// Package learn implements Bourbon's model manager and cost-benefit
// analyzer (paper section 4.4).
//
// The manager decides when a newly flushed SSTable should be trained
// with Greedy-PLR. It waits T_wait, estimates cost (C_model) versus
// benefit (B_model) from same-level statistics, and trains files in
// priority order. Modes: Never, Always, CostBenefit.
package learn

import (
	"container/heap"
	"sort"
	"sync"
	"time"

	"bourbon/plr"
)

// Mode selects the learning policy.
type Mode int

const (
	// ModeNever never builds models (baseline-only lookups).
	ModeNever Mode = iota
	// ModeAlways learns every file after T_wait (Bourbon-always).
	ModeAlways
	// ModeCostBenefit applies C_model < B_model (Bourbon-cba).
	ModeCostBenefit
)

func (m Mode) String() string {
	switch m {
	case ModeNever:
		return "never"
	case ModeAlways:
		return "always"
	case ModeCostBenefit:
		return "cba"
	default:
		return "unknown"
	}
}

// DefaultTWait matches the paper: 50 ms (max learn time ~40 ms).
const DefaultTWait = 50 * time.Millisecond

// FileStats accumulates lookup statistics for one file (same-level
// aggregates feed the analyzer).
type FileStats struct {
	mu sync.Mutex

	NegLookups int64
	PosLookups int64

	// Baseline path times (measured while waiting / before model).
	NegTime time.Duration
	PosTime time.Duration

	// Model path times (measured after learning).
	ModelNegTime time.Duration
	ModelPosTime time.Duration
	ModelNegN    int64
	ModelPosN    int64
}

func (s *FileStats) recordBaseline(positive bool, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if positive {
		s.PosLookups++
		s.PosTime += d
	} else {
		s.NegLookups++
		s.NegTime += d
	}
}

func (s *FileStats) recordModel(positive bool, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if positive {
		s.ModelPosN++
		s.ModelPosTime += d
	} else {
		s.ModelNegN++
		s.ModelNegTime += d
	}
}

func (s *FileStats) snapshot() FileSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	return FileSnap{
		NegLookups:   s.NegLookups,
		PosLookups:   s.PosLookups,
		NegTime:      s.NegTime,
		PosTime:      s.PosTime,
		ModelNegTime: s.ModelNegTime,
		ModelPosTime: s.ModelPosTime,
		ModelNegN:    s.ModelNegN,
		ModelPosN:    s.ModelPosN,
	}
}

// FileSnap is an immutable copy of FileStats.
type FileSnap struct {
	NegLookups   int64
	PosLookups   int64
	NegTime      time.Duration
	PosTime      time.Duration
	ModelNegTime time.Duration
	ModelPosTime time.Duration
	ModelNegN    int64
	ModelPosN    int64
}

// LevelAgg holds average statistics for files at one level that have
// completed a full lifetime (paper: statistics of replaced files).
type LevelAgg struct {
	Count int64

	// Averages over completed files.
	AvgNegLookups float64
	AvgPosLookups float64
	AvgNegTime    time.Duration // baseline neg
	AvgPosTime    time.Duration // baseline pos
	AvgModelNeg   time.Duration
	AvgModelPos   time.Duration
	AvgFileBytes  float64
}

// FileMeta is the learner's view of a candidate file.
type FileMeta struct {
	ID        string
	Level     int
	Bytes     uint64
	NumPoints int64
	CreatedAt time.Time

	// Train builds a PLR model from the file's sorted keys.
	Train func() (*plr.Model, error)
	// Attach installs the model on the live reader.
	Attach func(*plr.Model)
	// Stats is the live statistics object for this file.
	Stats *FileStats
}

// candidate is a file waiting for a learning decision.
type candidate struct {
	meta    FileMeta
	benefit float64 // B_model
	cost    float64 // C_model
	decided bool
	learn   bool
}

// item implements heap.Item ordered by (benefit - cost) descending.
type item struct {
	c   *candidate
	idx int
}

type prioQueue []*item

func (pq prioQueue) Len() int { return len(pq) }
func (pq prioQueue) Less(i, j int) bool {
	di := pq[i].c.benefit - pq[i].c.cost
	dj := pq[j].c.benefit - pq[j].c.cost
	return di > dj // max-heap
}
func (pq prioQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].idx = i
	pq[j].idx = j
}
func (pq *prioQueue) Push(x any) {
	it := x.(*item)
	it.idx = len(*pq)
	*pq = append(*pq, it)
}
func (pq *prioQueue) Pop() any {
	old := *pq
	n := len(old)
	it := old[n-1]
	old[n-1] = nil
	*pq = old[:n-1]
	return it
}

// Options configures the manager.
type Options struct {
	Mode Mode
	// TWait is how long to wait before considering a file.
	TWait time.Duration
	// TrainTimePerPoint is the calibrated average train cost
	// (C_model = TrainTimePerPoint * NumPoints).
	TrainTimePerPoint time.Duration
	// MinStatsFiles is how many completed files at a level are needed
	// before leaving always-learn bootstrap.
	MinStatsFiles int
	// Interval is how often the background loop runs.
	Interval time.Duration
}

func (o *Options) normalize() {
	if o.TWait <= 0 {
		o.TWait = DefaultTWait
	}
	if o.TrainTimePerPoint <= 0 {
		// Conservative default; calibrated at startup by the DB.
		o.TrainTimePerPoint = 200 * time.Nanosecond
	}
	if o.MinStatsFiles <= 0 {
		o.MinStatsFiles = 4
	}
	if o.Interval <= 0 {
		o.Interval = 10 * time.Millisecond
	}
}

// Manager runs the cost-benefit analyzer and trains models.
type Manager struct {
	opts Options

	mu         sync.Mutex
	cands      map[string]*candidate
	queue      prioQueue
	levelStats map[int][]completedFile // finished files per level
	byLevel    map[int]map[string]*FileStats

	// learnedFiles / counters for tests and reporting.
	learned  int64
	skipped  int64
	trainNs  int64 // total train time
	trainPts int64

	stop chan struct{}
	done chan struct{}
	once sync.Once

	// now is injectable for tests.
	now func() time.Time
}

type completedFile struct {
	bytes   uint64
	stats   FileSnap
	learned bool
}

// NewManager creates a manager.
func NewManager(opts Options) *Manager {
	opts.normalize()
	m := &Manager{
		opts:       opts,
		cands:      make(map[string]*candidate),
		levelStats: make(map[int][]completedFile),
		byLevel:    make(map[int]map[string]*FileStats),
		stop:       make(chan struct{}),
		done:       make(chan struct{}),
		now:        time.Now,
	}
	heap.Init(&m.queue)
	return m
}

// SetNow replaces the clock (tests).
func (m *Manager) SetNow(fn func() time.Time) {
	m.mu.Lock()
	m.now = fn
	m.mu.Unlock()
}

// RegisterFile enqueues a newly created SSTable for a learning decision.
func (m *Manager) RegisterFile(meta FileMeta) {
	if m.opts.Mode == ModeNever {
		return
	}
	if meta.Stats == nil {
		meta.Stats = &FileStats{}
	}
	c := &candidate{meta: meta}
	m.mu.Lock()
	// Replace any previous registration with same ID.
	if old, ok := m.cands[meta.ID]; ok {
		_ = old
	}
	m.cands[meta.ID] = c
	m.mu.Unlock()
}

// RetireFile removes a file (deleted by compaction) and folds its
// statistics into the level aggregate.
func (m *Manager) RetireFile(id string, level int, bytes uint64, learned bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	c, ok := m.cands[id]
	if !ok {
		return
	}
	delete(m.cands, id)
	snap := c.meta.Stats.snapshot()
	m.levelStats[level] = append(m.levelStats[level], completedFile{
		bytes:   bytes,
		stats:   snap,
		learned: learned,
	})
}

// RecordBaseline notes a lookup on the baseline path.
func (m *Manager) RecordBaseline(id string, positive bool, d time.Duration) {
	m.mu.Lock()
	c := m.cands[id]
	m.mu.Unlock()
	if c == nil || c.meta.Stats == nil {
		return
	}
	c.meta.Stats.recordBaseline(positive, d)
}

// RecordModel notes a lookup that used the model path.
func (m *Manager) RecordModel(id string, positive bool, d time.Duration) {
	m.mu.Lock()
	c := m.cands[id]
	m.mu.Unlock()
	if c == nil || c.meta.Stats == nil {
		return
	}
	c.meta.Stats.recordModel(positive, d)
}

// TrainCost returns C_model for a file of n points.
func (m *Manager) TrainCost(n int64) time.Duration {
	return time.Duration(int64(m.opts.TrainTimePerPoint) * n)
}

// levelAggregate builds LevelAgg from completed files at level.
func (m *Manager) levelAggregate(level int) LevelAgg {
	files := m.levelStats[level]
	var agg LevelAgg
	if len(files) == 0 {
		return agg
	}
	// Filter out very short-lived files (paper: analyzer filters them).
	var use []completedFile
	for _, f := range files {
		// Use files that either learned or served many lookups.
		if f.stats.NegLookups+f.stats.PosLookups >= 10 || f.learned {
			use = append(use, f)
		}
	}
	if len(use) == 0 {
		use = files
	}
	agg.Count = int64(len(use))
	var sumNeg, sumPos float64
	var sumNegT, sumPosT, sumMNeg, sumMPos time.Duration
	var sumBytes float64
	var mNegN, mPosN int64
	for _, f := range use {
		sumNeg += float64(f.stats.NegLookups)
		sumPos += float64(f.stats.PosLookups)
		sumNegT += f.stats.NegTime
		sumPosT += f.stats.PosTime
		sumMNeg += f.stats.ModelNegTime
		sumMPos += f.stats.ModelPosTime
		mNegN += f.stats.ModelNegN
		mPosN += f.stats.ModelPosN
		sumBytes += float64(f.bytes)
	}
	n := float64(len(use))
	agg.AvgNegLookups = sumNeg / n
	agg.AvgPosLookups = sumPos / n
	agg.AvgNegTime = time.Duration(int64(sumNegT) / int64(n))
	agg.AvgPosTime = time.Duration(int64(sumPosT) / int64(n))
	agg.AvgFileBytes = sumBytes / n
	if mNegN > 0 {
		agg.AvgModelNeg = time.Duration(int64(sumMNeg) / mNegN)
	} else {
		agg.AvgModelNeg = agg.AvgNegTime / 2 // optimistic default
	}
	if mPosN > 0 {
		agg.AvgModelPos = time.Duration(int64(sumMPos) / mPosN)
	} else {
		agg.AvgModelPos = agg.AvgPosTime / 2
	}
	return agg
}

// EstimateBenefit computes B_model for a candidate (paper eq.):
// B = (Tn.b - Tn.m)*Nn + (Tp.b - Tp.m)*Np
// with Nn, Np scaled by file size / average file size at the level.
func (m *Manager) EstimateBenefit(c *candidate) float64 {
	agg := m.levelAggregate(c.meta.Level)
	if agg.Count == 0 {
		// Bootstrap: no stats — report high benefit so we learn.
		return mathInf()
	}
	scale := 1.0
	if agg.AvgFileBytes > 0 {
		scale = float64(c.meta.Bytes) / agg.AvgFileBytes
	}
	nn := agg.AvgNegLookups * scale
	np := agg.AvgPosLookups * scale

	// Baseline times: use live stats if the file already served
	// lookups during T_wait; else level averages.
	snap := c.meta.Stats.snapshot()
	tnb := agg.AvgNegTime
	tpb := agg.AvgPosTime
	if snap.NegLookups > 0 {
		tnb = time.Duration(int64(snap.NegTime) / snap.NegLookups)
	}
	if snap.PosLookups > 0 {
		tpb = time.Duration(int64(snap.PosTime) / snap.PosLookups)
	}
	tnm := agg.AvgModelNeg
	tpm := agg.AvgModelPos

	bn := float64(tnb-tnm) * nn
	bp := float64(tpb-tpm) * np
	b := bn + bp
	if b < 0 {
		b = 0
	}
	return b
}

func mathInf() float64 {
	return 1e18
}

// Bootstrap reports whether the level still lacks statistics.
func (m *Manager) Bootstrap(level int) bool {
	return int(len(m.levelStats[level])) < m.opts.MinStatsFiles
}

// EvaluateOnce runs one analyzer pass: decide and train ready files.
func (m *Manager) EvaluateOnce() {
	m.mu.Lock()
	now := m.now()
	var ready []*candidate
	for _, c := range m.cands {
		if now.Sub(c.meta.CreatedAt) < m.opts.TWait {
			continue
		}
		if c.decided {
			continue
		}
		ready = append(ready, c)
	}
	m.mu.Unlock()

	if len(ready) == 0 {
		return
	}

	// Sort by creation time (older first).
	sort.Slice(ready, func(i, j int) bool {
		return ready[i].meta.CreatedAt.Before(ready[j].meta.CreatedAt)
	})

	var trainList []*candidate
	for _, c := range ready {
		var learn bool
		switch m.opts.Mode {
		case ModeAlways:
			learn = true
		case ModeCostBenefit:
			if m.Bootstrap(c.meta.Level) {
				learn = true // bootstrap: always-learn until stats exist
			} else {
				c.cost = float64(m.TrainCost(c.meta.NumPoints))
				c.benefit = m.EstimateBenefit(c)
				learn = c.cost < c.benefit
			}
			c.decided = true
			c.learn = learn
		default:
			learn = false
			c.decided = true
		}
		if m.opts.Mode == ModeAlways {
			c.decided = true
			c.learn = true
		}
		if learn {
			trainList = append(trainList, c)
		} else {
			m.mu.Lock()
			m.skipped++
			// Remove skipped files so they never train; keep stats
			// via RetireFile when compaction deletes them.
			// Mark decided; leave in cands for stats recording.
			m.mu.Unlock()
		}
	}

	// Priority queue ordered by B-C (always-mode: all equal by benefit).
	m.mu.Lock()
	m.queue = m.queue[:0]
	for _, c := range trainList {
		heap.Push(&m.queue, &item{c: c})
	}
	m.mu.Unlock()

	// Train in priority order.
	for m.queue.Len() > 0 {
		it := heap.Pop(&m.queue).(*item)
		m.trainOne(it.c)
	}
}

func (m *Manager) trainOne(c *candidate) {
	// Detach from cands so we do not double-train.
	m.mu.Lock()
	cur, ok := m.cands[c.meta.ID]
	if !ok || cur != c || (c.decided && !c.learn && m.opts.Mode != ModeAlways) {
		m.mu.Unlock()
		return
	}
	// Mark as training by deleting from map (stats move to level on retire).
	delete(m.cands, c.meta.ID)
	// Keep stats accessible: re-insert under special? Simpler: keep a
	// side map for stats after learn. We re-register stats pointer.
	m.mu.Unlock()

	start := time.Now()
	model, err := c.meta.Train()
	elapsed := time.Since(start)
	if err != nil || model == nil {
		m.mu.Lock()
		m.skipped++
		// Put back for stats.
		c.decided = false
		m.cands[c.meta.ID] = c
		m.mu.Unlock()
		return
	}
	if c.meta.Attach != nil {
		c.meta.Attach(model)
	}

	m.mu.Lock()
	m.learned++
	m.trainNs += elapsed.Nanoseconds()
	m.trainPts += c.meta.NumPoints
	// Keep stats in cands for ongoing lookup recording.
	c.decided = true
	c.learn = true
	m.cands[c.meta.ID] = c
	m.mu.Unlock()
}

// Start launches the background evaluation loop.
func (m *Manager) Start() {
	if m.opts.Mode == ModeNever {
		close(m.done)
		return
	}
	go func() {
		defer close(m.done)
		t := time.NewTicker(m.opts.Interval)
		defer t.Stop()
		for {
			select {
			case <-m.stop:
				m.EvaluateOnce()
				return
			case <-t.C:
				m.EvaluateOnce()
			}
		}
	}()
}

// Stop stops the background loop and waits.
func (m *Manager) Stop() {
	m.once.Do(func() {
		close(m.stop)
	})
	<-m.done
}

// Stats returns counters for tests and CLI output.
func (m *Manager) Stats() (learned, skipped int64, trainTime time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.learned, m.skipped, time.Duration(m.trainNs)
}

// CalibrateTrainTime measures average PLR train time per point using
// a sample of keys and stores it in the options.
func (m *Manager) CalibrateTrainTime(sample []uint64) {
	if len(sample) == 0 {
		return
	}
	start := time.Now()
	_ = plr.Train(sample, plr.DefaultDelta)
	elapsed := time.Since(start)
	per := time.Duration(int64(elapsed) / int64(len(sample)))
	if per <= 0 {
		per = time.Nanosecond
	}
	m.mu.Lock()
	m.opts.TrainTimePerPoint = per
	m.mu.Unlock()
}

// Mode returns the configured mode.
func (m *Manager) Mode() Mode { return m.opts.Mode }
