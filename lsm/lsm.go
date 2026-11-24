// Package lsm implements a WiscKey-style log-structured merge tree
// with optional learned indexes (Bourbon).
//
// Write path: WAL -> memtable -> flush to L0 SSTable (values in vlog).
// Read path: memtable, then FindFiles across levels; each candidate
// file uses the model path when learned, else the baseline path.
package lsm

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"bourbon/learn"
	"bourbon/memtable"
	"bourbon/plr"
	"bourbon/sstable"
	"bourbon/vlog"
)

// Options configures a DB.
type Options struct {
	Dir string

	// MemtableBytes triggers a flush when exceeded (default 4 MB).
	MemtableBytes int64
	// LevelRatio is the size ratio between levels (default 10).
	LevelRatio int
	// NumLevels is L0..L{n-1} (default 7).
	NumLevels int
	// BaseLevelBytes is the target size of L1 (default 10 MB).
	BaseLevelBytes int64
	// L0FileTrigger is the number of L0 files that forces compaction.
	L0FileTrigger int
	// BlockSize for SSTables (default 4 KB).
	BlockSize int
	// ErrorBound is the PLR delta (default 8).
	ErrorBound int64
	// LearnMode selects Never / Always / CostBenefit.
	LearnMode learn.Mode
	// TWait before learning (default 50 ms).
	TWait time.Duration
	// DisableWAL skips the write-ahead log (benchmarks only).
	DisableWAL bool
	// ValueLogDir overrides dir/vlog.
	ValueLogDir string
}

func (o *Options) normalize() error {
	if o.Dir == "" {
		return errors.New("lsm: Dir is required")
	}
	if o.MemtableBytes <= 0 {
		o.MemtableBytes = 4 << 20
	}
	if o.LevelRatio <= 1 {
		o.LevelRatio = 10
	}
	if o.NumLevels < 2 {
		o.NumLevels = 7
	}
	if o.BaseLevelBytes <= 0 {
		o.BaseLevelBytes = 10 << 20
	}
	if o.L0FileTrigger <= 0 {
		o.L0FileTrigger = 4
	}
	if o.BlockSize <= 0 {
		o.BlockSize = sstable.DefaultBlockSize
	}
	if o.ErrorBound <= 0 {
		o.ErrorBound = plr.DefaultDelta
	}
	if o.TWait <= 0 {
		o.TWait = learn.DefaultTWait
	}
	return nil
}

// fileMeta tracks one SSTable in the tree.
type fileMeta struct {
	ID        string // absolute path (unique)
	Level     int
	MinKey    uint64
	MaxKey    uint64
	NumRecs   int64
	Bytes     uint64
	CreatedAt time.Time
	Seq       uint64 // flush sequence; higher = newer

	reader *sstable.Reader
	stats  *learn.FileStats
}

func (f *fileMeta) contains(key uint64) bool {
	return key >= f.MinKey && key <= f.MaxKey
}

// DB is the Bourbon key-value store.
type DB struct {
	opts Options

	mu      sync.Mutex
	immCond *sync.Cond // signals when imm becomes nil
	flushMu sync.Mutex // serializes flushImm calls

	mem     *memtable.MemTable
	imm     *memtable.MemTable // flush in progress (stays readable until L0 insert)
	wal     *memtable.WAL
	vl      *vlog.Manager
	levels  [][]*fileMeta
	seq     uint64
	fileNum uint64

	learner *learn.Manager
	closed  bool
	flushCh chan struct{}
	stopCh  chan struct{}
	wg      sync.WaitGroup

	// nextID assigns monotonic file numbers.
	nextID uint64

	// Counters for tests/CLI.
	baselineHits int64
	modelHits    int64
}

// Open creates or recovers a database at opts.Dir.
func Open(opts Options) (*DB, error) {
	if err := opts.normalize(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(opts.Dir, 0o755); err != nil {
		return nil, err
	}
	levels := make([][]*fileMeta, opts.NumLevels)

	db := &DB{
		opts:    opts,
		mem:     memtable.New(),
		levels:  levels,
		flushCh: make(chan struct{}, 1),
		stopCh:  make(chan struct{}),
	}
	db.immCond = sync.NewCond(&db.mu)

	vdir := opts.ValueLogDir
	if vdir == "" {
		vdir = filepath.Join(opts.Dir, "vlog")
	}
	db.vl = vlog.NewManager(vdir)

	if !opts.DisableWAL {
		wal, err := memtable.NewWAL(filepath.Join(opts.Dir, "000001.wal"))
		if err != nil {
			return nil, err
		}
		db.wal = wal
	}

	db.learner = learn.NewManager(learn.Options{
		Mode:  opts.LearnMode,
		TWait: opts.TWait,
	})

	// Recover existing SSTables if present.
	if err := db.recover(); err != nil {
		return nil, err
	}

	db.learner.Start()
	db.wg.Add(1)
	go db.flushWorker()

	return db, nil
}

// recover scans Dir for *.sst and registers them into levels.
func (db *DB) recover() error {
	entries, err := os.ReadDir(db.opts.Dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		name := e.Name()
		if filepath.Ext(name) != ".sst" {
			continue
		}
		path := filepath.Join(db.opts.Dir, name)
		r, err := sstable.Open(path)
		if err != nil {
			return fmt.Errorf("recover %s: %w", path, err)
		}
		// Level is encoded as NNNNNN.Lll.sst
		level, num, err := parseFileName(name)
		if err != nil {
			r.Close()
			return err
		}
		if level >= len(db.levels) {
			r.Close()
			continue
		}
		min, max := r.Bounds()
		fm := &fileMeta{
			ID:        path,
			Level:     level,
			MinKey:    min,
			MaxKey:    max,
			NumRecs:   r.NumRecords(),
			CreatedAt: time.Now(),
			Seq:       num,
			reader:    r,
			stats:     &learn.FileStats{},
		}
		fi, _ := os.Stat(path)
		if fi != nil {
			fm.Bytes = uint64(fi.Size())
		}
		db.levels[level] = append(db.levels[level], fm)
		db.registerLearner(fm)
		if num >= db.nextID {
			db.nextID = num + 1
		}
	}
	for lvl := range db.levels {
		sortFiles(db.levels[lvl])
	}
	return nil
}

func parseFileName(name string) (level int, num uint64, err error) {
	// format: 000012.sst or 000012.L2.sst — we use %06d.L%d.sst
	var n uint64
	var l int
	if cnt, e := fmt.Sscanf(name, "%06d.L%d.sst", &n, &l); e != nil || cnt != 2 {
		if cnt, e2 := fmt.Sscanf(name, "%06d.sst", &n); e2 != nil || cnt != 1 {
			return 0, 0, fmt.Errorf("bad sst name %q", name)
		}
		return 0, n, nil
	}
	return l, n, nil
}

func sortFiles(files []*fileMeta) {
	sort.Slice(files, func(i, j int) bool {
		if files[i].MinKey != files[j].MinKey {
			return files[i].MinKey < files[j].MinKey
		}
		return files[i].Seq < files[j].Seq
	})
}

// Put writes a key-value pair.
func (db *DB) Put(key uint64, value []byte) error {
	return db.write(key, value, false)
}

// Delete removes a key (writes a tombstone).
func (db *DB) Delete(key uint64) error {
	return db.write(key, nil, true)
}

func (db *DB) write(key uint64, value []byte, tomb bool) error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return errors.New("lsm: closed")
	}

	db.seq++
	seq := db.seq

	if tomb {
		if db.wal != nil {
			if err := db.wal.AppendDelete(key, seq); err != nil {
				db.mu.Unlock()
				return err
			}
		}
		db.mem.Delete(key, seq)
	} else {
		ptr, err := db.vl.Append(value)
		if err != nil {
			db.mu.Unlock()
			return err
		}
		if db.wal != nil {
			if err := db.wal.AppendPut(key, ptr, value, seq); err != nil {
				db.mu.Unlock()
				return err
			}
		}
		db.mem.Put(key, ptr, len(value), seq)
	}

	needFlush := db.mem.ApproxBytes() >= db.opts.MemtableBytes
	if needFlush {
		// Never overwrite a pending imm table — wait for its flush.
		for db.imm != nil && !db.closed {
			db.immCond.Wait()
		}
		if db.closed {
			db.mu.Unlock()
			return errors.New("lsm: closed")
		}
		db.imm = db.mem
		db.mem = memtable.New()
	}
	db.mu.Unlock()

	if needFlush {
		select {
		case db.flushCh <- struct{}{}:
		default:
		}
	}
	return nil
}

// Get looks up key. Returns (value, true, nil) if found.
// Returns (nil, false, nil) if absent or deleted.
func (db *DB) Get(key uint64) ([]byte, bool, error) {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil, false, errors.New("lsm: closed")
	}

	// 1. Memtable.
	if e, ok := db.mem.Get(key); ok {
		db.mu.Unlock()
		if e.Tomb {
			return nil, false, nil
		}
		v, err := db.readPointer(e.Pointer)
		return v, err == nil && v != nil, err
	}
	// 2. Immutable memtable (flush in progress).
	if db.imm != nil {
		if e, ok := db.imm.Get(key); ok {
			db.mu.Unlock()
			if e.Tomb {
				return nil, false, nil
			}
			v, err := db.readPointer(e.Pointer)
			return v, err == nil && v != nil, err
		}
	}

	// 3. FindFiles: candidate SSTables across levels.
	cands := db.findFiles(key)
	db.mu.Unlock()

	// 4. Internal lookups on candidates (newest level first).
	var lastErr error
	for _, f := range cands {
		start := time.Now()
		res, err := db.lookupFile(f, key)
		elapsed := time.Since(start)
		if err != nil {
			lastErr = err
			continue
		}
		if res.positive {
			// Positive internal lookup: stop.
			db.recordLookup(f, true, res.usedModel, elapsed)
			if res.tomb {
				return nil, false, nil
			}
			v, err := db.readPointer(res.ptr)
			return v, err == nil && v != nil, err
		}
		// Negative internal lookup: continue to next candidate.
		db.recordLookup(f, false, res.usedModel, elapsed)
	}
	if lastErr != nil {
		return nil, false, lastErr
	}
	return nil, false, nil
}

type lookupOutcome struct {
	positive  bool
	tomb      bool
	usedModel bool
	ptr       sstable.ValuePointer
}

// lookupFile performs one internal lookup against a file.
func (db *DB) lookupFile(f *fileMeta, key uint64) (lookupOutcome, error) {
	var out lookupOutcome
	if f.reader == nil {
		return out, nil
	}
	// Bloom check is inside Get / GetWithModel.

	// Tombstones are stored as special pointers: Size=0 && LogID=tombLog?
	// Simpler: we store tombstone markers in the table via a reserved
	// pointer (LogID=^uint32(0)). Encode that below in flush.
	if f.reader.IsLearned() {
		res, err := f.reader.GetWithModel(key)
		out.usedModel = true
		if err != nil {
			return out, err
		}
		if !res.Found {
			return out, nil
		}
		out.positive = true
		if isTombPointer(res.Pointer) {
			out.tomb = true
			return out, nil
		}
		out.ptr = res.Pointer
		return out, nil
	}

	res, err := f.reader.Get(key)
	out.usedModel = false
	if err != nil {
		return out, err
	}
	if !res.Found {
		return out, nil
	}
	out.positive = true
	if isTombPointer(res.Pointer) {
		out.tomb = true
		return out, nil
	}
	out.ptr = res.Pointer
	return out, nil
}

const tombLogID = ^uint32(0)

func tombPointer() sstable.ValuePointer {
	return sstable.ValuePointer{LogID: tombLogID, Size: 0}
}

func isTombPointer(p sstable.ValuePointer) bool {
	return p.LogID == tombLogID
}

func (db *DB) recordLookup(f *fileMeta, positive, usedModel bool, d time.Duration) {
	if usedModel {
		db.mu.Lock()
		db.modelHits++
		db.mu.Unlock()
		db.learner.RecordModel(f.ID, positive, d)
	} else {
		db.mu.Lock()
		db.baselineHits++
		db.mu.Unlock()
		db.learner.RecordBaseline(f.ID, positive, d)
	}
}

func (db *DB) readPointer(ptr sstable.ValuePointer) ([]byte, error) {
	if isTombPointer(ptr) {
		return nil, nil
	}
	return db.vl.Read(ptr)
}

// findFiles returns candidate files for key, newest level first.
// Caller must hold db.mu (or accept a consistent snapshot).
func (db *DB) findFiles(key uint64) []*fileMeta {
	var out []*fileMeta
	// L0: all overlapping files (ranges may overlap); newest first.
	l0 := db.levels[0]
	for i := len(l0) - 1; i >= 0; i-- {
		f := l0[i]
		if f.contains(key) {
			out = append(out, f)
		}
	}
	// L1+: at most one file per level (disjoint ranges).
	for lvl := 1; lvl < len(db.levels); lvl++ {
		files := db.levels[lvl]
		// Binary search for file with MinKey <= key <= MaxKey.
		i := sort.Search(len(files), func(i int) bool {
			return files[i].MaxKey >= key
		})
		if i < len(files) && files[i].contains(key) {
			out = append(out, files[i])
		}
	}
	return out
}

// FindFilesCount exposes the candidate count (tests / benchmarks).
func (db *DB) FindFilesCount(key uint64) int {
	db.mu.Lock()
	defer db.mu.Unlock()
	return len(db.findFiles(key))
}

// Stats returns baseline and model path hit counters.
func (db *DB) Stats() (baseline, model int64) {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.baselineHits, db.modelHits
}

// Learner exposes the manager (tests / CLI).
func (db *DB) Learner() *learn.Manager { return db.learner }

// registerLearner wires a file into the cost-benefit analyzer.
func (db *DB) registerLearner(f *fileMeta) {
	if db.learner.Mode() == learn.ModeNever {
		return
	}
	meta := f
	db.learner.RegisterFile(learn.FileMeta{
		ID:        meta.ID,
		Level:     meta.Level,
		Bytes:     meta.Bytes,
		NumPoints: meta.NumRecs,
		CreatedAt: meta.CreatedAt,
		Stats:     meta.stats,
		Train: func() (*plr.Model, error) {
			keys, err := meta.reader.Keys()
			if err != nil {
				return nil, err
			}
			return plr.Train(keys, db.opts.ErrorBound), nil
		},
		Attach: func(m *plr.Model) {
			meta.reader.SetModel(m)
		},
	})
}

// flushWorker drains the flush channel and flushes the immutable memtable.
func (db *DB) flushWorker() {
	defer db.wg.Done()
	for {
		select {
		case <-db.stopCh:
			return
		case <-db.flushCh:
			db.flushImm()
		}
	}
}

// flushImm writes db.imm to an L0 SSTable.
// The imm table stays readable until the new file is registered in L0,
// so concurrent Gets never miss data mid-flush.
func (db *DB) flushImm() {
	db.flushMu.Lock()
	defer db.flushMu.Unlock()

	db.mu.Lock()
	imm := db.imm
	if imm == nil || db.closed {
		db.mu.Unlock()
		return
	}
	entries := imm.Iter()
	// Keep db.imm set until the SSTable is linked into L0.
	db.mu.Unlock()

	if len(entries) == 0 {
		db.mu.Lock()
		if db.imm == imm {
			db.imm = nil
			db.immCond.Broadcast()
		}
		db.mu.Unlock()
		return
	}

	db.mu.Lock()
	db.nextID++
	num := db.nextID
	db.mu.Unlock()

	level := 0
	path := filepath.Join(db.opts.Dir, fmt.Sprintf("%06d.L%d.sst", num, level))

	w, err := sstable.NewWriter(path, sstable.Options{BlockSize: db.opts.BlockSize})
	if err != nil {
		// Fail closed to avoid wedging writers waiting on immCond.
		// Data remains in the WAL for recovery.
		db.mu.Lock()
		if db.imm == imm {
			db.imm = nil
			db.immCond.Broadcast()
		}
		db.mu.Unlock()
		return
	}

	// Entries are sorted and unique. Tombstones use a reserved pointer.
	for _, e := range entries {
		ptr := e.Pointer
		if e.Tomb {
			ptr = tombPointer()
		}
		if err := w.Add(e.Key, ptr); err != nil {
			continue
		}
	}
	meta, err := w.Finish()
	if err != nil {
		db.mu.Lock()
		if db.imm == imm {
			db.imm = nil
			db.immCond.Broadcast()
		}
		db.mu.Unlock()
		return
	}

	// Reset WAL after successful flush.
	if db.wal != nil {
		db.wal.Close()
		if wal, err := memtable.NewWAL(filepath.Join(db.opts.Dir, fmt.Sprintf("%06d.wal", num))); err == nil {
			db.mu.Lock()
			db.wal = wal
			db.mu.Unlock()
		}
	}

	r, err := sstable.Open(path)
	if err != nil {
		db.mu.Lock()
		if db.imm == imm {
			db.imm = nil
			db.immCond.Broadcast()
		}
		db.mu.Unlock()
		return
	}
	min, max := r.Bounds()
	fm := &fileMeta{
		ID:        path,
		Level:     level,
		MinKey:    min,
		MaxKey:    max,
		NumRecs:   meta.NumRecs,
		Bytes:     meta.Bytes,
		CreatedAt: time.Now(),
		Seq:       num,
		reader:    r,
		stats:     &learn.FileStats{},
	}

	db.mu.Lock()
	db.levels[0] = append(db.levels[0], fm)
	sortFiles(db.levels[0])
	// Only now is imm data fully visible in L0 — drop the reference.
	if db.imm == imm {
		db.imm = nil
		db.immCond.Broadcast()
	}
	db.mu.Unlock()

	db.registerLearner(fm)
	db.maybeCompact()
}

// maybeCompact triggers compactions when limits are exceeded.
func (db *DB) maybeCompact() {
	db.mu.Lock()
	var tasks [][2]int // (fromLevel, toLevel)
	if len(db.levels[0]) >= db.opts.L0FileTrigger {
		tasks = append(tasks, [2]int{0, 1})
	}
	for lvl := 1; lvl < len(db.levels)-1; lvl++ {
		if db.levelBytes(lvl) > db.levelLimit(lvl) {
			tasks = append(tasks, [2]int{lvl, lvl + 1})
		}
	}
	db.mu.Unlock()

	for _, t := range tasks {
		db.compact(t[0], t[1])
	}
}

func (db *DB) levelBytes(lvl int) int64 {
	var sum int64
	for _, f := range db.levels[lvl] {
		sum += int64(f.Bytes)
	}
	return sum
}

func (db *DB) levelLimit(lvl int) int64 {
	// L1 = BaseLevelBytes, each next level * ratio.
	limit := db.opts.BaseLevelBytes
	for i := 1; i < lvl; i++ {
		limit *= int64(db.opts.LevelRatio)
	}
	return limit
}

// compact merges files from `from` into `to`.
func (db *DB) compact(from, to int) {
	db.mu.Lock()
	if from >= len(db.levels) || to >= len(db.levels) {
		db.mu.Unlock()
		return
	}
	var inputs []*fileMeta
	if from == 0 {
		// Compact all L0 files (simple policy).
		inputs = append(inputs, db.levels[0]...)
	} else {
		// Compact the largest file at `from`.
		var best *fileMeta
		for _, f := range db.levels[from] {
			if best == nil || f.Bytes > best.Bytes {
				best = f
			}
		}
		if best == nil {
			db.mu.Unlock()
			return
		}
		inputs = append(inputs, best)
	}
	if len(inputs) == 0 {
		db.mu.Unlock()
		return
	}

	// Overlapping files at `to`.
	var lo, hi uint64 = inputs[0].MinKey, inputs[0].MaxKey
	for _, f := range inputs {
		if f.MinKey < lo {
			lo = f.MinKey
		}
		if f.MaxKey > hi {
			hi = f.MaxKey
		}
	}
	var overlaps []*fileMeta
	for _, f := range db.levels[to] {
		if f.MaxKey >= lo && f.MinKey <= hi {
			overlaps = append(overlaps, f)
		}
	}

	// Remove selected inputs from their level (atomic under lock).
	db.removeFiles(from, inputs)
	db.removeFiles(to, overlaps)
	db.mu.Unlock()

	// Merge outside the lock.
	newFile := db.mergeFiles(from, inputs, overlaps, to)

	// Retire stats for deleted files.
	for _, f := range inputs {
		db.learner.RetireFile(f.ID, f.Level, f.Bytes, f.reader.IsLearned())
		f.reader.Close()
	}
	for _, f := range overlaps {
		db.learner.RetireFile(f.ID, f.Level, f.Bytes, f.reader.IsLearned())
		f.reader.Close()
	}

	if newFile != nil {
		db.mu.Lock()
		db.levels[to] = append(db.levels[to], newFile)
		sortFiles(db.levels[to])
		db.mu.Unlock()
		db.registerLearner(newFile)
	}
}

func (db *DB) removeFiles(level int, files []*fileMeta) {
	if len(files) == 0 {
		return
	}
	drop := make(map[string]bool, len(files))
	for _, f := range files {
		drop[f.ID] = true
	}
	kept := db.levels[level][:0]
	for _, f := range db.levels[level] {
		if !drop[f.ID] {
			kept = append(kept, f)
		}
	}
	db.levels[level] = kept
}

// mergeFiles k-way merges input tables and writes a new table at `to`.
// For equal keys, newer sources win: higher `from` level beats `to`,
// and among same-level inputs, higher Seq wins.
func (db *DB) mergeFiles(from int, inputs, overlaps []*fileMeta, to int) *fileMeta {
	type src struct {
		it  *sstable.Iterator
		fm  *fileMeta
		pri int // higher = newer
	}

	var srcs []src
	for _, f := range inputs {
		srcs = append(srcs, src{it: f.reader.NewIterator(), fm: f})
	}
	for _, f := range overlaps {
		srcs = append(srcs, src{it: f.reader.NewIterator(), fm: f})
	}
	for i := range srcs {
		f := srcs[i].fm
		if f.Level == from {
			srcs[i].pri = 1_000_000 + int(f.Seq)
		} else {
			srcs[i].pri = int(f.Seq)
		}
	}

	db.nextID++
	num := db.nextID
	path := filepath.Join(db.opts.Dir, fmt.Sprintf("%06d.L%d.sst", num, to))
	w, err := sstable.NewWriter(path, sstable.Options{BlockSize: db.opts.BlockSize})
	if err != nil {
		return nil
	}

	n := len(srcs)
	type rec struct {
		key uint64
		ptr sstable.ValuePointer
		pri int
	}
	heads := make([]rec, n)
	alive := make([]bool, n)
	for i, s := range srcs {
		if s.it.Valid() {
			heads[i] = rec{key: s.it.Key(), ptr: s.it.Pointer(), pri: s.pri}
			alive[i] = true
		}
	}

	advance := func(i int) {
		if srcs[i].it.Next() && srcs[i].it.Valid() {
			heads[i] = rec{
				key: srcs[i].it.Key(),
				ptr: srcs[i].it.Pointer(),
				pri: srcs[i].pri,
			}
		} else {
			alive[i] = false
		}
	}

	lastKey := uint64(0)
	haveLast := false
	dropTomb := to == len(db.levels)-1

	for {
		best := -1
		for i := 0; i < n; i++ {
			if !alive[i] {
				continue
			}
			if best == -1 ||
				heads[i].key < heads[best].key ||
				(heads[i].key == heads[best].key && heads[i].pri > heads[best].pri) {
				best = i
			}
		}
		if best == -1 {
			break
		}
		k := heads[best].key
		if haveLast && k == lastKey {
			for i := 0; i < n; i++ {
				if alive[i] && heads[i].key == k {
					advance(i)
				}
			}
			continue
		}
		ptr := heads[best].ptr
		if !(dropTomb && isTombPointer(ptr)) {
			if err := w.Add(k, ptr); err != nil {
				// Keys must ascend; skip on defensive failure.
				_ = err
			}
		}
		lastKey = k
		haveLast = true
		for i := 0; i < n; i++ {
			if alive[i] && heads[i].key == k {
				advance(i)
			}
		}
	}

	meta, err := w.Finish()
	if err != nil {
		return nil
	}
	r, err := sstable.Open(path)
	if err != nil {
		return nil
	}
	min, max := r.Bounds()
	if meta.NumRecs == 0 {
		r.Close()
		os.Remove(path)
		return nil
	}
	return &fileMeta{
		ID:        path,
		Level:     to,
		MinKey:    min,
		MaxKey:    max,
		NumRecs:   meta.NumRecs,
		Bytes:     meta.Bytes,
		CreatedAt: time.Now(),
		Seq:       num,
		reader:    r,
		stats:     &learn.FileStats{},
	}
}

// Flush forces the memtable to disk as an L0 file.
func (db *DB) Flush() error {
	db.mu.Lock()
	// Wait for any in-flight flush to finish before swapping.
	for db.imm != nil && !db.closed {
		db.immCond.Wait()
	}
	if db.closed {
		db.mu.Unlock()
		return errors.New("lsm: closed")
	}
	if db.mem.Len() == 0 {
		db.mu.Unlock()
		return nil
	}
	db.imm = db.mem
	db.mem = memtable.New()
	db.mu.Unlock()

	db.flushImm()
	return nil
}

// CompactAll runs compaction until every level is within limits
// (best-effort; used by tests to force merges).
func (db *DB) CompactAll() {
	for round := 0; round < 32; round++ {
		db.mu.Lock()
		triggered := false
		if len(db.levels[0]) > 1 {
			triggered = true
		} else {
			for lvl := 1; lvl < len(db.levels)-1; lvl++ {
				if db.levelBytes(lvl) > db.levelLimit(lvl) {
					triggered = true
					break
				}
			}
		}
		db.mu.Unlock()
		if !triggered {
			return
		}
		db.maybeCompact()
	}
}

// Calibrate measures PLR train cost and feeds the analyzer.
func (db *DB) Calibrate(sample []uint64) {
	db.learner.CalibrateTrainTime(sample)
}

// Close stops background work and releases resources.
func (db *DB) Close() error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil
	}
	db.closed = true
	db.immCond.Broadcast() // wake any writers waiting on imm
	db.mu.Unlock()

	close(db.stopCh)
	db.wg.Wait()
	db.learner.Stop()

	var err error
	if db.wal != nil {
		err = db.wal.Close()
	}
	db.vl.Close()

	db.mu.Lock()
	for _, lvl := range db.levels {
		for _, f := range lvl {
			f.reader.Close()
		}
	}
	db.mu.Unlock()
	return err
}

// Levels exposes file counts per level (tests).
func (db *DB) Levels() []int {
	db.mu.Lock()
	defer db.mu.Unlock()
	out := make([]int, len(db.levels))
	for i, lvl := range db.levels {
		out[i] = len(lvl)
	}
	return out
}
