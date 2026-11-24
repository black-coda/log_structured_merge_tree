package learn

import (
	"testing"
	"time"

	"bourbon/plr"
)

func TestAlwaysLearnsAfterTWait(t *testing.T) {
	m := NewManager(Options{
		Mode:     ModeAlways,
		TWait:    10 * time.Millisecond,
		Interval: 5 * time.Millisecond,
	})

	var trained bool
	m.RegisterFile(FileMeta{
		ID:        "f1",
		Level:     0,
		Bytes:     1000,
		NumPoints: 100,
		CreatedAt: time.Now().Add(-time.Second),
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			trained = true
			keys := make([]uint64, 100)
			for i := range keys {
				keys[i] = uint64(i)
			}
			return plr.Train(keys, 8), nil
		},
		Attach: func(m *plr.Model) {},
	})

	m.EvaluateOnce()
	learned, _, _ := m.Stats()
	if !trained || learned != 1 {
		t.Fatalf("trained=%v learned=%d", trained, learned)
	}
}

func TestWaitBeforeLearning(t *testing.T) {
	now := time.Now()
	m := NewManager(Options{Mode: ModeAlways, TWait: time.Hour})
	m.SetNow(func() time.Time { return now })

	m.RegisterFile(FileMeta{
		ID:        "f1",
		Level:     0,
		CreatedAt: now, // just created
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			t.Fatal("should not train yet")
			return nil, nil
		},
	})
	m.EvaluateOnce()
	learned, _, _ := m.Stats()
	if learned != 0 {
		t.Fatalf("learned=%d before T_wait", learned)
	}
}

func TestNeverModeSkips(t *testing.T) {
	m := NewManager(Options{Mode: ModeNever, TWait: 0})
	m.RegisterFile(FileMeta{
		ID:        "f1",
		CreatedAt: time.Now().Add(-time.Hour),
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			t.Fatal("never mode must not train")
			return nil, nil
		},
	})
	m.EvaluateOnce()
	learned, _, _ := m.Stats()
	if learned != 0 {
		t.Fatalf("learned=%d", learned)
	}
}

func TestCbaBootstrapAlwaysLearns(t *testing.T) {
	m := NewManager(Options{
		Mode:          ModeCostBenefit,
		TWait:         time.Millisecond,
		MinStatsFiles: 4,
	})
	m.RegisterFile(FileMeta{
		ID:        "f1",
		Level:     0,
		NumPoints: 50,
		CreatedAt: time.Now().Add(-time.Second),
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			keys := make([]uint64, 50)
			for i := range keys {
				keys[i] = uint64(i)
			}
			return plr.Train(keys, 8), nil
		},
		Attach: func(*plr.Model) {},
	})
	m.EvaluateOnce()
	learned, _, _ := m.Stats()
	if learned != 1 {
		t.Fatalf("bootstrap should learn, learned=%d", learned)
	}
}

func TestCbaSkipsWhenCostExceedsBenefit(t *testing.T) {
	m := NewManager(Options{
		Mode:          ModeCostBenefit,
		TWait:         time.Millisecond,
		MinStatsFiles: 1,
		// Huge train cost per point => C_model very large.
		TrainTimePerPoint: 10 * time.Millisecond,
	})
	// Seed level stats: completed files with few lookups.
	for i := 0; i < 4; i++ {
		m.RetireFile("old", 0, 1000, true)
	}
	// Fix aggregates: RetireFile needs different IDs to count as separate files.
	// RetireFile with same ID overwrites in cands but appends to levelStats
	// only when found in cands. Seed properly:
	m.mu.Lock()
	for i := 0; i < 4; i++ {
		m.levelStats[0] = append(m.levelStats[0], completedFile{
			bytes: 1000,
			stats: FileSnap{NegLookups: 5, PosLookups: 5, NegTime: 1000, PosTime: 1000},
		})
	}
	m.mu.Unlock()

	var trained bool
	m.RegisterFile(FileMeta{
		ID:        "new",
		Level:     0,
		Bytes:     1000,
		NumPoints: 100000, // expensive to train
		CreatedAt: time.Now().Add(-time.Second),
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			trained = true
			return nil, nil
		},
	})
	m.EvaluateOnce()
	learned, skipped, _ := m.Stats()
	if trained || learned != 0 {
		t.Fatalf("should skip: trained=%v learned=%d", trained, learned)
	}
	if skipped == 0 {
		t.Fatal("expected skipped counter")
	}
}

func TestCbaLearnsWhenBenefitOutweighsCost(t *testing.T) {
	m := NewManager(Options{
		Mode:              ModeCostBenefit,
		TWait:             time.Millisecond,
		MinStatsFiles:     1,
		TrainTimePerPoint: time.Nanosecond, // cheap
	})
	m.mu.Lock()
	for i := 0; i < 4; i++ {
		// Files that served many baseline lookups; model roughly 2x faster.
		m.levelStats[0] = append(m.levelStats[0], completedFile{
			bytes: 1000,
			stats: FileSnap{
				NegLookups:   100000,
				PosLookups:   100000,
				NegTime:      1000 * time.Microsecond,
				PosTime:      1000 * time.Microsecond,
				ModelNegTime: 400 * time.Microsecond,
				ModelPosTime: 400 * time.Microsecond,
				ModelNegN:    1000,
				ModelPosN:    1000,
			},
		})
	}
	m.mu.Unlock()

	trained := false
	m.RegisterFile(FileMeta{
		ID:        "new",
		Level:     0,
		Bytes:     1000,
		NumPoints: 10,
		CreatedAt: time.Now().Add(-time.Second),
		Stats:     &FileStats{},
		Train: func() (*plr.Model, error) {
			trained = true
			keys := []uint64{1, 2, 3}
			return plr.Train(keys, 8), nil
		},
		Attach: func(*plr.Model) {},
	})
	m.EvaluateOnce()
	learned, _, _ := m.Stats()
	if !trained || learned != 1 {
		t.Fatalf("should learn: trained=%v learned=%d", trained, learned)
	}
}

func TestCalibrate(t *testing.T) {
	m := NewManager(Options{Mode: ModeAlways})
	keys := make([]uint64, 10000)
	for i := range keys {
		keys[i] = uint64(i)
	}
	m.CalibrateTrainTime(keys)
	m.mu.Lock()
	per := m.opts.TrainTimePerPoint
	m.mu.Unlock()
	if per <= 0 {
		t.Fatal("calibration failed")
	}
}
