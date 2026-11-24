package bench

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"bourbon/learn"
	"bourbon/lsm"
	"bourbon/plr"
)

// openDB creates a DB for benchmarking under a temp dir.
func openDB(tb testing.TB, mode learn.Mode) *lsm.DB {
	tb.Helper()
	opts := lsm.Options{
		Dir:            filepath.Join(tb.TempDir(), "db"),
		MemtableBytes:  16 << 20,
		L0FileTrigger:  8,
		BaseLevelBytes: 64 << 20,
		LearnMode:      mode,
		DisableWAL:     true,
		TWait:          20 * time.Millisecond,
	}
	db, err := lsm.Open(opts)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { db.Close() })
	return db
}

func load(db *lsm.DB, keys []uint64, valueLen int) error {
	val := make([]byte, valueLen)
	for i := range val {
		val[i] = byte(i)
	}
	for _, k := range keys {
		if err := db.Put(k, val); err != nil {
			return err
		}
	}
	return db.Flush()
}

func loadTB(tb testing.TB, db *lsm.DB, keys []uint64, valueLen int) {
	tb.Helper()
	if err := load(db, keys, valueLen); err != nil {
		tb.Fatal(err)
	}
}

func waitLearned(tb testing.TB, db *lsm.DB, min int64) {
	tb.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		learned, _, _ := db.Learner().Stats()
		if learned >= min {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	learned, _, _ := db.Learner().Stats()
	tb.Fatalf("learned=%d, want >= %d", learned, min)
}

func trainPLR(keys []uint64) *plr.Model {
	return plr.Train(keys, plr.DefaultDelta)
}

// BenchmarkGetBaseline measures lookups with learning disabled.
func BenchmarkGetBaseline(b *testing.B) {
	for _, name := range []string{"linear", "seg1", "normal"} {
		b.Run(name, func(b *testing.B) {
			ds := Build(name, 200000)
			db := openDB(b, learn.ModeNever)
			loadTB(b, db, ds.Keys, 64)
			queries := ds.Queries(b.N, 7)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, ok, err := db.Get(queries[i]); err != nil || !ok {
					b.Fatalf("lookup fail ok=%v err=%v", ok, err)
				}
			}
		})
	}
}

// BenchmarkGetModel measures lookups after models are built.
func BenchmarkGetModel(b *testing.B) {
	for _, name := range []string{"linear", "seg1", "normal"} {
		b.Run(name, func(b *testing.B) {
			ds := Build(name, 200000)
			db := openDB(b, learn.ModeAlways)
			loadTB(b, db, ds.Keys, 64)
			waitLearned(b, db, 1)
			queries := ds.Queries(b.N, 7)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, ok, err := db.Get(queries[i]); err != nil || !ok {
					b.Fatalf("lookup fail ok=%v err=%v", ok, err)
				}
			}
		})
	}
}

// BenchmarkTrainPLR measures Greedy-PLR training cost.
func BenchmarkTrainPLR(b *testing.B) {
	ds := Build("normal", 200000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainPLR(ds.Keys)
	}
}

// Result of a one-shot speedup comparison.
type Result struct {
	Dataset   string
	Baseline  time.Duration // avg per lookup
	Model     time.Duration // avg per lookup
	Speedup   float64
	Found     int
	Learned   int64
	Skipped   int64
	ModelHits int64
}

// Compare runs baseline vs model lookups on a synthetic dataset and
// reports average latency for each path (CLI entry point).
func Compare(dataset string, nKeys, nOps int, mode learn.Mode) (Result, error) {
	res := Result{Dataset: dataset}
	ds := Build(dataset, nKeys)

	// Baseline run.
	{
		dir, err := os.MkdirTemp("", "bourbon-base-*")
		if err != nil {
			return res, err
		}
		defer os.RemoveAll(dir)
		db, err := lsm.Open(lsm.Options{
			Dir:           filepath.Join(dir, "db"),
			MemtableBytes: 64 << 20,
			LearnMode:     learn.ModeNever,
			DisableWAL:    true,
		})
		if err != nil {
			return res, err
		}
		load(db, ds.Keys, 64)
		q := ds.Queries(nOps, 7)
		start := time.Now()
		found := 0
		for _, k := range q {
			if _, ok, err := db.Get(k); err != nil {
				db.Close()
				return res, err
			} else if ok {
				found++
			}
		}
		elapsed := time.Since(start)
		res.Baseline = time.Duration(int64(elapsed) / int64(nOps))
		res.Found = found
		db.Close()
	}

	// Model run.
	{
		dir, err := os.MkdirTemp("", "bourbon-model-*")
		if err != nil {
			return res, err
		}
		defer os.RemoveAll(dir)
		db, err := lsm.Open(lsm.Options{
			Dir:           filepath.Join(dir, "db"),
			MemtableBytes: 64 << 20,
			LearnMode:     mode,
			DisableWAL:    true,
			TWait:         20 * time.Millisecond,
		})
		if err != nil {
			return res, err
		}
		load(db, ds.Keys, 64)
		// Wait for at least one model (skip when learning is off).
		if mode != learn.ModeNever {
			deadline := time.Now().Add(30 * time.Second)
			for time.Now().Before(deadline) {
				l, _, _ := db.Learner().Stats()
				if l >= 1 {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
		res.Learned, res.Skipped, _ = db.Learner().Stats()
		q := ds.Queries(nOps, 7)
		start := time.Now()
		for _, k := range q {
			if _, ok, err := db.Get(k); err != nil {
				db.Close()
				return res, err
			} else if !ok {
				db.Close()
				return res, fmt.Errorf("model run: missing key %d", k)
			}
		}
		elapsed := time.Since(start)
		res.Model = time.Duration(int64(elapsed) / int64(nOps))
		res.ModelHits = func() int64 { _, m := db.Stats(); return m }()
		db.Close()
	}

	if res.Model > 0 {
		res.Speedup = float64(res.Baseline) / float64(res.Model)
	}
	return res, nil
}
