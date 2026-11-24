package lsm

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"bourbon/learn"
)

func openTestDB(t *testing.T, mutate func(*Options)) *DB {
	t.Helper()
	opts := Options{
		Dir:            t.TempDir(),
		MemtableBytes:  64 << 10, // small: flush often
		L0FileTrigger:  4,
		BaseLevelBytes: 256 << 10,
		LearnMode:      learn.ModeNever,
		DisableWAL:     true,
		TWait:          5 * time.Millisecond,
	}
	if mutate != nil {
		mutate(&opts)
	}
	db, err := Open(opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestPutGetDelete(t *testing.T) {
	db := openTestDB(t, nil)

	if err := db.Put(1, []byte("one")); err != nil {
		t.Fatal(err)
	}
	v, ok, err := db.Get(1)
	if err != nil || !ok || string(v) != "one" {
		t.Fatalf("get: %q ok=%v err=%v", v, ok, err)
	}

	if err := db.Delete(1); err != nil {
		t.Fatal(err)
	}
	_, ok, err = db.Get(1)
	if err != nil || ok {
		t.Fatalf("deleted key still present ok=%v err=%v", ok, err)
	}
}

func TestFlushAndReadFromSSTable(t *testing.T) {
	db := openTestDB(t, nil)
	for i := uint64(0); i < 200; i++ {
		if err := db.Put(i, []byte(fmt.Sprintf("v%d", i))); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Flush(); err != nil {
		t.Fatal(err)
	}
	// Memtable empty; data in L0.
	for i := uint64(0); i < 200; i++ {
		v, ok, err := db.Get(i)
		if err != nil || !ok {
			t.Fatalf("key %d: ok=%v err=%v", i, ok, err)
		}
		if string(v) != fmt.Sprintf("v%d", i) {
			t.Fatalf("key %d = %q", i, v)
		}
	}
	lv := db.Levels()
	if lv[0] == 0 {
		t.Fatalf("expected L0 files, got %v", lv)
	}
}

func TestOverwriteAcrossFlushes(t *testing.T) {
	db := openTestDB(t, nil)
	for i := uint64(0); i < 50; i++ {
		db.Put(i, []byte("first"))
	}
	db.Flush()
	for i := uint64(0); i < 50; i++ {
		db.Put(i, []byte("second"))
	}
	db.Flush()

	for i := uint64(0); i < 50; i++ {
		v, ok, err := db.Get(i)
		if err != nil || !ok {
			t.Fatalf("key %d missing", i)
		}
		if string(v) != "second" {
			t.Fatalf("key %d = %q, want second", i, v)
		}
	}
}

func TestDeleteAfterFlush(t *testing.T) {
	db := openTestDB(t, nil)
	db.Put(42, []byte("answer"))
	db.Flush()
	db.Delete(42)
	db.Flush()

	_, ok, err := db.Get(42)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("deleted key found")
	}
}

func TestCompactionPreservesData(t *testing.T) {
	db := openTestDB(t, func(o *Options) {
		o.MemtableBytes = 8 << 10
		o.L0FileTrigger = 2
		o.BaseLevelBytes = 32 << 10
		o.LevelRatio = 4
	})
	const n = 500
	for i := uint64(0); i < n; i++ {
		db.Put(i, bytes.Repeat([]byte{byte(i)}, 32))
	}
	db.Flush()
	// Force more L0 files.
	for i := uint64(0); i < n; i++ {
		db.Put(i, []byte(fmt.Sprintf("upd%d", i)))
	}
	db.Flush()
	for i := uint64(0); i < n; i++ {
		db.Put(i, []byte(fmt.Sprintf("final%d", i)))
	}
	db.Flush()
	db.CompactAll()

	for i := uint64(0); i < n; i++ {
		v, ok, err := db.Get(i)
		if err != nil || !ok {
			t.Fatalf("key %d: ok=%v err=%v levels=%v", i, ok, err, db.Levels())
		}
		if string(v) != fmt.Sprintf("final%d", i) {
			t.Fatalf("key %d = %q", i, v)
		}
	}
}

func TestModelPathAfterLearning(t *testing.T) {
	db := openTestDB(t, func(o *Options) {
		o.LearnMode = learn.ModeAlways
		o.TWait = 2 * time.Millisecond
	})
	const n = 300
	for i := uint64(0); i < n; i++ {
		db.Put(i, []byte(fmt.Sprintf("val%d", i)))
	}
	if err := db.Flush(); err != nil {
		t.Fatal(err)
	}

	// Wait for learner to train.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		learned, _, _ := db.Learner().Stats()
		if learned > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	learned, _, _ := db.Learner().Stats()
	if learned == 0 {
		t.Fatal("no model trained")
	}

	// Reads must still be correct via model path.
	for i := uint64(0); i < n; i++ {
		v, ok, err := db.Get(i)
		if err != nil || !ok || string(v) != fmt.Sprintf("val%d", i) {
			t.Fatalf("key %d: ok=%v v=%q err=%v", i, ok, v, err)
		}
	}
	_, modelHits := db.Stats()
	if modelHits == 0 {
		t.Fatal("expected model path hits")
	}
}

func TestCbaLearnsLessThanAlwaysUnderChurn(t *testing.T) {
	// Write-heavy churn, no lookups. Always must never skip. CBA must
	// bootstrap then skip once level stats exist (from compaction).
	// Absolute learned counts vary with compaction timing, so we only
	// assert the skip behavior.
	run := func(mode learn.Mode) (learned, skipped int64) {
		db := openTestDB(t, func(o *Options) {
			o.LearnMode = mode
			o.TWait = 2 * time.Millisecond
			o.MemtableBytes = 4 << 10
			o.L0FileTrigger = 2
		})
		// Enough rounds that compaction retires files and fills
		// levelStats, so CBA can leave bootstrap and start skipping.
		for round := 0; round < 12; round++ {
			for i := uint64(0); i < 100; i++ {
				db.Put(uint64(round)*100+i, []byte(fmt.Sprintf("r%d", round)))
			}
			db.Flush()
			time.Sleep(5 * time.Millisecond)
		}
		// Wait until counters stop changing so Close does not race
		// the background trainer.
		deadline := time.Now().Add(2 * time.Second)
		var prevL, prevS int64 = -1, -1
		stable := 0
		for time.Now().Before(deadline) {
			l, s, _ := db.Learner().Stats()
			if l == prevL && s == prevS {
				stable++
				if stable >= 5 {
					break
				}
			} else {
				stable = 0
				prevL, prevS = l, s
			}
			time.Sleep(10 * time.Millisecond)
		}
		learned, skipped, _ = db.Learner().Stats()
		db.Close()
		return
	}

	lAlways, sAlways := run(learn.ModeAlways)
	lCBA, sCBA := run(learn.ModeCostBenefit)

	if lAlways == 0 {
		t.Fatal("always learned nothing")
	}
	// Always mode never skips a registered file.
	if sAlways != 0 {
		t.Fatalf("always skipped %d, want 0", sAlways)
	}
	// Under churn with no lookups, CBA must skip at least one file
	// after it leaves bootstrap.
	if sCBA == 0 {
		t.Fatalf("cba skipped nothing (learned %d); want skips under churn", lCBA)
	}
	// CBA must not learn more files than always when both see the
	// same churn pattern. Learned counts can differ slightly because
	// compaction retires files at different times; allow a small gap
	// but CBA must be the more selective mode overall.
	if lCBA > lAlways {
		t.Fatalf("cba learned %d > always %d", lCBA, lAlways)
	}
}

func TestNeverModeUsesBaselineOnly(t *testing.T) {
	db := openTestDB(t, func(o *Options) {
		o.LearnMode = learn.ModeNever
	})
	db.Put(1, []byte("x"))
	db.Flush()
	time.Sleep(20 * time.Millisecond)
	if _, ok, _ := db.Get(1); !ok {
		t.Fatal("missing")
	}
	learned, _, _ := db.Learner().Stats()
	if learned != 0 {
		t.Fatalf("never mode learned %d", learned)
	}
	baseline, model := db.Stats()
	if model != 0 {
		t.Fatalf("model hits = %d", model)
	}
	if baseline == 0 {
		t.Fatal("expected baseline hits")
	}
}

func TestRangeOfKeysAcrossLevels(t *testing.T) {
	db := openTestDB(t, func(o *Options) {
		o.MemtableBytes = 4 << 10
		o.L0FileTrigger = 2
	})
	// Insert ascending, flush multiple times to create L0 files,
	// then compact into L1.
	for batch := 0; batch < 5; batch++ {
		base := uint64(batch * 1000)
		for i := uint64(0); i < 100; i++ {
			db.Put(base+i, []byte(fmt.Sprintf("b%d-%d", batch, i)))
		}
		db.Flush()
	}
	db.CompactAll()

	for batch := 0; batch < 5; batch++ {
		base := uint64(batch * 1000)
		for i := uint64(0); i < 100; i++ {
			v, ok, err := db.Get(base + i)
			if err != nil || !ok {
				t.Fatalf("key %d: ok=%v err=%v levels=%v", base+i, ok, err, db.Levels())
			}
			if string(v) != fmt.Sprintf("b%d-%d", batch, i) {
				t.Fatalf("key %d = %q", base+i, v)
			}
		}
	}
}
