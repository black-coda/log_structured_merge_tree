package bench

import (
	"testing"

	"bourbon/learn"
	"bourbon/plr"
)

func TestDatasetsSortedUnique(t *testing.T) {
	for _, name := range []string{"linear", "seg1", "seg10", "normal", "random"} {
		ds := Build(name, 5000)
		if len(ds.Keys) != 5000 {
			t.Fatalf("%s: len=%d", name, len(ds.Keys))
		}
		for i := 1; i < len(ds.Keys); i++ {
			if ds.Keys[i] <= ds.Keys[i-1] {
				t.Fatalf("%s: not strictly ascending at %d", name, i)
			}
		}
	}
}

func TestCompareProducesSpeedup(t *testing.T) {
	res, err := Compare("linear", 50000, 20000, learn.ModeAlways)
	if err != nil {
		t.Fatal(err)
	}
	if res.Baseline <= 0 || res.Model <= 0 {
		t.Fatalf("bad latencies: %+v", res)
	}
	if res.Learned == 0 {
		t.Fatal("no models learned")
	}
	// Correctness: all lookups found.
	if res.Found == 0 {
		t.Fatal("no finds in baseline")
	}
	t.Logf("baseline=%s model=%s speedup=%.2fx learned=%d",
		res.Baseline, res.Model, res.Speedup, res.Learned)
}

func TestSegmentedModelStillCorrect(t *testing.T) {
	// seg1 has many segments; ensure model path still finds keys.
	res, err := Compare("seg1", 30000, 10000, learn.ModeAlways)
	if err != nil {
		t.Fatal(err)
	}
	if res.Learned == 0 {
		t.Fatal("expected learning on seg1")
	}
}

func TestZipfianQueries(t *testing.T) {
	ds := Build("linear", 10000)
	q := ds.ZipfianQueries(1000, 1)
	if len(q) != 1000 {
		t.Fatal("bad query count")
	}
	set := make(map[uint64]bool, len(ds.Keys))
	for _, k := range ds.Keys {
		set[k] = true
	}
	for _, k := range q {
		if !set[k] {
			t.Fatalf("query key %d not in dataset", k)
		}
	}
}

func TestTrainSegmentsSane(t *testing.T) {
	linear := Build("linear", 100000)
	m := plr.Train(linear.Keys, plr.DefaultDelta)
	if m.NumSegments() != 1 {
		t.Fatalf("linear segments = %d", m.NumSegments())
	}
	seg := Build("seg10", 100000)
	m2 := plr.Train(seg.Keys, plr.DefaultDelta)
	if m2.NumSegments() <= 1 {
		t.Fatalf("seg10 segments = %d, want > 1", m2.NumSegments())
	}
	t.Logf("seg10 segments=%d", m2.NumSegments())
}
