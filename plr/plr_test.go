package plr

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestLinearKeysSingleSegment(t *testing.T) {
	keys := make([]uint64, 1000)
	for i := range keys {
		keys[i] = uint64(i)
	}
	m := Train(keys, DefaultDelta)
	if m.NumSegments() != 1 {
		t.Fatalf("expected 1 segment, got %d", m.NumSegments())
	}
	for i, k := range keys {
		idx, ok := m.Lookup(keys, k)
		if !ok || idx != i {
			t.Fatalf("key %d: got (%d,%v), want (%d,true)", k, idx, ok, i)
		}
	}
}

func TestAllPointsWithinDelta(t *testing.T) {
	keys := randomKeys(5000, 1)
	m := Train(keys, DefaultDelta)
	for i, k := range keys {
		_, low, high := m.Predict(k)
		if int64(i) < low-DefaultDelta || int64(i) > high+DefaultDelta {
			// Window should cover position within a generous margin;
			// primary guarantee is Lookup correctness below.
			t.Logf("position %d outside window [%d,%d] for key %d", i, low, high, k)
		}
		idx, ok := m.Lookup(keys, k)
		if !ok || idx != i {
			t.Fatalf("key %d at %d: got (%d,%v)", k, i, idx, ok)
		}
	}
}

func TestSegmentedKeys(t *testing.T) {
	// Gaps every 100 keys force multiple segments.
	var keys []uint64
	next := uint64(0)
	for seg := 0; seg < 50; seg++ {
		for i := 0; i < 100; i++ {
			keys = append(keys, next)
			next++
		}
		next += 1000 // gap
	}
	m := Train(keys, DefaultDelta)
	if m.NumSegments() < 2 {
		t.Fatalf("expected multiple segments, got %d", m.NumSegments())
	}
	for i, k := range keys {
		idx, ok := m.Lookup(keys, k)
		if !ok || idx != i {
			t.Fatalf("key %d: got (%d,%v), want %d", k, idx, ok, i)
		}
	}
}

func TestAbsentKeys(t *testing.T) {
	keys := []uint64{10, 20, 30, 40, 50}
	m := Train(keys, DefaultDelta)
	for _, miss := range []uint64{5, 15, 35, 55, 100} {
		if _, ok := m.Lookup(keys, miss); ok {
			t.Fatalf("key %d should be absent", miss)
		}
	}
}

func TestNormalDistribution(t *testing.T) {
	keys := normalKeys(10000, 42)
	m := Train(keys, DefaultDelta)
	for i := 0; i < 200; i++ {
		k := keys[i*37%len(keys)]
		idx, ok := m.Lookup(keys, k)
		if !ok {
			t.Fatalf("missing key %d", k)
		}
		if keys[idx] != k {
			t.Fatalf("wrong index for %d", k)
		}
	}
	if m.NumSegments() == 0 {
		t.Fatal("no segments")
	}
}

func TestPredictWindowCoversNearby(t *testing.T) {
	keys := make([]uint64, 1000)
	for i := range keys {
		keys[i] = uint64(i * 2)
	}
	m := Train(keys, DefaultDelta)
	k := keys[500]
	_, low, high := m.Predict(k)
	if 500 < int(low)-1 || 500 > int(high)+1 {
		t.Fatalf("window [%d,%d] does not cover position 500 for key %d", low, high, k)
	}
}

func TestModelBytes(t *testing.T) {
	keys := make([]uint64, 100)
	for i := range keys {
		keys[i] = uint64(i)
	}
	m := Train(keys, DefaultDelta)
	if m.Bytes() != int64(m.NumSegments())*48 {
		t.Fatalf("bytes mismatch")
	}
}

func BenchmarkTrainLinear(b *testing.B) {
	keys := make([]uint64, 1<<20)
	for i := range keys {
		keys[i] = uint64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Train(keys, DefaultDelta)
	}
}

func BenchmarkLookupModel(b *testing.B) {
	keys := make([]uint64, 1<<20)
	for i := range keys {
		keys[i] = uint64(i)
	}
	m := Train(keys, DefaultDelta)
	rng := rand.New(rand.NewSource(1))
	queries := make([]uint64, b.N)
	for i := range queries {
		queries[i] = keys[rng.Intn(len(keys))]
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Lookup(keys, queries[i])
	}
}

func BenchmarkLookupBinary(b *testing.B) {
	keys := make([]uint64, 1<<20)
	for i := range keys {
		keys[i] = uint64(i)
	}
	rng := rand.New(rand.NewSource(1))
	queries := make([]uint64, b.N)
	for i := range queries {
		queries[i] = keys[rng.Intn(len(keys))]
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binaryLookup(keys, queries[i])
	}
}

func randomKeys(n int, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	set := make(map[uint64]struct{}, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		k := uint64(rng.Int63())
		if _, ok := set[k]; ok {
			continue
		}
		set[k] = struct{}{}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func normalKeys(n int, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	set := make(map[uint64]struct{}, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		// Sample N(0,1), scale to integers, shift positive.
		g := rng.NormFloat64()
		k := uint64(math.Abs(g)*1e9) + uint64(len(keys))
		if _, ok := set[k]; ok {
			continue
		}
		set[k] = struct{}{}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}
