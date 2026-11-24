package bloom

import (
	"fmt"
	"testing"
)

func TestAddContains(t *testing.T) {
	f := New(1000)
	keys := make([][]byte, 1000)
	for i := range keys {
		keys[i] = []byte(fmt.Sprintf("key-%d", i))
		f.Add(keys[i])
	}
	for _, k := range keys {
		if !f.MayContain(k) {
			t.Fatalf("false negative for %s", k)
		}
	}
}

func TestFalsePositiveRate(t *testing.T) {
	const n = 10000
	f := New(n)
	for i := 0; i < n; i++ {
		f.Add([]byte(fmt.Sprintf("present-%d", i)))
	}
	fp := 0
	const trials = 10000
	for i := 0; i < trials; i++ {
		if f.MayContain([]byte(fmt.Sprintf("absent-%d", i))) {
			fp++
		}
	}
	rate := float64(fp) / trials
	if rate > 0.05 {
		t.Fatalf("false positive rate too high: %.4f", rate)
	}
}

func TestEncodeDecode(t *testing.T) {
	f := New(500)
	for i := 0; i < 500; i++ {
		f.Add([]byte(fmt.Sprintf("k%d", i)))
	}
	data := f.Encode()
	g := Decode(data)
	if g == nil {
		t.Fatal("decode failed")
	}
	if g.K() != f.K() || g.M() != f.M() {
		t.Fatalf("params mismatch: (%d,%d) vs (%d,%d)", g.K(), g.M(), f.K(), f.M())
	}
	for i := 0; i < 500; i++ {
		k := []byte(fmt.Sprintf("k%d", i))
		if !g.MayContain(k) {
			t.Fatalf("missing after roundtrip: %s", k)
		}
	}
}

func TestNilFilter(t *testing.T) {
	var f *Filter
	if !f.MayContain([]byte("x")) {
		t.Fatal("nil filter must return true")
	}
}
