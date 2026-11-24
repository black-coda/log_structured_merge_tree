// Package bench provides synthetic datasets and end-to-end benchmarks
// that compare baseline lookups against Bourbon's learned-model path.
package bench

import (
	"math"
	"math/rand"
	"sort"
)

// Dataset is a sorted set of uint64 keys.
type Dataset struct {
	Name string
	Keys []uint64
}

// Linear returns consecutive keys 0..n-1 (paper: "linear").
func Linear(n int) Dataset {
	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = uint64(i)
	}
	return Dataset{Name: "linear", Keys: keys}
}

// Segmented returns consecutive runs with gaps. gapEvery keys form a
// run; then a gap of `gap` is inserted (paper: seg-1%, seg-10%).
func Segmented(n, runLen, gap int) Dataset {
	keys := make([]uint64, 0, n)
	next := uint64(0)
	for len(keys) < n {
		for i := 0; i < runLen && len(keys) < n; i++ {
			keys = append(keys, next)
			next++
		}
		next += uint64(gap)
	}
	return Dataset{Name: "segmented", Keys: keys}
}

// Normal samples unique keys from a scaled normal distribution
// (paper: "normal" dataset).
func Normal(n int, seed int64) Dataset {
	rng := rand.New(rand.NewSource(seed))
	set := make(map[uint64]struct{}, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		g := rng.NormFloat64()
		v := math.Abs(g) * 1e9
		k := uint64(v) + uint64(len(keys))
		if _, ok := set[k]; ok {
			continue
		}
		set[k] = struct{}{}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return Dataset{Name: "normal", Keys: keys}
}

// RandomUnique returns n unique random keys in ascending order.
func RandomUnique(n int, seed int64) Dataset {
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
	return Dataset{Name: "random", Keys: keys}
}

// Queries returns n random keys drawn from ds (with replacement).
func (d Dataset) Queries(n int, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	q := make([]uint64, n)
	for i := range q {
		q[i] = d.Keys[rng.Intn(len(d.Keys))]
	}
	return q
}

// ZipfianQueries returns n queries following a Zipf-like skew:
// most requests hit a small hot set (paper request distributions).
func (d Dataset) ZipfianQueries(n int, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	// Hot set: first 25% of keys get 80% of traffic.
	hot := len(d.Keys) / 4
	if hot < 1 {
		hot = 1
	}
	q := make([]uint64, n)
	for i := range q {
		if rng.Float64() < 0.8 {
			q[i] = d.Keys[rng.Intn(hot)]
		} else {
			q[i] = d.Keys[rng.Intn(len(d.Keys))]
		}
	}
	return q
}

// Build returns a dataset by name: linear|seg1|seg10|normal|random.
func Build(name string, n int) Dataset {
	switch name {
	case "linear":
		return Linear(n)
	case "seg1":
		// Gap every 100 keys (~1% boundary rate).
		return Segmented(n, 100, 1000)
	case "seg10":
		// Gap every 10 keys.
		return Segmented(n, 10, 100)
	case "normal":
		return Normal(n, 42)
	default:
		return RandomUnique(n, 42)
	}
}
