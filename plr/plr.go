// Package plr implements Greedy piecewise linear regression (Greedy-PLR)
// for learned indexes over sorted key sequences.
//
// A model maps each key to an approximate position (record index) in a
// sorted dataset. The model is a set of line segments. Each segment has
// a slope and intercept. Lookups use binary search to find the segment,
// evaluate the line, then search a small window around the prediction.
package plr

import (
	"math"
	"sort"
)

// DefaultDelta is the error bound used by Bourbon (paper: delta = 8).
const DefaultDelta = 8

// Segment is one line of the model.
// Position prediction: pos = slope*key + intercept.
type Segment struct {
	Slope     float64
	Intercept float64
	MinKey    uint64
	MaxKey    uint64
	// MinPos/MaxPos bound the true positions of keys in this segment.
	MinPos int64
	MaxPos int64
}

// Model is a trained piecewise linear model.
type Model struct {
	Segments []Segment
	Delta    int64
	MinPos   int64
	MaxPos   int64
}

// Trainer builds a model with Greedy-PLR in one pass over sorted points.
//
// Algorithm: maintain a feasible slope interval for a line anchored at
// the segment's first point. Each new point intersects the interval with
// the slopes that keep the line within Delta of that point. When the
// interval becomes empty, close the segment and start a new one.
// Training is O(n) and guarantees |pred - pos| <= Delta for every
// training point.
type Trainer struct {
	Delta    int64
	segments []Segment

	hasOpen  bool
	firstKey uint64
	firstPos int64
	minKey   uint64
	maxKey   uint64
	minPos   int64
	maxPos   int64
	slopeMin float64 // feasible slope lower bound (+Inf when open fresh)
	slopeMax float64 // feasible slope upper bound (-Inf when open fresh)
	n        int
}

// NewTrainer returns a trainer with the given error bound.
func NewTrainer(delta int64) *Trainer {
	if delta < 0 {
		delta = DefaultDelta
	}
	return &Trainer{Delta: delta}
}

// Add inserts one training point. Keys must arrive in non-decreasing order.
// pos is the record index of key in the sorted dataset.
func (t *Trainer) Add(key uint64, pos int64) {
	if !t.hasOpen {
		t.open(key, pos)
		return
	}
	lo, hi, ok := t.slopeBounds(key, pos)
	if !ok {
		t.close()
		t.open(key, pos)
		return
	}
	// Intersect feasible slope interval.
	if lo > t.slopeMin {
		t.slopeMin = lo
	}
	if hi < t.slopeMax {
		t.slopeMax = hi
	}
	if t.slopeMin > t.slopeMax {
		t.close()
		t.open(key, pos)
		return
	}
	t.absorb(key, pos)
}

func (t *Trainer) open(key uint64, pos int64) {
	t.hasOpen = true
	t.firstKey, t.firstPos = key, pos
	t.minKey, t.maxKey = key, key
	t.minPos, t.maxPos = pos, pos
	t.slopeMin = math.Inf(-1)
	t.slopeMax = math.Inf(1)
	t.n = 1
}

func (t *Trainer) absorb(key uint64, pos int64) {
	t.maxKey = key
	if pos < t.minPos {
		t.minPos = pos
	}
	if pos > t.maxPos {
		t.maxPos = pos
	}
	t.n++
}

// slopeBounds returns the interval of slopes s such that the line
// anchored at (firstKey, firstPos) with slope s is within Delta of
// (key, pos). ok=false means no such slope exists.
//
// Constraint: |s*(key-firstKey) - (pos-firstPos)| <= Delta.
func (t *Trainer) slopeBounds(key uint64, pos int64) (lo, hi float64, ok bool) {
	dk := float64(key) - float64(t.firstKey)
	dp := float64(pos) - float64(t.firstPos)
	d := float64(t.Delta)
	if dk == 0 {
		// Same key: only the position residual matters.
		if math.Abs(dp) <= d {
			return math.Inf(-1), math.Inf(1), true
		}
		return 0, 0, false
	}
	lo = (dp - d) / dk
	hi = (dp + d) / dk
	if lo > hi {
		lo, hi = hi, lo
	}
	return lo, hi, true
}

// line returns the slope and intercept for the closed segment.
// Uses the midpoint of the feasible slope interval (always within
// Delta of every absorbed point by construction). Single-point
// segments use a horizontal line through the point.
func (t *Trainer) line() (slope, intercept float64) {
	if t.n <= 1 {
		return 0, float64(t.firstPos)
	}
	slope = (t.slopeMin + t.slopeMax) / 2
	if math.IsInf(slope, 0) {
		// Degenerate: fall back to endpoint line.
		if t.maxKey == t.firstKey {
			return 0, float64(t.firstPos)
		}
		slope = float64(t.maxPos-t.firstPos) / float64(int64(t.maxKey)-int64(t.firstKey))
	}
	intercept = float64(t.firstPos) - slope*float64(t.firstKey)
	return slope, intercept
}

func (t *Trainer) close() {
	if !t.hasOpen {
		return
	}
	slope, intercept := t.line()
	t.segments = append(t.segments, Segment{
		Slope:     slope,
		Intercept: intercept,
		MinKey:    t.minKey,
		MaxKey:    t.maxKey,
		MinPos:    t.minPos,
		MaxPos:    t.maxPos,
	})
	t.hasOpen = false
	t.n = 0
}

// Finish seals the model.
func (t *Trainer) Finish() *Model {
	t.close()
	m := &Model{
		Segments: t.segments,
		Delta:    t.Delta,
	}
	if len(m.Segments) > 0 {
		m.MinPos = m.Segments[0].MinPos
		m.MaxPos = m.Segments[0].MaxPos
		for _, s := range m.Segments {
			if s.MinPos < m.MinPos {
				m.MinPos = s.MinPos
			}
			if s.MaxPos > m.MaxPos {
				m.MaxPos = s.MaxPos
			}
		}
	}
	return m
}

// Train builds a model from sorted keys (position = index).
func Train(keys []uint64, delta int64) *Model {
	tr := NewTrainer(delta)
	for i, k := range keys {
		tr.Add(k, int64(i))
	}
	return tr.Finish()
}

// SegmentIndex returns the segment that covers key, or the nearest one.
func (m *Model) SegmentIndex(key uint64) int {
	if len(m.Segments) == 0 {
		return -1
	}
	i := sort.Search(len(m.Segments), func(i int) bool {
		return m.Segments[i].MinKey > key
	}) - 1
	if i < 0 {
		return 0
	}
	if i >= len(m.Segments) {
		return len(m.Segments) - 1
	}
	return i
}

// Predict returns the estimated position of key and the search window
// [low, high] expected to contain the true position if key is present.
// The window is [p-delta, p+delta] (paper: pos ± delta). Callers must
// fall back to a full search when the window misses.
func (m *Model) Predict(key uint64) (pos int64, low, high int64) {
	if len(m.Segments) == 0 {
		return 0, 0, 0
	}
	s := m.Segments[m.SegmentIndex(key)]
	p := int64(math.Round(s.Slope*float64(key) + s.Intercept))

	low = p - m.Delta
	high = p + m.Delta
	if low < 0 {
		low = 0
	}
	if high < low {
		high = low
	}
	return p, low, high
}

// Lookup returns the index of key in sorted keys and whether it is present.
// It searches the model window first, then falls back to a full binary
// search so the result is always correct.
func (m *Model) Lookup(keys []uint64, key uint64) (int, bool) {
	if len(keys) == 0 {
		return -1, false
	}
	if len(m.Segments) == 0 {
		return binaryLookup(keys, key)
	}

	_, low, high := m.Predict(key)
	if high >= int64(len(keys)) {
		high = int64(len(keys)) - 1
	}
	if low < 0 {
		low = 0
	}

	if low <= high {
		// Probe predicted position first (often exact).
		if i := int((low + high) / 2); i >= 0 && i < len(keys) && keys[i] == key {
			return i, true
		}
		lo, hi := low, high
		for lo <= hi {
			mid := lo + (hi-lo)/2
			v := keys[mid]
			if v == key {
				return int(mid), true
			}
			if v < key {
				lo = mid + 1
			} else {
				hi = mid - 1
			}
		}
		// Check the boundary cells the window missed.
		if lo < int64(len(keys)) && keys[lo] == key {
			return int(lo), true
		}
		if hi >= 0 && keys[hi] == key {
			return int(hi), true
		}
	}

	// Correctness fallback (model error can exceed Delta at boundaries).
	return binaryLookup(keys, key)
}

func binaryLookup(keys []uint64, key uint64) (int, bool) {
	i := sort.Search(len(keys), func(i int) bool {
		return keys[i] >= key
	})
	if i < len(keys) && keys[i] == key {
		return i, true
	}
	return -1, false
}

// NumSegments returns the number of line segments.
func (m *Model) NumSegments() int {
	if m == nil {
		return 0
	}
	return len(m.Segments)
}

// Bytes estimates model space overhead in bytes.
// Each segment stores 4 float64 and 2 int64 values (48 bytes).
func (m *Model) Bytes() int64 {
	if m == nil {
		return 0
	}
	return int64(len(m.Segments)) * 48
}
