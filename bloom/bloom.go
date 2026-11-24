// Package bloom implements a simple Bloom filter used as the SSTable
// filter block. Bourbon queries the filter before loading a data block
// so that negative lookups end early.
package bloom

import (
	"encoding/binary"
	"hash/fnv"
	"math"
)

// Filter is a Bloom filter over a set of keys.
type Filter struct {
	bits []uint64
	k    uint32
	m    uint64 // number of bits (multiple of 64)
}

// New creates a filter sized for n expected keys with fp rate ~1%.
func New(n int) *Filter {
	if n < 1 {
		n = 1
	}
	// m = -n*ln(p)/ln(2)^2, p=0.01 => ~9.6 bits/key
	m := uint64(math.Ceil(float64(n) * 9.6))
	m = (m + 63) / 64 * 64
	if m < 64 {
		m = 64
	}
	// k = m/n * ln(2)
	k := uint32(math.Max(1, math.Round(float64(m)/float64(n)*math.Ln2)))
	return &Filter{
		bits: make([]uint64, m/64),
		k:    k,
		m:    m,
	}
}

// NewWithParams creates a filter with explicit bit count and hash count.
func NewWithParams(m uint64, k uint32) *Filter {
	if m < 64 {
		m = 64
	}
	m = (m + 63) / 64 * 64
	if k < 1 {
		k = 1
	}
	return &Filter{
		bits: make([]uint64, m/64),
		k:    k,
		m:    m,
	}
}

func hashes(key []byte) (h1, h2 uint64) {
	fn := fnv.New64a()
	fn.Write(key)
	h1 = fn.Sum64()
	// Second hash: FNV-1a over reversed bytes with different offset basis mix.
	fn2 := fnv.New64a()
	for i := len(key) - 1; i >= 0; i-- {
		fn2.Write([]byte{key[i]})
	}
	h2 = fn2.Sum64()
	if h2 == 0 {
		h2 = 0x9e3779b97f4a7c15
	}
	return h1, h2
}

// Add inserts key into the filter.
func (f *Filter) Add(key []byte) {
	if f == nil || f.m == 0 {
		return
	}
	h1, h2 := hashes(key)
	for i := uint32(0); i < f.k; i++ {
		pos := (h1 + uint64(i)*h2) % f.m
		f.bits[pos/64] |= 1 << (pos % 64)
	}
}

// MayContain reports whether key may be present (true = maybe).
func (f *Filter) MayContain(key []byte) bool {
	if f == nil || f.m == 0 {
		return true
	}
	h1, h2 := hashes(key)
	for i := uint32(0); i < f.k; i++ {
		pos := (h1 + uint64(i)*h2) % f.m
		if f.bits[pos/64]&(1<<(pos%64)) == 0 {
			return false
		}
	}
	return true
}

// K returns the number of hash functions.
func (f *Filter) K() uint32 { return f.k }

// M returns the number of bits.
func (f *Filter) M() uint64 { return f.m }

// Encode serializes the filter: [m uint64][k uint32][bits...].
func (f *Filter) Encode() []byte {
	buf := make([]byte, 12+len(f.bits)*8)
	binary.BigEndian.PutUint64(buf[0:8], f.m)
	binary.BigEndian.PutUint32(buf[8:12], f.k)
	for i, w := range f.bits {
		binary.BigEndian.PutUint64(buf[12+i*8:], w)
	}
	return buf
}

// Decode parses a serialized filter.
func Decode(data []byte) *Filter {
	if len(data) < 12 {
		return nil
	}
	m := binary.BigEndian.Uint64(data[0:8])
	k := binary.BigEndian.Uint32(data[8:12])
	words := (m + 63) / 64
	if uint64(len(data)) < 12+words*8 {
		return nil
	}
	bits := make([]uint64, words)
	for i := uint64(0); i < words; i++ {
		bits[i] = binary.BigEndian.Uint64(data[12+i*8:])
	}
	return &Filter{bits: bits, k: k, m: m}
}
