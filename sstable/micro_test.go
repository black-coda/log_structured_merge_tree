package sstable

import (
	"path/filepath"
	"testing"

	"bourbon/plr"
)

func BenchmarkGetBaseline(b *testing.B) {
	dir := b.TempDir()
	keys := make([]uint64, 200000)
	for i := range keys {
		keys[i] = uint64(i)
	}
	path := filepath.Join(dir, "t.sst")
	w, _ := NewWriter(path, Options{BlockSize: 4096})
	for i, k := range keys {
		w.Add(k, ValuePointer{LogID: 1, Offset: uint64(i * 64), Size: 64})
	}
	w.Finish()
	r, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	q := make([]uint64, b.N)
	for i := range q {
		q[i] = keys[i*7919%len(keys)]
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res, err := r.Get(q[i]); err != nil || !res.Found {
			b.Fatal("miss")
		}
	}
}

func BenchmarkGetModel(b *testing.B) {
	dir := b.TempDir()
	keys := make([]uint64, 200000)
	for i := range keys {
		keys[i] = uint64(i)
	}
	path := filepath.Join(dir, "t.sst")
	w, _ := NewWriter(path, Options{BlockSize: 4096})
	for i, k := range keys {
		w.Add(k, ValuePointer{LogID: 1, Offset: uint64(i * 64), Size: 64})
	}
	w.Finish()
	r, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	kk, _ := r.Keys()
	r.SetModel(plr.Train(kk, plr.DefaultDelta))
	q := make([]uint64, b.N)
	for i := range q {
		q[i] = keys[i*7919%len(keys)]
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res, err := r.GetWithModel(q[i]); err != nil || !res.Found {
			b.Fatal("miss")
		}
	}
}
