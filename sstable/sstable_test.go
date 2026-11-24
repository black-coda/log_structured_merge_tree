package sstable

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"bourbon/plr"
)

func writeTable(t *testing.T, dir string, keys []uint64) (*Reader, Meta) {
	t.Helper()
	path := filepath.Join(dir, fmt.Sprintf("%d.sst", keys[0]))
	w, err := NewWriter(path, Options{BlockSize: 256}) // small blocks for tests
	if err != nil {
		t.Fatal(err)
	}
	for i, k := range keys {
		ptr := ValuePointer{LogID: 1, Offset: uint64(i * 100), Size: 50}
		if err := w.Add(k, ptr); err != nil {
			t.Fatal(err)
		}
	}
	meta, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close(); os.Remove(path) })
	return r, meta
}

func TestWriteReadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	keys := make([]uint64, 500)
	for i := range keys {
		keys[i] = uint64(i * 3)
	}
	r, meta := writeTable(t, dir, keys)
	if meta.NumRecs != 500 {
		t.Fatalf("num recs = %d", meta.NumRecs)
	}
	if meta.MinKey != 0 || meta.MaxKey != 499*3 {
		t.Fatalf("bounds = %d..%d", meta.MinKey, meta.MaxKey)
	}
	if meta.NumBlocks < 2 {
		t.Fatalf("expected multiple blocks, got %d", meta.NumBlocks)
	}

	for i, k := range keys {
		res, err := r.Get(k)
		if err != nil {
			t.Fatal(err)
		}
		if !res.Found {
			t.Fatalf("missing key %d", k)
		}
		wantOff := uint64(i * 100)
		if res.Pointer.Offset != wantOff || res.Pointer.Size != 50 {
			t.Fatalf("key %d: ptr = %+v", k, res.Pointer)
		}
	}
}

func TestMissingKey(t *testing.T) {
	dir := t.TempDir()
	keys := []uint64{10, 20, 30, 40}
	r, _ := writeTable(t, dir, keys)
	for _, miss := range []uint64{5, 15, 35, 100} {
		res, err := r.Get(miss)
		if err != nil {
			t.Fatal(err)
		}
		if res.Found {
			t.Fatalf("unexpected find for %d", miss)
		}
	}
}

func TestModelPathMatchesBaseline(t *testing.T) {
	dir := t.TempDir()
	keys := make([]uint64, 1000)
	for i := range keys {
		keys[i] = uint64(i)
	}
	r, _ := writeTable(t, dir, keys)

	// Train and attach a model.
	k, err := r.Keys()
	if err != nil {
		t.Fatal(err)
	}
	model := plr.Train(k, plr.DefaultDelta)
	r.SetModel(model)

	for i, key := range keys {
		base, err := r.Get(key)
		if err != nil {
			t.Fatal(err)
		}
		via, err := r.GetWithModel(key)
		if err != nil {
			t.Fatal(err)
		}
		if !base.Found || !via.Found {
			t.Fatalf("key %d not found base=%v model=%v", key, base.Found, via.Found)
		}
		if base.Pointer != via.Pointer {
			t.Fatalf("key %d pointer mismatch", key)
		}
		if !via.UsedModel {
			t.Fatalf("model path not used for key %d (i=%d)", key, i)
		}
	}
}

func TestKeysReRead(t *testing.T) {
	dir := t.TempDir()
	keys := make([]uint64, 300)
	for i := range keys {
		keys[i] = uint64(1000 - i) // descending insert order reversed below
	}
	// Ascending:
	for i := range keys {
		keys[i] = uint64(i * 7)
	}
	r, _ := writeTable(t, dir, keys)
	got, err := r.Keys()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(keys) {
		t.Fatalf("len = %d", len(got))
	}
	for i := range keys {
		if got[i] != keys[i] {
			t.Fatalf("keys[%d] = %d want %d", i, got[i], keys[i])
		}
	}
}

func TestEmptyTable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.sst")
	w, err := NewWriter(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Finish(); err != nil {
		t.Fatal(err)
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	res, err := r.Get(1)
	if err != nil {
		t.Fatal(err)
	}
	if res.Found {
		t.Fatal("empty table should not find")
	}
}

func TestRecordAt(t *testing.T) {
	dir := t.TempDir()
	keys := make([]uint64, 100)
	for i := range keys {
		keys[i] = uint64(i)
	}
	r, _ := writeTable(t, dir, keys)
	for i, k := range keys {
		got, ptr, ok, err := r.RecordAt(int64(i))
		if err != nil || !ok {
			t.Fatalf("RecordAt(%d): ok=%v err=%v", i, ok, err)
		}
		if got != k {
			t.Fatalf("RecordAt(%d) key = %d want %d", i, got, k)
		}
		if ptr.Offset != uint64(i*100) {
			t.Fatalf("ptr offset = %d", ptr.Offset)
		}
	}
}
