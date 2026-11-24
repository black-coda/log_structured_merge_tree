package memtable

import (
	"path/filepath"
	"testing"

	"bourbon/sstable"
)

func ptr(off uint64) sstable.ValuePointer {
	return sstable.ValuePointer{LogID: 1, Offset: off, Size: 10}
}

func TestPutGetDelete(t *testing.T) {
	m := New()
	m.Put(1, ptr(10), 5, 1)
	m.Put(2, ptr(20), 5, 2)
	m.Delete(1, 3)

	e, ok := m.Get(1)
	if !ok || !e.Tomb {
		t.Fatalf("expected tombstone, got %+v ok=%v", e, ok)
	}
	e, ok = m.Get(2)
	if !ok || e.Tomb || e.Pointer.Offset != 20 {
		t.Fatalf("bad get: %+v", e)
	}
	if m.Len() != 2 {
		t.Fatalf("len = %d", m.Len())
	}
}

func TestSortedOrder(t *testing.T) {
	m := New()
	for _, k := range []uint64{30, 10, 20, 5} {
		m.Put(k, ptr(uint64(k)), 1, uint64(k))
	}
	keys := m.SortedKeys()
	want := []uint64{5, 10, 20, 30}
	for i := range want {
		if keys[i] != want[i] {
			t.Fatalf("keys = %v", keys)
		}
	}
}

func TestWALRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "000001.wal")
	w, err := NewWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AppendPut(7, ptr(70), []byte("v7"), 7); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendDelete(8, 8); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	var got []Entry
	err = Replay(path, func(e Entry) error {
		got = append(got, e)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("replayed %d records", len(got))
	}
	if got[0].Key != 7 || got[0].Tomb || got[0].Pointer.Offset != 70 {
		t.Fatalf("rec0 = %+v", got[0])
	}
	if got[1].Key != 8 || !got[1].Tomb {
		t.Fatalf("rec1 = %+v", got[1])
	}
}
