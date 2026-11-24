package vlog

import (
	"bytes"
	"testing"

	"bourbon/sstable"
)

func TestAppendRead(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(dir)
	defer m.Close()

	type item struct {
		ptr sstable.ValuePointer
		val []byte
	}
	values := [][]byte{
		[]byte("hello world"),
		bytes.Repeat([]byte("x"), 1000),
		{},
		[]byte("final"),
	}
	var items []item
	for _, v := range values {
		p, err := m.Append(v)
		if err != nil {
			t.Fatal(err)
		}
		items = append(items, item{p, v})
	}
	for _, it := range items {
		out, err := m.Read(it.ptr)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out, it.val) {
			t.Fatalf("mismatch: got %d bytes, want %d", len(out), len(it.val))
		}
	}
}

func TestReadFromReopenedManager(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(dir)
	v := []byte("persisted value")
	p, err := m.Append(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}

	m2 := NewManager(dir)
	defer m2.Close()
	out, err := m2.Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, v) {
		t.Fatalf("got %q", out)
	}
}
