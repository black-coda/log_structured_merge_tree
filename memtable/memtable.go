// Package memtable implements the in-memory sorted table and a simple
// write-ahead log. New writes land here first; when the table is full
// it is flushed to an L0 SSTable.
package memtable

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"

	"bourbon/sstable"
)

// Tombstone marks a deleted key in the memtable / SSTables.
const Tombstone = 1

// Entry is one version of a key.
type Entry struct {
	Key     uint64
	Value   []byte // nil for tombstone when Tomb=true
	Pointer sstable.ValuePointer
	Tomb    bool
	// Seq is a monotonically increasing sequence number (newest wins).
	Seq uint64
}

// MemTable is a sorted in-memory table protected by a mutex.
type MemTable struct {
	mu      sync.RWMutex
	entries map[uint64]Entry
	keys    []uint64 // sorted view, rebuilt on demand
	dirty   bool
	size    int64 // approximate bytes
	seq     uint64
}

// New creates an empty memtable.
func New() *MemTable {
	return &MemTable{
		entries: make(map[uint64]Entry),
	}
}

// Put inserts or overwrites a key with a value-log pointer.
func (m *MemTable) Put(key uint64, ptr sstable.ValuePointer, valueLen int, seq uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	old, existed := m.entries[key]
	m.entries[key] = Entry{
		Key:     key,
		Pointer: ptr,
		Tomb:    false,
		Seq:     seq,
	}
	if existed {
		m.size += int64(valueLen) - int64(len(old.Value))
	} else {
		m.size += int64(valueLen) + 16
	}
	m.dirty = true
}

// PutValue stores an inline value (used when values stay in the
// memtable path; still goes to vlog in Bourbon, so mostly pointer).
func (m *MemTable) PutValue(key uint64, value []byte, seq uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key] = Entry{
		Key:   key,
		Value: append([]byte(nil), value...),
		Seq:   seq,
	}
	m.size += int64(len(value)) + 16
	m.dirty = true
}

// Delete marks a key as deleted (tombstone).
func (m *MemTable) Delete(key uint64, seq uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key] = Entry{
		Key:  key,
		Tomb: true,
		Seq:  seq,
	}
	m.size += 16
	m.dirty = true
}

// Get returns the newest entry for key.
func (m *MemTable) Get(key uint64) (Entry, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	e, ok := m.entries[key]
	return e, ok
}

// Len returns the number of keys.
func (m *MemTable) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// ApproxBytes returns the approximate memory size.
func (m *MemTable) ApproxBytes() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.size
}

// SortedKeys returns keys in ascending order.
func (m *MemTable) SortedKeys() []uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if !m.dirty && m.keys != nil {
		return m.keys
	}
	keys := make([]uint64, 0, len(m.entries))
	for k := range m.entries {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	m.keys = keys
	m.dirty = false
	return keys
}

// Iter returns entries in ascending key order.
func (m *MemTable) Iter() []Entry {
	keys := m.SortedKeys()
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]Entry, 0, len(keys))
	for _, k := range keys {
		out = append(out, m.entries[k])
	}
	return out
}

// Reset clears the table (after flush).
func (m *MemTable) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make(map[uint64]Entry)
	m.keys = nil
	m.dirty = false
	m.size = 0
}

// WAL is a simple write-ahead log of memtable mutations.
type WAL struct {
	mu sync.Mutex
	f  *os.File
	w  *bufio.Writer
}

// NewWAL creates/truncates a WAL file at path.
func NewWAL(path string) (*WAL, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &WAL{f: f, w: bufio.NewWriterSize(f, 1<<16)}, nil
}

// Record types.
const (
	recPut    byte = 1
	recDelete byte = 2
)

// AppendPut records a put. Encoding:
// type u8 | key u64 | seq u64 | hasPtr u8 | [ptr 16 | valLen u32] | val bytes
func (w *WAL) AppendPut(key uint64, ptr sstable.ValuePointer, value []byte, seq uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	buf := make([]byte, 0, 1+8+8+1+16+4+len(value))
	buf = append(buf, recPut)
	buf = binary.BigEndian.AppendUint64(buf, key)
	buf = binary.BigEndian.AppendUint64(buf, seq)
	if value == nil && ptr.Size > 0 || ptr.LogID != 0 || ptr.Offset != 0 {
		buf = append(buf, 1)
		var p [16]byte
		ptr.Encode(p[:])
		buf = append(buf, p[:]...)
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(value)))
	} else {
		buf = append(buf, 0)
	}
	buf = append(buf, value...)
	if _, err := w.w.Write(buf); err != nil {
		return err
	}
	return nil
}

// AppendDelete records a delete: type u8 | key u64 | seq u64.
func (w *WAL) AppendDelete(key uint64, seq uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	var buf [17]byte
	buf[0] = recDelete
	binary.BigEndian.PutUint64(buf[1:9], key)
	binary.BigEndian.PutUint64(buf[9:17], seq)
	_, err := w.w.Write(buf[:])
	return err
}

// Sync flushes buffered writes to the file.
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.w.Flush(); err != nil {
		return err
	}
	return w.f.Sync()
}

// Close flushes and closes the file.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.w.Flush(); err != nil {
		return err
	}
	return w.f.Close()
}

// Replay reads the WAL and invokes fn for each record.
func Replay(path string, fn func(e Entry) error) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		var typ [1]byte
		if _, err := io.ReadFull(r, typ[:]); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		switch typ[0] {
		case recPut:
			keyBuf := make([]byte, 17)
			if _, err := io.ReadFull(r, keyBuf); err != nil {
				return err
			}
			key := binary.BigEndian.Uint64(keyBuf[0:8])
			seq := binary.BigEndian.Uint64(keyBuf[8:16])
			hasPtr := keyBuf[16]
			var e Entry
			e.Key = key
			e.Seq = seq
			if hasPtr == 1 {
				pb := make([]byte, 20)
				if _, err := io.ReadFull(r, pb); err != nil {
					return err
				}
				e.Pointer = sstable.ValuePointer{
					LogID:  binary.BigEndian.Uint32(pb[0:4]),
					Offset: binary.BigEndian.Uint64(pb[4:12]),
					Size:   binary.BigEndian.Uint32(pb[12:16]),
				}
				vlen := binary.BigEndian.Uint32(pb[16:20])
				if vlen > 0 {
					vb := make([]byte, vlen)
					if _, err := io.ReadFull(r, vb); err != nil {
						return err
					}
					e.Value = vb
				}
			} else {
				// length-prefixed inline value
				lb := make([]byte, 4)
				if _, err := io.ReadFull(r, lb); err != nil {
					return err
				}
				vlen := binary.BigEndian.Uint32(lb)
				if vlen > 0 {
					vb := make([]byte, vlen)
					if _, err := io.ReadFull(r, vb); err != nil {
						return err
					}
					e.Value = vb
				}
			}
			if err := fn(e); err != nil {
				return err
			}
		case recDelete:
			kb := make([]byte, 16)
			if _, err := io.ReadFull(r, kb); err != nil {
				return err
			}
			e := Entry{
				Key:  binary.BigEndian.Uint64(kb[0:8]),
				Seq:  binary.BigEndian.Uint64(kb[8:16]),
				Tomb: true,
			}
			if err := fn(e); err != nil {
				return err
			}
		default:
			return fmt.Errorf("wal: unknown record type %d", typ[0])
		}
	}
}
