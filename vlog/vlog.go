// Package vlog implements the value log used for key-value separation
// (WiscKey / Bourbon). SSTables store only keys and pointers; values
// live here. Compaction rewrites keys only and leaves values alone.
package vlog

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"bourbon/sstable"
)

const (
	headerSize = 8 // length uint32 + checksum-ish padding uint32 (simple)
)

// Log is an append-only value log file.
type Log struct {
	mu     sync.Mutex
	id     uint32
	f      *os.File
	path   string
	offset uint64
	closed bool
}

// Open opens or creates a value log at dir with the given id.
func Open(dir string, id uint32) (*Log, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, fmt.Sprintf("vlog-%06d.vlog", id))
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &Log{
		id:     id,
		f:      f,
		path:   path,
		offset: uint64(fi.Size()),
	}, nil
}

// ID returns the log identifier used in value pointers.
func (l *Log) ID() uint32 { return l.id }

// Offset returns the current append offset.
func (l *Log) Offset() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.offset
}

// Append writes value and returns its pointer.
// Record on disk: [len uint32][pad uint32][bytes].
func (l *Log) Append(value []byte) (sstable.ValuePointer, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return sstable.ValuePointer{}, errors.New("vlog: closed")
	}
	var hdr [headerSize]byte
	binary.BigEndian.PutUint32(hdr[0:4], uint32(len(value)))
	// pad/flags left zero; could hold CRC in a fuller implementation.

	start := l.offset
	if _, err := l.f.WriteAt(hdr[:], int64(start)); err != nil {
		return sstable.ValuePointer{}, err
	}
	if _, err := l.f.WriteAt(value, int64(start)+headerSize); err != nil {
		return sstable.ValuePointer{}, err
	}
	total := uint64(headerSize + len(value))
	l.offset += total
	return sstable.ValuePointer{
		LogID:  l.id,
		Offset: start,
		Size:   uint32(len(value)),
	}, nil
}

// Read fetches the value for ptr.
func (l *Log) Read(ptr sstable.ValuePointer) ([]byte, error) {
	if ptr.LogID != l.id {
		return nil, fmt.Errorf("vlog: wrong log id %d (want %d)", ptr.LogID, l.id)
	}
	l.mu.Lock()
	f := l.f
	l.mu.Unlock()
	if f == nil {
		return nil, errors.New("vlog: closed")
	}
	hdr := make([]byte, headerSize)
	if _, err := f.ReadAt(hdr, int64(ptr.Offset)); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[0:4])
	if n != ptr.Size {
		// Trust pointer size if header disagrees (truncated edge case).
		n = ptr.Size
	}
	buf := make([]byte, n)
	if _, err := f.ReadAt(buf, int64(ptr.Offset)+headerSize); err != nil && err != io.EOF {
		return nil, err
	}
	return buf, nil
}

// Sync flushes file data to stable storage.
func (l *Log) Sync() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed || l.f == nil {
		return nil
	}
	return l.f.Sync()
}

// Close closes the log file.
func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.f == nil {
		return nil
	}
	err := l.f.Close()
	l.f = nil
	return err
}

// Path returns the file path.
func (l *Log) Path() string { return l.path }

// Manager owns one or more value logs and routes reads by LogID.
type Manager struct {
	mu   sync.RWMutex
	dir  string
	logs map[uint32]*Log
	next uint32
}

// NewManager creates a manager for the directory.
func NewManager(dir string) *Manager {
	return &Manager{
		dir:  dir,
		logs: make(map[uint32]*Log),
		next: 1,
	}
}

// Current returns (creating if needed) the active log for appends.
func (m *Manager) Current() (*Log, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if l, ok := m.logs[m.next]; ok {
		return l, nil
	}
	l, err := Open(m.dir, m.next)
	if err != nil {
		return nil, err
	}
	m.logs[m.next] = l
	return l, nil
}

// Append writes value to the current log.
func (m *Manager) Append(value []byte) (sstable.ValuePointer, error) {
	l, err := m.Current()
	if err != nil {
		return sstable.ValuePointer{}, err
	}
	return l.Append(value)
}

// Read fetches value by pointer.
func (m *Manager) Read(ptr sstable.ValuePointer) ([]byte, error) {
	m.mu.RLock()
	l, ok := m.logs[ptr.LogID]
	m.mu.RUnlock()
	if !ok {
		// Try open from disk.
		nl, err := Open(m.dir, ptr.LogID)
		if err != nil {
			return nil, err
		}
		m.mu.Lock()
		m.logs[ptr.LogID] = nl
		m.mu.Unlock()
		return nl.Read(ptr)
	}
	return l.Read(ptr)
}

// Close closes all logs.
func (m *Manager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	var first error
	for _, l := range m.logs {
		if err := l.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
