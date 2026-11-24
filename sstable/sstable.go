// Package sstable implements the on-disk sorted table format used by
// Bourbon. A table holds fixed-size records (key + value pointer),
// grouped into data blocks, plus an index block and a Bloom filter.
//
// Layout:
//
//	[data block 0][data block 1]...[index block][filter block][footer]
//
// Records are fixed size so the model can convert a position into a
// byte offset without extra indirection (paper section 4.2).
package sstable

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"

	"bourbon/bloom"
	"bourbon/plr"
)

const (
	// Magic marks a valid footer.
	Magic uint32 = 0x424f5552 // "BOUR"
	// FormatVersion is the current table format.
	FormatVersion uint32 = 1

	// DefaultBlockSize is the target size of one data block in bytes.
	DefaultBlockSize = 4096

	// FooterSize is the fixed footer length.
	FooterSize = 40

	// PtrSize is the encoded size of a value pointer.
	PtrSize = 16

	// RecordSize is key (8) + pointer (16).
	RecordSize = 8 + PtrSize
)

// ValuePointer locates a value in the value log.
type ValuePointer struct {
	LogID  uint32
	Offset uint64
	Size   uint32
}

// Encode writes the pointer into a 16-byte buffer.
func (p ValuePointer) Encode(buf []byte) {
	binary.BigEndian.PutUint32(buf[0:4], p.LogID)
	binary.BigEndian.PutUint64(buf[4:12], p.Offset)
	binary.BigEndian.PutUint32(buf[12:16], p.Size)
}

// DecodeValuePointer reads a 16-byte buffer.
func DecodeValuePointer(buf []byte) ValuePointer {
	return ValuePointer{
		LogID:  binary.BigEndian.Uint32(buf[0:4]),
		Offset: binary.BigEndian.Uint64(buf[4:12]),
		Size:   binary.BigEndian.Uint32(buf[12:16]),
	}
}

// Options configures table writing.
type Options struct {
	BlockSize int
}

func (o *Options) normalize() {
	if o.BlockSize <= 0 {
		o.BlockSize = DefaultBlockSize
	}
}

// Meta describes a finished table.
type Meta struct {
	Path      string
	NumRecs   int64
	NumBlocks uint64
	MinKey    uint64
	MaxKey    uint64
	Bytes     uint64
	// Keys is kept for training the model after write.
	Keys []uint64
}

type indexEntry struct {
	MaxKey   uint64
	FirstKey uint64
	Offset   uint64
	Length   uint32
	NumRec   uint32
}

// Writer builds an SSTable.
type Writer struct {
	opts   Options
	f      *os.File
	path   string
	offset uint64

	keys       []uint64
	nrec       int64
	block      []byte
	blockKeys  []uint64
	blockStart uint64

	index   []indexEntry
	closed  bool
	syncDur bool
}

// NewWriter creates a writer for path.
func NewWriter(path string, opts Options) (*Writer, error) {
	opts.normalize()
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Writer{
		opts: opts,
		f:    f,
		path: path,
	}, nil
}

// Add appends one record. Keys must ascend.
func (w *Writer) Add(key uint64, ptr ValuePointer) error {
	if w.closed {
		return errors.New("sstable: writer closed")
	}
	if n := len(w.keys); n > 0 && key <= w.keys[n-1] {
		return fmt.Errorf("sstable: keys must ascend: %d after %d", key, w.keys[n-1])
	}

	if len(w.block) == 0 {
		w.blockStart = w.offset
		w.blockKeys = w.blockKeys[:0]
	}

	rec := make([]byte, RecordSize)
	binary.BigEndian.PutUint64(rec[0:8], key)
	ptr.Encode(rec[8:])
	w.block = append(w.block, rec...)
	w.blockKeys = append(w.blockKeys, key)
	w.keys = append(w.keys, key)
	w.nrec++

	if len(w.block) >= w.opts.BlockSize {
		return w.flushBlock()
	}
	return nil
}

func (w *Writer) flushBlock() error {
	if len(w.block) == 0 {
		return nil
	}
	n, err := w.f.Write(w.block)
	if err != nil {
		return err
	}
	w.index = append(w.index, indexEntry{
		MaxKey:   w.blockKeys[len(w.blockKeys)-1],
		FirstKey: w.blockKeys[0],
		Offset:   w.blockStart,
		Length:   uint32(n),
		NumRec:   uint32(len(w.blockKeys)),
	})
	w.offset += uint64(n)
	w.block = w.block[:0]
	w.blockKeys = w.blockKeys[:0]
	return nil
}

// Finish writes index, filter, and footer, then closes the file.
func (w *Writer) Finish() (Meta, error) {
	var meta Meta
	if w.closed {
		return meta, errors.New("sstable: already closed")
	}
	w.closed = true

	if len(w.keys) == 0 {
		// Still write a valid empty table.
		if err := w.writeEmpty(); err != nil {
			return meta, err
		}
		return Meta{Path: w.path, Keys: nil}, nil
	}

	if err := w.flushBlock(); err != nil {
		return meta, err
	}

	// Filter over all keys.
	filter := bloom.New(len(w.keys) + 1)
	var kb [8]byte
	for _, k := range w.keys {
		binary.BigEndian.PutUint64(kb[:], k)
		filter.Add(kb[:])
	}

	// Index block: count + 32-byte entries.
	idxOff := w.offset
	idx := make([]byte, 4+len(w.index)*32)
	binary.BigEndian.PutUint32(idx[0:4], uint32(len(w.index)))
	for i, e := range w.index {
		eb := idx[4+i*32:]
		binary.BigEndian.PutUint64(eb[0:8], e.MaxKey)
		binary.BigEndian.PutUint64(eb[8:16], e.FirstKey)
		binary.BigEndian.PutUint64(eb[16:24], e.Offset)
		binary.BigEndian.PutUint32(eb[24:28], e.Length)
		binary.BigEndian.PutUint32(eb[28:32], e.NumRec)
	}
	if _, err := w.f.Write(idx); err != nil {
		return meta, err
	}
	idxLen := uint64(len(idx))
	w.offset += idxLen

	// Filter block.
	fltOff := w.offset
	flt := filter.Encode()
	if _, err := w.f.Write(flt); err != nil {
		return meta, err
	}
	fltLen := uint64(len(flt))
	w.offset += fltLen

	// Footer.
	var foot [FooterSize]byte
	binary.BigEndian.PutUint32(foot[0:4], Magic)
	binary.BigEndian.PutUint32(foot[4:8], FormatVersion)
	binary.BigEndian.PutUint64(foot[8:16], idxOff)
	binary.BigEndian.PutUint64(foot[16:24], idxLen)
	binary.BigEndian.PutUint64(foot[24:32], fltOff)
	binary.BigEndian.PutUint64(foot[32:40], fltLen)
	if _, err := w.f.Write(foot[:]); err != nil {
		return meta, err
	}
	w.offset += FooterSize

	if err := w.f.Sync(); err != nil {
		return meta, err
	}
	if err := w.f.Close(); err != nil {
		return meta, err
	}

	return Meta{
		Path:      w.path,
		NumRecs:   w.nrec,
		NumBlocks: uint64(len(w.index)),
		MinKey:    w.keys[0],
		MaxKey:    w.keys[len(w.keys)-1],
		Bytes:     w.offset,
		Keys:      w.keys,
	}, nil
}

func (w *Writer) writeEmpty() error {
	// Empty index (count=0), minimal filter, footer.
	idx := make([]byte, 4)
	filter := bloom.New(1)
	flt := filter.Encode()

	idxOff := w.offset
	if _, err := w.f.Write(idx); err != nil {
		return err
	}
	w.offset += uint64(len(idx))

	fltOff := w.offset
	if _, err := w.f.Write(flt); err != nil {
		return err
	}
	w.offset += uint64(len(flt))

	var foot [FooterSize]byte
	binary.BigEndian.PutUint32(foot[0:4], Magic)
	binary.BigEndian.PutUint32(foot[4:8], FormatVersion)
	binary.BigEndian.PutUint64(foot[8:16], idxOff)
	binary.BigEndian.PutUint64(foot[16:24], 4)
	binary.BigEndian.PutUint64(foot[24:32], fltOff)
	binary.BigEndian.PutUint64(foot[32:40], uint64(len(flt)))
	if _, err := w.f.Write(foot[:]); err != nil {
		return err
	}
	return w.f.Close()
}

// Reader provides lookups against an SSTable.
type Reader struct {
	f       *os.File
	path    string
	index   []indexEntry
	filter  *bloom.Filter
	numRecs int64
	minKey  uint64
	maxKey  uint64
	mu      sync.RWMutex
	model   *plr.Model
	learned bool
	// posPrefix caches cumulative record counts (len = numBlocks+1).
	posPrefix []int64
}

// Open opens a table for reading.
func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*Reader, error) {
		f.Close()
		return nil, e
	}

	fi, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	size := fi.Size()
	if size < FooterSize {
		return fail(errors.New("sstable: file too small"))
	}
	foot := make([]byte, FooterSize)
	if _, err = f.ReadAt(foot, size-FooterSize); err != nil {
		return fail(err)
	}
	if binary.BigEndian.Uint32(foot[0:4]) != Magic {
		return fail(errors.New("sstable: bad magic"))
	}
	idxOff := binary.BigEndian.Uint64(foot[8:16])
	idxLen := binary.BigEndian.Uint64(foot[16:24])
	fltOff := binary.BigEndian.Uint64(foot[24:32])
	fltLen := binary.BigEndian.Uint64(foot[32:40])

	r := &Reader{f: f, path: path}

	// Index.
	idx := make([]byte, idxLen)
	if _, err = f.ReadAt(idx, int64(idxOff)); err != nil {
		return fail(err)
	}
	if len(idx) < 4 {
		return fail(errors.New("sstable: short index"))
	}
	n := binary.BigEndian.Uint32(idx[0:4])
	if uint64(len(idx)) < 4+uint64(n)*32 {
		return fail(errors.New("sstable: index truncated"))
	}
	r.index = make([]indexEntry, n)
	for i := uint32(0); i < n; i++ {
		eb := idx[4+i*32:]
		r.index[i] = indexEntry{
			MaxKey:   binary.BigEndian.Uint64(eb[0:8]),
			FirstKey: binary.BigEndian.Uint64(eb[8:16]),
			Offset:   binary.BigEndian.Uint64(eb[16:24]),
			Length:   binary.BigEndian.Uint32(eb[24:28]),
			NumRec:   binary.BigEndian.Uint32(eb[28:32]),
		}
		r.numRecs += int64(r.index[i].NumRec)
	}
	if n > 0 {
		r.minKey = r.index[0].FirstKey
		r.maxKey = r.index[n-1].MaxKey
	}

	// Filter.
	flt := make([]byte, fltLen)
	if _, err = f.ReadAt(flt, int64(fltOff)); err != nil {
		return fail(err)
	}
	r.filter = bloom.Decode(flt)
	if r.filter == nil {
		return fail(errors.New("sstable: bad filter"))
	}
	return r, nil
}

// Close releases the file handle.
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.f == nil {
		return nil
	}
	err := r.f.Close()
	r.f = nil
	return err
}

// Path returns the file path.
func (r *Reader) Path() string { return r.path }

// NumRecords returns the number of records.
func (r *Reader) NumRecords() int64 { return r.numRecs }

// Bounds returns the key range of the table.
func (r *Reader) Bounds() (min, max uint64) { return r.minKey, r.maxKey }

// MayContain checks the Bloom filter.
func (r *Reader) MayContain(key uint64) bool {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], key)
	return r.filter.MayContain(buf[:])
}

// findBlock returns the index of the first block whose MaxKey >= key.
func (r *Reader) findBlock(key uint64) int {
	lo, hi := 0, len(r.index)-1
	ans := -1
	for lo <= hi {
		mid := (lo + hi) / 2
		if r.index[mid].MaxKey >= key {
			ans = mid
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return ans
}

// readBlock loads block i into buf (caller-provided or internal).
func (r *Reader) readBlock(i int, buf []byte) ([]byte, error) {
	e := r.index[i]
	if cap(buf) < int(e.Length) {
		buf = make([]byte, e.Length)
	} else {
		buf = buf[:e.Length]
	}
	if _, err := r.f.ReadAt(buf, int64(e.Offset)); err != nil {
		return nil, err
	}
	return buf, nil
}

// GetResult is the outcome of a table lookup.
type GetResult struct {
	Found   bool
	Pointer ValuePointer
	// BlockIndex is the data block that holds the key (-1 if not found).
	BlockIndex int
	// Position is the record index within the table (for model path).
	Position int64
	// UsedModel reports whether the learned model guided the search.
	UsedModel bool
	// Chunk is the loaded byte range for the model path (optional).
	Chunk []byte
	// ChunkStart is the record index of the first record in Chunk.
	ChunkStart int64
}

// Get looks up key in the table using the baseline path:
// SearchIB (binary search index) -> SearchFB -> LoadDB -> SearchDB.
func (r *Reader) Get(key uint64) (GetResult, error) {
	var res GetResult
	res.BlockIndex = -1

	if len(r.index) == 0 || key < r.minKey || key > r.maxKey {
		return res, nil
	}
	if !r.MayContain(key) {
		return res, nil
	}

	bi := r.findBlock(key)
	if bi < 0 {
		return res, nil
	}

	block, err := r.readBlock(bi, nil)
	if err != nil {
		return res, err
	}
	return r.searchInBlock(bi, block, key, false), nil
}

// searchInBlock binary-searches a loaded block for key.
// chunkStart is the global record index of block[0]; if the loaded
// buffer is a sub-chunk, pass the global index of its first record.
func (r *Reader) searchInBlock(bi int, block []byte, key uint64, usedModel bool) GetResult {
	res := GetResult{BlockIndex: bi, UsedModel: usedModel}
	n := len(block) / RecordSize
	startPos := r.blockStartPos(bi)

	lo, hi := 0, n-1
	for lo <= hi {
		mid := (lo + hi) / 2
		off := mid * RecordSize
		k := binary.BigEndian.Uint64(block[off : off+8])
		if k == key {
			res.Found = true
			res.Pointer = DecodeValuePointer(block[off+8 : off+RecordSize])
			res.Position = startPos + int64(mid)
			return res
		}
		if k < key {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return res
}

// blockStartPos returns the global record index of the first record
// in block bi (binary search over cached prefix sums).
func (r *Reader) blockStartPos(bi int) int64 {
	if bi <= 0 {
		return 0
	}
	prefix := r.ensurePrefix()
	if bi >= len(prefix) {
		return 0
	}
	return prefix[bi]
}

func (r *Reader) ensurePrefix() []int64 {
	r.mu.RLock()
	p := r.posPrefix
	r.mu.RUnlock()
	if p != nil {
		return p
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.posPrefix == nil {
		p := make([]int64, len(r.index)+1)
		var sum int64
		for i, e := range r.index {
			p[i] = sum
			sum += int64(e.NumRec)
		}
		p[len(r.index)] = sum
		r.posPrefix = p
	}
	return r.posPrefix
}

// GetWithModel looks up key using the learned model when available
// (paper Figure 6b): ModelLookup -> SearchFB -> LoadChunk -> LocateKey.
func (r *Reader) GetWithModel(key uint64) (GetResult, error) {
	r.mu.RLock()
	model := r.model
	learned := r.learned
	r.mu.RUnlock()

	if !learned || model == nil {
		return r.Get(key)
	}

	if len(r.index) == 0 || key < r.minKey || key > r.maxKey {
		return GetResult{BlockIndex: -1}, nil
	}
	if !r.MayContain(key) {
		return GetResult{BlockIndex: -1}, nil
	}

	// ModelLookup: predicted position and window (paper: pos ± delta).
	pred, low, high := model.Predict(key)
	if high < 0 || low >= r.numRecs {
		// Fall back to index search.
		return r.Get(key)
	}
	if high >= r.numRecs {
		high = r.numRecs - 1
	}
	if low < 0 {
		low = 0
	}
	if pred < low {
		pred = low
	}
	if pred > high {
		pred = high
	}

	// Calculate the data block that contains the predicted record.
	// If [low, high] spans blocks, prefer the block of `pred` (paper note).
	bi := r.blockForPos(pred)
	if bi < 0 {
		return r.Get(key)
	}

	// LoadChunk: read only the byte range covering records [low, high]
	// within this block (fixed-size records make this exact).
	e := r.index[bi]
	blockStartPos := r.blockStartPos(bi)

	// Local record indices within the block.
	localLow := low - blockStartPos
	localHigh := high - blockStartPos
	if localLow < 0 {
		localLow = 0
	}
	if localHigh > int64(e.NumRec)-1 {
		localHigh = int64(e.NumRec) - 1
	}
	if localLow > localHigh {
		localLow, localHigh = localHigh, localLow
	}

	// Pad the window by a few records so a slightly-off prediction
	// still lands inside the loaded chunk.
	const pad = 4
	if localLow >= pad {
		localLow -= pad
	} else {
		localLow = 0
	}
	if localHigh+pad < int64(e.NumRec)-1 {
		localHigh += pad
	} else {
		localHigh = int64(e.NumRec) - 1
	}

	chunkOff := e.Offset + uint64(localLow)*RecordSize
	chunkLen := uint64(localHigh-localLow+1) * RecordSize
	if chunkLen == 0 {
		return r.Get(key)
	}
	bp := chunkPool.Get().(*[]byte)
	var chunk []byte
	if uint64(cap(*bp)) >= chunkLen {
		chunk = (*bp)[:chunkLen]
	} else {
		chunk = make([]byte, chunkLen)
		*bp = chunk
	}
	if _, err := r.f.ReadAt(chunk, int64(chunkOff)); err != nil {
		chunkPool.Put(bp)
		return res0(err)
	}

	// LocateKey: binary search inside the loaded chunk.
	res := r.searchChunk(bi, blockStartPos+localLow, chunk, key, true)
	chunkPool.Put(bp)
	if res.Found {
		return res, nil
	}

	// Key might sit outside the padded window (model error > Delta).
	// Try a wider load of the whole block before giving up.
	block, err := r.readBlock(bi, nil)
	if err != nil {
		return res0(err)
	}
	res = r.searchInBlock(bi, block, key, true)
	if res.Found {
		return res, nil
	}

	// Adjacent blocks if prediction straddled a boundary.
	for _, cand := range []int{bi - 1, bi + 1} {
		if cand < 0 || cand >= len(r.index) {
			continue
		}
		if key < r.index[cand].FirstKey || key > r.index[cand].MaxKey {
			continue
		}
		block, err := r.readBlock(cand, nil)
		if err != nil {
			return res0(err)
		}
		res = r.searchInBlock(cand, block, key, true)
		if res.Found {
			return res, nil
		}
	}

	// Model error exceeded the window: correctness fallback to the
	// baseline index path (paper assumes delta bounds the error; our
	// greedy trainer can drift slightly at segment seams).
	return r.Get(key)
}

// searchChunk binary-searches a loaded chunk. startPos is the global
// record index of chunk[0].
func (r *Reader) searchChunk(bi int, startPos int64, chunk []byte, key uint64, usedModel bool) GetResult {
	res := GetResult{BlockIndex: bi, UsedModel: usedModel}
	n := len(chunk) / RecordSize
	if n == 0 {
		return res
	}
	// Fast path: keys are often exactly at the predicted midpoint.
	mid := n / 2
	off := mid * RecordSize
	if k := binary.BigEndian.Uint64(chunk[off : off+8]); k == key {
		res.Found = true
		res.Pointer = DecodeValuePointer(chunk[off+8 : off+RecordSize])
		res.Position = startPos + int64(mid)
		return res
	}
	lo, hi := 0, n-1
	for lo <= hi {
		m := (lo + hi) / 2
		o := m * RecordSize
		k := binary.BigEndian.Uint64(chunk[o : o+8])
		if k == key {
			res.Found = true
			res.Pointer = DecodeValuePointer(chunk[o+8 : o+RecordSize])
			res.Position = startPos + int64(m)
			return res
		}
		if k < key {
			lo = m + 1
		} else {
			hi = m - 1
		}
	}
	return res
}

// chunkPool reduces allocations on the model lookup path.
var chunkPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 4096)
		return &b
	},
}

func res0(err error) (GetResult, error) {
	return GetResult{BlockIndex: -1}, err
}

// blockForPos returns the block index that contains record position pos.
// Uses a cached prefix-sum of per-block record counts + binary search.
func (r *Reader) blockForPos(pos int64) int {
	if len(r.index) == 0 {
		return -1
	}
	if pos < 0 {
		return 0
	}
	prefix := r.ensurePrefix()
	i := sort.Search(len(r.index), func(i int) bool {
		return prefix[i+1] > pos
	})
	if i >= len(r.index) {
		return len(r.index) - 1
	}
	return i
}

// SetModel attaches a learned model to the reader.
func (r *Reader) SetModel(m *plr.Model) {
	r.mu.Lock()
	r.model = m
	r.learned = m != nil
	r.mu.Unlock()
}

// Model returns the attached model (nil if not learned).
func (r *Reader) Model() *plr.Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.model
}

// IsLearned reports whether a model is attached.
func (r *Reader) IsLearned() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.learned
}

// Keys re-reads all keys from disk (used for training).
func (r *Reader) Keys() ([]uint64, error) {
	keys := make([]uint64, 0, r.numRecs)
	var buf []byte
	for i := range r.index {
		block, err := r.readBlock(i, buf)
		if err != nil {
			return nil, err
		}
		buf = block
		n := len(block) / RecordSize
		for j := 0; j < n; j++ {
			off := j * RecordSize
			keys = append(keys, binary.BigEndian.Uint64(block[off:off+8]))
		}
	}
	return keys, nil
}

// RecordAt returns the key and pointer at record position pos.
// Used by compaction to iterate tables in order.
func (r *Reader) RecordAt(pos int64) (uint64, ValuePointer, bool, error) {
	if pos < 0 || pos >= r.numRecs {
		return 0, ValuePointer{}, false, nil
	}
	bi := r.blockForPos(pos)
	if bi < 0 {
		return 0, ValuePointer{}, false, nil
	}
	var start int64
	for i := 0; i < bi; i++ {
		start += int64(r.index[i].NumRec)
	}
	local := int(pos - start)
	block, err := r.readBlock(bi, nil)
	if err != nil {
		return 0, ValuePointer{}, false, err
	}
	off := local * RecordSize
	if off+RecordSize > len(block) {
		return 0, ValuePointer{}, false, errors.New("sstable: record out of range")
	}
	k := binary.BigEndian.Uint64(block[off : off+8])
	return k, DecodeValuePointer(block[off+8 : off+RecordSize]), true, nil
}

// Iterator walks records of a table in ascending key order.
type Iterator struct {
	r        *Reader
	blockIdx int
	buf      []byte
	recIdx   int
	nrec     int
	valid    bool
	key      uint64
	ptr      ValuePointer
	err      error
}

// NewIterator returns an iterator positioned before the first record.
func (r *Reader) NewIterator() *Iterator {
	it := &Iterator{r: r, blockIdx: -1, valid: false}
	it.advanceBlock()
	return it
}

func (it *Iterator) advanceBlock() {
	it.blockIdx++
	it.recIdx = 0
	it.nrec = 0
	it.valid = false
	it.err = nil
	if it.blockIdx >= len(it.r.index) {
		return
	}
	buf, err := it.r.readBlock(it.blockIdx, it.buf)
	if err != nil {
		it.err = err
		return
	}
	it.buf = buf
	it.nrec = len(buf) / RecordSize
	it.step()
}

func (it *Iterator) step() {
	if it.err != nil {
		it.valid = false
		return
	}
	if it.recIdx >= it.nrec {
		it.advanceBlock()
		return
	}
	off := it.recIdx * RecordSize
	it.key = binary.BigEndian.Uint64(it.buf[off : off+8])
	it.ptr = DecodeValuePointer(it.buf[off+8 : off+RecordSize])
	it.valid = true
}

// Next advances to the next record. Returns false when done or on error.
func (it *Iterator) Next() bool {
	if !it.valid {
		return false
	}
	it.recIdx++
	it.step()
	return it.valid
}

// Valid reports whether the iterator points at a record.
func (it *Iterator) Valid() bool { return it.valid }

// Key returns the current key.
func (it *Iterator) Key() uint64 { return it.key }

// Pointer returns the current value pointer.
func (it *Iterator) Pointer() ValuePointer { return it.ptr }

// Err returns the first read error.
func (it *Iterator) Err() error { return it.err }
