# Bourbon LSM — Code Report and Navigation Guide

This document explains what this codebase does. It also tells you how to
read the code. The text follows ASD-STE100 Simplified Technical English.
Some word choices are soft. The structure rules are strict.

---

## 1. Purpose

This project is a Go implementation of the Bourbon paper
(arXiv:2005.14213). The paper adds learned indexes to a Log-Structured
Merge Tree (LSM tree). The system keeps the WiscKey design for values.
It adds Greedy-PLR models for key positions. It also adds a cost-benefit
analyzer that decides when to train a model.

The system does three jobs:

1. Store key-value pairs at high write speed.
2. Read values with two paths: a baseline path and a model path.
3. Learn a model per SSTable file when the model pays for its cost.

---

## 2. Words you will see

| Word | Meaning in this codebase |
|------|--------------------------|
| LSM tree | A tree of sorted string tables (SSTables) in levels. |
| SSTable | A sorted, immutable file on disk. |
| Memtable | An in-memory table that accepts new writes. |
| WAL | Write-ahead log. It records writes before they hit disk. |
| Value log | A separate log that stores large values. |
| Value pointer | An address into the value log: log ID, offset, size. |
| PLR | Piecewise linear regression. It maps a key to a position. |
| Model | A trained PLR. It predicts where a key sits in a file. |
| CBA | Cost-benefit analyzer. It decides if training is worth it. |
| Delta (δ) | The error bound of a model. Default is 8. |
| Level | A tier in the LSM tree: L0, L1, L2, and so on. |
| Compaction | The act of merge files into a lower level. |
| Tombstone | A delete marker. It hides an older value. |

---

## 3. Parts of the system

The code lives in nine packages. Each package has one job.

| Package | File | Job |
|---------|------|-----|
| `plr` | `plr/plr.go` | Train and query Greedy-PLR models. |
| `bloom` | `bloom/bloom.go` | Bloom filter for negative lookups. |
| `sstable` | `sstable/sstable.go` | Write and read SSTable files. |
| `vlog` | `vlog/vlog.go` | Append and read the value log. |
| `memtable` | `memtable/memtable.go` | Sorted map plus WAL. |
| `lsm` | `lsm/lsm.go` | The database. It ties all parts together. |
| `learn` | `learn/learn.go` | Modes and the cost-benefit analyzer. |
| `bench` | `bench/*.go` | Datasets, benchmarks, comparisons. |
| `cmd` | `cmd/bourbon/main.go` | Command-line entry point. |

Read order for a new reader:

1. `plr/plr.go` — small and pure. No I/O.
2. `bloom/bloom.go` — small and pure. No I/O.
3. `memtable/memtable.go` — in-memory writes plus WAL.
4. `vlog/vlog.go` — value storage.
5. `sstable/sstable.go` — file format and both read paths.
6. `lsm/lsm.go` — the glue. Start with `Get`, then `Put`.
7. `learn/learn.go` — when and how to train.
8. `bench/` and `cmd/` — how to run and measure.

---

## 4. Write path

Follow a `Put` from start to end.

1. The caller runs `db.Put(key, value)`.
2. `lsm.DB.write` appends the pair to the WAL.
3. `lsm.DB.write` appends the pair to the memtable.
4. The value goes to the value log. The memtable stores a pointer.
5. When the memtable is full, a worker flushes it.
6. The flush writes a new SSTable in L0.
7. The flush links the file into the level list.
8. The old memtable stays readable until the link is complete.
9. Compaction later merges L0 files into L1, then L1 into L2.

Delete works the same way. It writes a tombstone pointer. The tombstone
hides older values during reads and compaction.

Key functions to read:

- `lsm.DB.Put` — the public write entry.
- `lsm.DB.write` — WAL, memtable, value log.
- `lsm.DB.flushImm` — memtable to SSTable.
- `lsm.DB.compact` and `lsm.DB.mergeFiles` — level merges.

---

## 5. Read path — baseline

The baseline path does not use a model. It works like WiscKey.

1. The caller runs `db.Get(key)`.
2. The memtable checks the key. Hit means done.
3. The immutable memtable checks the key. Hit means done.
4. `db.findFiles` collects candidate SSTables. Newest level first.
5. For each candidate, `db.lookupFile` runs an internal lookup.
6. `Reader.Get` checks the bloom filter. A miss skips the file.
7. `Reader.findBlock` binary-searches the index block.
8. The reader loads the data block.
9. `searchInBlock` binary-searches inside the block.
10. A hit returns a value pointer. The reader fetches the value from
    the value log.

Key functions to read:

- `lsm.DB.Get` — the public read entry.
- `lsm.DB.findFiles` — which files can hold the key.
- `lsm.DB.lookupFile` — one internal lookup.
- `sstable.Reader.Get` — bloom, index, data block.
- `sstable.Reader.findBlock` and `searchInBlock` — the two binary
  searches.

---

## 6. Read path — model

The model path replaces step 7 and step 8 with a learned prediction.
The paper calls this the Bourbon lookup.

1. Steps 1 to 6 are the same as the baseline path.
2. `Reader.GetWithModel` runs when the file has a model.
3. The model predicts a position `p` and a window `[p-δ, p+δ]`.
4. `blockForPos` maps the position to a block index. A prefix-sum
   binary search does this job.
5. `LoadChunk` reads only the byte range for the window. It does not
   load the whole block.
6. `searchChunk` probes the middle of the chunk first. Then it runs a
   binary search.
7. On a miss, the reader loads the full block. Then it checks the
   adjacent blocks.
8. On a still-miss, the reader falls back to `Reader.Get`. This keeps
   the result correct when model error exceeds δ.

Key functions to read:

- `sstable.Reader.GetWithModel` — the full model path.
- `sstable.Reader.blockForPos` — position to block.
- `sstable.Reader.searchChunk` — search inside the loaded chunk.
- `sstable.Reader.Get` — the correctness fallback.
- `lsm.DB.lookupFile` — chooses model path when `IsLearned()` is true.

---

## 7. The model itself

`plr.Train(keys, delta)` builds a model from sorted keys.

The trainer keeps one open segment. Each new point must fit the current
line within δ. The trainer does not use a fixed line through two
endpoints. That method lets error drift. Instead, the trainer keeps a
feasible slope interval. Each point intersects the interval. An empty
interval closes the segment and starts a new one.

The result: every training point has error at most δ (plus rounding).

Key functions to read:

- `plr.Train` — the training entry.
- `plr.Trainer.Add` — absorb or close.
- `plr.Trainer.slopeBounds` — the feasible slope for one point.
- `plr.Model.Predict` — position and window for a key.
- `plr.Model.SegmentIndex` — which segment covers a key.

---

## 8. Learning modes

The `learn.Manager` decides when to train. Three modes exist.

| Mode | Behavior |
|------|----------|
| `never` | No training. All reads use the baseline path. |
| `always` | Train every registered file after `T_wait`. |
| `cba` | Train only when benefit exceeds cost. |

The cost-benefit analyzer (CBA) estimates two numbers for each file:

- Cost: train time for that file. The manager calibrates time per point.
- Benefit: how many baseline lookups a model would skip or speed up.

If benefit is greater than cost, the manager trains. If not, it skips.
A bootstrap rule forces the first few files to train so the manager has
data to compare.

Key functions to read:

- `learn.Manager.RegisterFile` — enroll a file.
- `learn.Manager.Start` / `Stop` — the background loop.
- `learn.EstimateBenefit` — the benefit estimate.
- `learn.Manager.TrainCost` — the cost estimate.
- `lsm.DB.registerLearner` — how the DB wires a file to the manager.

---

## 9. File format

An SSTable file has five parts in order.

| Part | Contents |
|------|----------|
| Data block(s) | Sorted records: 8-byte key + 16-byte value pointer. |
| Index block | One entry per data block: first key, max key, offset, size, record count. |
| Filter block | Bloom filter over all keys. |
| Footer | Magic number `0x424f5552`, offsets for index and filter. |

Record size is fixed at 24 bytes. Fixed size makes chunk loads exact.
The model can compute a byte range from a record position.

Key functions to read:

- `sstable.Writer.Add` and `Finish` — write path.
- `sstable.Reader.Open` — read the footer and build the index.
- `sstable.Reader.readBlock` — load one block by offset.

---

## 10. How to navigate the codebase

### 10.1 First session (about 30 minutes)

1. Open `plr/plr.go`. Read `Train`, `Add`, `Predict`. Run
   `go test ./plr/ -v`.
2. Open `bloom/bloom.go`. Read `New`, `Add`, `MayContain`. Run
   `go test ./bloom/ -v`.
3. Open `lsm/lsm.go`. Read `Options`, `Open`, `Put`, `Get`. Do not read
   compaction yet.

### 10.2 Second session (about 45 minutes)

1. Open `memtable/memtable.go`. Read `Put`, `Get`, `WAL.AppendPut`.
2. Open `vlog/vlog.go`. Read `Append`, `Read`, `ValuePointer`.
3. Open `sstable/sstable.go`. Read `Writer.Add`, `Writer.Finish`,
   `Reader.Get`.
4. Trace one `Put` by hand. Write the call chain on paper.

### 10.3 Third session (about 60 minutes)

1. Read `lsm.DB.flushImm`. Follow one flush from memtable to file.
2. Read `lsm.DB.compact` and `mergeFiles`. Follow one merge.
3. Read `sstable.Reader.GetWithModel` in full. Compare it with
   `Reader.Get`.
4. Read `learn.Manager` loop. Connect training to `SetModel`.

### 10.4 Grep recipes

| You want to find… | Run this |
|-------------------|----------|
| The public DB API | `grep -n 'func (db \*DB)' lsm/lsm.go` |
| Both read paths | `grep -n 'func (r \*Reader) Get' sstable/sstable.go` |
| Model attach point | `grep -n 'SetModel' -r .` |
| Learning modes | `grep -n 'Mode' learn/learn.go` |
| Test for a feature | `grep -n 'func Test' lsm/lsm_test.go` |
| Benchmark entry | `grep -n 'func Benchmark\|func Compare' bench/` |

### 10.5 Run and measure

```bash
# All tests
go test ./... -count=1

# One package, verbose
go test ./lsm/ -v -count=1

# Micro benchmarks
go test ./plr/ -bench=. -benchmem
go test ./sstable/ -bench='BenchmarkGet' -benchtime=200000x

# CLI comparison (baseline vs model)
go run ./cmd/bourbon -dataset linear -keys 200000 -ops 100000 -mode always
go run ./cmd/bourbon -dataset normal -keys 200000 -ops 100000 -mode always
go run ./cmd/bourbon -dataset seg10  -keys 200000 -ops 100000 -mode cba
```

Datasets for the CLI: `linear`, `seg1`, `seg10`, `normal`, `random`.
Modes: `always`, `cba`, `never`.

---

## 11. Measured results

Machine: Apple M1 Pro. Go 1.27.4. 200,000 keys. 100,000 lookups.
Mode: `always`. Values are 64 bytes.

| Dataset | Baseline avg | Model avg | Speedup |
|---------|--------------|-----------|---------|
| linear | 2.76 µs | 1.97 µs | 1.40× |
| normal | 2.68 µs | 2.09 µs | 1.28× |
| seg10 | 2.80 µs | 2.16 µs | 1.30× |
| random | 2.77 µs | 2.03 µs | 1.36× |

The paper reports 1.23× to 1.78×. This implementation lands inside that
range.

Micro benchmark on one SSTable (200,000 keys):

| Path | Time per lookup |
|------|-----------------|
| Baseline `Get` | ~1355 ns |
| Model `GetWithModel` | ~622 ns |

---

## 12. Limits

This is a core system, not a production database.

- Single process. No network, no replication, no transactions.
- Keys are `uint64`. Values are byte slices.
- One writer at a time. Reads are safe during writes.
- Compaction is simple. It merges whole levels. It has no size tiers.
- The value log has no GC yet. Deleted values stay until compaction
  drops them.
- The CBA uses estimates. It does not profile real lookups.
- The model is in memory. It does not persist across restarts.

---

## 13. Paper guidelines mapped to code

| Paper idea | Where it lives |
|------------|----------------|
| Greedy-PLR with δ = 8 | `plr.Train`, `plr.DefaultDelta` |
| Model per file | `learn.Manager`, `sstable.Reader.SetModel` |
| Baseline lookup (WiscKey style) | `sstable.Reader.Get` |
| Model lookup (LoadChunk) | `sstable.Reader.GetWithModel` |
| Cost-benefit learning | `learn.EstimateBenefit`, `learn.Manager` |
| Three modes (never/always/CBA) | `learn.Mode` |
| Synthetic datasets | `bench.Build`, `bench/dataset.go` |
| T_wait before training | `lsm.Options.TWait`, `learn.Options` |

---

## 14. Checklist for a new change

1. State the change in one sentence.
2. Name the package you will edit.
3. Write or extend a test first.
4. Make the change. Keep sentences in comments short. Use active voice.
5. Run `gofmt -w .` and `go vet ./...`.
6. Run `go test ./... -count=1`.
7. If the change touches lookups, run the CLI comparison.
8. Update this document if the navigation advice becomes wrong.
