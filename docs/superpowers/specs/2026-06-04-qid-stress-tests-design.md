# Design: rocm-xio QID stress tests (quiesce + delete/create)

**Date:** 2026-06-04
**Branch:** `fix/qid8-restore-kernel-queue` (in the VM repo at `/home/ubuntu/src/rocm-xio`)
**Status:** approved design, pending implementation

## Problem

The rocm-xio kernel module manipulates kernel-owned NVMe queue state from
outside the NVMe driver: it quiesces/unquiesces individual blk-mq hardware
contexts (PR #174's `QUIESCE_NS`/`UNQUIESCE_NS`), and it deletes and
re-creates device-side queues while resetting the kernel's host-side ring
pointers (the `fix/qid8-restore-kernel-queue` resurrect path). These
operations reach into private `struct nvme_queue` / `struct nvme_dev`
state via offset-verified mirror structs.

We have unit-level coverage of the resurrect path (the existing
`nvme-qid8-ring-wrap-stress` test), but no coverage that exercises these
operations against a **warm** kernel — i.e. after the in-kernel NVMe
driver and block layer have already done substantial I/O and advanced
their internal state. The risk we are testing for: rocm-xio's
quiesce/delete/create operations corrupting or desyncing kernel state in
a way that only manifests under sustained, realistic I/O load.

These tests are **CPU-only** — no GPU is required for the two
primary tests, which sidesteps the currently-wedged Navi 21 GPU.

## Goals

1. **Quiesce isolation:** prove that when rocm-xio quiesces a queue, the
   kernel does not dispatch to it under heavy I/O, that *other* queues
   keep flowing, and that after unquiesce the queue resumes correctly
   without breaking the kernel.
2. **Delete/create resilience:** prove that deleting and re-creating a
   kernel-owned queue (via rocm-xio's resurrect path) leaves the kernel
   able to do heavy sustained I/O on the recreated queue without
   breaking.

## Non-goals

- GPU peer-DMA correctness (covered elsewhere; GPU is wedged anyway).
- Data-path performance benchmarking. These are correctness/integrity
  tests, not throughput tests.
- Testing the QEMU emulated NVMe model itself.

## Constraints / environment

- Tests run **entirely inside the VM** (`ssh -p 2222 ubuntu@localhost`),
  so the deliverable tests rely only on in-VM observables:
  latency/backpressure ordering, cross-CPU control, data integrity, and
  dmesg scanning.
- The QEMU host-side doorbell trace (`NVME_TRACE=doorbell|all` in
  `qemu/run-vm`) traces the VM's emulated NVMe but is only reachable from
  the **host** that launched the VM. It is therefore a **development /
  debugging aid** for the implementer and verifier — NOT part of the
  deliverable test's pass/fail logic.
- VM: kernel 6.8.0-117-generic, 8 vCPUs. Managed-IRQ mapping is
  `CPU N → hctx N → QID N+1`, so CPU 7 → QID 8, CPU 3 → QID 4. (See
  `qemu/QID8_RACE_REPORT.md`.)
- NVMe controller node defaults to `/dev/nvme2`, block dev `/dev/nvme2n1`.

## Trigger mechanism

All kernel-module operations are invoked through the **production
ioctls**, driven by small standalone C helpers (the pattern established
by `scripts/test/resurrect-qid.c`). No GPU, no xio-tester required for
the primary tests. No new production kernel code; new code is limited to
test helpers and the (already-present) test-only debug ioctls.

- `quiesce-qid <bdev> <qid>` → `ROCM_XIO_QUIESCE_NS`
- `unquiesce-qid <bdev> <qid>` → `ROCM_XIO_UNQUIESCE_NS`
- DELETE_SQ/DELETE_CQ → `nvme admin-passthru --opcode=0x00/0x04 --cdw10=<qid>`
  (identical wire commands to what xio-tester issues on exit)
- CREATE_CQ/CREATE_SQ + host ring-pointer reset → existing `resurrect-qid`
  helper via `ROCM_XIO_DEBUG_RESURRECT_QID`

## Architecture

### Shared infrastructure

- **`scripts/test/lib-qid-stress.sh`** — sourced bash library:
  - `resolve_bdf <ctrl>` → `0xBBDD` form for ioctls
  - `lba_size <bdev>`, `qid_for_cpu <cpu>` (= cpu+1)
  - `controller_reset_fresh <ctrl>` → reset + settle so the module
    re-captures a clean queue snapshot and the ring starts fresh
  - `timed_io <op> <cpu> <bdev> <lba> <bytes> <pattern> <timeout>` →
    pinned write/read, echoes wall-clock latency, nonzero rc on
    timeout/failure
  - `dmesg_bad <qid>` → greps `QID <qid> timeout|reset controller|Oops|
    kernel BUG|Invalid SQE|invalid queue|general protection`
  - `prewrap_ring <cpu> <bdev> <half_wraps>` → drive an odd number of
    half-ring-wraps of direct I/O so `cq_phase` flips off fresh state
- **`scripts/test/quiesce-qid.c`, `scripts/test/unquiesce-qid.c`** —
  tiny ioctl wrappers (mirror of `resurrect-qid.c`).
- **CMake:** compile the three helpers as build targets; register three
  ctests with labels `hardware nvme stress`.

### Test #1 — quiesce isolation (`test-qid-quiesce-stress.sh`)

ctest name `nvme-qid-quiesce-stress`. GPU-free. Validated now.

Per iteration:
1. `controller_reset_fresh`.
2. Warm-up: heavy direct I/O on CPU 7 (QID 8) **and** CPU 3 (QID 4) so
   both hctxs are warm (pointers advanced, ring wrapped).
3. `quiesce-qid /dev/nvme2n1 8` (real `QUIESCE_NS`).
4. Proof, two signals:
   - **Backpressure:** start a CPU-7-pinned write in the background,
     record start time. Assert it does NOT complete within the hold
     window (default 8 s) — blk-mq queues but does not dispatch it.
   - **Cross-CPU control:** during the same window, CPU-3-pinned I/O
     must complete normally (only the target hctx is stopped).
5. `unquiesce-qid /dev/nvme2n1 8`. Assert the blocked CPU-7 write now
   completes AND its completion timestamp is after the unquiesce call
   (ordering proof it was genuinely gated, not merely slow).
6. Verify data integrity of that write; `dmesg_bad 8`.

FAIL if: the CPU-7 write completes during the quiesce window; or CPU-3
I/O blocks during it; or the write never resumes after unquiesce; or
dmesg shows timeout/oops.

Teeth check (run by verifier): against a module whose `UNQUIESCE_NS` is
stubbed to no-op, the blocked write must hang forever → test FAILs,
proving it detects a broken unquiesce.

### Test #2 — GPU-free delete/create (`test-qid-recreate-stress.sh`)

ctest name `nvme-qid-recreate-stress`. GPU-free. Validated now.

Isolates queue-lifecycle + host ring-pointer-reset from any
PRP1-injection corruption — there is no rocm-xio hijack in this test.

Per iteration:
1. `controller_reset_fresh`.
2. Pre-I/O: `prewrap_ring 7 /dev/nvme2n1 <odd>` — pure kernel I/O that
   wraps QID 8's ring an odd number of times so host pointers move off
   fresh state. No rocm-xio involvement.
3. Delete + create: `nvme admin-passthru` DELETE_SQ + DELETE_CQ for
   QID 8, then `resurrect-qid 0x0500 8` (real `rocm_xio_resurrect_work_fn`:
   CREATE_CQ + CREATE_SQ at snapshotted addrs + host ring-pointer reset).
4. Post-create heavy I/O: sustained mixed read/write on CPU 7, many ring
   wraps, high iodepth (fio if available, else looped `dd`), with
   verification. This is the "lots of kernel I/O on the newly created
   queue" emphasis. A final write/read/compare confirms integrity.
5. `dmesg_bad 8` after each phase.

FAIL if: any post-create I/O times out/hangs/mismatches; or dmesg shows
QID-8 timeout/oops.

Teeth check (run by verifier): against a module with the host
ring-pointer reset stripped, the test must FAIL (matching the proven 3/3
FAIL of the existing ring-wrap test); with the fix, 5/5 PASS.

### Test #3 — GPU delete/create (`test-qid-recreate-gpu-stress.sh`)

ctest name `nvme-qid-recreate-gpu-stress`. **Gated behind
`STRESS_USE_GPU=1`**; skips cleanly otherwise. Written now, validated
LATER (after a host power-cycle recovers the Navi 21 GPU).

Same skeleton as #2, but step-2 "warm" I/O is driven through rocm-xio's
real path: xio-tester GPU workload hijacks QID 8 (kprobe rewrites PRP1 →
device queue points at GPU memory), runs I/O, exits (DELETE fires, kprobe
schedules resurrect). Step 4 then hammers the recreated queue with kernel
I/O. Faithful end-to-end version: GPU corruption, then verify
create/delete resolves it.

Marked clearly in the spec and the script banner as unvalidated until
the GPU recovers.

## Testing the tests (teeth)

A stress test that cannot fail proves nothing. Each primary test is run
twice by the verification agent — once against a deliberately-broken
module, once against the correct module — and must distinguish them:

- Test #1: broken = `UNQUIESCE_NS` no-op → must hang/FAIL.
- Test #2: broken = host ring-pointer reset removed → must FAIL with the
  QID-8 timeout signature.

## Files

| Artifact | New/Changed |
|---|---|
| `scripts/test/lib-qid-stress.sh` | new |
| `scripts/test/quiesce-qid.c` | new |
| `scripts/test/unquiesce-qid.c` | new |
| `scripts/test/test-qid-quiesce-stress.sh` | new |
| `scripts/test/test-qid-recreate-stress.sh` | new |
| `scripts/test/test-qid-recreate-gpu-stress.sh` | new (gated) |
| `tests/system/nvme-ep/CMakeLists.txt` | changed (register helpers + 3 ctests) |

No production kernel-module or userspace source changes.

## Risks / open questions

- **Backpressure observability (#1):** the test infers "queue unused"
  from a blocked background write that resumes only after unquiesce.
  This is indirect (no in-VM way to see doorbell rings). The
  completion-after-unquiesce ordering plus the cross-CPU control is the
  mitigation. The implementer may use the host `NVME_TRACE=doorbell` path
  during development to confirm zero doorbell rings on QID 8 during the
  window, strengthening confidence even though the trace is not in the
  deliverable.
- **`prewrap_ring` phase arithmetic:** the number of completions from a
  looped `dd`/fio may not exactly equal the requested block count if the
  block layer merges/splits. The teeth check (must-fail-without-fix)
  guards against a silently-too-weak prewrap.
- **fio availability:** post-create heavy I/O prefers fio; falls back to
  looped `dd` if fio is absent in the VM.
