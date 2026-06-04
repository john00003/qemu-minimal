# rocm-xio QID Stress Tests Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three CPU-only stress tests that exercise rocm-xio's queue manipulation (quiesce/unquiesce, delete/create) against a warm kernel and prove the kernel's I/O state stays correct.

**Architecture:** Bash test scripts driven by tiny C ioctl-wrapper helpers that call the *production* rocm-xio ioctls. Tests run entirely inside the VM and judge correctness from in-VM observables only: latency/backpressure ordering, cross-CPU control, data integrity, and dmesg scanning. Each primary test has a "teeth" check (deliberately break the module → test must fail). The QEMU `NVME_TRACE` host-side doorbell trace is a dev/debug aid only, never part of deliverable pass/fail.

**Tech Stack:** Linux kernel module (rocm-xio) in a QEMU VM (kernel 6.8.0-117-generic, 8 vCPUs), bash, C (compiled as CXX via CMake), `nvme-cli`, ctest. All work in the VM repo `/home/ubuntu/src/rocm-xio` on branch `fix/qid8-restore-kernel-queue`.

**Spec:** `docs/superpowers/specs/2026-06-04-qid-stress-tests-design.md` (in the host repo `/home/johtyler/code/qemu-minimal`)

---

## Environment & conventions (read before any task)

- **All build/run happens in the VM:** `ssh -p 2222 ubuntu@localhost`. Repo: `/home/ubuntu/src/rocm-xio`. The host repo `/home/johtyler/code/qemu-minimal` holds the spec/plan only.
- **Branch:** confirm `git -C ~/src/rocm-xio branch --show-current` == `fix/qid8-restore-kernel-queue`. Do NOT touch `main`, `pr-174`, `fix/file-to-bdev-on-65plus`. Do NOT push.
- **Device topology:** controller `/dev/nvme2`, block dev `/dev/nvme2n1`, PCI `0000:05:00.0` → BDF `0x0500`. Managed-IRQ map: CPU N → hctx N → QID N+1. So CPU 7 → QID 8 (target), CPU 3 → QID 4 (control).
- **Existing template to mirror:** `scripts/test/resurrect-qid.c` (a working ioctl-wrapper helper) and `scripts/test/test-qid8-ring-wrap-stress.sh` (a working stress test). Read both before writing new files; copy their idioms (BDF resolution, `timed_io`, dmesg scanning, controller reset).
- **Production ioctls already exist:** `ROCM_XIO_QUIESCE_NS`, `ROCM_XIO_UNQUIESCE_NS` (cmds 13/14), `ROCM_XIO_DEBUG_RESURRECT_QID` (cmd 15). UAPI in `src/include/rocm-xio-uapi.h`. The quiesce req struct is `struct rocm_xio_quiesce_ns_req { __s32 bdev_fd; __u32 qid; }`.
- **C helpers compiled as CXX:** the project enables only CXX + HIP, not C. Follow the existing `set_source_files_properties(... LANGUAGE CXX)` idiom for every `.c` helper.
- **Module build/load cycle:**
  ```
  cd ~/src/rocm-xio/kernel/rocm-xio
  make clean && make KDIR=/usr/src/linux-headers-$(uname -r)
  sudo rmmod rocm_xio   # if wedged: sudo reboot, wait ~30s, reconnect
  sudo cp rocm-xio.ko /lib/modules/$(uname -r)/updates/rocm-xio.ko
  sudo depmod -a && sudo modprobe rocm_xio
  cat /sys/module/rocm_xio/srcversion   # must match: modinfo rocm-xio.ko | grep srcversion
  ```
- **Running a ctest:** `cd ~/src/rocm-xio/build && sudo env ROCXIO_NVME_DEVICE=/dev/nvme2 NVME_DEVICE=/dev/nvme2 ctest -R "<name>" --output-on-failure`.
- **Dev/debug trace (NOT in deliverable):** the agent may, while developing, ask the orchestrator to relaunch the VM with `NVME_TRACE=doorbell NVME_TRACE_FILE=/path` (in `qemu/run-vm` on the host) to confirm zero doorbell rings on a quiesced QID. This is optional and host-side; never wire it into test pass/fail.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/test/quiesce-qid.c` | CLI helper: open bdev, call `ROCM_XIO_QUIESCE_NS` for `<bdev> <qid>` |
| `scripts/test/unquiesce-qid.c` | CLI helper: open bdev, call `ROCM_XIO_UNQUIESCE_NS` for `<bdev> <qid>` |
| `scripts/test/lib-qid-stress.sh` | Sourced bash library: BDF resolve, LBA size, timed_io, dmesg scanning (windowed), controller reset, prewrap |
| `scripts/test/test-qid-quiesce-stress.sh` | Test #1: quiesce isolation + resume |
| `scripts/test/test-qid-recreate-stress.sh` | Test #2: GPU-free delete/create + post-create hammer |
| `scripts/test/test-qid-recreate-gpu-stress.sh` | Test #3: GPU-driven delete/create (gated `STRESS_USE_GPU=1`) |
| `tests/system/nvme-ep/CMakeLists.txt` | Register 2 new helpers + 3 new ctests |

---

## Chunk 1: Helpers + shared library

### Task 1: `quiesce-qid` / `unquiesce-qid` CLI helpers

**Files:**
- Create: `scripts/test/quiesce-qid.c`
- Create: `scripts/test/unquiesce-qid.c`
- Reference: `scripts/test/resurrect-qid.c` (template), `src/include/rocm-xio-uapi.h` (ioctl numbers + struct)

- [ ] **Step 1: Read the template and UAPI**

Run on the VM: `cat ~/src/rocm-xio/scripts/test/resurrect-qid.c` and `sed -n '70,100p;220,240p' ~/src/rocm-xio/src/include/rocm-xio-uapi.h`.
Confirm: `ROCM_XIO_QUIESCE_NS = _IOW('R', 13, struct rocm_xio_quiesce_ns_req)`, `ROCM_XIO_UNQUIESCE_NS = _IOW('R', 14, ...)`, and the struct is `{ __s32 bdev_fd; __u32 qid; }`. If the real numbers/struct differ, use what the header says (header is authoritative).

- [ ] **Step 2: Write `quiesce-qid.c`**

The helper opens BOTH the rocm-xio control device and the target block device, because `QUIESCE_NS` takes a `bdev_fd` (the namespace block device fd) inside the request plus is issued on the `/dev/rocm-xio` fd.

```c
/* TEST-ONLY helper: quiesce one NVMe IO queue's blk-mq hctx via the
 * production ROCM_XIO_QUIESCE_NS ioctl.
 *
 * Usage: quiesce-qid <bdev> <qid>
 *   e.g. quiesce-qid /dev/nvme2n1 8
 *
 * Used by test-qid-quiesce-stress.sh. Mirrors resurrect-qid.c.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/types.h>

#define ROCM_XIO_IOC_MAGIC 'R'
struct rocm_xio_quiesce_ns_req {
  __s32 bdev_fd;
  __u32 qid;
};
#define ROCM_XIO_QUIESCE_NS \
  _IOW(ROCM_XIO_IOC_MAGIC, 13, struct rocm_xio_quiesce_ns_req)

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <bdev> <qid>\n", argv[0]);
    return 2;
  }
  const char* bdev = argv[1];
  unsigned int qid = (unsigned int)strtoul(argv[2], NULL, 0);

  int bdev_fd = open(bdev, O_RDONLY | O_CLOEXEC);
  if (bdev_fd < 0) {
    perror("open bdev");
    return 1;
  }
  int kfd = open("/dev/rocm-xio", O_RDWR);
  if (kfd < 0) {
    perror("open /dev/rocm-xio");
    close(bdev_fd);
    return 1;
  }
  struct rocm_xio_quiesce_ns_req req;
  req.bdev_fd = bdev_fd;
  req.qid = qid;
  if (ioctl(kfd, ROCM_XIO_QUIESCE_NS, &req) < 0) {
    perror("ioctl QUIESCE_NS");
    close(kfd);
    close(bdev_fd);
    return 1;
  }
  close(kfd);
  close(bdev_fd);
  printf("quiesced: bdev=%s qid=%u\n", bdev, qid);
  return 0;
}
```

- [ ] **Step 3: Write `unquiesce-qid.c`**

Identical to `quiesce-qid.c` except the ioctl name/number and messages:

```c
/* TEST-ONLY helper: unquiesce one NVMe IO queue's blk-mq hctx via the
 * production ROCM_XIO_UNQUIESCE_NS ioctl.
 *
 * Usage: unquiesce-qid <bdev> <qid>
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/types.h>

#define ROCM_XIO_IOC_MAGIC 'R'
struct rocm_xio_quiesce_ns_req {
  __s32 bdev_fd;
  __u32 qid;
};
#define ROCM_XIO_UNQUIESCE_NS \
  _IOW(ROCM_XIO_IOC_MAGIC, 14, struct rocm_xio_quiesce_ns_req)

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <bdev> <qid>\n", argv[0]);
    return 2;
  }
  const char* bdev = argv[1];
  unsigned int qid = (unsigned int)strtoul(argv[2], NULL, 0);

  int bdev_fd = open(bdev, O_RDONLY | O_CLOEXEC);
  if (bdev_fd < 0) { perror("open bdev"); return 1; }
  int kfd = open("/dev/rocm-xio", O_RDWR);
  if (kfd < 0) { perror("open /dev/rocm-xio"); close(bdev_fd); return 1; }

  struct rocm_xio_quiesce_ns_req req;
  req.bdev_fd = bdev_fd;
  req.qid = qid;
  if (ioctl(kfd, ROCM_XIO_UNQUIESCE_NS, &req) < 0) {
    perror("ioctl UNQUIESCE_NS");
    close(kfd); close(bdev_fd); return 1;
  }
  close(kfd);
  close(bdev_fd);
  printf("unquiesced: bdev=%s qid=%u\n", bdev, qid);
  return 0;
}
```

- [ ] **Step 4: Sanity-compile both standalone**

Run on VM: `cc -O2 -o /tmp/quiesce-qid ~/src/rocm-xio/scripts/test/quiesce-qid.c && cc -O2 -o /tmp/unquiesce-qid ~/src/rocm-xio/scripts/test/unquiesce-qid.c && echo COMPILE_OK`
Expected: `COMPILE_OK` (no warnings/errors). This is a standalone smoke compile; CMake wiring is Task 4.

- [ ] **Step 5: Smoke-test against the live module**

Run on VM (module must be loaded):
```
sudo /tmp/quiesce-qid /dev/nvme2n1 8 && sleep 1 && sudo /tmp/unquiesce-qid /dev/nvme2n1 8
sudo dmesg | grep -iE "QUIESCE_NS|UNQUIESCE_NS" | tail -4
```
Expected: both print success; dmesg shows `stopped hctx 7 (qid 8)` then `started hctx 7 (qid 8)`. If quiesce succeeds but the QID is left stopped, ALWAYS run unquiesce to restore — a stuck-stopped hctx will wedge later tests.

- [ ] **Step 6: Commit**

```bash
cd ~/src/rocm-xio
git add scripts/test/quiesce-qid.c scripts/test/unquiesce-qid.c
git commit -m "test/qid: add quiesce-qid/unquiesce-qid ioctl helpers"
```

### Task 2: shared bash library `lib-qid-stress.sh`

**Files:**
- Create: `scripts/test/lib-qid-stress.sh`
- Reference: `scripts/test/test-qid8-ring-wrap-stress.sh` (copy idioms: BDF resolve, LBA size, `timed_io`, dmesg scan, prewrap, controller reset)

- [ ] **Step 1: Extract & generalize the shared helpers**

Create `scripts/test/lib-qid-stress.sh` as a sourced library (no `set -u` here; the sourcing script sets shell options). Functions, each mirroring the proven logic in `test-qid8-ring-wrap-stress.sh`:

```bash
# lib-qid-stress.sh — shared helpers for QID stress tests.
# Source this; do not execute. Mirrors idioms proven in
# test-qid8-ring-wrap-stress.sh.

# resolve_bdf <ctrl-node>  -> echoes 0xBBDD
resolve_bdf() {
    local ctrl_name pci bus slot func devfn
    ctrl_name="$(basename "$1")"
    pci="$(basename "$(readlink -f "/sys/class/nvme/$ctrl_name/device")")"
    bus="0x$(echo "$pci" | cut -d: -f2)"
    slot="$(echo "$pci" | cut -d: -f3 | cut -d. -f1)"
    func="$(echo "$pci" | cut -d. -f2)"
    devfn=$(( (0x$slot << 3) | func ))
    printf "0x%02x%02x" "$bus" "$devfn"
}

# lba_size <bdev> -> echoes bytes (default 512)
lba_size() {
    local lbads
    lbads="$(${NVME_CMD:-nvme} id-ns "$1" 2>/dev/null | grep -E '^lbaf' \
        | grep 'in use' | head -1 | sed -E 's/.*lbads:([0-9]+).*/\1/')"
    [ -z "$lbads" ] && lbads=9
    echo $(( 1 << lbads ))
}

# controller_reset_fresh <ctrl-name>  (resets + settles for fresh snapshot)
controller_reset_fresh() {
    echo 1 > "/sys/class/nvme/$1/reset_controller" 2>/dev/null
    sleep 4
}

# dmesg_since <marker-file> <qid>  -> echoes bad lines since the marker was
# stamped. Uses a baseline snapshot (NOT dmesg -C) so pre-existing benign
# kernel noise is never misattributed. Caller stamps the marker with
# dmesg_mark before the window.
dmesg_mark() {  # writes current last-line marker to $1
    dmesg | tail -1 > "$1" 2>/dev/null || true
}
dmesg_bad_since() {  # $1=marker-file $2=qid -> prints bad lines, empty if clean
    local marker qid
    marker="$(cat "$1" 2>/dev/null)"
    qid="$2"
    # Print everything after the marker line, then grep for bad signatures.
    dmesg | awk -v m="$marker" 'f{print} $0==m{f=1}' \
        | grep -iE "QID $qid timeout|reset controller|Oops|kernel BUG|Invalid SQE|invalid queue|general protection" \
        || true
}

# prewrap_ring <cpu> <bdev> <half_wraps> <queue_len> <lba_size>
# Drives an ODD number of half-ring-wraps of direct reads so cq_phase
# flips off the device-fresh state. half_wraps MUST be odd.
prewrap_ring() {
    local cpu="$1" bdev="$2" half="$3" qlen="$4" lbsz="$5"
    local cmds=$(( half * qlen / 2 ))
    taskset -c "$cpu" dd if="$bdev" of=/dev/null bs="$lbsz" \
        count="$cmds" iflag=direct >/dev/null 2>&1
}

# timed_io <op:write|read> <cpu> <bdev> <lba> <bytes> <datafile> <timeout_s>
# Echoes wall-clock latency (seconds, 2dp); returns nonzero on
# timeout/failure. NLB is zero-based (blocks-1).
timed_io() {
    local op="$1" cpu="$2" bdev="$3" lba="$4" bytes="$5" data="$6" to="$7"
    local lbsz blocks nlb t0 t1 rc
    lbsz="$(lba_size "$bdev")"
    blocks=$(( bytes / lbsz )); nlb=$(( blocks - 1 ))
    t0=$(date +%s.%N)
    if [ "$op" = write ]; then
        timeout "$to" taskset -c "$cpu" "${NVME_CMD:-nvme}" write "$bdev" \
            --start-block="$lba" --block-count="$nlb" --data-size="$bytes" \
            --data="$data" >/dev/null 2>&1
    else
        timeout "$to" taskset -c "$cpu" "${NVME_CMD:-nvme}" read "$bdev" \
            --start-block="$lba" --block-count="$nlb" --data-size="$bytes" \
            --data="$data" >/dev/null 2>&1
    fi
    rc=$?
    t1=$(date +%s.%N)
    awk "BEGIN{printf \"%.2f\", $t1-$t0}"
    return $rc
}
```

- [ ] **Step 2: Shellcheck / source smoke test**

Run on VM: `bash -n ~/src/rocm-xio/scripts/test/lib-qid-stress.sh && echo SYNTAX_OK`. Then quick functional check:
```
( . ~/src/rocm-xio/scripts/test/lib-qid-stress.sh; echo "bdf=$(resolve_bdf /dev/nvme2) lba=$(lba_size /dev/nvme2n1)" )
```
Expected: `SYNTAX_OK` then `bdf=0x0500 lba=512` (or the real LBA size).

- [ ] **Step 3: Commit**

```bash
cd ~/src/rocm-xio
git add scripts/test/lib-qid-stress.sh
git commit -m "test/qid: add shared lib-qid-stress.sh helper library"
```

---

## Chunk 2: Test #1 — quiesce isolation

### Task 3: `test-qid-quiesce-stress.sh`

**Files:**
- Create: `scripts/test/test-qid-quiesce-stress.sh`
- Reference: `scripts/test/lib-qid-stress.sh`, `scripts/test/test-qid8-ring-wrap-stress.sh` (banner/summary style)

- [ ] **Step 1: Write the test script**

Key correctness properties (all three must hold or the iteration FAILs):
1. During the quiesce window, a CPU-7 background write must NOT complete.
2. During that same window, CPU-3 I/O MUST complete normally (control).
3. After unquiesce, the CPU-7 write completes, its data verifies, and its completion is observed only after the unquiesce call.

```bash
#!/bin/bash
# QID quiesce isolation stress test.
#
# Proves rocm-xio's per-hctx QUIESCE_NS stops the kernel from dispatching
# to a quiesced queue under load, leaves OTHER queues flowing, and that
# UNQUIESCE_NS cleanly resumes the queue without breaking the kernel.
#
# Entirely in-VM. Proof is by backpressure ordering + cross-CPU control +
# data integrity + dmesg, since doorbell tracing is only host-side.
#
# Env: ROCXIO_NVME_DEVICE (default /dev/nvme2), ROCXIO_NVME_BDEV
#      (default <ctrl>n1), STRESS_ITERS (default 5),
#      STRESS_CPU (default 7 -> QID 8), STRESS_CTRL_CPU (default 3 -> QID4),
#      QUIESCE_HOLD_S (default 8), FIRST_IO_TIMEOUT_S (default 8),
#      QUIESCE_HELPER / UNQUIESCE_HELPER (default ./build/{,un}quiesce-qid).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib-qid-stress.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
NVME_CTRL="${ROCXIO_NVME_DEVICE:-/dev/nvme2}"
NVME_BDEV="${ROCXIO_NVME_BDEV:-${NVME_CTRL}n1}"
ITERS="${STRESS_ITERS:-5}"
CPU="${STRESS_CPU:-7}"
CTRL_CPU="${STRESS_CTRL_CPU:-3}"
QID=$((CPU + 1)); CTRL_QID=$((CTRL_CPU + 1))
HOLD="${QUIESCE_HOLD_S:-8}"
IO_TO="${FIRST_IO_TIMEOUT_S:-8}"
NVME_CMD="${NVME_CMD:-nvme}"

QH="${QUIESCE_HELPER:-./build/quiesce-qid}"
UH="${UNQUIESCE_HELPER:-./build/unquiesce-qid}"
for c in "$QH" ./quiesce-qid /tmp/quiesce-qid; do [ -x "$c" ] && QH="$c" && break; done
for c in "$UH" ./unquiesce-qid /tmp/unquiesce-qid; do [ -x "$c" ] && UH="$c" && break; done

[ "$EUID" -eq 0 ] || { echo -e "${RED}must run as root${NC}"; exit 2; }
[ -b "$NVME_BDEV" ] || { echo -e "${RED}$NVME_BDEV not a block dev${NC}"; exit 2; }
[ -x "$QH" ] && [ -x "$UH" ] || { echo -e "${RED}need quiesce/unquiesce helpers${NC}"; exit 2; }

CTRL_NAME="$(basename "$NVME_CTRL")"
LBSZ="$(lba_size "$NVME_BDEV")"
VERIFY_BYTES="${STRESS_VERIFY_BYTES:-$((256 * 1024))}"
VERIFY_LBA=$(( 256 * 1024 * 1024 / LBSZ ))     # well clear of LBA 0
CTRL_LBA=$(( 512 * 1024 * 1024 / LBSZ ))       # control region, distinct
TMP="$(mktemp -d /tmp/qid-quiesce.XXXXXX)"; trap 'rm -rf "$TMP"' EXIT
PATTERN="$TMP/pattern.bin"; head -c "$VERIFY_BYTES" /dev/urandom > "$PATTERN"

echo "========= QID $QID quiesce isolation stress ========="
echo "ctrl=$NVME_CTRL bdev=$NVME_BDEV target=CPU$CPU/QID$QID control=CPU$CTRL_CPU/QID$CTRL_QID"
echo "iters=$ITERS hold=${HOLD}s io_timeout=${IO_TO}s"
echo "===================================================="

PASS=0; FAIL=0; REASONS=""

for ((it=1; it<=ITERS; it++)); do
    echo ""; echo "---- iteration $it/$ITERS ----"
    controller_reset_fresh "$CTRL_NAME"
    MARK="$TMP/mark.$it"; dmesg_mark "$MARK"

    # Warm both hctxs so internal state is advanced.
    echo "  [1] warm-up I/O on CPU$CPU and CPU$CTRL_CPU"
    taskset -c "$CPU" dd if="$NVME_BDEV" of=/dev/null bs="$LBSZ" \
        count=2048 iflag=direct >/dev/null 2>&1
    taskset -c "$CTRL_CPU" dd if="$NVME_BDEV" of=/dev/null bs="$LBSZ" \
        count=2048 iflag=direct >/dev/null 2>&1

    # Quiesce the target QID.
    echo "  [2] quiesce QID $QID"
    if ! "$QH" "$NVME_BDEV" "$QID" >/dev/null 2>&1; then
        echo -e "  ${RED}quiesce failed${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:quiesce-failed"; continue
    fi

    # Background write pinned to the quiesced CPU; must NOT finish during hold.
    echo "  [3] launch CPU$CPU write (must block during quiesce)"
    BG_DONE="$TMP/bgdone.$it"; rm -f "$BG_DONE"
    (
        taskset -c "$CPU" "$NVME_CMD" write "$NVME_BDEV" \
            --start-block="$VERIFY_LBA" \
            --block-count=$(( VERIFY_BYTES / LBSZ - 1 )) \
            --data-size="$VERIFY_BYTES" --data="$PATTERN" >/dev/null 2>&1
        echo "$(date +%s.%N)" > "$BG_DONE"
    ) &
    BG_PID=$!

    # Control I/O on a different CPU MUST succeed during the window.
    echo "  [4] control I/O on CPU$CTRL_CPU (must succeed during quiesce)"
    cdt="$(timed_io read "$CTRL_CPU" "$NVME_BDEV" "$CTRL_LBA" "$VERIFY_BYTES" "$TMP/ctrl.bin" "$IO_TO")"
    crc=$?
    if [ $crc -ne 0 ]; then
        echo -e "  ${RED}control I/O on CPU$CTRL_CPU blocked (${cdt}s) -- over-broad quiesce${NC}"
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:control-blocked"
        "$UH" "$NVME_BDEV" "$QID" >/dev/null 2>&1; kill "$BG_PID" 2>/dev/null; wait "$BG_PID" 2>/dev/null
        continue
    fi

    # Hold the quiesce; the background write must still be pending.
    sleep "$HOLD"
    if [ -f "$BG_DONE" ]; then
        echo -e "  ${RED}CPU$CPU write completed DURING quiesce -- not isolated${NC}"
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:not-isolated"
        "$UH" "$NVME_BDEV" "$QID" >/dev/null 2>&1; wait "$BG_PID" 2>/dev/null
        continue
    fi

    # Unquiesce; record the time, then the write should complete.
    echo "  [5] unquiesce QID $QID -> write must now complete"
    UNQ_T="$(date +%s.%N)"
    if ! "$UH" "$NVME_BDEV" "$QID" >/dev/null 2>&1; then
        echo -e "  ${RED}unquiesce failed${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:unquiesce-failed"; kill "$BG_PID" 2>/dev/null; wait "$BG_PID" 2>/dev/null; continue
    fi

    # Wait (bounded) for the background write to finish post-unquiesce.
    waited=0
    while [ ! -f "$BG_DONE" ] && [ "$waited" -lt "$IO_TO" ]; do sleep 0.2; waited=$((waited+1)); done
    wait "$BG_PID" 2>/dev/null
    if [ ! -f "$BG_DONE" ]; then
        echo -e "  ${RED}CPU$CPU write never completed after unquiesce -- kernel broke${NC}"
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:no-resume"; continue
    fi

    # Ordering proof: completion timestamp must be >= unquiesce timestamp.
    DONE_T="$(cat "$BG_DONE")"
    if awk "BEGIN{exit !($DONE_T < $UNQ_T)}"; then
        echo -e "  ${YELLOW}WARN: write completed before unquiesce timestamp${NC}"
        # Not a hard fail (clock granularity), but flag it.
    fi

    # Data integrity of the resumed write.
    timed_io read "$CPU" "$NVME_BDEV" "$VERIFY_LBA" "$VERIFY_BYTES" "$TMP/rb.bin" "$IO_TO" >/dev/null
    if ! cmp -s "$PATTERN" "$TMP/rb.bin"; then
        echo -e "  ${RED}data mismatch after resume${NC}"
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:data-mismatch"; continue
    fi

    BAD="$(dmesg_bad_since "$MARK" "$QID")"
    if [ -n "$BAD" ]; then
        echo -e "  ${RED}dmesg error:${NC}"; echo "$BAD" | tail -4 | sed 's/^/      /'
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:dmesg-error"; continue
    fi

    echo -e "  ${GREEN}PASS${NC} (control ok, target isolated then resumed, data ok)"
    PASS=$((PASS+1))
done

echo ""; echo "========= SUMMARY ========="
echo "iters=$ITERS PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || { echo "reasons:$REASONS"; echo -e "${RED}OVERALL: FAIL${NC}"; exit 1; }
echo -e "${GREEN}OVERALL: PASS${NC}"; exit 0
```

- [ ] **Step 2: Syntax check**

Run on VM: `bash -n ~/src/rocm-xio/scripts/test/test-qid-quiesce-stress.sh && echo SYNTAX_OK`. Expected: `SYNTAX_OK`.

- [ ] **Step 3: Run against the CURRENT (correct) module — expect PASS**

Build helpers standalone for the trial run (CMake wiring is Task 4):
```
cc -O2 -o /tmp/quiesce-qid ~/src/rocm-xio/scripts/test/quiesce-qid.c
cc -O2 -o /tmp/unquiesce-qid ~/src/rocm-xio/scripts/test/unquiesce-qid.c
cd ~/src/rocm-xio
sudo env QUIESCE_HELPER=/tmp/quiesce-qid UNQUIESCE_HELPER=/tmp/unquiesce-qid \
    STRESS_ITERS=3 bash scripts/test/test-qid-quiesce-stress.sh
```
Expected: `OVERALL: PASS`, 3/3.

- [ ] **Step 4: TEETH — break UNQUIESCE_NS, expect FAIL**

In a scratch copy of the module, stub the `ROCM_XIO_UNQUIESCE_NS` ioctl to `return 0;` immediately (no actual `blk_mq_start_hw_queue`). Rebuild + load that broken module, rerun the test. Expected: the CPU-7 write never resumes → `iter*:no-resume` → `OVERALL: FAIL`. This proves the test detects a broken unquiesce.
**IMPORTANT:** after the teeth run, restore the correct module (rebuild from the committed source, reload) and confirm `cat /sys/module/rocm_xio/srcversion` matches the committed build, so later tasks run against the real module. If any QID was left quiesced, run `unquiesce-qid` to restore it.

- [ ] **Step 5: Commit (test only; do not commit the broken module)**

```bash
cd ~/src/rocm-xio
git add scripts/test/test-qid-quiesce-stress.sh
git commit -m "test/qid: quiesce isolation stress test (teeth-verified)"
```

---

## Chunk 3: Test #2 — GPU-free delete/create, and CMake wiring

### Task 4: `test-qid-recreate-stress.sh` (GPU-free)

**Files:**
- Create: `scripts/test/test-qid-recreate-stress.sh`
- Reference: `scripts/test/lib-qid-stress.sh`, `scripts/test/test-qid8-ring-wrap-stress.sh`, `scripts/test/resurrect-qid.c`

- [ ] **Step 1: Write the test script**

Per iteration: reset fresh → pre-wrap (odd half-wraps, flips cq_phase) → DELETE_SQ/DELETE_CQ via `nvme admin-passthru` → `resurrect-qid <bdf> <qid>` → **post-create heavy I/O** (sustained, many wraps, verified) → dmesg scan. This isolates queue-lifecycle correctness; NO rocm-xio hijack/PRP1 injection here.

```bash
#!/bin/bash
# GPU-free QID delete/create stress test.
#
# Pure-kernel I/O warms QID 8 (odd ring wraps flip cq_phase), then the
# device queue is DELETEd (admin-passthru) and recreated via the real
# resurrect path (resurrect-qid -> ROCM_XIO_DEBUG_RESURRECT_QID, which
# runs CREATE_CQ/CREATE_SQ + host ring-pointer reset). Then a SUSTAINED
# post-create I/O hammer on the recreated queue must complete cleanly.
#
# Isolates queue-lifecycle + host-ptr-reset from PRP1-injection: there is
# no rocm-xio hijack in this test. Entirely in-VM.
#
# Env: ROCXIO_NVME_DEVICE (/dev/nvme2), ROCXIO_NVME_BDEV (<ctrl>n1),
#      STRESS_ITERS (5), STRESS_CPU (7), STRESS_HALF_WRAPS (3, must be odd),
#      QUEUE_LENGTH (1024), POST_HAMMER_MB (256), FIRST_IO_TIMEOUT_S (8),
#      RESURRECT_HELPER (./build/resurrect-qid).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib-qid-stress.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
NVME_CTRL="${ROCXIO_NVME_DEVICE:-/dev/nvme2}"
NVME_BDEV="${ROCXIO_NVME_BDEV:-${NVME_CTRL}n1}"
ITERS="${STRESS_ITERS:-5}"
CPU="${STRESS_CPU:-7}"; QID=$((CPU+1))
HALF_WRAPS="${STRESS_HALF_WRAPS:-3}"
QUEUE_LENGTH="${QUEUE_LENGTH:-1024}"
POST_MB="${POST_HAMMER_MB:-256}"
IO_TO="${FIRST_IO_TIMEOUT_S:-8}"
NVME_CMD="${NVME_CMD:-nvme}"

RH="${RESURRECT_HELPER:-./build/resurrect-qid}"
for c in "$RH" ./resurrect-qid /tmp/resurrect-qid; do [ -x "$c" ] && RH="$c" && break; done

[ "$EUID" -eq 0 ] || { echo -e "${RED}must run as root${NC}"; exit 2; }
[ -b "$NVME_BDEV" ] || { echo -e "${RED}$NVME_BDEV not a block dev${NC}"; exit 2; }
[ -x "$RH" ] || { echo -e "${RED}need resurrect-qid helper${NC}"; exit 2; }
if [ $((HALF_WRAPS % 2)) -eq 0 ]; then
    echo -e "${YELLOW}WARN: STRESS_HALF_WRAPS even; cq_phase stays fresh (weak test)${NC}"
fi

CTRL_NAME="$(basename "$NVME_CTRL")"
BDF="$(resolve_bdf "$NVME_CTRL")"
LBSZ="$(lba_size "$NVME_BDEV")"
VERIFY_BYTES="${STRESS_VERIFY_BYTES:-$((256*1024))}"
VERIFY_LBA=$(( 256*1024*1024 / LBSZ ))
TMP="$(mktemp -d /tmp/qid-recreate.XXXXXX)"; trap 'rm -rf "$TMP"' EXIT
PATTERN="$TMP/pattern.bin"; head -c "$VERIFY_BYTES" /dev/urandom > "$PATTERN"
POST_COUNT=$(( POST_MB * 1024 * 1024 / LBSZ ))

echo "========= QID $QID delete/create stress (GPU-free) ========="
echo "ctrl=$NVME_CTRL ($BDF) bdev=$NVME_BDEV cpu=$CPU qid=$QID"
echo "iters=$ITERS half_wraps=$HALF_WRAPS post_hammer=${POST_MB}MB"
echo "==========================================================="

delete_dev_queue() {
    $NVME_CMD admin-passthru "$NVME_CTRL" --opcode=0x00 --cdw10="$QID" >/dev/null 2>&1
    $NVME_CMD admin-passthru "$NVME_CTRL" --opcode=0x04 --cdw10="$QID" >/dev/null 2>&1
}

PASS=0; FAIL=0; REASONS=""
for ((it=1; it<=ITERS; it++)); do
    echo ""; echo "---- iteration $it/$ITERS ----"
    controller_reset_fresh "$CTRL_NAME"
    MARK="$TMP/mark.$it"; dmesg_mark "$MARK"

    echo "  [1] pre-wrap on CPU$CPU ($HALF_WRAPS half-wraps -> flip cq_phase)"
    prewrap_ring "$CPU" "$NVME_BDEV" "$HALF_WRAPS" "$QUEUE_LENGTH" "$LBSZ"

    echo "  [2] DELETE_SQ+DELETE_CQ (admin-passthru) + resurrect (real path)"
    delete_dev_queue
    if ! "$RH" "$BDF" "$QID" >/dev/null 2>&1; then
        echo -e "  ${RED}resurrect helper failed${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:resurrect-failed"; sleep 35; continue
    fi
    sleep 1   # let the 250ms delayed resurrect work fire

    echo "  [3] post-create hammer: ${POST_MB}MB direct read on CPU$CPU"
    t0=$(date +%s.%N)
    if ! timeout $(( IO_TO * 8 )) taskset -c "$CPU" dd if="$NVME_BDEV" \
            of=/dev/null bs="$LBSZ" count="$POST_COUNT" iflag=direct \
            >/dev/null 2>&1; then
        echo -e "  ${RED}post-create hammer hung/timed out${NC}"
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:hammer-timeout"
        sleep 35; dmesg_bad_since "$MARK" "$QID" | tail -4 | sed 's/^/      DMESG: /'
        continue
    fi
    t1=$(date +%s.%N)

    echo "  [4] post-create verify write/read/compare on CPU$CPU"
    if ! timed_io write "$CPU" "$NVME_BDEV" "$VERIFY_LBA" "$VERIFY_BYTES" "$PATTERN" "$IO_TO" >/dev/null; then
        echo -e "  ${RED}verify write timeout${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:write-timeout"; sleep 35; continue
    fi
    if ! timed_io read "$CPU" "$NVME_BDEV" "$VERIFY_LBA" "$VERIFY_BYTES" "$TMP/rb.bin" "$IO_TO" >/dev/null; then
        echo -e "  ${RED}verify read timeout${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:read-timeout"; sleep 35; continue
    fi
    if ! cmp -s "$PATTERN" "$TMP/rb.bin"; then
        echo -e "  ${RED}data mismatch${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:data-mismatch"; continue
    fi

    BAD="$(dmesg_bad_since "$MARK" "$QID")"
    if [ -n "$BAD" ]; then
        echo -e "  ${RED}dmesg error:${NC}"; echo "$BAD" | tail -4 | sed 's/^/      /'
        FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:dmesg-error"; continue
    fi
    hdt="$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")"
    echo -e "  ${GREEN}PASS${NC} (hammer ${POST_MB}MB in ${hdt}s, data ok, dmesg clean)"
    PASS=$((PASS+1))
done

echo ""; echo "========= SUMMARY ========="
echo "iters=$ITERS PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || { echo "reasons:$REASONS"; echo -e "${RED}OVERALL: FAIL${NC}"; exit 1; }
echo -e "${GREEN}OVERALL: PASS${NC}"; exit 0
```

- [ ] **Step 2: Syntax check**

Run on VM: `bash -n ~/src/rocm-xio/scripts/test/test-qid-recreate-stress.sh && echo SYNTAX_OK`. Expected: `SYNTAX_OK`.

- [ ] **Step 3: Run against the CURRENT (correct) module — expect PASS**

```
cd ~/src/rocm-xio
sudo env RESURRECT_HELPER=/tmp/resurrect-qid STRESS_ITERS=3 \
    bash scripts/test/test-qid-recreate-stress.sh
```
(Reuse the `/tmp/resurrect-qid` built earlier, or build from `scripts/test/resurrect-qid.c`.) Expected: `OVERALL: PASS`, 3/3.

- [ ] **Step 4: TEETH — strip the host ring-pointer reset, expect FAIL**

In a scratch copy of the module, remove/short-circuit the four-pointer write-back in `rocm_xio_resurrect_work_fn` (the block that sets `sq_tail/last_sq_tail/cq_head/cq_phase`). Rebuild + load. Rerun the test. Expected: post-create hammer or verify hangs → `OVERALL: FAIL` with QID-8 timeout in dmesg, matching the proven ring-wrap behavior.
**IMPORTANT:** restore the correct committed module afterward (rebuild from committed source, reload, confirm srcversion) before continuing.

- [ ] **Step 5: Commit**

```bash
cd ~/src/rocm-xio
git add scripts/test/test-qid-recreate-stress.sh
git commit -m "test/qid: GPU-free delete/create stress test (teeth-verified)"
```

### Task 5: CMake wiring for helpers + tests #1/#2

**Files:**
- Modify: `tests/system/nvme-ep/CMakeLists.txt` (after the existing ring-wrap block at ~line 530)

- [ ] **Step 1: Add helper executables and ctests**

Append after the existing `nvme-qid8-ring-wrap-stress` registration, mirroring its idioms exactly (compile `.c` as CXX; output to build root; `xio_add_script_test` with the `_verify_wrapper` SCRIPT and the test script as ARGS; pass helper paths via ENVIRONMENT):

```cmake
# --- QID quiesce + delete/create stress tests --------------
# CPU-only stress tests exercising rocm-xio queue manipulation against a
# warm kernel. GPU-free (use production QUIESCE_NS/UNQUIESCE_NS and the
# DEBUG_RESURRECT_QID ioctls); run even with the GPU wedged.
set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/scripts/test/quiesce-qid.c
  ${CMAKE_SOURCE_DIR}/scripts/test/unquiesce-qid.c
  PROPERTIES LANGUAGE CXX)
add_executable(quiesce-qid ${CMAKE_SOURCE_DIR}/scripts/test/quiesce-qid.c)
add_executable(unquiesce-qid ${CMAKE_SOURCE_DIR}/scripts/test/unquiesce-qid.c)
set_target_properties(quiesce-qid unquiesce-qid PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(_qid_quiesce_stress
  ${CMAKE_SOURCE_DIR}/scripts/test/test-qid-quiesce-stress.sh)
xio_add_script_test(
  NAME nvme-qid-quiesce-stress
  SCRIPT ${_verify_wrapper}
  LABELS hardware nvme stress
  TIMEOUT 300
  ARGS ${_qid_quiesce_stress}
  ENVIRONMENT
    "STRESS_ITERS=5"
    "QUIESCE_HELPER=${CMAKE_BINARY_DIR}/quiesce-qid"
    "UNQUIESCE_HELPER=${CMAKE_BINARY_DIR}/unquiesce-qid"
)

set(_qid_recreate_stress
  ${CMAKE_SOURCE_DIR}/scripts/test/test-qid-recreate-stress.sh)
xio_add_script_test(
  NAME nvme-qid-recreate-stress
  SCRIPT ${_verify_wrapper}
  LABELS hardware nvme stress
  TIMEOUT 600
  ARGS ${_qid_recreate_stress}
  ENVIRONMENT
    "STRESS_ITERS=5"
    "STRESS_HALF_WRAPS=3"
    "RESURRECT_HELPER=${CMAKE_BINARY_DIR}/resurrect-qid"
)
```

- [ ] **Step 2: Reconfigure + build the helpers**

```
cd ~/src/rocm-xio/build && cmake . >/dev/null && make quiesce-qid unquiesce-qid resurrect-qid 2>&1 | tail -5
ls -la quiesce-qid unquiesce-qid resurrect-qid
```
Expected: three executables in the build dir.

- [ ] **Step 3: Run both ctests against the correct module — expect PASS**

```
cd ~/src/rocm-xio/build
sudo env ROCXIO_NVME_DEVICE=/dev/nvme2 NVME_DEVICE=/dev/nvme2 \
    ctest -R "nvme-qid-quiesce-stress|nvme-qid-recreate-stress" --output-on-failure
```
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
cd ~/src/rocm-xio
git add tests/system/nvme-ep/CMakeLists.txt
git commit -m "test/qid: register quiesce + delete/create stress ctests"
```

---

## Chunk 4: Test #3 — GPU delete/create (gated, written now / validated later)

### Task 6: `test-qid-recreate-gpu-stress.sh`

**Files:**
- Create: `scripts/test/test-qid-recreate-gpu-stress.sh`
- Modify: `tests/system/nvme-ep/CMakeLists.txt` (add gated ctest)
- Reference: `scripts/test/test-qid-recreate-stress.sh` (skeleton), `tests/system/nvme-ep/CMakeLists.txt` existing xio-tester invocation (for the exact GPU command)

- [ ] **Step 1: Write the gated GPU test**

Same skeleton as Test #2, but the warm phase is driven by xio-tester's real GPU hijack instead of pure-kernel pre-wrap. Gate the whole body behind `STRESS_USE_GPU=1`; otherwise print a clear SKIP and exit 0. Banner must state it is unvalidated until the Navi 21 GPU is power-cycled.

```bash
#!/bin/bash
# GPU-driven QID delete/create stress test (FAITHFUL end-to-end).
#
# Warm phase uses xio-tester's real GPU hijack of QID 8 (kprobe rewrites
# PRP1 -> device queue points at GPU memory), then xio-tester exit fires
# DELETE and the kprobe schedules the real resurrect. Then a sustained
# kernel I/O hammer on the recreated queue must complete cleanly.
#
# GATED behind STRESS_USE_GPU=1. UNVALIDATED until the Navi 21 GPU is
# power-cycled (AMD reset bug wedges the GPU; a guest reboot does not
# recover it). Skips cleanly when the gate is unset.
#
# Env: STRESS_USE_GPU (must be 1 to run), XIO_TESTER (./build/xio-tester),
#      plus the same vars as test-qid-recreate-stress.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib-qid-stress.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
if [ "${STRESS_USE_GPU:-0}" != "1" ]; then
    echo -e "${YELLOW}SKIP: GPU delete/create test gated behind STRESS_USE_GPU=1${NC}"
    echo "  (unvalidated until the Navi 21 GPU is power-cycled)"
    exit 0
fi

NVME_CTRL="${ROCXIO_NVME_DEVICE:-/dev/nvme2}"
NVME_BDEV="${ROCXIO_NVME_BDEV:-${NVME_CTRL}n1}"
ITERS="${STRESS_ITERS:-3}"
CPU="${STRESS_CPU:-7}"; QID=$((CPU+1))
POST_MB="${POST_HAMMER_MB:-256}"
IO_TO="${FIRST_IO_TIMEOUT_S:-8}"
NVME_CMD="${NVME_CMD:-nvme}"
XIO_TESTER="${XIO_TESTER:-./build/xio-tester}"
QUEUE_LENGTH="${QUEUE_LENGTH:-1024}"
LFSR_SEED="${LFSR_SEED:-0x1234}"; WRITE_IO="${WRITE_IO:-4}"; BLOCKS_PER_CMD="${BLOCKS_PER_CMD:-8}"

[ "$EUID" -eq 0 ] || { echo -e "${RED}must run as root${NC}"; exit 2; }
[ -b "$NVME_BDEV" ] || { echo -e "${RED}$NVME_BDEV not a block dev${NC}"; exit 2; }
[ -x "$XIO_TESTER" ] || { echo -e "${RED}xio-tester not found at $XIO_TESTER${NC}"; exit 2; }

CTRL_NAME="$(basename "$NVME_CTRL")"
LBSZ="$(lba_size "$NVME_BDEV")"
VERIFY_BYTES="${STRESS_VERIFY_BYTES:-$((256*1024))}"
VERIFY_LBA=$(( 256*1024*1024 / LBSZ ))
POST_COUNT=$(( POST_MB*1024*1024 / LBSZ ))
TMP="$(mktemp -d /tmp/qid-gpu.XXXXXX)"; trap 'rm -rf "$TMP"' EXIT
PATTERN="$TMP/pattern.bin"; head -c "$VERIFY_BYTES" /dev/urandom > "$PATTERN"

echo "===== QID $QID delete/create stress (GPU/xio-tester) ====="
echo "ctrl=$NVME_CTRL bdev=$NVME_BDEV cpu=$CPU qid=$QID xio=$XIO_TESTER"
echo "========================================================="

run_xio() {
    "$XIO_TESTER" nvme-ep -v --write-io "$WRITE_IO" --controller "$NVME_CTRL" \
        --queue-length "$QUEUE_LENGTH" --lfsr-seed "$LFSR_SEED" --base-lba 0 \
        --lbas-per-io "$BLOCKS_PER_CMD" --access-pattern sequential \
        --batch-size 0 -m 0 > "$TMP/xio.log" 2>&1
}

PASS=0; FAIL=0; REASONS=""
for ((it=1; it<=ITERS; it++)); do
    echo ""; echo "---- iteration $it/$ITERS ----"
    controller_reset_fresh "$CTRL_NAME"
    MARK="$TMP/mark.$it"; dmesg_mark "$MARK"

    echo "  [1] xio-tester GPU hijack of QID $QID (warm + DELETE on exit)"
    run_xio || { echo -e "  ${YELLOW}xio-tester nonzero; tail:${NC}"; tail -3 "$TMP/xio.log" | sed 's/^/      /'; }
    sleep 1   # let the 250ms delayed resurrect fire

    echo "  [2] post-create hammer ${POST_MB}MB on CPU$CPU"
    if ! timeout $(( IO_TO*8 )) taskset -c "$CPU" dd if="$NVME_BDEV" of=/dev/null \
            bs="$LBSZ" count="$POST_COUNT" iflag=direct >/dev/null 2>&1; then
        echo -e "  ${RED}hammer hung/timed out${NC}"; FAIL=$((FAIL+1))
        REASONS="$REASONS iter$it:hammer-timeout"; sleep 35
        dmesg_bad_since "$MARK" "$QID" | tail -4 | sed 's/^/      DMESG: /'; continue
    fi

    echo "  [3] verify write/read/compare on CPU$CPU"
    timed_io write "$CPU" "$NVME_BDEV" "$VERIFY_LBA" "$VERIFY_BYTES" "$PATTERN" "$IO_TO" >/dev/null || {
        echo -e "  ${RED}verify write timeout${NC}"; FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:write-timeout"; continue; }
    timed_io read "$CPU" "$NVME_BDEV" "$VERIFY_LBA" "$VERIFY_BYTES" "$TMP/rb.bin" "$IO_TO" >/dev/null || {
        echo -e "  ${RED}verify read timeout${NC}"; FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:read-timeout"; continue; }
    cmp -s "$PATTERN" "$TMP/rb.bin" || {
        echo -e "  ${RED}data mismatch${NC}"; FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:data-mismatch"; continue; }

    BAD="$(dmesg_bad_since "$MARK" "$QID")"
    [ -z "$BAD" ] || { echo -e "  ${RED}dmesg error:${NC}"; echo "$BAD" | tail -4 | sed 's/^/      /'; FAIL=$((FAIL+1)); REASONS="$REASONS iter$it:dmesg-error"; continue; }

    echo -e "  ${GREEN}PASS${NC} (GPU hijack + resurrect + hammer ok)"
    PASS=$((PASS+1))
done

echo ""; echo "========= SUMMARY ========="
echo "iters=$ITERS PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || { echo "reasons:$REASONS"; echo -e "${RED}OVERALL: FAIL${NC}"; exit 1; }
echo -e "${GREEN}OVERALL: PASS${NC}"; exit 0
```

- [ ] **Step 2: Syntax check + gate check**

Run on VM:
```
bash -n ~/src/rocm-xio/scripts/test/test-qid-recreate-gpu-stress.sh && echo SYNTAX_OK
sudo bash ~/src/rocm-xio/scripts/test/test-qid-recreate-gpu-stress.sh   # no STRESS_USE_GPU
```
Expected: `SYNTAX_OK`, then a clean `SKIP` and exit 0 (proves the gate works without a GPU).

- [ ] **Step 3: Register the gated ctest in CMake**

Append to `tests/system/nvme-ep/CMakeLists.txt` after the Task-5 block:

```cmake
set(_qid_recreate_gpu_stress
  ${CMAKE_SOURCE_DIR}/scripts/test/test-qid-recreate-gpu-stress.sh)
xio_add_script_test(
  NAME nvme-qid-recreate-gpu-stress
  SCRIPT ${_verify_wrapper}
  LABELS hardware nvme stress gpu
  TIMEOUT 600
  ARGS ${_qid_recreate_gpu_stress}
  ENVIRONMENT
    "STRESS_ITERS=3"
    "XIO_TESTER=${CMAKE_BINARY_DIR}/xio-tester"
)
# NOTE: this ctest SKIPs (exit 0) unless STRESS_USE_GPU=1 is also set;
# it is unvalidated until the Navi 21 GPU is power-cycled.
```

- [ ] **Step 4: Reconfigure + confirm the ctest registers and SKIPs**

```
cd ~/src/rocm-xio/build && cmake . >/dev/null
sudo env ROCXIO_NVME_DEVICE=/dev/nvme2 ctest -R "nvme-qid-recreate-gpu-stress" --output-on-failure
```
Expected: the test runs, prints SKIP, and ctest reports PASS (exit 0).

- [ ] **Step 5: Commit**

```bash
cd ~/src/rocm-xio
git add scripts/test/test-qid-recreate-gpu-stress.sh tests/system/nvme-ep/CMakeLists.txt
git commit -m "test/qid: gated GPU-driven delete/create stress test (run after GPU power-cycle)"
```

---

## Final verification (after all tasks)

- [ ] **Full GPU-free stress suite green against the correct module**

```
cd ~/src/rocm-xio/build
cat /sys/module/rocm_xio/srcversion   # must match committed module
sudo env ROCXIO_NVME_DEVICE=/dev/nvme2 NVME_DEVICE=/dev/nvme2 \
    ctest -R "nvme-qid-quiesce-stress|nvme-qid-recreate-stress" --output-on-failure
sudo env ROCXIO_NVME_DEVICE=/dev/nvme2 ctest -R "nvme-qid-recreate-gpu-stress" --output-on-failure  # SKIPs
```
Expected: quiesce + recreate PASS; gpu test SKIPs (PASS). Confirm no QID left quiesced and module srcversion matches committed source.

- [ ] **Confirm branch state**

```
cd ~/src/rocm-xio && git log --oneline -7 && git status --short
```
Expected: the new commits present, working tree clean, still on `fix/qid8-restore-kernel-queue`, nothing pushed.

---

## Notes for the executor

- **Teeth are mandatory.** Tasks 3 and 4 each require proving the test FAILS against a deliberately-broken module before committing. A test that passes against both the correct and broken module is worthless — make it harder until it distinguishes them. Always restore the correct committed module afterward (verify `srcversion`).
- **Broken-module hygiene (teeth steps).** Build the deliberately-broken `.ko` into a scratch location and load it for the teeth run ONLY; NEVER `git add`/`commit` the broken edit, and NEVER leave the broken `.ko` in `/lib/modules/$(uname -r)/updates/`. Explicit safe sequence: (1) `git stash` is NOT enough — instead make the breaking edit, `make`, `sudo cp rocm-xio.ko /lib/modules/$(uname -r)/updates/`, `sudo depmod -a`, reload, run the test (expect FAIL); (2) revert the edit with `git checkout -- kernel/rocm-xio/rocm-xio.c`, `make clean && make`, recopy, `depmod`, reload; (3) confirm `cat /sys/module/rocm_xio/srcversion` equals `modinfo kernel/rocm-xio/rocm-xio.ko | grep srcversion` built from committed source. Only then proceed.
- **dmesg windowing is intentional.** `lib-qid-stress.sh` uses `dmesg_mark`/`dmesg_bad_since` rather than the template's `dmesg -C`. This is a deliberate improvement (avoids misattributing pre-existing benign kernel noise) — do not "fix" it back to `dmesg -C`.
- **Never leave a QID quiesced.** If any quiesce step errors mid-iteration, run `unquiesce-qid` before moving on; a stuck-stopped hctx wedges later tests and confuses results.
- **dmesg windowing, not `dmesg -C`.** Use `dmesg_mark`/`dmesg_bad_since` so pre-existing benign kernel noise is never misattributed (spec reviewer's note).
- **Dev-time trace is allowed.** While debugging a failure you may ask the orchestrator to relaunch the VM with `NVME_TRACE=doorbell` (host-side, `qemu/run-vm`) to confirm doorbell behavior. Never wire it into deliverable pass/fail.
- **half_wraps must stay odd** (default 3) — an even count lands `cq_phase` back on fresh and silently weakens Test #2.
