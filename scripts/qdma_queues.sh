#!/usr/bin/env bash
# qdma_queues.sh — create/destroy the MM queue pair this platform needs.
#
#     sudo ./qdma_queues.sh setup      # load the driver, create + start the MM pair
#     ./qdma_queues.sh status          # what exists right now (no root needed)
#     sudo ./qdma_queues.sh teardown   # delete the queues, unload the driver
#
# THE DRIVER IS NOT INSTALLED
# It is used straight out of its build tree — `make` there, no `make install` — so
# it is not under /lib/modules and modprobe cannot see it.  setup insmods it by path
# and teardown rmmods it by name.  QDMA_TREE / KO override the paths.
#
# WHY THIS IS A SCRIPT AND NOT PART OF A LIBRARY
# Queue creation is a root-only, system-wide, once-per-boot administration step: it
# changes state other processes share, and a library that created queues behind a
# caller's back would make two tools fight over them.
#
# NODE NAMING — THE NUMBER IS THE QUEUE INDEX, NOT THE DIRECTION
# The name is <QDEV>-<MM|ST>-<qidx> and says nothing about direction: a node serves
# H2C on write() and C2H on read(), for whichever directions were `q add`ed at that
# index. "MM-0 is h2c, MM-1 is c2h" is a CONVENTION this script establishes, not a
# rule the driver enforces.
#
# Getting it wrong is NOT caught. The queue handle for H2C at index N is literally
# N, and the cdev struct is zero-allocated, so on a c2h-only node the h2c handle
# stays 0 — a perfectly valid handle for H2C queue 0. A write() to the wrong node
# lands on queue 0 with no error at all.
#
# DO NOT set 'aperture_sz'. It makes a queue's AXI address wrap inside a
# power-of-two window: a transfer past the aperture silently folds back to the
# start instead of walking up memory. Leaving it unset gives linear addressing.
#
# SIBLING SCRIPT
# bank_controller_top/SW/qdma_queues.sh is the same logic for the other platform.
# Fixes belong in both. The non-obvious ones already carried here:
#   - dma-ctl's EXIT STATUS CANNOT BE TRUSTED: with qmax==0 it prints "Zero Qs" and
#     still exits 0. Only the /dev node appearing proves anything.
#   - `q list` validates start_idx first, so it errors out if queue 0 was never
#     added — it cannot be used to blind-scan an empty device.
#   - idx_ringsz is an INDEX into a table, not a size (10 = 3072 on this build).
#     dma-ctl's "Default ring size set to 2048" message is wrong: omitting it
#     selects index 9, which is 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../platform/select.sh
source "$SCRIPT_DIR/../platform/select.sh"

# The driver is used FROM ITS BUILD TREE — `make` there, no `make install`.  So
# nothing is in /lib/modules and modprobe cannot find it: the module is insmod'd by
# path, and `make` puts both it and dma-ctl in bin/.
QDMA_TREE="${QDMA_TREE:-/home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel}"
DMACTL="${DMACTL:-$(command -v dma-ctl || true)}"
[[ -n "$DMACTL" ]] || DMACTL="$QDMA_TREE/bin/dma-ctl"
KO="${KO:-$QDMA_TREE/bin/${DRIVER}.ko}"
MODNAME="${DRIVER//-/_}"

H2C_DEV="/dev/${QDEV}-MM-${H2C_IDX}"
C2H_DEV="/dev/${QDEV}-MM-${C2H_IDX}"

log() { echo "[qdma] $*"; }

need_root() {
    [[ "$(id -u)" == 0 ]] || { echo "ERROR: must run as root (sudo $0 $*)" >&2; exit 2; }
}

loaded() { lsmod | grep -q "^${MODNAME} "; }

check_tools() {
    [[ -x "$DMACTL" ]] || {
        echo "ERROR: dma-ctl not found (tried PATH and $DMACTL)" >&2
        echo "         make -C $QDMA_TREE" >&2
        exit 2; }
    [[ -e "/sys/bus/pci/devices/$BDF" ]] || {
        echo "ERROR: $BDF is not enumerated. Program the PDI and rescan first:" >&2
        echo "         sudo ./reprogram.sh" >&2
        exit 2; }
}

# The module is not installed, so it is loaded by path and unloaded by name.  Doing
# it here rather than leaving it to the caller is the whole point: queues cannot
# exist without it, and a half-set-up board is what this script exists to prevent.
load_driver() {
    if loaded; then log "$MODNAME already loaded"; return 0; fi
    [[ -f "$KO" ]] || {
        echo "ERROR: $KO not found — the driver has not been built." >&2
        echo "         make -C $QDMA_TREE" >&2
        exit 2; }
    log "insmod $KO"
    insmod "$KO" || { echo "ERROR: insmod failed (dmesg | tail)" >&2; exit 1; }
    # Binding happens on load, through PCI ID matching, and sysfs follows a moment
    # later.  Nothing downstream works until qdma/ is there, so wait for it rather
    # than racing it.
    local i
    for i in $(seq 1 25); do
        [[ -d "/sys/bus/pci/devices/$BDF/qdma" ]] && { log "bound to $BDF"; return 0; }
        sleep 0.2
    done
    echo "ERROR: $MODNAME loaded but never bound to $BDF." >&2
    echo "       Is this the right card, and did the PDI expose the QDMA function?" >&2
    exit 1
}

unload_driver() {
    if ! loaded; then log "$MODNAME not loaded"; return 0; fi
    log "rmmod $MODNAME"
    rmmod "$MODNAME" || {
        echo "ERROR: rmmod failed — something still has a queue open." >&2
        echo "       Close any tool holding /dev/${QDEV}-* and try again." >&2
        exit 1; }
}

status() {
    log "device : $BDF  ($QDEV)"
    log "driver : $MODNAME $(loaded && echo LOADED || echo 'not loaded')  ($KO)"
    log "dma-ctl: $DMACTL"
    if [[ -r "/sys/bus/pci/devices/$BDF/qdma/qmax" ]]; then
        log "qmax   : $(cat "/sys/bus/pci/devices/$BDF/qdma/qmax")"
    fi
    for d in "$H2C_DEV" "$C2H_DEV"; do
        if [[ -e "$d" ]]; then log "  PRESENT $d"
        else                   log "  MISSING $d"; fi
    done
    if loaded; then
        echo
        log "queues known to the driver (q list takes <start_idx> <num_Qs>):"
        "$DMACTL" "$QDEV" q list 0 "${LIST_N:-8}" 2>&1 | sed 's/^/         /' || true
    fi
}

setup() {
    need_root setup
    check_tools
    load_driver

    # qmax is the per-function queue budget and is 0 at probe, so zero queues can
    # exist until it is raised.
    local qm="/sys/bus/pci/devices/$BDF/qdma/qmax"
    if [[ -w "$qm" ]] || chmod 666 "$qm" 2>/dev/null; then
        local cur; cur="$(cat "$qm" 2>/dev/null || echo 0)"
        if [[ "$cur" -lt 2 ]]; then
            log "qmax is $cur -> setting $QMAX"
            echo "$QMAX" > "$qm"
        else
            log "qmax is $cur (leaving it alone)"
        fi
    fi

    # Idempotent: skip a direction whose node already exists rather than failing the
    # whole run, so this is safe to call from a wrapper.
    if [[ -e "$H2C_DEV" ]]; then
        log "$H2C_DEV already present — skipping h2c"
    else
        log "adding h2c queue idx $H2C_IDX (mode defaults to mm)"
        "$DMACTL" "$QDEV" q add   idx "$H2C_IDX" dir h2c
        "$DMACTL" "$QDEV" q start idx "$H2C_IDX" dir h2c idx_ringsz "$RINGSZ_IDX" pfetch_en
    fi
    if [[ -e "$C2H_DEV" ]]; then
        log "$C2H_DEV already present — skipping c2h"
    else
        log "adding c2h queue idx $C2H_IDX"
        "$DMACTL" "$QDEV" q add   idx "$C2H_IDX" dir c2h
        "$DMACTL" "$QDEV" q start idx "$C2H_IDX" dir c2h idx_ringsz "$RINGSZ_IDX" pfetch_en
    fi

    # dma-ctl's exit status cannot be trusted here (see the header), so wait for the
    # nodes themselves. They appear at `q add`, asynchronously.
    for _ in $(seq 1 20); do
        [[ -e "$H2C_DEV" && -e "$C2H_DEV" ]] && break
        sleep 0.2
    done

    local bad=0
    for d in "$H2C_DEV" "$C2H_DEV"; do
        if [[ -e "$d" ]]; then log "ready: $d"
        else                   log "ERROR: $d did not appear"; bad=1; fi
    done
    (( bad == 0 )) || { log "setup incomplete — see 'q list' below"; status; exit 1; }

    # The nodes were just created, so re-apply non-root access if this machine opted
    # into it. No-op otherwise.
    [[ -x "$SCRIPT_DIR/setup_permissions.sh" ]] && \
        "$SCRIPT_DIR/setup_permissions.sh" --reapply || true
    log "MM queue pair ready."
}

teardown() {
    need_root teardown
    if ! loaded; then log "$MODNAME is not loaded — nothing to tear down"; exit 0; fi
    check_tools
    # Stop before delete, c2h before h2c, and never let one failure abort the rest —
    # a half-torn-down queue is harder to recover than a missing one.  Then the
    # module goes too: it was insmod'd here, so it is rmmod'd here.
    "$DMACTL" "$QDEV" q stop idx "$C2H_IDX" dir c2h || true
    "$DMACTL" "$QDEV" q stop idx "$H2C_IDX" dir h2c || true
    "$DMACTL" "$QDEV" q del  idx "$C2H_IDX" dir c2h || true
    "$DMACTL" "$QDEV" q del  idx "$H2C_IDX" dir h2c || true
    unload_driver
    log "torn down"
}

case "${1:-status}" in
    status)   status ;;
    setup)    setup ;;
    teardown) teardown ;;
    -h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}" ;;
    *) echo "usage: $0 {status|setup|teardown}" >&2; exit 2 ;;
esac
