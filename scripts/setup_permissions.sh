#!/usr/bin/env bash
# setup_permissions.sh — grant a NON-ROOT user access to this one FPGA card, once.
#
#     sudo ./setup_permissions.sh              # install rules + apply now
#     ./setup_permissions.sh --check           # report only (no root needed)
#     sudo ./setup_permissions.sh --revert     # remove the rules and re-lock
#     sudo ./setup_permissions.sh --reapply    # re-apply to whatever exists now
#
# WHY
# Driving the card needs two things a normal user cannot touch: the CSR (a PCI BAR)
# and the QDMA MM queue nodes. Running everything under sudo works but forces every
# experiment, script and notebook to be root. This grants exactly those two, to one
# group, for one device.
#
# WHAT IS BEING GRANTED — read before running
#   1. /sys/bus/pci/devices/<BDF>/resource<N> — mmap of the user BAR, i.e. full
#      read/write of the FPGA's register window. It does NOT reach host memory:
#      unlike /dev/mem this is one BAR and nothing else, which is exactly why this
#      is the route to open and /dev/mem is not.
#   2. /dev/<QDEV>-MM-* — the DMA queues. Whoever has these can read and WRITE ANY
#      AXI ADDRESS ON THE CARD. Also not host memory: an MM descriptor moves data
#      between a host buffer the process already owns and a card address.
# So the blast radius is "anything to the FPGA", not "anything to the machine". On
# a shared machine, put only the people who should own the card into the group.
# Membership is NOT changed here.
#
# SIBLING SCRIPT
# /home/kjy/pim/bank_controller/bank_controller_top/SW/setup_permissions.sh is the
# same logic for the other platform. If you fix something here, fix it there too —
# these carry several non-obvious corrections that were expensive to find:
#   - udev alone is not enough: --reapply is invoked by reprogram.sh and
#     qdma_queues.sh so permissions do not depend on udev winning a race against a
#     PCIe rescan or a queue node appearing.
#   - resource<N> is a sysfs ATTRIBUTE, not a device node, so it cannot take
#     GROUP=/MODE= in a rule; it needs a RUN+= chmod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../platform/select.sh
source "$SCRIPT_DIR/../platform/select.sh"

# The rules key off the BDF, not the channel count, so they are the same for every
# image.  They used to be named "-ch1-" back when the tree was under ch1/; the
# legacy names are removed below so both do not sit in rules.d applying the same
# thing under two names.
RULES_QDMA="/etc/udev/rules.d/71-pim-emu-qdma.rules"
RULES_PCI="/etc/udev/rules.d/71-pim-emu-bar.rules"
RULES_LEGACY=(/etc/udev/rules.d/71-pim-ch1-qdma.rules
              /etc/udev/rules.d/71-pim-ch1-bar.rules)
DEV="/sys/bus/pci/devices/$BDF"
RES="$DEV/resource$CSR_BAR"

MODE="apply"
case "${1:-}" in
    --check)   MODE="check" ;;
    --revert)  MODE="revert" ;;
    --reapply) MODE="reapply" ;;
    -h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 0 ;;
    "") ;;
    *) echo "usage: $0 [--check|--revert|--reapply]" >&2; exit 2 ;;
esac

log() { echo "[perm] $*"; }

report() {
    log "device      : $BDF   (dma-ctl name $QDEV)"
    log "group       : $GROUP"
    if [[ -e "$RES" ]]; then
        log "BAR$CSR_BAR resource: $(stat -c '%A %U:%G' "$RES")  $RES"
    else
        log "BAR$CSR_BAR resource: ABSENT — the bitstream exposes no user BAR, or the"
        log "                  device is not enumerated. Run reprogram.sh first."
    fi
    local found=0
    for d in /dev/${QDEV}-MM-*; do
        [[ -e "$d" ]] || continue
        found=1
        log "qdma node   : $(stat -c '%A %U:%G' "$d")  $d"
    done
    (( found )) || log "qdma node   : none present — run qdma_queues.sh setup first"
    for f in "$RULES_QDMA" "$RULES_PCI"; do
        [[ -f "$f" ]] && log "rule        : installed  $f" || log "rule        : absent     $f"
    done
}

if [[ "$MODE" == "check" ]]; then
    report
    exit 0
fi

if [[ "$MODE" == "reapply" ]]; then
    # Silent no-op unless the rules are installed: a machine that never opted in
    # must not have its permissions changed as a side effect of reprogramming.
    [[ -f "$RULES_QDMA" || -f "$RULES_PCI" ]] || exit 0
    [[ "$(id -u)" == 0 ]] || exit 0
    getent group "$GROUP" >/dev/null || exit 0
    n=0
    if [[ -e "$RES" ]]; then
        chgrp "$GROUP" "$RES" 2>/dev/null && chmod 0660 "$RES" 2>/dev/null && n=$((n+1))
    fi
    for d in /dev/${QDEV}-MM-*; do
        [[ -e "$d" ]] || continue
        chgrp "$GROUP" "$d" 2>/dev/null && chmod 0660 "$d" 2>/dev/null && n=$((n+1))
    done
    log "re-applied $GROUP access to $n object(s)"
    exit 0
fi

[[ "$(id -u)" == 0 ]] || { echo "ERROR: must run as root (sudo $0 ...)" >&2; exit 2; }
getent group "$GROUP" >/dev/null || {
    echo "ERROR: group '$GROUP' does not exist. Create it, or set GROUP=..." >&2; exit 2; }

if [[ "$MODE" == "revert" ]]; then
    rm -f "$RULES_QDMA" "$RULES_PCI" "${RULES_LEGACY[@]}"
    udevadm control --reload-rules || true
    [[ -e "$RES" ]] && chown root:root "$RES" && chmod 600 "$RES"
    for d in /dev/${QDEV}-MM-*; do
        [[ -e "$d" ]] && chown root:root "$d" && chmod 600 "$d"
    done
    log "reverted"
    report
    exit 0
fi

# ---------------------------------------------------------------- qdma nodes ----
# The char devices are created by the driver at `q add` and destroyed on `q del`,
# driver unload, or PCIe re-enumeration — so a udev rule is the only thing that
# survives; chmod'ing by hand lasts until the next reprogram. SUBSYSTEM is the
# driver's own name (confirm with `udevadm info -q all -n /dev/<node>`), and the
# node name carries the queue INDEX, not the direction.
rm -f "${RULES_LEGACY[@]}"
cat > "$RULES_QDMA" <<EOF
# PIM emulator_top — QDMA MM queue nodes.
# Installed by SW/setup_permissions.sh. Remove with --revert.
# Grants DMA to any AXI address on the card to $GROUP.
SUBSYSTEM=="$DRIVER", KERNEL=="${QDEV}-MM-*", GROUP="$GROUP", MODE="0660"
EOF
log "installed $RULES_QDMA"

# ------------------------------------------------------------------ BAR file ----
# resource<N> is a sysfs binary attribute, not a device node, so it cannot be given
# GROUP=/MODE= the way a node can — udev has to chmod it from a RUN rule. The rule
# fires on the PCI device's add/change event, which a rescan produces, so it
# survives reprogram.sh.
cat > "$RULES_PCI" <<EOF
# PIM emulator_top — BAR$CSR_BAR of the FPGA at $BDF.
# Installed by SW/setup_permissions.sh. Remove with --revert.
# Grants mmap of the user register window to $GROUP. This is ONE BAR, not
# /dev/mem: it does not expose host memory.
ACTION=="add|change", SUBSYSTEM=="pci", KERNEL=="$BDF", \\
  RUN+="/bin/sh -c 'chgrp $GROUP /sys/bus/pci/devices/$BDF/resource$CSR_BAR 2>/dev/null; chmod 0660 /sys/bus/pci/devices/$BDF/resource$CSR_BAR 2>/dev/null'"
EOF
log "installed $RULES_PCI"

udevadm control --reload-rules
udevadm trigger --subsystem-match=pci --attr-match=vendor=0x10ee 2>/dev/null || true

# Apply to what exists right now: a trigger does not reliably re-run RUN rules, and
# the queues may already be up.
if [[ -e "$RES" ]]; then
    chgrp "$GROUP" "$RES" && chmod 0660 "$RES"
    log "applied to $RES"
else
    log "WARN: $RES absent — nothing to apply yet. It appears once a bitstream with"
    log "      a user BAR is loaded and the device is re-enumerated."
fi
for d in /dev/${QDEV}-MM-*; do
    [[ -e "$d" ]] || continue
    chgrp "$GROUP" "$d" && chmod 0660 "$d"
    log "applied to $d"
done

echo
report
echo
log "Now verify AS THE NORMAL USER (not root):"
log "    ./bar_dump --check"
log "  If it fails with EACCES or EINVAL the kernel is refusing the mmap regardless"
log "  of the file mode — that is a kernel policy question, not a bug in this"
log "  script. Report it and keep using sudo."
