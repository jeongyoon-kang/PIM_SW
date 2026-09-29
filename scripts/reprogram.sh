#!/usr/bin/env bash
# usage: sudo reprogram.sh --hw-dir DIR
#        sudo reprogram.sh --pdi FILE
#
#   --hw-dir DIR   program the one *.pdi directly under DIR (error if none or
#                  more than one); the *.ltx of the same name is loaded with it
#   --pdi FILE     program FILE; FILE with .pdi replaced by .ltx is loaded if present
#   -h, --help     show this help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the settings this script uses: BDF (the card's PCIe address), DRIVER
# (unbound before programming), CSR_BAR (the BAR checked after rescan), and
# VIVADO_SETTINGS / PROGRAM_TCL (used to run Vivado and program the PDI).
# shellcheck source=../platform/common.conf
source "$SCRIPT_DIR/../platform/common.conf"

LOGDIR="$SCRIPT_DIR/logs"
RETRIES=2
DEV="/sys/bus/pci/devices/$BDF"

log() { echo "[reprogram] $*"; }
need() { [[ -n "${2:-}" ]] || { echo "ERROR: $1 needs a value (try --help)" >&2; exit 2; }; }

# The image comes from --hw-dir or --pdi; both start empty.
HW_DIR="" PDI=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hw-dir) need "$@"; HW_DIR="$2"; shift 2 ;;
        --pdi)    need "$@"; PDI="$2";    shift 2 ;;
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done
if [[ -n "$HW_DIR" && -n "$PDI" ]]; then
    echo "ERROR: --hw-dir and --pdi are exclusive (try --help)" >&2; exit 2
elif [[ -z "$HW_DIR" && -z "$PDI" ]]; then
    echo "ERROR: name the image: --hw-dir DIR or --pdi FILE (try --help)" >&2; exit 2
fi

# ---- find the PDI (exactly one *.pdi directly under --hw-dir) and the .ltx of
#      the same name next to it.
if [[ -n "$HW_DIR" ]]; then
    [[ -d "$HW_DIR" ]] || { log "ERROR: not a directory: $HW_DIR"; exit 2; }
    mapfile -t _pdis < <(find "$HW_DIR" -maxdepth 1 -name '*.pdi' 2>/dev/null | sort)
    if   (( ${#_pdis[@]} == 1 )); then PDI="${_pdis[0]}"
    elif (( ${#_pdis[@]} == 0 )); then log "ERROR: no *.pdi in $HW_DIR"; exit 2
    else
        log "ERROR: ${#_pdis[@]} PDIs in $HW_DIR — name one with --pdi:"
        printf '         %s\n' "${_pdis[@]}" >&2
        exit 2
    fi
fi
[[ -f "$PDI" ]] || { log "ERROR: PDI not found: $PDI"; exit 2; }
LTX="${PDI%.pdi}.ltx"
[[ -f "$LTX" ]] || LTX=""

# Absolute paths for Vivado, which runs from $LOGDIR.
PDI="$(readlink -f "$PDI")"
[[ -z "$LTX" ]] || LTX="$(readlink -f "$LTX")"
log "pdi : $PDI"
log "ltx : ${LTX:-<none>}"
[[ -n "$LTX" ]] || log "WARN: no LTX alongside the PDI — ILA debug will be unavailable"

dev_state() {
    if [[ -e "$DEV/resource" ]]; then
        log "  state: device PRESENT  driver=$(basename "$(readlink -f "$DEV/driver" 2>/dev/null)" 2>/dev/null || echo none)"
    else
        log "  state: device ABSENT (not enumerated)"
    fi
}

# Print BAR$CSR_BAR's base address from the resource file; fail if it is missing
# or 0 (an unallocated BAR is listed as 0).
read_bar() {
    local b
    b="$(awk -v n="$((CSR_BAR + 1))" 'NR==n{print $1}' "$DEV/resource" 2>/dev/null)" || return 1
    [[ -n "$b" && "$b" != 0x0000000000000000 && "$b" != 0x0 ]] || return 1
    echo "$b"
}

[[ "$(id -u)" == 0 ]] || { echo "ERROR: must run as root (sudo $0 ...)" >&2; exit 2; }
RUN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"

# Check program.tcl and Vivado first; the device is detached only after these pass.
[[ -f "$PROGRAM_TCL" ]] || { log "ERROR: program.tcl not found: $PROGRAM_TCL"; exit 2; }
if [[ ! -f "$VIVADO_SETTINGS" ]]; then
    log "ERROR: Vivado settings not found: $VIVADO_SETTINGS"
    log "       set VIVADO_SETTINGS=/path/to/settings64.sh"
    exit 2
fi
if ! sudo -u "$RUN_USER" -H bash -lc \
      "source '$VIVADO_SETTINGS' >/dev/null 2>&1 && command -v vivado >/dev/null"; then
    log "ERROR: vivado not runnable after sourcing $VIVADO_SETTINGS"
    exit 2
fi

# On a failure after detach, rescan to put the device back on PCIe.
DETACHED=0
restore_on_fail() {
    local rc=$?
    if (( rc != 0 )) && (( DETACHED == 1 )); then
        log "failure -> restoring PCIe device (rescan)"
        echo 1 > /sys/bus/pci/rescan 2>/dev/null || true
        sleep 2
        if [[ -e "$DEV/resource" ]]; then log "device restored at $BDF"
        else log "WARNING: device did NOT come back — a host reboot may be needed"; fi
    fi
}
trap restore_on_fail EXIT

# Unbind the driver and remove the device from PCIe before programming;
# rescan() brings it back with the new image.
detach() {
    if [[ -e "$DEV/driver" ]]; then
        log "unbinding $DRIVER from $BDF"
        echo "$BDF" > "/sys/bus/pci/drivers/$DRIVER/unbind" 2>/dev/null || true
    fi
    if [[ -e "$DEV" ]]; then
        log "removing $BDF from PCIe"
        echo 1 > "$DEV/remove"
    fi
    DETACHED=1
    sleep 1
    dev_state
}

# Rescan PCIe and wait up to 30 s for the device to appear.
rescan() {
    log "rescanning PCIe"
    echo 1 > /sys/bus/pci/rescan
    for _ in $(seq 1 30); do
        [[ -e "$DEV/resource" ]] && { sleep 1; dev_state; return 0; }
        sleep 1
    done
    dev_state
    return 1
}

# Secondary bus reset on the upstream bridge; used when a retry is needed.
bus_reset() {
    local bridge; bridge="$(basename "$(readlink -f "$DEV/..")" 2>/dev/null || true)"
    [[ -n "$bridge" && "$bridge" != "pci0000:00" ]] || return 1
    local short="${bridge#0000:}"
    log "secondary bus reset on bridge $short"
    setpci -s "$short" BRIDGE_CONTROL=40:40 || return 1
    sleep 1
    setpci -s "$short" BRIDGE_CONTROL=00:40 || return 1
    sleep 2
}

# ----------------------------------------------------------------- run ----
detach

mkdir -p "$LOGDIR"
VLOG="$LOGDIR/reprogram.vivado.log"
log "programming PDI as user '$RUN_USER'  (vivado log -> $VLOG)"
# Run Vivado from LOGDIR so its vivado.jou / vivado.log land there too.
sudo -u "$RUN_USER" -H bash -lc \
    "source '$VIVADO_SETTINGS' >/dev/null 2>&1; cd '$LOGDIR' && \
     vivado -mode batch -source '$PROGRAM_TCL' -tclargs '$PDI' '$LTX'" \
    > "$VLOG" 2>&1 \
    || { log "ERROR: Vivado programming failed — see $VLOG"; tail -5 "$VLOG" | sed 's/^/           /'; exit 1; }
log "  $(grep -m1 'program_hw_devices: Time' "$VLOG" 2>/dev/null || echo 'programmed')"
log "PDI programmed"

for attempt in $(seq 0 "$RETRIES"); do
    if (( attempt > 0 )); then
        log "retry $attempt/$RETRIES"
        detach; bus_reset || true
    fi
    rescan || { log "device did not reappear after rescan"; continue; }
    BAR="$(read_bar || true)"
    if [[ -z "${BAR:-}" ]]; then
        log "BAR$CSR_BAR is not allocated — the programmed bitstream exposes no user BAR"
        continue
    fi
    log "device back at $BDF, BAR$CSR_BAR=$BAR"

    # Re-apply non-root access to the BAR file the rescan created.
    [[ -x "$SCRIPT_DIR/setup_permissions.sh" ]] && \
        "$SCRIPT_DIR/setup_permissions.sh" --reapply || true
    exit 0
done

log "ERROR: the device did not come back with BAR$CSR_BAR allocated after"
log "       $((RETRIES+1)) attempts.  A host reboot is the remaining fallback."
exit 1
