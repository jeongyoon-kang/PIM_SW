#!/usr/bin/env bash
# reprogram.sh — load this platform's PDI without rebooting the host.
#
#     sudo ./reprogram.sh --ch 2                # load the 2-channel image
#     sudo ./reprogram.sh --ch 1 --hw-dir DIR   # ...from a different drop of it
#     sudo ./reprogram.sh --ch 4 --pdi FILE     # ...one exact file
#     ./reprogram.sh --ch 2 --select-only       # record the choice, touch nothing
#     sudo ./reprogram.sh --sanity-only         # check the BAR + run SANITY_CMD
#
# --ch IS THE CHANNEL COUNT OF THE IMAGE, AND THIS SCRIPT OWNS IT
# How many channels exist is a property of the bitstream, so the step that loads the
# bitstream is the one that knows.  --ch N picks platform/chN.conf, programs the
# *.pdi that conf's HW_DIR points at, and then WRITES THE CHOICE DOWN by pointing
# platform/active at that conf.  Everything downstream reads it there: setup.sh does
# not ask which platform to build for, it looks.
#
# That is the whole reason the flag lives here.  A channel count typed into a build
# script is a claim about hardware nobody checked; typed here it is the same act
# that put the hardware in place.
#
#     sudo ./reprogram.sh --ch 2      # board becomes 2-channel, and says so
#     ./setup.sh                      # builds for whatever the board became
#
# --hw-dir and --pdi override WHERE the image comes from, not WHICH platform it is:
# a new revision of the same channel count, or a drop kept outside the tree.  The
# directory must hold exactly one *.pdi; its *.ltx is picked up alongside.
#
# THIS SCRIPT DOES NOT REBUILD.  After it runs, the compiled-in channel count is
# whatever it was before, so the tools will refuse until setup.sh has run.  That is
# the intended order and the refusal says so.
#
# WHY THE PCIe DANCE
# The board is a PCIe card. Programming a PDI over JTAG invalidates PCIe
# enumeration, so the host must re-enumerate or every register read comes back
# 0xffffffff — which a naive tool reports as "a register full of ones" rather than
# as a dead link, and a measurement then runs for hours on garbage. Sequence:
# unbind + remove BEFORE programming (a live driver over a reconfiguring FPGA risks
# AER storms), rescan after, then PROVE the device is healthy before trusting it.
#
# IT DOES NOT RUN THE SANITY PROBE
# Programming and judging the result are separate jobs, and this script only does
# the first.  What it still proves is the part it is uniquely able to: that the
# device re-enumerated and the kernel allocated BAR$CSR_BAR — an unallocated BAR
# reads back as zeros rather than as a missing line, so a caller that skipped this
# would go on to mmap physical address 0.
#
# Everything past that belongs to the probe, and running it from here had a side
# effect that does not belong in a programming step: emu_sanity WRITES the channel
# address map and the DRAM timing registers.  Doing that silently at the end of a
# reprogram means the board's configuration changes as a byproduct of loading a
# bitstream.  Run it yourself, or let scripts/setup.sh do it — it does, right after
# it builds.
#
# --sanity-only still runs SANITY_CMD, because that is the whole of what that mode
# is for.
#
# SIBLING SCRIPT
# bank_controller_top/SW/reprogram_bank.sh is the same logic for the other
# platform. Fixes belong in both. The non-obvious ones already carried here:
#   - An UNALLOCATED BAR reads back from the `resource` file as
#     0x0000000000000000, not as a missing line, so a plain -n test passes and the
#     caller goes on to mmap physical address 0. read_bar() rejects it.
#   - Queues do NOT survive re-enumeration: qdma_queues.sh setup must be re-run
#     after every program. This script says so at the end rather than leaving it to
#     be rediscovered.
#   - Permissions are re-applied here, not left to udev winning a race with rescan.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --ch has to be known BEFORE the confs are sourced, because it chooses which conf.
# select.sh already honours PIM_PLATFORM as an override, so this feeds it there
# rather than growing a second selection path.
WANT_CH=""
_a=("$@")
for ((_i = 0; _i < ${#_a[@]}; _i++)); do
    if [[ "${_a[_i]}" == "--ch" ]]; then
        WANT_CH="${_a[_i+1]:-}"
        case "$WANT_CH" in
            [0-9]*) ;;
            *) echo "ERROR: --ch takes a channel count (1, 2, 4 — whatever platform/ch*.conf exists)" >&2
               exit 2 ;;
        esac
        export PIM_PLATFORM="ch${WANT_CH}"
    fi
done

# shellcheck source=../platform/select.sh
source "$SCRIPT_DIR/../platform/select.sh"

LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
RETRIES="${RETRIES:-2}"
DEV="/sys/bus/pci/devices/$BDF"

SANITY_ONLY=0
SELECT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ch)          shift 2 ;;      # already consumed above
        --select-only) SELECT_ONLY=1; shift ;;
        --pdi)         PDI="$2"; shift 2 ;;
        --hw-dir)      HW_DIR="$2"; shift 2 ;;
        --ltx)         LTX="$2"; shift 2 ;;
        --tcl)         PROGRAM_TCL="$2"; shift 2 ;;
        --bdf)         BDF="$2"; DEV="/sys/bus/pci/devices/$2"; shift 2 ;;
        --sanity-only) SANITY_ONLY=1; shift ;;
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

log() { echo "[reprogram] $*"; }

# --ch IS REQUIRED, because this is the step that decides it.  Falling back to
# whatever platform/active already said would let a reprogram quietly load the image
# for the platform that was selected LAST TIME — the one case where the record and
# the board can disagree without anyone typing anything wrong.
#
# --sanity-only is exempt: it programs nothing, so it has nothing to decide.
if (( ! SANITY_ONLY )) && [[ -z "$WANT_CH" ]]; then
    _av="$(cd "$SCRIPT_DIR/../platform" && ls ch*.conf 2>/dev/null | sed 's/\.conf$//;s/^ch//' | tr '\n' ' ')"
    echo "ERROR: --ch is required." >&2
    echo "       The board's channel count is what this step decides, and the tools" >&2
    echo "       downstream read it back from what this writes." >&2
    echo "       available: ${_av:-none}" >&2
    echo "         sudo $0 --ch 2" >&2
    exit 2
fi

# ---- resolve the images. Empty PDI means "the one in HW_DIR", which is the normal
#      case for a drop that contains exactly one bitstream; more than one is
#      ambiguous and must be named rather than guessed at.
if [[ -z "${PDI:-}" ]]; then
    if [[ ! -d "${HW_DIR:-}" ]]; then
        log "ERROR: HW_DIR is not a directory: ${HW_DIR:-<unset>}"
        log "       It comes from platform/active ($PIM_PLATFORM) unless --hw-dir overrode it."
        exit 2
    fi
    mapfile -t _pdis < <(find "$HW_DIR" -maxdepth 1 -name '*.pdi' 2>/dev/null | sort)
    if   (( ${#_pdis[@]} == 1 )); then PDI="${_pdis[0]}"
    elif (( ${#_pdis[@]} == 0 )); then
        log "ERROR: no *.pdi in $HW_DIR"
        log "       The HW drop has not landed yet, or PDI= must be set."
        exit 2
    else
        log "ERROR: ${#_pdis[@]} PDIs in $HW_DIR — name one with --pdi:"
        printf '         %s\n' "${_pdis[@]}" >&2
        exit 2
    fi
fi
if [[ -z "${LTX:-}" ]]; then
    _cand="${PDI%.pdi}.ltx"
    [[ -f "$_cand" ]] && LTX="$_cand" || LTX=""
fi

dev_state() {
    if [[ -e "$DEV/resource" ]]; then
        log "  state: device PRESENT  driver=$(basename "$(readlink -f "$DEV/driver" 2>/dev/null)" 2>/dev/null || echo none)"
    else
        log "  state: device ABSENT (not enumerated)"
    fi
}

# The CSR BAR's base, from line CSR_BAR+1 of the resource file.
#
# An UNALLOCATED BAR reads back as all zeros, not as a missing line, so a plain -n
# test on the result passes and the caller mmaps physical address 0. That is the
# state whenever the loaded bitstream does not expose the user BAR. Reject it.
read_bar() {
    local b
    b="$(awk -v n="$((CSR_BAR + 1))" 'NR==n{print $1}' "$DEV/resource" 2>/dev/null)" || return 1
    [[ -n "$b" && "$b" != 0x0000000000000000 && "$b" != 0x0 ]] || return 1
    echo "$b"
}

sanity() {
    local bar="$1" rc=0
    log "sanity: $SANITY_CMD"
    # shellcheck disable=SC2086
    eval $SANITY_CMD 2>&1 | sed 's/^/           /' || rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        log "sanity FAILED (exit $rc)"
        return 1
    fi
    log "sanity OK"
    return 0
}

# WRITE THE CHOICE DOWN.  Everything downstream reads platform/active to learn what
# the board is; this is where it gets set, because this is where the board became
# that.  Doing it before programming would leave a lie behind on a failed run, so it
# happens only after the PDI is in.
record_choice() {
    [[ -n "$WANT_CH" ]] || return 0
    local want="ch${WANT_CH}.conf"
    local dir; dir="$(cd "$SCRIPT_DIR/../platform" && pwd)"
    [[ -r "$dir/$want" ]] || { log "ERROR: $dir/$want is not readable"; return 1; }
    local now=""
    [[ -e "$dir/active" ]] && now="$(basename "$(readlink -f "$dir/active")")"
    if [[ "$now" == "$want" ]]; then
        log "platform/active already $want"
    else
        ln -sfn "$want" "$dir/active"
        log "platform/active ${now:-none} -> $want"
    fi
    log ""
    log "The build is still for whatever was selected before, so the tools will"
    log "refuse until:   $SCRIPT_DIR/setup.sh"
}

# ---------------------------------------------------- select-only path ----
# For a machine with the tree but no board, or to fix up the record by hand.
if (( SELECT_ONLY )); then
    [[ -n "$WANT_CH" ]] || { log "ERROR: --select-only needs --ch N"; exit 2; }
    record_choice || exit 1
    exit 0
fi

# ---------------------------------------------------- sanity-only path ----
if (( SANITY_ONLY )); then
    dev_state
    BAR="$(read_bar || true)"
    if [[ -z "${BAR:-}" ]]; then
        log "ERROR: BAR$CSR_BAR is not allocated at $BDF."
        log "       The loaded bitstream exposes no user BAR, so there is nothing to"
        log "       talk to. Program the PDI first:  sudo $0"
        exit 1
    fi
    log "BAR$CSR_BAR=$BAR"
    sanity "$BAR" || exit 1
    echo "BAR=$BAR"
    exit 0
fi

[[ "$(id -u)" == 0 ]] || { echo "ERROR: must run as root (sudo $0 ...)" >&2; exit 2; }
RUN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"

# Preflight BEFORE touching PCIe: if we cannot program, we must not detach the
# device — that would leave the board unusable for no reason.
# ABSOLUTE BEFORE VIVADO SEES IT.  Vivado is launched with `cd "$LOGDIR"` so its
# own journal lands there, which means a relative --pdi resolves against a DIFFERENT
# directory than the check below — the shell says the file is there and the Tcl says
# it is not, naming a path that does exist from where you typed it.  Absolutising
# here makes the two agree whatever the caller passed.
[[ -n "${PDI:-}" ]] && PDI="$(readlink -f "$PDI" 2>/dev/null || echo "$PDI")"
[[ -n "${LTX:-}" ]] && LTX="$(readlink -f "$LTX" 2>/dev/null || echo "$LTX")"
[[ -f "$PDI" ]] || { log "ERROR: PDI not found: $PDI"; exit 2; }
[[ -n "$LTX" ]] || log "WARN: no LTX alongside the PDI — ILA debug will be unavailable"
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

# If anything fails after we detach, put the device back rather than leaving it
# removed from PCIe — that state looks like dead hardware.
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

# Fallback when the link does not retrain: secondary bus reset on the bridge.
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
log "=== emulator_top ($PIM_PLATFORM) ==="
[[ -n "$WANT_CH" ]] || log "  NOTE: no --ch, so this reprograms whatever platform/active already says"
log "  dir : $HW_DIR"
log "  pdi : $PDI"
log "  ltx : ${LTX:-<none>}"
detach

mkdir -p "$LOGDIR"
VLOG="$LOGDIR/reprogram.vivado.log"
log "programming PDI as user '$RUN_USER'  (vivado log -> $VLOG)"
# cd into LOGDIR so Vivado's own vivado.jou / vivado.log land there too.
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

    # The BAR file is brand new after a rescan; re-apply non-root access if this
    # machine opted into it. No-op otherwise.
    [[ -x "$SCRIPT_DIR/setup_permissions.sh" ]] && \
        "$SCRIPT_DIR/setup_permissions.sh" --reapply || true

    record_choice || exit 1
    echo "BAR=$BAR"
    log ""
    log "NEXT — programming is all this does.  Two things did NOT survive it:"
    log "  the QDMA queues (re-enumeration took them):"
    log "    sudo $SCRIPT_DIR/qdma_queues.sh setup"
    log "  the compiled-in channel count and the channel address map:"
    log "    $SCRIPT_DIR/setup.sh                # rebuilds for what is now selected,"
    log "                                        # and applies the map to the board"
    exit 0
done

log "ERROR: the device did not come back with BAR$CSR_BAR allocated after "
log "       $((RETRIES+1)) attempts.  A host reboot is the remaining fallback."
exit 1
