#!/usr/bin/env bash
# setup.sh — choose which PIM emulator image the tools talk to.
#
#     ./setup.sh                             # build for whatever board is selected
#     ./setup.sh --map 2                     # ...with the interleaved channel map
#     ./setup.sh --status                    # what is selected and what was built
#     sudo ./setup.sh --all                  # build + qdma queues + permissions
#
# IT DOES NOT CHOOSE THE CHANNEL COUNT.  reprogram.sh does.
# How many channels exist is a property of the bitstream on the board, so the step
# that loads the bitstream is the one that knows — `reprogram.sh --ch N` programs
# the image and points platform/active at that platform's conf.  This script reads
# it.  A channel count typed into a build script would be a claim about hardware
# nobody checked, and the failure it produces is the worst kind this hardware
# offers: a 1-channel runtime on a 2-channel board writes half the operands, reads
# half the results, and returns plausible numbers.
#
# WHAT IT DOES CHOOSE is the channel address map (--map), because that one really is
# a software decision: the bitstream supports both and a register selects between
# them.  On a 1-channel image it makes no difference and says so.
#
# THE CHANNEL ADDRESS MAP  (--map)
# Which field of an address names the channel.  It is a register on the board, so
# switching needs a rebuild but NOT a new bitstream — seconds instead of minutes.
#
#   --map 1   ChRoBaCo   | CH | ROW | BA | CO | byte |
#             Channel is the TOP field, so it changes once every 4 GiB and a linear
#             transfer stays inside one channel.  This is the register's reset state
#             and what the address map was before the mode existed.
#
#   --map 2   RoChBaCo   | ROW | CH | BA | CO | byte |
#             Channel drops to just under ROW, so it changes every 32 KiB and a
#             linear transfer spreads across every channel.  What is left below the
#             channel field — BA + CO + byte — is exactly one RoBaCo row, which is
#             what one all-bank MAC consumes: an operand never straddles a channel.
#
# Meaningless on a 1-channel image.  --map WRITES the choice into that platform's
# conf, so it survives every later rebuild; without it the conf's ADDR_MAP stands.
#
# IT IS COMPILED IN, like the channel count.  Nothing above hwdef takes a map
# argument or reads the register per access: pim_decode() answers for the map this
# build was made for, and pim_addr_map_check() refuses at open if the board disagrees.
# Setting the board to match is hwdef/test/emu_sanity, which reprogram.sh already runs.
#
# CHANGING IT INVALIDATES EVERYTHING RESIDENT.  No byte moves; the meaning of the
# address that reaches it changes.  Place data after choosing, never before.
#
# HOW THE VALUES GET IN
# `hwdef/gen_config.sh --defs` turns the selected conf into a -D list and this
# script passes it to all four makes as PIM_DEFS.  hwdef/pim_config.h holds a
# DEFAULT for every value (ch4) so a bare `make` still compiles — but such a build
# leaves PIM_CONFIG_FROM_CONF at 0 and pim_platform_check() then REFUSES to run it.
# The defaults are there to keep the tree buildable, not to be used.
#
# WHY IT CLEANS FIRST
# make cannot see a -D change.  Nothing in the tree is newer after a platform
# switch, so an incremental build would relink the previous platform's objects and
# say nothing.  The clean is the only thing standing between you and that, which is
# why a failed clean aborts instead of continuing.  Same DEFS as last time -> no
# clean, no rebuild.
#
# --no-build skips both if you want to build by hand.
#
# AND IT PUTS THE BOARD WHERE THE BUILD EXPECTS IT.  The channel map lives in a
# register, so after a successful build this runs hwdef/test/emu_sanity — the one
# thing that writes it — and you are done in one command.  --no-apply skips that.
# A card that is absent or unreadable is reported, not treated as a build failure.
#
# WHAT THIS DOES NOT DO
# It does not program the BITSTREAM.  reprogram.sh does that, and it reads HW_DIR
# from the conf this script selects — so the order is: setup.sh, then reprogram.sh.
# (reprogram.sh runs emu_sanity too, so the map is applied there as well.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(cd "$SCRIPT_DIR/../platform" && pwd)"
EMU_TOP="$(cd "$SCRIPT_DIR/.." && pwd)"   # the confs reference this; see platform/select.sh

WANT=""
MAP=""
DO_QUEUES=0
DO_PERMS=0
DO_BUILD=1
DO_APPLY=1
STATUS_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --map)      MAP="${2:-}"; shift 2 ;;
        --queues)   DO_QUEUES=1; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        --no-apply) DO_APPLY=0; shift ;;
        --perms)    DO_PERMS=1; shift ;;
        --all)      DO_QUEUES=1; DO_PERMS=1; shift ;;
        --status)   STATUS_ONLY=1; shift ;;
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "usage: $0 [--map 1|2] [--no-build] [--no-apply] [--queues] [--perms] [--all] [--status]" >&2
           echo "       the channel count comes from platform/active — set it with" >&2
           echo "         sudo ./reprogram.sh --ch N" >&2
           echo "       --map 1 = ChRoBaCo (bypass), 2 = RoChBaCo (interleave); $0 -h explains" >&2
           exit 2 ;;
    esac
done

log() { echo "[setup] $*"; }

avail() { (cd "$PLATFORM_DIR" && ls *.conf 2>/dev/null | grep -v '^common\.conf$' | sed 's/\.conf$//'); }

status() {
    if [[ -e "$PLATFORM_DIR/active" ]]; then
        local t; t="$(basename "$(readlink -f "$PLATFORM_DIR/active")" .conf)"
        log "active   : $t"
        # Read the three values that decide whether a wrong choice is catastrophic.
        ( set +u; source "$PLATFORM_DIR/active"
          log "  NCH=$NCH  NBANK=$NBANK  bank_stride=$HBM_BANK_STRIDE  window=$BANK_WINDOW"
          log "  UPLOAD_PATH=$UPLOAD_PATH  SCHEDULE=$SCHEDULE  FEATURE_T_LATCH=$FEATURE_T_LATCH" )
        # The map that was BUILT, which is the stamp's business — the conf only says
        # what the next build would use.
        if [[ -r "$PLATFORM_DIR/.built" ]]; then
            local m; m="$(sed -n '2p' "$PLATFORM_DIR/.built" | grep -o 'PIM_ADDR_MAP=[01]' | cut -d= -f2)"
            case "${m:-}" in
                0) log "  chanmap=1 ChRoBaCo (bypass, channel every 4 GiB)" ;;
                1) log "  chanmap=2 RoChBaCo (interleave, channel every 32 KiB)" ;;
            esac
        fi
        # What the BINARIES were built for, which is a different question now: the
        # values live in -D, so nothing in the tree records them.  `cat pim_config.h`
        # answers "ch4" forever regardless of what was built.
        local b="UNKNOWN"
        [[ -r "$PLATFORM_DIR/.built" ]] && b="$(sed -n '1p' "$PLATFORM_DIR/.built")"
        if [[ "$b" == "$t" ]]; then
            log "built    : $b"
        else
            log "built    : $b   <-- DOES NOT MATCH.  Every tool will refuse to start."
            log "           $0 --platform $t"
        fi
    else
        log "active   : NONE — run  $0 --platform <name>"
    fi
    log "available: $(avail | tr '\n' ' ')"
}

# WHAT TO BUILD FOR COMES FROM THE BOARD'S RECORD, never from an argument.
if [[ -e "$PLATFORM_DIR/active" ]]; then
    WANT="$(basename "$(readlink -f "$PLATFORM_DIR/active")" .conf)"
fi

if (( STATUS_ONLY )); then status; exit 0; fi

if [[ -z "$WANT" ]]; then
    echo "ERROR: no platform selected — nothing says what is on the board." >&2
    echo "       Program it and record what it is:" >&2
    echo "         sudo $SCRIPT_DIR/reprogram.sh --ch 2      (or 1 / 4)" >&2
    echo "       Or record it without touching the board:" >&2
    echo "         $SCRIPT_DIR/reprogram.sh --ch 2 --select-only" >&2
    exit 2
fi

if [[ -n "$WANT" ]]; then
    CONF="$PLATFORM_DIR/${WANT}.conf"
    if [[ ! -r "$CONF" ]]; then
        echo "ERROR: platform/active points at '$WANT' but $CONF is not readable" >&2
        exit 2
    fi
    log "board is $WANT (from platform/active)"

    # AFTER the symlink moved: gen_config.sh reads platform/active, not its argument.
    # Built ONCE and reused for all four makes, so a symlink changed mid-build cannot
    # produce a half-and-half tree.
    # --map EDITS THE CONF.  The map is a choice like the platform itself, not a
    # switch for one invocation: a build a week later, or after touching a source
    # file, has to come out the same.  Writing it here is the only way the conf stays
    # the single place the answer lives — an override that lived only in this run
    # would silently revert the next time anyone rebuilt without repeating it.
    if [[ -n "$MAP" ]]; then
        case "$MAP" in 1|2) ;; *) echo "ERROR: --map must be 1 (ChRoBaCo) or 2 (RoChBaCo)" >&2; exit 2 ;; esac
        grep -q '^ADDR_MAP=' "$CONF" || {
            echo "ERROR: $CONF has no ADDR_MAP line to set" >&2; exit 2; }
        WAS="$(sed -n 's/^ADDR_MAP=\([0-9]*\).*/\1/p' "$CONF")"
        if [[ "$WAS" != "$MAP" ]]; then
            sed -i "s/^ADDR_MAP=.*/ADDR_MAP=$MAP/" "$CONF"
            log "${WANT}.conf: ADDR_MAP $WAS -> $MAP"
        fi
    fi
    DEFS="$("$EMU_TOP/hwdef/gen_config.sh" --defs)" || exit 1
    [[ -n "$DEFS" ]] || { echo "[setup] gen_config.sh --defs produced nothing" >&2; exit 1; }

    # THE TOPOLOGY IS COMPILED IN, so selecting is only half the job.  Skipping the
    # build leaves binaries that refuse to start (pim_platform_check) rather than run
    # against the wrong channel count — but refusing is not the goal, working is.
    if (( DO_BUILD )); then
        # Compare the whole -D string, not the platform name: editing a value inside
        # ch2.conf and re-selecting ch2 has to rebuild too, and the name alone would
        # call that a no-op.
        STAMP="$PLATFORM_DIR/.built"
        PREV=""
        [[ -r "$STAMP" ]] && PREV="$(sed -n '2p' "$STAMP")"

        # ...and whether anything was built behind this script's back.  The stamp is
        # written AFTER the makes, so every artifact should be OLDER than it.  One
        # that is newer came from a bare `make`, which compiled the ch4 defaults —
        # and since PIM_DEFS is unchanged, make would consider those objects current
        # and leave them in place.  The tree would keep refusing to start with no
        # obvious way out.
        STRAY=""
        if [[ -r "$STAMP" ]]; then
            STRAY="$(find "$EMU_TOP/hwdef" "$EMU_TOP/runtime" -type f -newer "$STAMP" \
                        ! -name '*.c' ! -name '*.h' ! -name 'Makefile' ! -name '*.sh' \
                        ! -name '*.md' -print -quit 2>/dev/null)"
        fi

        if [[ "$PREV" != "$DEFS" || -n "$STRAY" ]]; then
            if [[ -n "$STRAY" ]]; then
                log "built outside this script (${STRAY#$EMU_TOP/}) -> cleaning first"
            else
                log "constants changed -> cleaning first (make cannot see a -D change)"
            fi
            make -s -C "$EMU_TOP/runtime" clean || { echo "[setup] runtime clean FAILED" >&2; exit 1; }
            make -s -C "$EMU_TOP/hwdef"   clean || { echo "[setup] hwdef clean FAILED" >&2; exit 1; }
            # A clean that exits 0 without removing anything (permissions) would let
            # the previous platform's objects link into this one.
            for f in runtime/libpim.so runtime/libpim.a hwdef/libhwdef.a; do
                [[ -e "$EMU_TOP/$f" ]] && { echo "[setup] clean incomplete: $f survived" >&2; exit 1; }
            done
            # And the one thing clean must NOT have taken: pim_config.h is source now.
            [[ -r "$EMU_TOP/hwdef/pim_config.h" ]] || {
                echo "[setup] hwdef/pim_config.h is gone — clean removed source" >&2; exit 1; }
        fi

        log "rebuilding for $WANT ..."
        make -s -C "$EMU_TOP/hwdef"        "PIM_DEFS=$DEFS" || { echo "[setup] hwdef build FAILED" >&2; exit 1; }
        make -s -C "$EMU_TOP/hwdef" test   "PIM_DEFS=$DEFS" || { echo "[setup] probe build FAILED" >&2; exit 1; }
        make -s -C "$EMU_TOP/runtime"      "PIM_DEFS=$DEFS" || { echo "[setup] runtime build FAILED" >&2; exit 1; }
        make -s -C "$EMU_TOP/runtime" test "PIM_DEFS=$DEFS" || { echo "[setup] runtime test build FAILED" >&2; exit 1; }
        # Only after all four succeeded: a stamp written earlier would let a failed
        # build look up to date and skip the clean next time.
        { printf '%s\n' "$WANT"; printf '%s\n' "$DEFS"; } > "$STAMP"
        log "built"

        # AND PUT THE BOARD WHERE THE BUILD EXPECTS IT.  The channel map is a
        # register that resets with the bitstream, so choosing it and applying it are
        # two different things — and leaving the second to the caller means every
        # tool refuses until they remember.  emu_sanity is what writes it (the only
        # thing that does), and it is cheap.
        #
        # A BOARD PROBLEM IS NOT A BUILD PROBLEM.  If the card is not enumerated, or
        # permissions were never set, the build still succeeded and this says so
        # rather than failing.  Nothing is lost by not applying: every tool checks
        # the register at startup and refuses with the command to run.
        if (( DO_APPLY )); then
            if "$EMU_TOP/hwdef/test/emu_sanity" --quiet >/dev/null 2>&1; then
                log "board configured to match (emu_sanity)"
            else
                log "NOTE: could not configure the board — the BUILD is fine."
                log "      Run it yourself when the card is up:"
                log "        $EMU_TOP/hwdef/test/emu_sanity"
                log "      Until then every tool will refuse, saying so."
            fi
        fi
    else
        log "NOT rebuilt (--no-build) — binaries still compiled for whatever came before"
    fi

    # The image on the board is NOT checked against this — the hardware exposes no
    # way to read its channel count.  Choosing the platform and programming the PDI
    # are one human step (reprogram.sh takes HW_DIR from this same conf), which is
    # what keeps them together.  If a revision ever adds that register, the
    # cross-check goes here.
    ( set +u; source "$PLATFORM_DIR/$WANT.conf"
      if [[ ! -d "${HW_DIR:-}" ]]; then
          log "NOTE: HW_DIR does not exist yet: $HW_DIR"
          log "      selection is still valid; reprogram.sh will need it."
      fi ) || true
fi

if (( DO_QUEUES )); then
    [[ "$(id -u)" == 0 ]] || { echo "ERROR: --queues needs root (sudo $0 ...)" >&2; exit 2; }
    log "qdma queues ..."
    "$SCRIPT_DIR/qdma_queues.sh" setup
fi

if (( DO_PERMS )); then
    [[ "$(id -u)" == 0 ]] || { echo "ERROR: --perms needs root (sudo $0 ...)" >&2; exit 2; }
    log "permissions ..."
    "$SCRIPT_DIR/setup_permissions.sh"
fi

echo
status
