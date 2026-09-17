#!/usr/bin/env bash
# gen_config.sh — turn the ACTIVE platform conf into compiler arguments.
#
#     ./gen_config.sh --defs      # -D list for the build.  THIS is what setup.sh uses.
#     ./gen_config.sh --header    # the same values as a pim_config.h body, by hand
#
# WHY THIS FILE STILL EXISTS
# The conf is shell and the shell scripts source it directly.  Parsing it again in C
# meant a hand-written KEY=VALUE parser, dladdr() to find the file from inside a .so,
# and a path walk — about 200 lines whose only job was to turn text into numbers that
# never change while a board is programmed.  So the numbers are baked in instead.
#
# What changed is only HOW they get baked in: pim_config.h is now static source
# carrying ch4 defaults, and this script emits the -D list that overrides them.  The
# conf-key-to-macro-name mapping lives HERE and nowhere else — putting it in setup.sh
# would make two copies of it, and the ones that differ in name (HBM_BANK_STRIDE ->
# PIM_BANK_STRIDE, MC_SPAN -> PIM_MC_CH_SPAN) are exactly where a second copy rots.
#
# --header is a hand tool for refreshing the defaults in pim_config.h when ch4's conf
# changes.  No Makefile calls it; nothing consumes its output automatically.
#
# THE TYPE SUFFIXES ARE NOT DECORATION.  ULL on the addresses and u on the counts
# have to match what pim_config.h declares, because a missing suffix is invisible:
# the values are already long enough on LP64 that neither -Werror nor a _Static_assert
# can see the difference, and the first symptom is an address computed in the wrong
# width.
set -euo pipefail

MODE=defs
case "${1:---defs}" in
    --defs)   MODE=defs ;;
    --header) MODE=header ;;
    -h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 0 ;;
    *) echo "gen_config.sh: unknown argument '$1' (--defs | --header)" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "gen_config.sh: too many arguments" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(cd "$HERE/../platform" && pwd)"

# NOT an argument.  PIM_CONF_PATH must name the symlink the runtime check reads, so
# the file this script reads and the file that check watches have to be the same one.
CONF="$PLATFORM_DIR/active"

if [[ ! -e "$CONF" ]]; then
    echo "gen_config.sh: no platform selected ($CONF)." >&2
    echo "               $HERE/../scripts/setup.sh --platform ch2   (or ch1 / ch4)" >&2
    exit 2
fi
NAME="$(basename "$(readlink -f "$CONF")" .conf)"

EMU_TOP="$(cd "$HERE/.." && pwd)"        # the confs reference this
# shellcheck disable=SC1090
source "$CONF"

# EVERY key, every time — including the ones that are identical in all three confs
# today.  Emitting only what differs would mean a conf that changes OFF_VIOL (ch1.conf
# already warns that it might) silently keeps the ch4 default instead.
req() { [[ -n "${!1:-}" ]] || { echo "gen_config.sh: $CONF has no $1" >&2; exit 2; }; }
for k in NCH NBANK BAR2_AXI_BASE OFF_GPR OFF_CFR OFF_VIOL OFF_IMEM \
         HBM_BASE HBM_CH_SPAN HBM_BANK_STRIDE BANK_WINDOW \
         MC_BASE MC_SPAN UPLOAD_PATH NTAIL_POLICY SCHEDULE FEATURE_T_LATCH ADDR_MAP; do
    req "$k"
done

# ADDR_MAP is a menu number, 1 or 2; the macro carries the REGISTER value, 0 or 1.
# Keeping the two apart on purpose: the conf and the command line speak in choices,
# MODE_CTRL speaks in a bit, and converting in one place stops the two being confused.
case "$ADDR_MAP" in
    1) MAPBIT=0 ;;
    2) MAPBIT=1 ;;
    *) echo "gen_config.sh: ADDR_MAP must be 1 (ChRoBaCo) or 2 (RoChBaCo), got '$ADDR_MAP'" >&2
       exit 2 ;;
esac

# A single quote in either path would break out of the quoting below and hand the
# compiler something else entirely.  Spaces survive it; single quotes cannot.
case "$EMU_TOP$CONF" in
    *"'"*) echo "gen_config.sh: a single quote in the tree path cannot be quoted safely:" >&2
           echo "               $EMU_TOP" >&2
           exit 2 ;;
esac

if [[ "$MODE" == defs ]]; then
    # -DX='"str"' — the outer single quotes survive make and are stripped by the
    # recipe's shell, so the compiler sees -DX="str".  Bare -DX="str" does not work:
    # the recipe shell eats the double quotes and the compiler reads an identifier.
    D=(
        "-DPIM_CONFIG_FROM_CONF=1"
        "-DPIM_PLATFORM_NAME='\"$NAME\"'"
        "-DPIM_CONF_PATH='\"$CONF\"'"
        "-DPIM_NCH=${NCH}u"
        "-DPIM_NBANK=${NBANK}u"
        "-DPIM_BAR2_AXI_BASE=${BAR2_AXI_BASE}ULL"
        "-DPIM_OFF_GPR=${OFF_GPR}ULL"
        "-DPIM_OFF_CFR=${OFF_CFR}ULL"
        "-DPIM_OFF_VIOL=${OFF_VIOL}ULL"
        "-DPIM_OFF_IMEM=${OFF_IMEM}ULL"
        "-DPIM_HBM_BASE=${HBM_BASE}ULL"
        "-DPIM_HBM_CH_SPAN=${HBM_CH_SPAN}ULL"
        "-DPIM_BANK_STRIDE=${HBM_BANK_STRIDE}ULL"
        "-DPIM_BANK_WINDOW=${BANK_WINDOW}ULL"
        "-DPIM_MC_BASE=${MC_BASE}ULL"
        "-DPIM_MC_CH_SPAN=${MC_SPAN}ULL"
        "-DPIM_UPLOAD_PATH='\"$UPLOAD_PATH\"'"
        "-DPIM_NTAIL_POLICY='\"$NTAIL_POLICY\"'"
        "-DPIM_SCHEDULE='\"$SCHEDULE\"'"
        "-DPIM_FEATURE_T_LATCH=${FEATURE_T_LATCH}"
        "-DPIM_ADDR_MAP=${MAPBIT}"
    )
    echo "${D[*]}"
    exit 0
fi

cat <<EOF
// ---- generated by hwdef/gen_config.sh --header from $CONF ($NAME) ----
// Paste into pim_config.h to refresh the compiled-in defaults.  Keep the #ifndef
// guards: they are what lets setup.sh override each value with -D.
#ifndef PIM_CONFIG_FROM_CONF
#define PIM_CONFIG_FROM_CONF 0
#endif
#ifndef PIM_PLATFORM_NAME
#define PIM_PLATFORM_NAME   "$NAME"
#endif
#ifndef PIM_CONF_PATH
#define PIM_CONF_PATH       ""
#endif
#ifndef PIM_NCH
#define PIM_NCH             ${NCH}u
#endif
#ifndef PIM_NBANK
#define PIM_NBANK           ${NBANK}u
#endif
#ifndef PIM_BAR2_AXI_BASE
#define PIM_BAR2_AXI_BASE   ${BAR2_AXI_BASE}ULL
#endif
#ifndef PIM_OFF_GPR
#define PIM_OFF_GPR         ${OFF_GPR}ULL
#endif
#ifndef PIM_OFF_CFR
#define PIM_OFF_CFR         ${OFF_CFR}ULL
#endif
#ifndef PIM_OFF_VIOL
#define PIM_OFF_VIOL        ${OFF_VIOL}ULL
#endif
#ifndef PIM_OFF_IMEM
#define PIM_OFF_IMEM        ${OFF_IMEM}ULL
#endif
#ifndef PIM_HBM_BASE
#define PIM_HBM_BASE        ${HBM_BASE}ULL
#endif
#ifndef PIM_HBM_CH_SPAN
#define PIM_HBM_CH_SPAN     ${HBM_CH_SPAN}ULL
#endif
#ifndef PIM_BANK_STRIDE
#define PIM_BANK_STRIDE     ${HBM_BANK_STRIDE}ULL
#endif
#ifndef PIM_BANK_WINDOW
#define PIM_BANK_WINDOW     ${BANK_WINDOW}ULL
#endif
#ifndef PIM_MC_BASE
#define PIM_MC_BASE         ${MC_BASE}ULL
#endif
#ifndef PIM_MC_CH_SPAN
#define PIM_MC_CH_SPAN      ${MC_SPAN}ULL
#endif
#ifndef PIM_UPLOAD_PATH
#define PIM_UPLOAD_PATH     "${UPLOAD_PATH}"
#endif
#ifndef PIM_NTAIL_POLICY
#define PIM_NTAIL_POLICY    "${NTAIL_POLICY}"
#endif
#ifndef PIM_SCHEDULE
#define PIM_SCHEDULE        "${SCHEDULE}"
#endif
#ifndef PIM_FEATURE_T_LATCH
#define PIM_FEATURE_T_LATCH ${FEATURE_T_LATCH}
#endif
#ifndef PIM_ADDR_MAP
#define PIM_ADDR_MAP        ${MAPBIT}
#endif
EOF
