# platform/select.sh — sourced by every script in sw/.
#
# Resolves WHICH platform is active and sources both config layers.  Every script
# does this the same way so that "which channel count am I talking to" can never
# differ between two tools in the same session.
#
# Order of precedence:
#   1. $PIM_PLATFORM              e.g. PIM_PLATFORM=ch2 ./reprogram.sh
#   2. platform/active            a symlink written by sw/setup.sh
#   3. nothing -> ERROR
#
# THERE IS NO DEFAULT.  A wrong channel count does not fail loudly on this
# hardware: a 1-channel runtime talking to a 2-channel board writes half the
# operands, reads half the results, and returns plausible numbers.  Refusing to
# start is the only behaviour that cannot be mistaken for success.

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMU_TOP="$(cd "$PLATFORM_DIR/.." && pwd)"
SCRIPT_DIR="$EMU_TOP/scripts"

# shellcheck source=common.conf
source "$PLATFORM_DIR/common.conf"

if [[ -n "${PIM_PLATFORM:-}" ]]; then
    PIM_CONF="$PLATFORM_DIR/${PIM_PLATFORM}.conf"
    if [[ ! -r "$PIM_CONF" ]]; then
        echo "ERROR: PIM_PLATFORM=$PIM_PLATFORM but $PIM_CONF is not readable." >&2
        echo "       available: $(cd "$PLATFORM_DIR" && ls *.conf | grep -v common | tr '\n' ' ')" >&2
        exit 2
    fi
elif [[ -e "$PLATFORM_DIR/active" ]]; then
    PIM_CONF="$PLATFORM_DIR/active"
    PIM_PLATFORM="$(basename "$(readlink -f "$PIM_CONF")" .conf)"
else
    echo "ERROR: no platform selected." >&2
    echo "       sudo $SCRIPT_DIR/setup.sh --platform ch2      (or ch1 / ch4)" >&2
    echo "       or set PIM_PLATFORM=ch2 for one command." >&2
    exit 2
fi

# shellcheck source=ch2.conf
source "$PIM_CONF"
export PIM_PLATFORM
