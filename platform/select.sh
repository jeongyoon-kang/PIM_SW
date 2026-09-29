# platform/select.sh — sourced by qdma_queues.sh and setup_permissions.sh.
#
# Loads the board values (common.conf: BDF, driver, queues, permissions) and sets
# EMU_TOP / SCRIPT_DIR.

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMU_TOP="$(cd "$PLATFORM_DIR/.." && pwd)"
SCRIPT_DIR="$EMU_TOP/scripts"

# shellcheck source=common.conf
source "$PLATFORM_DIR/common.conf"
