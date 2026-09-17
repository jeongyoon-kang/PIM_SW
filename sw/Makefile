# SPDX-License-Identifier: MIT
# sw/ — the PIM software stack.  See docs/pim_sw_design.md.
#
#     make            # libpim + libpimrt  (user space only)
#     make drv        # pim.ko             (needs a kernel build tree)
#     make all        # both
#     make test       # the offline tests — no board, no loaded module
#     make clean
#
# THE DRIVER IS NOT PART OF THE DEFAULT GOAL.  It builds against a kernel tree that
# may not be installed, and — more to the point — the user-space half does not depend
# on it at build time.  The two sides share exactly one file, include/uapi/pim_ioctl.h,
# and that is a header.  Being able to build either one alone is the check that the
# separation is real.

.PHONY: all user drv lib runtime test clean
all: user drv

user: lib runtime

lib:
	$(MAKE) -C lib

runtime: lib
	$(MAKE) -C runtime

drv:
	$(MAKE) -C drv

# lib/test needs neither board nor module.  runtime/test runs its host-only phases
# either way and says loudly when it skips the board.
test: user
	$(MAKE) -C lib test
	@echo
	$(MAKE) -C runtime test

clean:
	$(MAKE) -C runtime clean
	$(MAKE) -C lib clean
	-$(MAKE) -C drv clean
