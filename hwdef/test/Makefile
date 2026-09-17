# SPDX-License-Identifier: MIT
# Probes that drive the hardware DIRECTLY.
#
#     make                      # all of them
#     make emu_chain            # one
#
# These build ISR words by hand and talk to the board over QDMA and MMIO.  They do
# NOT link libpim.so and must not: their job is to say what the hardware does, and a
# test that goes through the abstraction it is checking can only confirm that the
# abstraction is self-consistent.
#
# The one thing they share with the runtime is hwdef — the address map, the ISR
# encoding and the platform conf, which are facts about the board rather than
# anybody's design.
#
# Choose an image ONCE and every tool follows:
#     ../../scripts/setup.sh --platform ch2

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -I..
# Passed down from scripts/setup.sh.  A probe built without these gets the ch4
# defaults and links happily against a libhwdef.a built for something else — half
# stride, wrong bank spacing, and the banner naming the other platform.  These tools
# drive the board directly, so go through setup.sh.
override CFLAGS += $(PIM_DEFS)
LDLIBS  ?= -lm

HWDEF   := ../libhwdef.a
PROBES  := emu_sanity emu_gpr_loop emu_hbm_direct emu_mc emu_gemv emu_ewmul emu_chain \
           emu_timing

.PHONY: all clean
all: $(PROBES)

# pim_config.h now carries defaults and the real values come as -D, which make cannot
# see — the file dependency below no longer catches a platform switch on its own.
# setup.sh cleans before it rebuilds; that is what catches it.
$(PROBES): %: %.c $(HWDEF) ../emu_regs.h ../pim_platform.h ../pim_config.h
	$(CC) $(CFLAGS) -o $@ $< $(HWDEF) $(LDLIBS)

$(HWDEF):
	$(MAKE) -C ..

clean:
	rm -f $(PROBES)
