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
# The platform comes from platform/config through ../pim_config.h.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -I..
LDLIBS  ?= -lm

HWDEF   := ../libhwdef.a
PROBES  := emu_sanity emu_gpr_loop emu_hbm_direct emu_mc emu_gemv emu_ewmul emu_chain \
           emu_timing emu_aimcmp

.PHONY: all clean FORCE
all: $(PROBES)

$(PROBES): %: %.c $(HWDEF) ../emu_regs.h ../pim_platform.h ../pim_config.h
	$(CC) $(CFLAGS) -o $@ $< $(HWDEF) $(LDLIBS)

# Always descend: hwdef/ regenerates pim_config.h and rebuilds libhwdef.a if it changed.
$(HWDEF): FORCE
	$(MAKE) -C .. libhwdef.a
../pim_config.h: $(HWDEF) ;

clean:
	rm -f $(PROBES)
