# SPDX-License-Identifier: MIT
# The GEMV tests.
#
#     make run                  # both
#     ./gemv_test 0000:01:00.0  # one tile end to end; phase 0 needs no board
#     ./tlatch_test             # one accumulator latch vs two (ISR[35])
#
# THE TIMING KNOBS ARE NOT HERE.  They are control-plane registers, and until the
# driver owns the CFR (design §9.1 plan A) the tier that reaches them is
# hwdef/test — direct MMIO, no module, no queues.  See hwdef/test/emu_timing.c.
#
# LINKS libpimrt.so AND libpim.so, NOT their objects.  Unlike lib/test/pool_test —
# which has to substitute an internal symbol to fake the driver — this test needs
# nothing internal, so it is held to the rule: a test may only call what an
# application could call.  If something here needs a symbol that is not in
# include/pim/, that is a finding about the API, not a reason to link the objects.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g
override CFLAGS += -I../../include
LDFLAGS ?= -L.. -L../../lib -Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,'$$ORIGIN/../../lib'
LDLIBS  ?= -lpimrt -lpim

.PHONY: all run clean
all: gemv_test tlatch_test col_probe eager_board xfer_bench

%_test: %_test.c ../libpimrt.so ../../lib/libpim.so \
        ../../include/pimrt/pim_gemv.h ../../include/pimrt/pim_exec.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

../libpimrt.so:
	$(MAKE) -C ..

../../lib/libpim.so:
	$(MAKE) -C ../../lib

# THE HARDWARE IS SERIAL.  Never run this alongside a probe from hwdef/test.
# col_probe asks the board a question rather than checking a known answer, so it is
# not part of `run`.  It needs hwdef for the ISR encoder: it builds its own program.
col_probe: col_probe.c
	$(CC) $(CFLAGS) -I../../../hwdef -o $@ $^ $(LDFLAGS) $(LDLIBS)

eager_board: eager_board.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Asks what a small transfer costs, rather than checking a known answer.  Not in
# `run` for the same reason col_probe is not.
xfer_bench: xfer_bench.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lpthread

run: all
	./gemv_test
	@echo
	./tlatch_test

clean:
	rm -f gemv_test tlatch_test col_probe eager_board xfer_bench
