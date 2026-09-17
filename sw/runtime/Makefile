# SPDX-License-Identifier: MIT
# libpimrt — the compute runtime.  Design §9.
#
#     make            # libpimrt.a, libpimrt.so
#     make clean
#
# ITS HEADERS LIVE IN ../include/pimrt; libpim's live in ../include/pim.  Both are
# on the include path, and the direction is one-way: this layer includes "pim/..."
# freely, and nothing under ../lib may include "pimrt/...".
#
# LINKS libpim, NOT its objects.  A runtime that reached into the library's internals
# would make the library's public surface a fiction; here it can only call what an
# application could call, which is what keeps pim.h honest.
#
# IT INCLUDES hwdef/emu_regs.h — the ISR encoding and the control-plane offsets, which
# are facts about the board and have exactly one home.  That file is TOPOLOGY-FREE:
# GPR, CFR and IMEM are one each however many channels the image has, so including it
# brings no compile-time channel count with it.
#
# IT MUST NOT INCLUDE hwdef/pim_platform.h, which does carry one.  This stack takes
# topology from the driver at run time, and a binary with two answers for the channel
# count is worse than one with none.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -fPIC
# ../../hwdef for emu_regs.h — the ISR encoding and the control plane, which are
# facts about the board and have one home.  NOT for pim_platform.h; see below.
override CFLAGS += -I../include -I../../hwdef
LDFLAGS ?= -L../lib -Wl,-rpath,'$$ORIGIN/../lib'
LDLIBS  ?= -lpim

SRC := pim_exec.c pim_lower.c pim_gemv.c pim_mac_exact.c
OBJ := $(SRC:.c=.o)
HDR := ../include/pim/pim_addr.h ../include/pimrt/pim_logical.h ../include/pimrt/pim_exec.h ../include/pimrt/pim_gemv.h \
       ../include/pim/pim.h ../include/pim/pim_geometry.h ../../hwdef/emu_regs.h

.DEFAULT_GOAL := all

.PHONY: all test clean
all: libpimrt.a libpimrt.so

test: all
	$(MAKE) -C test run

../lib/libpim.so:
	$(MAKE) -C ../lib

libpimrt.a: $(OBJ)
	rm -f $@
	$(AR) rcs $@ $(OBJ)

libpimrt.so: $(OBJ) ../lib/libpim.so
	$(CC) $(CFLAGS) $(LDFLAGS) -shared -o $@ $(OBJ) $(LDLIBS)

$(OBJ): %.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) libpimrt.a libpimrt.so
	$(MAKE) -C test clean
