# SPDX-License-Identifier: MIT
# hwdef — what the hardware IS, as C.
#
#     make            # libhwdef.a
#     make test       # the probes in test/
#
# WHY THIS IS ITS OWN DIRECTORY
# Two things need the address map, the ISR encoding and the platform conf: the
# probes that drive the hardware directly, and the runtime that abstracts it.
# Neither may depend on the other — a probe must be able to contradict the runtime,
# and the runtime must not change when a test does.  So the shared part is neither
# of them, and lives here.
#
#     hwdef/          <- this.  depends on nothing
#       test/         <- probes.  depend on hwdef only
#     runtime/        <- libpim.so.  depends on hwdef only
#       test/         <- links libpim.so
#
# It used to live in the probe directory, which made libpim.so depend on the test
# tree: editing a probe's header changed the shipped library.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -fPIC
AR      ?= ar

# The platform constants arrive here, from scripts/setup.sh via gen_config.sh --defs.
# `override` because CFLAGS is ?= and a caller who sets it on the command line would
# otherwise drop this whole line — silently building the ch4 defaults instead.
override CFLAGS += $(PIM_DEFS)

SRC  := pim_platform.c
OBJ  := $(SRC:.c=.o)
HDRS := emu_regs.h pim_platform.h pim_config.h

# Stated rather than left to ordering: whichever rule ends up first, `make` here
# means the library.
.DEFAULT_GOAL := all

.PHONY: all test clean
all: libhwdef.a

libhwdef.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

test: libhwdef.a
	$(MAKE) -C test

# pim_config.h is NOT listed here.  It is source now — carrying the ch4 defaults that
# -D overrides — and setup.sh runs `clean` before every platform switch, so deleting
# it here would take out the header on the first switch and there is nothing to
# regenerate it from.
clean:
	rm -f $(OBJ) libhwdef.a
	$(MAKE) -C test clean
