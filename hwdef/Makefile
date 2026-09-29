# SPDX-License-Identifier: MIT
# hwdef — what the hardware IS, as C.
#
#     make            # pim_config.h (from platform/config), then libhwdef.a
#     make test       # the probes in test/
#
# WHY THIS IS ITS OWN DIRECTORY
# Two things need the address map, the ISR encoding and the platform conf: the
# probes that drive the hardware directly, and the runtime that abstracts it.
# Neither may depend on the other — a probe must be able to contradict the runtime,
# and the runtime must not change when a test does.  So the shared part is neither
# of them, and lives here.
#
#     hwdef/          <- this.  depends on platform/config
#       test/         <- probes.  depend on hwdef only
#
# pim_config.h is generated from platform/config by gen_config.sh on every make and
# rewritten only when its content changes; everything that includes it depends on it.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -fPIC
AR      ?= ar

SRC  := pim_platform.c
OBJ  := $(SRC:.c=.o)
HDRS := emu_regs.h pim_platform.h pim_config.h

.DEFAULT_GOAL := all

.PHONY: all test clean FORCE
all: libhwdef.a

# FORCE: re-read platform/config every time.
pim_config.h: FORCE
	./gen_config.sh --header $@

libhwdef.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

test: libhwdef.a
	$(MAKE) -C test

clean:
	rm -f $(OBJ) libhwdef.a pim_config.h
	$(MAKE) -C test clean
