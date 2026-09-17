# SPDX-License-Identifier: MIT
# runtime — the HW/SW abstraction layer.  Produces libpim.so.
#
#     make            # libpim.a, libpim.so
#     make test       # build and run the unit tests in test/
#
# DEPENDS ON hwdef AND NOTHING ELSE.
# hwdef carries the address map, the ISR encoding and the platform conf — facts
# about the board.  Everything above them (allocation, layout, scheduling) is this
# library's design and lives here.
#
# It must NOT depend on hwdef/test.  Those probes exist to contradict this library
# when it is wrong; a library that shared code with them could not be contradicted.
# (It did, briefly: libpim.so compiled a .c out of the probe directory.)

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -fPIC -I../hwdef
# See hwdef/Makefile.  A bare `make` here compiles the ch4 defaults; go through
# scripts/setup.sh --platform chN so these arrive.
override CFLAGS += $(PIM_DEFS)
LDLIBS  ?= -lm -ldl          # hwdef no longer needs -ldl: the platform is compiled in, not discovered

HWDEF_DIR := ../hwdef
HWDEF_A   := $(HWDEF_DIR)/libhwdef.a

LIB_SRC := pim_dev.c pim_alloc.c pim_layout.c pim_prog.c pim_gemv.c pim_attn.c \
           pim_golden.c pim_mac_exact.c
LIB_OBJ := $(LIB_SRC:.c=.o)
# pim_config.h is a real header and stays a dependency, but it no longer CHANGES on a
# platform switch — the values arrive as -D now, and make cannot see those.  So this
# line no longer protects against relinking from stale objects; setup.sh running
# `clean` before every switch is what does, and that is not optional either.
HDRS    := pim.h pim_internal.h pim_layout.h \
           $(HWDEF_DIR)/emu_regs.h $(HWDEF_DIR)/pim_platform.h \
           $(HWDEF_DIR)/pim_config.h

# WITHOUT THIS, `make` here built NOTHING.  The first target rule in a makefile
# becomes the default goal, and the rule that used to sit right here named
# ../hwdef/pim_config.h — so `make -C runtime` reported "Nothing to be done" and
# libpim.a was never built by setup.sh at all.  Only libpim.so got made, by the
# `test` target that follows.
.DEFAULT_GOAL := all

.PHONY: all test clean
all: libpim.a libpim.so

$(HWDEF_A):
	$(MAKE) -C $(HWDEF_DIR)

# The static archive carries hwdef's objects too, so a caller linking libpim.a needs
# nothing else.  The shared object links them in for the same reason.
libpim.a: $(LIB_OBJ) $(HWDEF_A)
	rm -f $@
	$(AR) rcsT $@ $(LIB_OBJ) $(HWDEF_A) 2>/dev/null || \
	  { mkdir -p .thin && cd .thin && $(AR) x ../$(HWDEF_A) && cd .. && \
	    $(AR) rcs $@ $(LIB_OBJ) .thin/*.o && rm -rf .thin; }

libpim.so: $(LIB_OBJ) $(HWDEF_A)
	$(CC) $(CFLAGS) -shared -o $@ $(LIB_OBJ) $(HWDEF_A) $(LDLIBS)

$(LIB_OBJ): %.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

test: libpim.so
	$(MAKE) -C test

clean:
	rm -f $(LIB_OBJ) libpim.a libpim.so
	rm -rf .thin
	$(MAKE) -C test clean
