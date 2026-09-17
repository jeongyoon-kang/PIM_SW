# SPDX-License-Identifier: MIT
# libpim — allocation, translation and data movement.  Design §7 and §8.
#
#     make            # libpim.a, libpim.so
#     make clean
#
# DEPENDS ON ../include/pim AND NOTHING ELSE.
#
# The headers are split by layer so that the dependency direction is visible in
# every #include line rather than only in this comment:
#
#     include/pim/    libpim    pim.h  pim_geometry.h  pim_addr.h
#     include/pimrt/  libpimrt  pim_exec.h  pim_logical.h  pim_gemv.h
#
# A `#include "pimrt/..."` appearing in a file under lib/ IS the violation; there
# is no need to reason about which layer a name belongs to.
#
# In particular it does NOT depend on ../../hwdef.  That is the whole point of this
# rewrite: hwdef compiles the topology in with -D, so a channel-count change means
# rebuilding every binary that touched it.  Here the topology arrives from the
# driver at open (design §6.4), so this library is built once and runs against any
# image the module was told about.
#
# It also does not depend on the compute runtime above it.  ../runtime links this;
# nothing here may reach the other way.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -fPIC
override CFLAGS += -I../include
LDLIBS  ?= -lpthread

SRC := pim_dev.c pim_geom.c pim_alloc.c pim_addr.c pim_memcpy.c
OBJ := $(SRC:.c=.o)
HDR := pim_internal.h ../include/pim/pim.h ../include/pim/pim_geometry.h \
       ../include/uapi/pim_ioctl.h

.DEFAULT_GOAL := all

.PHONY: all test clean
all: libpim.a libpim.so

# Neither test needs a board or a loaded module; see test/Makefile.
test:
	$(MAKE) -C test run

libpim.a: $(OBJ)
	rm -f $@
	$(AR) rcs $@ $(OBJ)

libpim.so: $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(OBJ) $(LDLIBS)

$(OBJ): %.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) libpim.a libpim.so
	$(MAKE) -C test clean
