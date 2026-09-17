# SPDX-License-Identifier: MIT
# Unit tests for the runtime, THROUGH the runtime.
#
#     make          # build
#     make run      # build and run (needs the board)
#
# LINKS libpim.so, NOT the objects.
# That is the whole point of this directory.  Linking the static objects would let a
# test reach an internal symbol, and then it would be testing the implementation
# instead of the thing that ships.  Here a test can only call what pim.h exports —
# which is exactly what the application will be able to call.
#
# rpath so the tests run from the build tree without LD_LIBRARY_PATH.  $$ORIGIN is
# escaped for make; the linker sees $ORIGIN.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -Werror -std=c11 -O2 -g -I.. -I../../hwdef
# load_test.c reads the address map directly, so it needs these as much as the
# library does.  From scripts/setup.sh.
override CFLAGS += $(PIM_DEFS)
LDFLAGS ?= -L.. -Wl,-rpath,'$$ORIGIN/..'
LDLIBS  ?= -lpim -lm

TESTS := pim_test load_test

.PHONY: all run clean
all: $(TESTS)

$(TESTS): %: %.c ../libpim.so ../pim.h ../pim_layout.h \
                 ../../hwdef/pim_config.h ../../hwdef/pim_platform.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

../libpim.so:
	$(MAKE) -C ..

# Hardware is SERIAL — never run two of these at once, and never alongside a probe.
run: $(TESTS)
	./pim_test
	./load_test

clean:
	rm -f $(TESTS)
