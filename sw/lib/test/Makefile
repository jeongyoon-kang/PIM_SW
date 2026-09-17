# SPDX-License-Identifier: MIT
# Tests for libpim that need NEITHER a board NOR a loaded pim.ko.
#
#     make run
#
# Both of these check things that are wrong long before hardware could say so:
# the address arithmetic of both regions, and the allocators' refusal to alias.  The integration
# test — open the real /dev/pim, move real bytes — is a different program and
# waits on the module being loadable.
#
# geom_test reaches into ../../../hwdef on purpose; see the comment at its top.

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -std=c11 -O1 -g
override CFLAGS += -I.. -I../../include
LDLIBS  ?= -lpthread

HWDEF   := ../../../hwdef
# pim_rt.c comes along so that the GPR word conversion — the runtime's half of
# design §9.4 — is covered by the same simulated card.  It is pure arithmetic on
# libpim's public API, so it needs nothing else.
LIB     := ../pim_alloc.c ../pim_geom.c ../pim_addr.c

.PHONY: all run clean
all: geom_test pool_test lower_test eager_test

# hwdef is compiled with whatever platform/active says; go through
# scripts/setup.sh if you want a different one compared.
geom_test: geom_test.c ../pim_geom.c $(HWDEF)/pim_platform.c
	$(CC) $(CFLAGS) -I$(HWDEF) -o $@ $^ -lm

pool_test: pool_test.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# lower_test links the runtime's two code-generation passes as well, and hwdef for
# the ISR encoder.  It still needs no board: the addresses it resolves come from the
# same simulated driver pool_test uses.
lower_test: lower_test.c $(LIB) ../pim_memcpy.c ../../runtime/pim_lower.c ../../runtime/pim_gemv.c \
            ../../runtime/pim_exec.c ../../runtime/pim_mac_exact.c
	$(CC) $(CFLAGS) -I$(HWDEF) -o $@ $^ $(LDLIBS)

eager_test: eager_test.c $(LIB) ../pim_memcpy.c ../../runtime/pim_gemv.c \
            ../../runtime/pim_lower.c ../../runtime/pim_exec.c ../../runtime/pim_mac_exact.c
	$(CC) $(CFLAGS) -I$(HWDEF) -o $@ $^ $(LDLIBS)

run: all
	./geom_test
	@echo
	./pool_test
	@echo
	./lower_test
	@echo
	./eager_test

clean:
	rm -f geom_test pool_test lower_test eager_test
