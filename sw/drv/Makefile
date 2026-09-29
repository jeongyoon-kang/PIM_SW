# SPDX-License-Identifier: MIT
# pim.ko — the /dev/pim device-memory ledgers.  See ../docs/pim_sw_design.md §6, §9.4.
#
#     make                                  # build against the running kernel
#     make KDIR=/lib/modules/5.15.165/build # or a specific one
#     make load                             # insmod
#     make unload
#     make clean
#
# The module parameters default to platform/config's values, compiled in through
# hwdef/pim_config.h, so a plain `insmod pim.ko` matches the board.  After editing
# platform/config: make, then `make reload`.  Parameters given to insmod still win.

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

.PHONY: all clean load unload reload status
all:
	$(MAKE) -C ../../hwdef pim_config.h
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

load: all
	sudo insmod pim.ko
	@dmesg | tail -4

unload:
	sudo rmmod pim

reload: unload load

status:
	@lsmod | /bin/grep -q '^pim ' && cat /proc/pim || echo "pim not loaded"
