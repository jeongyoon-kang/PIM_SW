# SPDX-License-Identifier: MIT
# pim.ko — the /dev/pim device-memory ledgers.  See ../docs/pim_sw_design.md §6, §9.4.
#
#     make                                  # build against the running kernel
#     make KDIR=/lib/modules/5.15.165/build # or a specific one
#     make load  CH=4                       # insmod with a channel count
#     make unload
#     make clean
#
# THE CHANNEL COUNT LIVES IN THE INSMOD LINE, not in a build.  That is design §6.4:
# one source of truth for topology, injected once, reported to libpim over GET_INFO.
# Rebuilding the module to change channels would recreate exactly the drift this
# stack exists to remove — so there is no CH= that affects `make`, only `make load`.

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# Defaults for `make load`.  Override on the command line: make load CH=2
CH       ?= 4
BANKS    ?= 16
ROW      ?= 2048
BASE     ?= 0x020400000000
CH_SPAN  ?= 0x100000000
MAP      ?= 1
# The GPR region (design §9.4): BAR2's AXI base + OFF_GPR, 4 MiB, 4 KiB pages.
# The CFR — whose offset 0 is the DOORBELL — starts immediately after it, so do not
# round GPR_SIZE up.
GPR_BASE ?= 0x020200000000
GPR_SIZE ?= 0x400000
GPR_PAGE ?= 4096

.PHONY: all clean load unload reload status
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

load: all
	sudo insmod pim.ko pim_channels=$(CH) pim_banks=$(BANKS) \
	                   pim_row_size=$(ROW) pim_base=$(BASE) \
	                   pim_ch_span=$(CH_SPAN) pim_addr_map=$(MAP) \
	                   pim_gpr_base=$(GPR_BASE) pim_gpr_size=$(GPR_SIZE) \
	                   pim_gpr_page=$(GPR_PAGE)
	@dmesg | tail -4

unload:
	sudo rmmod pim

reload: unload load

status:
	@lsmod | /bin/grep -q '^pim ' && cat /proc/pim || echo "pim not loaded"
