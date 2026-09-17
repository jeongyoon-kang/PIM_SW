# SPDX-License-Identifier: MIT
# _pim — the per-op binding.  Marshalling only; see _pim.cpp.
#
#     make                 # build _pim*.so against the python on PATH
#     make PYTHON=/path/to/python
#     make check           # import it and print the geometry
#     make clean
#
# THE ONLY C++ IN THE TREE, and it links libpim/libpimrt exactly as an application
# would — through their public headers, not their objects.  If something here needs
# a symbol those libraries do not export, the answer is to export it deliberately,
# not to reach past them.
#
# WHY NOT setup.py.  There is one extension, it is built in place, and it is
# imported from the directory it lives in.  A packaging layer would add a build
# system to learn for no behaviour.

PYTHON  ?= python3
SW      := ..

CXXFLAGS ?= -O2 -g -Wall -Wextra -std=c++17 -fPIC -fvisibility=hidden
override CXXFLAGS += -I$(SW)/include $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS  ?= -L$(SW)/runtime -L$(SW)/lib \
            -Wl,-rpath,'$$ORIGIN/$(SW)/runtime' -Wl,-rpath,'$$ORIGIN/$(SW)/lib'
LDLIBS   ?= -lpimrt -lpim

EXT := _pim$(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo .so)

.PHONY: all check clean
all: $(EXT)

$(SW)/runtime/libpimrt.so:
	$(MAKE) -C $(SW)/runtime

$(EXT): _pim.cpp $(SW)/runtime/libpimrt.so
	$(CXX) $(CXXFLAGS) -shared -o $@ _pim.cpp $(LDFLAGS) $(LDLIBS)

# Imports and asks the driver for the geometry, which is the shortest thing that
# proves the whole chain: the extension loaded, it found both libraries at the
# rpaths above, and the module is up.
check: $(EXT)
	@$(PYTHON) -c "import _pim; print('_pim ok:', _pim.geometry())"

clean:
	rm -f _pim*.so
