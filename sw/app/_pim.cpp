// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// _pim.cpp — the only C++ in the tree, and it does nothing but marshal.
//
// WHY IT EXISTS WHEN device.py ALREADY USES ctypes.  Different call rates.  The
// geometry and the budget are read once at startup and ctypes is right for them.
// The ops below run about 500 times per token; ctypes costs 1-3 us per call, which
// would be more than a small op's entire launch.  pybind11 is ~0.1 us.
//
// WHY POINTERS AND NOT BUFFERS.  Everything here is BF16, and numpy has no bf16, so
// a torch tensor cannot expose one through the buffer protocol.  torch's data_ptr()
// is the honest way in, and it means this file never has to know what torch is.
// The cost is that a caller can pass a pointer to the wrong thing; the Python layer
// above (pimllm/ops.py) is where that is made hard, because it can see dtype and
// shape and this cannot.
//
// EVERY ERROR IS A SENTENCE.  libpim returns const char* rather than an errno and
// those sentences say what to do about it, so they become the exception message
// verbatim.  Nothing here invents a message of its own except where it is checking
// something libpim cannot see.
//////////////////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
#include "pim/pim.h"
#include "pimrt/pim_op.h"
#include "pimrt/pim_tensor.h"
}

namespace py = pybind11;

static void ck(const char *bad)
{
    if (bad) throw std::runtime_error(bad);
}

static pim_ctx *ctx()
{
    pim_ctx *c = pim_default();
    if (!c)
        throw std::runtime_error(std::string("cannot open the PIM device: ")
                                 + pim_last_error()
                                 + "\n  is pim.ko loaded?  sudo make -C sw/drv load CH=2");
    return c;
}

// ---------------------------------------------------------------- Tensor ------
// Owns its allocation.  Freed on __del__ or .free(); a tensor that outlives the
// interpreter is the process exiting, which frees the card memory anyway.
struct Tensor {
    pim_tensor t{};
    bool live = false;

    Tensor(int layout, uint32_t nout, uint32_t nred, unsigned aflags)
    {
        ck(pim_tensor_alloc(ctx(), (pim_layout)layout, nout, nred, aflags, &t));
        live = true;
    }
    ~Tensor() { free_(); }

    void free_()
    {
        if (live) { pim_tensor_free(ctx(), &t); live = false; }
    }
    void check() const
    {
        if (!live) throw std::runtime_error("this pim.Tensor has been freed");
    }

    // `src` is host BF16 in the tensor's layout order — [nout][nred] for OUT_MAJOR,
    // [nred][nout] for RED_MAJOR.  `nelem` is checked against the shape rather than
    // trusted, because the failure it prevents is a read past the end of a torch
    // tensor and that is a segfault at best.
    void upload(uintptr_t src, size_t nelem)
    {
        check();
        size_t want = (size_t)t.nout * t.nred;
        if (nelem != want)
            throw std::runtime_error("upload: " + std::to_string(nelem)
                                     + " elements for a [" + std::to_string(t.nout)
                                     + " x " + std::to_string(t.nred) + "] tensor, which wants "
                                     + std::to_string(want));
        ck(pim_tensor_upload(ctx(), &t, (const void *)src));
    }

    void append(uint32_t first, uint32_t count, uintptr_t src, size_t nelem)
    {
        check();
        uint32_t other = (t.layout == PIM_LAYOUT_OUT_MAJOR) ? t.nred : t.nout;
        size_t want = (size_t)count * other;
        if (nelem != want)
            throw std::runtime_error("append: " + std::to_string(nelem)
                                     + " elements for " + std::to_string(count)
                                     + " step(s) of " + std::to_string(other));
        ck(pim_tensor_append(ctx(), &t, first, count, (const void *)src));
    }

    void truncate(uint32_t pos) { check(); ck(pim_tensor_truncate(ctx(), &t, pos)); }

    size_t bytes() const { return pim_tensor_bytes(pim_geom(), &t); }
};

// --------------------------------------------------------------- Runtime ------
struct Runtime {
    pim_rt *rt = nullptr;

    Runtime(uint32_t max_red, uint32_t max_out_groups, bool allow_t, bool set_timing)
    {
        pim_rt_config cfg{};
        cfg.max_red = max_red;
        cfg.max_out_groups = max_out_groups;
        cfg.allow_t_latch = allow_t;
        cfg.set_timing = set_timing;
        ck(pim_rt_open(ctx(), &cfg, &rt));
    }
    ~Runtime() { if (rt) pim_rt_close(rt); }

    // y must have room for out_count * nch * nbank BF16 — INCLUDING the padding
    // outputs of the last supergroup, which are written and then ignored by the
    // caller.  Sizing it to the model's real output count is a two-byte overrun per
    // padded lane and it is the mistake this signature makes easy to catch.
    void matvec(Tensor &m, uint32_t out_first, uint32_t out_count,
                uint32_t red_off, uint32_t red_len,
                uintptr_t v, size_t vn, uintptr_t y, size_t yn, int mode)
    {
        m.check();
        if (vn != red_len)
            throw std::runtime_error("matvec: vector has " + std::to_string(vn)
                                     + " elements, red_len is " + std::to_string(red_len));
        size_t want = pim_op_outputs(rt, out_count);
        if (yn < want)
            throw std::runtime_error("matvec: output buffer holds " + std::to_string(yn)
                                     + " but " + std::to_string(out_count)
                                     + " supergroups write " + std::to_string(want)
                                     + " (padding outputs included)");
        // The launch can be milliseconds and it polls; nothing here touches Python
        // objects, so hold nothing up.
        py::gil_scoped_release nogil;
        ck(pim_op_matvec(rt, &m.t, out_first, out_count, red_off, red_len,
                         (const uint16_t *)v, (uint16_t *)y, (pim_acc_mode)mode));
    }

    py::dict stats() const
    {
        pim_rt_stat s{};
        pim_rt_stat_get(rt, &s);
        py::dict d;
        d["launch_us"] = s.launch_us;
        d["polls"] = s.polls;
        d["nlaunch"] = s.nlaunch;
        d["nisr"] = s.nisr;
        d["nwrvec"] = s.nwrvec;
        d["nop"] = s.nop;
        return d;
    }
    void stats_reset() { pim_rt_stat_reset(rt); }
};

PYBIND11_MODULE(_pim, m)
{
    m.doc() = "Marshalling only.  The design is in include/pimrt/pim_op.h.";

    m.attr("OUT_MAJOR") = (int)PIM_LAYOUT_OUT_MAJOR;
    m.attr("RED_MAJOR") = (int)PIM_LAYOUT_RED_MAJOR;
    m.attr("ACC_SINGLE") = (int)PIM_ACC_SINGLE;
    m.attr("ACC_DUAL") = (int)PIM_ACC_DUAL;
    m.attr("ALLOC_ZERO") = (unsigned)PIM_ALLOC_F_ZERO;

    m.def("geometry", [] {
        const pim_geometry *g = pim_geom();
        if (!g) throw std::runtime_error(std::string("no device: ") + pim_last_error());
        py::dict d;
        d["nch"] = g->nch;
        d["nbank"] = g->nbank;
        d["row_bytes"] = g->row_bytes;
        d["unit_bytes"] = g->unit_bytes;
        d["dram_bytes"] = g->mem[PIM_MEM_DRAM].bytes;
        d["gpr_bytes"] = g->mem[PIM_MEM_GPR].bytes;
        return d;
    }, "The loaded driver's geometry, as a dict.");

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<int, uint32_t, uint32_t, unsigned>(),
             py::arg("layout"), py::arg("nout"), py::arg("nred"), py::arg("flags") = 0)
        .def("upload", &Tensor::upload, py::arg("ptr"), py::arg("nelem"))
        .def("append", &Tensor::append, py::arg("first"), py::arg("count"),
             py::arg("ptr"), py::arg("nelem"))
        .def("truncate", &Tensor::truncate, py::arg("pos"))
        .def("free", &Tensor::free_)
        .def_property_readonly("nout", [](const Tensor &t) { return t.t.nout; })
        .def_property_readonly("nred", [](const Tensor &t) { return t.t.nred; })
        .def_property_readonly("noutpad", [](const Tensor &t) { return t.t.noutpad; })
        .def_property_readonly("nredpad", [](const Tensor &t) { return t.t.nredpad; })
        .def_property_readonly("ngroups", [](const Tensor &t) { return t.t.ngroups; })
        .def_property_readonly("nchunks", [](const Tensor &t) { return t.t.nchunks; })
        .def_property_readonly("frontier", [](const Tensor &t) { return t.t.frontier; })
        .def_property_readonly("layout", [](const Tensor &t) { return (int)t.t.layout; })
        .def_property_readonly("bytes", &Tensor::bytes)
        .def("__repr__", [](const Tensor &t) {
            return "<pim.Tensor [" + std::to_string(t.t.nout) + " x "
                 + std::to_string(t.t.nred) + "] "
                 + (t.t.layout == PIM_LAYOUT_OUT_MAJOR ? "OUT_MAJOR" : "RED_MAJOR")
                 + " " + std::to_string(t.bytes() >> 10) + " KiB"
                 + (t.live ? "" : " FREED") + ">";
        });

    py::class_<Runtime>(m, "Runtime")
        .def(py::init<uint32_t, uint32_t, bool, bool>(),
             py::arg("max_red"), py::arg("max_out_groups"),
             py::arg("allow_t_latch") = false, py::arg("set_timing") = true)
        .def("matvec", &Runtime::matvec,
             py::arg("m"), py::arg("out_first"), py::arg("out_count"),
             py::arg("red_off"), py::arg("red_len"),
             py::arg("v"), py::arg("vn"), py::arg("y"), py::arg("yn"),
             py::arg("mode") = (int)PIM_ACC_SINGLE)
        .def("stats", &Runtime::stats)
        .def("stats_reset", &Runtime::stats_reset)
        .def("outputs", [](const Runtime &r, uint32_t n) {
            return pim_op_outputs(r.rt, n);
        }, py::arg("out_count"));
}
