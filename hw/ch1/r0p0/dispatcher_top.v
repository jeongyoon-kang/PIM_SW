`timescale 1ns / 1ps
`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: KETI
// Engineer: Jeongyoon Kang
//
// Module Name: dispatcher_top
// Project Name: Physical-timing PIM emulator - command_dispatcher top
// Target Devices: VHK158, V80
//
// Description:
//   command_dispatcher top.  Two AXI slaves + the fetch path (DESIGN_NOTES):
//
//   1) AXI4-Full slave  (512KB window, 256b data) - PROGRAM LOAD ONLY:
//        instruction_memory (INTERNAL instance), word index = A[18:5] (14b).
//      INCR bursts increment the word index per beat.
//      This window used to be 1MB with A[19] selecting IMEM vs GPR.  The GPR now
//      has its own AXI slave on emu_gpr_wrap, so the host reaches it there and
//      this port carries instructions and nothing else -- one window, one meaning.
//      AR/R host readback is a zero-return stub; IMEM readback was never used and
//      a GPR result is now read over the GPR's own slave.
//
//   1b) GPR, both directions, on behalf of the MC.  BOTH are addressed from the
//      ISR at the instant it is issued, so the MC never handles a GPR address:
//        RD_MAC  the MC hands the readout up (s_res_*, DATA only) and this top
//                writes it to GPR[r_res_addr] -- ISR[22:6], latched at issue
//        WRVEC   this top starts u_gpr_bridge at issue with ISR[22:6] as the base
//                and ISR[58:49] (OPSIZE) as the count; the bridge walks those
//                addresses and streams the vector down (m_gpr_*)
//      Before, RD_MAC came through here but WRVEC was served by a bridge inside
//      emu_gpr_wrap, which made the MC a third master on the GPR and left "the
//      dispatcher owns the GPR" only half true.  With the bridge moved in here
//      the GPR sees exactly two masters, this one and the host -- and it stays
//      two however many channels are added, because the channel fabric folds
//      them into this single port.
//
//   2) AXI4-Lite slave  (CFR) - doorbell / status + timing-parameter fan-out:
//        0x00 CTRL   W  [0]=doorbell(self-clear) [1]=mode(0 NORMAL / 1 ALL_BANK)
//        0x04 STATUS R  [3:0]=fsm_state [21:4]=RESERVED, reads 0 [31]=done
//                       ([4] inv_sticky / [7:5] last_inv_kind / [21:8] last_inv_idx
//                        held the validity-gate report; the gate is parked - see
//                        below - and the bit positions are kept free for it)
//        0x08 T_FAW  W    0x0C T_RRD  W                        (-> MC)
//        0x10 T_RCD  W    0x14 T_CCD  W   0x18 T_RTP W
//        0x1C T_RP   W    0x20 T_WR   W   0x24 T_RAS W          (-> bank, 16 common)
//        0x28 PROG_LEN W  ISR count (fetch termination)
//      Writing CTRL[0]=1 emits a 1-cycle doorbell pulse (self-clear; never stored)
//      that launches the fetcher and also strobes o_doorbell for measurement-ip
//      (t_doorbell latch).
//
//   The internal IMEM command port is shared by the host-write path and the
//   fetch-read path; they are temporally disjoint (load, then doorbell, then
//   fetch), so a fetch-priority mux suffices.
//
//   VALIDITY GATE - PARKED 2026-07-29 (timing; see fetch_decode.v header).  The
//   fetcher no longer checks anything: every word in IMEM reaches the MC.  Nothing
//   downstream re-checks either, so the SOFTWARE that builds the program must
//   guarantee CH_MASK + V1-V4.  o_decode_invalid / o_inv_kind / o_inv_idx are kept
//   as ports (tied off by fetch_decode) so the block-design port list does not
//   move; the STATUS bits they fed now read 0.  The gate itself is parked at
//   ../../validity_gate/ - its README lists what to redo before reviving it.
//
//   OUT OF SCOPE THIS STEP: CH_MASK multi-channel fan-out, GPR<->MC WRVEC read mux
//   (integration step).  See fetch_decode + _PIM_CONTRACT.
//////////////////////////////////////////////////////////////////////////////////

module dispatcher_top #(
    // 512KB window = 16K words x 32B, which is the whole of IMEM.  It was 20 (1MB)
    // while this slave carried BOTH memories and A[19] chose between them; since
    // the GPR moved to its own AXI slave the top bit has no reader, and leaving
    // the port a bit wider than the decode only invites a 1MB segment that
    // aliases IMEM onto itself.  Width now equals the addressable window.
    parameter integer FULL_ADDR_WIDTH = 19,    // 512KB IMEM window (awaddr[18:5] = word)
    parameter integer FULL_DATA_WIDTH = 256,
    parameter integer FULL_ID_WIDTH   = 4,
    parameter integer LITE_ADDR_WIDTH = 8,
    parameter integer LITE_DATA_WIDTH = 32,
    parameter integer IMEM_AW         = 14,    // IMEM: 16K words = 512KB
    // GPR word index.  Its own parameter, not IMEM_AW: the two memories are
    // different sizes now (GPR 128K words = 4 MiB) and sharing one width silently
    // truncated every GPR address above 16K.  It must equal ISR_ROW_W, since a
    // GPR address arrives as the ISR's ROW/GPR_ADDR field.
    parameter integer GPR_AW          = 17,    // GPR: 128K words = 4 MiB
    parameter integer TW              = 8      // timing-parameter width (MC/bank ports)
)(
    // clk drives both AXI slaves; rst_n is their active-low reset.  Vivado infers
    // s_axi (AXI4) and s_lite (AXI4-Lite) from the port names on its own, but
    // WITHOUT them being listed here it leaves them on the default FREQ_HZ
    // 100000000 while a 200 MHz block-design clock drives the cell -- connect_bd_
    // intf_net then hits a FREQ_HZ propagation mismatch against the VIP master.
    // Same reason as emulator_controller.v:69-77.
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_lite:m_gpr:s_res, ASSOCIATED_RESET rst_n" *)
    input  wire                          clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                          rst_n,

    // ================= AXI4-Full slave (program / vector load) =================
    // write address
    input  wire                          s_axi_awvalid,
    output wire                          s_axi_awready,
    input  wire [FULL_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire [7:0]                    s_axi_awlen,
    input  wire [2:0]                    s_axi_awsize,
    input  wire [1:0]                    s_axi_awburst,
    input  wire [FULL_ID_WIDTH-1:0]      s_axi_awid,
    // write data
    input  wire                          s_axi_wvalid,
    output wire                          s_axi_wready,
    input  wire [FULL_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [(FULL_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                          s_axi_wlast,
    // write response
    output wire                          s_axi_bvalid,
    input  wire                          s_axi_bready,
    output wire [FULL_ID_WIDTH-1:0]      s_axi_bid,
    output wire [1:0]                    s_axi_bresp,
    // read address
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    input  wire [FULL_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire [7:0]                    s_axi_arlen,
    input  wire [2:0]                    s_axi_arsize,
    input  wire [1:0]                    s_axi_arburst,
    input  wire [FULL_ID_WIDTH-1:0]      s_axi_arid,
    // read data
    output wire                          s_axi_rvalid,
    input  wire                          s_axi_rready,
    output wire [FULL_DATA_WIDTH-1:0]    s_axi_rdata,
    output wire [FULL_ID_WIDTH-1:0]      s_axi_rid,
    output wire                          s_axi_rlast,
    output wire [1:0]                    s_axi_rresp,

    // ===================== AXI4-Lite slave (CFR) ==============================
    input  wire                          s_lite_awvalid,
    output wire                          s_lite_awready,
    input  wire [LITE_ADDR_WIDTH-1:0]    s_lite_awaddr,
    input  wire                          s_lite_wvalid,
    output wire                          s_lite_wready,
    input  wire [LITE_DATA_WIDTH-1:0]    s_lite_wdata,
    input  wire [(LITE_DATA_WIDTH/8)-1:0] s_lite_wstrb,
    output wire                          s_lite_bvalid,
    input  wire                          s_lite_bready,
    output wire [1:0]                    s_lite_bresp,
    input  wire                          s_lite_arvalid,
    output wire                          s_lite_arready,
    input  wire [LITE_ADDR_WIDTH-1:0]    s_lite_araddr,
    output wire                          s_lite_rvalid,
    input  wire                          s_lite_rready,
    output wire [LITE_DATA_WIDTH-1:0]    s_lite_rdata,
    output wire [1:0]                    s_lite_rresp,

    // =============== external GPR command port (write path) ===================
    output wire                          o_gpr_cmd_valid,
    input  wire                          i_gpr_cmd_ready,
    output wire                          o_gpr_cmd_we,
    output wire [GPR_AW-1:0]             o_gpr_cmd_addr,
    output wire [FULL_DATA_WIDTH-1:0]    o_gpr_cmd_wdata,
    // GPR read return.  The dispatcher reads the GPR now as well as writing it --
    // see the WRVEC block below for why the read bridge moved in here.
    input  wire                          i_gpr_rd_valid,
    output wire                          o_gpr_rd_ready,
    input  wire [FULL_DATA_WIDTH-1:0]    i_gpr_rd_data,

    // ---- MC WRVEC vector service ----
    //   The MC consumes a vector as a plain stream; the sequencing behind it is
    //   gpr_read_bridge, instantiated below.  The bridge lived in emu_gpr_wrap
    //   once, which made the MC a third master on the GPR while RD_MAC already
    //   came back through here -- the two directions did not match.  Both now go
    //   MC <-> dispatcher <-> GPR.
    //
    //   The MC no longer asks.  It used to hand the request back as a level plus
    //   an address (i_gpr_rd_en / i_gpr_rd_addr), which meant the GPR word index
    //   travelled down inside the ISR and then straight back up -- the same
    //   number in two places, with two chances to be wrong.  This top starts the
    //   bridge from the ISR it is issuing instead, the same instant and the same
    //   word it already reads for r_res_addr.  Nothing on the MC's boundary
    //   carries a GPR address in either direction now.
    output wire [FULL_DATA_WIDTH-1:0]    m_gpr_tdata,
    output wire                          m_gpr_tvalid,
    input  wire                          m_gpr_tready,

    // ---- RD_MAC result landing (from the MC, up the Data link) ----
    //   spec v2.6 §3e: the 16 x BF16 readout is written to GPR[GPR_ADDR] with no
    //   transform.  The MC holds valid until this accepts; the address rides
    //   along, so nothing here has to remember which ISR asked.
    //   Result landing from the MC.  DATA ONLY -- the destination is ISR[19:6] of
    //   the ISR this top handed down, so it is latched here at issue rather than
    //   carried back.  Safe because the MC's s_isr_ready is low from accept until
    //   it retires (emulator_controller.v:855), so the next ISR cannot be issued
    //   before this result arrives and r_res_addr still belongs to it.
    //   Named for what it carries, not for RD_MAC: RD_AF will use the same path.
    //   Carried as AXI4-Stream (s_res) so the block design links it as one
    //   interface rather than three loose nets, matching m_gpr on this module.
    input  wire                          s_res_tvalid,
    output wire                          s_res_tready,
    input  wire [FULL_DATA_WIDTH-1:0]    s_res_tdata,

    // ===================== ISR stream to emulator MC ==========================
    output wire [FULL_DATA_WIDTH-1:0]    s_isr_data,
    output wire                          s_isr_valid,
    input  wire                          s_isr_ready,

    // ================= CFR distributed timing / mode outputs ==================
    output wire [TW-1:0]                 o_t_faw,     // -> MC
    output wire [TW-1:0]                 o_t_rrd,     // -> MC
    output wire                          o_mode,      // -> MC (0 NORMAL / 1 ALL_BANK)
    output wire [TW-1:0]                 o_t_rcd,     // -> bank (16 common)
    output wire [TW-1:0]                 o_t_ccd,
    output wire [TW-1:0]                 o_t_rtp,
    output wire [TW-1:0]                 o_t_rp,
    output wire [TW-1:0]                 o_t_wr,
    output wire [TW-1:0]                 o_t_ras,

    // ============= doorbell strobe to measurement-ip (t_doorbell) =============
    output wire                          o_doorbell,

    // ===== decode-invalid report to measurement-ip (3-party contract, §5) =====
    //   validity_gate dropped a fetched ISR (spec v2.5 §5): it never reaches the
    //   MC.  strobe = 1 cycle ; kind = which V failed ; idx = ISR index (PC).
    output wire                          o_decode_invalid,
    output wire [2:0]                    o_inv_kind,
    output wire [IMEM_AW-1:0]            o_inv_idx,

    // ===== fetch-busy level (system integration — emulator_top i_pim_active) ====
    //   High from a doorbell until the fetcher has issued the whole program (the
    //   last ISR accepted onto s_isr).  emulator_top latches i_pim_active on the
    //   doorbell pulse and clears it when this falls, so the PIM engine owns the
    //   emulator across the whole kernel while the MC r_mode drain handles the
    //   post-fetch tail (계약 §0b).  A doorbell-priority latch mirrors the fetch
    //   lifecycle so a relaunch from a stuck DONE re-arms cleanly.
    output wire                          o_fetch_busy
);

    // ISR field positions.  This top does not decode ISRs -- that is the MC's job
    // -- but it does latch ONE field: the ROW/GPR_ADDR of the ISR it is handing
    // down, so it knows where a result will land.  Reading the position from the
    // shared header rather than re-typing the bits is what keeps that latch in
    // step with the decoder when the layout moves.
    `include "pim_isr_defs.vh"

    // ======================================================================
    // CFR registers (AXI4-Lite)
    // ======================================================================
    reg [TW-1:0]      r_t_faw, r_t_rrd, r_t_rcd, r_t_ccd, r_t_rtp, r_t_rp, r_t_wr, r_t_ras;
    reg               r_mode;
    reg [IMEM_AW-1:0] r_prog_len;
    reg               r_doorbell;    // 1-cycle self-clearing pulse

    // ---- Lite write channel : AW and W latched INDEPENDENTLY, single-beat ----
    //   AW and W are independent AXI channels: the master may drive either one
    //   first and it must not wait for the other.  Each channel is captured on
    //   its OWN handshake and held; the register write commits (and B is issued)
    //   only when both halves are present.  A channel already held drops its
    //   READY so the same transfer cannot be accepted twice; both holds are
    //   released on the B handshake, which re-opens the pair for the next write.
    reg                       r_lite_bvalid;
    reg                       r_aw_hold;      // AW captured, waiting for W
    reg [LITE_ADDR_WIDTH-1:0] r_aw_addr;
    reg                       r_w_hold;       // W captured, waiting for AW
    reg [LITE_DATA_WIDTH-1:0] r_w_data;

    // READY low while that half is already held, or while B is still outstanding
    assign s_lite_awready = ~r_aw_hold & ~r_lite_bvalid;
    assign s_lite_wready  = ~r_w_hold  & ~r_lite_bvalid;
    assign s_lite_bvalid  = r_lite_bvalid;
    assign s_lite_bresp   = 2'b00;

    wire lite_aw_hs = s_lite_awvalid & s_lite_awready;   // this-cycle AW transfer
    wire lite_w_hs  = s_lite_wvalid  & s_lite_wready;    // this-cycle W  transfer
    wire lite_aw_ok = r_aw_hold | lite_aw_hs;
    wire lite_w_ok  = r_w_hold  | lite_w_hs;
    // commit when both halves are available (held from earlier and/or this cycle)
    wire lite_wr_accept = lite_aw_ok & lite_w_ok & ~r_lite_bvalid;
    wire [LITE_ADDR_WIDTH-1:0] lite_wr_addr = r_aw_hold ? r_aw_addr : s_lite_awaddr;
    wire [LITE_DATA_WIDTH-1:0] lite_wr_data = r_w_hold  ? r_w_data  : s_lite_wdata;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_t_faw <= {TW{1'b0}}; r_t_rrd <= {TW{1'b0}};
            r_t_rcd <= {TW{1'b0}}; r_t_ccd <= {TW{1'b0}}; r_t_rtp <= {TW{1'b0}};
            r_t_rp  <= {TW{1'b0}}; r_t_wr  <= {TW{1'b0}}; r_t_ras <= {TW{1'b0}};
            r_mode      <= 1'b0;
            r_prog_len  <= {IMEM_AW{1'b0}};
            r_doorbell  <= 1'b0;
            r_lite_bvalid <= 1'b0;
            r_aw_hold   <= 1'b0;
            r_aw_addr   <= {LITE_ADDR_WIDTH{1'b0}};
            r_w_hold    <= 1'b0;
            r_w_data    <= {LITE_DATA_WIDTH{1'b0}};
        end else begin
            r_doorbell <= 1'b0;    // default: pulse lasts one cycle (self-clear)
            // AW / W capture : held until the pair commits
            if (lite_wr_accept) begin
                r_aw_hold <= 1'b0;
                r_w_hold  <= 1'b0;
            end else begin
                if (lite_aw_hs) begin r_aw_hold <= 1'b1; r_aw_addr <= s_lite_awaddr; end
                if (lite_w_hs)  begin r_w_hold  <= 1'b1; r_w_data  <= s_lite_wdata;  end
            end
            // B response handshake
            if (lite_wr_accept)                     r_lite_bvalid <= 1'b1;
            else if (r_lite_bvalid & s_lite_bready) r_lite_bvalid <= 1'b0;
            // register write decode (byte offset)
            if (lite_wr_accept) begin
                case (lite_wr_addr[7:0])
                    8'h00: begin r_mode <= lite_wr_data[1]; r_doorbell <= lite_wr_data[0]; end
                    8'h08: r_t_faw    <= lite_wr_data[TW-1:0];
                    8'h0C: r_t_rrd    <= lite_wr_data[TW-1:0];
                    8'h10: r_t_rcd    <= lite_wr_data[TW-1:0];
                    8'h14: r_t_ccd    <= lite_wr_data[TW-1:0];
                    8'h18: r_t_rtp    <= lite_wr_data[TW-1:0];
                    8'h1C: r_t_rp     <= lite_wr_data[TW-1:0];
                    8'h20: r_t_wr     <= lite_wr_data[TW-1:0];
                    8'h24: r_t_ras    <= lite_wr_data[TW-1:0];
                    8'h28: r_prog_len <= lite_wr_data[IMEM_AW-1:0];
                    default: ; // reserved: ignore
                endcase
            end
        end
    end

    // ---- decode-invalid report : pass-through only ----
    // fetch_decode ties these off (validity gate parked).  The sticky latch that
    // used to sit here was removed with it; STATUS[21:4] now reads a constant 0.
    wire               fd_decode_invalid;
    wire [2:0]         fd_inv_kind;
    wire [IMEM_AW-1:0] fd_inv_idx;

    assign o_decode_invalid = fd_decode_invalid;
    assign o_inv_kind       = fd_inv_kind;
    assign o_inv_idx        = fd_inv_idx;

    // ---- Lite read channel : STATUS + register read-back ----
    wire [3:0] fd_state;
    wire       fd_done;
    reg               r_lite_rvalid;
    reg [LITE_DATA_WIDTH-1:0] r_lite_rdata;
    assign s_lite_arready = ~r_lite_rvalid;
    assign s_lite_rvalid  = r_lite_rvalid;
    assign s_lite_rdata   = r_lite_rdata;
    assign s_lite_rresp   = 2'b00;
    wire lite_rd_accept = s_lite_arvalid & ~r_lite_rvalid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_lite_rvalid <= 1'b0;
            r_lite_rdata  <= {LITE_DATA_WIDTH{1'b0}};
        end else begin
            if (lite_rd_accept) begin
                r_lite_rvalid <= 1'b1;
                case (s_lite_araddr[7:0])
                    8'h00: r_lite_rdata <= {30'b0, r_mode, 1'b0};
                    // STATUS: [3:0]=state [21:4]=reserved(0, ex validity gate) [31]=done
                    8'h04: r_lite_rdata <= {fd_done, 9'b0, 18'b0, fd_state};
                    8'h08: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_faw};
                    8'h0C: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_rrd};
                    8'h10: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_rcd};
                    8'h14: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_ccd};
                    8'h18: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_rtp};
                    8'h1C: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_rp};
                    8'h20: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_wr};
                    8'h24: r_lite_rdata <= {{(LITE_DATA_WIDTH-TW){1'b0}}, r_t_ras};
                    8'h28: r_lite_rdata <= {{(LITE_DATA_WIDTH-IMEM_AW){1'b0}}, r_prog_len};
                    default: r_lite_rdata <= {LITE_DATA_WIDTH{1'b0}};
                endcase
            end else if (r_lite_rvalid & s_lite_rready) begin
                r_lite_rvalid <= 1'b0;
            end
        end
    end

    // CFR fan-out
    assign o_t_faw = r_t_faw;  assign o_t_rrd = r_t_rrd;  assign o_mode  = r_mode;
    assign o_t_rcd = r_t_rcd;  assign o_t_ccd = r_t_ccd;  assign o_t_rtp = r_t_rtp;
    assign o_t_rp  = r_t_rp;   assign o_t_wr  = r_t_wr;   assign o_t_ras = r_t_ras;
    assign o_doorbell = r_doorbell;

    // ======================================================================
    // AXI4-Full write path : A[19] branch -> internal IMEM or external GPR
    // ======================================================================
    localparam [1:0] WS_IDLE = 2'd0, WS_DATA = 2'd1, WS_RESP = 2'd2;
    reg [1:0]            ws;
    reg [IMEM_AW-1:0]   r_word_addr;    // current 256b word index
    reg [FULL_ID_WIDTH-1:0] r_bid;

    // ---- the one instant this module reads the ISR it is handing over --------
    //   BOTH GPR directions are addressed from here, off the same handshake, so
    //   the MC never has to know a GPR word index (see its s_gpr port comment).
    //   This is deliberately NOT in the fetch path: the validity gate was parked
    //   for timing on 2026-07-29 (WNS -14.136, 계약 §1d) and nothing new belongs
    //   in front of the fetcher.  It sits behind the issue handshake instead --
    //   one 5-bit opcode compare and two field slices, on an already-registered
    //   boundary.
    wire w_isr_hs = s_isr_valid & s_isr_ready;

    // Destination for the next result the MC sends up.  ISR[22:6] is ROW for a
    // DRAM op and GPR_ADDR for RD_MAC/WRVEC; capturing it unconditionally is
    // harmless because it is only USED when a result lands, and only RD_MAC
    // produces one.
    reg [GPR_AW-1:0] r_res_addr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          r_res_addr <= {GPR_AW{1'b0}};
        else if (w_isr_hs)   r_res_addr <= s_isr_data[ISR_ROW_LSB +: ISR_ROW_W];
    end

    // ---- WRVEC : start the vector walk out of the SAME handshake -------------
    //   The MC used to ask for this with a level and an address once it reached
    //   WFILL.  It cannot ask any more, so the request is built here from the
    //   word being issued: base = ISR[22:6], count = ISR[58:49] (OPSIZE) -- the
    //   same OPSIZE the MC's own fill counter loads, so the two counts come from
    //   one field of one word and cannot disagree.
    //
    //   Starting BEFORE the MC is in WFILL is fine and is the intended order: the
    //   bridge presents the first beat and holds it, and s_gpr_tready rises a
    //   couple of cycles later.  It is an ordinary handshake -- valid may lead
    //   ready.
    //
    //   i_abort on every issue is the one guard the removed level used to give
    //   for free.  In the normal flow it is a no-op: an ISR is single-outstanding
    //   and the MC only re-raises s_isr_ready after the fill's WSAVE/DONE, so the
    //   bridge is already idle by then.  It only bites when the MC never took the
    //   vector, and it stops that from shifting the NEXT vector by a beat.
    wire w_isr_is_wrvec = (s_isr_data[ISR_OPCODE_LSB +: ISR_OPCODE_W] == ISR_OP_WRVEC);
    wire w_brg_start    = w_isr_hs & w_isr_is_wrvec;
    wire w_brg_abort    = w_isr_hs & ~w_isr_is_wrvec;

    // WRVEC read bridge <-> GPR command port
    wire              brg_cmd_valid, brg_cmd_ready;
    wire [GPR_AW-1:0] brg_cmd_addr;

    // fetch read request into the shared IMEM command port
    wire              fd_cmd_valid;
    wire [IMEM_AW-1:0] fd_cmd_addr;

    // IMEM command port (shared: fetch read has priority over host write)
    wire              imem_cmd_valid;
    wire              imem_cmd_ready;
    wire              imem_cmd_we;
    wire [IMEM_AW-1:0] imem_cmd_addr;
    wire [FULL_DATA_WIDTH-1:0] imem_cmd_wdata;
    wire              imem_rd_valid;
    wire              imem_rd_ready;
    wire [FULL_DATA_WIDTH-1:0] imem_rd_data;

    // s_axi is IMEM-ONLY.  The host used to reach the GPR through this same
    // window with A[19] selecting between them; the GPR now has its own AXI slave
    // on emu_gpr_wrap, so this port carries instructions and nothing else.  One
    // window, one meaning.
    wire imem_wr_req = (ws == WS_DATA) & s_axi_wvalid;

    // host IMEM write can only take the port when the fetcher is not reading it
    wire imem_wr_ready = imem_cmd_ready & ~fd_cmd_valid;

    assign imem_cmd_valid = fd_cmd_valid | imem_wr_req;
    assign imem_cmd_we    = imem_wr_req & ~fd_cmd_valid;
    assign imem_cmd_addr  = fd_cmd_valid ? fd_cmd_addr : r_word_addr;
    assign imem_cmd_wdata = s_axi_wdata;

    // ---- external GPR write : host load, or an RD_MAC result landing ----
    // The MC hands the RD_MAC readout UP the Data link with its GPR_ADDR attached
    // (spec v2.6 §3e), and the dispatcher -- which owns the GPR -- writes it.  No
    // ISR decode is needed here because the address travels with the data.
    //
    // The two sources never collide: a host load happens before the doorbell, an
    // RD_MAC landing during the kernel.  The result is given priority anyway, so
    // that if a host ever did write mid-kernel it would be the HOST that stalls,
    // not the result that is dropped -- a stalled host write is visible on the
    // bus, a dropped result is not.
    //   Two requesters on the GPR command port, both of them the MC's business:
    //     RD_MAC landing  write, address = the GPR_ADDR the ISR named
    //     WRVEC fill      read, address walked by the bridge
    //   They are different ISRs and the ISR stream is single outstanding, so they
    //   never ask in the same cycle.  The priority is there so a future overlap
    //   degrades into a stall instead of two accepts at once, and the write wins
    //   because a stalled RD_MAC parks the MC in EC_P_RD_GPRW.
    assign o_gpr_cmd_valid = s_res_tvalid | brg_cmd_valid;
    assign o_gpr_cmd_we    = s_res_tvalid;                     // read when the bridge wins
    assign o_gpr_cmd_addr  = s_res_tvalid ? r_res_addr : brg_cmd_addr;
    assign o_gpr_cmd_wdata = s_res_tdata;                      // the bridge never writes
    assign s_res_tready    = s_res_tvalid & i_gpr_cmd_ready;
    assign brg_cmd_ready   = ~s_res_tvalid & brg_cmd_valid & i_gpr_cmd_ready;

    assign s_axi_awready = (ws == WS_IDLE);
    assign s_axi_wready  = (ws == WS_DATA) & imem_wr_ready;
    assign s_axi_bvalid  = (ws == WS_RESP);
    assign s_axi_bid     = r_bid;
    assign s_axi_bresp   = 2'b00;

    wire w_beat_accept = s_axi_wvalid & s_axi_wready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ws          <= WS_IDLE;
            r_word_addr <= {IMEM_AW{1'b0}};
            r_bid       <= {FULL_ID_WIDTH{1'b0}};
        end else begin
            case (ws)
                WS_IDLE: begin
                    if (s_axi_awvalid) begin
                        r_word_addr <= s_axi_awaddr[18:5];
                        r_bid       <= s_axi_awid;
                        ws          <= WS_DATA;
                    end
                end
                WS_DATA: begin
                    if (w_beat_accept) begin
                        r_word_addr <= r_word_addr + 1'b1;   // INCR burst
                        if (s_axi_wlast) ws <= WS_RESP;
                    end
                end
                WS_RESP: begin
                    if (s_axi_bready) ws <= WS_IDLE;
                end
                default: ws <= WS_IDLE;
            endcase
        end
    end

    // ======================================================================
    // AXI4-Full read path : zero-return stub (host readback optional)
    // ======================================================================
    localparam [0:0] RS_IDLE = 1'b0, RS_DATA = 1'b1;
    reg               rs;
    reg [FULL_ID_WIDTH-1:0] r_rid;
    reg [7:0]         r_rbeats;
    assign s_axi_arready = (rs == RS_IDLE);
    assign s_axi_rvalid  = (rs == RS_DATA);
    assign s_axi_rdata   = {FULL_DATA_WIDTH{1'b0}};
    assign s_axi_rid     = r_rid;
    assign s_axi_rlast   = (rs == RS_DATA) & (r_rbeats == 8'd0);
    assign s_axi_rresp   = 2'b00;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs       <= RS_IDLE;
            r_rid    <= {FULL_ID_WIDTH{1'b0}};
            r_rbeats <= 8'd0;
        end else begin
            case (rs)
                RS_IDLE: if (s_axi_arvalid) begin
                             r_rid    <= s_axi_arid;
                             r_rbeats <= s_axi_arlen;
                             rs       <= RS_DATA;
                         end
                RS_DATA: if (s_axi_rready) begin
                             if (r_rbeats == 8'd0) rs <= RS_IDLE;
                             else r_rbeats <= r_rbeats - 8'd1;
                         end
                default: rs <= RS_IDLE;
            endcase
        end
    end

    // ======================================================================
    // Internal instruction memory + fetcher
    // ======================================================================
    instruction_memory #(
        .AW (IMEM_AW),
        .DW (FULL_DATA_WIDTH)
    ) u_imem (
        .clk        (clk),
        .rst_n      (rst_n),
        .s_cmd_valid(imem_cmd_valid),
        .s_cmd_ready(imem_cmd_ready),
        .s_cmd_we   (imem_cmd_we),
        .s_cmd_addr (imem_cmd_addr),
        .s_cmd_wdata(imem_cmd_wdata),
        .m_rd_valid (imem_rd_valid),
        .m_rd_ready (imem_rd_ready),
        .m_rd_data  (imem_rd_data)
    );

    // fetch-busy level : set on the doorbell pulse, cleared when the fetcher
    // reports the program fully issued (fd_done).  Doorbell has priority so a
    // relaunch from a stuck S_DONE re-arms in the same cycle.
    reg r_fetch_busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          r_fetch_busy <= 1'b0;
        else if (r_doorbell) r_fetch_busy <= 1'b1;
        else if (fd_done)    r_fetch_busy <= 1'b0;
    end
    assign o_fetch_busy = r_fetch_busy;

    fetch_decode #(
        .AW (IMEM_AW),
        .DW (FULL_DATA_WIDTH)
    ) u_fetch (
        .clk             (clk),
        .rst_n           (rst_n),
        .i_doorbell      (r_doorbell),
        .i_prog_len      (r_prog_len),
        .o_imem_cmd_valid(fd_cmd_valid),
        .i_imem_cmd_ready(imem_cmd_ready),
        .o_imem_cmd_addr (fd_cmd_addr),
        .i_imem_rd_valid (imem_rd_valid),
        .o_imem_rd_ready (imem_rd_ready),
        .i_imem_rd_data  (imem_rd_data),
        .o_isr_data      (s_isr_data),
        .o_isr_valid     (s_isr_valid),
        .i_isr_ready     (s_isr_ready),
        .o_decode_invalid(fd_decode_invalid),
        .o_inv_kind      (fd_inv_kind),
        .o_inv_idx       (fd_inv_idx),
        .o_state         (fd_state),
        .o_done          (fd_done)
    );

    // =========================================================================
    // WRVEC vector service.  Lives here, not in emu_gpr_wrap, so that BOTH GPR
    // directions are MC <-> dispatcher <-> GPR: RD_MAC already came back this
    // way.  The bridge itself is unchanged -- its interface was already
    // "request in / GPR command out / read data in / stream out", which is
    // exactly the shape this position needs.
    // =========================================================================
    gpr_read_bridge #(
        .AW (GPR_AW),
        .DW (FULL_DATA_WIDTH),
        .LW (ISR_OPSIZE_W)
    ) u_gpr_bridge (
        .clk(clk), .rst_n(rst_n),
        .i_start(w_brg_start),
        .i_addr (s_isr_data[ISR_ROW_LSB    +: ISR_ROW_W]),
        .i_len  (s_isr_data[ISR_OPSIZE_LSB +: ISR_OPSIZE_W]),
        .i_abort(w_brg_abort),
        .o_gpr_cmd_valid(brg_cmd_valid), .i_gpr_cmd_ready(brg_cmd_ready),
        .o_gpr_cmd_addr(brg_cmd_addr),
        .i_gpr_rd_valid(i_gpr_rd_valid), .o_gpr_rd_ready(o_gpr_rd_ready),
        .i_gpr_rd_data(i_gpr_rd_data),
        .o_gpr_tdata(m_gpr_tdata), .o_gpr_tvalid(m_gpr_tvalid), .i_gpr_tready(m_gpr_tready)
    );

endmodule

`default_nettype wire
