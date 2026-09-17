`timescale 1ns / 1ps
`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: KETI
// Engineer: Jeongyoon Kang
//
// Module Name: emu_gpr_wrap
// Project Name: Physical-timing PIM emulator — emulator_top BD-assembly glue
// Target Devices: VHK158, V80
//
// Description:
//   BD-assembly glue for the GPR side of the front end.  Same role that
//   emu_gb_wrap plays for global_bus: it packages a memory plus the wiring that
//   a block design cannot express as loose continuous assignments, so the whole
//   thing drops into the BD as ONE cell.
//
//   Contents:
//     gpr              — the 16K x 256b scratchpad (single command port)
//     host AXI4 slave  — the host's own way in, read AND write
//     command-port mux — the GPR has ONE command port and TWO MASTERS; this
//                        arbitrates them by priority (gpr.v header: "MC<->host
//                        ARBITRATION IS NOT HERE. A wrapper ... muxes the two
//                        requesters into this single command port")
//     read-return demux — one read-data output, two readers, one ownership bit
//     kernel-active latch — see below
//
//   EXACTLY TWO MASTERS, and that is the point.  The GPR used to be reached three
//   ways: the dispatcher wrote it, a gpr_read_bridge living HERE read it for the
//   MC's WRVEC fill, and the host wrote it through the dispatcher's A[19] window.
//   So RD_MAC went MC -> dispatcher -> GPR while WRVEC went MC <- GPR directly --
//   the two directions did not match, and "the dispatcher owns the GPR"
//   (dispatcher_top.v:352) was only half true.  The bridge now lives in
//   dispatcher_top, so BOTH directions are MC <-> dispatcher <-> GPR and this
//   cell sees dispatcher and host, nothing else.  With NC channels that stays
//   two: the channel fabric folds NC MCs into the one dispatcher port.
//
//   WHY THE LATCH STILL LIVES HERE.  o_pim_active used to be the select for the
//   port mux above; since RD_MAC's GPR landing (spec v2.6 §3e) that mux is a
//   plain priority and needs no state, so the latch's only consumer is now the
//   MC's HOST_PORT mux (i_pim_active, spec §6e).  It stays here because it is
//   built from the dispatcher signals that arrive at this cell anyway, and
//   moving it would only add a port.
//     set   : on the doorbell pulse (kernel start)
//     clear : when the fetcher goes idle (whole program issued)
//   Set has priority, so a doorbell arriving while ~i_fetch_busy still arms it.
//   With a program terminated by ISR_OP_EOS the clear lands after the last real
//   ISR retired -- see pim_isr_defs.vh.
//
//   Port naming: i_/o_ for scalars, m_gpr_* for the AXI-Stream the MC consumes
//   (_PIM_CONTRACT §4).  The dispatcher-facing command port keeps the
//   dispatcher's own names (s_disp_cmd_*) so the BD net is name-obvious.
//////////////////////////////////////////////////////////////////////////////////

module emu_gpr_wrap #(
    parameter integer AW = 17,     // GPR word-index width (128K entries)
    parameter integer DW = 256     // GPR word / s_gpr beat width
)(
    // s_axi must be listed here too, or Vivado leaves the host slave with no
    // associated clock and validate_bd_design fails on a FREQ_HZ mismatch.
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET rst_n" *)
    input  wire           clk,
    input  wire           rst_n,

    // ---- kernel-active latch source (from dispatcher_top) ----
    input  wire           i_doorbell,      // 1-cycle pulse: kernel start
    input  wire           i_fetch_busy,    // level: fetcher still issuing
    output wire           o_pim_active,    // -> MC i_pim_active, and the mux select

    // ---- dispatcher host-load path : GPR write (program/vector load) ----
    input  wire           s_disp_cmd_valid,
    output wire           s_disp_cmd_ready,
    input  wire           s_disp_cmd_we,
    input  wire [AW-1:0]  s_disp_cmd_addr,
    input  wire [DW-1:0]  s_disp_cmd_wdata,

    // ---- dispatcher read return (the dispatcher now reads too) ----
    //   The WRVEC fill used to be served by a gpr_read_bridge sitting in THIS
    //   cell, which made the MC a third master on the GPR and left the
    //   dispatcher owning only half of it.  The bridge moved into
    //   dispatcher_top, so the dispatcher issues both directions and this cell
    //   sees exactly two masters: dispatcher and host.
    output wire           s_disp_rd_valid,
    input  wire           s_disp_rd_ready,
    output wire [DW-1:0]  s_disp_rd_data,

    // ---- host AXI4 slave : direct GPR read/write ------------------------------
    //   The dispatcher's AXI4-Full slave can WRITE the GPR (it owns the write
    //   path) but its AR/R is a zero-return stub, so on hardware there was no way
    //   to read a result back -- simulation reads the array hierarchically, which
    //   a board cannot do.  This port is that missing read path, and it comes in
    //   beside the dispatcher rather than through it: routing host reads through
    //   the dispatcher would have meant a read-return path across a cell that
    //   exists to fetch instructions.
    //
    //   4 MiB window = 128K words x 32 B.  Word index is AWADDR[21:5] -- the
    //   same <<5 the dispatcher's IMEM window uses, so one host-side address
    //   helper serves both; only the field is wider.  INCR only, full-width
    //   beats only (no narrow, no WRAP): the GPR is a word array and a partial
    //   word has no meaning here -- note there is no WSTRB port at all, so a
    //   host write always lands on a whole 32 B word.
    input  wire           s_axi_awvalid,
    output wire           s_axi_awready,
    input  wire [21:0]    s_axi_awaddr,
    input  wire [7:0]     s_axi_awlen,
    input  wire [3:0]     s_axi_awid,
    input  wire           s_axi_wvalid,
    output wire           s_axi_wready,
    input  wire [DW-1:0]  s_axi_wdata,
    input  wire           s_axi_wlast,
    output wire           s_axi_bvalid,
    input  wire           s_axi_bready,
    output wire [3:0]     s_axi_bid,
    output wire [1:0]     s_axi_bresp,
    input  wire           s_axi_arvalid,
    output wire           s_axi_arready,
    input  wire [21:0]    s_axi_araddr,
    input  wire [7:0]     s_axi_arlen,
    input  wire [3:0]     s_axi_arid,
    output wire           s_axi_rvalid,
    input  wire           s_axi_rready,
    output wire [DW-1:0]  s_axi_rdata,
    output wire           s_axi_rlast,
    output wire [3:0]     s_axi_rid,
    output wire [1:0]     s_axi_rresp
);

    // =====================================================================
    // kernel-active latch
    // =====================================================================
    reg r_pim_active;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)             r_pim_active <= 1'b0;
        else if (i_doorbell)    r_pim_active <= 1'b1;   // priority: set at kernel start
        else if (!i_fetch_busy) r_pim_active <= 1'b0;   // clear once the program is issued
    end
    assign o_pim_active = r_pim_active;

    // =====================================================================
    // GPR command port : one port, two masters, PRIORITY (dispatcher first)
    //
    //   Unlike the earlier three-requester version these two CAN genuinely
    //   collide -- the host is free to touch the GPR at any time, including
    //   mid-kernel, because it is a bus and nothing sequences it against the ISR
    //   stream.  So the priority here really does arbitrate, and the loser really
    //   does stall.  That is why the host is the loser (see below).
    //
    //   r_pim_active is not part of this mux; it only selects the MC's HOST_PORT
    //   (spec §6e).
    // =====================================================================
    wire          gpr_cmd_valid, gpr_cmd_ready, gpr_cmd_we;
    wire [AW-1:0] gpr_cmd_addr;
    wire [DW-1:0] gpr_cmd_wdata;

    // TWO masters, pure priority: dispatcher > host.
    //   The host is last on purpose.  A stalled host transaction is visible on
    //   the bus and the host can wait; the dispatcher cannot -- it is serving the
    //   MC, whose WRVEC fill starves mid-flight and whose RD_MAC parks in
    //   EC_P_RD_GPRW with nowhere to put the result.  Backpressure that shows up
    //   beats backpressure that hangs.
    wire          hst_cmd_valid, hst_cmd_ready, hst_cmd_we;
    wire [AW-1:0] hst_cmd_addr;
    wire [DW-1:0] hst_cmd_wdata;

    wire dsp_grant = s_disp_cmd_valid;
    wire hst_grant = ~dsp_grant & hst_cmd_valid;

    assign gpr_cmd_valid = s_disp_cmd_valid | hst_cmd_valid;
    assign gpr_cmd_we    = dsp_grant ? s_disp_cmd_we    : hst_cmd_we;
    assign gpr_cmd_addr  = dsp_grant ? s_disp_cmd_addr  : hst_cmd_addr;
    assign gpr_cmd_wdata = dsp_grant ? s_disp_cmd_wdata : hst_cmd_wdata;
    // ready goes to the GRANTED master only, so the other cannot consume an
    // accept that was meant for it.
    assign s_disp_cmd_ready = dsp_grant & gpr_cmd_ready;
    assign hst_cmd_ready    = hst_grant & gpr_cmd_ready;

    wire          gpr_mrd_valid, gpr_mrd_ready;
    wire [DW-1:0] gpr_mrd_data;

    // ---- read-return ownership -----------------------------------------------
    //   The GPR has ONE read-data output and TWO readers (dispatcher, host).  It
    //   accepts a new command only when that output is empty or draining this
    //   cycle (gpr.v:63), so at most one read is ever in flight -- one bit is
    //   therefore enough to remember whose it is.  Without it a host read would
    //   be handed to the dispatcher and injected into a WRVEC vector, which is a
    //   silent wrong answer rather than a visible fault.
    localparam OWN_DSP = 1'b0, OWN_HST = 1'b1;
    reg r_rd_owner;
    wire gpr_rd_accept = gpr_cmd_valid & gpr_cmd_ready & ~gpr_cmd_we;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)             r_rd_owner <= OWN_DSP;
        else if (gpr_rd_accept) r_rd_owner <= dsp_grant ? OWN_DSP : OWN_HST;
    end

    wire hst_rd_valid = gpr_mrd_valid & (r_rd_owner == OWN_HST);
    wire hst_rd_ready;
    assign s_disp_rd_valid = gpr_mrd_valid & (r_rd_owner == OWN_DSP);
    assign s_disp_rd_data  = gpr_mrd_data;
    assign gpr_mrd_ready   = (r_rd_owner == OWN_DSP) ? s_disp_rd_ready : hst_rd_ready;

    gpr #(.AW(AW), .DW(DW)) u_gpr (
        .clk(clk), .rst_n(rst_n),
        .s_cmd_valid(gpr_cmd_valid), .s_cmd_ready(gpr_cmd_ready),
        .s_cmd_we(gpr_cmd_we), .s_cmd_addr(gpr_cmd_addr), .s_cmd_wdata(gpr_cmd_wdata),
        .m_rd_valid(gpr_mrd_valid), .m_rd_ready(gpr_mrd_ready), .m_rd_data(gpr_mrd_data)
    );


    // =====================================================================
    // Host AXI4 slave -> GPR command port
    //
    //   Deliberately unpipelined: one beat is issued, its data waited for, then
    //   the next.  A host readback is a handful of words at the end of a kernel,
    //   so throughput is worth nothing here and every cycle of extra state is a
    //   place for the GPR's single read-return to be mis-owned.  Write and read
    //   never run at once (separate states), which is what keeps hst_cmd_* a
    //   single requester on the mux above.
    // =====================================================================
    localparam H_IDLE = 3'd0, H_WDATA = 3'd1, H_BRESP = 3'd2,
               H_RCMD = 3'd3, H_RDATA = 3'd4;
    reg [2:0]     h_state;
    reg [AW-1:0]  h_addr;
    reg [7:0]     h_left;      // beats remaining after the current one
    reg [3:0]     h_id;

    wire h_aw_hs = s_axi_awvalid & s_axi_awready;
    wire h_ar_hs = s_axi_arvalid & s_axi_arready;
    wire h_w_hs  = s_axi_wvalid  & s_axi_wready;
    wire h_r_hs  = s_axi_rvalid  & s_axi_rready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_state <= H_IDLE;  h_addr <= {AW{1'b0}};  h_left <= 8'd0;  h_id <= 4'd0;
        end else case (h_state)
            H_IDLE: begin
                // AW first when both arrive: a write in flight is holding host
                // data that has nowhere else to sit.
                if (h_aw_hs) begin
                    h_addr <= s_axi_awaddr[21:5];  h_left <= s_axi_awlen;
                    h_id   <= s_axi_awid;          h_state <= H_WDATA;
                end else if (h_ar_hs) begin
                    h_addr <= s_axi_araddr[21:5];  h_left <= s_axi_arlen;
                    h_id   <= s_axi_arid;          h_state <= H_RCMD;
                end
            end
            H_WDATA: if (h_w_hs) begin
                h_addr <= h_addr + 1'b1;
                if (s_axi_wlast) h_state <= H_BRESP;
                else             h_left  <= h_left - 1'b1;
            end
            H_BRESP: if (s_axi_bvalid & s_axi_bready) h_state <= H_IDLE;
            // issue one read command, then wait for its data
            H_RCMD:  if (hst_cmd_valid & hst_cmd_ready) h_state <= H_RDATA;
            H_RDATA: if (h_r_hs) begin
                h_addr <= h_addr + 1'b1;
                if (h_left == 8'd0) h_state <= H_IDLE;
                else begin h_left <= h_left - 1'b1; h_state <= H_RCMD; end
            end
            default: h_state <= H_IDLE;
        endcase
    end

    assign s_axi_awready = (h_state == H_IDLE);
    assign s_axi_arready = (h_state == H_IDLE) & ~s_axi_awvalid;   // AW wins the tie
    assign s_axi_wready  = (h_state == H_WDATA) & hst_cmd_ready;
    assign s_axi_bvalid  = (h_state == H_BRESP);
    assign s_axi_bid     = h_id;
    assign s_axi_bresp   = 2'b00;                                  // OKAY
    assign s_axi_rvalid  = (h_state == H_RDATA) & hst_rd_valid;
    assign s_axi_rdata   = gpr_mrd_data;
    assign s_axi_rlast   = (h_state == H_RDATA) & (h_left == 8'd0);
    assign s_axi_rid     = h_id;
    assign s_axi_rresp   = 2'b00;
    assign hst_rd_ready  = (h_state == H_RDATA) & s_axi_rready;

    assign hst_cmd_valid = ((h_state == H_WDATA) & s_axi_wvalid) | (h_state == H_RCMD);
    assign hst_cmd_we    =  (h_state == H_WDATA);
    assign hst_cmd_addr  = h_addr;
    assign hst_cmd_wdata = s_axi_wdata;

endmodule

`default_nettype wire
