`timescale 1ns / 1ps
`default_nettype none
//////////////////////////////////////////////////////////////////////////////////
// Company: KETI
// Engineer: Jeongyoon Kang
//
// Module Name: emu_viol_csr
// Project Name: Physical-timing PIM emulator — emulator_top FPGA violation CSR
// Target Devices: VHK158, V80
//
// Description:
//   AXI4-Lite slave that makes the 16 banks' violation events readable from the
//   host.  FPGA-ONLY: in simulation the same events are read hierarchically or
//   off the waveform, so nothing like this is built into the simulation BD.
//
//   AGGREGATION, NOT A LOG.  Per bank and per violation kind it keeps exactly
//   three numbers -- the same shape bank_probe_csr.v:714-770 established:
//     sticky            did this kind ever fire
//     saturating count  how many times (holds at 0xFF, never wraps)
//     max overrun       the worst overshoot seen, in cycles
//   There is no per-event FIFO on purpose: a timing violation cannot be repaired
//   after the fact, and the only remedy is to re-run with a larger K.  "Did it
//   happen and by how much" is therefore the whole of what the host needs.
//
//   FIVE KINDS, NOT THREE.  bank_probe_csr predates the direction tag and takes
//   i_viol_kind as 2 bits.  bank_controller_top now emits BC_VIOL_KIND_W = 3 with
//   five kinds (bank_ctrl_defs.vh:100-107):
//     1 RCD_RD   2 CCD_RD   3 RCD_WR   4 CCD_WR   5 RECOVERY_WR
//   Folding read and write into one bucket would lose which direction broke, so
//   all five are kept apart here.
//
//   REGISTER MAP  (32b registers, byte addressed)
//     per bank b, base = b*0x40 :
//       +0x00  STICKY   [0] RCD_RD  [1] CCD_RD  [2] RCD_WR  [3] CCD_WR
//                       [4] RECOVERY_WR                     [8] ewmul_drop
//                       [31] any of the above
//       +0x04  CNT_A    [7:0] RCD_RD  [15:8] CCD_RD  [23:16] RCD_WR  [31:24] CCD_WR
//       +0x08  CNT_B    [7:0] RECOVERY_WR   [15:8] ewmul_drop
//       +0x0C  MAX_A    [7:0] RCD_RD  [15:8] CCD_RD  [23:16] RCD_WR  [31:24] CCD_WR
//       +0x10  MAX_B    [7:0] RECOVERY_WR
//     global :
//       0x400  CTRL     [0] clrstats -- write 1 to clear every bank (self-clearing)
//       0x404  ANY      [15:0] one bit per bank, set while that bank's STICKY[31]
//                       is set.  Read this first: it names the guilty bank in one
//                       transaction instead of sixteen.
//
//   Reads of an unmapped offset return 0.  Writes to anything but CTRL are
//   ignored (the statistics are read-only by construction -- the host cannot
//   forge a clean run).
//
//   Port naming: per-bank inputs are spelled out as _00.._15 rather than packed,
//   the same way emu_cmd_bus does it, so the block design connects them directly
//   with no xlconcat glue.
//////////////////////////////////////////////////////////////////////////////////

module emu_viol_csr #(
    parameter integer LITE_ADDR_WIDTH = 12,   // 4 KiB : 16x0x40 banks + globals
    parameter integer LITE_DATA_WIDTH = 32
)(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire                          s_axi_aclk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                          s_axi_aresetn,

    // ---- AXI4-Lite slave ----
    input  wire                          s_axi_awvalid,
    output wire                          s_axi_awready,
    input  wire [LITE_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire                          s_axi_wvalid,
    output wire                          s_axi_wready,
    input  wire [LITE_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [(LITE_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    output wire                          s_axi_bvalid,
    input  wire                          s_axi_bready,
    output wire [1:0]                    s_axi_bresp,
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    input  wire [LITE_ADDR_WIDTH-1:0]    s_axi_araddr,
    output wire                          s_axi_rvalid,
    input  wire                          s_axi_rready,
    output wire [LITE_DATA_WIDTH-1:0]    s_axi_rdata,
    output wire [1:0]                    s_axi_rresp,

    // ---- per-bank violation events (1-cycle strobes from bank_controller_top) ----
    input  wire                          i_viol_valid_00,
    input  wire [2:0]                    i_viol_kind_00,
    input  wire [7:0]                    i_viol_overrun_00,
    input  wire                          i_ewmul_drop_00,
    input  wire                          i_viol_valid_01,
    input  wire [2:0]                    i_viol_kind_01,
    input  wire [7:0]                    i_viol_overrun_01,
    input  wire                          i_ewmul_drop_01,
    input  wire                          i_viol_valid_02,
    input  wire [2:0]                    i_viol_kind_02,
    input  wire [7:0]                    i_viol_overrun_02,
    input  wire                          i_ewmul_drop_02,
    input  wire                          i_viol_valid_03,
    input  wire [2:0]                    i_viol_kind_03,
    input  wire [7:0]                    i_viol_overrun_03,
    input  wire                          i_ewmul_drop_03,
    input  wire                          i_viol_valid_04,
    input  wire [2:0]                    i_viol_kind_04,
    input  wire [7:0]                    i_viol_overrun_04,
    input  wire                          i_ewmul_drop_04,
    input  wire                          i_viol_valid_05,
    input  wire [2:0]                    i_viol_kind_05,
    input  wire [7:0]                    i_viol_overrun_05,
    input  wire                          i_ewmul_drop_05,
    input  wire                          i_viol_valid_06,
    input  wire [2:0]                    i_viol_kind_06,
    input  wire [7:0]                    i_viol_overrun_06,
    input  wire                          i_ewmul_drop_06,
    input  wire                          i_viol_valid_07,
    input  wire [2:0]                    i_viol_kind_07,
    input  wire [7:0]                    i_viol_overrun_07,
    input  wire                          i_ewmul_drop_07,
    input  wire                          i_viol_valid_08,
    input  wire [2:0]                    i_viol_kind_08,
    input  wire [7:0]                    i_viol_overrun_08,
    input  wire                          i_ewmul_drop_08,
    input  wire                          i_viol_valid_09,
    input  wire [2:0]                    i_viol_kind_09,
    input  wire [7:0]                    i_viol_overrun_09,
    input  wire                          i_ewmul_drop_09,
    input  wire                          i_viol_valid_10,
    input  wire [2:0]                    i_viol_kind_10,
    input  wire [7:0]                    i_viol_overrun_10,
    input  wire                          i_ewmul_drop_10,
    input  wire                          i_viol_valid_11,
    input  wire [2:0]                    i_viol_kind_11,
    input  wire [7:0]                    i_viol_overrun_11,
    input  wire                          i_ewmul_drop_11,
    input  wire                          i_viol_valid_12,
    input  wire [2:0]                    i_viol_kind_12,
    input  wire [7:0]                    i_viol_overrun_12,
    input  wire                          i_ewmul_drop_12,
    input  wire                          i_viol_valid_13,
    input  wire [2:0]                    i_viol_kind_13,
    input  wire [7:0]                    i_viol_overrun_13,
    input  wire                          i_ewmul_drop_13,
    input  wire                          i_viol_valid_14,
    input  wire [2:0]                    i_viol_kind_14,
    input  wire [7:0]                    i_viol_overrun_14,
    input  wire                          i_ewmul_drop_14,
    input  wire                          i_viol_valid_15,
    input  wire [2:0]                    i_viol_kind_15,
    input  wire [7:0]                    i_viol_overrun_15,
    input  wire                          i_ewmul_drop_15
);

    // Violation-kind symbols.  Never spell the numbers here -- bank-controller-ip
    // owns them and a renumber must break loudly, not silently mis-bucket.
    `include "bank_ctrl_defs.vh"

    localparam integer NB = 16;

    // ---------------------------------------------------------------------
    // AXI4-Lite : write channel.  AW and W are gathered independently, so the
    // host may drive them in either order (a master that leads with W would
    // otherwise deadlock against an AW-first slave).
    // ---------------------------------------------------------------------
    reg                       r_aw_hold, r_w_hold;
    reg [LITE_ADDR_WIDTH-1:0] r_aw_addr;
    reg [LITE_DATA_WIDTH-1:0] r_w_data;
    reg                       r_bvalid;

    wire w_aw_hs = s_axi_awvalid & s_axi_awready;
    wire w_w_hs  = s_axi_wvalid  & s_axi_wready;
    // accept when both halves are in hand this cycle (held or arriving) and the
    // previous response has been taken
    wire w_aw_have = r_aw_hold | w_aw_hs;
    wire w_w_have  = r_w_hold  | w_w_hs;
    wire w_wr_accept = w_aw_have & w_w_have & ~r_bvalid;

    wire [LITE_ADDR_WIDTH-1:0] w_wr_addr = r_aw_hold ? r_aw_addr : s_axi_awaddr;
    wire [LITE_DATA_WIDTH-1:0] w_wr_data = r_w_hold  ? r_w_data  : s_axi_wdata;

    assign s_axi_awready = ~r_aw_hold & ~r_bvalid;
    assign s_axi_wready  = ~r_w_hold  & ~r_bvalid;
    assign s_axi_bvalid  = r_bvalid;
    assign s_axi_bresp   = 2'b00;

    // CTRL.clrstats : write-1, self-clearing.  Never stored, so a host that
    // forgets to write 0 back cannot wedge the counters at zero.
    wire w_clrstats = w_wr_accept & (w_wr_addr[11:0] == 12'h400) & w_wr_data[0];

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            r_aw_hold <= 1'b0; r_w_hold <= 1'b0; r_bvalid <= 1'b0;
            r_aw_addr <= {LITE_ADDR_WIDTH{1'b0}};
            r_w_data  <= {LITE_DATA_WIDTH{1'b0}};
        end else begin
            if (w_aw_hs && !w_wr_accept) begin r_aw_hold <= 1'b1; r_aw_addr <= s_axi_awaddr; end
            if (w_w_hs  && !w_wr_accept) begin r_w_hold  <= 1'b1; r_w_data  <= s_axi_wdata;  end
            if (w_wr_accept) begin r_aw_hold <= 1'b0; r_w_hold <= 1'b0; r_bvalid <= 1'b1; end
            else if (r_bvalid & s_axi_bready) r_bvalid <= 1'b0;
        end
    end

    // ---------------------------------------------------------------------
    // Statistics : one block per bank, built by generate so all 16 are
    // provably identical.  The per-bank input scalars are gathered into local
    // buses first; the ports stay spelled out for the block design's sake.
    // ---------------------------------------------------------------------
    wire [NB-1:0]     w_v_valid;
    wire [NB*3-1:0]   w_v_kind;
    wire [NB*8-1:0]   w_v_over;
    wire [NB-1:0]     w_drop;

    assign w_v_valid[0]         = i_viol_valid_00;
    assign w_v_kind[0*3 +: 3]   = i_viol_kind_00;
    assign w_v_over[0*8 +: 8]   = i_viol_overrun_00;
    assign w_drop[0]            = i_ewmul_drop_00;
    assign w_v_valid[1]         = i_viol_valid_01;
    assign w_v_kind[1*3 +: 3]   = i_viol_kind_01;
    assign w_v_over[1*8 +: 8]   = i_viol_overrun_01;
    assign w_drop[1]            = i_ewmul_drop_01;
    assign w_v_valid[2]         = i_viol_valid_02;
    assign w_v_kind[2*3 +: 3]   = i_viol_kind_02;
    assign w_v_over[2*8 +: 8]   = i_viol_overrun_02;
    assign w_drop[2]            = i_ewmul_drop_02;
    assign w_v_valid[3]         = i_viol_valid_03;
    assign w_v_kind[3*3 +: 3]   = i_viol_kind_03;
    assign w_v_over[3*8 +: 8]   = i_viol_overrun_03;
    assign w_drop[3]            = i_ewmul_drop_03;
    assign w_v_valid[4]         = i_viol_valid_04;
    assign w_v_kind[4*3 +: 3]   = i_viol_kind_04;
    assign w_v_over[4*8 +: 8]   = i_viol_overrun_04;
    assign w_drop[4]            = i_ewmul_drop_04;
    assign w_v_valid[5]         = i_viol_valid_05;
    assign w_v_kind[5*3 +: 3]   = i_viol_kind_05;
    assign w_v_over[5*8 +: 8]   = i_viol_overrun_05;
    assign w_drop[5]            = i_ewmul_drop_05;
    assign w_v_valid[6]         = i_viol_valid_06;
    assign w_v_kind[6*3 +: 3]   = i_viol_kind_06;
    assign w_v_over[6*8 +: 8]   = i_viol_overrun_06;
    assign w_drop[6]            = i_ewmul_drop_06;
    assign w_v_valid[7]         = i_viol_valid_07;
    assign w_v_kind[7*3 +: 3]   = i_viol_kind_07;
    assign w_v_over[7*8 +: 8]   = i_viol_overrun_07;
    assign w_drop[7]            = i_ewmul_drop_07;
    assign w_v_valid[8]         = i_viol_valid_08;
    assign w_v_kind[8*3 +: 3]   = i_viol_kind_08;
    assign w_v_over[8*8 +: 8]   = i_viol_overrun_08;
    assign w_drop[8]            = i_ewmul_drop_08;
    assign w_v_valid[9]         = i_viol_valid_09;
    assign w_v_kind[9*3 +: 3]   = i_viol_kind_09;
    assign w_v_over[9*8 +: 8]   = i_viol_overrun_09;
    assign w_drop[9]            = i_ewmul_drop_09;
    assign w_v_valid[10]         = i_viol_valid_10;
    assign w_v_kind[10*3 +: 3]   = i_viol_kind_10;
    assign w_v_over[10*8 +: 8]   = i_viol_overrun_10;
    assign w_drop[10]            = i_ewmul_drop_10;
    assign w_v_valid[11]         = i_viol_valid_11;
    assign w_v_kind[11*3 +: 3]   = i_viol_kind_11;
    assign w_v_over[11*8 +: 8]   = i_viol_overrun_11;
    assign w_drop[11]            = i_ewmul_drop_11;
    assign w_v_valid[12]         = i_viol_valid_12;
    assign w_v_kind[12*3 +: 3]   = i_viol_kind_12;
    assign w_v_over[12*8 +: 8]   = i_viol_overrun_12;
    assign w_drop[12]            = i_ewmul_drop_12;
    assign w_v_valid[13]         = i_viol_valid_13;
    assign w_v_kind[13*3 +: 3]   = i_viol_kind_13;
    assign w_v_over[13*8 +: 8]   = i_viol_overrun_13;
    assign w_drop[13]            = i_ewmul_drop_13;
    assign w_v_valid[14]         = i_viol_valid_14;
    assign w_v_kind[14*3 +: 3]   = i_viol_kind_14;
    assign w_v_over[14*8 +: 8]   = i_viol_overrun_14;
    assign w_drop[14]            = i_ewmul_drop_14;
    assign w_v_valid[15]         = i_viol_valid_15;
    assign w_v_kind[15*3 +: 3]   = i_viol_kind_15;
    assign w_v_over[15*8 +: 8]   = i_viol_overrun_15;
    assign w_drop[15]            = i_ewmul_drop_15;
    // 5 kinds x {sticky, count, max} + drop {sticky, count}, per bank.
    reg [NB-1:0]   r_stick [1:5];
    reg [7:0]      r_cnt   [1:5][0:NB-1];
    reg [7:0]      r_max   [1:5][0:NB-1];
    reg [NB-1:0]   r_drop_s;
    reg [7:0]      r_drop_c [0:NB-1];

    integer bi, ki;
    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn || w_clrstats) begin
            for (ki = 1; ki <= 5; ki = ki + 1) begin
                r_stick[ki] <= {NB{1'b0}};
                for (bi = 0; bi < NB; bi = bi + 1) begin
                    r_cnt[ki][bi] <= 8'd0;
                    r_max[ki][bi] <= 8'd0;
                end
            end
            r_drop_s <= {NB{1'b0}};
            for (bi = 0; bi < NB; bi = bi + 1) r_drop_c[bi] <= 8'd0;
        end else begin
            for (bi = 0; bi < NB; bi = bi + 1) begin
                if (w_v_valid[bi]) begin
                    ki = w_v_kind[bi*3 +: 3];
                    // BC_VIOL_NONE (0) is not an event; anything above 5 would be
                    // a renumber upstream and is dropped rather than mis-bucketed.
                    if (ki >= 1 && ki <= 5) begin
                        r_stick[ki][bi] <= 1'b1;
                        if (r_cnt[ki][bi] != 8'hFF) r_cnt[ki][bi] <= r_cnt[ki][bi] + 8'd1;
                        if (w_v_over[bi*8 +: 8] > r_max[ki][bi])
                            r_max[ki][bi] <= w_v_over[bi*8 +: 8];
                    end
                end
                if (w_drop[bi]) begin
                    r_drop_s[bi] <= 1'b1;
                    if (r_drop_c[bi] != 8'hFF) r_drop_c[bi] <= r_drop_c[bi] + 8'd1;
                end
            end
        end
    end

    // ANY bitmap : one read names the guilty bank.
    wire [NB-1:0] w_any;
    genvar gb;
    generate for (gb = 0; gb < NB; gb = gb + 1) begin : g_any
        assign w_any[gb] = r_stick[BC_VIOL_RCD_RD][gb] | r_stick[BC_VIOL_CCD_RD][gb]
                         | r_stick[BC_VIOL_RCD_WR][gb] | r_stick[BC_VIOL_CCD_WR][gb]
                         | r_stick[BC_VIOL_RECOVERY_WR][gb] | r_drop_s[gb];
    end endgenerate

    // ---------------------------------------------------------------------
    // AXI4-Lite : read channel.  Single outstanding, registered data.
    // ---------------------------------------------------------------------
    reg                       r_rvalid;
    reg [LITE_DATA_WIDTH-1:0] r_rdata;
    assign s_axi_arready = ~r_rvalid;
    assign s_axi_rvalid  = r_rvalid;
    assign s_axi_rdata   = r_rdata;
    assign s_axi_rresp   = 2'b00;

    wire [3:0] w_rd_bank = s_axi_araddr[9:6];   // b*0x40
    wire [3:0] w_rd_reg  = s_axi_araddr[5:2];   // register within the block
    wire       w_rd_glob = s_axi_araddr[10];    // 0x400+

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            r_rvalid <= 1'b0;
            r_rdata  <= {LITE_DATA_WIDTH{1'b0}};
        end else begin
            if (s_axi_arvalid & s_axi_arready) begin
                r_rvalid <= 1'b1;
                r_rdata  <= {LITE_DATA_WIDTH{1'b0}};
                if (w_rd_glob) begin
                    case (s_axi_araddr[3:2])
                        2'd0: r_rdata <= {LITE_DATA_WIDTH{1'b0}};       // 0x400 CTRL reads 0
                        2'd1: r_rdata <= {{(LITE_DATA_WIDTH-NB){1'b0}}, w_any};  // 0x404 ANY
                        default: ;
                    endcase
                end else begin
                    case (w_rd_reg)
                        4'd0: r_rdata <= {                              // STICKY
                                w_any[w_rd_bank], 22'd0,
                                r_drop_s[w_rd_bank], 3'd0,
                                r_stick[BC_VIOL_RECOVERY_WR][w_rd_bank],
                                r_stick[BC_VIOL_CCD_WR][w_rd_bank],
                                r_stick[BC_VIOL_RCD_WR][w_rd_bank],
                                r_stick[BC_VIOL_CCD_RD][w_rd_bank],
                                r_stick[BC_VIOL_RCD_RD][w_rd_bank]};
                        4'd1: r_rdata <= {r_cnt[BC_VIOL_CCD_WR][w_rd_bank],   // CNT_A
                                          r_cnt[BC_VIOL_RCD_WR][w_rd_bank],
                                          r_cnt[BC_VIOL_CCD_RD][w_rd_bank],
                                          r_cnt[BC_VIOL_RCD_RD][w_rd_bank]};
                        4'd2: r_rdata <= {16'd0, r_drop_c[w_rd_bank],         // CNT_B
                                          r_cnt[BC_VIOL_RECOVERY_WR][w_rd_bank]};
                        4'd3: r_rdata <= {r_max[BC_VIOL_CCD_WR][w_rd_bank],   // MAX_A
                                          r_max[BC_VIOL_RCD_WR][w_rd_bank],
                                          r_max[BC_VIOL_CCD_RD][w_rd_bank],
                                          r_max[BC_VIOL_RCD_RD][w_rd_bank]};
                        4'd4: r_rdata <= {24'd0,                              // MAX_B
                                          r_max[BC_VIOL_RECOVERY_WR][w_rd_bank]};
                        default: ;                                            // unmapped -> 0
                    endcase
                end
            end else if (r_rvalid & s_axi_rready) begin
                r_rvalid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
