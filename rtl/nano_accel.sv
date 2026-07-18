// nano_accel - autoregressive nano language model accelerator.
//
// Serves a char-level MLP (context C=8, embed E=24, hidden H=512 x2, vocab V=128)
// with INT4 weights / INT8 activations, all state on-chip.
//
// Pipeline: 64 INT4xINT8 MACs/cycle (dot64) -> INT32 accumulator ->
// fused dequant/requant drain: y = clamp(round((acc + bias) * M[c]) >> sh)
// where M[c] is a per-output-channel fixed-point scale precomputed by the
// compiler (folds weight scale, input activation scale, output scale).
//
// ISA (128-bit instructions, ISRAM):
//   [3:0] op      [7:4] flags (bit0 RELU, bit1 TO_LBUF)   [12:8] sh (drain shift)
//   [31:16] n     [47:32] m     [63:48] a0   [79:64] a1
//   [95:80] a2    [111:96] a3   [127:112] a4
//   op=1 SETLEN : seqlen <- a0
//   op=2 GATHER : x[a1..] <- embedding lookup of tokens seqlen-n .. seqlen-1
//                 (a0=emb base word in ESRAM, n=C tokens, m=E bytes/token)
//   op=3 MATVEC : y[a2] <- W[a0..] (m x n INT4) * x[a1] (INT8), bias BSRAM[a3],
//                 scales MSRAM[a4]; flags: RELU, TO_LBUF (write 48b raw to LBUF)
//   op=4 ARGMAX : tsram[seqlen] <- argmax(LBUF[0..m-1]); seqlen++
//   op=7 HALT   : done <- 1
//
// All memories are behavioral (ram.sv); a commercial flow swaps in SRAM macros.

module nano_accel (
    input  wire       clk,
    input  wire       rst_n,
    output reg        done,
    // host output interface: one pulse per generated token
    output reg        out_valid,
    output reg [7:0]  out_tok
);
    // ---------------- memories ----------------
    // program
    wire [127:0] ir_w;
    reg  [7:0]   pc;
    ram #(.WIDTH(128), .DEPTH(256), .AW(8)) isram (
        .clk(clk), .we(1'b0), .waddr(8'd0), .wdata(128'd0),
        .raddr(pc), .rdata(ir_w));

    // weights: 8192 x 256b = 256 KB (INT4 packed 64/word)
    reg  [12:0]  ws_raddr;
    wire [255:0] ws_rdata;
    ram #(.WIDTH(256), .DEPTH(8192), .AW(13)) wsram (
        .clk(clk), .we(1'b0), .waddr(13'd0), .wdata(256'd0),
        .raddr(ws_raddr), .rdata(ws_rdata));

    // activations: 64 x 512b = 4 KB (INT8 packed 64/word)
    reg  [5:0]   xs_raddr;
    wire [511:0] xs_rdata;
    (* keep = 1 *) reg          xs_we;
    (* keep = 1 *) reg  [5:0]   xs_waddr;
    (* keep = 1 *) reg  [511:0] xs_wdata;
    ram #(.WIDTH(512), .DEPTH(64), .AW(6)) xsram (
        .clk(clk), .we(xs_we), .waddr(xs_waddr), .wdata(xs_wdata),
        .raddr(xs_raddr), .rdata(xs_rdata));

    // embeddings: 1024 x 32b = 4 KB (INT8 packed 4/word)
    reg  [9:0]   es_raddr;
    wire [31:0]  es_rdata;
    ram #(.WIDTH(32), .DEPTH(1024), .AW(10)) esram (
        .clk(clk), .we(1'b0), .waddr(10'd0), .wdata(32'd0),
        .raddr(es_raddr), .rdata(es_rdata));

    // biases: 2048 x 32b (INT32, in acc units); L1@0, L2@512, HEAD@1024
    reg  [10:0]  bs_raddr;
    wire [31:0]  bs_rdata;
    ram #(.WIDTH(32), .DEPTH(2048), .AW(11)) bsram (
        .clk(clk), .we(1'b0), .waddr(11'd0), .wdata(32'd0),
        .raddr(bs_raddr), .rdata(bs_rdata));

    // per-channel dequant scales: 2048 x 16b (unsigned fixed point)
    reg  [10:0]  ms_raddr;
    wire [15:0]  ms_rdata;
    ram #(.WIDTH(16), .DEPTH(2048), .AW(11)) msram (
        .clk(clk), .we(1'b0), .waddr(11'd0), .wdata(16'd0),
        .raddr(ms_raddr), .rdata(ms_rdata));

    // tokens: 256 x 8b (prompt preloaded; ARGMAX appends)
    reg  [7:0]   ts_raddr;
    wire [7:0]   ts_rdata;
    reg          ts_we;
    reg  [7:0]   ts_waddr;
    reg  [7:0]   ts_wdata;
    ram #(.WIDTH(8), .DEPTH(256), .AW(8)) tsram (
        .clk(clk), .we(ts_we), .waddr(ts_waddr), .wdata(ts_wdata),
        .raddr(ts_raddr), .rdata(ts_rdata));

    // logits: 128 x 48b signed (flip-flops)
    reg signed [47:0] lbuf [0:127];

    // ---------------- instruction fields ----------------
    wire [3:0]  op    = ir_w[3:0];
    wire [3:0]  flags = ir_w[7:4];
    wire [4:0]  sh    = ir_w[12:8];
    wire [15:0] n     = ir_w[31:16];
    wire [15:0] m     = ir_w[47:32];
    wire [15:0] a0    = ir_w[63:48];
    wire [15:0] a1    = ir_w[79:64];
    wire [15:0] a2    = ir_w[95:80];
    wire [15:0] a3    = ir_w[111:96];
    wire [15:0] a4    = ir_w[127:112];
    wire        f_relu = flags[0];
    wire        f_lbuf = flags[1];

    localparam OP_SETLEN = 4'd1, OP_GATHER = 4'd2, OP_MATVEC = 4'd3,
               OP_ARGMAX = 4'd4, OP_HALT   = 4'd7;

    // ---------------- FSM ----------------
    localparam S_FETCH = 3'd0, S_EXEC = 3'd1, S_GATHER = 3'd2,
               S_MV    = 3'd3, S_AM   = 3'd4, S_HALT   = 3'd5;
    reg [2:0]  state;
    reg [127:0] ir;

    reg [7:0]  seqlen;
    (* keep = 1 *) reg [31:0] cycles;

    // GATHER regs
    reg [15:0]  g_i, g_k, g_cnt;
    (* keep = 1 *) reg [511:0] gstaging;
    wire [7:0]  g_tok = ts_rdata;
    wire [15:0] g_words = (m >> 2);          // E/4 words per token

    // MATVEC regs
    reg [15:0]  mv_c, mv_k, row_base;
    reg signed [31:0] acc;
    wire [15:0] row_words = (n >> 6);        // N/64 words per row
    wire signed [31:0] dot_y;

    dot64 dot_unit (.w(ws_rdata), .x(xs_rdata), .y(dot_y));

    // drain pipeline
    reg               valid1, valid2;
    reg  [15:0]       cd1, cd2;
    reg signed [31:0] acc_snap;
    reg signed [47:0] pb;
    reg               lbuf1, lbuf2, relu1, relu2;
    reg  [4:0]        sh1, sh2;
    (* keep = 1 *) reg [511:0] ystage;

    wire signed [31:0] bias_w = bs_rdata;
    wire signed [16:0] msc_w  = {1'b0, ms_rdata};
    wire signed [47:0] rnd    = (sh2 > 0) ? (48'sd1 <<< (sh2 - 1)) : 48'sd0;
    wire signed [47:0] yshift = (pb + rnd) >>> sh2;
    wire signed [47:0] yrelu  = (relu2 && yshift < 0) ? 48'sd0 : yshift;
    wire [7:0]         y8     = (yrelu > 48'sd127)  ? 8'd127 :
                                (yrelu < -48'sd128) ? 8'h80 : yrelu[7:0];

    // ARGMAX regs
    reg [7:0]        am_i;
    reg signed [47:0] am_best;
    reg [7:0]        am_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_FETCH; pc <= 0; done <= 0; seqlen <= 0; cycles <= 0;
            g_i <= 0; g_k <= 0; g_cnt <= 0; gstaging <= 0;
            mv_c <= 0; mv_k <= 0; row_base <= 0; acc <= 0;
            valid1 <= 0; valid2 <= 0; ystage <= 0;
            xs_we <= 0; ts_we <= 0;
            am_i <= 0; am_best <= 0; am_idx <= 0;
            out_valid <= 0; out_tok <= 0;
        end else begin
            valid1 <= 0; valid2 <= valid1;
            xs_we <= 0; ts_we <= 0; out_valid <= 0;
            if (!done) cycles <= cycles + 1;

            // drain stage 2 (independent of FSM state)
            if (valid2) begin
                if (lbuf2) begin
                    lbuf[cd2[6:0]] <= pb;
                end else begin
                    // on the 64th byte of a word, flush ystage including this byte
                    if (cd2[5:0] == 6'd63) begin
                        xs_we    <= 1;
                        xs_waddr <= a2[5:0] + {2'b00, cd2[9:6]};
                        xs_wdata <= {y8, ystage[503:0]};
                    end else begin
                        ystage[{cd2[5:0], 3'b000} +: 8] <= y8;
                    end
                end
            end
            // drain stage 1
            if (valid1) begin
                pb    <= (acc_snap + bias_w) * msc_w;
                cd2   <= cd1; lbuf2 <= lbuf1; relu2 <= relu1; sh2 <= sh1;
            end

            case (state)
            S_FETCH: begin
                ir <= ir_w;
                state <= S_EXEC;
            end
            S_EXEC: begin
                case (ir[3:0])
                OP_SETLEN: begin seqlen <= ir[63:48]; pc <= pc + 1; state <= S_FETCH; end
                OP_GATHER: begin
                    g_i <= 0; g_k <= 0; g_cnt <= 0; gstaging <= 0;
                    state <= S_GATHER;
                end
                OP_MATVEC: begin
                    mv_c <= 0; mv_k <= 0; row_base <= ir[63:48]; acc <= 0;
                    ystage <= 0;
                    state <= S_MV;
                end
                OP_ARGMAX: begin
                    am_best <= lbuf[0]; am_idx <= 0; am_i <= 1;
                    state <= S_AM;
                end
                OP_HALT: begin done <= 1; state <= S_HALT; end
                default: begin pc <= pc + 1; state <= S_FETCH; end
                endcase
            end

            S_GATHER: begin
                // ts_raddr/es_raddr driven combinationally below
                gstaging[{g_cnt[5:0], 3'b000} +: 32] <= es_rdata;
                if (g_cnt[5:0] == 6'd60) begin
                    xs_we    <= 1;
                    xs_waddr <= a1[5:0] + {2'b00, g_cnt[7:6]};
                    xs_wdata <= {es_rdata, gstaging[479:0]};
                end
                g_cnt <= g_cnt + 4;
                if (g_k == g_words - 1) begin
                    g_k <= 0;
                    if (g_i == n - 1) begin pc <= pc + 1; state <= S_FETCH; end
                    else g_i <= g_i + 1;
                end else g_k <= g_k + 1;
            end

            S_MV: begin
                if (mv_c < m) begin
                    // one 64-wide word per cycle through dot64
                    if (mv_k == row_words - 1) begin
                        // snapshot this channel's result; drain overlaps next channel
                        acc_snap <= acc + dot_y;
                        cd1   <= mv_c; valid1 <= 1;
                        lbuf1 <= f_lbuf; relu1 <= f_relu; sh1 <= sh;
                        acc   <= 0;
                        mv_k  <= 0;
                        row_base <= row_base + row_words;
                        mv_c  <= mv_c + 1;
                    end else begin
                        acc <= acc + dot_y;
                        mv_k <= mv_k + 1;
                    end
                end else begin
                    // all channels dotted; wait for the last drain to complete
                    if (valid2 && cd2 == m - 1) begin
                        pc <= pc + 1; state <= S_FETCH;
                    end
                end
            end

            S_AM: begin
                if (lbuf[am_i[6:0]] > am_best) begin
                    am_best <= lbuf[am_i[6:0]];
                    am_idx  <= am_i;
                end
                if (am_i == m[7:0] - 1) begin
                    ts_we   <= 1;
                    ts_waddr <= seqlen;
                    ts_wdata <= (lbuf[am_i[6:0]] > am_best) ? am_i : am_idx;
                    out_valid <= 1;
                    out_tok  <= (lbuf[am_i[6:0]] > am_best) ? am_i : am_idx;
                    seqlen  <= seqlen + 1;
                    pc <= pc + 1; state <= S_FETCH;
                end else am_i <= am_i + 1;
            end

            S_HALT: ;
            default: state <= S_FETCH;
            endcase
        end
    end

    // ---------------- combinational read addresses ----------------
    always @(*) begin
        // defaults
        ts_raddr = 8'd0;
        es_raddr = 10'd0;
        ws_raddr = 13'd0;
        xs_raddr = 6'd0;
        bs_raddr = 9'd0;
        ms_raddr = 9'd0;
        case (state)
        S_GATHER: begin
            ts_raddr = seqlen - n[7:0] + g_i[7:0];
            es_raddr = a0[9:0] + {g_tok, 2'b00} + {g_tok, 1'b0} + g_k[9:0]; // a0 + tok*6 + k
        end
        S_MV: begin
            ws_raddr = row_base[12:0] + mv_k[12:0];
            xs_raddr = a1[5:0] + mv_k[5:0];
            bs_raddr = a3[10:0] + cd1[10:0];
            ms_raddr = a4[10:0] + cd1[10:0];
        end
        endcase
    end
endmodule
