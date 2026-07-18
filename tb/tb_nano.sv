// Testbench: load compiled program + memories, run autoregressive generation,
// compare generated tokens bit-exactly against the golden model.
`timescale 1ns/1ps
module tb;
`ifdef GEN_TOKENS
    localparam G      = `GEN_TOKENS;
`else
    localparam G      = 32;   // generated tokens (must match sw/build.py)
`endif
    localparam PLEN   = 8;    // prompt length = context size

    reg  clk = 0, rst_n = 1;
    wire done;
    wire out_valid;
    wire [7:0] out_tok;
    nano_accel dut (.clk(clk), .rst_n(rst_n), .done(done),
                    .out_valid(out_valid), .out_tok(out_tok));

    always #5 clk = ~clk;   // 100 MHz

    // live token stream from the chip's output interface
    always @(posedge clk) if (rst_n && out_valid) $write("%c", out_tok);

    reg [7:0] expected [0:G-1];
    integer i, errors;

    initial begin
        $readmemh("out/isram.hex", dut.isram.mem);
        $readmemh("out/wsram.hex", dut.wsram.mem);
        $readmemh("out/esram.hex", dut.esram.mem);
        $readmemh("out/bsram.hex", dut.bsram.mem);
        $readmemh("out/msram.hex", dut.msram.mem);
        $readmemh("out/tsram.hex", dut.tsram.mem);
        $readmemh("out/expected.hex", expected);

        #1  rst_n = 0;
        #20 rst_n = 1;
        $write("streaming live    = \"");

        wait (done);
        #10;

        $display("\"");
        $display("----------------------------------------------------");
        $display("DONE: %0d cycles for %0d generated tokens", dut.cycles, G);
        $display("cycles/token      = %0d", dut.cycles / G);
        $display("tokens/s @100MHz  = %0d", 100000000 / (dut.cycles / G));
        $write  ("generated text    = \"");
        for (i = 0; i < G; i = i + 1) $write("%c", dut.tsram.mem[PLEN+i]);
        $display("\"");

        errors = 0;
        for (i = 0; i < G; i = i + 1)
            if (dut.tsram.mem[PLEN+i] !== expected[i]) begin
                errors = errors + 1;
                $display("MISMATCH token %0d: got 0x%02x expected 0x%02x",
                         i, dut.tsram.mem[PLEN+i], expected[i]);
            end
        if (errors == 0) $display("PASS: bit-exact match with golden model");
        else             $display("FAIL: %0d/%0d token mismatches", errors, G);
        $finish;
    end

    initial begin
        #50000000;  // 5M cycles timeout
        $display("TIMEOUT: done never asserted");
        $finish;
    end
endmodule
