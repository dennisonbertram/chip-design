// Generic single-write-port, async-read RAM (behavioral; synthesizes to FF array).
// Real SRAM macros don't exist in the open Nangate45 PDK; in a commercial flow
// these get replaced by foundry SRAM compilers. Sizes are reported separately.
// The (* blackbox *) attribute excludes it from synthesis (reported as SRAM);
// simulators ignore the attribute and use the behavioral model.
(* blackbox *)
module ram #(
    parameter WIDTH = 32,
    parameter DEPTH = 256,
    parameter AW    = 8
) (
    input  wire             clk,
    input  wire             we,
    input  wire [AW-1:0]    waddr,
    input  wire [WIDTH-1:0] wdata,
    input  wire [AW-1:0]    raddr,
    output wire [WIDTH-1:0] rdata
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end
    assign rdata = mem[raddr];
endmodule
