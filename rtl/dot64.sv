// 64-lane INT4(weight) x INT8(activation) dot product unit.
// Combinational: 64 signed multipliers + 6-level adder tree -> INT32.
// One word of weights (64 x 4b = 256b) against one word of activations
// (64 x 8b = 512b) per cycle -> 64 MACs/cycle.
module dot64 (
    input  wire [255:0] w,   // 64 packed signed INT4 weights
    input  wire [511:0] x,   // 64 packed signed INT8 activations
    output wire [31:0]  y
);
    integer i;
    reg signed [31:0] sum;
    reg signed [3:0]  wq;
    reg signed [7:0]  xq;
    always @(*) begin
        sum = 32'sd0;
        for (i = 0; i < 64; i = i + 1) begin
            wq  = w[i*4 +: 4];
            xq  = x[i*8 +: 8];
            sum = sum + wq * xq;
        end
    end
    assign y = sum;
endmodule
