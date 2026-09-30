`default_nettype none

// One synchronous true-dual-port array; the containing macro interface adds
// the second read cycle. Reset never clears SRAM contents.
module local_sram_array #(parameter integer DEPTH = 1024) (
    input logic clk,
    input logic a_enable, a_write,
    input logic [11:0] a_address,
    input logic [127:0] a_write_data, a_write_mask_n,
    output logic [127:0] a_read_data,
    input logic b_enable, b_write,
    input logic [11:0] b_address,
    input logic [127:0] b_write_data, b_write_mask_n,
    output logic [127:0] b_read_data
);
    localparam integer ADDRESS_BITS = $clog2(DEPTH);
    logic [127:0] storage [0:DEPTH-1];
    always_ff @(posedge clk) begin
        if (a_enable && !a_write) a_read_data <= storage[a_address[ADDRESS_BITS-1:0]];
        if (b_enable && !b_write) b_read_data <= storage[b_address[ADDRESS_BITS-1:0]];
        if (a_enable && a_write)
            storage[a_address[ADDRESS_BITS-1:0]] <= (storage[a_address[ADDRESS_BITS-1:0]] & a_write_mask_n) | (a_write_data & ~a_write_mask_n);
        if (b_enable && b_write)
            storage[b_address[ADDRESS_BITS-1:0]] <= (storage[b_address[ADDRESS_BITS-1:0]] & b_write_mask_n) | (b_write_data & ~b_write_mask_n);
        assert ((!a_enable || int'(a_address) < DEPTH) && (!b_enable || int'(b_address) < DEPTH))
            else $error("local_sram_array address exceeds depth %0d", DEPTH);
    end
    initial begin
        if (DEPTH != 1024 && DEPTH != 2048 && DEPTH != 4096)
            $error("local_sram_array depth must be 1024, 2048, or 4096");
    end
endmodule

`default_nettype wire
