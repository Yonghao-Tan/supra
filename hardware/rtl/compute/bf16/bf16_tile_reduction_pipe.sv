`default_nettype none

module bf16_tile_reduction_pipe #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  req_valid,
    output logic                  req_ready,
    input  logic [1023:0]         req_values,
    input  logic [63:0]           req_lane_mask,
    input  logic [TAG_WIDTH-1:0]  req_tag,
    output logic                  rsp_valid,
    input  logic                  rsp_ready,
    output logic [127:0]          rsp_values,
    output logic [7:0]            rsp_row_mask,
    output logic [TAG_WIDTH-1:0]  rsp_tag
);
    logic [511:0] level0_lhs;
    logic [511:0] level0_rhs;
    logic [31:0] level0_mask;
    logic level0_rsp_valid;
    logic level0_rsp_ready;
    logic [511:0] level0_values;
    logic [31:0] level0_rsp_mask;
    logic [TAG_WIDTH-1:0] level0_tag;
    logic [255:0] level1_lhs;
    logic [255:0] level1_rhs;
    logic [15:0] level1_mask;
    logic level1_req_ready;
    logic level1_rsp_valid;
    logic level1_rsp_ready;
    logic [255:0] level1_values;
    logic [15:0] level1_rsp_mask;
    logic [TAG_WIDTH-1:0] level1_tag;
    logic [127:0] level2_lhs;
    logic [127:0] level2_rhs;
    logic [7:0] level2_mask;
    logic level2_req_ready;

    always_comb begin
        level0_lhs = '0;
        level0_rhs = '0;
        level0_mask = '0;
        for (integer row = 0; row < 8; row = row + 1) begin
            for (integer pair = 0; pair < 4; pair = pair + 1) begin
                integer input_base;
                integer output_index;
                input_base = row*8 + pair*2;
                output_index = row*4 + pair;
                level0_lhs[output_index*16 +: 16] = req_lane_mask[input_base] ?
                    req_values[input_base*16 +: 16] : 16'h0000;
                level0_rhs[output_index*16 +: 16] = req_lane_mask[input_base+1] ?
                    req_values[(input_base+1)*16 +: 16] : 16'h0000;
                level0_mask[output_index] = req_lane_mask[input_base] ||
                    req_lane_mask[input_base+1];
            end
        end

        level1_lhs = '0;
        level1_rhs = '0;
        level1_mask = '0;
        for (integer row = 0; row < 8; row = row + 1) begin
            for (integer pair = 0; pair < 2; pair = pair + 1) begin
                integer input_base;
                integer output_index;
                input_base = row*4 + pair*2;
                output_index = row*2 + pair;
                level1_lhs[output_index*16 +: 16] = level0_rsp_mask[input_base] ?
                    level0_values[input_base*16 +: 16] : 16'h0000;
                level1_rhs[output_index*16 +: 16] = level0_rsp_mask[input_base+1] ?
                    level0_values[(input_base+1)*16 +: 16] : 16'h0000;
                level1_mask[output_index] = level0_rsp_mask[input_base] ||
                    level0_rsp_mask[input_base+1];
            end
        end

        level2_lhs = '0;
        level2_rhs = '0;
        level2_mask = '0;
        for (integer row = 0; row < 8; row = row + 1) begin
            level2_lhs[row*16 +: 16] = level1_rsp_mask[row*2] ?
                level1_values[(row*2)*16 +: 16] : 16'h0000;
            level2_rhs[row*16 +: 16] = level1_rsp_mask[row*2+1] ?
                level1_values[(row*2+1)*16 +: 16] : 16'h0000;
            level2_mask[row] = level1_rsp_mask[row*2] || level1_rsp_mask[row*2+1];
        end
    end

    bf16_add_pipe #(.LANES(32), .TAG_WIDTH(TAG_WIDTH)) level0 (
        .clk(clk), .rst(rst), .req_valid(req_valid), .req_ready(req_ready),
        .req_lhs(level0_lhs), .req_rhs(level0_rhs), .req_lane_mask(level0_mask),
        .req_tag(req_tag), .rsp_valid(level0_rsp_valid), .rsp_ready(level0_rsp_ready),
        .rsp_values(level0_values), .rsp_lane_mask(level0_rsp_mask), .rsp_tag(level0_tag));

    assign level0_rsp_ready = level1_req_ready;
    bf16_add_pipe #(.LANES(16), .TAG_WIDTH(TAG_WIDTH)) level1 (
        .clk(clk), .rst(rst), .req_valid(level0_rsp_valid), .req_ready(level1_req_ready),
        .req_lhs(level1_lhs), .req_rhs(level1_rhs), .req_lane_mask(level1_mask),
        .req_tag(level0_tag), .rsp_valid(level1_rsp_valid), .rsp_ready(level1_rsp_ready),
        .rsp_values(level1_values), .rsp_lane_mask(level1_rsp_mask), .rsp_tag(level1_tag));

    assign level1_rsp_ready = level2_req_ready;
    bf16_add_pipe #(.LANES(8), .TAG_WIDTH(TAG_WIDTH)) level2 (
        .clk(clk), .rst(rst), .req_valid(level1_rsp_valid), .req_ready(level2_req_ready),
        .req_lhs(level2_lhs), .req_rhs(level2_rhs), .req_lane_mask(level2_mask),
        .req_tag(level1_tag), .rsp_valid(rsp_valid), .rsp_ready(rsp_ready),
        .rsp_values(rsp_values), .rsp_lane_mask(rsp_row_mask), .rsp_tag(rsp_tag));

`ifndef SYNTHESIS
    logic stalled;
    logic [127:0] held_values;
    logic [7:0] held_mask;
    logic [TAG_WIDTH-1:0] held_tag;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled <= 1'b0;
            held_values <= '0;
            held_mask <= '0;
            held_tag <= '0;
        end else begin
            if (stalled)
                assert (rsp_valid && rsp_values == held_values &&
                        rsp_row_mask == held_mask && rsp_tag == held_tag)
                    else $error("bf16_tile_reduction_pipe changed a stalled response");
            stalled <= rsp_valid && !rsp_ready;
            if (rsp_valid && !rsp_ready) begin
                held_values <= rsp_values;
                held_mask <= rsp_row_mask;
                held_tag <= rsp_tag;
            end
        end
    end
`endif
endmodule

`default_nettype wire
