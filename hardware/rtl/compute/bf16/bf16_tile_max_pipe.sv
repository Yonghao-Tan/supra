`default_nettype none

module bf16_tile_max_pipe #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  req_valid,
    output logic                  req_ready,
    input  logic                  req_magnitude,
    input  logic [1023:0]         req_values,
    input  logic [63:0]           req_lane_mask,
    input  logic [TAG_WIDTH-1:0]  req_tag,
    output logic                  rsp_valid,
    input  logic                  rsp_ready,
    output logic [127:0]          rsp_values,
    output logic [7:0]            rsp_row_mask,
    output logic [TAG_WIDTH-1:0]  rsp_tag
);
    logic stage0_valid;
    logic [511:0] stage0_values;
    logic [31:0] stage0_mask;
    logic [TAG_WIDTH-1:0] stage0_tag;
    logic stage1_valid;
    logic [255:0] stage1_values;
    logic [15:0] stage1_mask;
    logic [TAG_WIDTH-1:0] stage1_tag;
    logic stage2_valid;
    logic [127:0] stage2_values;
    logic [7:0] stage2_mask;
    logic [TAG_WIDTH-1:0] stage2_tag;
    logic stage0_ready;
    logic stage1_ready;
    logic stage2_ready;

    function automatic logic numeric_greater(
        input logic [15:0] lhs,
        input logic [15:0] rhs
    );
        if (lhs[14:0] == 0 && rhs[14:0] == 0)
            numeric_greater = 1'b0;
        else if (lhs[15] != rhs[15])
            numeric_greater = rhs[15];
        else if (!lhs[15])
            numeric_greater = lhs[14:0] > rhs[14:0];
        else
            numeric_greater = lhs[14:0] < rhs[14:0];
    endfunction

    function automatic logic choose_right(
        input logic        magnitude,
        input logic [15:0] left_value,
        input logic        left_valid,
        input logic [15:0] right_value,
        input logic        right_valid
    );
        if (!right_valid)
            choose_right = 1'b0;
        else if (!left_valid)
            choose_right = 1'b1;
        else if (magnitude)
            choose_right = right_value[14:0] > left_value[14:0];
        else
            choose_right = numeric_greater(right_value, left_value);
    endfunction

    assign stage2_ready = !stage2_valid || rsp_ready;
    assign stage1_ready = !stage1_valid || stage2_ready;
    assign stage0_ready = !stage0_valid || stage1_ready;
    assign req_ready = stage0_ready;
    assign rsp_valid = stage2_valid;
    assign rsp_values = stage2_values;
    assign rsp_row_mask = stage2_mask;
    assign rsp_tag = stage2_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            stage0_valid <= 1'b0;
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
        end else begin
            if (stage2_ready) begin
                stage2_valid <= stage1_valid;
                if (stage1_valid) begin
                    for (integer row = 0; row < 8; row = row + 1) begin
                        logic select_right;
                        select_right = choose_right(1'b0,
                            stage1_values[(row*2)*16 +: 16], stage1_mask[row*2],
                            stage1_values[(row*2+1)*16 +: 16], stage1_mask[row*2+1]);
                        stage2_values[row*16 +: 16] <= select_right ?
                            stage1_values[(row*2+1)*16 +: 16] :
                            stage1_values[(row*2)*16 +: 16];
                        stage2_mask[row] <= stage1_mask[row*2] || stage1_mask[row*2+1];
                    end
                    stage2_tag <= stage1_tag;
                end
            end
            if (stage1_ready) begin
                stage1_valid <= stage0_valid;
                if (stage0_valid) begin
                    for (integer item = 0; item < 16; item = item + 1) begin
                        logic select_right;
                        select_right = choose_right(1'b0,
                            stage0_values[(item*2)*16 +: 16], stage0_mask[item*2],
                            stage0_values[(item*2+1)*16 +: 16], stage0_mask[item*2+1]);
                        stage1_values[item*16 +: 16] <= select_right ?
                            stage0_values[(item*2+1)*16 +: 16] :
                            stage0_values[(item*2)*16 +: 16];
                        stage1_mask[item] <= stage0_mask[item*2] || stage0_mask[item*2+1];
                    end
                    stage1_tag <= stage0_tag;
                end
            end
            if (stage0_ready) begin
                stage0_valid <= req_valid;
                if (req_valid) begin
                    for (integer item = 0; item < 32; item = item + 1) begin
                        logic [15:0] left_value;
                        logic [15:0] right_value;
                        logic [15:0] selected_value;
                        logic select_right;
                        left_value = req_values[(item*2)*16 +: 16];
                        right_value = req_values[(item*2+1)*16 +: 16];
                        select_right = choose_right(req_magnitude,
                            left_value, req_lane_mask[item*2],
                            right_value, req_lane_mask[item*2+1]);
                        selected_value = select_right ? right_value : left_value;
                        if (req_magnitude)
                            stage0_values[item*16 +: 16] <= {1'b0, selected_value[14:0]};
                        else
                            stage0_values[item*16 +: 16] <= selected_value;
                        stage0_mask[item] <= req_lane_mask[item*2] ||
                            req_lane_mask[item*2+1];
                    end
                    stage0_tag <= req_tag;
                end
            end
        end
    end

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
                    else $error("bf16_tile_max_pipe changed a stalled response");
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
