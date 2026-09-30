`default_nettype none

module bf16_vector8_max_pipe #(
    parameter integer TAG_WIDTH = 6
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 req_valid,
    output logic                 req_ready,
    input  logic [8*16-1:0]      req_values,
    input  logic [7:0]           req_lane_mask,
    input  logic [TAG_WIDTH-1:0] req_tag,
    output logic                 rsp_valid,
    input  logic                 rsp_ready,
    output logic [15:0]          rsp_value,
    output logic [TAG_WIDTH-1:0] rsp_tag
);
    logic stage0_valid;
    logic [4*16-1:0] stage0_values;
    logic [3:0] stage0_mask;
    logic [TAG_WIDTH-1:0] stage0_tag;
    logic stage1_valid;
    logic [2*16-1:0] stage1_values;
    logic [1:0] stage1_mask;
    logic [TAG_WIDTH-1:0] stage1_tag;
    logic stage2_valid;
    logic [15:0] stage2_value;
    logic [TAG_WIDTH-1:0] stage2_tag;
    logic stage0_ready, stage1_ready, stage2_ready;

    function automatic logic choose_right(
        input logic [15:0] left_value,
        input logic left_valid,
        input logic [15:0] right_value,
        input logic right_valid
    );
        choose_right = right_valid &&
            (!left_valid || right_value[14:0] > left_value[14:0]);
    endfunction

    assign stage2_ready = !stage2_valid || rsp_ready;
    assign stage1_ready = !stage1_valid || stage2_ready;
    assign stage0_ready = !stage0_valid || stage1_ready;
    assign req_ready = stage0_ready;
    assign rsp_valid = stage2_valid;
    assign rsp_value = stage2_value;
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
                    stage2_value <= choose_right(
                        stage1_values[0 +: 16], stage1_mask[0],
                        stage1_values[16 +: 16], stage1_mask[1]) ?
                        stage1_values[16 +: 16] : stage1_values[0 +: 16];
                    stage2_tag <= stage1_tag;
                end
            end
            if (stage1_ready) begin
                stage1_valid <= stage0_valid;
                if (stage0_valid) begin
                    for (integer pair = 0; pair < 2; pair++) begin
                        stage1_values[pair*16 +: 16] <= choose_right(
                            stage0_values[(pair*2)*16 +: 16],
                            stage0_mask[pair*2],
                            stage0_values[(pair*2+1)*16 +: 16],
                            stage0_mask[pair*2+1]) ?
                            stage0_values[(pair*2+1)*16 +: 16] :
                            stage0_values[(pair*2)*16 +: 16];
                        stage1_mask[pair] <= stage0_mask[pair*2] ||
                            stage0_mask[pair*2+1];
                    end
                    stage1_tag <= stage0_tag;
                end
            end
            if (stage0_ready) begin
                stage0_valid <= req_valid;
                if (req_valid) begin
                    for (integer pair = 0; pair < 4; pair++) begin
                        logic [15:0] selected_value;
                        selected_value = choose_right(
                            req_values[(pair*2)*16 +: 16],
                            req_lane_mask[pair*2],
                            req_values[(pair*2+1)*16 +: 16],
                            req_lane_mask[pair*2+1]) ?
                            req_values[(pair*2+1)*16 +: 16] :
                            req_values[(pair*2)*16 +: 16];
                        stage0_values[pair*16 +: 16] <=
                            {1'b0, selected_value[14:0]};
                        stage0_mask[pair] <= req_lane_mask[pair*2] ||
                            req_lane_mask[pair*2+1];
                    end
                    stage0_tag <= req_tag;
                end
            end
        end
    end

`ifndef SYNTHESIS
    logic stalled;
    logic [15:0] held_value;
    logic [TAG_WIDTH-1:0] held_tag;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled <= 1'b0;
            held_value <= '0;
            held_tag <= '0;
        end else begin
            if (stalled)
                assert (rsp_valid && rsp_value == held_value &&
                        rsp_tag == held_tag)
                    else $error("bf16_vector8_max_pipe changed a stalled response");
            stalled <= rsp_valid && !rsp_ready;
            if (rsp_valid && !rsp_ready) begin
                held_value <= rsp_value;
                held_tag <= rsp_tag;
            end
        end
    end
`endif
endmodule

`default_nettype wire
