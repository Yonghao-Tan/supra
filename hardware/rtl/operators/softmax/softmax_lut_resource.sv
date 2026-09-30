`default_nettype none

// Shared exponent and reciprocal lookup hardware. Each external request uses
// two internal cycles: 32 exponent lanes or four reciprocal rows per cycle.
module softmax_lut_resource (
    input  logic clk,
    input  logic rst,

    input  logic exp_req_valid,
    output logic exp_req_ready,
    input  hardware_types_pkg::softmax_exp_request_t exp_req,
    output logic exp_rsp_valid,
    input  logic exp_rsp_ready,
    output hardware_types_pkg::softmax_exp_response_t exp_rsp,

    input  logic reciprocal_req_valid,
    output logic reciprocal_req_ready,
    input  hardware_types_pkg::softmax_reciprocal_request_t reciprocal_req,
    output logic reciprocal_rsp_valid,
    input  logic reciprocal_rsp_ready,
    output hardware_types_pkg::softmax_reciprocal_response_t reciprocal_rsp
);
    import softmax_lut_pkg::*;
    localparam integer EXP_LANES = 32;
    localparam integer RECIPROCAL_ROWS = 4;

    logic exp_stage0_valid;
    logic exp_stage1_valid;
    logic exp_stage2_valid;
    logic exp_stage0_half;
    logic exp_stage1_half;
    logic exp_stage2_half;
    logic [EXP_LANES-1:0] exp_stage0_mask;
    logic [EXP_LANES-1:0] exp_stage1_mask;
    logic [EXP_LANES-1:0] exp_stage2_mask;
    logic [15:0] exp_stage0_tag;
    logic [15:0] exp_stage1_tag;
    logic [15:0] exp_stage2_tag;
    logic [EXP_LANES-1:0] exp_prepare_special;
    logic [EXP_LANES*8-1:0] exp_prepare_special_address;
    logic [EXP_LANES*17-1:0] exp_prepare_unscaled;
    logic [EXP_LANES*5-1:0] exp_prepare_shift;
    logic [EXP_LANES-1:0] exp_stage0_special;
    logic [EXP_LANES*8-1:0] exp_stage0_special_address;
    logic [EXP_LANES*17-1:0] exp_stage0_unscaled;
    logic [EXP_LANES*5-1:0] exp_stage0_shift;
    logic [EXP_LANES-1:0] exp_stage1_special;
    logic [EXP_LANES*8-1:0] exp_stage1_special_address;
    logic [EXP_LANES*25-1:0] exp_stage1_numerator;
    logic [EXP_LANES*5-1:0] exp_stage1_shift;
    logic [EXP_LANES*25-1:0] exp_scaled_numerator;
    logic [EXP_LANES*8-1:0] exp_finished_address;
    logic [EXP_LANES*8-1:0] exp_stage2_address;
    logic exp_input_half_pending;
    logic [EXP_LANES*16-1:0] exp_second_half_delta;
    logic [EXP_LANES-1:0] exp_second_half_mask;
    logic [15:0] exp_second_half_tag;
    logic exp_selected_valid;
    logic [EXP_LANES*16-1:0] exp_selected_delta;
    logic [EXP_LANES-1:0] exp_selected_mask;
    logic [15:0] exp_selected_tag;
    logic exp_selected_half;
    logic exp_result_valid;
    logic [1023:0] exp_result_values;
    logic [63:0] exp_result_mask;
    logic [15:0] exp_result_tag;
    logic exp_pipeline_advance;

    assign exp_pipeline_advance = !exp_result_valid || exp_rsp_ready;
    assign exp_req_ready = !exp_input_half_pending && exp_pipeline_advance;
    assign exp_selected_valid = exp_input_half_pending ||
        (exp_req_valid && exp_req_ready);
    assign exp_selected_delta = exp_input_half_pending ?
        exp_second_half_delta : exp_req.delta_values[511:0];
    assign exp_selected_mask = exp_input_half_pending ?
        exp_second_half_mask : exp_req.lane_mask[31:0];
    assign exp_selected_tag = exp_input_half_pending ?
        exp_second_half_tag : exp_req.tag;
    assign exp_selected_half = exp_input_half_pending;
    assign exp_rsp_valid = exp_result_valid;
    assign exp_rsp.values = exp_result_values;
    assign exp_rsp.lane_mask = exp_result_mask;
    assign exp_rsp.tag = exp_result_tag;

    for (genvar lane = 0; lane < EXP_LANES; lane = lane + 1) begin : g_exp_address
        softmax_lut_address_prepare prepare (
            .delta(exp_selected_delta[lane*16 +: 16]),
            .special_case(exp_prepare_special[lane]),
            .special_address(exp_prepare_special_address[lane*8 +: 8]),
            .unscaled_numerator(exp_prepare_unscaled[lane*17 +: 17]),
            .total_shift(exp_prepare_shift[lane*5 +: 5]));
        softmax_lut_address_scale scale (
            .unscaled_numerator(exp_stage0_unscaled[lane*17 +: 17]),
            .numerator(exp_scaled_numerator[lane*25 +: 25]));
        softmax_lut_address_finish finish (
            .special_case(exp_stage1_special[lane]),
            .special_address(exp_stage1_special_address[lane*8 +: 8]),
            .numerator(exp_stage1_numerator[lane*25 +: 25]),
            .total_shift(exp_stage1_shift[lane*5 +: 5]),
            .address(exp_finished_address[lane*8 +: 8]));
    end

    always_ff @(posedge clk) begin
        integer lane;
        if (rst) begin
            exp_stage0_valid <= 1'b0;
            exp_stage1_valid <= 1'b0;
            exp_stage2_valid <= 1'b0;
            exp_stage0_half <= 1'b0;
            exp_stage1_half <= 1'b0;
            exp_stage2_half <= 1'b0;
            exp_input_half_pending <= 1'b0;
            exp_second_half_delta <= '0;
            exp_second_half_mask <= '0;
            exp_second_half_tag <= '0;
            exp_result_valid <= 1'b0;
            exp_result_values <= '0;
            exp_result_mask <= '0;
            exp_result_tag <= '0;
        end else if (exp_pipeline_advance) begin
            exp_result_valid <= exp_stage2_valid && exp_stage2_half;
            if (exp_stage2_valid) begin
                for (lane = 0; lane < EXP_LANES; lane = lane + 1) begin
                    if (exp_stage2_half) begin
                        exp_result_values[(EXP_LANES+lane)*16 +: 16] <=
                            SOFTMAX_EXP_BASE_LUT[
                                exp_stage2_address[lane*8 + 4 +: 4]] +
                            {8'd0, SOFTMAX_EXP_DELTA_LUT[
                                exp_stage2_address[lane*8 +: 8]]};
                        exp_result_mask[EXP_LANES+lane] <= exp_stage2_mask[lane];
                    end else begin
                        exp_result_values[lane*16 +: 16] <=
                            SOFTMAX_EXP_BASE_LUT[
                                exp_stage2_address[lane*8 + 4 +: 4]] +
                            {8'd0, SOFTMAX_EXP_DELTA_LUT[
                                exp_stage2_address[lane*8 +: 8]]};
                        exp_result_mask[lane] <= exp_stage2_mask[lane];
                    end
                end
                if (exp_stage2_half)
                    exp_result_tag <= exp_stage2_tag;
            end
            exp_stage2_valid <= exp_stage1_valid;
            exp_stage2_half <= exp_stage1_half;
            exp_stage2_mask <= exp_stage1_mask;
            exp_stage2_tag <= exp_stage1_tag;
            exp_stage2_address <= exp_finished_address;
            exp_stage1_valid <= exp_stage0_valid;
            exp_stage1_half <= exp_stage0_half;
            exp_stage1_mask <= exp_stage0_mask;
            exp_stage1_tag <= exp_stage0_tag;
            exp_stage1_special <= exp_stage0_special;
            exp_stage1_special_address <= exp_stage0_special_address;
            exp_stage1_numerator <= exp_scaled_numerator;
            exp_stage1_shift <= exp_stage0_shift;
            exp_stage0_valid <= exp_selected_valid;
            if (exp_selected_valid) begin
                exp_stage0_half <= exp_selected_half;
                exp_stage0_mask <= exp_selected_mask;
                exp_stage0_tag <= exp_selected_tag;
                exp_stage0_special <= exp_prepare_special;
                exp_stage0_special_address <= exp_prepare_special_address;
                exp_stage0_unscaled <= exp_prepare_unscaled;
                exp_stage0_shift <= exp_prepare_shift;
            end
            if (exp_input_half_pending) begin
                exp_input_half_pending <= 1'b0;
            end else if (exp_req_valid && exp_req_ready) begin
                exp_input_half_pending <= 1'b1;
                exp_second_half_delta <= exp_req.delta_values[1023:512];
                exp_second_half_mask <= exp_req.lane_mask[63:32];
                exp_second_half_tag <= exp_req.tag;
            end
        end
    end

    logic [RECIPROCAL_ROWS*8-1:0] reciprocal_address;
    logic [RECIPROCAL_ROWS*16-1:0] reciprocal_lut_values;
    logic [RECIPROCAL_ROWS*16-1:0] reciprocal_result_next;
    logic [127:0] reciprocal_result;
    logic [7:0] reciprocal_result_row_mask;
    logic [15:0] reciprocal_result_tag;
    logic reciprocal_input_half_pending;
    logic [RECIPROCAL_ROWS*16-1:0] reciprocal_second_half_sum;
    logic [RECIPROCAL_ROWS-1:0] reciprocal_second_half_mask;
    logic [15:0] reciprocal_second_half_tag;
    logic [RECIPROCAL_ROWS*16-1:0] reciprocal_selected_sum;

    assign reciprocal_req_ready = !reciprocal_input_half_pending &&
        (!reciprocal_rsp_valid || reciprocal_rsp_ready);
    assign reciprocal_selected_sum = reciprocal_input_half_pending ?
        reciprocal_second_half_sum : reciprocal_req.sum_values[63:0];
    assign reciprocal_rsp.values = reciprocal_result;
    assign reciprocal_rsp.row_mask = reciprocal_result_row_mask;
    assign reciprocal_rsp.tag = reciprocal_result_tag;

    for (genvar row = 0; row < RECIPROCAL_ROWS; row = row + 1) begin : g_reciprocal
        assign reciprocal_lut_values[row*16 +: 16] =
            SOFTMAX_RECIPROCAL_LUT[reciprocal_address[row*8 +: 8]];
        softmax_reciprocal_bf16 reciprocal_prepare (
            .sum_value(reciprocal_selected_sum[row*16 +: 16]),
            .lut_value(reciprocal_lut_values[row*16 +: 16]),
            .lut_address(reciprocal_address[row*8 +: 8]),
            .reciprocal(reciprocal_result_next[row*16 +: 16]));
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            reciprocal_rsp_valid <= 1'b0;
            reciprocal_result <= '0;
            reciprocal_result_row_mask <= '0;
            reciprocal_result_tag <= '0;
            reciprocal_input_half_pending <= 1'b0;
            reciprocal_second_half_sum <= '0;
            reciprocal_second_half_mask <= '0;
            reciprocal_second_half_tag <= '0;
        end else if (!reciprocal_rsp_valid || reciprocal_rsp_ready) begin
            reciprocal_rsp_valid <= reciprocal_input_half_pending;
            if (reciprocal_input_half_pending) begin
                reciprocal_result[127:64] <= reciprocal_result_next;
                reciprocal_result_row_mask[7:4] <=
                    reciprocal_second_half_mask;
                reciprocal_result_tag <= reciprocal_second_half_tag;
                reciprocal_input_half_pending <= 1'b0;
            end else if (reciprocal_req_valid && reciprocal_req_ready) begin
                reciprocal_result[63:0] <= reciprocal_result_next;
                reciprocal_result_row_mask[3:0] <= reciprocal_req.row_mask[3:0];
                reciprocal_second_half_sum <= reciprocal_req.sum_values[127:64];
                reciprocal_second_half_mask <= reciprocal_req.row_mask[7:4];
                reciprocal_second_half_tag <= reciprocal_req.tag;
                reciprocal_input_half_pending <= 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin : check_exp_lut_factorization
        for (int address = 0; address < 256; address = address + 1) begin
            assert (SOFTMAX_EXP_BASE_LUT[address >> 4] +
                    {8'd0, SOFTMAX_EXP_DELTA_LUT[address]} ==
                    SOFTMAX_EXP_LUT[address])
                else $error("softmax exp LUT factorization mismatch at address %0d",
                            address);
        end
    end

    assert property (@(posedge clk) disable iff (rst)
        exp_rsp_valid && !exp_rsp_ready |=>
            exp_rsp_valid && $stable(exp_rsp))
        else $error("softmax exp response changed while stalled");
    assert property (@(posedge clk) disable iff (rst)
        reciprocal_rsp_valid && !reciprocal_rsp_ready |=>
            reciprocal_rsp_valid && $stable(reciprocal_rsp))
        else $error("softmax reciprocal response changed while stalled");
`endif
endmodule

`default_nettype wire
