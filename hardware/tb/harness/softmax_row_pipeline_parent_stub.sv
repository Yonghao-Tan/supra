`default_nettype none

// Verification-only batch endpoint for checking softmax_engine iteration.
// The production row pipeline is tested separately with its numeric resources.
module softmax_row_pipeline #(
    parameter integer MAX_LENGTH = 2048,
    parameter integer DIM_WIDTH = 12,
    parameter integer TAG_WIDTH = 16
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [3:0]    row_count,
    input  logic [DIM_WIDTH-1:0] row_length,
    input  logic          capture_probability_enable,
    output logic          capture_probability_valid,
    input  logic          capture_probability_ready,
    output logic          source_req_valid,
    input  logic          source_req_ready,
    output logic [1:0]    source_req_pass,
    output logic [DIM_WIDTH-1:0] source_req_key,
    output logic [TAG_WIDTH-1:0] source_req_tag,
    input  logic          source_rsp_valid,
    output logic          source_rsp_ready,
    input  logic [1023:0] source_rsp_values,
    input  logic [63:0]   source_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] source_rsp_tag,
    output logic          bf16_write_valid,
    input  logic          bf16_write_ready,
    output logic          bf16_write_probability,
    output logic [DIM_WIDTH-1:0] bf16_write_key,
    output logic [1023:0] bf16_write_values,
    output logic [63:0]   bf16_write_lane_mask,
    output logic [TAG_WIDTH-1:0] bf16_write_tag,
    output logic          quantized_write_valid,
    input  logic          quantized_write_ready,
    output logic [DIM_WIDTH-1:0] quantized_write_key,
    output logic [511:0]  quantized_write_values,
    output logic [63:0]   quantized_write_lane_mask,
    output logic [TAG_WIDTH-1:0] quantized_write_tag,
    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [127:0]  scale_write_values,
    output logic [7:0]    scale_write_row_mask,
    output logic          scratch_read_valid,
    input  logic          scratch_read_ready,
    output logic          scratch_read_bank,
    output logic [9:0]    scratch_read_left_address,
    output logic [9:0]    scratch_read_right_address,
    output logic [TAG_WIDTH-1:0] scratch_read_tag,
    input  logic          scratch_rsp_valid,
    output logic          scratch_rsp_ready,
    input  logic [127:0]  scratch_rsp_left_data,
    input  logic [127:0]  scratch_rsp_right_data,
    input  logic [TAG_WIDTH-1:0] scratch_rsp_tag,
    output logic          scratch_write_valid,
    input  logic          scratch_write_ready,
    output logic          scratch_write_bank,
    output logic [9:0]    scratch_write_address,
    output logic [127:0]  scratch_write_data,
    output logic [15:0]   scratch_write_byte_enable,
    output logic          vector_req_valid,
    input  logic          vector_req_ready,
    output logic [2:0]    vector_req_operation,
    output logic [1023:0] vector_req_values,
    output logic [1023:0] vector_req_paired_values,
    output logic [1023:0] vector_req_factor0_values,
    output logic [1023:0] vector_req_factor1_values,
    output logic [63:0]   vector_req_lane_mask,
    output logic [TAG_WIDTH-1:0] vector_req_tag,
    input  logic          vector_rsp_valid,
    output logic          vector_rsp_ready,
    input  logic [1023:0] vector_rsp_values,
    input  logic [63:0]   vector_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] vector_rsp_tag,
    output logic          max_req_valid,
    input  logic          max_req_ready,
    output logic          max_req_magnitude,
    output logic [1023:0] max_req_values,
    output logic [63:0]   max_req_lane_mask,
    output logic [TAG_WIDTH-1:0] max_req_tag,
    input  logic          max_rsp_valid,
    output logic          max_rsp_ready,
    input  logic [127:0]  max_rsp_values,
    input  logic [7:0]    max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] max_rsp_tag,
    output logic          reduction_req_valid,
    input  logic          reduction_req_ready,
    output logic [1023:0] reduction_req_values,
    output logic [63:0]   reduction_req_lane_mask,
    output logic [TAG_WIDTH-1:0] reduction_req_tag,
    input  logic          reduction_rsp_valid,
    output logic          reduction_rsp_ready,
    input  logic [127:0]  reduction_rsp_values,
    input  logic [7:0]    reduction_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] reduction_rsp_tag,
    output logic          exp_req_valid,
    input  logic          exp_req_ready,
    output logic [1023:0] exp_req_delta_values,
    output logic [63:0]   exp_req_lane_mask,
    output logic [TAG_WIDTH-1:0] exp_req_tag,
    input  logic          exp_rsp_valid,
    output logic          exp_rsp_ready,
    input  logic [1023:0] exp_rsp_values,
    input  logic [63:0]   exp_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] exp_rsp_tag,
    output logic          reciprocal_req_valid,
    input  logic          reciprocal_req_ready,
    output logic [127:0]  reciprocal_req_sum_values,
    output logic [7:0]    reciprocal_req_row_mask,
    output logic [TAG_WIDTH-1:0] reciprocal_req_tag,
    input  logic          reciprocal_rsp_valid,
    output logic          reciprocal_rsp_ready,
    input  logic [127:0]  reciprocal_rsp_values,
    input  logic [7:0]    reciprocal_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] reciprocal_rsp_tag,
    output logic          quant_scale_req_valid,
    input  logic          quant_scale_req_ready,
    output logic [127:0]  quant_scale_req_row_max_abs,
    output logic [7:0]    quant_scale_req_row_mask,
    input  logic          quant_scale_rsp_valid,
    output logic          quant_scale_rsp_ready,
    input  logic [127:0]  quant_scale_rsp_values,
    output logic          quant_values_req_valid,
    input  logic          quant_values_req_ready,
    output logic [1023:0] quant_values_req_values,
    output logic [63:0]   quant_values_req_lane_mask,
    output logic [TAG_WIDTH-1:0] quant_values_req_tag,
    input  logic          quant_values_rsp_valid,
    output logic          quant_values_rsp_ready,
    input  logic [511:0]  quant_values_rsp_values,
    input  logic [63:0]   quant_values_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] quant_values_rsp_tag,
    output logic          done_pulse,
    output logic          error,
    output logic [63:0]   accepted_source_tile_count,
    output logic [63:0]   accepted_bf16_write_count,
    output logic [63:0]   accepted_quantized_write_count,
    output logic [63:0]   accepted_scratch_read_count,
    output logic [63:0]   accepted_scratch_write_count,
    output logic [63:0]   accepted_vector_request_count,
    output logic [63:0]   accepted_max_request_count,
    output logic [63:0]   accepted_reduction_request_count,
    output logic [63:0]   accepted_quantized_value_count
);
    logic active;
    logic request_pending;
    logic completion_pending;

    assign start_ready = !active && !abort_request;
    assign source_req_valid = request_pending;
    assign capture_probability_valid = 1'b0;
    assign source_req_pass = 2'd0;
    assign source_req_key = '0;
    assign source_req_tag = TAG_WIDTH'(16'ha55a);
    assign source_rsp_ready = 1'b0;
    assign bf16_write_valid = 1'b0;
    assign bf16_write_probability = 1'b0;
    assign bf16_write_key = '0;
    assign bf16_write_values = '0;
    assign bf16_write_lane_mask = '0;
    assign bf16_write_tag = '0;
    assign quantized_write_valid = 1'b0;
    assign quantized_write_key = '0;
    assign quantized_write_values = '0;
    assign quantized_write_lane_mask = '0;
    assign quantized_write_tag = '0;
    assign scale_write_valid = 1'b0;
    assign scale_write_values = '0;
    assign scale_write_row_mask = '0;
    assign scratch_read_valid = 1'b0;
    assign scratch_read_bank = 1'b0;
    assign scratch_read_left_address = '0;
    assign scratch_read_right_address = '0;
    assign scratch_read_tag = '0;
    assign scratch_rsp_ready = 1'b0;
    assign scratch_write_valid = 1'b0;
    assign scratch_write_bank = 1'b0;
    assign scratch_write_address = '0;
    assign scratch_write_data = '0;
    assign scratch_write_byte_enable = '0;
    assign vector_req_valid = 1'b0;
    assign vector_req_operation = '0;
    assign vector_req_values = '0;
    assign vector_req_paired_values = '0;
    assign vector_req_factor0_values = '0;
    assign vector_req_factor1_values = '0;
    assign vector_req_lane_mask = '0;
    assign vector_req_tag = '0;
    assign vector_rsp_ready = 1'b0;
    assign max_req_valid = 1'b0;
    assign max_req_magnitude = 1'b0;
    assign max_req_values = '0;
    assign max_req_lane_mask = '0;
    assign max_req_tag = '0;
    assign max_rsp_ready = 1'b0;
    assign reduction_req_valid = 1'b0;
    assign reduction_req_values = '0;
    assign reduction_req_lane_mask = '0;
    assign reduction_req_tag = '0;
    assign reduction_rsp_ready = 1'b0;
    assign exp_req_valid = 1'b0;
    assign exp_req_delta_values = '0;
    assign exp_req_lane_mask = '0;
    assign exp_req_tag = '0;
    assign exp_rsp_ready = 1'b0;
    assign reciprocal_req_valid = 1'b0;
    assign reciprocal_req_sum_values = '0;
    assign reciprocal_req_row_mask = '0;
    assign reciprocal_req_tag = '0;
    assign reciprocal_rsp_ready = 1'b0;
    assign quant_scale_req_valid = 1'b0;
    assign quant_scale_req_row_max_abs = '0;
    assign quant_scale_req_row_mask = '0;
    assign quant_scale_rsp_ready = 1'b0;
    assign quant_values_req_valid = 1'b0;
    assign quant_values_req_values = '0;
    assign quant_values_req_lane_mask = '0;
    assign quant_values_req_tag = '0;
    assign quant_values_rsp_ready = 1'b0;
    assign error = 1'b0;
    assign accepted_source_tile_count = '0;
    assign accepted_bf16_write_count = '0;
    assign accepted_quantized_write_count = '0;
    assign accepted_scratch_read_count = '0;
    assign accepted_scratch_write_count = '0;
    assign accepted_vector_request_count = '0;
    assign accepted_max_request_count = '0;
    assign accepted_reduction_request_count = '0;
    assign accepted_quantized_value_count = '0;

    always_ff @(posedge clk) begin
        if (rst) begin
            active <= 1'b0;
            request_pending <= 1'b0;
            completion_pending <= 1'b0;
            done_pulse <= 1'b0;
            abort_ack <= 1'b0;
        end else begin
            done_pulse <= 1'b0;
            abort_ack <= 1'b0;
            if (abort_request && active) begin
                active <= 1'b0;
                request_pending <= 1'b0;
                completion_pending <= 1'b0;
                abort_ack <= 1'b1;
            end else if (start_valid && start_ready) begin
                active <= 1'b1;
                request_pending <= 1'b1;
            end else if (request_pending && source_req_ready) begin
                request_pending <= 1'b0;
                completion_pending <= 1'b1;
            end else if (completion_pending) begin
                active <= 1'b0;
                completion_pending <= 1'b0;
                done_pulse <= 1'b1;
            end
        end
    end

    initial begin
        if (MAX_LENGTH != 2048 || DIM_WIDTH != 12 || TAG_WIDTH != 16)
            $error("softmax parent stub parameter mismatch");
    end
endmodule

`default_nettype wire
