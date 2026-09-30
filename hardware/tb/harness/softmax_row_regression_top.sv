`default_nettype none

module softmax_row_regression_top (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    input  logic          hold_scratch_response,
    output logic          scratch_read_accepted,
    output logic          scratch_response_accepted,
    output logic          quant_scale_accepted,
    output logic          quant_scale_response_accepted,
    output logic          abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [3:0]    row_count,
    input  logic [11:0]   row_length,
    input  logic          capture_probability_enable,
    output logic          capture_probability_valid,
    input  logic          capture_probability_ready,
    output logic          source_req_valid,
    input  logic          source_req_ready,
    output logic [1:0]    source_req_pass,
    output logic [11:0]   source_req_key,
    output logic [15:0]   source_req_tag,
    input  logic          source_rsp_valid,
    output logic          source_rsp_ready,
    input  logic [1023:0] source_rsp_values,
    input  logic [63:0]   source_rsp_lane_mask,
    input  logic [15:0]   source_rsp_tag,
    output logic          bf16_write_valid,
    input  logic          bf16_write_ready,
    output logic          bf16_write_probability,
    output logic [11:0]   bf16_write_key,
    output logic [1023:0] bf16_write_values,
    output logic [63:0]   bf16_write_lane_mask,
    output logic [15:0]   bf16_write_tag,
    output logic          quantized_write_valid,
    input  logic          quantized_write_ready,
    output logic [11:0]   quantized_write_key,
    output logic [511:0]  quantized_write_values,
    output logic [63:0]   quantized_write_lane_mask,
    output logic [15:0]   quantized_write_tag,
    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [127:0]  scale_write_values,
    output logic [7:0]    scale_write_row_mask,
    output logic          done,
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
    logic scratch_read_valid, scratch_read_ready, scratch_read_bank;
    logic [9:0] scratch_read_left_address, scratch_read_right_address;
    logic [15:0] scratch_read_tag;
    logic scratch_rsp_valid, scratch_rsp_ready;
    logic [127:0] scratch_rsp_left_data, scratch_rsp_right_data;
    logic [15:0] scratch_rsp_tag;
    logic scratch_write_valid, scratch_write_ready, scratch_write_bank;
    logic [9:0] scratch_write_address;
    logic [127:0] scratch_write_data;
    logic [15:0] scratch_write_byte_enable;
    logic [127:0] scratch_memory [0:1][0:1023];
    logic scratch_response_pending;
    logic scratch_response_bank;
    logic [9:0] scratch_response_left_address;
    logic [9:0] scratch_response_right_address;
    logic [15:0] scratch_response_tag;

    logic vector_req_valid, vector_req_ready;
    logic [2:0] vector_req_operation;
    logic [1023:0] vector_req_values, vector_req_paired_values;
    logic [1023:0] vector_req_factor0_values, vector_req_factor1_values;
    logic [63:0] vector_req_lane_mask;
    logic [15:0] vector_req_tag;
    logic vector_rsp_valid, vector_rsp_ready;
    logic [1023:0] vector_rsp_values;
    logic [63:0] vector_rsp_lane_mask;
    logic [15:0] vector_rsp_tag;

    logic max_req_valid, max_req_ready, max_req_magnitude;
    logic [1023:0] max_req_values;
    logic [63:0] max_req_lane_mask;
    logic [15:0] max_req_tag;
    logic max_rsp_valid, max_rsp_ready;
    logic [127:0] max_rsp_values;
    logic [7:0] max_rsp_row_mask;
    logic [15:0] max_rsp_tag;

    logic reduction_req_valid, reduction_req_ready;
    logic [1023:0] reduction_req_values;
    logic [63:0] reduction_req_lane_mask;
    logic [15:0] reduction_req_tag;
    logic reduction_rsp_valid, reduction_rsp_ready;
    logic [127:0] reduction_rsp_values;
    logic [7:0] reduction_rsp_row_mask;
    logic [15:0] reduction_rsp_tag;

    logic exp_req_valid, exp_req_ready, exp_rsp_valid, exp_rsp_ready;
    logic reciprocal_req_valid, reciprocal_req_ready;
    logic reciprocal_rsp_valid, reciprocal_rsp_ready;
    hardware_types_pkg::softmax_exp_request_t exp_req;
    hardware_types_pkg::softmax_exp_response_t exp_rsp;
    hardware_types_pkg::softmax_reciprocal_request_t reciprocal_req;
    hardware_types_pkg::softmax_reciprocal_response_t reciprocal_rsp;

    logic quant_scale_req_valid, quant_scale_req_ready;
    logic [127:0] quant_scale_req_row_max_abs;
    logic [7:0] quant_scale_req_row_mask;
    logic quant_scale_rsp_valid, quant_scale_rsp_ready;
    logic [127:0] quant_scale_rsp_values;
    logic quant_values_req_valid, quant_values_req_ready;
    logic [1023:0] quant_values_req_values;
    logic [63:0] quant_values_req_lane_mask;
    logic [15:0] quant_values_req_tag;
    logic quant_values_rsp_valid, quant_values_rsp_ready;
    logic [511:0] quant_values_rsp_values;
    logic [63:0] quant_values_rsp_lane_mask;
    logic [15:0] quant_values_rsp_tag;

    assign scratch_read_ready = !scratch_response_pending ||
        (scratch_rsp_valid && scratch_rsp_ready);
    assign scratch_write_ready = 1'b1;
    assign scratch_rsp_valid = scratch_response_pending && !hold_scratch_response;
    assign scratch_read_accepted = scratch_read_valid && scratch_read_ready;
    assign scratch_response_accepted = scratch_rsp_valid && scratch_rsp_ready;
    assign quant_scale_accepted = quant_scale_req_valid && quant_scale_req_ready;
    assign quant_scale_response_accepted = quant_scale_rsp_valid && quant_scale_rsp_ready;
    assign scratch_rsp_left_data = scratch_memory[scratch_response_bank]
        [scratch_response_left_address];
    assign scratch_rsp_right_data = scratch_memory[scratch_response_bank]
        [scratch_response_right_address];
    assign scratch_rsp_tag = scratch_response_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            scratch_response_pending <= 1'b0;
            scratch_response_bank <= 1'b0;
            scratch_response_left_address <= '0;
            scratch_response_right_address <= '0;
            scratch_response_tag <= '0;
        end else begin
            if (scratch_rsp_valid && scratch_rsp_ready)
                scratch_response_pending <= 1'b0;
            if (scratch_read_valid && scratch_read_ready) begin
                scratch_response_pending <= 1'b1;
                scratch_response_bank <= scratch_read_bank;
                scratch_response_left_address <= scratch_read_left_address;
                scratch_response_right_address <= scratch_read_right_address;
                scratch_response_tag <= scratch_read_tag;
            end
            if (scratch_write_valid && scratch_write_ready)
                for (integer byte_index = 0; byte_index < 16;
                     byte_index = byte_index + 1)
                    if (scratch_write_byte_enable[byte_index])
                        scratch_memory[scratch_write_bank][scratch_write_address]
                            [byte_index*8 +: 8] <=
                            scratch_write_data[byte_index*8 +: 8];
        end
    end

    softmax_row_pipeline pipeline (
        .capture_probability_enable, .capture_probability_valid, .capture_probability_ready,
        .clk(clk), .rst(rst), .abort_request(abort_request),
        .abort_ack(abort_ack), .start_valid(start_valid),
        .start_ready(start_ready), .row_count(row_count),
        .row_length(row_length), .source_req_valid(source_req_valid),
        .source_req_ready(source_req_ready), .source_req_pass(source_req_pass),
        .source_req_key(source_req_key), .source_req_tag(source_req_tag),
        .source_rsp_valid(source_rsp_valid), .source_rsp_ready(source_rsp_ready),
        .source_rsp_values(source_rsp_values),
        .source_rsp_lane_mask(source_rsp_lane_mask),
        .source_rsp_tag(source_rsp_tag), .bf16_write_valid(bf16_write_valid),
        .bf16_write_ready(bf16_write_ready),
        .bf16_write_probability(bf16_write_probability),
        .bf16_write_key(bf16_write_key), .bf16_write_values(bf16_write_values),
        .bf16_write_lane_mask(bf16_write_lane_mask),
        .bf16_write_tag(bf16_write_tag), .quantized_write_valid(quantized_write_valid),
        .quantized_write_ready(quantized_write_ready), .quantized_write_key(quantized_write_key),
        .quantized_write_values(quantized_write_values),
        .quantized_write_lane_mask(quantized_write_lane_mask),
        .quantized_write_tag(quantized_write_tag), .scale_write_valid(scale_write_valid),
        .scale_write_ready(scale_write_ready),
        .scale_write_values(scale_write_values),
        .scale_write_row_mask(scale_write_row_mask),
        .scratch_read_valid(scratch_read_valid),
        .scratch_read_ready(scratch_read_ready),
        .scratch_read_bank(scratch_read_bank),
        .scratch_read_left_address(scratch_read_left_address),
        .scratch_read_right_address(scratch_read_right_address),
        .scratch_read_tag(scratch_read_tag), .scratch_rsp_valid(scratch_rsp_valid),
        .scratch_rsp_ready(scratch_rsp_ready),
        .scratch_rsp_left_data(scratch_rsp_left_data),
        .scratch_rsp_right_data(scratch_rsp_right_data),
        .scratch_rsp_tag(scratch_rsp_tag),
        .scratch_write_valid(scratch_write_valid),
        .scratch_write_ready(scratch_write_ready),
        .scratch_write_bank(scratch_write_bank),
        .scratch_write_address(scratch_write_address),
        .scratch_write_data(scratch_write_data),
        .scratch_write_byte_enable(scratch_write_byte_enable),
        .vector_req_valid(vector_req_valid), .vector_req_ready(vector_req_ready),
        .vector_req_operation(vector_req_operation),
        .vector_req_values(vector_req_values),
        .vector_req_paired_values(vector_req_paired_values),
        .vector_req_factor0_values(vector_req_factor0_values),
        .vector_req_factor1_values(vector_req_factor1_values),
        .vector_req_lane_mask(vector_req_lane_mask), .vector_req_tag(vector_req_tag),
        .vector_rsp_valid(vector_rsp_valid), .vector_rsp_ready(vector_rsp_ready),
        .vector_rsp_values(vector_rsp_values),
        .vector_rsp_lane_mask(vector_rsp_lane_mask), .vector_rsp_tag(vector_rsp_tag),
        .max_req_valid(max_req_valid), .max_req_ready(max_req_ready),
        .max_req_magnitude(max_req_magnitude), .max_req_values(max_req_values),
        .max_req_lane_mask(max_req_lane_mask), .max_req_tag(max_req_tag),
        .max_rsp_valid(max_rsp_valid), .max_rsp_ready(max_rsp_ready),
        .max_rsp_values(max_rsp_values), .max_rsp_row_mask(max_rsp_row_mask),
        .max_rsp_tag(max_rsp_tag), .reduction_req_valid(reduction_req_valid),
        .reduction_req_ready(reduction_req_ready),
        .reduction_req_values(reduction_req_values),
        .reduction_req_lane_mask(reduction_req_lane_mask),
        .reduction_req_tag(reduction_req_tag),
        .reduction_rsp_valid(reduction_rsp_valid),
        .reduction_rsp_ready(reduction_rsp_ready),
        .reduction_rsp_values(reduction_rsp_values),
        .reduction_rsp_row_mask(reduction_rsp_row_mask),
        .reduction_rsp_tag(reduction_rsp_tag),
        .exp_req_valid(exp_req_valid), .exp_req_ready(exp_req_ready),
        .exp_req_delta_values(exp_req.delta_values),
        .exp_req_lane_mask(exp_req.lane_mask), .exp_req_tag(exp_req.tag),
        .exp_rsp_valid(exp_rsp_valid), .exp_rsp_ready(exp_rsp_ready),
        .exp_rsp_values(exp_rsp.values),
        .exp_rsp_lane_mask(exp_rsp.lane_mask), .exp_rsp_tag(exp_rsp.tag),
        .reciprocal_req_valid(reciprocal_req_valid),
        .reciprocal_req_ready(reciprocal_req_ready),
        .reciprocal_req_sum_values(reciprocal_req.sum_values),
        .reciprocal_req_row_mask(reciprocal_req.row_mask),
        .reciprocal_req_tag(reciprocal_req.tag),
        .reciprocal_rsp_valid(reciprocal_rsp_valid),
        .reciprocal_rsp_ready(reciprocal_rsp_ready),
        .reciprocal_rsp_values(reciprocal_rsp.values),
        .reciprocal_rsp_row_mask(reciprocal_rsp.row_mask),
        .reciprocal_rsp_tag(reciprocal_rsp.tag),
        .quant_scale_req_valid(quant_scale_req_valid),
        .quant_scale_req_ready(quant_scale_req_ready),
        .quant_scale_req_row_max_abs(quant_scale_req_row_max_abs),
        .quant_scale_req_row_mask(quant_scale_req_row_mask),
        .quant_scale_rsp_valid(quant_scale_rsp_valid),
        .quant_scale_rsp_ready(quant_scale_rsp_ready),
        .quant_scale_rsp_values(quant_scale_rsp_values),
        .quant_values_req_valid(quant_values_req_valid),
        .quant_values_req_ready(quant_values_req_ready),
        .quant_values_req_values(quant_values_req_values),
        .quant_values_req_lane_mask(quant_values_req_lane_mask),
        .quant_values_req_tag(quant_values_req_tag),
        .quant_values_rsp_valid(quant_values_rsp_valid),
        .quant_values_rsp_ready(quant_values_rsp_ready),
        .quant_values_rsp_values(quant_values_rsp_values),
        .quant_values_rsp_lane_mask(quant_values_rsp_lane_mask),
        .quant_values_rsp_tag(quant_values_rsp_tag), .done_pulse(done),
        .error(error),
        .accepted_source_tile_count(accepted_source_tile_count),
        .accepted_bf16_write_count(accepted_bf16_write_count),
        .accepted_quantized_write_count(accepted_quantized_write_count),
        .accepted_scratch_read_count(accepted_scratch_read_count),
        .accepted_scratch_write_count(accepted_scratch_write_count),
        .accepted_vector_request_count(accepted_vector_request_count),
        .accepted_max_request_count(accepted_max_request_count),
        .accepted_reduction_request_count(accepted_reduction_request_count),
        .accepted_quantized_value_count(accepted_quantized_value_count));

    bf16_vector_pipe #(.LANES(64), .TAG_WIDTH(16)) vector_pipe (
        .clk(clk), .rst(rst), .abort_request(1'b0), .abort_ack(),
        .req_valid(vector_req_valid),
        .req_ready(vector_req_ready), .req_operation(vector_req_operation),
        .req_values(vector_req_values), .req_paired_values(vector_req_paired_values),
        .req_factor0_values(vector_req_factor0_values),
        .req_factor1_values(vector_req_factor1_values),
        .req_lane_mask(vector_req_lane_mask), .req_tag(vector_req_tag),
        .rsp_valid(vector_rsp_valid), .rsp_ready(vector_rsp_ready),
        .rsp_values(vector_rsp_values), .rsp_lane_mask(vector_rsp_lane_mask),
        .rsp_tag(vector_rsp_tag), .idle(), .trace_sample_valid(),
        .trace_sample_stage(), .trace_sample_values(), .trace_sample_lane_mask(),
        .trace_sample_tag());

    bf16_tile_max_pipe #(.TAG_WIDTH(16)) max_pipe (
        .clk(clk), .rst(rst), .req_valid(max_req_valid), .req_ready(max_req_ready),
        .req_magnitude(max_req_magnitude), .req_values(max_req_values),
        .req_lane_mask(max_req_lane_mask), .req_tag(max_req_tag),
        .rsp_valid(max_rsp_valid), .rsp_ready(max_rsp_ready),
        .rsp_values(max_rsp_values), .rsp_row_mask(max_rsp_row_mask),
        .rsp_tag(max_rsp_tag));

    bf16_tile_reduction_pipe #(.TAG_WIDTH(16)) reduction_pipe (
        .clk(clk), .rst(rst), .req_valid(reduction_req_valid),
        .req_ready(reduction_req_ready), .req_values(reduction_req_values),
        .req_lane_mask(reduction_req_lane_mask), .req_tag(reduction_req_tag),
        .rsp_valid(reduction_rsp_valid), .rsp_ready(reduction_rsp_ready),
        .rsp_values(reduction_rsp_values),
        .rsp_row_mask(reduction_rsp_row_mask), .rsp_tag(reduction_rsp_tag));

    softmax_lut_resource lut_resource (
        .clk(clk), .rst(rst), .exp_req_valid(exp_req_valid),
        .exp_req_ready(exp_req_ready), .exp_req(exp_req),
        .exp_rsp_valid(exp_rsp_valid), .exp_rsp_ready(exp_rsp_ready),
        .exp_rsp(exp_rsp), .reciprocal_req_valid(reciprocal_req_valid),
        .reciprocal_req_ready(reciprocal_req_ready),
        .reciprocal_req(reciprocal_req),
        .reciprocal_rsp_valid(reciprocal_rsp_valid),
        .reciprocal_rsp_ready(reciprocal_rsp_ready),
        .reciprocal_rsp(reciprocal_rsp));

    activation_quantizer #(.LANES(64), .TAG_WIDTH(16)) quantizer (
        .clk(clk), .rst(rst), .abort_request(1'b0), .abort_ack(),
        .scale_req_valid(quant_scale_req_valid),
        .scale_req_ready(quant_scale_req_ready), .scale_req_a4_row_mask(8'h00),
        .scale_req_clip_ratio_bf16(16'd0),
        .scale_req_use_static_scale(1'b0), .scale_req_static_scales_bf16('0),
        .scale_req_row_max_abs(quant_scale_req_row_max_abs),
        .scale_req_row_mask(quant_scale_req_row_mask),
        .scale_rsp_valid(quant_scale_rsp_valid),
        .scale_rsp_ready(quant_scale_rsp_ready),
        .scale_rsp_values_bf16(quant_scale_rsp_values),
        .quantized_req_valid(quant_values_req_valid), .quantized_req_ready(quant_values_req_ready),
        .quantized_req_values_bf16(quant_values_req_values),
        .quantized_req_lane_mask(quant_values_req_lane_mask),
        .quantized_req_tag(quant_values_req_tag), .quantized_rsp_valid(quant_values_rsp_valid),
        .quantized_rsp_ready(quant_values_rsp_ready), .quantized_rsp_values(quant_values_rsp_values),
        .quantized_rsp_lane_mask(quant_values_rsp_lane_mask),
        .quantized_rsp_tag(quant_values_rsp_tag), .idle());
endmodule

`default_nettype wire
