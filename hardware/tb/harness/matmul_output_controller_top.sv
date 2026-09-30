`default_nettype none

module matmul_output_controller_top (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    input  logic          cfg_valid,
    output logic          cfg_ready,
    input  logic [2:0]    cfg_mode,
    input  logic [5:0]    cfg_physical_row_base,
    input  logic [5:0]    cfg_row_count,
    input  logic [2:0]    cfg_batch_count,
    input  logic [5:0]    cfg_second_batch_rows,
    input  logic [5:0]    cfg_third_batch_rows,
    input  logic [5:0]    cfg_fourth_batch_rows,
    input  logic [5:0]    cfg_fifth_batch_rows,
    input  logic [5:0]    cfg_sixth_batch_rows,
    input  logic [15:0]   cfg_output_channels,
    input  logic [63:0]   cfg_output_base,
    input  logic [63:0]   cfg_output_limit,
    input  logic [31:0]   cfg_local_row_stride,
    input  logic          cfg_residual_from_ddr,
    input  logic [63:0]   cfg_residual_base,
    input  logic [63:0]   cfg_residual_limit,
    input  logic [5:0]    cfg_residual_storage_rows,
    input  logic          in_valid,
    input  logic [2:0]    in_batch_index,
    output logic          in_ready,
    input  logic [5:0]    in_physical_row,
    input  logic [15:0]   in_output_channel,
    input  logic [63:0]   in_row_byte_base,
    input  logic [127:0]  in_data,
    input  logic [15:0]   in_byte_enable,

    output logic          residual_read_request_valid,
    input  logic          residual_read_request_ready,
    output logic [63:0]   residual_read_request_address,
    output logic [31:0]   residual_read_request_bytes,
    output logic [7:0]    residual_read_request_tag,
    output logic          residual_read_local,
    input  logic          residual_read_request_done,
    input  logic          residual_read_request_error,
    input  logic          residual_read_data_valid,
    output logic          residual_read_data_ready,
    input  logic [127:0]  residual_read_data,
    input  logic [15:0]   residual_read_byte_enable,
    input  logic          residual_read_data_last,
    input  logic [7:0]    residual_read_data_tag,

    output logic          local_write_valid,
    input  logic          local_write_ready,
    output logic [63:0]   local_write_byte_address,
    output logic [127:0]  local_write_data,
    output logic [15:0]   local_write_byte_enable,
    output logic          output_write_valid,
    input  logic          output_write_ready,
    output logic [63:0]   output_write_byte_address,
    output logic [31:0]   output_write_transaction_bytes,
    output logic [127:0]  output_write_data,
    output logic [15:0]   output_write_byte_enable,
    output logic          output_write_first,
    output logic          output_write_last,
    input  logic          output_write_done,
    input  logic          output_write_error,

    output logic          arithmetic_trace_sample_valid,
    output logic [255:0]  arithmetic_trace_sample_values,
    output logic [15:0]   arithmetic_trace_sample_lane_mask,
    output logic [15:0]   arithmetic_trace_sample_tag,
    output logic          row_commit,
    output logic          done,
    output logic          busy,
    output logic          error,
    output logic [3:0]    error_id,

    output logic          trace_staging_write_valid,
    output logic          trace_staging_write_ready,
    output logic [9:0]    trace_staging_write_address,
    output logic          trace_staging_read_valid,
    output logic          trace_staging_read_ready,
    output logic [9:0]    trace_staging_read_address,
    output logic          trace_staging_read_response_valid,
    output logic          trace_arithmetic_req_valid,
    output logic          trace_arithmetic_req_ready,
    output logic [15:0]   trace_arithmetic_req_tag,
    output logic          trace_arithmetic_rsp_valid,
    output logic          trace_arithmetic_rsp_ready,
    output logic [15:0]   trace_arithmetic_rsp_tag,
    output logic [4:0]    trace_output_state
);
    logic staging_write_valid, staging_write_ready;
    logic [9:0] staging_write_address;
    logic [127:0] staging_write_data;
    logic [15:0] staging_write_byte_enable;
    logic staging_read_request_valid, staging_read_request_ready;
    logic [9:0] staging_read_address;
    logic staging_read_response_valid;
    logic [127:0] staging_read_data;
    logic arithmetic_req_valid, arithmetic_req_ready;
    logic [2:0] arithmetic_req_operation;
    logic [255:0] arithmetic_req_values;
    logic [255:0] arithmetic_req_paired_values;
    logic [255:0] arithmetic_req_factor0_values;
    logic [255:0] arithmetic_req_factor1_values;
    logic [15:0] arithmetic_req_lane_mask, arithmetic_req_tag;
    logic arithmetic_rsp_valid, arithmetic_rsp_ready;
    logic [255:0] arithmetic_rsp_values;
    logic [15:0] arithmetic_rsp_lane_mask, arithmetic_rsp_tag;
    logic arithmetic_abort_ack;
    logic unused_residual_ddr_stage;
    logic [1:0] unused_trace_sample_stage;

    assign trace_staging_write_valid = staging_write_valid;
    assign trace_staging_write_ready = staging_write_ready;
    assign trace_staging_write_address = staging_write_address;
    assign trace_staging_read_valid = staging_read_request_valid;
    assign trace_staging_read_ready = staging_read_request_ready;
    assign trace_staging_read_address = staging_read_address;
    assign trace_staging_read_response_valid = staging_read_response_valid;
    assign trace_arithmetic_req_valid = arithmetic_req_valid;
    assign trace_arithmetic_req_ready = arithmetic_req_ready;
    assign trace_arithmetic_req_tag = arithmetic_req_tag;
    assign trace_arithmetic_rsp_valid = arithmetic_rsp_valid;
    assign trace_arithmetic_rsp_ready = arithmetic_rsp_ready;
    assign trace_arithmetic_rsp_tag = arithmetic_rsp_tag;
    assign trace_output_state = output_controller.state;

    local_sram_macro #(.DEPTH(1024)) staging_memory (
        .clk, .rst,
        .a_req_valid(staging_write_valid),
        .a_req_ready(staging_write_ready),
        .a_write(1'b1),
        .a_address({2'd0, staging_write_address}),
        .a_write_data(staging_write_data),
        .a_write_mask_n(~{{8{staging_write_byte_enable[15]}},
            {8{staging_write_byte_enable[14]}}, {8{staging_write_byte_enable[13]}},
            {8{staging_write_byte_enable[12]}}, {8{staging_write_byte_enable[11]}},
            {8{staging_write_byte_enable[10]}}, {8{staging_write_byte_enable[9]}},
            {8{staging_write_byte_enable[8]}}, {8{staging_write_byte_enable[7]}},
            {8{staging_write_byte_enable[6]}}, {8{staging_write_byte_enable[5]}},
            {8{staging_write_byte_enable[4]}}, {8{staging_write_byte_enable[3]}},
            {8{staging_write_byte_enable[2]}}, {8{staging_write_byte_enable[1]}},
            {8{staging_write_byte_enable[0]}}}),
        .a_read_valid(), .a_read_data(),
        .b_req_valid(staging_read_request_valid),
        .b_req_ready(staging_read_request_ready),
        .b_write(1'b0),
        .b_address({2'd0, staging_read_address}),
        .b_write_data('0), .b_write_mask_n('1),
        .b_read_valid(staging_read_response_valid),
        .b_read_data(staging_read_data),
        .same_address_conflict()
    );

    bf16_vector_pipe #(.LANES(16), .TAG_WIDTH(16)) arithmetic (
        .clk, .rst, .abort_request,
        .abort_ack(arithmetic_abort_ack),
        .req_valid(arithmetic_req_valid),
        .req_ready(arithmetic_req_ready),
        .req_operation(arithmetic_req_operation),
        .req_values(arithmetic_req_values),
        .req_paired_values(arithmetic_req_paired_values),
        .req_factor0_values(arithmetic_req_factor0_values),
        .req_factor1_values(arithmetic_req_factor1_values),
        .req_lane_mask(arithmetic_req_lane_mask),
        .req_tag(arithmetic_req_tag),
        .rsp_valid(arithmetic_rsp_valid),
        .rsp_ready(arithmetic_rsp_ready),
        .rsp_values(arithmetic_rsp_values),
        .rsp_lane_mask(arithmetic_rsp_lane_mask),
        .rsp_tag(arithmetic_rsp_tag), .idle(),
        .trace_sample_valid(arithmetic_trace_sample_valid),
        .trace_sample_stage(unused_trace_sample_stage),
        .trace_sample_values(arithmetic_trace_sample_values),
        .trace_sample_lane_mask(arithmetic_trace_sample_lane_mask),
        .trace_sample_tag(arithmetic_trace_sample_tag)
    );

    matmul_output_controller output_controller (
        .cfg_fused_product(cfg_mode == 3'd5),
        .cfg_batch_count,
        .cfg_second_batch_rows,
        .cfg_third_batch_rows, .cfg_fourth_batch_rows,
        .cfg_fifth_batch_rows, .cfg_sixth_batch_rows,
        .in_batch_index,
        .clk, .rst, .abort_request, .abort_ack,
        .cfg_valid, .cfg_ready, .cfg_mode, .cfg_physical_row_base,
        .cfg_row_count, .cfg_output_channels, .cfg_output_base,
        .cfg_output_limit, .cfg_local_row_stride,
        .cfg_residual_from_ddr, .cfg_residual_base, .cfg_residual_limit,
        .cfg_residual_storage_rows,
        .in_valid, .in_ready, .in_physical_row, .in_output_channel,
        .in_row_byte_base, .in_data, .in_byte_enable,
        .residual_read_request_valid, .residual_read_request_ready,
        .residual_ddr_stage(unused_residual_ddr_stage),
        .residual_read_request_address, .residual_read_request_bytes,
        .residual_read_request_tag, .residual_read_request_done,
        .residual_read_local,
        .residual_read_request_error, .residual_read_data_valid,
        .residual_read_data_ready, .residual_read_data,
        .residual_read_byte_enable, .residual_read_data_last,
        .residual_read_data_tag,
        .arithmetic_req_valid, .arithmetic_req_ready,
        .arithmetic_req_operation, .arithmetic_req_values,
        .arithmetic_req_paired_values, .arithmetic_req_factor0_values,
        .arithmetic_req_factor1_values, .arithmetic_req_lane_mask,
        .arithmetic_req_tag, .arithmetic_rsp_valid,
        .arithmetic_rsp_ready, .arithmetic_rsp_values,
        .arithmetic_rsp_lane_mask, .arithmetic_rsp_tag,
        .arithmetic_abort_ack,
        .local_write_valid, .local_write_ready,
        .local_write_byte_address, .local_write_data,
        .local_write_byte_enable,
        .output_write_valid, .output_write_ready,
        .output_write_byte_address, .output_write_transaction_bytes,
        .output_write_data, .output_write_byte_enable,
        .output_write_first, .output_write_last,
        .output_write_done, .output_write_error,
        .staging_write_valid, .staging_write_ready,
        .staging_write_address, .staging_write_data,
        .staging_write_byte_enable,
        .staging_read_request_valid, .staging_read_request_ready,
        .staging_read_address, .staging_read_response_valid,
        .staging_read_data,
        .row_commit, .done_pulse(done), .busy, .error, .error_id
    );
endmodule

`default_nettype wire
