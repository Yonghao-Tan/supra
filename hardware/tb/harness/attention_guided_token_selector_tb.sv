// Connect the selector to BF16 arithmetic, the row encoder, and a stalled
// metadata scratch model.
`default_nettype none

module attention_guided_token_selector_tb (
    input logic loaded_token_valid, loaded_token_kv_write,
    input logic [10:0] loaded_token_position,
    input logic consume_source_a,
    output logic source_a_valid,
    output logic [31:0] source_a_mask,
    output logic [10:0] source_a_block_start,
    output logic [31:0] source_a_capture_index,
    input logic clk,
    input logic rst,
    input logic abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [63:0] start_descriptor_address,
    input logic [63:0] start_joint_descriptor_address,
    input logic start_prepare,
    input logic start_relation_only,
    input logic start_relation_l31,
    input logic start_closeout,
    output logic prepared_for_forward,
    output logic probability_enable,
    output logic [31:0] relation_layer_mask,
    output logic [63:0] probability_output_base,
    output logic [11:0] probability_query_end,
    output logic joint_add_observe_valid,
    output logic scratch_observe_valid,
    output logic [7:0] selector_observe_state,
    output logic [31:0] metadata_bytes,
    output logic [3:0] metadata_rounds,
    output logic dma_request_valid,
    input logic dma_request_ready,
    output logic [63:0] dma_request_address,
    output logic [31:0] dma_request_bytes,
    output logic [7:0] dma_request_tag,
    input logic dma_read_valid,
    output logic dma_read_ready,
    input logic [127:0] dma_read_data,
    input logic [15:0] dma_read_byte_enable,
    input logic dma_read_last,
    input logic [7:0] dma_read_tag,
    input logic dma_error,
    input logic dma_abort_ack,
    output logic dma_write_request_valid,
    input logic dma_write_request_ready,
    output logic [63:0] dma_write_request_address,
    output logic [31:0] dma_write_request_bytes,
    output logic [7:0] dma_write_request_tag,
    output logic dma_write_valid,
    input logic dma_write_ready,
    output logic [127:0] dma_write_data,
    output logic [15:0] dma_write_byte_enable,
    output logic dma_write_last,
    input logic dma_write_done,
    input logic dma_write_error,

    output logic selected_valid,
    input logic selected_ready,
    output logic [10:0] selected_position,
    output logic selected_last,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id
);

    hardware_types_pkg::attention_probability_config_t probability_config;
    assign probability_enable = probability_config.enable;
    assign probability_output_base = probability_config.output_base;
    assign probability_query_end = probability_config.query_end;
    logic packer_abort_request;
    logic bf16_abort_request, bf16_abort_ack, bf16_req_valid, bf16_req_ready, bf16_rsp_valid, bf16_rsp_ready;
    hardware_types_pkg::bf16_request_t bf16_req;
    hardware_types_pkg::bf16_response_t bf16_rsp;
    assign joint_add_observe_valid = bf16_req_valid && bf16_req_ready;
    assign bf16_rsp.values[1023:128] = '0;
    assign bf16_rsp.lane_mask[63:8] = '0;
    bf16_vector_pipe #(.LANES(8), .TAG_WIDTH(16)) joint_arithmetic (
        .clk, .rst, .abort_request(bf16_abort_request), .abort_ack(bf16_abort_ack),
        .req_valid(bf16_req_valid), .req_ready(bf16_req_ready),
        .req_operation(bf16_req.operation), .req_values(bf16_req.values[127:0]),
        .req_paired_values(bf16_req.paired_values[127:0]), .req_factor0_values(bf16_req.factor0_values[127:0]), .req_factor1_values(128'd0),
        .req_lane_mask(bf16_req.lane_mask[7:0]), .req_tag(bf16_req.tag),
        .rsp_valid(bf16_rsp_valid), .rsp_ready(bf16_rsp_ready), .rsp_values(bf16_rsp.values[127:0]),
        .rsp_lane_mask(bf16_rsp.lane_mask[7:0]), .rsp_tag(bf16_rsp.tag), .idle(),
        .trace_sample_valid(), .trace_sample_stage(), .trace_sample_values(), .trace_sample_lane_mask(), .trace_sample_tag()
    );
    logic packer_abort_ack;
    logic packer_start_valid;
    logic packer_start_ready;
    hardware_types_pkg::token_metadata_config_t packer_config;
    logic packer_row_valid;
    logic packer_row_ready;
    hardware_types_pkg::token_metadata_row_t packer_row;
    logic packer_beat_valid;
    logic packer_beat_ready;
    logic [127:0] packer_beat_data;
    logic packer_beat_last;
    logic packer_done_valid;
    logic packer_done_ready;
    logic packer_error;
    logic [7:0] packer_error_id;
    logic [5:0] packer_compute_groups;
    logic [5:0] packer_semantic_groups;
    logic metadata_scratch_req_valid, metadata_scratch_req_ready;
    logic metadata_scratch_write;
    logic [8:0] metadata_scratch_address;
    logic [58:0] metadata_scratch_write_data;
    logic [58:0] metadata_scratch_write_enable;
    logic metadata_scratch_rsp_valid;
    logic [58:0] metadata_scratch_rsp_data;
    logic metadata_scratch_aux_write_valid, metadata_scratch_aux_write_ready;
    logic [8:0] metadata_scratch_aux_write_address;
    logic [19:0] metadata_scratch_aux_write_data;
    logic [19:0] metadata_scratch_aux_write_enable;
    logic [58:0] metadata_scratch_memory [0:431];
    logic [1:0] metadata_scratch_read_pending;
    logic [58:0] metadata_scratch_read_data [0:1];

    logic [2:0] scratch_delay;
    always_ff @(posedge clk) begin
        if (rst) scratch_delay <= '0;
        else scratch_delay <= scratch_delay + 3'd1;
    end
    assign metadata_scratch_req_ready = !rst && scratch_delay != 3'd0;
    assign scratch_observe_valid = (metadata_scratch_req_valid && metadata_scratch_req_ready) || metadata_scratch_rsp_valid;
    assign selector_observe_state = 8'(dut.state);
    assign metadata_scratch_aux_write_ready = !rst && scratch_delay != 3'd2 && scratch_delay != 3'd3;
    assign metadata_scratch_rsp_valid = metadata_scratch_read_pending[1];
    assign metadata_scratch_rsp_data = metadata_scratch_read_data[1];
    always_ff @(posedge clk) begin
        if (rst) begin
            metadata_scratch_read_pending <= '0;
            metadata_scratch_read_data[0] <= '0;
            metadata_scratch_read_data[1] <= '0;
        end else begin
            metadata_scratch_read_pending <= {
                metadata_scratch_read_pending[0],
                metadata_scratch_req_valid && metadata_scratch_req_ready &&
                !metadata_scratch_write};
            metadata_scratch_read_data[1] <= metadata_scratch_read_data[0];
            if (metadata_scratch_req_valid && metadata_scratch_req_ready) begin
                if (metadata_scratch_write)
                    metadata_scratch_memory[metadata_scratch_address] <=
                        (metadata_scratch_memory[metadata_scratch_address] &
                         ~metadata_scratch_write_enable) |
                        (metadata_scratch_write_data &
                         metadata_scratch_write_enable);
                else
                    metadata_scratch_read_data[0] <=
                        metadata_scratch_memory[metadata_scratch_address];
            end
            if (metadata_scratch_aux_write_valid && metadata_scratch_aux_write_ready)
                metadata_scratch_memory[metadata_scratch_aux_write_address] <=
                    (metadata_scratch_memory[metadata_scratch_aux_write_address] &
                     ~{metadata_scratch_aux_write_enable, 39'd0}) |
                    ({metadata_scratch_aux_write_data, 39'd0} &
                     {metadata_scratch_aux_write_enable, 39'd0});
        end
    end

    attention_guided_token_selector dut (.*);
    token_issue_packer shared_token_issue_packer (
        .clk, .rst, .abort_request(packer_abort_request), .abort_ack(packer_abort_ack),
        .start_valid(packer_start_valid), .start_ready(packer_start_ready),
        .start_row_count(packer_config.row_count), .start_sequence_length(packer_config.sequence_length),
        .start_token_batch_index(packer_config.token_batch_index), .start_first_token_ordinal(packer_config.first_token_ordinal),
        .start_metadata_version(packer_config.metadata_version), .start_capture_index(packer_config.capture_index),
        .row_valid(packer_row_valid), .row_ready(packer_row_ready), .row_index(packer_row.index),
        .row_source_index(packer_row.source_index), .row_token_position(packer_row.token_position),
        .row_kv_write_disable(packer_row.kv_write_disable), .row_kv_index(packer_row.kv_index), .row_embedding_source(packer_row.embedding_source),
        .activation_bits(packer_row.bits), .row_query_group(packer_row.query_group), .row_cache_group(packer_row.cache_group),
        .beat_valid(packer_beat_valid), .beat_ready(packer_beat_ready), .beat_data(packer_beat_data),
        .beat_last(packer_beat_last), .done_valid(packer_done_valid), .done_ready(packer_done_ready),
        .error(packer_error), .error_id(packer_error_id), .compute_group_count(packer_compute_groups),
        .semantic_group_count(packer_semantic_groups)
    );
endmodule
`default_nettype wire
