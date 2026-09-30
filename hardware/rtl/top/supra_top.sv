`timescale 1ns/1ps
`default_nettype none

// SUPRA accelerator with launch/completion and AXI4 memory interfaces.
module supra_top #(
    parameter integer ATTENTION_HEAD_COUNT = 32,
    parameter integer HIDDEN_FEATURES = 4096,
    parameter integer FFN_FEATURES = 12288
) (
    input  logic         core_clk,
    input  logic         core_rst,
    input  logic         launch_valid,
    output logic         launch_ready,
    input  logic [63:0]  launch_config_address,
    input  logic [31:0]  launch_id,
    output logic         completion_valid,
    input  logic         completion_ready,
    output logic [31:0]  completion_id,
    output logic         completion_error,
    output logic [15:0]  completion_error_id,
    output logic [3:0]   m_axi_awid,
    output logic [63:0]  m_axi_awaddr,
    output logic [7:0]   m_axi_awlen,
    output logic [2:0]   m_axi_awsize,
    output logic [1:0]   m_axi_awburst,
    output logic         m_axi_awvalid,
    input  logic         m_axi_awready,
    output logic [255:0] m_axi_wdata,
    output logic [31:0]  m_axi_wstrb,
    output logic         m_axi_wlast,
    output logic         m_axi_wvalid,
    input  logic         m_axi_wready,
    input  logic [3:0]   m_axi_bid,
    input  logic [1:0]   m_axi_bresp,
    input  logic         m_axi_bvalid,
    output logic         m_axi_bready,
    output logic [3:0]   m_axi_arid,
    output logic [63:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,
    input  logic [3:0]   m_axi_rid,
    input  logic [255:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready
);
    wire clk = core_clk;
    wire rst = core_rst;

    logic [2:0] execution_memory_stage;
    logic execution_block_schedule_active;
    logic execution_forward_postprocess_start_valid;
    logic execution_forward_postprocess_start_ready;
    logic [2559:0] execution_forward_postprocess_configuration_bits;
    logic [63:0] execution_refresh_configuration_address;
    logic execution_refresh_prepare;
    logic execution_refresh_loaded_token_valid, execution_refresh_loaded_token_kv_write;
    logic [10:0] execution_refresh_loaded_token_position;
    logic execution_refresh_select;
    logic execution_refresh_relation_only;
    logic execution_refresh_relation_l31;
    logic execution_refresh_closeout, execution_forward_postprocess_block_complete;
    logic [31:0] execution_refresh_relation_layer_mask;
    logic [63:0] execution_joint_configuration_address;
    logic execution_forward_postprocess_done_valid;
    logic execution_forward_postprocess_done_ready;
    logic execution_forward_postprocess_done_error;
    logic [7:0] execution_forward_postprocess_done_error_id;
    logic [2:0] execution_dma_read_request_valid;
    logic [2:0] execution_dma_read_request_ready;
    hardware_types_pkg::dma_read_request_t execution_dma_read_request [0:2];
    logic [2:0] execution_dma_read_data_valid;
    logic [2:0] execution_dma_read_data_ready;
    hardware_types_pkg::dma_read_beat_t execution_dma_read_data [0:2];
    hardware_types_pkg::dma_completion_t execution_dma_read_completion [0:2];
    logic execution_hidden_start_valid;
    logic execution_hidden_start_ready;
    logic execution_hidden_start_write;
    logic execution_hidden_start_residual;
    logic [63:0] execution_hidden_start_address;
    logic [31:0] execution_hidden_start_bytes;
    logic [5:0] execution_hidden_start_active_rows;
    logic [287:0] execution_hidden_start_row_permutation;
    logic execution_hidden_start_direct_row_index_enable;
    logic [527:0] execution_hidden_start_direct_row_index;
    logic execution_hidden_start_source_descriptor_enable;
    logic [815:0] execution_hidden_start_source_index;
    logic [47:0] execution_hidden_start_source_embedding;
    logic [63:0] execution_hidden_start_embedding_base;
    logic [63:0] execution_hidden_start_embedding_limit;
    logic [63:0] execution_hidden_start_limit;
    logic execution_hidden_done_valid;
    logic execution_hidden_done_ready;
    logic execution_hidden_error;
    logic operator_command_valid;
    logic operator_command_ready;
    logic [5:0] operator_command_layer;
    logic [3:0] execution_layer_operator_index;
    logic operator_command_cache_release;
    logic operator_command_qkv_residual_spill;
    logic operator_command_rms_qkv_prefetch_q;
    logic operator_command_q_head_preprocess;
    hardware_types_pkg::operator_command_t operator_command;
    logic operator_command_done_valid;
    logic operator_command_done_ready;
    logic operator_command_done_error;
    logic [15:0] operator_command_done_error_id;
    hardware_types_pkg::operator_completion_t operator_command_completion;
    logic [5:0] row_active_rows;
    logic [5:0] row_residual_storage_rows;
    logic [11:0] row_sequence_length;
    hardware_types_pkg::matmul_row_config_t matmul_row_config;
    hardware_types_pkg::hidden_row_config_t hidden_row_config;
    hardware_types_pkg::attention_row_config_t attention_row_config;
    logic layer_entry_valid;
    logic [4:0] layer_entry_index;
    hardware_types_pkg::matmul_weight_config_t matmul_weight_config;
    hardware_types_pkg::rmsnorm_layer_config_t rmsnorm_layer_config;
    hardware_types_pkg::qkv_layer_config_t qkv_layer_config;
    hardware_types_pkg::attention_cache_config_t attention_cache_config;
    hardware_types_pkg::attention_context_config_t attention_context_config;
    hardware_types_pkg::ffn_workspace_config_t ffn_workspace_config;
    hardware_types_pkg::ffn_batch_pair_config_t ffn_batch_pair_config;
    logic attention_pair_enable, attention_second_batch;
    logic attention_third_batch_enable;
    logic [1:0] attention_batch_index;
    logic kv_pair_enable;
    hardware_types_pkg::attention_row_config_t kv_pair_second_config;
    hardware_types_pkg::attention_batch_group_config_t kv_batch_group_config;
    logic attention_normalized_capture;
    logic attention_output_subset_enable;
    logic [5:0] attention_output_token_count;
    logic [5:0] attention_pair_second_rows;
    logic [5:0] attention_third_rows;

    assign operator_command.layer = operator_command_layer;
    assign operator_command.layer_operator_index = execution_layer_operator_index;
    assign operator_command.cache_release = operator_command_cache_release;
    assign operator_command.qkv_residual_spill =
        operator_command_qkv_residual_spill;
    assign operator_command.rms_qkv_prefetch_q =
        operator_command_rms_qkv_prefetch_q;
    assign operator_command.q_head_preprocess =
        operator_command_q_head_preprocess;
    assign operator_command_done_error =
        operator_command_completion.error;
    assign operator_command_done_error_id =
        operator_command_completion.error_id;
    logic [1:0] qkv_memory_stage;
    logic [1:0] matmul_memory_stage;
    logic rms_gamma_loaded;
    hardware_types_pkg::attention_read_source_t attention_read_source;
    logic dma_read_stream_idle;
    logic qkv_residual_spill_active;
    logic [6:0] operator_dma_read_request_valid;
    logic [6:0] operator_dma_read_request_ready;
    hardware_types_pkg::dma_read_request_t operator_dma_read_request [0:6];
    logic elementwise_read_second_span_valid;
    logic [63:0] elementwise_read_second_span_address;
    logic [31:0] elementwise_read_pair_stride;
    logic [10:0] elementwise_read_pair_count;
    logic matmul_aux_read_second_span_valid;
    logic [63:0] matmul_aux_read_second_span_address;
    logic [31:0] matmul_aux_read_pair_stride;
    logic [10:0] matmul_aux_read_pair_count;
    logic [6:0] operator_dma_read_data_valid;
    logic [6:0] operator_dma_read_data_ready;
    hardware_types_pkg::dma_read_beat_t operator_dma_read_data [0:6];
    hardware_types_pkg::dma_wide_read_beat_t operator_dma_wide_read_data [0:6];
    hardware_types_pkg::dma_completion_t operator_dma_read_completion [0:6];
    logic elementwise_read_data_span;
    logic [3:0] operator_dma_write_request_valid;
    logic [3:0] operator_dma_write_request_ready;
    hardware_types_pkg::dma_write_request_t operator_dma_write_request [0:3];
    logic [3:0] operator_dma_write_data_valid;
    logic [3:0] operator_dma_write_data_ready;
    hardware_types_pkg::dma_write_beat_t operator_dma_write_data [0:3];
    hardware_types_pkg::dma_completion_t operator_dma_write_completion [0:3];
    logic memory_read_request_valid;
    logic memory_read_request_ready;
    hardware_types_pkg::dma_read_request_t memory_read_request;
    logic memory_read_second_span_valid;
    logic [63:0] memory_read_second_span_address;
    logic [31:0] memory_read_pair_stride;
    logic [10:0] memory_read_pair_count;
    hardware_types_pkg::dma_completion_t memory_read_completion;
    logic memory_read_data_valid;
    logic memory_read_data_ready;
    hardware_types_pkg::dma_read_beat_t memory_read_data;
    logic memory_read_data_span;
    logic memory_wide_read_data_valid;
    logic memory_wide_read_data_ready;
    hardware_types_pkg::dma_wide_read_beat_t memory_wide_read_data;
    logic memory_write_request_valid;
    logic memory_write_request_ready;
    hardware_types_pkg::dma_write_request_t memory_write_request;
    hardware_types_pkg::dma_completion_t memory_write_completion;
    logic memory_write_data_valid;
    logic memory_write_data_ready;
    hardware_types_pkg::dma_write_beat_t memory_write_data;
    logic memory_abort_request;
    logic memory_read_busy;
    logic memory_write_busy;
    logic memory_drained;
    logic [4:0] memory_read_outstanding;
    logic [4:0] memory_write_outstanding;
    logic [63:0] memory_read_credit_stall_cycles;
    logic [63:0] memory_write_credit_stall_cycles;
    logic [63:0] memory_read_burst_count;
    logic [63:0] memory_write_burst_count;
    logic [63:0] memory_actual_read_bytes;
    logic [63:0] memory_actual_write_bytes;
    logic [63:0] memory_useful_read_bytes;
    logic [63:0] memory_useful_write_bytes;
    logic local_layout_update_valid;
    logic local_layout_update_ready;
    local_memory_layout_pkg::layout_requirement_t requested_local_layout;
    local_memory_layout_pkg::layout_requirement_t current_local_layout;
    logic qkv_matmul_overlap_active;
    wire [3:0] requested_local_phase = requested_local_layout.phase;
    wire requested_active_panel = requested_local_layout.active_panel;
    wire requested_qkv_head_staging =
        requested_local_layout.qkv_head_staging;
    wire [1:0] requested_panel_layout =
        requested_local_layout.panel_layout;
    wire [3:0] current_local_phase = current_local_layout.phase;
    wire current_active_panel = current_local_layout.active_panel;
    wire current_qkv_head_staging = current_local_layout.qkv_head_staging;
    wire [1:0] current_panel_layout = current_local_layout.panel_layout;
    logic rms_memory_phase_active;
    logic rms_ffn_command;
    logic rms_gamma_bypass;
    logic rms_gamma_stage_valid;
    logic rms_gamma_stage_ready;
    hardware_types_pkg::rms_gamma_write_t rms_gamma_write;
    logic rms_tile_read_req_valid;
    logic rms_tile_read_req_ready;
    hardware_types_pkg::rms_tile_read_request_t rms_tile_read_req;
    logic rms_tile_read_rsp_valid;
    logic rms_tile_read_rsp_ready;
    hardware_types_pkg::rms_tile_read_response_t rms_tile_read_rsp;
    logic rms_scratch_read_valid;
    logic rms_scratch_read_ready;
    hardware_types_pkg::local_pair_read_request_t rms_scratch_read_req;
    logic rms_scratch_read_rsp_valid;
    logic rms_scratch_read_rsp_ready;
    hardware_types_pkg::local_pair_read_response_t rms_scratch_read_rsp;
    logic rms_scratch_write_valid;
    logic rms_scratch_write_ready;
    hardware_types_pkg::local_banked_write_t rms_scratch_write;
    logic refresh_metadata_scratch_req_valid;
    logic refresh_metadata_scratch_req_ready;
    logic refresh_metadata_scratch_write;
    logic [8:0] refresh_metadata_scratch_address;
    logic [58:0] refresh_metadata_scratch_write_data;
    logic [58:0] refresh_metadata_scratch_write_enable;
    logic refresh_metadata_scratch_rsp_valid;
    logic [58:0] refresh_metadata_scratch_rsp_data;
    logic refresh_metadata_scratch_aux_write_valid;
    logic refresh_metadata_scratch_aux_write_ready;
    logic [8:0] refresh_metadata_scratch_aux_write_address;
    logic [19:0] refresh_metadata_scratch_aux_write_data;
    logic [19:0] refresh_metadata_scratch_aux_write_enable;
    logic rms_norm_write_valid;
    logic rms_norm_write_ready;
    hardware_types_pkg::rms_norm_write_t rms_norm_write;
    logic hidden_local_write_valid;
    logic hidden_local_write_ready;
    hardware_types_pkg::hidden_local_write_t hidden_local_write;
    logic hidden_local_read_valid;
    logic hidden_local_read_ready;
    hardware_types_pkg::hidden_local_read_request_t hidden_local_read_req;
    logic hidden_local_read_response_valid;
    hardware_types_pkg::local_128b_read_response_t hidden_local_read_rsp;
    logic attention_context_buffer_write_valid;
    logic attention_context_buffer_write_ready;
    hardware_types_pkg::local_128b_write_t attention_context_write;
    logic attention_context_read_req_valid;
    logic attention_context_read_req_ready;
    hardware_types_pkg::tagged_local_read_request_t attention_context_read_req;
    logic attention_context_read_rsp_valid;
    logic attention_context_read_rsp_ready;
    hardware_types_pkg::tagged_local_read_response_t attention_context_read_rsp;
    logic attention_softmax_scratch_read_valid;
    logic attention_softmax_scratch_read_ready;
    hardware_types_pkg::local_pair_read_request_t
        attention_softmax_scratch_read_req;
    logic attention_softmax_scratch_rsp_valid;
    logic attention_softmax_scratch_rsp_ready;
    hardware_types_pkg::local_pair_read_response_t attention_softmax_scratch_rsp;
    logic attention_softmax_scratch_write_valid;
    logic attention_softmax_scratch_write_ready;
    hardware_types_pkg::local_banked_write_t attention_softmax_scratch_write;
    logic qkv_head_read_req_valid;
    logic qkv_head_read_req_ready;
    hardware_types_pkg::qkv_head_read_request_t qkv_head_read_req;
    logic qkv_head_read_rsp_valid;
    logic qkv_head_read_rsp_ready;
    hardware_types_pkg::qkv_head_read_response_t qkv_head_read_rsp;
    logic rope_source_read_req_valid;
    logic rope_source_read_req_ready;
    hardware_types_pkg::rope_source_read_request_t rope_source_read_req;
    logic rope_source_read_rsp_valid;
    logic rope_source_read_rsp_ready;
    hardware_types_pkg::rope_source_read_response_t rope_source_read_rsp;
    logic rope_cos_read_req_valid;
    logic rope_cos_read_req_ready;
    hardware_types_pkg::rope_constant_read_request_t rope_constant_read_req;
    logic rope_cos_read_rsp_valid;
    logic rope_cos_read_rsp_ready;
    hardware_types_pkg::rope_constant_read_response_t rope_cos_read_rsp;
    logic rope_sin_read_req_valid;
    logic rope_sin_read_req_ready;
    logic rope_sin_read_rsp_valid;
    logic rope_sin_read_rsp_ready;
    hardware_types_pkg::rope_constant_read_response_t rope_sin_read_rsp;
    logic rope_destination_write_valid;
    logic rope_destination_write_ready;
    hardware_types_pkg::rope_destination_write_t rope_destination_write;
    logic qkv_head_stage_write_valid;
    logic qkv_head_stage_write_ready;
    hardware_types_pkg::qkv_head_stage_write_t qkv_head_stage_write;
    logic qkv_head_tile_memory_req_valid;
    logic qkv_head_tile_memory_req_ready;
    hardware_types_pkg::qkv_head_tile_read_request_t qkv_head_tile_read_req;
    logic qkv_head_tile_memory_rsp_valid;
    hardware_types_pkg::qkv_head_tile_read_response_t qkv_head_tile_read_rsp;
    logic qkv_constant_stage_valid;
    logic qkv_constant_stage_ready;
    hardware_types_pkg::qkv_constant_write_t qkv_constant_write;
    logic qkv_q_local_write_valid;
    logic qkv_q_local_write_ready;
    hardware_types_pkg::qkv_q_write_t qkv_q_write;
    logic attention_panel_write_valid;
    logic attention_panel_write_ready;
    hardware_types_pkg::attention_panel_write_t attention_panel_write;
    logic attention_scale_write_valid;
    logic attention_scale_write_ready;
    hardware_types_pkg::attention_panel_scale_write_t
        attention_panel_scale_write;
    logic attention_operand_req_valid;
    logic attention_operand_req_ready;
    hardware_types_pkg::attention_operand_request_t attention_operand_req;
    logic attention_operand_rsp_valid;
    hardware_types_pkg::attention_operand_response_t attention_operand_rsp;
    logic attention_operand_abort_request;
    logic attention_operand_abort_ack;
    logic attention_score_write_valid;
    logic attention_score_write_ready;
    hardware_types_pkg::attention_score_write_t attention_score_write;
    logic attention_softmax_score_req_valid;
    logic attention_softmax_score_req_ready;
    logic attention_softmax_score_slot_ready;
    hardware_types_pkg::attention_score_read_request_t attention_score_read_req;
    logic attention_softmax_score_memory_rsp_valid;
    hardware_types_pkg::attention_score_read_response_t attention_score_read_rsp;
    logic attention_probability_write_valid;
    logic attention_probability_write_ready;
    hardware_types_pkg::attention_probability_write_t
        attention_probability_write;
    logic attention_probability_quantized_write_valid;
    logic attention_probability_quantized_write_ready;
    hardware_types_pkg::attention_probability_quantized_write_t
        attention_probability_quantized_write;
    logic attention_probability_scale_write_valid;
    logic attention_probability_scale_write_ready;
    hardware_types_pkg::attention_probability_scale_write_t
        attention_probability_scale_write;
    logic matmul_source_req_valid;
    logic matmul_local_source_req_ready;
    hardware_types_pkg::matmul_source_request_t matmul_source_req;
    logic matmul_local_source_rsp_valid;
    logic matmul_source_rsp_ready;
    hardware_types_pkg::matmul_source_response_t matmul_source_rsp;
    logic matmul_activation_write_valid;
    logic matmul_activation_write_ready;
    hardware_types_pkg::matmul_activation_write_t matmul_activation_write;
    logic matmul_read_bundle_valid;
    logic matmul_read_bundle_ready;
    hardware_types_pkg::matmul_read_request_t matmul_read_req;
    logic matmul_read_response_valid;
    hardware_types_pkg::matmul_read_response_t matmul_read_rsp;
    logic matmul_panel_write_valid;
    logic matmul_panel_write_ready;
    hardware_types_pkg::matmul_panel_write_t matmul_panel_write;
    logic matmul_combine_write_valid;
    logic matmul_combine_write_ready;
    hardware_types_pkg::local_128b_write_t matmul_combine_write;
    logic matmul_combine_read_valid;
    logic matmul_combine_read_ready;
    hardware_types_pkg::local_128b_read_request_t matmul_combine_read_req;
    logic matmul_combine_read_response_valid;
    hardware_types_pkg::local_128b_read_response_t matmul_combine_read_rsp;
    logic matmul_local_output_write_valid;
    logic matmul_local_output_write_ready;
    hardware_types_pkg::matmul_output_write_t matmul_local_output_write;
    logic elementwise_product_req_valid;
    logic elementwise_product_req_ready;
    hardware_types_pkg::elementwise_product_request_t elementwise_product_req;
    logic elementwise_product_rsp_valid;
    logic elementwise_product_rsp_ready;
    hardware_types_pkg::elementwise_product_response_t elementwise_product_rsp;
    logic matmul_residual_local_read_issue;
    logic matmul_residual_memory_read_ready;
    hardware_types_pkg::matmul_residual_read_request_t matmul_residual_read_req;
    logic matmul_residual_memory_read_rsp_valid;
    hardware_types_pkg::local_128b_read_response_t matmul_residual_read_rsp;
    logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        local_port_req_ready;
    logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        local_port_read_valid;
    logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        local_port_read_data;
    logic post_table_access_valid, post_table_access_ready;
    logic post_table_access_enable;
    logic post_state_write_valid, post_state_write_ready;
    logic [9:0] post_state_write_word_address;
    logic [127:0] post_state_write_data;
    logic post_state_read_valid, post_state_read_ready;
    logic [9:0] post_state_read_word_address;
    logic [15:0] post_state_read_tag;
    logic post_state_read_rsp_valid;
    logic [127:0] post_state_read_rsp_data;
    logic [15:0] post_state_read_rsp_tag;
    logic post_suppressed_write_valid, post_suppressed_write_ready;
    logic [5:0] post_suppressed_write_address;
    logic [127:0] post_suppressed_write_data;
    logic post_suppressed_read_valid, post_suppressed_read_ready;
    logic [7:0] post_suppressed_read_index;
    logic post_suppressed_read_rsp_valid;
    logic [16:0] post_suppressed_read_rsp_token_id;
    logic lm_head_scale_write_valid, lm_head_scale_write_ready;
    logic [6:0] lm_head_scale_write_address;
    logic [127:0] lm_head_scale_write_data;
    logic lm_head_scale_read_valid, lm_head_scale_read_ready;
    logic [6:0] lm_head_scale_read_address;
    logic lm_head_scale_read_rsp_valid;
    logic [127:0] lm_head_scale_read_rsp_data;
    logic candidate_state_read_valid, candidate_state_read_ready;
    logic [3:0] candidate_state_read_row_group;
    logic [2:0] candidate_state_read_lane_block;
    logic candidate_state_read_row_half;
    logic [1:0] candidate_state_read_word;
    logic [3:0] candidate_state_read_row_mask;
    logic [15:0] candidate_state_read_tag;
    logic candidate_state_read_rsp_valid, candidate_state_read_rsp_ready;
    logic [511:0] candidate_state_read_rsp_data;
    logic [3:0] candidate_state_read_rsp_row_mask;
    logic [15:0] candidate_state_read_rsp_tag;
    logic candidate_state_write_valid, candidate_state_write_ready;
    logic [3:0] candidate_state_write_row_group;
    logic [2:0] candidate_state_write_lane_block;
    logic candidate_state_write_row_half;
    logic [1:0] candidate_state_write_word;
    logic [3:0] candidate_state_write_row_mask;
    logic [511:0] candidate_state_write_data;

    execution_controller #(.FFN_FEATURES(FFN_FEATURES)) execution (
        .clk(clk), .rst(rst),
        .launch_valid(launch_valid), .launch_ready(launch_ready),
        .launch_config_addr(launch_config_address), .launch_id(launch_id),
        .completion_valid(completion_valid),
        .completion_ready(completion_ready),
        .completion_id(completion_id),
        .completion_error(completion_error),
        .completion_error_id(completion_error_id),
        .memory_stage(execution_memory_stage),
        .block_schedule_active(execution_block_schedule_active),
        .forward_postprocess_start_valid(execution_forward_postprocess_start_valid),
        .forward_postprocess_start_ready(execution_forward_postprocess_start_ready),
        .forward_postprocess_configuration_bits(
            execution_forward_postprocess_configuration_bits),
        .refresh_configuration_address(execution_refresh_configuration_address),
        .refresh_prepare(execution_refresh_prepare),
        .refresh_loaded_token_valid(execution_refresh_loaded_token_valid),
        .refresh_loaded_token_position(execution_refresh_loaded_token_position),
        .refresh_loaded_token_kv_write(execution_refresh_loaded_token_kv_write),
        .refresh_select(execution_refresh_select),
        .refresh_relation_only(execution_refresh_relation_only),
        .refresh_relation_l31(execution_refresh_relation_l31),
        .refresh_closeout(execution_refresh_closeout),
        .forward_postprocess_block_complete(execution_forward_postprocess_block_complete),
        .refresh_relation_layer_mask(execution_refresh_relation_layer_mask),
        .joint_configuration_address(execution_joint_configuration_address),
        .forward_postprocess_done_valid(execution_forward_postprocess_done_valid),
        .forward_postprocess_done_ready(execution_forward_postprocess_done_ready),
        .forward_postprocess_done_error(execution_forward_postprocess_done_error),
        .forward_postprocess_done_error_id(execution_forward_postprocess_done_error_id),
        .dma_read_request_valid(execution_dma_read_request_valid),
        .dma_read_request_ready(execution_dma_read_request_ready),
        .dma_read_request(execution_dma_read_request),
        .dma_read_data_valid(execution_dma_read_data_valid),
        .dma_read_data_ready(execution_dma_read_data_ready),
        .dma_read_data(execution_dma_read_data),
        .dma_read_completion(execution_dma_read_completion),
        .hidden_start_valid(execution_hidden_start_valid),
        .hidden_start_ready(execution_hidden_start_ready),
        .hidden_start_write(execution_hidden_start_write),
        .hidden_start_residual(execution_hidden_start_residual),
        .hidden_start_address(execution_hidden_start_address),
        .hidden_start_bytes(execution_hidden_start_bytes),
        .hidden_start_active_rows(execution_hidden_start_active_rows),
        .hidden_start_row_permutation(execution_hidden_start_row_permutation),
        .hidden_start_direct_row_index_enable(
            execution_hidden_start_direct_row_index_enable),
        .hidden_start_direct_row_index(execution_hidden_start_direct_row_index),
        .hidden_start_source_descriptor_enable(
            execution_hidden_start_source_descriptor_enable),
        .hidden_start_source_index(execution_hidden_start_source_index),
        .hidden_start_source_embedding(
            execution_hidden_start_source_embedding),
        .hidden_start_embedding_base(execution_hidden_start_embedding_base),
        .hidden_start_embedding_limit(execution_hidden_start_embedding_limit),
        .hidden_start_limit(execution_hidden_start_limit),
        .hidden_done_valid(execution_hidden_done_valid),
        .hidden_done_ready(execution_hidden_done_ready),
        .hidden_error(execution_hidden_error),
        .command_valid(operator_command_valid),
        .command_ready(operator_command_ready),
        .command_layer(operator_command_layer),
        .layer_operator_index(execution_layer_operator_index),
        .command_cache_release(operator_command_cache_release),
        .command_qkv_residual_spill(operator_command_qkv_residual_spill),
        .command_rms_qkv_prefetch_q(operator_command_rms_qkv_prefetch_q),
        .command_q_head_preprocess(operator_command_q_head_preprocess),
        .command_done_valid(operator_command_done_valid),
        .command_done_ready(operator_command_done_ready),
        .command_done_error(operator_command_done_error),
        .command_done_error_id(operator_command_done_error_id),
        .row_active_rows(row_active_rows),
        .row_residual_storage_rows(row_residual_storage_rows),
        .row_sequence_length(row_sequence_length),
        .matmul_row_config(matmul_row_config),
        .hidden_row_config(hidden_row_config),
        .attention_row_config(attention_row_config),
        .layer_entry_valid(layer_entry_valid),
        .layer_entry_index(layer_entry_index),
        .matmul_weight_config(matmul_weight_config),
        .rmsnorm_layer_config(rmsnorm_layer_config),
        .qkv_layer_config(qkv_layer_config),
        .attention_cache_config(attention_cache_config),
        .attention_context_config(attention_context_config),
        .ffn_workspace_config(ffn_workspace_config),
        .ffn_batch_pair_config(ffn_batch_pair_config),
        .attention_pair_enable(attention_pair_enable),
        .attention_normalized_capture(attention_normalized_capture),
        .kv_pair_enable(kv_pair_enable), .kv_pair_second_config(kv_pair_second_config),
        .kv_batch_group_config(kv_batch_group_config),
        .attention_output_subset_enable(attention_output_subset_enable),
        .attention_output_token_count(attention_output_token_count),
        .attention_pair_second_rows(attention_pair_second_rows),
        .attention_third_batch_enable(attention_third_batch_enable),
        .attention_third_rows(attention_third_rows),
        .attention_second_batch(attention_second_batch),
        .attention_batch_index(attention_batch_index)
    );

    dma_stream_router dma_router (
        .clk(clk), .rst(rst),
        .execution_memory_stage(execution_memory_stage),
        .execution_layer_operator_index(execution_layer_operator_index),
        .qkv_memory_stage(qkv_memory_stage),
        .matmul_memory_stage(matmul_memory_stage),
        .rms_gamma_loaded(rms_gamma_loaded),
        .attention_read_source(attention_read_source),
        .read_stream_idle(dma_read_stream_idle),
        .qkv_residual_spill_active(qkv_residual_spill_active),
        .execution_read_request_valid(execution_dma_read_request_valid),
        .execution_read_request_ready(execution_dma_read_request_ready),
        .execution_read_request(execution_dma_read_request),
        .execution_read_data_valid(execution_dma_read_data_valid),
        .execution_read_data_ready(execution_dma_read_data_ready),
        .execution_read_data(execution_dma_read_data),
        .execution_read_completion(execution_dma_read_completion),
        .operator_read_request_valid(operator_dma_read_request_valid),
        .operator_read_request_ready(operator_dma_read_request_ready),
        .operator_read_request(operator_dma_read_request),
        .elementwise_read_second_span_valid,
        .elementwise_read_second_span_address,
        .elementwise_read_pair_stride,
        .elementwise_read_pair_count,
        .matmul_aux_read_second_span_valid,
        .matmul_aux_read_second_span_address,
        .matmul_aux_read_pair_stride,
        .matmul_aux_read_pair_count,
        .operator_read_data_valid(operator_dma_read_data_valid),
        .operator_read_data_ready(operator_dma_read_data_ready),
        .operator_read_data(operator_dma_read_data),
        .operator_wide_read_data(operator_dma_wide_read_data),
        .operator_read_completion(operator_dma_read_completion),
        .elementwise_read_data_span,
        .operator_write_request_valid(operator_dma_write_request_valid),
        .operator_write_request_ready(operator_dma_write_request_ready),
        .operator_write_request(operator_dma_write_request),
        .operator_write_data_valid(operator_dma_write_data_valid),
        .operator_write_data_ready(operator_dma_write_data_ready),
        .operator_write_data(operator_dma_write_data),
        .operator_write_completion(operator_dma_write_completion),
        .memory_read_request_valid(memory_read_request_valid),
        .memory_read_request_ready(memory_read_request_ready),
        .memory_read_request(memory_read_request),
        .memory_read_second_span_valid,
        .memory_read_second_span_address,
        .memory_read_pair_stride,
        .memory_read_pair_count,
        .memory_read_completion(memory_read_completion),
        .memory_read_data_valid(memory_read_data_valid),
        .memory_read_data_ready(memory_read_data_ready),
        .memory_read_data(memory_read_data),
        .memory_read_data_span,
        .memory_wide_read_data_valid(memory_wide_read_data_valid),
        .memory_wide_read_data_ready(memory_wide_read_data_ready),
        .memory_wide_read_data(memory_wide_read_data),
        .memory_write_request_valid(memory_write_request_valid),
        .memory_write_request_ready(memory_write_request_ready),
        .memory_write_request(memory_write_request),
        .memory_write_completion(memory_write_completion),
        .memory_write_data_valid(memory_write_data_valid),
        .memory_write_data_ready(memory_write_data_ready),
        .memory_write_data(memory_write_data)
    );

    memory_controller memory (
        .clk(clk), .rst(rst), .abort_request(memory_abort_request),
        .read_request_valid(memory_read_request_valid),
        .read_request_ready(memory_read_request_ready),
        .read_request(memory_read_request),
        .read_second_span_valid(memory_read_second_span_valid),
        .read_second_span_address(memory_read_second_span_address),
        .read_pair_stride(memory_read_pair_stride),
        .read_pair_count(memory_read_pair_count),
        .read_completion(memory_read_completion),
        .read_data_valid(memory_read_data_valid),
        .read_data_ready(memory_read_data_ready),
        .read_data(memory_read_data),
        .read_data_span(memory_read_data_span),
        .read_wide_data_valid(memory_wide_read_data_valid),
        .read_wide_data_ready(memory_wide_read_data_ready),
        .read_wide_data(memory_wide_read_data),
        .write_request_valid(memory_write_request_valid),
        .write_request_ready(memory_write_request_ready),
        .write_request(memory_write_request),
        .write_completion(memory_write_completion),
        .write_data_valid(memory_write_data_valid),
        .write_data_ready(memory_write_data_ready),
        .write_data(memory_write_data),
        .read_busy(memory_read_busy),
        .write_busy(memory_write_busy),
        .memory_drained(memory_drained),
        .read_outstanding(memory_read_outstanding),
        .write_outstanding(memory_write_outstanding),
        .read_outstanding_high_water(), .write_outstanding_high_water(),
        .read_credit_stall_cycles(memory_read_credit_stall_cycles),
        .write_credit_stall_cycles(memory_write_credit_stall_cycles),
        .read_burst_count(memory_read_burst_count),
        .write_burst_count(memory_write_burst_count),
        .actual_read_bytes(memory_actual_read_bytes),
        .actual_write_bytes(memory_actual_write_bytes),
        .useful_read_bytes(memory_useful_read_bytes),
        .useful_write_bytes(memory_useful_write_bytes),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready), .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    local_memory local_sram (
        .clk(clk), .rst(rst),
        .layout_update_valid(local_layout_update_valid),
        .layout_update_ready(local_layout_update_ready),
        .next_layout(requested_local_layout),
        .current_layout(current_local_layout),
        .qkv_matmul_overlap_active(qkv_matmul_overlap_active),
        .active_row_count(row_active_rows),
        .rms_active(rms_memory_phase_active),
        .rms_ffn_command(rms_ffn_command),
        .attention_normalized_capture(attention_normalized_capture),
        .rms_gamma_bypass(rms_gamma_bypass),
        .rms_gamma_stage_valid(rms_gamma_stage_valid),
        .rms_gamma_stage_ready(rms_gamma_stage_ready),
        .rms_gamma_write(rms_gamma_write),
        .rms_tile_read_req_valid(rms_tile_read_req_valid),
        .rms_tile_read_req_ready(rms_tile_read_req_ready),
        .rms_tile_read_req(rms_tile_read_req),
        .rms_tile_read_rsp_valid(rms_tile_read_rsp_valid),
        .rms_tile_read_rsp_ready(rms_tile_read_rsp_ready),
        .rms_tile_read_rsp(rms_tile_read_rsp),
        .rms_scratch_read_valid(rms_scratch_read_valid),
        .rms_scratch_read_ready(rms_scratch_read_ready),
        .rms_scratch_read_req(rms_scratch_read_req),
        .rms_scratch_read_rsp_valid(rms_scratch_read_rsp_valid),
        .rms_scratch_read_rsp_ready(rms_scratch_read_rsp_ready),
        .rms_scratch_read_rsp(rms_scratch_read_rsp),
        .rms_scratch_write_valid(rms_scratch_write_valid),
        .rms_scratch_write_ready(rms_scratch_write_ready),
        .rms_scratch_write(rms_scratch_write),
        .refresh_metadata_scratch_req_valid(refresh_metadata_scratch_req_valid),
        .refresh_metadata_scratch_req_ready(refresh_metadata_scratch_req_ready),
        .refresh_metadata_scratch_write(refresh_metadata_scratch_write),
        .refresh_metadata_scratch_address(refresh_metadata_scratch_address),
        .refresh_metadata_scratch_write_data(refresh_metadata_scratch_write_data),
        .refresh_metadata_scratch_write_enable(refresh_metadata_scratch_write_enable),
        .refresh_metadata_scratch_rsp_valid(refresh_metadata_scratch_rsp_valid),
        .refresh_metadata_scratch_rsp_data(refresh_metadata_scratch_rsp_data),
        .refresh_metadata_scratch_aux_write_valid(refresh_metadata_scratch_aux_write_valid),
        .refresh_metadata_scratch_aux_write_ready(refresh_metadata_scratch_aux_write_ready),
        .refresh_metadata_scratch_aux_write_address(refresh_metadata_scratch_aux_write_address),
        .refresh_metadata_scratch_aux_write_data(refresh_metadata_scratch_aux_write_data),
        .refresh_metadata_scratch_aux_write_enable(refresh_metadata_scratch_aux_write_enable),
        .rms_norm_write_valid(rms_norm_write_valid),
        .rms_norm_write_ready(rms_norm_write_ready),
        .rms_norm_write(rms_norm_write),
        .hidden_write_valid(hidden_local_write_valid),
        .hidden_write_ready(hidden_local_write_ready),
        .hidden_write_req(hidden_local_write),
        .hidden_read_valid(hidden_local_read_valid),
        .hidden_read_ready(hidden_local_read_ready),
        .hidden_read_req(hidden_local_read_req),
        .hidden_read_response_valid(hidden_local_read_response_valid),
        .hidden_read_rsp(hidden_local_read_rsp),
        .context_write_valid(attention_context_buffer_write_valid),
        .context_write_ready(attention_context_buffer_write_ready),
        .context_write_req(attention_context_write),
        .context_read_req_valid(attention_context_read_req_valid),
        .context_read_req_ready(attention_context_read_req_ready),
        .context_read_req(attention_context_read_req),
        .context_read_rsp_valid(attention_context_read_rsp_valid),
        .context_read_rsp_ready(attention_context_read_rsp_ready),
        .context_read_rsp(attention_context_read_rsp),
        .attention_scratch_read_valid(
            attention_softmax_scratch_read_valid),
        .attention_scratch_read_ready(
            attention_softmax_scratch_read_ready),
        .attention_scratch_read_req(attention_softmax_scratch_read_req),
        .attention_scratch_rsp_valid(attention_softmax_scratch_rsp_valid),
        .attention_scratch_rsp_ready(attention_softmax_scratch_rsp_ready),
        .attention_scratch_rsp(attention_softmax_scratch_rsp),
        .attention_scratch_write_valid(
            attention_softmax_scratch_write_valid),
        .attention_scratch_write_ready(
            attention_softmax_scratch_write_ready),
        .attention_scratch_write_req(attention_softmax_scratch_write),
        .qkv_head_read_req_valid(qkv_head_read_req_valid),
        .qkv_head_read_req_ready(qkv_head_read_req_ready),
        .qkv_head_read_req(qkv_head_read_req),
        .qkv_head_read_rsp_valid(qkv_head_read_rsp_valid),
        .qkv_head_read_rsp_ready(qkv_head_read_rsp_ready),
        .qkv_head_read_rsp(qkv_head_read_rsp),
        .rope_source_read_req_valid(rope_source_read_req_valid),
        .rope_source_read_req_ready(rope_source_read_req_ready),
        .rope_source_read_req(rope_source_read_req),
        .rope_source_read_rsp_valid(rope_source_read_rsp_valid),
        .rope_source_read_rsp_ready(rope_source_read_rsp_ready),
        .rope_source_read_rsp(rope_source_read_rsp),
        .rope_cos_read_req_valid(rope_cos_read_req_valid),
        .rope_cos_read_req_ready(rope_cos_read_req_ready),
        .rope_constant_read_req(rope_constant_read_req),
        .rope_cos_read_rsp_valid(rope_cos_read_rsp_valid),
        .rope_cos_read_rsp_ready(rope_cos_read_rsp_ready),
        .rope_cos_read_rsp(rope_cos_read_rsp),
        .rope_sin_read_req_valid(rope_sin_read_req_valid),
        .rope_sin_read_req_ready(rope_sin_read_req_ready),
        .rope_sin_read_rsp_valid(rope_sin_read_rsp_valid),
        .rope_sin_read_rsp_ready(rope_sin_read_rsp_ready),
        .rope_sin_read_rsp(rope_sin_read_rsp),
        .rope_destination_write_valid(rope_destination_write_valid),
        .rope_destination_write_ready(rope_destination_write_ready),
        .rope_destination_write(rope_destination_write),
        .qkv_head_stage_write_valid(qkv_head_stage_write_valid),
        .qkv_head_stage_write_ready(qkv_head_stage_write_ready),
        .qkv_head_stage_write(qkv_head_stage_write),
        .qkv_head_tile_read_req_valid(qkv_head_tile_memory_req_valid),
        .qkv_head_tile_read_req_ready(qkv_head_tile_memory_req_ready),
        .qkv_head_tile_read_req(qkv_head_tile_read_req),
        .qkv_head_tile_read_rsp_valid(qkv_head_tile_memory_rsp_valid),
        .qkv_head_tile_read_rsp(qkv_head_tile_read_rsp),
        .qkv_constant_write_valid(qkv_constant_stage_valid),
        .qkv_constant_write_ready(qkv_constant_stage_ready),
        .qkv_constant_write(qkv_constant_write),
        .qkv_q_write_valid(qkv_q_local_write_valid),
        .qkv_q_write_ready(qkv_q_local_write_ready),
        .qkv_q_write(qkv_q_write),
        .attention_panel_write_valid(attention_panel_write_valid),
        .attention_panel_write_ready(attention_panel_write_ready),
        .attention_panel_write(attention_panel_write),
        .attention_panel_scale_write_valid(attention_scale_write_valid),
        .attention_panel_scale_write_ready(attention_scale_write_ready),
        .attention_panel_scale_write(attention_panel_scale_write),
        .attention_operand_req_valid(attention_operand_req_valid),
        .attention_operand_req_ready(attention_operand_req_ready),
        .attention_operand_req(attention_operand_req),
        .attention_operand_rsp_valid(attention_operand_rsp_valid),
        .attention_operand_rsp(attention_operand_rsp),
        .attention_operand_abort_request(attention_operand_abort_request),
        .attention_operand_abort_ack(attention_operand_abort_ack),
        .attention_score_write_valid(attention_score_write_valid),
        .attention_score_write_ready(attention_score_write_ready),
        .attention_score_write(attention_score_write),
        .attention_score_read_req_valid(attention_softmax_score_req_valid),
        .attention_score_read_req_ready(attention_softmax_score_req_ready),
        .attention_score_read_slot_ready(attention_softmax_score_slot_ready),
        .attention_score_read_req(attention_score_read_req),
        .attention_score_read_rsp_valid(
            attention_softmax_score_memory_rsp_valid),
        .attention_score_read_rsp(attention_score_read_rsp),
        .attention_probability_write_valid(attention_probability_write_valid),
        .attention_probability_write_ready(attention_probability_write_ready),
        .attention_probability_write(attention_probability_write),
        .attention_probability_quantized_write_valid(
            attention_probability_quantized_write_valid),
        .attention_probability_quantized_write_ready(
            attention_probability_quantized_write_ready),
        .attention_probability_quantized_write(
            attention_probability_quantized_write),
        .attention_probability_scale_write_valid(
            attention_probability_scale_write_valid),
        .attention_probability_scale_write_ready(
            attention_probability_scale_write_ready),
        .attention_probability_scale_write(attention_probability_scale_write),
        .matmul_local_source_req_valid(matmul_source_req_valid),
        .matmul_local_source_req_ready(matmul_local_source_req_ready),
        .matmul_local_source_req(matmul_source_req),
        .matmul_local_source_rsp_valid(matmul_local_source_rsp_valid),
        .matmul_local_source_rsp_ready(matmul_source_rsp_ready),
        .matmul_local_source_rsp(matmul_source_rsp),
        .matmul_activation_write_valid(matmul_activation_write_valid),
        .matmul_activation_write_ready(matmul_activation_write_ready),
        .matmul_activation_write(matmul_activation_write),
        .matmul_read_bundle_valid(matmul_read_bundle_valid),
        .matmul_read_bundle_ready(matmul_read_bundle_ready),
        .matmul_read_req(matmul_read_req),
        .matmul_read_response_valid(matmul_read_response_valid),
        .matmul_read_rsp(matmul_read_rsp),
        .matmul_panel_write_valid(matmul_panel_write_valid),
        .matmul_panel_write_ready(matmul_panel_write_ready),
        .matmul_panel_write(matmul_panel_write),
        .lm_head_scale_write_valid, .lm_head_scale_write_ready,
        .lm_head_scale_write_address, .lm_head_scale_write_data,
        .lm_head_scale_read_valid, .lm_head_scale_read_ready,
        .lm_head_scale_read_address, .lm_head_scale_read_rsp_valid,
        .lm_head_scale_read_rsp_data, .candidate_state_read_valid,
        .candidate_state_read_ready, .candidate_state_read_row_group,
        .candidate_state_read_lane_block, .candidate_state_read_row_half,
        .candidate_state_read_word, .candidate_state_read_row_mask,
        .candidate_state_read_tag, .candidate_state_read_rsp_valid,
        .candidate_state_read_rsp_ready, .candidate_state_read_rsp_data,
        .candidate_state_read_rsp_row_mask, .candidate_state_read_rsp_tag,
        .candidate_state_write_valid, .candidate_state_write_ready,
        .candidate_state_write_row_group, .candidate_state_write_lane_block,
        .candidate_state_write_row_half, .candidate_state_write_word,
        .candidate_state_write_row_mask, .candidate_state_write_data,
        .post_table_access_valid, .post_table_access_ready,
        .post_table_access_enable, .post_state_write_valid,
        .post_state_write_ready, .post_state_write_word_address,
        .post_state_write_data, .post_state_read_valid,
        .post_state_read_ready, .post_state_read_word_address,
        .post_state_read_tag, .post_state_read_rsp_valid,
        .post_state_read_rsp_data, .post_state_read_rsp_tag,
        .post_suppressed_write_valid, .post_suppressed_write_ready,
        .post_suppressed_write_address, .post_suppressed_write_data,
        .post_suppressed_read_valid, .post_suppressed_read_ready,
        .post_suppressed_read_index, .post_suppressed_read_rsp_valid,
        .post_suppressed_read_rsp_token_id,
        .matmul_combine_write_valid(matmul_combine_write_valid),
        .matmul_combine_write_ready(matmul_combine_write_ready),
        .matmul_combine_write(matmul_combine_write),
        .matmul_combine_read_valid(matmul_combine_read_valid),
        .matmul_combine_read_ready(matmul_combine_read_ready),
        .matmul_combine_read_req(matmul_combine_read_req),
        .matmul_combine_read_response_valid(
            matmul_combine_read_response_valid),
        .matmul_combine_read_rsp(matmul_combine_read_rsp),
        .matmul_local_output_write_valid(matmul_local_output_write_valid),
        .matmul_local_output_write_ready(matmul_local_output_write_ready),
        .matmul_local_output_write(matmul_local_output_write),
        .elementwise_product_req_valid(elementwise_product_req_valid),
        .elementwise_product_req_ready(elementwise_product_req_ready),
        .elementwise_product_req(elementwise_product_req),
        .elementwise_product_rsp_valid(elementwise_product_rsp_valid),
        .elementwise_product_rsp_ready(elementwise_product_rsp_ready),
        .elementwise_product_rsp(elementwise_product_rsp),
        .matmul_residual_read_valid(matmul_residual_local_read_issue),
        .matmul_residual_read_ready(matmul_residual_memory_read_ready),
        .matmul_residual_read_req(matmul_residual_read_req),
        .matmul_residual_read_rsp_valid(
            matmul_residual_memory_read_rsp_valid),
        .matmul_residual_read_rsp(matmul_residual_read_rsp),
        .port_req_ready(local_port_req_ready),
        .port_read_valid(local_port_read_valid),
        .port_read_data(local_port_read_data)
    );

    operator_subsystem #(
        .ATTENTION_HEAD_COUNT(ATTENTION_HEAD_COUNT),
        .HIDDEN_FEATURES(HIDDEN_FEATURES),
        .FFN_FEATURES(FFN_FEATURES)
    ) operators (
        .clk(clk),
        .rst(rst),
        .execution_memory_stage(execution_memory_stage),
        .execution_block_schedule_active(execution_block_schedule_active),
        .execution_forward_postprocess_start_valid(
            execution_forward_postprocess_start_valid),
        .execution_forward_postprocess_start_ready(
            execution_forward_postprocess_start_ready),
        .execution_forward_postprocess_configuration_bits(
            execution_forward_postprocess_configuration_bits),
        .execution_refresh_configuration_address,
        .execution_refresh_prepare,
        .execution_refresh_loaded_token_valid, .execution_refresh_loaded_token_position, .execution_refresh_loaded_token_kv_write,
        .execution_refresh_select,
        .execution_refresh_relation_only,
        .execution_refresh_relation_l31, .execution_refresh_closeout,
        .execution_forward_postprocess_block_complete,
        .execution_refresh_relation_layer_mask,
        .execution_joint_configuration_address,
        .execution_forward_postprocess_done_valid(execution_forward_postprocess_done_valid),
        .execution_forward_postprocess_done_ready(execution_forward_postprocess_done_ready),
        .execution_forward_postprocess_done_error(execution_forward_postprocess_done_error),
        .execution_forward_postprocess_done_error_id(
            execution_forward_postprocess_done_error_id),
        .execution_hidden_start_valid(execution_hidden_start_valid),
        .execution_hidden_start_ready(execution_hidden_start_ready),
        .execution_hidden_start_write(execution_hidden_start_write),
        .execution_hidden_start_residual(execution_hidden_start_residual),
        .execution_hidden_start_address(execution_hidden_start_address),
        .execution_hidden_start_bytes(execution_hidden_start_bytes),
        .execution_hidden_start_active_rows(execution_hidden_start_active_rows),
        .execution_hidden_start_row_permutation(execution_hidden_start_row_permutation),
        .execution_hidden_start_direct_row_index_enable(
            execution_hidden_start_direct_row_index_enable),
        .execution_hidden_start_direct_row_index(
            execution_hidden_start_direct_row_index),
        .execution_hidden_start_source_descriptor_enable(
            execution_hidden_start_source_descriptor_enable),
        .execution_hidden_start_source_index(
            execution_hidden_start_source_index),
        .execution_hidden_start_source_embedding(
            execution_hidden_start_source_embedding),
        .execution_hidden_start_embedding_base(
            execution_hidden_start_embedding_base),
        .execution_hidden_start_embedding_limit(
            execution_hidden_start_embedding_limit),
        .execution_hidden_start_limit(execution_hidden_start_limit),
        .execution_hidden_done_valid(execution_hidden_done_valid),
        .execution_hidden_done_ready(execution_hidden_done_ready),
        .execution_hidden_error(execution_hidden_error),
        .operator_command_valid(operator_command_valid),
        .operator_command_ready(operator_command_ready),
        .pending_command(operator_command),
        .operator_command_done_valid(operator_command_done_valid),
        .operator_command_done_ready(operator_command_done_ready),
        .execution_completion(operator_command_completion),
        .row_active_rows(row_active_rows),
        .row_residual_storage_rows(row_residual_storage_rows),
        .row_sequence_length(row_sequence_length),
        .matmul_row_config(matmul_row_config),
        .hidden_row_config(hidden_row_config),
        .attention_row_config(attention_row_config),
        .layer_entry_valid(layer_entry_valid),
        .layer_entry_index(layer_entry_index),
        .matmul_weight_config(matmul_weight_config),
        .rmsnorm_layer_config(rmsnorm_layer_config),
        .qkv_layer_config(qkv_layer_config),
        .attention_cache_config(attention_cache_config),
        .attention_context_config(attention_context_config),
        .ffn_workspace_config(ffn_workspace_config),
        .ffn_batch_pair_config(ffn_batch_pair_config),
        .attention_pair_enable(attention_pair_enable),
        .attention_output_subset_enable(attention_output_subset_enable),
        .attention_output_token_count(attention_output_token_count),
        .attention_pair_second_rows(attention_pair_second_rows),
        .attention_third_batch_enable(attention_third_batch_enable),
        .attention_third_rows(attention_third_rows),
        .attention_second_batch(attention_second_batch),
        .attention_batch_index(attention_batch_index),
        .kv_pair_enable(kv_pair_enable), .kv_pair_second_config(kv_pair_second_config),
        .kv_batch_group_config(kv_batch_group_config),
        .qkv_memory_stage(qkv_memory_stage),
        .attention_normalized_capture(attention_normalized_capture),
        .matmul_memory_stage(matmul_memory_stage),
        .rms_gamma_loaded(rms_gamma_loaded),
        .attention_read_source(attention_read_source),
        .dma_read_stream_idle(dma_read_stream_idle),
        .qkv_residual_spill_active(qkv_residual_spill_active),
        .dma_read_request_valid(operator_dma_read_request_valid),
        .dma_read_request_ready(operator_dma_read_request_ready),
        .dma_read_request(operator_dma_read_request),
        .elementwise_read_second_span_valid,
        .elementwise_read_second_span_address,
        .elementwise_read_pair_stride,
        .elementwise_read_pair_count,
        .matmul_aux_read_second_span_valid,
        .matmul_aux_read_second_span_address,
        .matmul_aux_read_pair_stride,
        .matmul_aux_read_pair_count,
        .dma_read_data_valid(operator_dma_read_data_valid),
        .dma_read_data_ready(operator_dma_read_data_ready),
        .dma_read_data(operator_dma_read_data),
        .dma_wide_read_data(operator_dma_wide_read_data),
        .elementwise_read_data_span,
        .dma_read_completion(operator_dma_read_completion),
        .dma_write_request_valid(operator_dma_write_request_valid),
        .dma_write_request_ready(operator_dma_write_request_ready),
        .dma_write_request(operator_dma_write_request),
        .dma_write_data_valid(operator_dma_write_data_valid),
        .dma_write_data_ready(operator_dma_write_data_ready),
        .dma_write_data(operator_dma_write_data),
        .dma_write_completion(operator_dma_write_completion),
        .memory_abort_request(memory_abort_request),
        .local_layout_update_valid(local_layout_update_valid),
        .requested_layout(requested_local_layout),
        .current_layout(current_local_layout),
        .qkv_matmul_overlap_active(qkv_matmul_overlap_active),
        .rms_memory_phase_active(rms_memory_phase_active),
        .rms_ffn_command(rms_ffn_command),
        .rms_gamma_bypass(rms_gamma_bypass),
        .rms_gamma_stage_valid(rms_gamma_stage_valid),
        .rms_gamma_stage_ready(rms_gamma_stage_ready),
        .rms_gamma_write(rms_gamma_write),
        .rms_tile_read_req_valid(rms_tile_read_req_valid),
        .rms_tile_read_req_ready(rms_tile_read_req_ready),
        .rms_tile_read_req(rms_tile_read_req),
        .rms_tile_read_rsp_valid(rms_tile_read_rsp_valid),
        .rms_tile_read_rsp_ready(rms_tile_read_rsp_ready),
        .rms_tile_read_rsp(rms_tile_read_rsp),
        .rms_scratch_read_valid(rms_scratch_read_valid),
        .rms_scratch_read_ready(rms_scratch_read_ready),
        .rms_scratch_read_req(rms_scratch_read_req),
        .rms_scratch_read_rsp_valid(rms_scratch_read_rsp_valid),
        .rms_scratch_read_rsp_ready(rms_scratch_read_rsp_ready),
        .rms_scratch_read_rsp(rms_scratch_read_rsp),
        .rms_scratch_write_valid(rms_scratch_write_valid),
        .rms_scratch_write_ready(rms_scratch_write_ready),
        .rms_scratch_write(rms_scratch_write),
        .refresh_metadata_scratch_req_valid(refresh_metadata_scratch_req_valid),
        .refresh_metadata_scratch_req_ready(refresh_metadata_scratch_req_ready),
        .refresh_metadata_scratch_write(refresh_metadata_scratch_write),
        .refresh_metadata_scratch_address(refresh_metadata_scratch_address),
        .refresh_metadata_scratch_write_data(refresh_metadata_scratch_write_data),
        .refresh_metadata_scratch_write_enable(refresh_metadata_scratch_write_enable),
        .refresh_metadata_scratch_rsp_valid(refresh_metadata_scratch_rsp_valid),
        .refresh_metadata_scratch_rsp_data(refresh_metadata_scratch_rsp_data),
        .refresh_metadata_scratch_aux_write_valid(refresh_metadata_scratch_aux_write_valid),
        .refresh_metadata_scratch_aux_write_ready(refresh_metadata_scratch_aux_write_ready),
        .refresh_metadata_scratch_aux_write_address(refresh_metadata_scratch_aux_write_address),
        .refresh_metadata_scratch_aux_write_data(refresh_metadata_scratch_aux_write_data),
        .refresh_metadata_scratch_aux_write_enable(refresh_metadata_scratch_aux_write_enable),
        .rms_norm_write_valid(rms_norm_write_valid),
        .rms_norm_write_ready(rms_norm_write_ready),
        .rms_norm_write(rms_norm_write),
        .hidden_local_write_valid(hidden_local_write_valid),
        .hidden_local_write_ready(hidden_local_write_ready),
        .hidden_local_write(hidden_local_write),
        .hidden_local_read_valid(hidden_local_read_valid),
        .hidden_local_read_ready(hidden_local_read_ready),
        .hidden_local_read_req(hidden_local_read_req),
        .hidden_local_read_response_valid(hidden_local_read_response_valid),
        .hidden_local_read_rsp(hidden_local_read_rsp),
        .attention_context_buffer_write_valid(attention_context_buffer_write_valid),
        .attention_context_buffer_write_ready(attention_context_buffer_write_ready),
        .attention_context_write(attention_context_write),
        .attention_context_read_req_valid(attention_context_read_req_valid),
        .attention_context_read_req_ready(attention_context_read_req_ready),
        .attention_context_read_req(attention_context_read_req),
        .attention_context_read_rsp_valid(attention_context_read_rsp_valid),
        .attention_context_read_rsp_ready(attention_context_read_rsp_ready),
        .attention_context_read_rsp(attention_context_read_rsp),
        .attention_softmax_scratch_read_valid(attention_softmax_scratch_read_valid),
        .attention_softmax_scratch_read_ready(attention_softmax_scratch_read_ready),
        .attention_softmax_scratch_read_req(attention_softmax_scratch_read_req),
        .attention_softmax_scratch_rsp_valid(attention_softmax_scratch_rsp_valid),
        .attention_softmax_scratch_rsp_ready(attention_softmax_scratch_rsp_ready),
        .attention_softmax_scratch_rsp(attention_softmax_scratch_rsp),
        .attention_softmax_scratch_write_valid(attention_softmax_scratch_write_valid),
        .attention_softmax_scratch_write_ready(attention_softmax_scratch_write_ready),
        .attention_softmax_scratch_write(attention_softmax_scratch_write),
        .qkv_head_read_req_valid(qkv_head_read_req_valid),
        .qkv_head_read_req_ready(qkv_head_read_req_ready),
        .qkv_head_read_req(qkv_head_read_req),
        .qkv_head_read_rsp_valid(qkv_head_read_rsp_valid),
        .qkv_head_read_rsp_ready(qkv_head_read_rsp_ready),
        .qkv_head_read_rsp(qkv_head_read_rsp),
        .rope_source_read_req_valid(rope_source_read_req_valid),
        .rope_source_read_req_ready(rope_source_read_req_ready),
        .rope_source_read_req(rope_source_read_req),
        .rope_source_read_rsp_valid(rope_source_read_rsp_valid),
        .rope_source_read_rsp_ready(rope_source_read_rsp_ready),
        .rope_source_read_rsp(rope_source_read_rsp),
        .rope_cos_read_req_valid(rope_cos_read_req_valid),
        .rope_cos_read_req_ready(rope_cos_read_req_ready),
        .rope_constant_read_req(rope_constant_read_req),
        .rope_cos_read_rsp_valid(rope_cos_read_rsp_valid),
        .rope_cos_read_rsp_ready(rope_cos_read_rsp_ready),
        .rope_cos_read_rsp(rope_cos_read_rsp),
        .rope_sin_read_req_valid(rope_sin_read_req_valid),
        .rope_sin_read_req_ready(rope_sin_read_req_ready),
        .rope_sin_read_rsp_valid(rope_sin_read_rsp_valid),
        .rope_sin_read_rsp_ready(rope_sin_read_rsp_ready),
        .rope_sin_read_rsp(rope_sin_read_rsp),
        .rope_destination_write_valid(rope_destination_write_valid),
        .rope_destination_write_ready(rope_destination_write_ready),
        .rope_destination_write(rope_destination_write),
        .qkv_head_stage_write_valid(qkv_head_stage_write_valid),
        .qkv_head_stage_write_ready(qkv_head_stage_write_ready),
        .qkv_head_stage_write(qkv_head_stage_write),
        .qkv_head_tile_memory_req_valid(qkv_head_tile_memory_req_valid),
        .qkv_head_tile_memory_req_ready(qkv_head_tile_memory_req_ready),
        .qkv_head_tile_read_req(qkv_head_tile_read_req),
        .qkv_head_tile_memory_rsp_valid(qkv_head_tile_memory_rsp_valid),
        .qkv_head_tile_read_rsp(qkv_head_tile_read_rsp),
        .qkv_constant_stage_valid(qkv_constant_stage_valid),
        .qkv_constant_stage_ready(qkv_constant_stage_ready),
        .qkv_constant_write(qkv_constant_write),
        .qkv_q_local_write_valid(qkv_q_local_write_valid),
        .qkv_q_local_write_ready(qkv_q_local_write_ready),
        .qkv_q_write(qkv_q_write),
        .attention_panel_write_valid(attention_panel_write_valid),
        .attention_panel_write_ready(attention_panel_write_ready),
        .attention_panel_write(attention_panel_write),
        .attention_scale_write_valid(attention_scale_write_valid),
        .attention_scale_write_ready(attention_scale_write_ready),
        .attention_panel_scale_write(attention_panel_scale_write),
        .attention_operand_req_valid(attention_operand_req_valid),
        .attention_operand_req_ready(attention_operand_req_ready),
        .attention_operand_req(attention_operand_req),
        .attention_operand_rsp_valid(attention_operand_rsp_valid),
        .attention_operand_rsp(attention_operand_rsp),
        .attention_operand_abort_request(attention_operand_abort_request),
        .attention_operand_abort_ack(attention_operand_abort_ack),
        .attention_score_write_valid(attention_score_write_valid),
        .attention_score_write_ready(attention_score_write_ready),
        .attention_score_write(attention_score_write),
        .attention_softmax_score_req_valid(attention_softmax_score_req_valid),
        .attention_softmax_score_req_ready(attention_softmax_score_req_ready),
        .attention_softmax_score_slot_ready(attention_softmax_score_slot_ready),
        .attention_score_read_req(attention_score_read_req),
        .attention_softmax_score_memory_rsp_valid(attention_softmax_score_memory_rsp_valid),
        .attention_score_read_rsp(attention_score_read_rsp),
        .attention_probability_write_valid(attention_probability_write_valid),
        .attention_probability_write_ready(attention_probability_write_ready),
        .attention_probability_write(attention_probability_write),
        .attention_probability_quantized_write_valid(attention_probability_quantized_write_valid),
        .attention_probability_quantized_write_ready(attention_probability_quantized_write_ready),
        .attention_probability_quantized_write(
            attention_probability_quantized_write),
        .attention_probability_scale_write_valid(attention_probability_scale_write_valid),
        .attention_probability_scale_write_ready(attention_probability_scale_write_ready),
        .attention_probability_scale_write(attention_probability_scale_write),
        .matmul_source_req_valid(matmul_source_req_valid),
        .matmul_local_source_req_ready(matmul_local_source_req_ready),
        .matmul_source_req(matmul_source_req),
        .matmul_local_source_rsp_valid(matmul_local_source_rsp_valid),
        .matmul_source_rsp_ready(matmul_source_rsp_ready),
        .matmul_source_rsp(matmul_source_rsp),
        .matmul_activation_write_valid(matmul_activation_write_valid),
        .matmul_activation_write_ready(matmul_activation_write_ready),
        .matmul_activation_write(matmul_activation_write),
        .matmul_read_bundle_valid(matmul_read_bundle_valid),
        .matmul_read_bundle_ready(matmul_read_bundle_ready),
        .matmul_read_req(matmul_read_req),
        .matmul_read_response_valid(matmul_read_response_valid),
        .matmul_read_rsp(matmul_read_rsp),
        .matmul_panel_write_valid(matmul_panel_write_valid),
        .matmul_panel_write_ready(matmul_panel_write_ready),
        .matmul_panel_write(matmul_panel_write),
        .matmul_combine_write_valid(matmul_combine_write_valid),
        .matmul_combine_write_ready(matmul_combine_write_ready),
        .matmul_combine_write(matmul_combine_write),
        .matmul_combine_read_valid(matmul_combine_read_valid),
        .matmul_combine_read_ready(matmul_combine_read_ready),
        .matmul_combine_read_req(matmul_combine_read_req),
        .matmul_combine_read_response_valid(matmul_combine_read_response_valid),
        .matmul_combine_read_rsp(matmul_combine_read_rsp),
        .matmul_local_output_write_valid(matmul_local_output_write_valid),
        .matmul_local_output_write_ready(matmul_local_output_write_ready),
        .matmul_local_output_write(matmul_local_output_write),
        .elementwise_product_req_valid(elementwise_product_req_valid),
        .elementwise_product_req_ready(elementwise_product_req_ready),
        .elementwise_product_req(elementwise_product_req),
        .elementwise_product_rsp_valid(elementwise_product_rsp_valid),
        .elementwise_product_rsp_ready(elementwise_product_rsp_ready),
        .elementwise_product_rsp(elementwise_product_rsp),
        .matmul_residual_local_read_issue(matmul_residual_local_read_issue),
        .matmul_residual_memory_read_ready(matmul_residual_memory_read_ready),
        .matmul_residual_read_req(matmul_residual_read_req),
        .matmul_residual_memory_read_rsp_valid(matmul_residual_memory_read_rsp_valid),
        .matmul_residual_read_rsp(matmul_residual_read_rsp),
        .lm_head_scale_write_valid, .lm_head_scale_write_ready,
        .lm_head_scale_write_address, .lm_head_scale_write_data,
        .lm_head_scale_read_valid, .lm_head_scale_read_ready,
        .lm_head_scale_read_address, .lm_head_scale_read_rsp_valid,
        .lm_head_scale_read_rsp_data, .candidate_state_read_valid,
        .candidate_state_read_ready, .candidate_state_read_row_group,
        .candidate_state_read_lane_block, .candidate_state_read_row_half,
        .candidate_state_read_word, .candidate_state_read_row_mask,
        .candidate_state_read_tag, .candidate_state_read_rsp_valid,
        .candidate_state_read_rsp_ready, .candidate_state_read_rsp_data,
        .candidate_state_read_rsp_row_mask, .candidate_state_read_rsp_tag,
        .candidate_state_write_valid, .candidate_state_write_ready,
        .candidate_state_write_row_group, .candidate_state_write_lane_block,
        .candidate_state_write_row_half, .candidate_state_write_word,
        .candidate_state_write_row_mask, .candidate_state_write_data,
        .post_table_access_valid, .post_table_access_ready,
        .post_table_access_enable, .post_state_write_valid,
        .post_state_write_ready, .post_state_write_word_address,
        .post_state_write_data, .post_state_read_valid,
        .post_state_read_ready, .post_state_read_word_address,
        .post_state_read_tag, .post_state_read_rsp_valid,
        .post_state_read_rsp_data, .post_state_read_rsp_tag,
        .post_suppressed_write_valid, .post_suppressed_write_ready,
        .post_suppressed_write_address, .post_suppressed_write_data,
        .post_suppressed_read_valid, .post_suppressed_read_ready,
        .post_suppressed_read_index, .post_suppressed_read_rsp_valid,
        .post_suppressed_read_rsp_token_id
    );

`ifndef SYNTHESIS
    initial begin
        assert (ATTENTION_HEAD_COUNT >= 1 && ATTENTION_HEAD_COUNT <= 32)
            else $error("supra_top ATTENTION_HEAD_COUNT must be in 1..32");
    end
`endif
endmodule

`default_nettype wire
