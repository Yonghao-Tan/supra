`timescale 1ns/1ps
`default_nettype none

// Executes accepted operator commands and connects their SRAM/DMA requests
// to shared compute resources according to the fixed operator schedule.
module operator_subsystem #(
    parameter integer ATTENTION_HEAD_COUNT = 32,
    parameter integer HIDDEN_FEATURES = 4096,
    parameter integer FFN_FEATURES = 12288
) (
    input  logic clk,
    input  logic rst,
    input  logic [2:0] execution_memory_stage,
    input  logic execution_block_schedule_active,
    input  logic execution_forward_postprocess_start_valid,
    output logic execution_forward_postprocess_start_ready,
    input  logic [2559:0] execution_forward_postprocess_configuration_bits,
    input  logic [63:0] execution_refresh_configuration_address,
    input  logic execution_refresh_prepare,
    input logic execution_refresh_loaded_token_valid, execution_refresh_loaded_token_kv_write,
    input logic [10:0] execution_refresh_loaded_token_position,
    input  logic execution_refresh_select,
    input  logic execution_refresh_relation_only,
    input  logic execution_refresh_relation_l31,
    input  logic execution_refresh_closeout,
    output logic [31:0] execution_refresh_relation_layer_mask,
    input  logic [63:0] execution_joint_configuration_address,
    output logic execution_forward_postprocess_done_valid,
    output logic execution_forward_postprocess_block_complete,
    input  logic execution_forward_postprocess_done_ready,
    output logic execution_forward_postprocess_done_error,
    output logic [7:0] execution_forward_postprocess_done_error_id,
    input  logic execution_hidden_start_valid,
    output logic execution_hidden_start_ready,
    input  logic execution_hidden_start_write,
    input  logic execution_hidden_start_residual,
    input  logic [63:0] execution_hidden_start_address,
    input  logic [31:0] execution_hidden_start_bytes,
    input  logic [5:0] execution_hidden_start_active_rows,
    input  logic [287:0] execution_hidden_start_row_permutation,
    input  logic execution_hidden_start_direct_row_index_enable,
    input  logic [527:0] execution_hidden_start_direct_row_index,
    input  logic execution_hidden_start_source_descriptor_enable,
    input  logic [815:0] execution_hidden_start_source_index,
    input  logic [47:0] execution_hidden_start_source_embedding,
    input  logic [63:0] execution_hidden_start_embedding_base,
    input  logic [63:0] execution_hidden_start_embedding_limit,
    input  logic [63:0] execution_hidden_start_limit,
    output logic execution_hidden_done_valid,
    input  logic execution_hidden_done_ready,
    output logic execution_hidden_error,
    input  logic operator_command_valid,
    output logic operator_command_ready,
    input  hardware_types_pkg::operator_command_t pending_command,
    output logic operator_command_done_valid,
    input  logic operator_command_done_ready,
    output hardware_types_pkg::operator_completion_t execution_completion,
    input  logic [5:0] row_active_rows,
    input  logic [5:0] row_residual_storage_rows,
    input  logic [11:0] row_sequence_length,
    input  hardware_types_pkg::matmul_row_config_t matmul_row_config,
    input  hardware_types_pkg::hidden_row_config_t hidden_row_config,
    input  hardware_types_pkg::attention_row_config_t attention_row_config,
    input  logic layer_entry_valid,
    input  logic [4:0] layer_entry_index,
    input  hardware_types_pkg::matmul_weight_config_t matmul_weight_config,
    input  hardware_types_pkg::rmsnorm_layer_config_t rmsnorm_layer_config,
    input  hardware_types_pkg::qkv_layer_config_t qkv_layer_config,
    input  hardware_types_pkg::attention_cache_config_t attention_cache_config,
    input  hardware_types_pkg::attention_context_config_t attention_context_config,
    input  hardware_types_pkg::ffn_workspace_config_t ffn_workspace_config,
    input  hardware_types_pkg::ffn_batch_pair_config_t ffn_batch_pair_config,
    input logic attention_pair_enable,
    input logic attention_normalized_capture,
    input logic kv_pair_enable,
    input hardware_types_pkg::attention_row_config_t kv_pair_second_config,
    input hardware_types_pkg::attention_batch_group_config_t kv_batch_group_config,
    input logic attention_output_subset_enable,
    input logic [5:0] attention_output_token_count,
    input logic [5:0] attention_pair_second_rows,
    input logic attention_third_batch_enable,
    input logic [5:0] attention_third_rows,
    output logic attention_second_batch,
    output logic [1:0] attention_batch_index,
    output logic [1:0] qkv_memory_stage,
    output logic [1:0] matmul_memory_stage,
    output logic rms_gamma_loaded,
    output hardware_types_pkg::attention_read_source_t attention_read_source,
    input  logic dma_read_stream_idle,
    output logic qkv_residual_spill_active,
    output logic [6:0] dma_read_request_valid,
    input  logic [6:0] dma_read_request_ready,
    output hardware_types_pkg::dma_read_request_t dma_read_request [0:6],
    output logic elementwise_read_second_span_valid,
    output logic [63:0] elementwise_read_second_span_address,
    output logic [31:0] elementwise_read_pair_stride,
    output logic [10:0] elementwise_read_pair_count,
    output logic matmul_aux_read_second_span_valid,
    output logic [63:0] matmul_aux_read_second_span_address,
    output logic [31:0] matmul_aux_read_pair_stride,
    output logic [10:0] matmul_aux_read_pair_count,
    input  logic [6:0] dma_read_data_valid,
    output logic [6:0] dma_read_data_ready,
    input  hardware_types_pkg::dma_read_beat_t dma_read_data [0:6],
    input  hardware_types_pkg::dma_wide_read_beat_t
        dma_wide_read_data [0:6],
    input  logic elementwise_read_data_span,
    input  hardware_types_pkg::dma_completion_t dma_read_completion [0:6],
    output logic [3:0] dma_write_request_valid,
    input  logic [3:0] dma_write_request_ready,
    output hardware_types_pkg::dma_write_request_t dma_write_request [0:3],
    output logic [3:0] dma_write_data_valid,
    input  logic [3:0] dma_write_data_ready,
    output hardware_types_pkg::dma_write_beat_t dma_write_data [0:3],
    input  hardware_types_pkg::dma_completion_t dma_write_completion [0:3],
    output logic memory_abort_request,
    output logic local_layout_update_valid,
    output local_memory_layout_pkg::layout_requirement_t requested_layout,
    input  local_memory_layout_pkg::layout_requirement_t current_layout,
    output logic qkv_matmul_overlap_active,
    output logic rms_memory_phase_active,
    output logic rms_ffn_command,
    output logic rms_gamma_bypass,
    output logic rms_gamma_stage_valid,
    input  logic rms_gamma_stage_ready,
    output hardware_types_pkg::rms_gamma_write_t rms_gamma_write,
    output logic rms_tile_read_req_valid,
    input  logic rms_tile_read_req_ready,
    output hardware_types_pkg::rms_tile_read_request_t rms_tile_read_req,
    input  logic rms_tile_read_rsp_valid,
    output logic rms_tile_read_rsp_ready,
    input  hardware_types_pkg::rms_tile_read_response_t rms_tile_read_rsp,
    output logic rms_scratch_read_valid,
    input  logic rms_scratch_read_ready,
    output hardware_types_pkg::local_pair_read_request_t rms_scratch_read_req,
    input  logic rms_scratch_read_rsp_valid,
    output logic rms_scratch_read_rsp_ready,
    input  hardware_types_pkg::local_pair_read_response_t rms_scratch_read_rsp,
    output logic rms_scratch_write_valid,
    input  logic rms_scratch_write_ready,
    output hardware_types_pkg::local_banked_write_t rms_scratch_write,
    output logic refresh_metadata_scratch_req_valid,
    input  logic refresh_metadata_scratch_req_ready,
    output logic refresh_metadata_scratch_write,
    output logic [8:0] refresh_metadata_scratch_address,
    output logic [58:0] refresh_metadata_scratch_write_data,
    output logic [58:0] refresh_metadata_scratch_write_enable,
    input  logic refresh_metadata_scratch_rsp_valid,
    input  logic [58:0] refresh_metadata_scratch_rsp_data,
    output logic refresh_metadata_scratch_aux_write_valid,
    input  logic refresh_metadata_scratch_aux_write_ready,
    output logic [8:0] refresh_metadata_scratch_aux_write_address,
    output logic [19:0] refresh_metadata_scratch_aux_write_data,
    output logic [19:0] refresh_metadata_scratch_aux_write_enable,
    output logic rms_norm_write_valid,
    input  logic rms_norm_write_ready,
    output hardware_types_pkg::rms_norm_write_t rms_norm_write,
    output logic hidden_local_write_valid,
    input  logic hidden_local_write_ready,
    output hardware_types_pkg::hidden_local_write_t hidden_local_write,
    output logic hidden_local_read_valid,
    input  logic hidden_local_read_ready,
    output hardware_types_pkg::hidden_local_read_request_t hidden_local_read_req,
    input  logic hidden_local_read_response_valid,
    input  hardware_types_pkg::local_128b_read_response_t hidden_local_read_rsp,
    output logic attention_context_buffer_write_valid,
    input  logic attention_context_buffer_write_ready,
    output hardware_types_pkg::local_128b_write_t attention_context_write,
    output logic attention_context_read_req_valid,
    input  logic attention_context_read_req_ready,
    output hardware_types_pkg::tagged_local_read_request_t
        attention_context_read_req,
    input  logic attention_context_read_rsp_valid,
    output logic attention_context_read_rsp_ready,
    input  hardware_types_pkg::tagged_local_read_response_t
        attention_context_read_rsp,
    output logic attention_softmax_scratch_read_valid,
    input  logic attention_softmax_scratch_read_ready,
    output hardware_types_pkg::local_pair_read_request_t
        attention_softmax_scratch_read_req,
    input  logic attention_softmax_scratch_rsp_valid,
    output logic attention_softmax_scratch_rsp_ready,
    input  hardware_types_pkg::local_pair_read_response_t
        attention_softmax_scratch_rsp,
    output logic attention_softmax_scratch_write_valid,
    input  logic attention_softmax_scratch_write_ready,
    output hardware_types_pkg::local_banked_write_t
        attention_softmax_scratch_write,
    output logic qkv_head_read_req_valid,
    input  logic qkv_head_read_req_ready,
    output hardware_types_pkg::qkv_head_read_request_t qkv_head_read_req,
    input  logic qkv_head_read_rsp_valid,
    output logic qkv_head_read_rsp_ready,
    input  hardware_types_pkg::qkv_head_read_response_t qkv_head_read_rsp,
    output logic rope_source_read_req_valid,
    input  logic rope_source_read_req_ready,
    output hardware_types_pkg::rope_source_read_request_t rope_source_read_req,
    input  logic rope_source_read_rsp_valid,
    output logic rope_source_read_rsp_ready,
    input  hardware_types_pkg::rope_source_read_response_t rope_source_read_rsp,
    output logic rope_cos_read_req_valid,
    input  logic rope_cos_read_req_ready,
    output hardware_types_pkg::rope_constant_read_request_t
        rope_constant_read_req,
    input  logic rope_cos_read_rsp_valid,
    output logic rope_cos_read_rsp_ready,
    input  hardware_types_pkg::rope_constant_read_response_t rope_cos_read_rsp,
    output logic rope_sin_read_req_valid,
    input  logic rope_sin_read_req_ready,
    input  logic rope_sin_read_rsp_valid,
    output logic rope_sin_read_rsp_ready,
    input  hardware_types_pkg::rope_constant_read_response_t rope_sin_read_rsp,
    output logic rope_destination_write_valid,
    input  logic rope_destination_write_ready,
    output hardware_types_pkg::rope_destination_write_t rope_destination_write,
    output logic qkv_head_stage_write_valid,
    input  logic qkv_head_stage_write_ready,
    output hardware_types_pkg::qkv_head_stage_write_t qkv_head_stage_write,
    output logic qkv_head_tile_memory_req_valid,
    input  logic qkv_head_tile_memory_req_ready,
    output hardware_types_pkg::qkv_head_tile_read_request_t
        qkv_head_tile_read_req,
    input  logic qkv_head_tile_memory_rsp_valid,
    input  hardware_types_pkg::qkv_head_tile_read_response_t
        qkv_head_tile_read_rsp,
    output logic qkv_constant_stage_valid,
    input  logic qkv_constant_stage_ready,
    output hardware_types_pkg::qkv_constant_write_t qkv_constant_write,
    output logic qkv_q_local_write_valid,
    input  logic qkv_q_local_write_ready,
    output hardware_types_pkg::qkv_q_write_t qkv_q_write,
    output logic attention_panel_write_valid,
    input  logic attention_panel_write_ready,
    output hardware_types_pkg::attention_panel_write_t attention_panel_write,
    output logic attention_scale_write_valid,
    input  logic attention_scale_write_ready,
    output hardware_types_pkg::attention_panel_scale_write_t
        attention_panel_scale_write,
    output logic attention_operand_req_valid,
    input  logic attention_operand_req_ready,
    output hardware_types_pkg::attention_operand_request_t attention_operand_req,
    input  logic attention_operand_rsp_valid,
    input  hardware_types_pkg::attention_operand_response_t
        attention_operand_rsp,
    output logic attention_operand_abort_request,
    input  logic attention_operand_abort_ack,
    output logic attention_score_write_valid,
    input  logic attention_score_write_ready,
    output hardware_types_pkg::attention_score_write_t attention_score_write,
    output logic attention_softmax_score_req_valid,
    input  logic attention_softmax_score_req_ready,
    output logic attention_softmax_score_slot_ready,
    output hardware_types_pkg::attention_score_read_request_t
        attention_score_read_req,
    input  logic attention_softmax_score_memory_rsp_valid,
    input  hardware_types_pkg::attention_score_read_response_t
        attention_score_read_rsp,
    output logic attention_probability_write_valid,
    input  logic attention_probability_write_ready,
    output hardware_types_pkg::attention_probability_write_t
        attention_probability_write,
    output logic attention_probability_quantized_write_valid,
    input  logic attention_probability_quantized_write_ready,
    output hardware_types_pkg::attention_probability_quantized_write_t
        attention_probability_quantized_write,
    output logic attention_probability_scale_write_valid,
    input  logic attention_probability_scale_write_ready,
    output hardware_types_pkg::attention_probability_scale_write_t
        attention_probability_scale_write,
    output logic matmul_source_req_valid,
    input  logic matmul_local_source_req_ready,
    output hardware_types_pkg::matmul_source_request_t matmul_source_req,
    input  logic matmul_local_source_rsp_valid,
    output logic matmul_source_rsp_ready,
    input  hardware_types_pkg::matmul_source_response_t matmul_source_rsp,
    output logic matmul_activation_write_valid,
    input  logic matmul_activation_write_ready,
    output hardware_types_pkg::matmul_activation_write_t matmul_activation_write,
    output logic matmul_read_bundle_valid,
    input  logic matmul_read_bundle_ready,
    output hardware_types_pkg::matmul_read_request_t matmul_read_req,
    input  logic matmul_read_response_valid,
    input  hardware_types_pkg::matmul_read_response_t matmul_read_rsp,
    output logic matmul_panel_write_valid,
    input  logic matmul_panel_write_ready,
    output hardware_types_pkg::matmul_panel_write_t matmul_panel_write,
    output logic matmul_combine_write_valid,
    input  logic matmul_combine_write_ready,
    output hardware_types_pkg::local_128b_write_t matmul_combine_write,
    output logic matmul_combine_read_valid,
    input  logic matmul_combine_read_ready,
    output hardware_types_pkg::local_128b_read_request_t matmul_combine_read_req,
    input  logic matmul_combine_read_response_valid,
    input  hardware_types_pkg::local_128b_read_response_t matmul_combine_read_rsp,
    output logic matmul_local_output_write_valid,
    input  logic matmul_local_output_write_ready,
    output hardware_types_pkg::matmul_output_write_t matmul_local_output_write,
    output logic elementwise_product_req_valid,
    input  logic elementwise_product_req_ready,
    output hardware_types_pkg::elementwise_product_request_t
        elementwise_product_req,
    input  logic elementwise_product_rsp_valid,
    output logic elementwise_product_rsp_ready,
    input  hardware_types_pkg::elementwise_product_response_t
        elementwise_product_rsp,
    output logic matmul_residual_local_read_issue,
    input  logic matmul_residual_memory_read_ready,
    output hardware_types_pkg::matmul_residual_read_request_t
        matmul_residual_read_req,
    input  logic matmul_residual_memory_read_rsp_valid,
    input  hardware_types_pkg::local_128b_read_response_t
        matmul_residual_read_rsp,
    output logic lm_head_scale_write_valid,
    input  logic lm_head_scale_write_ready,
    output logic [6:0] lm_head_scale_write_address,
    output logic [127:0] lm_head_scale_write_data,
    output logic lm_head_scale_read_valid,
    input  logic lm_head_scale_read_ready,
    output logic [6:0] lm_head_scale_read_address,
    input  logic lm_head_scale_read_rsp_valid,
    input  logic [127:0] lm_head_scale_read_rsp_data,
    output logic candidate_state_read_valid,
    input  logic candidate_state_read_ready,
    output logic [3:0] candidate_state_read_row_group,
    output logic [2:0] candidate_state_read_lane_block,
    output logic candidate_state_read_row_half,
    output logic [1:0] candidate_state_read_word,
    output logic [3:0] candidate_state_read_row_mask,
    output logic [15:0] candidate_state_read_tag,
    input  logic candidate_state_read_rsp_valid,
    output logic candidate_state_read_rsp_ready,
    input  logic [511:0] candidate_state_read_rsp_data,
    input  logic [3:0] candidate_state_read_rsp_row_mask,
    input  logic [15:0] candidate_state_read_rsp_tag,
    output logic candidate_state_write_valid,
    input  logic candidate_state_write_ready,
    output logic [3:0] candidate_state_write_row_group,
    output logic [2:0] candidate_state_write_lane_block,
    output logic candidate_state_write_row_half,
    output logic [1:0] candidate_state_write_word,
    output logic [3:0] candidate_state_write_row_mask,
    output logic [511:0] candidate_state_write_data,
    output logic post_table_access_valid,
    input  logic post_table_access_ready,
    output logic post_table_access_enable,
    output logic post_state_write_valid,
    input  logic post_state_write_ready,
    output logic [9:0] post_state_write_word_address,
    output logic [127:0] post_state_write_data,
    output logic post_state_read_valid,
    input  logic post_state_read_ready,
    output logic [9:0] post_state_read_word_address,
    output logic [15:0] post_state_read_tag,
    input  logic post_state_read_rsp_valid,
    input  logic [127:0] post_state_read_rsp_data,
    input  logic [15:0] post_state_read_rsp_tag,
    output logic post_suppressed_write_valid,
    input  logic post_suppressed_write_ready,
    output logic [5:0] post_suppressed_write_address,
    output logic [127:0] post_suppressed_write_data,
    output logic post_suppressed_read_valid,
    input  logic post_suppressed_read_ready,
    output logic [7:0] post_suppressed_read_index,
    input  logic post_suppressed_read_rsp_valid,
    input  logic [16:0] post_suppressed_read_rsp_token_id
);
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_RMSNORM;
    import layer_schedule_pkg::LAYER_OPERATOR_QKV_PREPARATION;
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION;
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_RESIDUAL_SPILL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_RMSNORM;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_GATE_MATMUL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_UP_MATMUL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_HIDDEN_REFILL;
    import local_memory_layout_pkg::local_memory_phase_t;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_IDLE;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_MATMUL;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_QKV_PREPARATION;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_ATTENTION_QK;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_RMSNORM;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_ELEMENTWISE;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_R4;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_R4_TWO_TOKENS;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN;
    import local_memory_layout_pkg::LOCAL_MEMORY_PHASE_FINAL_OUTPUT;
    import local_memory_layout_pkg::LOCAL_PANEL_LAYOUT_NONE;
    import local_memory_layout_pkg::LOCAL_PANEL_LAYOUT_QKV;
    import local_memory_layout_pkg::LOCAL_PANEL_LAYOUT_MIXED;
    import local_memory_layout_pkg::LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL;

    logic active_command_valid;
    hardware_types_pkg::operator_command_t active_command;
    localparam int unsigned COMMAND_START_RMS = 0;
    localparam int unsigned COMMAND_START_QKV = 1;
    localparam int unsigned COMMAND_START_ATTENTION = 2;
    localparam int unsigned COMMAND_START_MATMUL = 3;
    localparam int unsigned COMMAND_START_HIDDEN = 4;
    localparam int unsigned COMMAND_START_ELEMENTWISE = 5;
    localparam int unsigned DMA_READ_HIDDEN = 0;
    localparam int unsigned DMA_READ_RMS_GAMMA = 1;
    localparam int unsigned DMA_READ_QKV_METADATA = 2;
    localparam int unsigned DMA_READ_MATMUL = 3;
    localparam int unsigned DMA_READ_MATMUL_AUX = 4;
    localparam int unsigned DMA_READ_ATTENTION = 5;
    localparam int unsigned DMA_READ_ELEMENTWISE = 6;
    localparam int unsigned DMA_WRITE_HIDDEN = 0;
    localparam int unsigned DMA_WRITE_QKV_CACHE = 1;
    localparam int unsigned DMA_WRITE_ATTENTION = 2;
    localparam int unsigned DMA_WRITE_MATMUL = 3;

    assign dma_read_request[DMA_READ_HIDDEN].wide_data = 1'b0;
    assign dma_read_request[DMA_READ_RMS_GAMMA].tag = 8'h04;
    assign dma_read_request[DMA_READ_RMS_GAMMA].wide_data = 1'b0;
    assign dma_read_request[DMA_READ_MATMUL].tag = 8'h80;
    assign dma_read_request[DMA_READ_MATMUL].wide_data = 1'b1;
    assign dma_read_request[DMA_READ_ATTENTION].wide_data = 1'b1;
    assign dma_read_request[DMA_READ_ELEMENTWISE].wide_data = 1'b0;
    logic [5:0] pending_start_select;
    logic [5:0] pending_start_ready;
    logic [5:0] command_child_start_fire;
    logic pending_start_enable;
    logic pending_layout_matches;
    local_memory_layout_pkg::layout_requirement_t pending_initial_layout;
    local_memory_layout_pkg::layout_requirement_t accepted_layout;
    logic [6:0] ffn_pair_activation_group_count;
    logic qkvo_query_requires_expanded_layout;
    logic qkvo_kv_requires_expanded_layout;
    logic qkvo_kv_expanded_matmul;
    logic qkvo_query_group_precompute;
    logic ffn_pair_requires_expanded_layout;
    logic [5:0] operator_command_layer;
    logic [3:0] execution_layer_operator_index;
    logic active_command_rms_qkv_prefetch_q;
    logic operator_command_done_error;
    logic [15:0] operator_command_done_error_id;
    logic command_child_done_valid;
    logic command_child_done_error;
    logic [15:0] command_child_done_error_id;
    logic command_done_can_complete;
    logic attention_q_head_active;
    assign operator_command_layer = active_command_valid ?
        active_command.layer : pending_command.layer;
    assign execution_layer_operator_index = active_command_valid ?
        active_command.layer_operator_index : pending_command.layer_operator_index;
    assign active_command_rms_qkv_prefetch_q = active_command_valid ?
        active_command.rms_qkv_prefetch_q : pending_command.rms_qkv_prefetch_q;
    assign execution_completion.operator_tag = {
        active_command.layer, active_command.layer_operator_index, 6'd0};
    assign execution_completion.error = operator_command_done_error;
    assign execution_completion.error_id =
        operator_command_done_error_id;
    always_comb begin
        ffn_pair_activation_group_count =
            {1'b0, matmul_row_config.compute_group_count};
        if (ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.batch_count >= 3'd2)
            ffn_pair_activation_group_count +=
                {1'b0, ffn_batch_pair_config.second_row_config.compute_group_count};
        if (ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.batch_count >= 3'd3)
            ffn_pair_activation_group_count +=
                {1'b0, ffn_batch_pair_config.third_row_config.compute_group_count};
        if (ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.batch_count >= 3'd4)
            ffn_pair_activation_group_count +=
                {1'b0, ffn_batch_pair_config.fourth_row_config.compute_group_count};
        if (ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.batch_count >= 3'd5)
            ffn_pair_activation_group_count +=
                {1'b0, ffn_batch_pair_config.fifth_row_config.compute_group_count};
        if (ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.batch_count >= 3'd6)
            ffn_pair_activation_group_count +=
                {1'b0, ffn_batch_pair_config.sixth_row_config.compute_group_count};
    end
    assign ffn_pair_requires_expanded_layout =
        ffn_batch_pair_config.enable &&
        ffn_batch_pair_config.compact_a4_layout &&
        ffn_pair_activation_group_count > 7'd12;
    assign qkvo_query_requires_expanded_layout =
        attention_q_head_active && ffn_batch_pair_config.enable &&
        ffn_batch_pair_config.qkvo_group &&
        ffn_batch_pair_config.compact_a4_layout &&
        ffn_pair_activation_group_count > 7'd4;
    // The compact K/V-pair image holds six 16-slot groups (192 KiB).
    // Larger groups use the expanded image only during Matmul. K/V head
    // staging subsequently reuses its banks, so each head Matmul reloads activation.
    assign qkvo_kv_requires_expanded_layout =
        kv_pair_enable && ffn_batch_pair_config.enable &&
        ffn_batch_pair_config.qkvo_group &&
        ffn_batch_pair_config.compact_a4_layout &&
        ffn_pair_activation_group_count > 7'd6;
    assign qkvo_kv_expanded_matmul = qkvo_kv_requires_expanded_layout &&
        qkv_memory_stage == 2'd2; // qkv_preparation_controller MEMORY_STAGE_MATMUL
    assign qkvo_query_group_precompute =
        ffn_batch_pair_config.enable && ffn_batch_pair_config.qkvo_group &&
        ffn_batch_pair_config.compact_a4_layout &&
        kv_batch_group_config.active_batch_index == 3'd0;

    always_ff @(posedge clk) begin
        if (rst) begin
            active_command_valid <= 1'b0;
            active_command <= '0;
        end else begin
            if (operator_command_valid && operator_command_ready) begin
                active_command_valid <= 1'b1;
                active_command <= pending_command;
            end
            if (active_command_valid && operator_command_done_valid &&
                operator_command_done_ready)
                active_command_valid <= 1'b0;
        end
    end

`ifndef SYNTHESIS
    hardware_types_pkg::operator_completion_t stalled_operator_completion;
`endif

    // Simulation wrappers observe accepted events through these internal signals.
    logic [31:0] qkv_accepted_matmul_count;
    logic [31:0] qkv_accepted_rope_count;
    logic [31:0] qkv_accepted_head_read_count;
    logic [31:0] qkv_accepted_quant_count;
    logic [31:0] qkv_accepted_quantized_commit_count;
    logic [31:0] qkv_completed_cache_write_count;
    logic [15:0] qkv_outstanding_cache_writes;
    logic [63:0] matmul_accepted_issue_count;
    logic [63:0] matmul_accepted_result_count;
    logic [63:0] matmul_base_request_count;
    logic [63:0] matmul_enhancement_request_count;
    logic [63:0] matmul_scale_request_count;
    logic [63:0] matmul_base_request_bytes;
    logic [63:0] matmul_enhancement_request_bytes;
    logic [63:0] matmul_scale_request_bytes;
    logic rms_trace_sample_valid;
    logic [5:0] rms_trace_sample_row;
    logic [15:0] rms_trace_sample_element;
    logic [1023:0] rms_trace_sample_data;
    logic [127:0] rms_trace_sample_byte_enable;
    logic elementwise_trace_sample_valid;
    logic elementwise_trace_sample_pass;
    logic [10:0] elementwise_trace_sample_stripe;
    logic [5:0] elementwise_trace_sample_row_base;
    logic [63:0] elementwise_trace_sample_lane_mask;
    logic [1023:0] elementwise_trace_sample_silu;
    logic [1023:0] elementwise_trace_sample_product;

    wire [5:0] row_segment_mode = matmul_row_config.segment_mode;
    wire [17:0] row_segment_base = matmul_row_config.segment_row_base;
    wire [17:0] row_segment_count = matmul_row_config.segment_row_count;
    wire [287:0] row_physical_to_local =
        hidden_row_config.physical_to_local;
    wire [527:0] row_token_position =
        attention_row_config.token_position;
    wire [47:0] row_matmul_valid = matmul_row_config.row_enable;

    logic matmul_local_layout;

    localparam logic [2:0] EXEC_MEMORY_CONFIGURATION = 3'd1;
    localparam logic [2:0] EXEC_MEMORY_METADATA = 3'd2;
    localparam logic [2:0] EXEC_MEMORY_HIDDEN_READ = 3'd3;
    localparam logic [2:0] EXEC_MEMORY_HIDDEN_WRITE = 3'd4;
    localparam logic [2:0] EXEC_MEMORY_FORWARD_POSTPROCESS = 3'd5;

    logic hidden_abort_request;
    logic hidden_abort_ack;
    logic hidden_start_valid;
    logic hidden_start_ready;
    logic [1:0] hidden_start_operation;
    logic hidden_busy;
    logic hidden_done_valid;
    logic hidden_done_ready;
    logic hidden_done_for_ffn;
    logic hidden_error;
    logic [3:0] hidden_error_id;
    logic execution_hidden_source_selected;
    logic post_hidden_source_selected;
    logic ffn_hidden_source_selected;

    logic rms_abort_request;
    logic rms_abort_ack;
    logic rms_start_valid;
    logic block_rms_start_valid;
    logic block_rms_gamma_bypass;
    logic rms_start_ready;

    logic rms_tile_read_output_pass;

    logic [15:0] rms_norm_write_tag;

    logic rms_done_valid;
    logic rms_done_ready;
    logic block_rms_done_ready;
    logic rms_error;
    logic [3:0] rms_error_id;
    logic rms_active;

    logic rms_arithmetic_req_valid;
    logic rms_arithmetic_req_ready;
    logic rms_arithmetic_rsp_valid;
    logic rms_arithmetic_rsp_ready;
    logic rms_reduction_req_valid;
    logic rms_reduction_req_ready;
    logic rms_reduction_rsp_valid;
    logic rms_reduction_rsp_ready;
    logic engine_rms_tile_read_req_valid, engine_rms_tile_read_req_ready;
    hardware_types_pkg::rms_tile_read_request_t engine_rms_tile_read_req;
    logic engine_rms_tile_read_rsp_valid, engine_rms_tile_read_rsp_ready;
    hardware_types_pkg::rms_tile_read_response_t engine_rms_tile_read_rsp;

    logic qkv_phase_active;

    logic [15:0] qkv_v_static_scale;
    logic qkv_start_valid;
    logic qkv_start_ready;
    logic qkv_pair_active, qkv_second_batch;
    logic [2:0] qkv_batch_index;
    logic [5:0] qkv_active_token_count;
    logic qkv_done_valid;
    logic qkv_done_ready;
    logic qkv_error;
    logic [7:0] qkv_error_id;
    logic [1:0] qkv_compute_stage;

    logic qkv_abort_request;
    logic qkv_abort_ack;

    logic qkv_matmul_start_valid;
    logic qkv_matmul_start_ready;
    logic qkv_matmul_start_enable;
    logic [1:0] qkv_matmul_qkv_select;
    logic [5:0] qkv_matmul_head;
    logic qkv_matmul_preprocess_input;
    logic qkv_matmul_use_resident_a8;
    logic qkv_group_activation_reuse;
    logic qkv_matmul_qkv_w8_layout;
    logic [63:0] qkv_matmul_group_staging_base;
    logic qkv_matmul_prefetch_valid;
    logic qkv_matmul_prefetch_ready;
    logic [1:0] qkv_matmul_prefetch_qkv_select;
    logic [5:0] qkv_matmul_prefetch_head;
    logic qkv_matmul_done_valid;
    logic qkv_matmul_error;
    logic qkv_matmul_abort_request;
    logic qkv_matmul_abort_ack;
    hardware_types_pkg::ffn_batch_pair_config_t matmul_ffn_batch_pair_config;

    logic qkv_rope_start_valid;
    logic qkv_rope_start_ready;
    logic qkv_rope_source_is_k;
    logic [5:0] qkv_rope_head;
    logic qkv_rope_done_pulse;
    logic qkv_rope_error;
    logic qkv_rope_abort_request;
    logic qkv_rope_abort_ack;

    logic qkv_head_staging_layout;
    logic qkv_head_staging_layout_ready;

    logic qkv_max_req_valid, qkv_max_req_ready;
    logic qkv_max_rsp_valid, qkv_max_rsp_ready;

    logic qkv_quant_scale_req_valid;
    logic qkv_quant_scale_req_ready;
    logic qkv_quant_scale_rsp_valid;
    logic qkv_quant_scale_rsp_ready;
    logic qkv_quant_req_valid;
    logic qkv_quant_req_ready;
    logic qkv_quant_rsp_valid;
    logic qkv_quant_rsp_ready;
    logic qkv_quant_abort_request;
    logic qkv_quant_abort_ack;

    logic qkv_quantized_commit_valid;
    logic qkv_quantized_commit_ready;
    logic [1:0] qkv_quantized_commit_qkv_select;
    logic [5:0] qkv_quantized_commit_head;
    logic [5:0] qkv_quantized_commit_physical_row;
    logic [10:0] qkv_quantized_commit_token_position;
    logic qkv_quantized_commit_half;
    logic [511:0] qkv_quantized_commit_values;
    logic [63:0] qkv_quantized_commit_lane_mask;
    logic [15:0] qkv_quantized_commit_scale;
    logic [15:0] qkv_quantized_commit_tag;
    logic qkv_quantized_commit_requires_response;
    logic qkv_cache_write_complete_valid;
    logic qkv_cache_write_complete_ready;
    logic qkv_cache_write_complete_error;
    logic qkv_cache_mapping_commit_valid;
    logic qkv_cache_mapping_commit_ready;
    logic qkv_cache_mapping_commit_error;

    logic cache_start_valid;
    logic cache_start_ready;
    logic cache_config_valid;
    logic cache_config_release;
    logic qkv_cache_config_release;
    logic cache_config_done_pulse;
    logic cache_error;
    logic [3:0] cache_error_id;
    logic cache_lookup_abort_ack;
    logic cache_current_write_valid;
    logic cache_current_write_ready;
    logic cache_current_write_accepted;
    logic [63:0] cache_current_k_write_address;
    logic [63:0] cache_current_v_write_address;
    logic [63:0] cache_current_k_scale_write_address;

    logic attention_abort_request;
    logic attention_abort_ack;
    logic attention_start_valid;
    logic attention_start_ready;
    logic attention_busy;
    logic attention_done_valid;
    logic attention_done_ready;
    logic attention_error;
    logic [7:0] attention_error_id;
    logic [47:0] attention_context_row_max_valid;
    logic [48*16-1:0] attention_context_row_max_values;
    logic attention_q_head_start_valid;
    logic attention_q_head_start_ready;
    logic [5:0] attention_q_head_start_head;
    logic attention_q_head_start_slot;
    logic attention_active_second_batch;
    logic [1:0] attention_active_batch_index;
    logic attention_pair_active;
    logic [7:0] attention_pair_score_words;
    logic attention_q_head_postprocess_enable;
    logic attention_q_head_precompute_only;
    logic attention_q_head_use_staged;
    logic attention_q_head_matmul_done;
    logic attention_q_head_done_valid;
    logic attention_q_head_done_ready;
    logic attention_q_head_error;
    logic [7:0] attention_q_head_error_id;
    logic attention_q_head_abort_request;
    logic attention_q_head_abort_ack;
    logic attention_q_head_slot;

    logic attention_cache_lookup_req_valid;
    logic attention_cache_lookup_req_ready;
    logic [5:0] attention_cache_lookup_req_head;
    logic [10:0] attention_cache_lookup_req_key_group;
    logic [4:0] attention_cache_lookup_req_chunk;
    logic [15:0] attention_cache_lookup_req_tag;
    logic attention_cache_lookup_rsp_valid;
    logic attention_cache_lookup_rsp_ready;
    logic [10:0] attention_cache_lookup_rsp_key_group;
    logic [7:0] attention_cache_lookup_rsp_current_mask;
    logic [7:0] attention_cache_lookup_rsp_retained_mask;
    logic [63:0] attention_cache_lookup_rsp_current_k_address;
    logic [63:0] attention_cache_lookup_rsp_current_v_address;
    logic [63:0] attention_cache_lookup_rsp_current_k_scale_address;
    logic [63:0] attention_cache_lookup_rsp_retained_k_address;
    logic [63:0] attention_cache_lookup_rsp_retained_v_address;
    logic [63:0] attention_cache_lookup_rsp_retained_k_scale_address;
    logic [15:0] attention_cache_lookup_rsp_tag;
    logic attention_cache_lookup_abort_request;

    logic [15:0] attention_panel_write_tag;

    logic [15:0] attention_scale_write_tag;

    logic attention_pe_req_valid;
    logic attention_pe_req_ready;
    logic attention_pe_accum_result_valid;
    logic attention_pe_accum_result_ready;
    logic attention_pe_abort_request;
    logic attention_pe_abort_ack;

    logic attention_rescale_req_valid;
    logic attention_rescale_req_ready;
    logic attention_rescale_rsp_valid;
    logic attention_rescale_rsp_ready;
    logic attention_rescale_abort_request;
    logic attention_rescale_abort_ack;

    logic [5:0] attention_score_write_head;

    logic [15:0] attention_score_write_tag;

    logic [15:0] attention_context_buffer_write_tag;

    logic [1:0] attention_softmax_score_pass;
    logic [23:0] attention_softmax_score_group_base;
    logic [5:0] attention_softmax_score_head;

    logic [15:0] attention_softmax_score_tag;

    logic attention_probability_write_probability;
    logic [23:0] attention_probability_write_group_base;
    logic [5:0] attention_probability_write_head;

    logic [15:0] attention_probability_write_tag;

    logic [5:0] attention_probability_quantized_write_head;

    logic [15:0] attention_probability_quantized_write_tag;

    logic attention_softmax_bf16_req_valid;
    logic attention_softmax_bf16_req_ready;
    logic attention_softmax_bf16_rsp_valid;
    logic attention_softmax_bf16_rsp_ready;
    logic attention_softmax_max_req_valid, attention_softmax_max_req_ready;
    logic attention_softmax_max_rsp_valid, attention_softmax_max_rsp_ready;
    logic attention_softmax_reduction_req_valid;
    logic attention_softmax_reduction_req_ready;
    logic attention_softmax_reduction_rsp_valid;
    logic attention_softmax_reduction_rsp_ready;
    logic attention_softmax_exp_req_valid;
    logic attention_softmax_exp_req_ready;
    logic attention_softmax_exp_rsp_valid;
    logic attention_softmax_exp_rsp_ready;
    logic attention_softmax_reciprocal_req_valid;
    logic attention_softmax_reciprocal_req_ready;
    logic attention_softmax_reciprocal_rsp_valid;
    logic attention_softmax_reciprocal_rsp_ready;
    hardware_types_pkg::softmax_exp_request_t attention_softmax_exp_req;
    hardware_types_pkg::softmax_exp_response_t attention_softmax_exp_rsp;
    hardware_types_pkg::softmax_reciprocal_request_t
        attention_softmax_reciprocal_req;
    hardware_types_pkg::softmax_reciprocal_response_t
        attention_softmax_reciprocal_rsp;
    logic attention_softmax_quant_scale_req_valid;
    logic attention_softmax_quant_scale_req_ready;
    logic attention_softmax_quant_scale_rsp_valid;
    logic attention_softmax_quant_scale_rsp_ready;
    logic attention_softmax_quant_values_req_valid;
    logic attention_softmax_quant_values_req_ready;
    logic attention_softmax_quant_values_rsp_valid;
    logic attention_softmax_quant_values_rsp_ready;

    logic [63:0] attention_completed_head_count;
    logic [63:0] attention_qk_count;
    logic [63:0] attention_pv_count;
    local_memory_phase_t attention_local_phase;

    logic matmul_abort_request;
    logic matmul_abort_ack;
    logic matmul_start_valid;
    logic matmul_start_ready;
    logic block_matmul_start_valid;
    hardware_types_pkg::matmul_command_t matmul_command;
    logic [15:0] matmul_start_operator_tag;
    logic matmul_prefetch_valid;
    logic matmul_prefetch_ready;
    logic matmul_prefetch_empty;
    logic matmul_prefetch_cancel;
    hardware_types_pkg::matmul_command_t matmul_prefetch_command;
    logic [1:0] matmul_prefetch_qkv_select;
    logic [5:0] matmul_prefetch_qkv_head;
    logic [15:0] matmul_prefetch_operator_tag;
    localparam int unsigned MATMUL_PREFETCH_RMS_QKV = 0;
    localparam int unsigned MATMUL_PREFETCH_QKV_NEXT = 1;
    localparam int unsigned MATMUL_PREFETCH_FFN_GATE = 2;
    localparam int unsigned MATMUL_PREFETCH_FFN_UP = 3;
    localparam int unsigned MATMUL_PREFETCH_FFN_DOWN = 4;
    logic [4:0] matmul_prefetch_source_valid;

    logic [1:0] matmul_dma_req_plane;
    logic matmul_dma_req_panel;
    logic matmul_dma_req_last_for_plane;

    logic matmul_dma_reader_request_valid;
    logic matmul_dma_reader_request_ready;
    logic [63:0] matmul_dma_reader_request_address;
    logic [31:0] matmul_dma_reader_request_bytes;
    logic [7:0] matmul_dma_reader_request_tag;
    logic matmul_dma_reader_read_valid;
    logic matmul_dma_reader_read_ready;
    logic matmul_dma_reader_done;
    logic matmul_dma_reader_error;
    logic matmul_dma_reader_abort_ack;

    logic matmul_done_valid;
    logic matmul_done_ready;
    logic matmul_error;

    logic matmul_accum_result_valid;
    logic [2047:0] matmul_accum_result_accumulators;
    logic [63:0] matmul_accum_result_mask;
    logic [1:0] matmul_accum_result_mode;
    logic matmul_accum_result_last_k_step;
    logic [15:0] matmul_accum_result_tag;
    logic matmul_output_controller_write_spill;
    logic [5:0] matmul_output_controller_write_physical_row;

    logic matmul_bf16_abort_request;
    logic matmul_bf16_req_valid;
    logic matmul_bf16_req_ready;
    logic matmul_bf16_rsp_valid;
    logic matmul_bf16_rsp_ready;

    logic [15:0] matmul_activation_write_tag;
    logic matmul_activation_writer_abort_request;
    logic matmul_activation_writer_cfg_valid;
    logic matmul_activation_writer_cfg_ready;
    logic matmul_activation_writer_quantized_valid;
    logic matmul_activation_writer_quantized_ready;
    logic matmul_activation_writer_scale_valid;
    logic matmul_activation_writer_scale_ready;
    logic shared_activation_writer_done_pulse;
    logic shared_activation_writer_error;
    logic shared_activation_scale_valid;
    logic shared_activation_scale_ready;
    logic block_activation_scale_ready;
    logic [5:0] shared_activation_scale_row_base;
    logic [7:0] shared_activation_scale_row_mask;
    logic [127:0] shared_activation_scale_values;

    logic elementwise_abort_request;
    logic elementwise_abort_ack;
    logic elementwise_start_valid;
    logic elementwise_start_ready;
    logic elementwise_busy;
    logic elementwise_done_valid;
    logic elementwise_done_ready;
    logic elementwise_read_phase_complete;
    logic elementwise_down_prefetch_ready;
    logic elementwise_error;
    logic [3:0] elementwise_error_id;
    hardware_types_pkg::activation_layout_t elementwise_completed_layout;

    logic elementwise_bf16_req_valid;
    logic elementwise_bf16_req_ready;
    logic elementwise_bf16_rsp_valid;
    logic elementwise_bf16_rsp_ready;
    logic elementwise_bf16_abort_ack;
    logic elementwise_max_req_valid;
    logic elementwise_max_req_ready;
    logic elementwise_max_rsp_valid;
    logic elementwise_max_rsp_ready;
    hardware_types_pkg::reduction_request_t reduction_client_req [0:1];
    hardware_types_pkg::reduction_response_t reduction_client_rsp [0:1];
    hardware_types_pkg::maximum_request_t maximum_client_req [0:3];
    hardware_types_pkg::maximum_response_t maximum_client_rsp [0:3];
    logic elementwise_quant_scale_req_valid;
    logic elementwise_quant_scale_req_ready;
    logic [7:0] elementwise_quant_scale_req_a4_row_mask;
    logic elementwise_quant_scale_rsp_valid;
    logic elementwise_quant_scale_rsp_ready;
    logic elementwise_quant_values_req_valid;
    logic elementwise_quant_values_req_ready;
    logic elementwise_quant_values_rsp_valid;
    logic elementwise_quant_values_rsp_ready;
    logic elementwise_quant_abort_ack;
    logic elementwise_writer_abort_request;
    logic elementwise_writer_cfg_valid;
    logic elementwise_writer_cfg_ready;
    logic elementwise_writer_scale_valid;
    logic elementwise_writer_scale_ready;
    logic elementwise_writer_quantized_valid;
    logic elementwise_writer_quantized_ready;
    logic elementwise_writer_done_pulse;
    logic elementwise_writer_error;
    logic ffn_hidden_start_valid;
    logic qkv_residual_spill_start_valid;
    logic qkv_residual_spill_complete;
    logic rms_after_spill_pending;
    logic rms_spill_start_valid;
    logic qkv_residual_spill_error;
    logic [3:0] qkv_residual_spill_error_id;
    logic ffn_hidden_refill;

    logic matmul_tile_abort_request;
    logic matmul_tile_abort_ack;
    logic matmul_tile_req_valid;
    logic matmul_tile_req_ready;
    logic matmul_tile_accum_result_valid;
    logic matmul_tile_accum_result_ready;

    logic matmul_rescale_abort_request;
    logic matmul_rescale_abort_ack;
    logic matmul_rescale_claim_valid;
    logic matmul_rescale_claim_ready;
    logic matmul_rescale_req_valid;
    logic matmul_rescale_req_ready;
    logic matmul_rescale_rsp_valid;
    logic matmul_rescale_rsp_ready;

    logic matmul_max_req_valid, matmul_max_req_ready;
    logic matmul_max_rsp_valid, matmul_max_rsp_ready;
    logic matmul_quant_scale_req_valid, matmul_quant_scale_req_ready;
    logic matmul_quant_scale_rsp_valid, matmul_quant_scale_rsp_ready;
    logic matmul_quant_values_req_valid, matmul_quant_values_req_ready;
    logic matmul_quant_values_rsp_valid, matmul_quant_values_rsp_ready;

    hardware_types_pkg::quant_scale_request_t quant_scale_client_req [0:3];
    hardware_types_pkg::quant_scale_response_t quant_scale_client_rsp [0:3];
    hardware_types_pkg::quant_values_request_t quant_values_client_req [0:3];
    hardware_types_pkg::quant_values_response_t quant_values_client_rsp [0:3];
    localparam logic [1:0] QKV_COMPUTE_STAGE_MATMUL = 2'd1;
    localparam logic [1:0] QKV_COMPUTE_STAGE_ROPE = 2'd2;
    localparam logic [1:0] QKV_COMPUTE_STAGE_QUANT = 2'd3;
    hardware_types_pkg::bf16_request_t bf16_client_req [0:4];
    hardware_types_pkg::bf16_response_t bf16_client_rsp [0:4];
    hardware_types_pkg::pe_request_t pe_client_req [0:1];
    hardware_types_pkg::pe_response_t pe_client_rsp [0:1];
    hardware_types_pkg::rescale_request_t rescale_client_req [0:1];
    hardware_types_pkg::rescale_response_t rescale_client_rsp [0:1];
    hardware_types_pkg::activation_writer_config_t writer_client_cfg [0:1];
    hardware_types_pkg::activation_writer_scale_t writer_client_scale [0:1];
    hardware_types_pkg::activation_writer_values_t writer_client_values [0:1];
    logic unused_attention_rescale_claim_ready;

    assign pe_client_req[1].mixed_phase = 1'b0;
    assign pe_client_req[1].mixed_phase_first = 1'b0;
    assign pe_client_req[1].mixed_a8_rows = 8'd0;

    logic post_active;
    logic refresh_requested, refresh_active, refresh_start_ready, refresh_prepared;
    logic refresh_source_a_valid;
    logic [31:0] refresh_source_a_mask, refresh_source_a_capture_index;
    logic [10:0] refresh_source_a_block_start;
    logic [31:0] refresh_relation_layer_mask;
    assign execution_refresh_relation_layer_mask = refresh_requested ? refresh_relation_layer_mask : 32'd0;
    hardware_types_pkg::attention_probability_config_t refresh_probability_config, attention_probability_config;
    always_comb begin
        attention_probability_config = refresh_probability_config;
        attention_probability_config.enable = refresh_probability_config.enable && refresh_requested && refresh_prepared;
    end
    logic refresh_done, refresh_error;
    logic [7:0] refresh_error_id;
    logic post_done, post_error;
    logic [7:0] post_error_id;
    logic refresh_read_request_valid, refresh_read_data_ready;
    hardware_types_pkg::dma_read_request_t refresh_read_request;
    logic refresh_write_request_valid, refresh_write_data_valid;
    hardware_types_pkg::dma_write_request_t refresh_write_request;
    hardware_types_pkg::dma_write_beat_t refresh_write_data;
    logic [31:0] refresh_metadata_bytes;
    logic [3:0] refresh_metadata_rounds;
    logic refresh_selected_valid, refresh_selected_last;
    logic [10:0] refresh_selected_position;
    logic [1:0] row_packer_abort, row_packer_start_valid, row_packer_row_valid;
    logic [1:0] row_packer_beat_ready, row_packer_done_ready;
    hardware_types_pkg::token_metadata_config_t row_packer_config [0:1];
    hardware_types_pkg::token_metadata_row_t row_packer_row [0:1];
    logic row_packer_busy, row_packer_client_post, row_packer_selected_post;
    logic row_packer_start_ready, row_packer_row_ready, row_packer_abort_ack;
    logic row_packer_beat_valid, row_packer_beat_last, row_packer_done_valid, row_packer_error;
    logic [127:0] row_packer_beat_data;
    logic [7:0] row_packer_error_id;
    logic [5:0] row_packer_compute_groups, row_packer_semantic_groups;
    hardware_types_pkg::token_metadata_config_t selected_row_packer_config;
    hardware_types_pkg::token_metadata_row_t selected_row_packer_row;
    logic post_controller_start_valid, post_controller_start_ready;
    logic post_meta_request_valid, post_meta_request_ready;
    hardware_types_pkg::dma_read_request_t post_meta_request;
    logic post_meta_data_valid, post_meta_data_ready;
    hardware_types_pkg::dma_read_beat_t post_meta_data;
    hardware_types_pkg::dma_completion_t post_meta_completion;
    logic post_lm_request_valid, post_lm_request_ready;
    hardware_types_pkg::dma_read_request_t post_lm_request;
    logic post_lm_data_valid, post_lm_data_ready;
    hardware_types_pkg::dma_read_beat_t post_lm_data;
    hardware_types_pkg::dma_completion_t post_lm_completion;
    logic post_write_request_valid, post_write_request_ready;
    hardware_types_pkg::dma_write_request_t post_write_request;
    logic post_write_data_valid, post_write_data_ready;
    hardware_types_pkg::dma_write_beat_t post_write_data;
    hardware_types_pkg::dma_completion_t post_write_completion;
    logic post_hidden_start_valid, post_hidden_start_ready;
    logic [63:0] post_hidden_start_address, post_hidden_start_limit;
    logic [31:0] post_hidden_start_bytes;
    logic [5:0] post_hidden_start_rows;
    logic [527:0] post_hidden_start_ddr_row_index;
    logic [55:0] post_hidden_start_source_round;
    logic [47:0] post_hidden_start_source_row;
    logic post_hidden_done_valid, post_hidden_done_ready, post_hidden_error;
    logic post_rms_start_valid, post_rms_start_ready;
    logic [5:0] post_rms_start_rows;
    logic [5:0] post_rms_active_rows;
    logic [15:0] post_rms_start_elements;
    logic [15:0] post_rms_start_epsilon;
    logic post_rms_start_gamma_bypass;
    logic post_rms_done_valid, post_rms_done_ready, post_rms_error;
    logic post_rms_source_req_valid, post_rms_source_req_ready;
    hardware_types_pkg::rms_tile_read_request_t post_rms_source_req;
    logic post_rms_source_rsp_valid, post_rms_source_rsp_ready;
    hardware_types_pkg::rms_tile_read_response_t post_rms_source_rsp;
    logic post_max_req_valid, post_max_req_ready;
    hardware_types_pkg::maximum_request_t post_max_req;
    logic post_max_rsp_valid, post_max_rsp_ready;
    hardware_types_pkg::maximum_response_t post_max_rsp;
    logic post_quant_scale_req_valid, post_quant_scale_req_ready;
    hardware_types_pkg::quant_scale_request_t post_quant_scale_req;
    logic post_quant_scale_rsp_valid, post_quant_scale_rsp_ready;
    hardware_types_pkg::quant_scale_response_t post_quant_scale_rsp;
    logic post_quant_values_req_valid, post_quant_values_req_ready;
    hardware_types_pkg::quant_values_request_t post_quant_values_req;
    logic post_quant_values_rsp_valid, post_quant_values_rsp_ready;
    hardware_types_pkg::quant_values_response_t post_quant_values_rsp;
    logic post_writer_cfg_valid, post_writer_cfg_ready;
    hardware_types_pkg::activation_writer_config_t post_writer_cfg;
    logic post_writer_scale_valid, post_writer_scale_ready;
    hardware_types_pkg::activation_writer_scale_t post_writer_scale;
    logic post_writer_values_valid, post_writer_values_ready;
    hardware_types_pkg::activation_writer_values_t post_writer_values;
    logic post_panel_write_valid, post_panel_write_ready;
    hardware_types_pkg::matmul_panel_write_t post_panel_write;
    logic post_operand_read_valid, post_operand_read_ready;
    hardware_types_pkg::matmul_read_request_t post_operand_read;
    logic post_operand_rsp_valid;
    hardware_types_pkg::matmul_read_response_t post_operand_rsp;
    logic post_pe_abort_request, post_pe_abort_ack;
    logic post_pe_req_valid, post_pe_req_ready;
    hardware_types_pkg::pe_request_t post_pe_req;
    logic post_pe_rsp_valid, post_pe_rsp_ready;
    hardware_types_pkg::pe_response_t post_pe_rsp;
    logic post_rescale_abort_request, post_rescale_abort_ack;
    logic post_rescale_req_valid, post_rescale_req_ready;
    hardware_types_pkg::rescale_request_t post_rescale_req;
    logic post_rescale_rsp_valid, post_rescale_rsp_ready;
    hardware_types_pkg::rescale_response_t post_rescale_rsp;
    logic post_bf16_abort_request, post_bf16_abort_ack;
    logic refresh_bf16_abort_request, refresh_bf16_abort_ack;
    logic refresh_bf16_req_valid, refresh_bf16_req_ready, refresh_bf16_rsp_valid, refresh_bf16_rsp_ready;
    hardware_types_pkg::bf16_request_t refresh_bf16_req;
    logic post_bf16_req_valid, post_bf16_req_ready;
    hardware_types_pkg::bf16_request_t post_bf16_req;
    logic post_bf16_rsp_valid, post_bf16_rsp_ready;
    hardware_types_pkg::bf16_response_t post_bf16_rsp;
    logic post_reduction_req_valid, post_reduction_req_ready;
    hardware_types_pkg::reduction_request_t post_reduction_req;
    logic post_reduction_rsp_valid, post_reduction_rsp_ready;
    hardware_types_pkg::reduction_response_t post_reduction_rsp;
    logic post_exp_req_valid, post_exp_req_ready;
    hardware_types_pkg::softmax_exp_request_t post_exp_req;
    logic post_exp_rsp_valid, post_exp_rsp_ready;
    hardware_types_pkg::softmax_exp_response_t post_exp_rsp;
    logic post_reciprocal_req_valid, post_reciprocal_req_ready;
    hardware_types_pkg::softmax_reciprocal_request_t post_reciprocal_req;
    logic post_reciprocal_rsp_valid, post_reciprocal_rsp_ready;
    hardware_types_pkg::softmax_reciprocal_response_t post_reciprocal_rsp;
    logic [63:0] post_weight_bytes, post_scale_bytes;
    logic [63:0] post_pe_count, post_candidate_wait_cycles;
    logic shared_pe_idle, shared_rescale_idle;
    logic [4:0] resource_bf16_abort_request, resource_bf16_abort_ack;
    logic [4:0] resource_bf16_req_valid, resource_bf16_req_ready;
    logic [4:0] resource_bf16_rsp_valid, resource_bf16_rsp_ready;
    hardware_types_pkg::bf16_request_t resource_bf16_req [0:4];
    hardware_types_pkg::bf16_response_t resource_bf16_rsp [0:4];
    logic [3:0] resource_max_req_valid, resource_max_req_ready;
    logic [3:0] resource_max_rsp_valid, resource_max_rsp_ready;
    hardware_types_pkg::maximum_request_t resource_max_req [0:3];
    hardware_types_pkg::maximum_response_t resource_max_rsp [0:3];
    logic [1:0] resource_reduction_req_valid, resource_reduction_req_ready;
    logic [1:0] resource_reduction_rsp_valid, resource_reduction_rsp_ready;
    hardware_types_pkg::reduction_request_t resource_reduction_req [0:1];
    hardware_types_pkg::reduction_response_t resource_reduction_rsp [0:1];
    logic [1:0] resource_pe_abort_request, resource_pe_abort_ack;
    logic [1:0] resource_pe_req_valid, resource_pe_req_ready;
    logic [1:0] resource_pe_rsp_valid, resource_pe_rsp_ready;
    hardware_types_pkg::pe_request_t resource_pe_req [0:1];
    hardware_types_pkg::pe_response_t resource_pe_rsp [0:1];
    logic [1:0] resource_rescale_abort_request, resource_rescale_abort_ack;
    logic [1:0] resource_rescale_req_valid, resource_rescale_req_ready;
    logic [1:0] resource_rescale_rsp_valid, resource_rescale_rsp_ready;
    hardware_types_pkg::rescale_request_t resource_rescale_req [0:1];
    hardware_types_pkg::rescale_response_t resource_rescale_rsp [0:1];
    logic [3:0] resource_quant_abort_request, resource_quant_abort_ack;
    logic [3:0] resource_quant_scale_req_valid;
    logic [3:0] resource_quant_scale_req_ready;
    logic [3:0] resource_quant_scale_rsp_valid;
    logic [3:0] resource_quant_scale_rsp_ready;
    logic [3:0] resource_quant_values_req_valid;
    logic [3:0] resource_quant_values_req_ready;
    logic [3:0] resource_quant_values_rsp_valid;
    logic [3:0] resource_quant_values_rsp_ready;
    hardware_types_pkg::quant_scale_request_t resource_quant_scale_req [0:3];
    hardware_types_pkg::quant_scale_response_t resource_quant_scale_rsp [0:3];
    hardware_types_pkg::quant_values_request_t resource_quant_values_req [0:3];
    hardware_types_pkg::quant_values_response_t resource_quant_values_rsp [0:3];
    logic [1:0] resource_writer_abort_request;
    logic [1:0] resource_writer_cfg_valid, resource_writer_cfg_ready;
    logic [1:0] resource_writer_scale_valid, resource_writer_scale_ready;
    logic [1:0] resource_writer_values_valid, resource_writer_values_ready;
    hardware_types_pkg::activation_writer_config_t resource_writer_cfg [0:1];
    hardware_types_pkg::activation_writer_scale_t resource_writer_scale [0:1];
    hardware_types_pkg::activation_writer_values_t resource_writer_values [0:1];
    logic [1:0] resource_writer_done_pulse, resource_writer_error;
    logic resource_exp_req_valid, resource_exp_req_ready;
    logic resource_exp_rsp_valid, resource_exp_rsp_ready;
    hardware_types_pkg::softmax_exp_request_t resource_exp_req;
    hardware_types_pkg::softmax_exp_response_t resource_exp_rsp;
    logic resource_reciprocal_req_valid, resource_reciprocal_req_ready;
    logic resource_reciprocal_rsp_valid, resource_reciprocal_rsp_ready;
    hardware_types_pkg::softmax_reciprocal_request_t resource_reciprocal_req;
    hardware_types_pkg::softmax_reciprocal_response_t resource_reciprocal_rsp;
    logic qkv_metadata_request_valid, qkv_metadata_request_ready;
    hardware_types_pkg::dma_read_request_t qkv_metadata_request;
    logic qkv_metadata_data_valid, qkv_metadata_data_ready;
    hardware_types_pkg::dma_read_beat_t qkv_metadata_data;
    hardware_types_pkg::dma_completion_t qkv_metadata_completion;
    logic matmul_aux_request_valid, matmul_aux_request_ready;
    hardware_types_pkg::dma_read_request_t matmul_aux_request;
    logic matmul_aux_data_valid, matmul_aux_data_ready;
    hardware_types_pkg::dma_read_beat_t matmul_aux_data;
    hardware_types_pkg::dma_completion_t matmul_aux_completion;
    logic matmul_output_request_valid, matmul_output_request_ready;
    hardware_types_pkg::dma_write_request_t matmul_output_request;
    logic matmul_output_data_valid, matmul_output_data_ready;
    hardware_types_pkg::dma_write_beat_t matmul_output_data;
    hardware_types_pkg::dma_completion_t matmul_output_completion;
    logic block_panel_write_valid, block_panel_write_ready;
    hardware_types_pkg::matmul_panel_write_t block_panel_write;
    logic block_operand_read_valid, block_operand_read_ready;
    hardware_types_pkg::matmul_read_request_t block_operand_read;
    logic block_operand_rsp_valid;
    hardware_types_pkg::matmul_read_response_t block_operand_rsp;

    assign qkv_metadata_request.tag = 8'h05;
    assign qkv_metadata_request.wide_data = 1'b0;
    assign matmul_aux_request.wide_data = 1'b0;
    assign refresh_read_request.wide_data = 1'b0;
    assign refresh_requested = execution_forward_postprocess_configuration_bits[
        execution_config_pkg::EXECUTION_CONFIG_MAGIC_OFFSET*8 +: 32] == 32'h344e4c44 &&
        execution_forward_postprocess_configuration_bits[
            execution_config_pkg::EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET*8 +: 16] != 0;
    assign refresh_active = execution_refresh_select && execution_memory_stage == 3'd5;

    always_comb begin : post_dma_slot_select
        dma_read_request_valid[DMA_READ_QKV_METADATA] = refresh_active ? refresh_read_request_valid : post_active ?
            post_meta_request_valid : qkv_metadata_request_valid;
        dma_read_request[DMA_READ_QKV_METADATA] = refresh_active ? refresh_read_request : post_active ?
            post_meta_request : qkv_metadata_request;
        post_meta_request_ready = post_active &&
            dma_read_request_ready[DMA_READ_QKV_METADATA];
        qkv_metadata_request_ready = !post_active && !refresh_active &&
            dma_read_request_ready[DMA_READ_QKV_METADATA];
        post_meta_data_valid = post_active &&
            dma_read_data_valid[DMA_READ_QKV_METADATA];
        qkv_metadata_data_valid = !post_active && !refresh_active &&
            dma_read_data_valid[DMA_READ_QKV_METADATA];
        post_meta_data = dma_read_data[DMA_READ_QKV_METADATA];
        qkv_metadata_data = dma_read_data[DMA_READ_QKV_METADATA];
        dma_read_data_ready[DMA_READ_QKV_METADATA] = refresh_active ? refresh_read_data_ready : post_active ?
            post_meta_data_ready : qkv_metadata_data_ready;
        post_meta_completion = post_active ?
            dma_read_completion[DMA_READ_QKV_METADATA] : '0;
        qkv_metadata_completion = !post_active && !refresh_active ?
            dma_read_completion[DMA_READ_QKV_METADATA] : '0;

        dma_read_request_valid[DMA_READ_MATMUL_AUX] = post_active ?
            post_lm_request_valid : matmul_aux_request_valid;
        dma_read_request[DMA_READ_MATMUL_AUX] = post_active ?
            post_lm_request : matmul_aux_request;
        post_lm_request_ready = post_active &&
            dma_read_request_ready[DMA_READ_MATMUL_AUX];
        matmul_aux_request_ready = !post_active &&
            dma_read_request_ready[DMA_READ_MATMUL_AUX];
        post_lm_data_valid = post_active &&
            dma_read_data_valid[DMA_READ_MATMUL_AUX];
        matmul_aux_data_valid = !post_active &&
            dma_read_data_valid[DMA_READ_MATMUL_AUX];
        post_lm_data = dma_read_data[DMA_READ_MATMUL_AUX];
        matmul_aux_data = dma_read_data[DMA_READ_MATMUL_AUX];
        dma_read_data_ready[DMA_READ_MATMUL_AUX] = post_active ?
            post_lm_data_ready : matmul_aux_data_ready;
        post_lm_completion = post_active ?
            dma_read_completion[DMA_READ_MATMUL_AUX] : '0;
        matmul_aux_completion = !post_active ?
            dma_read_completion[DMA_READ_MATMUL_AUX] : '0;

        dma_write_request_valid[DMA_WRITE_MATMUL] = refresh_active ? refresh_write_request_valid : post_active ?
            post_write_request_valid : matmul_output_request_valid;
        dma_write_request[DMA_WRITE_MATMUL] = refresh_active ? refresh_write_request : post_active ?
            post_write_request : matmul_output_request;
        post_write_request_ready = post_active &&
            dma_write_request_ready[DMA_WRITE_MATMUL];
        matmul_output_request_ready = !post_active && !refresh_active &&
            dma_write_request_ready[DMA_WRITE_MATMUL];
        dma_write_data_valid[DMA_WRITE_MATMUL] = refresh_active ? refresh_write_data_valid : post_active ?
            post_write_data_valid : matmul_output_data_valid;
        dma_write_data[DMA_WRITE_MATMUL] = refresh_active ? refresh_write_data : post_active ?
            post_write_data : matmul_output_data;
        post_write_data_ready = post_active &&
            dma_write_data_ready[DMA_WRITE_MATMUL];
        matmul_output_data_ready = !post_active && !refresh_active &&
            dma_write_data_ready[DMA_WRITE_MATMUL];
        post_write_completion = post_active ?
            dma_write_completion[DMA_WRITE_MATMUL] : '0;
        matmul_output_completion = !post_active && !refresh_active ?
            dma_write_completion[DMA_WRITE_MATMUL] : '0;
    end

    always_comb begin : post_local_matmul_select
        matmul_panel_write_valid = post_active ? post_panel_write_valid :
            block_panel_write_valid;
        matmul_panel_write = post_active ? post_panel_write :
            block_panel_write;
        post_panel_write_ready = post_active && matmul_panel_write_ready;
        block_panel_write_ready = !post_active && matmul_panel_write_ready;
        matmul_read_bundle_valid = post_active ? post_operand_read_valid :
            block_operand_read_valid;
        matmul_read_req = post_active ? post_operand_read :
            block_operand_read;
        post_operand_read_ready = post_active && matmul_read_bundle_ready;
        block_operand_read_ready = !post_active && matmul_read_bundle_ready;
        post_operand_rsp_valid = post_active && matmul_read_response_valid;
        block_operand_rsp_valid = !post_active && matmul_read_response_valid;
        post_operand_rsp = matmul_read_rsp;
        block_operand_rsp = matmul_read_rsp;
    end

    logic rope_busy;

    logic rope_source_read_is_k;
    logic [15:0] rope_source_read_lane_mask;

    logic [63:0] rope_cos_read_req_address;
    logic [15:0] rope_cos_read_req_bytes;

    logic [63:0] rope_sin_read_req_address;
    logic [15:0] rope_sin_read_req_bytes;

    logic rope_destination_write_is_k;
    logic [15:0] rope_destination_write_lane_mask;

    logic rope_arithmetic_req_valid;
    logic rope_arithmetic_req_ready;
    logic rope_arithmetic_rsp_valid;
    logic rope_arithmetic_rsp_ready;

    logic [5:0] qkv_current_write_head;
    logic [5:0] qkv_current_write_row;
    logic [10:0] qkv_current_write_slot;
    logic [4:0] qkv_current_write_chunk;
    assign matmul_local_layout =
        current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED ||
        current_layout.panel_layout == LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL ||
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT;
    assign qkv_head_staging_layout_ready =
        ((qkv_phase_active &&
          current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION) ||
         (attention_q_head_active && current_layout.q_head_active)) &&
        current_layout.qkv_head_staging == qkv_head_staging_layout &&
        !local_layout_update_valid;

    assign execution_hidden_source_selected =
        execution_memory_stage == EXEC_MEMORY_HIDDEN_READ ||
        execution_memory_stage == EXEC_MEMORY_HIDDEN_WRITE;
    assign post_hidden_source_selected = post_active;
    assign ffn_hidden_source_selected = execution_block_schedule_active &&
        (execution_layer_operator_index == LAYER_OPERATOR_FFN_RESIDUAL_SPILL ||
         execution_layer_operator_index == LAYER_OPERATOR_FFN_HIDDEN_REFILL);
    assign ffn_hidden_refill = execution_layer_operator_index ==
        LAYER_OPERATOR_FFN_HIDDEN_REFILL;
    assign ffn_hidden_start_valid = operator_command_valid &&
        pending_start_enable && pending_start_select[COMMAND_START_HIDDEN];
    assign rms_spill_start_valid = operator_command_valid && pending_start_enable &&
        pending_command.layer_operator_index == LAYER_OPERATOR_ATTENTION_RMSNORM &&
        pending_command.qkv_residual_spill && pending_start_select[COMMAND_START_HIDDEN];
    assign qkv_residual_spill_start_valid = rms_spill_start_valid || (active_command_valid &&
        active_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION &&
        active_command.qkv_residual_spill &&
        !qkv_residual_spill_active && !qkv_residual_spill_complete);
    assign hidden_start_operation =
        (ffn_hidden_source_selected || qkv_residual_spill_start_valid) ?
        (ffn_hidden_refill ? 2'd3 : 2'd2) :
        post_hidden_source_selected ? 2'd0 :
        execution_hidden_start_residual ? 2'd2 :
        (execution_hidden_start_write ? 2'd1 : 2'd0);
    assign hidden_start_valid =
        ((execution_hidden_source_selected && execution_hidden_start_valid) ||
         (ffn_hidden_source_selected && ffn_hidden_start_valid) ||
         qkv_residual_spill_start_valid ||
         (post_hidden_source_selected && post_hidden_start_valid)) &&
        ((qkv_residual_spill_start_valid &&
          (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
           (rms_spill_start_valid && current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM))) ||
         (!qkv_residual_spill_start_valid &&
          (current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM ||
           current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT))) &&
        !local_layout_update_valid;
    assign execution_hidden_start_ready = hidden_start_ready &&
        execution_hidden_source_selected &&
        current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM &&
        !local_layout_update_valid;
    assign post_hidden_start_ready = hidden_start_ready &&
        post_hidden_source_selected &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !local_layout_update_valid;
    assign hidden_done_ready = qkv_residual_spill_active ? 1'b1 :
        hidden_done_for_ffn ?
        (active_command_valid &&
         (active_command.layer_operator_index == LAYER_OPERATOR_FFN_RESIDUAL_SPILL ||
          active_command.layer_operator_index == LAYER_OPERATOR_FFN_HIDDEN_REFILL) &&
         operator_command_done_ready && command_done_can_complete) :
        post_hidden_source_selected ? post_hidden_done_ready :
        execution_hidden_done_ready;
    assign execution_hidden_done_valid = hidden_done_valid &&
        !hidden_done_for_ffn && execution_hidden_source_selected;
    assign execution_hidden_error = hidden_error && !hidden_done_for_ffn;
    assign post_hidden_done_valid = hidden_done_valid &&
        !hidden_done_for_ffn && post_hidden_source_selected;
    assign post_hidden_error = hidden_error;
    assign hidden_abort_request = 1'b0;

    hidden_transfer hidden_io (
        .clk(clk), .rst(rst), .abort_request(hidden_abort_request),
        .abort_ack(hidden_abort_ack), .start_valid(hidden_start_valid),
        .start_ready(hidden_start_ready), .start_operation(hidden_start_operation),
        .start_active_rows(post_hidden_source_selected ?
            post_hidden_start_rows : (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ? row_active_rows :
            execution_hidden_start_active_rows),
        .start_physical_to_token_ordinal(post_hidden_source_selected ?
            288'd0 : (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ?
            row_physical_to_local : execution_hidden_start_row_permutation),
        .start_direct_row_index_enable(post_hidden_source_selected ? 1'b1 :
            (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ?
            1'b0 : execution_hidden_start_direct_row_index_enable),
        .start_direct_row_index(post_hidden_source_selected ?
            post_hidden_start_ddr_row_index : (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ?
            528'd0 : execution_hidden_start_direct_row_index),
        .start_source_descriptor_enable(
            !post_hidden_source_selected && !ffn_hidden_source_selected &&
            !qkv_residual_spill_start_valid &&
            execution_hidden_start_source_descriptor_enable),
        .start_source_index(execution_hidden_start_source_index),
        .start_source_embedding(execution_hidden_start_source_embedding),
        .start_embedding_base(execution_hidden_start_embedding_base),
        .start_embedding_limit(execution_hidden_start_embedding_limit),
        .start_ddr_base(post_hidden_source_selected ?
            post_hidden_start_address : (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ?
            ffn_workspace_config.residual_base :
            execution_hidden_start_address),
        .start_ddr_limit(post_hidden_source_selected ?
            post_hidden_start_limit : (ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid) ?
            ffn_workspace_config.residual_limit :
            execution_hidden_start_limit),
        .start_for_ffn(ffn_hidden_source_selected ||
            qkv_residual_spill_start_valid),
        .busy(hidden_busy), .done_valid(hidden_done_valid),
        .done_ready(hidden_done_ready), .done_for_ffn(hidden_done_for_ffn),
        .error(hidden_error),
        .error_id(hidden_error_id),
        .read_request_valid(dma_read_request_valid[DMA_READ_HIDDEN]),
        .read_request_ready(dma_read_request_ready[DMA_READ_HIDDEN]),
        .read_request_address(dma_read_request[DMA_READ_HIDDEN].byte_address),
        .read_request_bytes(dma_read_request[DMA_READ_HIDDEN].byte_count),
        .read_request_tag(dma_read_request[DMA_READ_HIDDEN].tag),
        .read_request_done(dma_read_completion[DMA_READ_HIDDEN].done_pulse),
        .read_request_error(dma_read_completion[DMA_READ_HIDDEN].error),
        .read_data_valid(dma_read_data_valid[DMA_READ_HIDDEN]),
        .read_data_ready(dma_read_data_ready[DMA_READ_HIDDEN]),
        .read_data(dma_read_data[DMA_READ_HIDDEN].data),
        .read_byte_enable(dma_read_data[DMA_READ_HIDDEN].byte_enable),
        .read_data_last(dma_read_data[DMA_READ_HIDDEN].last),
        .read_data_tag(dma_read_data[DMA_READ_HIDDEN].tag),
        .write_request_valid(dma_write_request_valid[DMA_WRITE_HIDDEN]),
        .write_request_ready(dma_write_request_ready[DMA_WRITE_HIDDEN]),
        .write_request_address(
            dma_write_request[DMA_WRITE_HIDDEN].byte_address),
        .write_request_bytes(dma_write_request[DMA_WRITE_HIDDEN].byte_count),
        .write_request_tag(dma_write_request[DMA_WRITE_HIDDEN].tag),
        .write_request_done(dma_write_completion[DMA_WRITE_HIDDEN].done_pulse),
        .write_request_error(dma_write_completion[DMA_WRITE_HIDDEN].error),
        .write_data_valid(dma_write_data_valid[DMA_WRITE_HIDDEN]),
        .write_data_ready(dma_write_data_ready[DMA_WRITE_HIDDEN]),
        .write_data(dma_write_data[DMA_WRITE_HIDDEN].data),
        .write_byte_enable(dma_write_data[DMA_WRITE_HIDDEN].byte_enable),
        .write_data_last(dma_write_data[DMA_WRITE_HIDDEN].last),
        .local_write_valid(hidden_local_write_valid),
        .local_write_ready(hidden_local_write_ready),
        .local_write_physical_row(hidden_local_write.physical_row),
        .local_write_channel_word(hidden_local_write.channel_word),
        .local_write_data(hidden_local_write.data),
        .local_write_byte_enable(hidden_local_write.byte_enable),
        .local_read_valid(hidden_local_read_valid),
        .local_read_ready(hidden_local_read_ready),
        .local_read_physical_row(hidden_local_read_req.physical_row),
        .local_read_channel_word(hidden_local_read_req.channel_word),
        .local_read_response_valid(hidden_local_read_response_valid),
        .local_read_data(hidden_local_read_rsp.data),
        .accepted_dma_request_count(), .accepted_dma_data_count(),
        .accepted_local_request_count()
    );

    always_ff @(posedge clk) begin : qkv_residual_spill_state
        if (rst) begin
            qkv_residual_spill_active <= 1'b0;
            rms_after_spill_pending <= 1'b0;
            qkv_residual_spill_complete <= 1'b0;
            qkv_residual_spill_error <= 1'b0;
            qkv_residual_spill_error_id <= '0;
        end else begin
            if (operator_command_valid && operator_command_ready &&
                (pending_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION ||
                 pending_command.layer_operator_index == LAYER_OPERATOR_ATTENTION_RMSNORM)) begin
                qkv_residual_spill_active <= 1'b0;
                qkv_residual_spill_complete <=
                    !pending_command.qkv_residual_spill;
                qkv_residual_spill_error <= 1'b0;
                qkv_residual_spill_error_id <= '0;
                rms_after_spill_pending <= rms_spill_start_valid;
            end
            if (rms_after_spill_pending && block_rms_start_valid && rms_start_ready)
                rms_after_spill_pending <= 1'b0;
            if (qkv_residual_spill_start_valid && hidden_start_valid &&
                hidden_start_ready)
                qkv_residual_spill_active <= 1'b1;
            if (qkv_residual_spill_active && hidden_done_valid &&
                hidden_done_ready) begin
                qkv_residual_spill_active <= 1'b0;
                qkv_residual_spill_complete <= 1'b1;
                qkv_residual_spill_error <= hidden_error;
                qkv_residual_spill_error_id <= hidden_error_id;
            end
            if (active_command_valid &&
                (active_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION ||
                 active_command.layer_operator_index == LAYER_OPERATOR_ATTENTION_RMSNORM) &&
                operator_command_done_valid && operator_command_done_ready) begin
                qkv_residual_spill_active <= 1'b0;
                qkv_residual_spill_complete <= 1'b0;
                rms_after_spill_pending <= 1'b0;
            end
        end
    end

    assign rms_abort_request = 1'b0;
    assign block_rms_start_valid = (operator_command_valid &&
        pending_start_enable && pending_start_select[COMMAND_START_RMS]) ||
        (active_command_valid && rms_after_spill_pending &&
         qkv_residual_spill_complete && !qkv_residual_spill_error);
    assign block_rms_gamma_bypass =
        execution_forward_postprocess_configuration_bits[
            execution_config_pkg::EXECUTION_CONFIG_MAGIC_OFFSET*8 +: 32] ==
            32'h344e4c44 &&
        execution_forward_postprocess_configuration_bits[
            execution_config_pkg::EXECUTION_CONFIG_VERSION_OFFSET*8 +: 16] ==
            16'd4;
    assign rms_start_valid = post_rms_start_valid || block_rms_start_valid;
    assign post_rms_start_ready = rms_start_ready && post_active;
    assign post_rms_done_valid = rms_done_valid && post_active;
    assign post_rms_error = rms_error;
    assign rms_done_ready = post_active ? post_rms_done_ready :
        block_rms_done_ready;
    assign rms_ffn_command = execution_layer_operator_index ==
        LAYER_OPERATOR_FFN_RMSNORM;
    assign rms_memory_phase_active = post_active ||
        (execution_block_schedule_active &&
         (execution_layer_operator_index == LAYER_OPERATOR_ATTENTION_RMSNORM ||
          execution_layer_operator_index == LAYER_OPERATOR_FFN_RMSNORM));

    always_ff @(posedge clk) begin
        if (rst)
            post_rms_active_rows <= '0;
        else if (post_rms_start_valid && post_rms_start_ready)
            post_rms_active_rows <= post_rms_start_rows;
    end

    rmsnorm_engine #(
        .MAX_ELEMENTS(4096), .MAX_ROWS(48), .TAG_WIDTH(16),
        .LOAD_GAMMA_FROM_DDR(1'b1)
    ) shared_rmsnorm (
        .clk(clk), .rst(rst), .abort_request(rms_abort_request),
        .abort_ack(rms_abort_ack), .start_valid(rms_start_valid),
        .start_ready(rms_start_ready),
        .start_row_count(post_rms_start_valid ? post_rms_start_rows :
            row_active_rows),
        .start_element_count(post_rms_start_valid ?
            post_rms_start_elements : 16'd4096),
        .start_epsilon_bf16(post_rms_start_valid ? post_rms_start_epsilon :
            rms_ffn_command ?
            rmsnorm_layer_config.ffn_epsilon_bf16 :
            rmsnorm_layer_config.attention_epsilon_bf16),
        .start_gamma_ddr_address(rms_ffn_command ?
            rmsnorm_layer_config.ffn_gamma_base :
            rmsnorm_layer_config.attention_gamma_base),
        .start_gamma_bypass(post_rms_start_valid ?
            post_rms_start_gamma_bypass : block_rms_gamma_bypass),
        .tile_read_req_valid(engine_rms_tile_read_req_valid),
        .tile_read_req_ready(engine_rms_tile_read_req_ready),
        .tile_read_output_pass(rms_tile_read_output_pass),
        .tile_read_gamma_bypass(rms_gamma_bypass),
        .tile_read_row_base(engine_rms_tile_read_req.row_base),
        .tile_read_element(engine_rms_tile_read_req.element),
        .tile_read_tag(engine_rms_tile_read_req.tag),
        .tile_read_rsp_valid(engine_rms_tile_read_rsp_valid),
        .tile_read_rsp_ready(engine_rms_tile_read_rsp_ready),
        .tile_read_values(engine_rms_tile_read_rsp.values),
        .tile_read_gamma(engine_rms_tile_read_rsp.gamma),
        .tile_read_lane_mask(engine_rms_tile_read_rsp.lane_mask),
        .tile_read_rsp_tag(engine_rms_tile_read_rsp.tag),
        .gamma_dma_request_valid(
            dma_read_request_valid[DMA_READ_RMS_GAMMA]),
        .gamma_dma_request_ready(
            dma_read_request_ready[DMA_READ_RMS_GAMMA]),
        .gamma_dma_request_address(
            dma_read_request[DMA_READ_RMS_GAMMA].byte_address),
        .gamma_dma_request_bytes(
            dma_read_request[DMA_READ_RMS_GAMMA].byte_count),
        .gamma_dma_request_done(
            dma_read_completion[DMA_READ_RMS_GAMMA].done_pulse),
        .gamma_dma_request_error(
            dma_read_completion[DMA_READ_RMS_GAMMA].error),
        .gamma_dma_data_valid(dma_read_data_valid[DMA_READ_RMS_GAMMA]),
        .gamma_dma_data_ready(dma_read_data_ready[DMA_READ_RMS_GAMMA]),
        .gamma_dma_data(dma_read_data[DMA_READ_RMS_GAMMA].data),
        .gamma_dma_byte_enable(
            dma_read_data[DMA_READ_RMS_GAMMA].byte_enable),
        .gamma_stage_write_valid(rms_gamma_stage_valid),
        .gamma_stage_write_ready(rms_gamma_stage_ready),
        .gamma_stage_write_word(rms_gamma_write.word),
        .gamma_stage_write_data(rms_gamma_write.data),
        .gamma_stage_write_byte_enable(rms_gamma_write.byte_enable),
        .scratch_read_valid(rms_scratch_read_valid),
        .scratch_read_ready(rms_scratch_read_ready),
        .scratch_read_bank(rms_scratch_read_req.bank),
        .scratch_read_left_address(rms_scratch_read_req.left_address),
        .scratch_read_right_address(rms_scratch_read_req.right_address),
        .scratch_read_tag(rms_scratch_read_req.tag),
        .scratch_read_rsp_valid(rms_scratch_read_rsp_valid),
        .scratch_read_rsp_ready(rms_scratch_read_rsp_ready),
        .scratch_read_left_data(rms_scratch_read_rsp.left_data),
        .scratch_read_right_data(rms_scratch_read_rsp.right_data),
        .scratch_read_rsp_tag(rms_scratch_read_rsp.tag),
        .scratch_write_valid(rms_scratch_write_valid),
        .scratch_write_ready(rms_scratch_write_ready),
        .scratch_write_bank(rms_scratch_write.bank),
        .scratch_write_address(rms_scratch_write.address),
        .scratch_write_data(rms_scratch_write.data),
        .scratch_write_byte_enable(rms_scratch_write.byte_enable),
        .norm_write_valid(rms_norm_write_valid),
        .norm_write_ready(rms_norm_write_ready),
        .norm_write_row_base(rms_norm_write.row_base),
        .norm_write_element(rms_norm_write.element),
        .norm_write_data(rms_norm_write.data),
        .norm_write_lane_mask(rms_norm_write.lane_mask),
        .norm_write_tag(rms_norm_write_tag),
        .trace_sample_valid(rms_trace_sample_valid), .trace_sample_ready(1'b1),
        .trace_sample_row_base(rms_trace_sample_row),
        .trace_sample_element(rms_trace_sample_element),
        .trace_sample_data(rms_trace_sample_data),
        .trace_sample_byte_enable(rms_trace_sample_byte_enable),
        .vector_req_valid(rms_arithmetic_req_valid),
        .vector_req_ready(rms_arithmetic_req_ready),
        .vector_req_operation(bf16_client_req[0].operation),
        .vector_req_values(bf16_client_req[0].values),
        .vector_req_paired_values(bf16_client_req[0].paired_values),
        .vector_req_factor0_values(bf16_client_req[0].factor0_values),
        .vector_req_factor1_values(bf16_client_req[0].factor1_values),
        .vector_req_lane_mask(bf16_client_req[0].lane_mask),
        .vector_req_tag(bf16_client_req[0].tag),
        .vector_rsp_valid(rms_arithmetic_rsp_valid),
        .vector_rsp_ready(rms_arithmetic_rsp_ready),
        .vector_rsp_values(bf16_client_rsp[0].values),
        .vector_rsp_lane_mask(bf16_client_rsp[0].lane_mask),
        .vector_rsp_tag(bf16_client_rsp[0].tag),
        .reduction_req_valid(rms_reduction_req_valid),
        .reduction_req_ready(rms_reduction_req_ready),
        .reduction_req_values(reduction_client_req[0].values),
        .reduction_req_lane_mask(reduction_client_req[0].lane_mask),
        .reduction_req_tag(reduction_client_req[0].tag),
        .reduction_rsp_valid(rms_reduction_rsp_valid),
        .reduction_rsp_ready(rms_reduction_rsp_ready),
        .reduction_rsp_values(reduction_client_rsp[0].values),
        .reduction_rsp_row_mask(reduction_client_rsp[0].row_mask),
        .reduction_rsp_tag(reduction_client_rsp[0].tag),
        .done_valid(rms_done_valid), .done_ready(rms_done_ready),
        .error(rms_error), .error_id(rms_error_id),
        .gamma_loaded(rms_gamma_loaded),
        .execution_active(rms_active), .accepted_tile_read_count(),
        .completed_tile_read_count(), .accepted_norm_write_count(),
        .completed_row_count(),
        .completed_command_count()
    );

    always_comb begin : rms_tile_source_select
        rms_tile_read_req_valid = rms_active ?
            engine_rms_tile_read_req_valid : post_rms_source_req_valid;
        rms_tile_read_req = rms_active ? engine_rms_tile_read_req :
            post_rms_source_req;
        engine_rms_tile_read_req_ready = rms_active &&
            rms_tile_read_req_ready;
        post_rms_source_req_ready = !rms_active &&
            rms_tile_read_req_ready;
        engine_rms_tile_read_rsp_valid = rms_active &&
            rms_tile_read_rsp_valid;
        post_rms_source_rsp_valid = !rms_active &&
            rms_tile_read_rsp_valid;
        engine_rms_tile_read_rsp = rms_tile_read_rsp;
        post_rms_source_rsp = rms_tile_read_rsp;
        if (post_active) begin
            for (integer post_rms_row = 0; post_rms_row < 8;
                 post_rms_row++) begin
                engine_rms_tile_read_rsp.lane_mask[
                    post_rms_row*8 +: 8] &=
                    {8{post_rms_row < post_rms_active_rows}};
                post_rms_source_rsp.lane_mask[post_rms_row*8 +: 8] &=
                    {8{post_rms_row < post_rms_active_rows}};
            end
        end
        rms_tile_read_rsp_ready = rms_active ?
            engine_rms_tile_read_rsp_ready : post_rms_source_rsp_ready;
    end

    qkv_preparation_controller #(
        .MAX_ROWS(48), .INTEGRATED_CACHE_COMMIT(1'b1),
        .TRUSTED_START_CONFIGURATION(1'b1)
    )
    qkv_preparation (
        .clk(clk), .rst(rst), .abort_request(qkv_abort_request),
        .abort_ack(qkv_abort_ack), .start_valid(qkv_start_valid),
        .start_ready(qkv_start_ready), .start_row_count(row_active_rows),
        .start_head_count(6'(ATTENTION_HEAD_COUNT)), .start_head_dim(9'd128),
        .start_row_enable(row_matmul_valid),
        .start_kv_write_disable(attention_row_config.kv_write_disable),
        .start_token_position(row_token_position),
        .start_kv_pair_enable(kv_pair_enable),
        .start_second_row_count(ffn_batch_pair_config.second_rows),
        .start_second_row_enable(kv_pair_second_config.row_enable),
        .start_second_kv_write_disable(kv_pair_second_config.kv_write_disable),
        .start_second_token_position(kv_pair_second_config.token_position),
        .start_batch_group(kv_batch_group_config),
        .kv_pair_layout_ready(current_layout == accepted_layout),
        .kv_pair_active(qkv_pair_active), .kv_second_batch(qkv_second_batch),
        .kv_batch_index(qkv_batch_index),
        .active_token_count(qkv_active_token_count), .active_token_enable(),
        .start_v_static_scale_bf16(16'd0),
        .start_v_scale_per_head(qkv_layer_config.v_scale_per_head),
        .start_rope_cos_base(qkv_layer_config.rope_cos_base),
        .start_rope_sin_base(qkv_layer_config.rope_sin_base),
        .start_v_scale_base(qkv_layer_config.v_scale_base),
        .q_head_start_valid(attention_q_head_start_valid),
        .q_head_start_ready(attention_q_head_start_ready),
        .q_head_start_head(attention_q_head_start_head),
        .q_head_start_slot(attention_q_head_start_slot),
        .q_head_postprocess_enable(attention_q_head_postprocess_enable),
        .q_head_preprocess_input(active_command_valid &&
            active_command.q_head_preprocess),
        .q_head_start_precompute_only(attention_q_head_precompute_only),
        .q_head_start_use_staged(attention_q_head_use_staged),
        .q_head_matmul_done(attention_q_head_matmul_done),
        .q_head_done_valid(attention_q_head_done_valid),
        .q_head_done_ready(attention_q_head_done_ready),
        .q_head_error(attention_q_head_error),
        .q_head_error_id(attention_q_head_error_id),
        .done_valid(qkv_done_valid), .done_ready(qkv_done_ready),
        .error(qkv_error), .error_id(qkv_error_id),
        .v_scale_select_head(attention_operand_req.active_head),
        .loaded_v_static_scale_bf16(qkv_v_static_scale),
        .compute_stage(qkv_compute_stage),
        .memory_stage(qkv_memory_stage),
        .matmul_start_valid(qkv_matmul_start_valid),
        .matmul_start_ready(qkv_matmul_start_ready),
        .matmul_qkv_select(qkv_matmul_qkv_select), .matmul_head(qkv_matmul_head),
        .matmul_preprocess_input(qkv_matmul_preprocess_input),
        .matmul_use_resident_a8(qkv_matmul_use_resident_a8),
        .kv_group_activation_reuse(qkv_group_activation_reuse),
        .matmul_qkv_w8_layout(qkv_matmul_qkv_w8_layout),
        .matmul_group_staging_base(qkv_matmul_group_staging_base),
        .matmul_prefetch_valid(qkv_matmul_prefetch_valid),
        .matmul_prefetch_ready(qkv_matmul_prefetch_ready),
        .matmul_prefetch_qkv_select(qkv_matmul_prefetch_qkv_select),
        .matmul_prefetch_head(qkv_matmul_prefetch_head),
        .matmul_done_valid(qkv_matmul_done_valid),
        .matmul_error(qkv_matmul_error),
        .matmul_abort_request(qkv_matmul_abort_request),
        .matmul_abort_ack(qkv_matmul_abort_ack),
        .rope_start_valid(qkv_rope_start_valid),
        .rope_start_ready(qkv_rope_start_ready),
        .rope_source_is_k(qkv_rope_source_is_k), .rope_head(qkv_rope_head),
        .rope_done_pulse(qkv_rope_done_pulse), .rope_error(qkv_rope_error),
        .rope_abort_request(qkv_rope_abort_request),
        .rope_abort_ack(qkv_rope_abort_ack),
        .head_staging_layout(qkv_head_staging_layout),
        .head_staging_layout_ready(qkv_head_staging_layout_ready),
        .matmul_overlap_active(qkv_matmul_overlap_active),
        .head_read_req_valid(qkv_head_read_req_valid),
        .head_read_req_ready(qkv_head_read_req_ready),
        .head_read_rope_destination(qkv_head_read_req.rope_destination),
        .head_read_physical_row(qkv_head_read_req.physical_row),
        .head_read_word(qkv_head_read_req.word),
        .head_read_rsp_valid(qkv_head_read_rsp_valid),
        .head_read_rsp_ready(qkv_head_read_rsp_ready),
        .head_read_rsp_data(qkv_head_read_rsp.data),
        .head_read_rsp_lane_mask(qkv_head_read_rsp.lane_mask),
        .head_stage_write_valid(qkv_head_stage_write_valid),
        .head_stage_write_ready(qkv_head_stage_write_ready),
        .head_stage_write_source(qkv_head_stage_write.source),
        .head_stage_write_bank(qkv_head_stage_write.bank),
        .head_stage_write_port(qkv_head_stage_write.port),
        .head_stage_write_word(qkv_head_stage_write.word),
        .head_stage_write_data(qkv_head_stage_write.data),
        .head_stage_write_byte_enable(qkv_head_stage_write.byte_enable),
        .head_tile_memory_req_valid(qkv_head_tile_memory_req_valid),
        .head_tile_memory_req_ready(qkv_head_tile_memory_req_ready),
        .head_tile_memory_row_base(qkv_head_tile_read_req.row_base),
        .head_tile_memory_word(qkv_head_tile_read_req.word),
        .head_tile_memory_row_mask(qkv_head_tile_read_req.row_mask),
        .head_tile_memory_rsp_valid(qkv_head_tile_memory_rsp_valid),
        .head_tile_memory_rsp_data(qkv_head_tile_read_rsp.data),
        .head_tile_memory_rsp_lane_mask(qkv_head_tile_read_rsp.lane_mask),
        .max_req_valid(qkv_max_req_valid), .max_req_ready(qkv_max_req_ready),
        .max_req_values(maximum_client_req[0].values),
        .max_req_lane_mask(maximum_client_req[0].lane_mask),
        .max_req_tag(maximum_client_req[0].tag),
        .max_rsp_valid(qkv_max_rsp_valid), .max_rsp_ready(qkv_max_rsp_ready),
        .max_rsp_values(maximum_client_rsp[0].values),
        .max_rsp_row_mask(maximum_client_rsp[0].row_mask),
        .max_rsp_tag(maximum_client_rsp[0].tag),
        .quant_scale_req_valid(qkv_quant_scale_req_valid),
        .quant_scale_req_ready(qkv_quant_scale_req_ready),
        .quant_scale_req_use_static_scale(quant_scale_client_req[0].use_static_scale),
        .quant_scale_req_static_scales_bf16(quant_scale_client_req[0].static_scales_bf16),
        .quant_scale_req_row_max_abs(quant_scale_client_req[0].row_max_abs),
        .quant_scale_req_row_mask(quant_scale_client_req[0].row_mask),
        .quant_scale_rsp_valid(qkv_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(qkv_quant_scale_rsp_ready),
        .quant_scale_rsp_values_bf16(quant_scale_client_rsp[0].values_bf16),
        .quant_req_valid(qkv_quant_req_valid), .quant_req_ready(qkv_quant_req_ready),
        .quant_req_values_bf16(quant_values_client_req[0].values_bf16),
        .quant_req_lane_mask(quant_values_client_req[0].lane_mask),
        .quant_req_tag(quant_values_client_req[0].tag),
        .quant_rsp_valid(qkv_quant_rsp_valid), .quant_rsp_ready(qkv_quant_rsp_ready),
        .quant_rsp_values(quant_values_client_rsp[0].values),
        .quant_rsp_lane_mask(quant_values_client_rsp[0].lane_mask),
        .quant_rsp_tag(quant_values_client_rsp[0].tag),
        .quant_abort_request(qkv_quant_abort_request),
        .quant_abort_ack(qkv_quant_abort_ack),
        .quantized_commit_valid(qkv_quantized_commit_valid),
        .quantized_commit_ready(qkv_quantized_commit_ready),
        .quantized_commit_qkv_select(qkv_quantized_commit_qkv_select),
        .quantized_commit_head(qkv_quantized_commit_head),
        .quantized_commit_physical_row(qkv_quantized_commit_physical_row),
        .quantized_commit_token_position(qkv_quantized_commit_token_position),
        .quantized_commit_half(qkv_quantized_commit_half),
        .quantized_commit_values(qkv_quantized_commit_values),
        .quantized_commit_lane_mask(qkv_quantized_commit_lane_mask),
        .quantized_commit_scale_bf16(qkv_quantized_commit_scale),
        .quantized_commit_tag(qkv_quantized_commit_tag),
        .quantized_commit_requires_response(qkv_quantized_commit_requires_response),
        .cache_write_complete_valid(qkv_cache_write_complete_valid),
        .cache_write_complete_ready(qkv_cache_write_complete_ready),
        .cache_write_complete_error(qkv_cache_write_complete_error),
        .cache_abort_request(),
        .cache_abort_ack(1'b0),
        .cache_mapping_commit_valid(qkv_cache_mapping_commit_valid),
        .cache_mapping_commit_ready(qkv_cache_mapping_commit_ready),
        .cache_mapping_commit_error(qkv_cache_mapping_commit_error),
        .cache_config_start_valid(cache_start_valid),
        .cache_config_start_ready(cache_start_ready),
        .cache_config_done_pulse(cache_config_done_pulse),
        .cache_config_error(cache_error),
        .cache_config_release(qkv_cache_config_release),
        .metadata_read_request_valid(qkv_metadata_request_valid),
        .metadata_read_request_ready(qkv_metadata_request_ready),
        .metadata_read_request_address(qkv_metadata_request.byte_address),
        .metadata_read_request_bytes(qkv_metadata_request.byte_count),
        .metadata_read_request_done(qkv_metadata_completion.done_pulse),
        .metadata_read_request_error(qkv_metadata_completion.error),
        .metadata_read_data_valid(qkv_metadata_data_valid),
        .metadata_read_data_ready(qkv_metadata_data_ready),
        .metadata_read_data(qkv_metadata_data.data),
        .metadata_read_byte_enable(qkv_metadata_data.byte_enable),
        .constant_stage_write_valid(qkv_constant_stage_valid),
        .constant_stage_write_ready(qkv_constant_stage_ready),
        .constant_stage_write_word(qkv_constant_write.word),
        .constant_stage_write_data(qkv_constant_write.data),
        .constant_stage_write_byte_enable(qkv_constant_write.byte_enable),
        .q_local_write_valid(qkv_q_local_write_valid),
        .q_local_write_ready(qkv_q_local_write_ready),
        .q_local_write_scale(qkv_q_write.scale),
        .q_local_write_slot(qkv_q_write.slot),
        .q_local_write_physical_row(qkv_q_write.physical_row),
        .q_local_write_word(qkv_q_write.word),
        .q_local_write_data(qkv_q_write.data),
        .q_local_write_byte_enable(qkv_q_write.byte_enable),
        .current_write_valid(cache_current_write_valid),
        .current_write_ready(cache_current_write_ready),
        .current_write_head(qkv_current_write_head),
        .current_write_physical_row(qkv_current_write_row),
        .current_write_logical_slot(qkv_current_write_slot),
        .current_write_chunk(qkv_current_write_chunk),
        .current_write_accepted(cache_current_write_accepted),
        .current_k_write_address(cache_current_k_write_address),
        .current_v_write_address(cache_current_v_write_address),
        .current_k_scale_write_address(cache_current_k_scale_write_address),
        .cache_writer_request_valid(
            dma_write_request_valid[DMA_WRITE_QKV_CACHE]),
        .cache_writer_request_ready(
            dma_write_request_ready[DMA_WRITE_QKV_CACHE]),
        .cache_writer_request_address(
            dma_write_request[DMA_WRITE_QKV_CACHE].byte_address),
        .cache_writer_request_bytes(
            dma_write_request[DMA_WRITE_QKV_CACHE].byte_count),
        .cache_writer_request_tag(
            dma_write_request[DMA_WRITE_QKV_CACHE].tag),
        .cache_writer_request_done(
            dma_write_completion[DMA_WRITE_QKV_CACHE].done_pulse),
        .cache_writer_request_error(
            dma_write_completion[DMA_WRITE_QKV_CACHE].error),
        .cache_writer_write_valid(
            dma_write_data_valid[DMA_WRITE_QKV_CACHE]),
        .cache_writer_write_ready(
            dma_write_data_ready[DMA_WRITE_QKV_CACHE]),
        .cache_writer_write_data(dma_write_data[DMA_WRITE_QKV_CACHE].data),
        .cache_writer_write_byte_enable(
            dma_write_data[DMA_WRITE_QKV_CACHE].byte_enable),
        .cache_writer_write_last(dma_write_data[DMA_WRITE_QKV_CACHE].last),
        .accepted_matmul_count(qkv_accepted_matmul_count),
        .accepted_rope_count(qkv_accepted_rope_count),
        .accepted_head_read_count(qkv_accepted_head_read_count),
        .accepted_quant_count(qkv_accepted_quant_count),
        .accepted_quantized_commit_count(qkv_accepted_quantized_commit_count),
        .completed_cache_write_count(qkv_completed_cache_write_count),
        .outstanding_cache_writes(qkv_outstanding_cache_writes)
    );

    matmul_controller #(
        .FIXED_SCHEDULE_INGRESS(1'b1),
        .TRUSTED_FIXED_CONFIGURATION(1'b1),
        .ATTENTION_HEAD_COUNT(ATTENTION_HEAD_COUNT),
        .HIDDEN_FEATURES(HIDDEN_FEATURES),
        .FFN_FEATURES(FFN_FEATURES)
    ) shared_matmul (
        .clk(clk), .rst(rst), .abort_request(matmul_abort_request),
        .abort_ack(matmul_abort_ack), .start_valid(matmul_start_valid),
        .start_ready(matmul_start_ready), .start_operator_tag(matmul_start_operator_tag),
        .start_rows('0), .start_input_features('0),
        .start_output_features('0), .start_segment_count('0),
        .start_segment_mode('0), .start_segment_row_base('0),
        .start_segment_row_count('0), .start_segment_activation_base_byte_offset('0),
        .start_row_enable('0),
        .start_source_ddr(1'b0), .start_source_base_addr('0),
        .start_use_resident_activation(1'b0), .start_qkv_w8_layout(1'b0),
        .start_output_mode('0), .start_operator_has_w8(1'b0),
        .start_weight_base_addr('0), .start_weight_enhancement_addr('0),
        .start_weight_scale_addr('0), .start_output_base('0),
        .start_output_limit('0), .start_output_row_stride('0),
        .fixed_command(matmul_command),
        .fixed_command_rows(row_active_rows),
        .fixed_command_residual_storage_rows(row_residual_storage_rows),
        .fixed_command_row_config(matmul_row_config),
        .fixed_command_qkv_select(qkv_matmul_qkv_select),
        .fixed_command_qkv_head(qkv_matmul_head),
        .fixed_command_qkv_use_resident_activation(
            qkv_matmul_use_resident_a8 &&
            (!qkvo_kv_requires_expanded_layout || qkv_group_activation_reuse)),
        .fixed_command_qkv_w8_layout(qkv_matmul_qkv_w8_layout),
        .fixed_weight_config(matmul_weight_config),
        .fixed_attention_context_config(attention_context_config),
        .fixed_attention_context_precomputed_max_disable(attention_pair_enable),
        .fixed_attention_context_max_valid(attention_context_row_max_valid),
        .fixed_attention_context_max_values(attention_context_row_max_values),
        .fixed_ffn_workspace_config(ffn_workspace_config),
        .fixed_ffn_batch_pair(matmul_ffn_batch_pair_config),
        .prefetch_valid(matmul_prefetch_valid),
        .prefetch_ready(matmul_prefetch_ready),
        .prefetch_empty(matmul_prefetch_empty),
        .prefetch_cancel(matmul_prefetch_cancel),
        .prefetch_operator_tag(matmul_prefetch_operator_tag),
        .prefetch_input_features('0), .prefetch_weight_base_addr('0),
        .prefetch_weight_enhancement_addr('0),
        .prefetch_weight_scale_addr('0), .prefetch_operator_has_w8(1'b0),
        .prefetch_qkv_w8_layout(1'b0),
        .fixed_prefetch_command(matmul_prefetch_command),
        .fixed_prefetch_qkv_select(matmul_prefetch_qkv_select),
        .fixed_prefetch_qkv_head(matmul_prefetch_qkv_head),
        .source_req_valid(matmul_source_req_valid),
        .source_req_ready(matmul_local_source_req_ready),
        .source_req_row(matmul_source_req.row),
        .source_req_rows(matmul_source_req.row_count),
        .source_req_element(matmul_source_req.element),
        .source_req_bank_base(matmul_source_req.bank_base),
        .source_rsp_valid(matmul_local_source_rsp_valid),
        .source_rsp_ready(matmul_source_rsp_ready),
        .source_rsp_values(matmul_source_rsp.values),
        .source_rsp_lane_mask(matmul_source_rsp.lane_mask),
        .aux_read_request_valid(matmul_aux_request_valid),
        .aux_read_request_ready(matmul_aux_request_ready),
        .aux_read_request_address(matmul_aux_request.byte_address),
        .aux_read_request_bytes(matmul_aux_request.byte_count),
        .aux_read_request_tag(matmul_aux_request.tag),
        .aux_read_second_span_valid(matmul_aux_read_second_span_valid),
        .aux_read_second_span_address(matmul_aux_read_second_span_address),
        .aux_read_pair_stride(matmul_aux_read_pair_stride),
        .aux_read_pair_count(matmul_aux_read_pair_count),
        .aux_read_request_error(matmul_aux_completion.error),
        .aux_read_data_valid(matmul_aux_data_valid),
        .aux_read_data_ready(matmul_aux_data_ready),
        .aux_read_data(matmul_aux_data.data),
        .aux_read_byte_enable(matmul_aux_data.byte_enable),
        .aux_read_data_last(matmul_aux_data.last),
        .aux_read_data_tag(matmul_aux_data.tag),
        .output_dma_request_valid(matmul_output_request_valid),
        .output_dma_request_ready(matmul_output_request_ready),
        .output_dma_request_address(matmul_output_request.byte_address),
        .output_dma_request_bytes(matmul_output_request.byte_count),
        .output_dma_request_tag(matmul_output_request.tag),
        .output_dma_request_done(matmul_output_completion.done_pulse),
        .output_dma_request_error(matmul_output_completion.error),
        .output_dma_data_valid(matmul_output_data_valid),
        .output_dma_data_ready(matmul_output_data_ready),
        .output_dma_data(matmul_output_data.data),
        .output_dma_byte_enable(matmul_output_data.byte_enable),
        .output_dma_data_last(matmul_output_data.last),
        .residual_local_read_valid(matmul_residual_local_read_issue),
        .residual_local_read_ready(matmul_residual_memory_read_ready),
        .residual_local_read_row(matmul_residual_read_req.row),
        .residual_local_read_channel_chunk(
            matmul_residual_read_req.channel_chunk),
        .residual_local_read_rsp_valid(
            matmul_residual_memory_read_rsp_valid),
        .residual_local_read_rsp_data(matmul_residual_read_rsp.data),
        .dma_req_valid(dma_read_request_valid[DMA_READ_MATMUL]),
        .dma_req_ready(dma_read_request_ready[DMA_READ_MATMUL]),
        .dma_req_addr(dma_read_request[DMA_READ_MATMUL].byte_address),
        .dma_req_bytes(dma_read_request[DMA_READ_MATMUL].byte_count),
        .dma_req_plane(matmul_dma_req_plane), .dma_req_panel(matmul_dma_req_panel),
        .dma_req_last_for_plane(matmul_dma_req_last_for_plane),
        .dma_req_done(dma_read_completion[DMA_READ_MATMUL].done_pulse),
        .dma_req_error(dma_read_completion[DMA_READ_MATMUL].error),
        .dma_rsp_valid(dma_read_data_valid[DMA_READ_MATMUL]),
        .dma_rsp_ready(dma_read_data_ready[DMA_READ_MATMUL]),
        .dma_rsp_data(dma_wide_read_data[DMA_READ_MATMUL].data),
        .dma_rsp_byte_enable(
            dma_wide_read_data[DMA_READ_MATMUL].byte_enable),
        .dma_rsp_error(dma_read_completion[DMA_READ_MATMUL].error),
        .dma_rsp_last(dma_wide_read_data[DMA_READ_MATMUL].last),
        .dma_abort_ack(matmul_dma_reader_abort_ack),
        .activation_writer_abort_request(matmul_activation_writer_abort_request),
        .activation_writer_cfg_valid(matmul_activation_writer_cfg_valid),
        .activation_writer_cfg_ready(matmul_activation_writer_cfg_ready),
        .activation_writer_cfg_mode(writer_client_cfg[0].mode),
        .activation_writer_cfg_physical_row_base(
            writer_client_cfg[0].physical_row_base),
        .activation_writer_cfg_row_count(writer_client_cfg[0].row_count),
        .activation_writer_cfg_elements_per_row(
            writer_client_cfg[0].elements_per_row),
        .activation_writer_cfg_activation_base_byte_offset(
            writer_client_cfg[0].activation_base_byte_offset),
        .activation_writer_cfg_activation_limit_byte_offset(
            writer_client_cfg[0].activation_limit_byte_offset),
        .activation_writer_cfg_chunk_major(writer_client_cfg[0].mixed_rows),
        .activation_writer_cfg_row_partial(writer_client_cfg[0].row_partial),
        .activation_writer_cfg_segment_count(
            writer_client_cfg[0].segment_count),
        .activation_writer_cfg_segment_mode(writer_client_cfg[0].segment_mode),
        .activation_writer_cfg_segment_row_base(
            writer_client_cfg[0].segment_row_base),
        .activation_writer_cfg_segment_row_count(
            writer_client_cfg[0].segment_row_count),
        .activation_writer_cfg_mixed_group_enable(
            writer_client_cfg[0].mixed_group_enable),
        .activation_writer_cfg_compute_group_count(
            writer_client_cfg[0].compute_group_count),
        .activation_writer_cfg_row_precision_a8(
            writer_client_cfg[0].row_precision_a8),
        .activation_writer_cfg_row_compute_group(
            writer_client_cfg[0].row_compute_group),
        .activation_writer_cfg_row_pe_slot(
            writer_client_cfg[0].row_pe_slot),
        .activation_writer_cfg_row_phase_mask(
            writer_client_cfg[0].row_phase_mask),
        .activation_writer_quantized_valid(matmul_activation_writer_quantized_valid),
        .activation_writer_quantized_ready(matmul_activation_writer_quantized_ready),
        .activation_writer_quantized_row(
            writer_client_values[0].physical_row_base),
        .activation_writer_quantized_element(
            writer_client_values[0].element_base),
        .activation_writer_quantized_values(writer_client_values[0].values),
        .activation_writer_quantized_lane_mask(
            writer_client_values[0].lane_mask),
        .activation_writer_quantized_scale(),
        .activation_writer_scale_valid(matmul_activation_writer_scale_valid),
        .activation_writer_scale_ready(matmul_activation_writer_scale_ready),
        .activation_writer_scale_row_base(
            writer_client_scale[0].row_base),
        .activation_writer_scale_row_mask(
            writer_client_scale[0].row_mask),
        .activation_writer_scale_values(writer_client_scale[0].values_bf16),
        .activation_writer_done_pulse(shared_activation_writer_done_pulse),
        .activation_writer_error(shared_activation_writer_error),
        .activation_scale_valid(shared_activation_scale_valid && !post_active),
        .activation_scale_ready(block_activation_scale_ready),
        .activation_scale_row_base(shared_activation_scale_row_base),
        .activation_scale_row_mask(shared_activation_scale_row_mask),
        .activation_scale_values(shared_activation_scale_values),
        .resident_activation_commit_valid(elementwise_done_valid &&
            elementwise_done_ready && !elementwise_error),
        .resident_activation_commit_rows(row_active_rows),
        .resident_activation_commit_features(16'(FFN_FEATURES)),
        .resident_activation_commit_segment_count(
            elementwise_completed_layout.segment_count),
        .resident_activation_commit_segment_mode(
            elementwise_completed_layout.segment_mode),
        .resident_activation_commit_segment_row_base(
            elementwise_completed_layout.segment_row_base),
        .resident_activation_commit_segment_row_count(
            elementwise_completed_layout.segment_row_count),
        .resident_activation_commit_segment_activation_base_byte_offset(
            elementwise_completed_layout.segment_activation_base_byte_offset),
        .resident_activation_commit_row_enable(
            elementwise_completed_layout.row_enable),
        .panel_write_accepted(), .panel_write_panel(),
        .panel_write_plane(), .panel_write_start_bank(), .panel_write_word_row(),
        .panel_write_data(), .panel_write_byte_enable(),
        .accum_result_valid(matmul_accum_result_valid), .accum_result_ready(1'b1),
        .accum_result_accumulators(matmul_accum_result_accumulators),
        .accum_result_mask(matmul_accum_result_mask),
        .accum_result_mode(matmul_accum_result_mode),
        .accum_result_last_k_step(matmul_accum_result_last_k_step),
        .accum_result_tag(matmul_accum_result_tag),
        .output_write_spill(matmul_output_controller_write_spill),
        .output_write_physical_row(matmul_output_controller_write_physical_row),
        .shared_bf16_abort_request(matmul_bf16_abort_request),
        .shared_bf16_abort_ack(resource_bf16_abort_ack[1]),
        .shared_bf16_req_valid(matmul_bf16_req_valid),
        .shared_bf16_req_ready(matmul_bf16_req_ready),
        .shared_bf16_req_operation(bf16_client_req[1].operation),
        .shared_bf16_req_values(bf16_client_req[1].values[255:0]),
        .shared_bf16_req_paired_values(
            bf16_client_req[1].paired_values[255:0]),
        .shared_bf16_req_factor0_values(
            bf16_client_req[1].factor0_values[255:0]),
        .shared_bf16_req_factor1_values(
            bf16_client_req[1].factor1_values[255:0]),
        .shared_bf16_req_lane_mask(bf16_client_req[1].lane_mask[15:0]),
        .shared_bf16_req_tag(bf16_client_req[1].tag),
        .shared_bf16_rsp_valid(matmul_bf16_rsp_valid),
        .shared_bf16_rsp_ready(matmul_bf16_rsp_ready),
        .shared_bf16_rsp_values(bf16_client_rsp[1].values[255:0]),
        .shared_bf16_rsp_lane_mask(bf16_client_rsp[1].lane_mask[15:0]),
        .shared_bf16_rsp_tag(bf16_client_rsp[1].tag),
        .done_valid(matmul_done_valid), .done_ready(matmul_done_ready),
        .error(matmul_error),
        .accepted_issue_count(matmul_accepted_issue_count),
        .accepted_result_count(matmul_accepted_result_count),
        .accepted_base_request_count(matmul_base_request_count),
        .accepted_enhancement_request_count(matmul_enhancement_request_count),
        .accepted_scale_request_count(matmul_scale_request_count),
        .accepted_base_request_bytes(matmul_base_request_bytes),
        .accepted_enhancement_request_bytes(matmul_enhancement_request_bytes),
        .accepted_scale_request_bytes(matmul_scale_request_bytes),
        .memory_stage(matmul_memory_stage),
        .shared_tile_abort_request(matmul_tile_abort_request),
        .shared_tile_abort_ack(matmul_tile_abort_ack),
        .shared_tile_req_valid(matmul_tile_req_valid),
        .shared_tile_req_ready(matmul_tile_req_ready),
        .shared_tile_req_mode(pe_client_req[0].mode),
        .shared_tile_req_activation_payload(pe_client_req[0].activation_payload),
        .shared_tile_req_weight_payload(pe_client_req[0].weight_payload),
        .shared_tile_req_row_mask(pe_client_req[0].row_mask),
        .shared_tile_req_k_mask(pe_client_req[0].k_mask),
        .shared_tile_req_col_mask(pe_client_req[0].column_mask),
        .shared_tile_req_first_k_step(pe_client_req[0].first_k_step),
        .shared_tile_req_last_k_step(pe_client_req[0].last_k_step),
        .shared_tile_req_mixed_phase(pe_client_req[0].mixed_phase),
        .shared_tile_req_mixed_phase_first(
            pe_client_req[0].mixed_phase_first),
        .shared_tile_req_mixed_a8_rows(pe_client_req[0].mixed_a8_rows),
        .shared_tile_req_activation_scales(pe_client_req[0].activation_scales),
        .shared_tile_req_weight_scales(pe_client_req[0].weight_scales),
        .shared_tile_req_tag(pe_client_req[0].tag),
        .shared_tile_accum_result_valid(matmul_tile_accum_result_valid),
        .shared_tile_accum_result_ready(matmul_tile_accum_result_ready),
        .shared_tile_accum_result_accumulators(pe_client_rsp[0].accumulators),
        .shared_tile_accum_result_activation_scales(
            pe_client_rsp[0].activation_scales),
        .shared_tile_accum_result_weight_scales(pe_client_rsp[0].weight_scales),
        .shared_tile_accum_result_mask(pe_client_rsp[0].accumulator_mask),
        .shared_tile_accum_result_mode(pe_client_rsp[0].mode),
        .shared_tile_accum_result_last_k_step(pe_client_rsp[0].last_k_step),
        .shared_tile_accum_result_tag(pe_client_rsp[0].tag),
        .shared_rescale_claim_valid(matmul_rescale_claim_valid),
        .shared_rescale_claim_ready(matmul_rescale_claim_ready),
        .shared_rescale_abort_request(matmul_rescale_abort_request),
        .shared_rescale_abort_ack(matmul_rescale_abort_ack),
        .shared_rescale_req_valid(matmul_rescale_req_valid),
        .shared_rescale_req_ready(matmul_rescale_req_ready),
        .shared_rescale_req_accumulators(rescale_client_req[0].accumulators),
        .shared_rescale_req_activation_scales(
            rescale_client_req[0].activation_scales),
        .shared_rescale_req_weight_scales(rescale_client_req[0].weight_scales),
        .shared_rescale_req_qk_scale_bf16(rescale_client_req[0].qk_scale_bf16),
        .shared_rescale_req_rescale_mode(rescale_client_req[0].rescale_mode),
        .shared_rescale_req_lane_mask(rescale_client_req[0].lane_mask),
        .shared_rescale_req_tag(rescale_client_req[0].tag),
        .shared_rescale_rsp_valid(matmul_rescale_rsp_valid),
        .shared_rescale_rsp_ready(matmul_rescale_rsp_ready),
        .shared_rescale_rsp_values(rescale_client_rsp[0].values),
        .shared_rescale_rsp_lane_mask(rescale_client_rsp[0].lane_mask),
        .shared_rescale_rsp_tag(rescale_client_rsp[0].tag),
        .max_req_valid(matmul_max_req_valid), .max_req_ready(matmul_max_req_ready),
        .max_req_values(maximum_client_req[1].values),
        .max_req_lane_mask(maximum_client_req[1].lane_mask),
        .max_req_tag(maximum_client_req[1].tag),
        .max_rsp_valid(matmul_max_rsp_valid), .max_rsp_ready(matmul_max_rsp_ready),
        .max_rsp_values(maximum_client_rsp[1].values),
        .max_rsp_row_mask(maximum_client_rsp[1].row_mask),
        .max_rsp_tag(maximum_client_rsp[1].tag),
        .quant_scale_req_valid(matmul_quant_scale_req_valid),
        .quant_scale_req_ready(matmul_quant_scale_req_ready),
        .quant_scale_req_a4_row_mask(quant_scale_client_req[1].a4_row_mask),
        .quant_scale_req_clip_ratio_bf16(quant_scale_client_req[1].clip_ratio_bf16),
        .quant_scale_req_row_max_abs(quant_scale_client_req[1].row_max_abs),
        .quant_scale_req_row_mask(quant_scale_client_req[1].row_mask),
        .quant_scale_rsp_valid(matmul_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(matmul_quant_scale_rsp_ready),
        .quant_scale_rsp_values(quant_scale_client_rsp[1].values_bf16),
        .quant_values_req_valid(matmul_quant_values_req_valid),
        .quant_values_req_ready(matmul_quant_values_req_ready),
        .quant_values_req_values(quant_values_client_req[1].values_bf16),
        .quant_values_req_lane_mask(quant_values_client_req[1].lane_mask),
        .quant_values_req_tag(quant_values_client_req[1].tag),
        .quant_values_rsp_valid(matmul_quant_values_rsp_valid),
        .quant_values_rsp_ready(matmul_quant_values_rsp_ready),
        .quant_values_rsp_values(quant_values_client_rsp[1].values),
        .quant_values_rsp_lane_mask(quant_values_client_rsp[1].lane_mask),
        .quant_values_rsp_tag(quant_values_client_rsp[1].tag),
        .panel_memory_write_valid(block_panel_write_valid),
        .panel_memory_write_ready(block_panel_write_ready),
        .panel_memory_write_panel(block_panel_write.panel),
        .panel_memory_write_plane(block_panel_write.plane),
        .panel_memory_write_start_bank(block_panel_write.start_bank),
        .panel_memory_write_word_row(block_panel_write.word_row),
        .panel_memory_write_data(block_panel_write.data),
        .panel_memory_write_byte_enable(block_panel_write.byte_enable),
        .read_bundle_valid(block_operand_read_valid),
        .read_bundle_ready(block_operand_read_ready),
        .activation_read_word_index(block_operand_read.activation_word_index),
        .activation_read_word_count(block_operand_read.activation_word_count),
        .activation_read_qkv_padded_layout(block_operand_read.qkv_padded_layout),
        .activation_read_physical_row_base(block_operand_read.physical_row_base),
        .activation_read_explicit_rows(block_operand_read.explicit_activation_rows),
        .activation_read_physical_rows(block_operand_read.activation_physical_row),
        .activation_read_lane_word_index(block_operand_read.activation_lane_word_index),
        .panel_read_select(block_operand_read.panel_select),
        .panel_read_port_mask(block_operand_read.panel_port_mask),
        .panel_read_address(block_operand_read.panel_address),
        .read_response_valid(block_operand_rsp_valid),
        .activation_read_data(block_operand_rsp.activation_data),
        .panel_read_data(block_operand_rsp.panel_data),
        .combine_write_valid(matmul_combine_write_valid),
        .combine_write_ready(matmul_combine_write_ready),
        .combine_write_address(matmul_combine_write.address),
        .combine_write_data(matmul_combine_write.data),
        .combine_write_byte_enable(matmul_combine_write.byte_enable),
        .combine_read_valid(matmul_combine_read_valid),
        .combine_read_ready(matmul_combine_read_ready),
        .combine_read_address(matmul_combine_read_req.address),
        .combine_read_response_valid(matmul_combine_read_response_valid),
        .combine_read_data(matmul_combine_read_rsp.data),
        .local_output_write_valid(matmul_local_output_write_valid),
        .local_output_write_ready(matmul_local_output_write_ready),
        .local_output_write_address(matmul_local_output_write.byte_address),
        .local_output_write_data(matmul_local_output_write.data),
        .local_output_write_byte_enable(matmul_local_output_write.byte_enable)
    );

    assign shared_activation_scale_ready = post_active ? 1'b1 :
        block_activation_scale_ready;

    assign elementwise_abort_request = 1'b0;
    assign elementwise_start_valid = operator_command_valid &&
        pending_start_enable &&
        pending_start_select[COMMAND_START_ELEMENTWISE];

    elementwise_engine #(
        .FFN_FEATURES(FFN_FEATURES),
        .MAX_ROWS(48),
        .FIXED_SCHEDULE_INGRESS(1'b1)
    ) ffn_elementwise (
        .clk(clk), .rst(rst), .abort_request(elementwise_abort_request),
        .abort_ack(elementwise_abort_ack), .start_valid(elementwise_start_valid),
        .start_ready(elementwise_start_ready),
        .start_precomputed_product(
            ffn_batch_pair_config.fused_product &&
            pending_command.layer_operator_index ==
                LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE),
        .start_r4_enable(matmul_row_config.mixed_group_enable),
        .start_two_token_r4(ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.compact_a4_layout),
        .start_active_rows(row_active_rows),
        .start_segment_count('0), .start_segment_mode('0),
        .start_segment_row_base('0), .start_segment_row_count('0),
        .start_segment_activation_base_byte_offset('0), .fixed_row_config(matmul_row_config),
        .start_activation_limit_byte_offset(ffn_batch_pair_config.enable ? 32'd589824 :
            32'(FFN_FEATURES * 32)),
        .start_activation_base_byte_offset(ffn_batch_pair_config.prepare_second ?
            ffn_batch_pair_config.second_activation_base_byte_offset : 32'd0),
        .start_workspace_base(ffn_workspace_config.gate_up_base),
        .start_workspace_limit(ffn_workspace_config.gate_up_limit),
        .busy(elementwise_busy), .done_valid(elementwise_done_valid),
        .done_ready(elementwise_done_ready),
        .completed_layout(elementwise_completed_layout),
        .read_phase_complete(elementwise_read_phase_complete),
        .down_prefetch_ready(elementwise_down_prefetch_ready),
        .error(elementwise_error), .error_id(elementwise_error_id),
        .read_request_valid(dma_read_request_valid[DMA_READ_ELEMENTWISE]),
        .read_request_ready(dma_read_request_ready[DMA_READ_ELEMENTWISE]),
        .read_request_address(
            dma_read_request[DMA_READ_ELEMENTWISE].byte_address),
        .read_request_bytes(dma_read_request[DMA_READ_ELEMENTWISE].byte_count),
        .read_request_tag(dma_read_request[DMA_READ_ELEMENTWISE].tag),
        .read_second_span_valid(elementwise_read_second_span_valid),
        .read_second_span_address(elementwise_read_second_span_address),
        .read_pair_stride(elementwise_read_pair_stride),
        .read_pair_count(elementwise_read_pair_count),
        .read_request_done(dma_read_completion[DMA_READ_ELEMENTWISE].done_pulse),
        .read_request_error(dma_read_completion[DMA_READ_ELEMENTWISE].error),
        .read_data_valid(dma_read_data_valid[DMA_READ_ELEMENTWISE]),
        .read_data_ready(dma_read_data_ready[DMA_READ_ELEMENTWISE]),
        .read_data(dma_read_data[DMA_READ_ELEMENTWISE].data),
        .read_byte_enable(dma_read_data[DMA_READ_ELEMENTWISE].byte_enable),
        .read_data_last(dma_read_data[DMA_READ_ELEMENTWISE].last),
        .read_data_tag(dma_read_data[DMA_READ_ELEMENTWISE].tag),
        .read_data_span(elementwise_read_data_span),
        .arithmetic_req_valid(elementwise_bf16_req_valid),
        .arithmetic_req_ready(elementwise_bf16_req_ready),
        .arithmetic_req_operation(bf16_client_req[4].operation),
        .arithmetic_req_values(bf16_client_req[4].values),
        .arithmetic_req_paired_values(bf16_client_req[4].paired_values),
        .arithmetic_req_factor0_values(bf16_client_req[4].factor0_values),
        .arithmetic_req_factor1_values(bf16_client_req[4].factor1_values),
        .arithmetic_req_lane_mask(bf16_client_req[4].lane_mask),
        .arithmetic_req_tag(bf16_client_req[4].tag),
        .arithmetic_rsp_valid(elementwise_bf16_rsp_valid),
        .arithmetic_rsp_ready(elementwise_bf16_rsp_ready),
        .arithmetic_rsp_values(bf16_client_rsp[4].values),
        .arithmetic_rsp_lane_mask(bf16_client_rsp[4].lane_mask),
        .arithmetic_rsp_tag(bf16_client_rsp[4].tag),
        .arithmetic_abort_ack(elementwise_bf16_abort_ack),
        .max_req_valid(elementwise_max_req_valid),
        .max_req_ready(elementwise_max_req_ready),
        .max_req_values(maximum_client_req[3].values),
        .max_req_lane_mask(maximum_client_req[3].lane_mask),
        .max_req_tag(maximum_client_req[3].tag),
        .max_rsp_valid(elementwise_max_rsp_valid),
        .max_rsp_ready(elementwise_max_rsp_ready),
        .max_rsp_values(maximum_client_rsp[3].values),
        .max_rsp_row_mask(maximum_client_rsp[3].row_mask),
        .max_rsp_tag(maximum_client_rsp[3].tag),
        .quant_scale_req_valid(elementwise_quant_scale_req_valid),
        .quant_scale_req_ready(elementwise_quant_scale_req_ready),
        .quant_scale_req_a4_row_mask(elementwise_quant_scale_req_a4_row_mask),
        .quant_scale_req_row_max_abs(
            quant_scale_client_req[3].row_max_abs),
        .quant_scale_req_row_mask(quant_scale_client_req[3].row_mask),
        .quant_scale_rsp_valid(elementwise_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(elementwise_quant_scale_rsp_ready),
        .quant_scale_rsp_values_bf16(
            quant_scale_client_rsp[3].values_bf16),
        .quant_values_req_valid(elementwise_quant_values_req_valid),
        .quant_values_req_ready(elementwise_quant_values_req_ready),
        .quant_values_req_values_bf16(quant_values_client_req[3].values_bf16),
        .quant_values_req_lane_mask(quant_values_client_req[3].lane_mask),
        .quant_values_req_tag(quant_values_client_req[3].tag),
        .quant_values_rsp_valid(elementwise_quant_values_rsp_valid),
        .quant_values_rsp_ready(elementwise_quant_values_rsp_ready),
        .quant_values_rsp_values(quant_values_client_rsp[3].values),
        .quant_values_rsp_lane_mask(quant_values_client_rsp[3].lane_mask),
        .quant_values_rsp_tag(quant_values_client_rsp[3].tag),
        .quant_abort_ack(elementwise_quant_abort_ack),
        .writer_cfg_valid(elementwise_writer_cfg_valid),
        .writer_cfg_ready(elementwise_writer_cfg_ready),
        .writer_cfg_mode(writer_client_cfg[1].mode),
        .writer_cfg_physical_row_base(writer_client_cfg[1].physical_row_base),
        .writer_cfg_row_count(writer_client_cfg[1].row_count),
        .writer_cfg_elements_per_row(writer_client_cfg[1].elements_per_row),
        .writer_cfg_activation_base_byte_offset(writer_client_cfg[1].activation_base_byte_offset),
        .writer_cfg_activation_limit_byte_offset(writer_client_cfg[1].activation_limit_byte_offset),
        .writer_cfg_mixed_group_enable(
            writer_client_cfg[1].mixed_group_enable),
        .writer_cfg_compute_group_count(
            writer_client_cfg[1].compute_group_count),
        .writer_cfg_row_precision_a8(writer_client_cfg[1].row_precision_a8),
        .writer_cfg_row_compute_group(writer_client_cfg[1].row_compute_group),
        .writer_cfg_row_pe_slot(writer_client_cfg[1].row_pe_slot),
        .writer_cfg_row_phase_mask(writer_client_cfg[1].row_phase_mask),
        .writer_scale_valid(elementwise_writer_scale_valid),
        .writer_scale_ready(elementwise_writer_scale_ready),
        .writer_scale_row_base(writer_client_scale[1].row_base),
        .writer_scale_row_mask(writer_client_scale[1].row_mask),
        .writer_scale_values_bf16(writer_client_scale[1].values_bf16),
        .writer_quantized_valid(elementwise_writer_quantized_valid),
        .writer_quantized_ready(elementwise_writer_quantized_ready),
        .writer_quantized_row_base(
            writer_client_values[1].physical_row_base),
        .writer_quantized_element_base(writer_client_values[1].element_base),
        .writer_quantized_values(writer_client_values[1].values),
        .writer_quantized_lane_mask(writer_client_values[1].lane_mask),
        .writer_quantized_tag(writer_client_values[1].tag),
        .writer_done_pulse(elementwise_writer_done_pulse),
        .writer_error(elementwise_writer_error),
        .writer_abort_request(elementwise_writer_abort_request),
        .product_memory_req_valid(elementwise_product_req_valid),
        .product_memory_req_ready(elementwise_product_req_ready),
        .product_memory_req(elementwise_product_req),
        .product_memory_rsp_valid(elementwise_product_rsp_valid),
        .product_memory_rsp_ready(elementwise_product_rsp_ready),
        .product_memory_rsp(elementwise_product_rsp),
        .trace_sample_valid(elementwise_trace_sample_valid),
        .trace_sample_ready(1'b1),
        .trace_sample_pass(elementwise_trace_sample_pass),
        .trace_sample_stripe(elementwise_trace_sample_stripe),
        .trace_sample_row_base(elementwise_trace_sample_row_base),
        .trace_sample_lane_mask(elementwise_trace_sample_lane_mask),
        .trace_sample_silu(elementwise_trace_sample_silu),
        .trace_sample_product(elementwise_trace_sample_product),
        .accepted_read_request_count(), .accepted_read_byte_count(),
        .accepted_arithmetic_request_count(), .completed_product_tile_count(),
        .accepted_max_request_count(), .accepted_quant_scale_count(),
        .accepted_quantized_value_count()
    );

    rope_engine #(.MAX_ROWS(48)) qkv_rope (
        .clk(clk), .rst(rst), .abort_request(qkv_rope_abort_request),
        .abort_ack(qkv_rope_abort_ack), .start_valid(qkv_rope_start_valid),
        .start_ready(qkv_rope_start_ready), .start_row_count(qkv_active_token_count),
        .start_head_count(6'd1), .start_head_dim(9'd128),
        .start_source_is_k(qkv_rope_source_is_k),
        .start_token_position(qkv_pair_active && qkv_second_batch ?
            kv_pair_second_config.token_position : row_token_position), .start_source_base(18'd0),
        .start_source_row_stride(18'd128), .start_destination_base(18'd0),
        .start_destination_row_stride(18'd128),
        .start_cos_lut_base(qkv_layer_config.rope_cos_base),
        .start_sin_lut_base(qkv_layer_config.rope_sin_base),
        .busy(rope_busy), .done_pulse(qkv_rope_done_pulse),
        .error(qkv_rope_error),
        .error_id(), .source_read_req_valid(rope_source_read_req_valid),
        .source_read_req_ready(rope_source_read_req_ready),
        .source_read_is_k(rope_source_read_is_k),
        .source_read_lane_mask(rope_source_read_lane_mask),
        .source_read_lane_address(rope_source_read_req.lane_address),
        .source_read_rsp_valid(rope_source_read_rsp_valid),
        .source_read_rsp_ready(rope_source_read_rsp_ready),
        .source_read_rsp_lane_mask(rope_source_read_rsp.lane_mask),
        .source_read_rsp_lane_data(rope_source_read_rsp.data),
        .cos_read_req_valid(rope_cos_read_req_valid),
        .cos_read_req_ready(rope_cos_read_req_ready),
        .cos_read_req_address(rope_cos_read_req_address),
        .cos_read_req_bytes(rope_cos_read_req_bytes),
        .constant_read_physical_row(rope_constant_read_req.physical_row),
        .constant_read_pair_base(rope_constant_read_req.pair_base),
        .cos_read_rsp_valid(rope_cos_read_rsp_valid),
        .cos_read_rsp_ready(rope_cos_read_rsp_ready),
        .cos_read_rsp_lane_mask(rope_cos_read_rsp.lane_mask),
        .cos_read_rsp_lane_data(rope_cos_read_rsp.data),
        .sin_read_req_valid(rope_sin_read_req_valid),
        .sin_read_req_ready(rope_sin_read_req_ready),
        .sin_read_req_address(rope_sin_read_req_address),
        .sin_read_req_bytes(rope_sin_read_req_bytes),
        .sin_read_rsp_valid(rope_sin_read_rsp_valid),
        .sin_read_rsp_ready(rope_sin_read_rsp_ready),
        .sin_read_rsp_lane_mask(rope_sin_read_rsp.lane_mask),
        .sin_read_rsp_lane_data(rope_sin_read_rsp.data),
        .destination_write_valid(rope_destination_write_valid),
        .destination_write_ready(rope_destination_write_ready),
        .destination_write_is_k(rope_destination_write_is_k),
        .destination_write_lane_mask(rope_destination_write_lane_mask),
        .destination_write_lane_address(rope_destination_write.lane_address),
        .destination_write_lane_data(rope_destination_write.lane_data),
        .trace_sample_valid(), .trace_sample_ready(1'b1), .trace_sample_physical_row(),
        .trace_sample_head(), .trace_sample_pair_base(),
        .trace_sample_token_position(), .trace_sample_data(),
        .trace_sample_byte_enable(),
        .arithmetic_req_valid(rope_arithmetic_req_valid),
        .arithmetic_req_ready(rope_arithmetic_req_ready),
        .arithmetic_req_operation(bf16_client_req[2].operation),
        .arithmetic_req_values(bf16_client_req[2].values[255:0]),
        .arithmetic_req_paired_values(
            bf16_client_req[2].paired_values[255:0]),
        .arithmetic_req_factor0_values(
            bf16_client_req[2].factor0_values[255:0]),
        .arithmetic_req_factor1_values(
            bf16_client_req[2].factor1_values[255:0]),
        .arithmetic_req_lane_mask(bf16_client_req[2].lane_mask[15:0]),
        .arithmetic_req_tag(bf16_client_req[2].tag),
        .arithmetic_rsp_valid(rope_arithmetic_rsp_valid),
        .arithmetic_rsp_ready(rope_arithmetic_rsp_ready),
        .arithmetic_rsp_values(bf16_client_rsp[2].values[255:0]),
        .arithmetic_rsp_lane_mask(bf16_client_rsp[2].lane_mask[15:0]),
        .arithmetic_rsp_tag(bf16_client_rsp[2].tag),
        .accepted_source_request_count(), .accepted_cos_request_count(),
        .accepted_sin_request_count(), .accepted_arithmetic_request_count(),
        .accepted_destination_write_count(), .accepted_trace_sample_count(),
        .completed_chunk_count(), .completed_command_count()
    );

    kv_cache_access_controller #(
        .MAX_CURRENT_ROWS(48),
        .TRUSTED_START_CONFIGURATION(1'b1)
    ) cache_state (
        .clk(clk), .rst(rst), .abort_request(attention_cache_lookup_abort_request),
        .start_pair_enable(kv_pair_enable),
        .start_second_rows(ffn_batch_pair_config.second_rows),
        .start_second_kv_write_disable(kv_pair_second_config.kv_write_disable),
        .start_second_token_position(kv_pair_second_config.token_position),
        .start_batch_group(kv_batch_group_config),
        .current_write_second_batch(qkv_pair_active && qkv_second_batch),
        .current_write_batch_index(qkv_pair_active ? qkv_batch_index : 3'd0),
        .abort_ack(cache_lookup_abort_ack), .start_valid(cache_start_valid),
        .start_ready(cache_start_ready), .start_current_rows(row_active_rows),
        .start_token_batch_index(attention_row_config.token_batch_index),
        .start_kv_write_disable(attention_row_config.kv_write_disable),
        .start_token_position(row_token_position),
        .start_current_k_base(attention_cache_config.current_k_base),
        .start_current_v_base(attention_cache_config.current_v_base),
        .start_current_k_scale_base(attention_cache_config.k_scale_base),
        .start_retained_k_base(attention_cache_config.retained_k_base),
        .start_retained_v_base(attention_cache_config.retained_v_base),
        .start_retained_k_scale_base(attention_cache_config.retained_k_scale_base != 0 ?
            attention_cache_config.retained_k_scale_base : attention_cache_config.k_scale_base),
        .start_head_stride({32'd0, attention_cache_config.head_stride_bytes}),
        .start_token_stride({32'd0, attention_cache_config.token_stride_bytes}),
        .start_k_scale_head_stride(
            {32'd0, attention_cache_config.k_scale_head_stride_bytes}),
        .config_valid(cache_config_valid), .config_release(cache_config_release),
        .config_done_pulse(cache_config_done_pulse), .error(cache_error),
        .error_id(cache_error_id),
        .current_write_valid(cache_current_write_valid),
        .current_write_ready(cache_current_write_ready),
        .current_write_head(qkv_current_write_head),
        .current_write_physical_row(qkv_current_write_row),
        .current_write_logical_slot(qkv_current_write_slot),
        .current_write_chunk(qkv_current_write_chunk),
        .current_write_accepted(cache_current_write_accepted),
        .current_k_write_address(cache_current_k_write_address),
        .current_v_write_address(cache_current_v_write_address),
        .current_k_scale_write_address(cache_current_k_scale_write_address),
        .current_write_physical_tag(), .current_write_logical_tag(),
        .read_req_valid(attention_cache_lookup_req_valid),
        .read_req_ready(attention_cache_lookup_req_ready),
        .read_req_head(attention_cache_lookup_req_head),
        .read_req_key_group(attention_cache_lookup_req_key_group),
        .read_req_chunk(attention_cache_lookup_req_chunk),
        .read_req_tag(attention_cache_lookup_req_tag),
        .read_rsp_valid(attention_cache_lookup_rsp_valid),
        .read_rsp_ready(attention_cache_lookup_rsp_ready),
        .read_rsp_key_group(attention_cache_lookup_rsp_key_group),
        .read_rsp_current_mask(attention_cache_lookup_rsp_current_mask),
        .read_rsp_retained_mask(attention_cache_lookup_rsp_retained_mask),
        .read_rsp_current_k_address(attention_cache_lookup_rsp_current_k_address),
        .read_rsp_current_v_address(attention_cache_lookup_rsp_current_v_address),
        .read_rsp_current_k_scale_address(
            attention_cache_lookup_rsp_current_k_scale_address),
        .read_rsp_retained_k_address(attention_cache_lookup_rsp_retained_k_address),
        .read_rsp_retained_v_address(attention_cache_lookup_rsp_retained_v_address),
        .read_rsp_retained_k_scale_address(
            attention_cache_lookup_rsp_retained_k_scale_address),
        .read_rsp_tag(attention_cache_lookup_rsp_tag), .accepted_current_write_count(),
        .accepted_read_request_count(), .completed_read_response_count(),
        .retained_write_count()
    );

    assign attention_operand_req.sequence_length = row_sequence_length;
    assign attention_score_read_req.active_row_count = row_active_rows;
    assign attention_score_read_req.sequence_length = row_sequence_length;

    assign attention_score_write.sequence_length =
        row_sequence_length;

    attention_controller attention (
        .start_pair_enable(attention_pair_enable),
        .start_output_subset_enable(attention_output_subset_enable),
        .start_output_query_count(attention_output_token_count),
        .start_second_query_count(attention_pair_second_rows),
        .start_third_enable(attention_third_batch_enable),
        .start_third_query_count(attention_third_rows),
        .layout_ready(current_layout == accepted_layout),
        .active_second_batch(attention_active_second_batch),
        .active_batch_index(attention_active_batch_index),
        .pair_active(attention_pair_active), .pair_score_words(attention_pair_score_words),
        .start_probability_config(attention_probability_config),
        .start_token_batch_index(attention_row_config.token_batch_index), .start_layer(operator_command_layer),
        .clk(clk), .rst(rst), .abort_request(attention_abort_request),
        .abort_ack(attention_abort_ack), .start_valid(attention_start_valid),
        .start_ready(attention_start_ready), .start_query_count(row_active_rows),
        .start_token_position(row_token_position),
        .start_sequence_length(row_sequence_length),
        .start_head_count(6'(ATTENTION_HEAD_COUNT)),
        .start_q_group_use_staged(ffn_batch_pair_config.enable &&
            ffn_batch_pair_config.qkvo_group &&
            ffn_batch_pair_config.compact_a4_layout),
        .start_q_group_precompute(qkvo_query_group_precompute),
        .start_context_max_enable(matmul_row_config.mixed_group_enable),
        .start_context_base(attention_context_config.context_base),
        .start_context_query_stride(
            attention_context_config.context_row_stride_bytes),
        .start_tag(16'h4702), .busy(attention_busy),
        .done_valid(attention_done_valid), .done_ready(attention_done_ready),
        .error(attention_error), .error_id(attention_error_id),
        .context_row_max_valid(attention_context_row_max_valid),
        .context_row_max_values(attention_context_row_max_values),
        .local_memory_phase(attention_local_phase),
        .q_head_start_valid(attention_q_head_start_valid),
        .q_head_start_ready(attention_q_head_start_ready),
        .q_head_start_head(attention_q_head_start_head),
        .q_head_start_slot(attention_q_head_start_slot),
        .q_head_postprocess_enable(attention_q_head_postprocess_enable),
        .q_head_precompute_only(attention_q_head_precompute_only),
        .q_head_use_staged(attention_q_head_use_staged),
        .q_head_matmul_done(attention_q_head_matmul_done),
        .q_head_done_valid(attention_q_head_done_valid),
        .q_head_done_ready(attention_q_head_done_ready),
        .q_head_error(attention_q_head_error),
        .q_head_error_id(attention_q_head_error_id),
        .q_head_abort_request(attention_q_head_abort_request),
        .q_head_abort_ack(attention_q_head_abort_ack),
        .q_head_active(attention_q_head_active),
        .q_head_slot(attention_q_head_slot),
        .attention_read_source(attention_read_source),
        .dma_read_stream_idle(dma_read_stream_idle),
        .cache_lookup_req_valid(attention_cache_lookup_req_valid),
        .cache_lookup_req_ready(attention_cache_lookup_req_ready),
        .cache_lookup_req_head(attention_cache_lookup_req_head),
        .cache_lookup_req_key_group(attention_cache_lookup_req_key_group),
        .cache_lookup_req_chunk(attention_cache_lookup_req_chunk),
        .cache_lookup_req_tag(attention_cache_lookup_req_tag),
        .cache_lookup_rsp_valid(attention_cache_lookup_rsp_valid),
        .cache_lookup_rsp_ready(attention_cache_lookup_rsp_ready),
        .cache_lookup_rsp_key_group(attention_cache_lookup_rsp_key_group),
        .cache_lookup_rsp_current_mask(attention_cache_lookup_rsp_current_mask),
        .cache_lookup_rsp_retained_mask(attention_cache_lookup_rsp_retained_mask),
        .cache_lookup_rsp_current_k_address(
            attention_cache_lookup_rsp_current_k_address),
        .cache_lookup_rsp_current_v_address(
            attention_cache_lookup_rsp_current_v_address),
        .cache_lookup_rsp_current_k_scale_address(
            attention_cache_lookup_rsp_current_k_scale_address),
        .cache_lookup_rsp_retained_k_address(
            attention_cache_lookup_rsp_retained_k_address),
        .cache_lookup_rsp_retained_v_address(
            attention_cache_lookup_rsp_retained_v_address),
        .cache_lookup_rsp_retained_k_scale_address(
            attention_cache_lookup_rsp_retained_k_scale_address),
        .cache_lookup_rsp_tag(attention_cache_lookup_rsp_tag),
        .cache_lookup_abort_request(attention_cache_lookup_abort_request),
        .cache_lookup_abort_ack(cache_lookup_abort_ack),
        .cache_dma_request_valid(
            dma_read_request_valid[DMA_READ_ATTENTION]),
        .cache_dma_request_ready(
            dma_read_request_ready[DMA_READ_ATTENTION]),
        .cache_dma_request_address(
            dma_read_request[DMA_READ_ATTENTION].byte_address),
        .cache_dma_request_bytes(
            dma_read_request[DMA_READ_ATTENTION].byte_count),
        .cache_dma_request_tag(dma_read_request[DMA_READ_ATTENTION].tag),
        .cache_dma_request_error(
            dma_read_completion[DMA_READ_ATTENTION].error),
        .cache_dma_read_valid(dma_read_data_valid[DMA_READ_ATTENTION]),
        .cache_dma_read_ready(dma_read_data_ready[DMA_READ_ATTENTION]),
        .cache_dma_read_data(dma_wide_read_data[DMA_READ_ATTENTION].data),
        .cache_dma_read_byte_enable(
            dma_wide_read_data[DMA_READ_ATTENTION].byte_enable),
        .cache_dma_read_last(dma_wide_read_data[DMA_READ_ATTENTION].last),
        .cache_dma_read_tag(dma_wide_read_data[DMA_READ_ATTENTION].tag),
        .cache_panel_write_valid(attention_panel_write_valid),
        .cache_panel_write_ready(attention_panel_write_ready),
        .cache_panel_write_target_panel(attention_panel_write.target_panel),
        .cache_panel_write_target_quarter(
            attention_panel_write.target_quarter),
        .cache_panel_write_bank(attention_panel_write.bank),
        .cache_panel_write_address(attention_panel_write.address),
        .cache_panel_write_data(attention_panel_write.data),
        .cache_panel_write_byte_enable(attention_panel_write.byte_enable),
        .cache_panel_write_tag(attention_panel_write_tag),
        .cache_scale_write_valid(attention_scale_write_valid),
        .cache_scale_write_ready(attention_scale_write_ready),
        .cache_scale_write_target_panel(attention_panel_scale_write.target_panel),
        .cache_scale_write_token_base(attention_panel_scale_write.token_base),
        .cache_scale_write_data(attention_panel_scale_write.data),
        .cache_scale_write_byte_enable(attention_panel_scale_write.byte_enable),
        .cache_scale_write_tag(attention_scale_write_tag),
        .operand_req_valid(attention_operand_req_valid),
        .operand_req_ready(attention_operand_req_ready),
        .operand_req_is_pv(attention_operand_req.is_pv),
        .operand_req_query_base(attention_operand_req.query_base),
        .operand_req_output_base(attention_operand_req.output_base),
        .operand_req_k_base(attention_operand_req.k_base),
        .operand_active_head(attention_operand_req.active_head),
        .operand_active_q_slot(attention_operand_req.q_slot),
        .operand_active_panel(attention_operand_req.active_panel),
        .operand_active_quarter(attention_operand_req.active_quarter),
        .operand_v_static_scale(qkv_v_static_scale),
        .operand_rsp_valid(attention_operand_rsp_valid),
        .operand_rsp_activation_rows(attention_operand_rsp.activation_rows),
        .operand_rsp_panel_rows(attention_operand_rsp.panel_rows),
        .operand_rsp_metadata(attention_operand_rsp.metadata),
        .operand_rsp_panel_metadata(attention_operand_rsp.panel_metadata),
        .operand_abort_request(attention_operand_abort_request),
        .operand_abort_ack(attention_operand_abort_ack),
        .pe_req_valid(attention_pe_req_valid), .pe_req_ready(attention_pe_req_ready),
        .pe_req_mode(pe_client_req[1].mode),
        .pe_req_activation_payload(pe_client_req[1].activation_payload),
        .pe_req_weight_payload(pe_client_req[1].weight_payload),
        .pe_req_row_mask(pe_client_req[1].row_mask),
        .pe_req_k_mask(pe_client_req[1].k_mask),
        .pe_req_col_mask(pe_client_req[1].column_mask),
        .pe_req_first_k_step(pe_client_req[1].first_k_step),
        .pe_req_last_k_step(pe_client_req[1].last_k_step),
        .pe_req_activation_scales(pe_client_req[1].activation_scales),
        .pe_req_weight_scales(pe_client_req[1].weight_scales),
        .pe_req_tag(pe_client_req[1].tag),
        .pe_accum_result_valid(attention_pe_accum_result_valid),
        .pe_accum_result_ready(attention_pe_accum_result_ready),
        .pe_accum_result_accumulators(pe_client_rsp[1].accumulators),
        .pe_accum_result_mask(pe_client_rsp[1].accumulator_mask),
        .pe_accum_result_mode(pe_client_rsp[1].mode),
        .pe_accum_result_last_k_step(pe_client_rsp[1].last_k_step),
        .pe_accum_result_activation_scales(pe_client_rsp[1].activation_scales),
        .pe_accum_result_weight_scales(pe_client_rsp[1].weight_scales),
        .pe_accum_result_tag(pe_client_rsp[1].tag),
        .pe_abort_request(attention_pe_abort_request),
        .pe_abort_ack(attention_pe_abort_ack),
        .rescale_req_valid(attention_rescale_req_valid),
        .rescale_req_ready(attention_rescale_req_ready),
        .rescale_req_accumulators(rescale_client_req[1].accumulators),
        .rescale_req_activation_scales(rescale_client_req[1].activation_scales),
        .rescale_req_weight_scales(rescale_client_req[1].weight_scales),
        .rescale_req_qk_scale_bf16(rescale_client_req[1].qk_scale_bf16),
        .rescale_req_rescale_mode(rescale_client_req[1].rescale_mode),
        .rescale_req_lane_mask(rescale_client_req[1].lane_mask),
        .rescale_req_tag(rescale_client_req[1].tag),
        .rescale_rsp_valid(attention_rescale_rsp_valid),
        .rescale_rsp_ready(attention_rescale_rsp_ready),
        .rescale_rsp_values(rescale_client_rsp[1].values),
        .rescale_rsp_lane_mask(rescale_client_rsp[1].lane_mask),
        .rescale_rsp_tag(rescale_client_rsp[1].tag),
        .rescale_abort_request(attention_rescale_abort_request),
        .rescale_abort_ack(attention_rescale_abort_ack),
        .score_write_valid(attention_score_write_valid),
        .score_write_ready(attention_score_write_ready),
        .score_write_head(attention_score_write_head),
        .score_write_query_base(attention_score_write.row_base),
        .score_write_key_base(attention_score_write.key_base),
        .score_write_values(attention_score_write.values),
        .score_write_lane_mask(attention_score_write.lane_mask),
        .score_write_tag(attention_score_write_tag),
        .context_buffer_write_valid(attention_context_buffer_write_valid),
        .context_buffer_write_ready(attention_context_buffer_write_ready),
        .context_buffer_write_address(attention_context_write.address),
        .context_buffer_write_data(attention_context_write.data),
        .context_buffer_write_byte_enable(
            attention_context_write.byte_enable),
        .context_buffer_write_tag(attention_context_buffer_write_tag),
        .softmax_score_read_req_valid(attention_softmax_score_req_valid),
        .softmax_score_read_req_ready(attention_softmax_score_req_ready),
        .softmax_score_read_pass(attention_softmax_score_pass),
        .softmax_score_read_group_base(attention_softmax_score_group_base),
        .softmax_score_read_head(attention_softmax_score_head),
        .softmax_score_read_row_base(attention_score_read_req.row_base),
        .softmax_score_read_key(attention_score_read_req.key),
        .softmax_score_read_tag(attention_softmax_score_tag),
        .softmax_score_read_slot_ready(attention_softmax_score_slot_ready),
        .softmax_score_memory_rsp_valid(
            attention_softmax_score_memory_rsp_valid),
        .softmax_score_memory_rsp_values(
            attention_score_read_rsp.values),
        .softmax_probability_write_valid(attention_probability_write_valid),
        .softmax_probability_write_ready(attention_probability_write_ready),
        .softmax_probability_write_probability(
            attention_probability_write_probability),
        .softmax_probability_write_group_base(
            attention_probability_write_group_base),
        .softmax_probability_write_head(attention_probability_write_head),
        .softmax_probability_write_row_base(
            attention_probability_write.row_base),
        .softmax_probability_write_key(attention_probability_write.key),
        .softmax_probability_write_lane_mask(
            attention_probability_write.lane_mask),
        .softmax_probability_write_values(attention_probability_write.values),
        .softmax_probability_write_tag(attention_probability_write_tag),
        .softmax_quantized_write_valid(attention_probability_quantized_write_valid),
        .softmax_quantized_write_ready(attention_probability_quantized_write_ready),
        .softmax_quantized_write_head(attention_probability_quantized_write_head),
        .softmax_quantized_write_row_base(attention_probability_quantized_write.row_base),
        .softmax_quantized_write_key(attention_probability_quantized_write.key),
        .softmax_quantized_write_values(attention_probability_quantized_write.values),
        .softmax_quantized_write_lane_mask(
            attention_probability_quantized_write.lane_mask),
        .softmax_quantized_write_tag(attention_probability_quantized_write_tag),
        .softmax_scale_write_valid(attention_probability_scale_write_valid),
        .softmax_scale_write_ready(attention_probability_scale_write_ready),
        .softmax_scale_write_head(attention_probability_scale_write.head),
        .softmax_scale_write_row_base(
            attention_probability_scale_write.row_base),
        .softmax_scale_write_values(attention_probability_scale_write.values),
        .softmax_scale_write_row_mask(
            attention_probability_scale_write.row_mask),
        .softmax_arithmetic_req_valid(attention_softmax_bf16_req_valid),
        .softmax_arithmetic_req_ready(attention_softmax_bf16_req_ready),
        .softmax_arithmetic_req_operation(bf16_client_req[3].operation),
        .softmax_arithmetic_req_values(bf16_client_req[3].values),
        .softmax_arithmetic_req_paired_values(
            bf16_client_req[3].paired_values),
        .softmax_arithmetic_req_factor0_values(
            bf16_client_req[3].factor0_values),
        .softmax_arithmetic_req_factor1_values(
            bf16_client_req[3].factor1_values),
        .softmax_arithmetic_req_lane_mask(bf16_client_req[3].lane_mask),
        .softmax_arithmetic_req_tag(bf16_client_req[3].tag),
        .softmax_arithmetic_rsp_valid(attention_softmax_bf16_rsp_valid),
        .softmax_arithmetic_rsp_ready(attention_softmax_bf16_rsp_ready),
        .softmax_arithmetic_rsp_values(bf16_client_rsp[3].values),
        .softmax_arithmetic_rsp_lane_mask(bf16_client_rsp[3].lane_mask),
        .softmax_arithmetic_rsp_tag(bf16_client_rsp[3].tag),
        .softmax_max_req_valid(attention_softmax_max_req_valid),
        .softmax_max_req_ready(attention_softmax_max_req_ready),
        .softmax_max_req_magnitude(maximum_client_req[2].magnitude),
        .softmax_max_req_values(maximum_client_req[2].values),
        .softmax_max_req_lane_mask(maximum_client_req[2].lane_mask),
        .softmax_max_req_tag(maximum_client_req[2].tag),
        .softmax_max_rsp_valid(attention_softmax_max_rsp_valid),
        .softmax_max_rsp_ready(attention_softmax_max_rsp_ready),
        .softmax_max_rsp_values(maximum_client_rsp[2].values),
        .softmax_max_rsp_row_mask(maximum_client_rsp[2].row_mask),
        .softmax_max_rsp_tag(maximum_client_rsp[2].tag),
        .softmax_reduction_req_valid(attention_softmax_reduction_req_valid),
        .softmax_reduction_req_ready(attention_softmax_reduction_req_ready),
        .softmax_reduction_req_values(reduction_client_req[1].values),
        .softmax_reduction_req_lane_mask(
            reduction_client_req[1].lane_mask),
        .softmax_reduction_req_tag(reduction_client_req[1].tag),
        .softmax_reduction_rsp_valid(attention_softmax_reduction_rsp_valid),
        .softmax_reduction_rsp_ready(attention_softmax_reduction_rsp_ready),
        .softmax_reduction_rsp_values(reduction_client_rsp[1].values),
        .softmax_reduction_rsp_row_mask(
            reduction_client_rsp[1].row_mask),
        .softmax_reduction_rsp_tag(reduction_client_rsp[1].tag),
        .softmax_exp_req_valid(attention_softmax_exp_req_valid),
        .softmax_exp_req_ready(attention_softmax_exp_req_ready),
        .softmax_exp_req_delta_values(
            attention_softmax_exp_req.delta_values),
        .softmax_exp_req_lane_mask(attention_softmax_exp_req.lane_mask),
        .softmax_exp_req_tag(attention_softmax_exp_req.tag),
        .softmax_exp_rsp_valid(attention_softmax_exp_rsp_valid),
        .softmax_exp_rsp_ready(attention_softmax_exp_rsp_ready),
        .softmax_exp_rsp_values(attention_softmax_exp_rsp.values),
        .softmax_exp_rsp_lane_mask(attention_softmax_exp_rsp.lane_mask),
        .softmax_exp_rsp_tag(attention_softmax_exp_rsp.tag),
        .softmax_reciprocal_req_valid(
            attention_softmax_reciprocal_req_valid),
        .softmax_reciprocal_req_ready(
            attention_softmax_reciprocal_req_ready),
        .softmax_reciprocal_req_sum_values(
            attention_softmax_reciprocal_req.sum_values),
        .softmax_reciprocal_req_row_mask(
            attention_softmax_reciprocal_req.row_mask),
        .softmax_reciprocal_req_tag(
            attention_softmax_reciprocal_req.tag),
        .softmax_reciprocal_rsp_valid(
            attention_softmax_reciprocal_rsp_valid),
        .softmax_reciprocal_rsp_ready(
            attention_softmax_reciprocal_rsp_ready),
        .softmax_reciprocal_rsp_values(
            attention_softmax_reciprocal_rsp.values),
        .softmax_reciprocal_rsp_row_mask(
            attention_softmax_reciprocal_rsp.row_mask),
        .softmax_reciprocal_rsp_tag(
            attention_softmax_reciprocal_rsp.tag),
        .softmax_quant_scale_req_valid(attention_softmax_quant_scale_req_valid),
        .softmax_quant_scale_req_ready(attention_softmax_quant_scale_req_ready),
        .softmax_quant_scale_req_row_max_abs(
            quant_scale_client_req[2].row_max_abs),
        .softmax_quant_scale_req_row_mask(
            quant_scale_client_req[2].row_mask),
        .softmax_quant_scale_rsp_valid(attention_softmax_quant_scale_rsp_valid),
        .softmax_quant_scale_rsp_ready(attention_softmax_quant_scale_rsp_ready),
        .softmax_quant_scale_rsp_values(
            quant_scale_client_rsp[2].values_bf16),
        .softmax_quant_values_req_valid(attention_softmax_quant_values_req_valid),
        .softmax_quant_values_req_ready(attention_softmax_quant_values_req_ready),
        .softmax_quant_values_req_values(
            quant_values_client_req[2].values_bf16),
        .softmax_quant_values_req_lane_mask(
            quant_values_client_req[2].lane_mask),
        .softmax_quant_values_req_tag(quant_values_client_req[2].tag),
        .softmax_quant_values_rsp_valid(attention_softmax_quant_values_rsp_valid),
        .softmax_quant_values_rsp_ready(attention_softmax_quant_values_rsp_ready),
        .softmax_quant_values_rsp_values(quant_values_client_rsp[2].values),
        .softmax_quant_values_rsp_lane_mask(
            quant_values_client_rsp[2].lane_mask),
        .softmax_quant_values_rsp_tag(quant_values_client_rsp[2].tag),
        .softmax_scratch_read_valid(attention_softmax_scratch_read_valid),
        .softmax_scratch_read_ready(attention_softmax_scratch_read_ready),
        .softmax_scratch_read_bank(attention_softmax_scratch_read_req.bank),
        .softmax_scratch_read_left_address(
            attention_softmax_scratch_read_req.left_address),
        .softmax_scratch_read_right_address(
            attention_softmax_scratch_read_req.right_address),
        .softmax_scratch_read_tag(attention_softmax_scratch_read_req.tag),
        .softmax_scratch_rsp_valid(attention_softmax_scratch_rsp_valid),
        .softmax_scratch_rsp_ready(attention_softmax_scratch_rsp_ready),
        .softmax_scratch_rsp_left_data(
            attention_softmax_scratch_rsp.left_data),
        .softmax_scratch_rsp_right_data(
            attention_softmax_scratch_rsp.right_data),
        .softmax_scratch_rsp_tag(attention_softmax_scratch_rsp.tag),
        .softmax_scratch_write_valid(attention_softmax_scratch_write_valid),
        .softmax_scratch_write_ready(attention_softmax_scratch_write_ready),
        .softmax_scratch_write_bank(attention_softmax_scratch_write.bank),
        .softmax_scratch_write_address(attention_softmax_scratch_write.address),
        .softmax_scratch_write_data(attention_softmax_scratch_write.data),
        .softmax_scratch_write_byte_enable(
            attention_softmax_scratch_write.byte_enable),
        .context_read_req_valid(attention_context_read_req_valid),
        .context_read_req_ready(attention_context_read_req_ready),
        .context_read_req_address(attention_context_read_req.address),
        .context_read_req_tag(attention_context_read_req.tag),
        .context_read_rsp_valid(attention_context_read_rsp_valid),
        .context_read_rsp_ready(attention_context_read_rsp_ready),
        .context_read_rsp_data(attention_context_read_rsp.data),
        .context_read_rsp_tag(attention_context_read_rsp.tag),
        .context_write_request_valid(
            dma_write_request_valid[DMA_WRITE_ATTENTION]),
        .context_write_request_ready(
            dma_write_request_ready[DMA_WRITE_ATTENTION]),
        .context_write_request_address(
            dma_write_request[DMA_WRITE_ATTENTION].byte_address),
        .context_write_request_bytes(
            dma_write_request[DMA_WRITE_ATTENTION].byte_count),
        .context_write_request_tag(
            dma_write_request[DMA_WRITE_ATTENTION].tag),
        .context_write_request_done(
            dma_write_completion[DMA_WRITE_ATTENTION].done_pulse),
        .context_write_request_error(
            dma_write_completion[DMA_WRITE_ATTENTION].error),
        .context_write_data_valid(
            dma_write_data_valid[DMA_WRITE_ATTENTION]),
        .context_write_data_ready(
            dma_write_data_ready[DMA_WRITE_ATTENTION]),
        .context_write_data(dma_write_data[DMA_WRITE_ATTENTION].data),
        .context_write_byte_enable(
            dma_write_data[DMA_WRITE_ATTENTION].byte_enable),
        .context_write_data_last(dma_write_data[DMA_WRITE_ATTENTION].last),
        .completed_head_count(attention_completed_head_count),
        .qk_command_count(attention_qk_count),
        .pv_command_count(attention_pv_count),
        .cache_compute_overlap_cycle_count(),
        .context_compute_overlap_cycle_count()
    );

    assign qkv_quant_abort_ack = resource_quant_abort_ack[0];
    assign elementwise_quant_abort_ack = resource_quant_abort_ack[3];
    assign elementwise_bf16_abort_ack = resource_bf16_abort_ack[4] &&
        elementwise_abort_request;

    assign bf16_client_req[1].values[1023:256] = '0;
    assign bf16_client_req[1].paired_values[1023:256] = '0;
    assign bf16_client_req[1].factor0_values[1023:256] = '0;
    assign bf16_client_req[1].factor1_values[1023:256] = '0;
    assign bf16_client_req[1].lane_mask[63:16] = '0;
    assign bf16_client_req[2].values[1023:256] = '0;
    assign bf16_client_req[2].paired_values[1023:256] = '0;
    assign bf16_client_req[2].factor0_values[1023:256] = '0;
    assign bf16_client_req[2].factor1_values[1023:256] = '0;
    assign bf16_client_req[2].lane_mask[63:16] = '0;
    assign maximum_client_req[0].magnitude = 1'b1;
    assign maximum_client_req[1].magnitude = 1'b1;
    assign maximum_client_req[3].magnitude = 1'b1;
    assign quant_scale_client_req[0].a4_row_mask = 8'h00;
    assign quant_scale_client_req[0].clip_ratio_bf16 = 16'd0;
    assign quant_scale_client_req[2].clip_ratio_bf16 = 16'd0;
    assign quant_scale_client_req[3].clip_ratio_bf16 = matmul_weight_config.ffn_down_clip_ratio_bf16;
    assign quant_scale_client_req[1].use_static_scale = 1'b0;
    assign quant_scale_client_req[1].static_scales_bf16 = '0;
    assign quant_scale_client_req[2].a4_row_mask = 8'h00;
    assign quant_scale_client_req[2].use_static_scale = 1'b0;
    assign quant_scale_client_req[2].static_scales_bf16 = '0;
    assign quant_scale_client_req[3].use_static_scale = 1'b0;
    assign quant_scale_client_req[3].static_scales_bf16 = '0;
    assign quant_scale_client_req[3].a4_row_mask =
        elementwise_quant_scale_req_a4_row_mask;
    assign writer_client_values[0].tag = '0;
    assign writer_client_cfg[1].mixed_rows =
        writer_client_cfg[1].mixed_group_enable;
    assign writer_client_cfg[1].row_partial = 1'b0;
    assign writer_client_cfg[1].segment_count = '0;
    assign writer_client_cfg[1].segment_mode = '0;
    assign writer_client_cfg[1].segment_row_base = '0;
    assign writer_client_cfg[1].segment_row_count = '0;

    assign post_controller_start_valid = execution_forward_postprocess_start_valid && !execution_refresh_select &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !local_layout_update_valid;
    assign execution_forward_postprocess_start_ready = execution_refresh_select ? refresh_start_ready : post_controller_start_ready &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !local_layout_update_valid;
    assign execution_forward_postprocess_done_valid = execution_refresh_select ? refresh_done : post_done;
    assign execution_forward_postprocess_done_error = execution_refresh_select ? refresh_error : post_error;
    assign execution_forward_postprocess_done_error_id = execution_refresh_select ? refresh_error_id : post_error_id;

    assign row_packer_selected_post = row_packer_busy ? row_packer_client_post : post_active;
    assign selected_row_packer_config = row_packer_config[!row_packer_selected_post];
    assign selected_row_packer_row = row_packer_row[!row_packer_selected_post];
    always_ff @(posedge clk) begin
        if (rst) begin
            row_packer_busy <= 1'b0;
            row_packer_client_post <= 1'b0;
        end else if (row_packer_start_valid[!row_packer_selected_post] && row_packer_start_ready) begin
            row_packer_busy <= 1'b1;
            row_packer_client_post <= row_packer_selected_post;
        end else if (row_packer_abort_ack ||
                (row_packer_done_valid && row_packer_done_ready[!row_packer_selected_post]))
            row_packer_busy <= 1'b0;
    end
    token_issue_packer shared_token_issue_packer (
        .clk, .rst, .abort_request(row_packer_abort[!row_packer_selected_post]), .abort_ack(row_packer_abort_ack),
        .start_valid(row_packer_start_valid[!row_packer_selected_post]), .start_ready(row_packer_start_ready),
        .start_row_count(selected_row_packer_config.row_count), .start_sequence_length(selected_row_packer_config.sequence_length),
        .start_token_batch_index(selected_row_packer_config.token_batch_index), .start_first_token_ordinal(selected_row_packer_config.first_token_ordinal),
        .start_metadata_version(selected_row_packer_config.metadata_version), .start_capture_index(selected_row_packer_config.capture_index),
        .row_valid(row_packer_row_valid[!row_packer_selected_post]), .row_ready(row_packer_row_ready),
        .row_index(selected_row_packer_row.index), .row_source_index(selected_row_packer_row.source_index),
        .row_token_position(selected_row_packer_row.token_position), .row_kv_index(selected_row_packer_row.kv_index),
        .row_kv_write_disable(selected_row_packer_row.kv_write_disable),
        .row_embedding_source(selected_row_packer_row.embedding_source), .activation_bits(selected_row_packer_row.bits),
        .row_query_group(selected_row_packer_row.query_group), .row_cache_group(selected_row_packer_row.cache_group),
        .beat_valid(row_packer_beat_valid), .beat_ready(row_packer_beat_ready[!row_packer_selected_post]),
        .beat_data(row_packer_beat_data), .beat_last(row_packer_beat_last),
        .done_valid(row_packer_done_valid), .done_ready(row_packer_done_ready[!row_packer_selected_post]),
        .error(row_packer_error), .error_id(row_packer_error_id),
        .compute_group_count(row_packer_compute_groups), .semantic_group_count(row_packer_semantic_groups)
    );
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (row_packer_start_valid != 2'b11)
                else $error("forward-postprocess and refresh requested the token issue packer together");
            if (row_packer_row_valid[!row_packer_selected_post])
                assert (row_packer_busy)
                    else $error("row packer received rows before its configuration");
        end
    end
`endif

    attention_guided_token_selector refresh (
        .loaded_token_valid(execution_refresh_loaded_token_valid), .loaded_token_position(execution_refresh_loaded_token_position),
        .loaded_token_kv_write(execution_refresh_loaded_token_kv_write),
        .consume_source_a(post_controller_start_valid && post_controller_start_ready),
        .source_a_valid(refresh_source_a_valid), .source_a_mask(refresh_source_a_mask),
        .source_a_block_start(refresh_source_a_block_start), .source_a_capture_index(refresh_source_a_capture_index),
        .bf16_abort_request(refresh_bf16_abort_request), .bf16_abort_ack(refresh_bf16_abort_ack),
        .bf16_req_valid(refresh_bf16_req_valid), .bf16_req_ready(refresh_bf16_req_ready), .bf16_req(refresh_bf16_req),
        .bf16_rsp_valid(refresh_bf16_rsp_valid), .bf16_rsp_ready(refresh_bf16_rsp_ready), .bf16_rsp(resource_bf16_rsp[3]),
        .probability_config(refresh_probability_config),
        .clk, .rst, .abort_request(1'b0), .abort_ack(),
        .packer_abort_request(row_packer_abort[1]), .packer_abort_ack(row_packer_abort_ack && !row_packer_selected_post),
        .packer_start_valid(row_packer_start_valid[1]), .packer_start_ready(row_packer_start_ready && !row_packer_selected_post),
        .packer_config(row_packer_config[1]), .packer_row_valid(row_packer_row_valid[1]),
        .packer_row_ready(row_packer_row_ready && !row_packer_selected_post), .packer_row(row_packer_row[1]),
        .packer_beat_valid(row_packer_beat_valid && !row_packer_selected_post), .packer_beat_ready(row_packer_beat_ready[1]),
        .packer_beat_data(row_packer_beat_data), .packer_beat_last(row_packer_beat_last),
        .packer_done_valid(row_packer_done_valid && !row_packer_selected_post), .packer_done_ready(row_packer_done_ready[1]),
        .packer_error(row_packer_error && !row_packer_selected_post), .packer_error_id(row_packer_error_id),
        .packer_compute_groups(row_packer_compute_groups), .packer_semantic_groups(row_packer_semantic_groups),
        .start_valid(execution_forward_postprocess_start_valid && execution_refresh_select), .start_ready(refresh_start_ready),
        .start_descriptor_address(execution_refresh_configuration_address),
        .start_joint_descriptor_address(execution_joint_configuration_address),
        .start_prepare(execution_refresh_prepare), .prepared_for_forward(refresh_prepared),
        .start_relation_only(execution_refresh_relation_only), .start_relation_l31(execution_refresh_relation_l31), .start_closeout(execution_refresh_closeout),
        .relation_layer_mask(refresh_relation_layer_mask),
        .metadata_bytes(refresh_metadata_bytes), .metadata_rounds(refresh_metadata_rounds),
        .dma_request_valid(refresh_read_request_valid),
        .dma_request_ready(dma_read_request_ready[DMA_READ_QKV_METADATA] && refresh_active),
        .dma_request_address(refresh_read_request.byte_address), .dma_request_bytes(refresh_read_request.byte_count),
        .dma_request_tag(refresh_read_request.tag),
        .dma_read_valid(dma_read_data_valid[DMA_READ_QKV_METADATA] && refresh_active),
        .dma_read_ready(refresh_read_data_ready), .dma_read_data(dma_read_data[DMA_READ_QKV_METADATA].data),
        .dma_read_byte_enable(dma_read_data[DMA_READ_QKV_METADATA].byte_enable),
        .dma_read_last(dma_read_data[DMA_READ_QKV_METADATA].last), .dma_read_tag(dma_read_data[DMA_READ_QKV_METADATA].tag),
        .dma_error(dma_read_completion[DMA_READ_QKV_METADATA].error && refresh_active), .dma_abort_ack(1'b0),
        .dma_write_request_valid(refresh_write_request_valid),
        .dma_write_request_ready(dma_write_request_ready[DMA_WRITE_MATMUL] && refresh_active),
        .dma_write_request_address(refresh_write_request.byte_address), .dma_write_request_bytes(refresh_write_request.byte_count),
        .dma_write_request_tag(refresh_write_request.tag), .dma_write_valid(refresh_write_data_valid),
        .dma_write_ready(dma_write_data_ready[DMA_WRITE_MATMUL] && refresh_active),
        .dma_write_data(refresh_write_data.data), .dma_write_byte_enable(refresh_write_data.byte_enable),
        .dma_write_last(refresh_write_data.last), .dma_write_done(dma_write_completion[DMA_WRITE_MATMUL].done_pulse && refresh_active),
        .dma_write_error(dma_write_completion[DMA_WRITE_MATMUL].error && refresh_active),
        .metadata_scratch_req_valid(refresh_metadata_scratch_req_valid),
        .metadata_scratch_req_ready(refresh_metadata_scratch_req_ready),
        .metadata_scratch_write(refresh_metadata_scratch_write),
        .metadata_scratch_address(refresh_metadata_scratch_address),
        .metadata_scratch_write_data(refresh_metadata_scratch_write_data),
        .metadata_scratch_write_enable(refresh_metadata_scratch_write_enable),
        .metadata_scratch_rsp_valid(refresh_metadata_scratch_rsp_valid),
        .metadata_scratch_rsp_data(refresh_metadata_scratch_rsp_data),
        .metadata_scratch_aux_write_valid(refresh_metadata_scratch_aux_write_valid),
        .metadata_scratch_aux_write_ready(refresh_metadata_scratch_aux_write_ready),
        .metadata_scratch_aux_write_address(refresh_metadata_scratch_aux_write_address),
        .metadata_scratch_aux_write_data(refresh_metadata_scratch_aux_write_data),
        .metadata_scratch_aux_write_enable(refresh_metadata_scratch_aux_write_enable),
        .selected_valid(refresh_selected_valid), .selected_ready(1'b1), .selected_position(refresh_selected_position),
        .selected_last(refresh_selected_last), .done_valid(refresh_done),
        .done_ready(execution_forward_postprocess_done_ready && execution_refresh_select), .error(refresh_error), .error_id(refresh_error_id)
    );

    forward_postprocess_controller forward_postprocess (
        .start_source_a_valid(refresh_source_a_valid), .start_source_a_mask(refresh_source_a_mask),
        .start_source_a_block_start(refresh_source_a_block_start), .start_source_a_capture_index(refresh_source_a_capture_index),
        .clk, .rst, .start_valid(post_controller_start_valid),
        .packer_abort_request(row_packer_abort[0]), .packer_abort_ack(row_packer_abort_ack && row_packer_selected_post),
        .packer_start_valid(row_packer_start_valid[0]), .packer_start_ready(row_packer_start_ready && row_packer_selected_post),
        .packer_config(row_packer_config[0]), .packer_row_valid(row_packer_row_valid[0]),
        .packer_row_ready(row_packer_row_ready && row_packer_selected_post), .packer_row(row_packer_row[0]),
        .packer_beat_valid(row_packer_beat_valid && row_packer_selected_post), .packer_beat_ready(row_packer_beat_ready[0]),
        .packer_beat_data(row_packer_beat_data), .packer_beat_last(row_packer_beat_last),
        .packer_done_valid(row_packer_done_valid && row_packer_selected_post), .packer_done_ready(row_packer_done_ready[0]),
        .packer_error(row_packer_error && row_packer_selected_post), .packer_error_id(row_packer_error_id),
        .packer_compute_groups(row_packer_compute_groups), .packer_semantic_groups(row_packer_semantic_groups),
        .start_ready(post_controller_start_ready),
        .start_configuration_bits(execution_forward_postprocess_configuration_bits),
        .done_valid(post_done), .current_block_complete(execution_forward_postprocess_block_complete),
        .done_ready(execution_forward_postprocess_done_ready && !execution_refresh_select),
        .error(post_error),
        .error_id(post_error_id), .active(post_active),
        .meta_request_valid(post_meta_request_valid),
        .meta_request_ready(post_meta_request_ready),
        .meta_request(post_meta_request), .meta_data_valid(post_meta_data_valid),
        .meta_data_ready(post_meta_data_ready), .meta_data(post_meta_data),
        .meta_completion(post_meta_completion),
        .lm_request_valid(post_lm_request_valid),
        .lm_request_ready(post_lm_request_ready), .lm_request(post_lm_request),
        .lm_data_valid(post_lm_data_valid), .lm_data_ready(post_lm_data_ready),
        .lm_data(post_lm_data), .lm_completion(post_lm_completion),
        .write_request_valid(post_write_request_valid),
        .write_request_ready(post_write_request_ready),
        .write_request(post_write_request),
        .write_data_valid(post_write_data_valid),
        .write_data_ready(post_write_data_ready), .write_data(post_write_data),
        .write_completion(post_write_completion),
        .hidden_start_valid(post_hidden_start_valid),
        .hidden_start_ready(post_hidden_start_ready),
        .hidden_start_address(post_hidden_start_address),
        .hidden_start_limit(post_hidden_start_limit),
        .hidden_start_bytes(post_hidden_start_bytes),
        .hidden_start_rows(post_hidden_start_rows),
        .hidden_start_ddr_row_index(post_hidden_start_ddr_row_index),
        .hidden_start_source_round(post_hidden_start_source_round),
        .hidden_start_source_row(post_hidden_start_source_row),
        .hidden_done_valid(post_hidden_done_valid),
        .hidden_done_ready(post_hidden_done_ready),
        .hidden_error(post_hidden_error),
        .rms_start_valid(post_rms_start_valid),
        .rms_start_ready(post_rms_start_ready),
        .rms_start_rows(post_rms_start_rows),
        .rms_start_elements(post_rms_start_elements),
        .rms_start_epsilon_bf16(post_rms_start_epsilon),
        .rms_start_gamma_bypass(post_rms_start_gamma_bypass),
        .rms_done_valid(post_rms_done_valid),
        .rms_done_ready(post_rms_done_ready), .rms_error(post_rms_error),
        .rms_source_req_valid(post_rms_source_req_valid),
        .rms_source_req_ready(post_rms_source_req_ready),
        .rms_source_req(post_rms_source_req),
        .rms_source_rsp_valid(post_rms_source_rsp_valid),
        .rms_source_rsp_ready(post_rms_source_rsp_ready),
        .rms_source_rsp(post_rms_source_rsp),
        .max_req_valid(post_max_req_valid), .max_req_ready(post_max_req_ready),
        .max_req(post_max_req), .max_rsp_valid(post_max_rsp_valid),
        .max_rsp_ready(post_max_rsp_ready), .max_rsp(post_max_rsp),
        .quant_scale_req_valid(post_quant_scale_req_valid),
        .quant_scale_req_ready(post_quant_scale_req_ready),
        .quant_scale_req(post_quant_scale_req),
        .quant_scale_rsp_valid(post_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(post_quant_scale_rsp_ready),
        .quant_scale_rsp(post_quant_scale_rsp),
        .quant_values_req_valid(post_quant_values_req_valid),
        .quant_values_req_ready(post_quant_values_req_ready),
        .quant_values_req(post_quant_values_req),
        .quant_values_rsp_valid(post_quant_values_rsp_valid),
        .quant_values_rsp_ready(post_quant_values_rsp_ready),
        .quant_values_rsp(post_quant_values_rsp),
        .writer_cfg_valid(post_writer_cfg_valid),
        .writer_cfg_ready(post_writer_cfg_ready), .writer_cfg(post_writer_cfg),
        .writer_scale_valid(post_writer_scale_valid),
        .writer_scale_ready(post_writer_scale_ready),
        .writer_scale(post_writer_scale),
        .writer_values_valid(post_writer_values_valid),
        .writer_values_ready(post_writer_values_ready),
        .writer_values(post_writer_values),
        .writer_done_pulse(resource_writer_done_pulse[0]),
        .writer_error(resource_writer_error[0]),
        .panel_write_valid(post_panel_write_valid),
        .panel_write_ready(post_panel_write_ready),
        .panel_write(post_panel_write),
        .operand_read_valid(post_operand_read_valid),
        .operand_read_ready(post_operand_read_ready),
        .operand_read(post_operand_read), .operand_rsp_valid(post_operand_rsp_valid),
        .operand_rsp(post_operand_rsp),
        .scale_write_valid(lm_head_scale_write_valid),
        .scale_write_ready(lm_head_scale_write_ready),
        .scale_write_address(lm_head_scale_write_address),
        .scale_write_data(lm_head_scale_write_data),
        .scale_read_valid(lm_head_scale_read_valid),
        .scale_read_ready(lm_head_scale_read_ready),
        .scale_read_address(lm_head_scale_read_address),
        .scale_read_rsp_valid(lm_head_scale_read_rsp_valid),
        .scale_read_rsp_data(lm_head_scale_read_rsp_data),
        .candidate_state_read_valid, .candidate_state_read_ready,
        .candidate_state_read_row_group, .candidate_state_read_lane_block,
        .candidate_state_read_row_half, .candidate_state_read_word,
        .candidate_state_read_row_mask, .candidate_state_read_tag,
        .candidate_state_read_rsp_valid, .candidate_state_read_rsp_ready,
        .candidate_state_read_rsp_data, .candidate_state_read_rsp_row_mask,
        .candidate_state_read_rsp_tag, .candidate_state_write_valid,
        .candidate_state_write_ready, .candidate_state_write_row_group,
        .candidate_state_write_lane_block, .candidate_state_write_row_half,
        .candidate_state_write_word, .candidate_state_write_row_mask,
        .candidate_state_write_data, .post_table_access_valid,
        .post_table_access_ready, .post_table_access_enable,
        .post_state_write_valid, .post_state_write_ready,
        .post_state_write_word_address, .post_state_write_data,
        .post_state_read_valid, .post_state_read_ready,
        .post_state_read_word_address, .post_state_read_tag,
        .post_state_read_rsp_valid, .post_state_read_rsp_data,
        .post_state_read_rsp_tag, .post_suppressed_write_valid,
        .post_suppressed_write_ready, .post_suppressed_write_address,
        .post_suppressed_write_data, .post_suppressed_read_valid,
        .post_suppressed_read_ready, .post_suppressed_read_index,
        .post_suppressed_read_rsp_valid,
        .post_suppressed_read_rsp_token_id,
        .pe_abort_request(post_pe_abort_request),
        .pe_abort_ack(post_pe_abort_ack), .pe_req_valid(post_pe_req_valid),
        .pe_req_ready(post_pe_req_ready), .pe_req(post_pe_req),
        .pe_rsp_valid(post_pe_rsp_valid), .pe_rsp_ready(post_pe_rsp_ready),
        .pe_rsp(post_pe_rsp), .pe_idle(shared_pe_idle),
        .rescale_abort_request(post_rescale_abort_request),
        .rescale_abort_ack(post_rescale_abort_ack),
        .rescale_req_valid(post_rescale_req_valid),
        .rescale_req_ready(post_rescale_req_ready),
        .rescale_req(post_rescale_req),
        .rescale_rsp_valid(post_rescale_rsp_valid),
        .rescale_rsp_ready(post_rescale_rsp_ready),
        .rescale_rsp(post_rescale_rsp), .rescale_idle(shared_rescale_idle),
        .bf16_abort_request(post_bf16_abort_request),
        .bf16_abort_ack(post_bf16_abort_ack),
        .bf16_req_valid(post_bf16_req_valid),
        .bf16_req_ready(post_bf16_req_ready), .bf16_req(post_bf16_req),
        .bf16_rsp_valid(post_bf16_rsp_valid),
        .bf16_rsp_ready(post_bf16_rsp_ready), .bf16_rsp(post_bf16_rsp),
        .reduction_req_valid(post_reduction_req_valid),
        .reduction_req_ready(post_reduction_req_ready),
        .reduction_req(post_reduction_req),
        .reduction_rsp_valid(post_reduction_rsp_valid),
        .reduction_rsp_ready(post_reduction_rsp_ready),
        .reduction_rsp(post_reduction_rsp),
        .exp_req_valid(post_exp_req_valid), .exp_req_ready(post_exp_req_ready),
        .exp_req(post_exp_req), .exp_rsp_valid(post_exp_rsp_valid),
        .exp_rsp_ready(post_exp_rsp_ready), .exp_rsp(post_exp_rsp),
        .reciprocal_req_valid(post_reciprocal_req_valid),
        .reciprocal_req_ready(post_reciprocal_req_ready),
        .reciprocal_req(post_reciprocal_req),
        .reciprocal_rsp_valid(post_reciprocal_rsp_valid),
        .reciprocal_rsp_ready(post_reciprocal_rsp_ready),
        .reciprocal_rsp(post_reciprocal_rsp),
        .accepted_weight_bytes(post_weight_bytes),
        .accepted_scale_bytes(post_scale_bytes),
        .accepted_pe_count(post_pe_count),
        .candidate_wait_cycles(post_candidate_wait_cycles));

    always_comb begin : post_resource_select
        resource_bf16_abort_request = {
            elementwise_abort_request, 1'b0, 1'b0,
            matmul_bf16_abort_request, 1'b0};
        resource_bf16_req_valid = {elementwise_bf16_req_valid,
            attention_softmax_bf16_req_valid, rope_arithmetic_req_valid,
            matmul_bf16_req_valid, rms_arithmetic_req_valid};
        for (integer client = 0; client < 5; client++) begin
            resource_bf16_req[client] = bf16_client_req[client];
            resource_bf16_rsp_ready[client] = 1'b0;
            bf16_client_rsp[client] = resource_bf16_rsp[client];
        end
        resource_bf16_rsp_ready[0] = rms_arithmetic_rsp_ready;
        resource_bf16_rsp_ready[1] = matmul_bf16_rsp_ready;
        resource_bf16_rsp_ready[2] = rope_arithmetic_rsp_ready;
        resource_bf16_rsp_ready[3] = refresh_active ? refresh_bf16_rsp_ready : post_active ? post_bf16_rsp_ready :
            attention_softmax_bf16_rsp_ready;
        resource_bf16_rsp_ready[4] = elementwise_bf16_rsp_ready;
        rms_arithmetic_req_ready = resource_bf16_req_ready[0];
        matmul_bf16_req_ready = resource_bf16_req_ready[1];
        rope_arithmetic_req_ready = resource_bf16_req_ready[2];
        attention_softmax_bf16_req_ready = post_active || refresh_active ? 1'b0 :
            resource_bf16_req_ready[3];
        elementwise_bf16_req_ready = resource_bf16_req_ready[4];
        rms_arithmetic_rsp_valid = resource_bf16_rsp_valid[0];
        matmul_bf16_rsp_valid = resource_bf16_rsp_valid[1];
        rope_arithmetic_rsp_valid = resource_bf16_rsp_valid[2];
        attention_softmax_bf16_rsp_valid = post_active || refresh_active ? 1'b0 :
            resource_bf16_rsp_valid[3];
        elementwise_bf16_rsp_valid = resource_bf16_rsp_valid[4];
        post_bf16_abort_ack = post_active && resource_bf16_abort_ack[3];
        post_bf16_req_ready = post_active && resource_bf16_req_ready[3];
        post_bf16_rsp_valid = post_active && resource_bf16_rsp_valid[3];
        post_bf16_rsp = resource_bf16_rsp[3];
        refresh_bf16_req_ready = refresh_active && resource_bf16_req_ready[3];
        refresh_bf16_rsp_valid = refresh_active && resource_bf16_rsp_valid[3];
        refresh_bf16_abort_ack = refresh_active && resource_bf16_abort_ack[3];
        if (post_active) begin
            resource_bf16_abort_request[3] = post_bf16_abort_request;
            resource_bf16_req_valid[3] = post_bf16_req_valid;
            resource_bf16_req[3] = post_bf16_req;
        end
        if (refresh_active) begin
            resource_bf16_abort_request[3] = refresh_bf16_abort_request;
            resource_bf16_req_valid[3] = refresh_bf16_req_valid;
            // Joint selection uses only lane 0; other lanes remain masked.
            resource_bf16_req[3].operation = refresh_bf16_req.operation;
            resource_bf16_req[3].values[127:0] = refresh_bf16_req.values[127:0];
            resource_bf16_req[3].paired_values[127:0] = refresh_bf16_req.paired_values[127:0];
            resource_bf16_req[3].factor0_values[127:0] = refresh_bf16_req.factor0_values[127:0];
            resource_bf16_req[3].lane_mask = refresh_bf16_req.lane_mask;
            resource_bf16_req[3].tag = refresh_bf16_req.tag;
        end

        resource_max_req_valid = {elementwise_max_req_valid,
            post_active ? post_max_req_valid :
                attention_softmax_max_req_valid,
            matmul_max_req_valid, qkv_max_req_valid};
        for (integer client = 0; client < 4; client++) begin
            resource_max_req[client] = maximum_client_req[client];
            resource_max_rsp_ready[client] = 1'b0;
            maximum_client_rsp[client] = resource_max_rsp[client];
        end
        if (post_active)
            resource_max_req[2] = post_max_req;
        qkv_max_req_ready = resource_max_req_ready[0];
        matmul_max_req_ready = resource_max_req_ready[1];
        attention_softmax_max_req_ready = post_active ? 1'b0 :
            resource_max_req_ready[2];
        elementwise_max_req_ready = resource_max_req_ready[3];
        resource_max_rsp_ready[0] = qkv_max_rsp_ready;
        resource_max_rsp_ready[1] = matmul_max_rsp_ready;
        resource_max_rsp_ready[2] = post_active ? post_max_rsp_ready :
            attention_softmax_max_rsp_ready;
        resource_max_rsp_ready[3] = elementwise_max_rsp_ready;
        qkv_max_rsp_valid = resource_max_rsp_valid[0];
        matmul_max_rsp_valid = resource_max_rsp_valid[1];
        attention_softmax_max_rsp_valid = post_active ? 1'b0 :
            resource_max_rsp_valid[2];
        elementwise_max_rsp_valid = resource_max_rsp_valid[3];
        post_max_req_ready = post_active && resource_max_req_ready[2];
        post_max_rsp_valid = post_active && resource_max_rsp_valid[2];
        post_max_rsp = resource_max_rsp[2];

        resource_reduction_req_valid = {
            post_active ? post_reduction_req_valid :
                attention_softmax_reduction_req_valid,
            rms_reduction_req_valid};
        resource_reduction_req[0] = reduction_client_req[0];
        resource_reduction_req[1] = post_active ? post_reduction_req :
            reduction_client_req[1];
        rms_reduction_req_ready = resource_reduction_req_ready[0];
        attention_softmax_reduction_req_ready = post_active ? 1'b0 :
            resource_reduction_req_ready[1];
        post_reduction_req_ready = post_active &&
            resource_reduction_req_ready[1];
        resource_reduction_rsp_ready[0] = rms_reduction_rsp_ready;
        resource_reduction_rsp_ready[1] = post_active ?
            post_reduction_rsp_ready : attention_softmax_reduction_rsp_ready;
        reduction_client_rsp[0] = resource_reduction_rsp[0];
        reduction_client_rsp[1] = resource_reduction_rsp[1];
        rms_reduction_rsp_valid = resource_reduction_rsp_valid[0];
        attention_softmax_reduction_rsp_valid = post_active ? 1'b0 :
            resource_reduction_rsp_valid[1];
        post_reduction_rsp_valid = post_active &&
            resource_reduction_rsp_valid[1];
        post_reduction_rsp = resource_reduction_rsp[1];

        resource_pe_abort_request = {attention_pe_abort_request,
            post_active ? post_pe_abort_request : matmul_tile_abort_request};
        resource_pe_req_valid = {attention_pe_req_valid,
            post_active ? post_pe_req_valid : matmul_tile_req_valid};
        resource_pe_req[0] = post_active ? post_pe_req : pe_client_req[0];
        resource_pe_req[1] = pe_client_req[1];
        matmul_tile_abort_ack = post_active ? 1'b0 : resource_pe_abort_ack[0];
        post_pe_abort_ack = post_active && resource_pe_abort_ack[0];
        attention_pe_abort_ack = resource_pe_abort_ack[1];
        matmul_tile_req_ready = post_active ? 1'b0 : resource_pe_req_ready[0];
        post_pe_req_ready = post_active && resource_pe_req_ready[0];
        attention_pe_req_ready = resource_pe_req_ready[1];
        resource_pe_rsp_ready[0] = post_active ? post_pe_rsp_ready :
            matmul_tile_accum_result_ready;
        resource_pe_rsp_ready[1] = attention_pe_accum_result_ready;
        pe_client_rsp[0] = resource_pe_rsp[0];
        pe_client_rsp[1] = resource_pe_rsp[1];
        matmul_tile_accum_result_valid = post_active ? 1'b0 :
            resource_pe_rsp_valid[0];
        post_pe_rsp_valid = post_active && resource_pe_rsp_valid[0];
        post_pe_rsp = resource_pe_rsp[0];
        attention_pe_accum_result_valid = resource_pe_rsp_valid[1];

        resource_rescale_abort_request = {attention_rescale_abort_request,
            post_active ? post_rescale_abort_request :
                matmul_rescale_abort_request};
        resource_rescale_req_valid = {attention_rescale_req_valid,
            post_active ? post_rescale_req_valid : matmul_rescale_req_valid};
        resource_rescale_req[0] = post_active ? post_rescale_req :
            rescale_client_req[0];
        resource_rescale_req[1] = rescale_client_req[1];
        matmul_rescale_abort_ack = post_active ? 1'b0 :
            resource_rescale_abort_ack[0];
        post_rescale_abort_ack = post_active && resource_rescale_abort_ack[0];
        attention_rescale_abort_ack = resource_rescale_abort_ack[1];
        matmul_rescale_req_ready = post_active ? 1'b0 :
            resource_rescale_req_ready[0];
        post_rescale_req_ready = post_active && resource_rescale_req_ready[0];
        attention_rescale_req_ready = resource_rescale_req_ready[1];
        resource_rescale_rsp_ready[0] = post_active ? post_rescale_rsp_ready :
            matmul_rescale_rsp_ready;
        resource_rescale_rsp_ready[1] = attention_rescale_rsp_ready;
        rescale_client_rsp[0] = resource_rescale_rsp[0];
        rescale_client_rsp[1] = resource_rescale_rsp[1];
        matmul_rescale_rsp_valid = post_active ? 1'b0 :
            resource_rescale_rsp_valid[0];
        post_rescale_rsp_valid = post_active && resource_rescale_rsp_valid[0];
        post_rescale_rsp = resource_rescale_rsp[0];
        attention_rescale_rsp_valid = resource_rescale_rsp_valid[1];

        resource_quant_abort_request = {elementwise_abort_request,
            post_active ? 1'b0 : 1'b0,
            matmul_activation_writer_abort_request,
            qkv_quant_abort_request};
        resource_quant_scale_req_valid = {
            elementwise_quant_scale_req_valid,
            post_active ? post_quant_scale_req_valid :
                attention_softmax_quant_scale_req_valid,
            matmul_quant_scale_req_valid, qkv_quant_scale_req_valid};
        resource_quant_values_req_valid = {
            elementwise_quant_values_req_valid,
            post_active ? post_quant_values_req_valid :
                attention_softmax_quant_values_req_valid,
            matmul_quant_values_req_valid, qkv_quant_req_valid};
        for (integer client = 0; client < 4; client++) begin
            resource_quant_scale_req[client] = quant_scale_client_req[client];
            resource_quant_values_req[client] = quant_values_client_req[client];
            resource_quant_scale_rsp_ready[client] = 1'b0;
            resource_quant_values_rsp_ready[client] = 1'b0;
            quant_scale_client_rsp[client] = resource_quant_scale_rsp[client];
            quant_values_client_rsp[client] = resource_quant_values_rsp[client];
        end
        if (post_active) begin
            resource_quant_scale_req[2] = post_quant_scale_req;
            resource_quant_values_req[2] = post_quant_values_req;
        end
        qkv_quant_scale_req_ready = resource_quant_scale_req_ready[0];
        matmul_quant_scale_req_ready = resource_quant_scale_req_ready[1];
        attention_softmax_quant_scale_req_ready = post_active ? 1'b0 :
            resource_quant_scale_req_ready[2];
        elementwise_quant_scale_req_ready = resource_quant_scale_req_ready[3];
        qkv_quant_req_ready = resource_quant_values_req_ready[0];
        matmul_quant_values_req_ready = resource_quant_values_req_ready[1];
        attention_softmax_quant_values_req_ready = post_active ? 1'b0 :
            resource_quant_values_req_ready[2];
        elementwise_quant_values_req_ready = resource_quant_values_req_ready[3];
        resource_quant_scale_rsp_ready[0] = qkv_quant_scale_rsp_ready;
        resource_quant_scale_rsp_ready[1] = matmul_quant_scale_rsp_ready;
        resource_quant_scale_rsp_ready[2] = post_active ?
            post_quant_scale_rsp_ready : attention_softmax_quant_scale_rsp_ready;
        resource_quant_scale_rsp_ready[3] = elementwise_quant_scale_rsp_ready;
        resource_quant_values_rsp_ready[0] = qkv_quant_rsp_ready;
        resource_quant_values_rsp_ready[1] = matmul_quant_values_rsp_ready;
        resource_quant_values_rsp_ready[2] = post_active ?
            post_quant_values_rsp_ready : attention_softmax_quant_values_rsp_ready;
        resource_quant_values_rsp_ready[3] = elementwise_quant_values_rsp_ready;
        qkv_quant_scale_rsp_valid = resource_quant_scale_rsp_valid[0];
        matmul_quant_scale_rsp_valid = resource_quant_scale_rsp_valid[1];
        attention_softmax_quant_scale_rsp_valid = post_active ? 1'b0 :
            resource_quant_scale_rsp_valid[2];
        elementwise_quant_scale_rsp_valid = resource_quant_scale_rsp_valid[3];
        qkv_quant_rsp_valid = resource_quant_values_rsp_valid[0];
        matmul_quant_values_rsp_valid = resource_quant_values_rsp_valid[1];
        attention_softmax_quant_values_rsp_valid = post_active ? 1'b0 :
            resource_quant_values_rsp_valid[2];
        elementwise_quant_values_rsp_valid = resource_quant_values_rsp_valid[3];
        post_quant_scale_req_ready = post_active &&
            resource_quant_scale_req_ready[2];
        post_quant_scale_rsp_valid = post_active &&
            resource_quant_scale_rsp_valid[2];
        post_quant_scale_rsp = resource_quant_scale_rsp[2];
        post_quant_values_req_ready = post_active &&
            resource_quant_values_req_ready[2];
        post_quant_values_rsp_valid = post_active &&
            resource_quant_values_rsp_valid[2];
        post_quant_values_rsp = resource_quant_values_rsp[2];

        resource_writer_abort_request = {elementwise_writer_abort_request,
            post_active ? 1'b0 : matmul_activation_writer_abort_request};
        resource_writer_cfg_valid = {elementwise_writer_cfg_valid,
            post_active ? post_writer_cfg_valid :
                matmul_activation_writer_cfg_valid};
        resource_writer_scale_valid = {elementwise_writer_scale_valid,
            post_active ? post_writer_scale_valid :
                matmul_activation_writer_scale_valid};
        resource_writer_values_valid = {elementwise_writer_quantized_valid,
            post_active ? post_writer_values_valid :
                matmul_activation_writer_quantized_valid};
        resource_writer_cfg[0] = post_active ? post_writer_cfg :
            writer_client_cfg[0];
        resource_writer_cfg[1] = writer_client_cfg[1];
        resource_writer_scale[0] = post_active ? post_writer_scale :
            writer_client_scale[0];
        resource_writer_scale[1] = writer_client_scale[1];
        resource_writer_values[0] = post_active ? post_writer_values :
            writer_client_values[0];
        resource_writer_values[1] = writer_client_values[1];
        matmul_activation_writer_cfg_ready = post_active ? 1'b0 :
            resource_writer_cfg_ready[0];
        post_writer_cfg_ready = post_active && resource_writer_cfg_ready[0];
        elementwise_writer_cfg_ready = resource_writer_cfg_ready[1];
        matmul_activation_writer_scale_ready = post_active ? 1'b0 :
            resource_writer_scale_ready[0];
        post_writer_scale_ready = post_active &&
            resource_writer_scale_ready[0];
        elementwise_writer_scale_ready = resource_writer_scale_ready[1];
        matmul_activation_writer_quantized_ready = post_active ? 1'b0 :
            resource_writer_values_ready[0];
        post_writer_values_ready = post_active &&
            resource_writer_values_ready[0];
        elementwise_writer_quantized_ready = resource_writer_values_ready[1];
        shared_activation_writer_done_pulse = !post_active &&
            resource_writer_done_pulse[0];
        shared_activation_writer_error = !post_active &&
            resource_writer_error[0];
        elementwise_writer_done_pulse = resource_writer_done_pulse[1];
        elementwise_writer_error = resource_writer_error[1];

        resource_exp_req_valid = post_active ? post_exp_req_valid :
            attention_softmax_exp_req_valid;
        resource_exp_req = post_active ? post_exp_req :
            attention_softmax_exp_req;
        post_exp_req_ready = post_active && resource_exp_req_ready;
        attention_softmax_exp_req_ready = post_active ? 1'b0 :
            resource_exp_req_ready;
        resource_exp_rsp_ready = post_active ? post_exp_rsp_ready :
            attention_softmax_exp_rsp_ready;
        post_exp_rsp_valid = post_active && resource_exp_rsp_valid;
        post_exp_rsp = resource_exp_rsp;
        attention_softmax_exp_rsp_valid = post_active ? 1'b0 :
            resource_exp_rsp_valid;
        attention_softmax_exp_rsp = resource_exp_rsp;

        resource_reciprocal_req_valid = post_active ?
            post_reciprocal_req_valid : attention_softmax_reciprocal_req_valid;
        resource_reciprocal_req = post_active ? post_reciprocal_req :
            attention_softmax_reciprocal_req;
        post_reciprocal_req_ready = post_active &&
            resource_reciprocal_req_ready;
        attention_softmax_reciprocal_req_ready = post_active ? 1'b0 :
            resource_reciprocal_req_ready;
        resource_reciprocal_rsp_ready = post_active ?
            post_reciprocal_rsp_ready : attention_softmax_reciprocal_rsp_ready;
        post_reciprocal_rsp_valid = post_active &&
            resource_reciprocal_rsp_valid;
        post_reciprocal_rsp = resource_reciprocal_rsp;
        attention_softmax_reciprocal_rsp_valid = post_active ? 1'b0 :
            resource_reciprocal_rsp_valid;
        attention_softmax_reciprocal_rsp = resource_reciprocal_rsp;
    end

    shared_compute_resources shared_compute (
        .clk(clk),
        .rst(rst),
        .reduction_req_valid(resource_reduction_req_valid),
        .reduction_req_ready(resource_reduction_req_ready),
        .reduction_req(resource_reduction_req),
        .reduction_rsp_valid(resource_reduction_rsp_valid),
        .reduction_rsp_ready(resource_reduction_rsp_ready),
        .reduction_rsp(resource_reduction_rsp),
        .softmax_exp_req_valid(resource_exp_req_valid),
        .softmax_exp_req_ready(resource_exp_req_ready),
        .softmax_exp_req(resource_exp_req),
        .softmax_exp_rsp_valid(resource_exp_rsp_valid),
        .softmax_exp_rsp_ready(resource_exp_rsp_ready),
        .softmax_exp_rsp(resource_exp_rsp),
        .softmax_reciprocal_req_valid(resource_reciprocal_req_valid),
        .softmax_reciprocal_req_ready(resource_reciprocal_req_ready),
        .softmax_reciprocal_req(resource_reciprocal_req),
        .softmax_reciprocal_rsp_valid(resource_reciprocal_rsp_valid),
        .softmax_reciprocal_rsp_ready(resource_reciprocal_rsp_ready),
        .softmax_reciprocal_rsp(resource_reciprocal_rsp),
        .bf16_abort_request(resource_bf16_abort_request),
        .bf16_abort_ack(resource_bf16_abort_ack),
        .bf16_req_valid(resource_bf16_req_valid),
        .bf16_req_ready(resource_bf16_req_ready),
        .bf16_req(resource_bf16_req),
        .bf16_rsp_valid(resource_bf16_rsp_valid),
        .bf16_rsp_ready(resource_bf16_rsp_ready),
        .bf16_rsp(resource_bf16_rsp),
        .max_req_valid(resource_max_req_valid),
        .max_req_ready(resource_max_req_ready),
        .max_req(resource_max_req),
        .max_rsp_valid(resource_max_rsp_valid),
        .max_rsp_ready(resource_max_rsp_ready),
        .max_rsp(resource_max_rsp),
        .pe_abort_request(resource_pe_abort_request),
        .pe_abort_ack(resource_pe_abort_ack),
        .pe_req_valid(resource_pe_req_valid),
        .pe_req_ready(resource_pe_req_ready),
        .pe_req(resource_pe_req),
        .pe_accum_result_valid(resource_pe_rsp_valid),
        .pe_accum_result_ready(resource_pe_rsp_ready),
        .pe_accum_result(resource_pe_rsp),
        .pe_idle(shared_pe_idle),
        .rescale_claim_valid({1'b0,
            post_active ? 1'b0 : matmul_rescale_claim_valid}),
        .rescale_claim_ready({unused_attention_rescale_claim_ready,
                              matmul_rescale_claim_ready}),
        .rescale_abort_request(resource_rescale_abort_request),
        .rescale_abort_ack(resource_rescale_abort_ack),
        .rescale_req_valid(resource_rescale_req_valid),
        .rescale_req_ready(resource_rescale_req_ready),
        .rescale_req(resource_rescale_req),
        .rescale_rsp_valid(resource_rescale_rsp_valid),
        .rescale_rsp_ready(resource_rescale_rsp_ready),
        .rescale_rsp(resource_rescale_rsp),
        .rescale_idle(shared_rescale_idle),
        .quant_abort_request(resource_quant_abort_request),
        .quant_abort_ack(resource_quant_abort_ack),
        .quant_scale_req_valid(resource_quant_scale_req_valid),
        .quant_scale_req_ready(resource_quant_scale_req_ready),
        .quant_scale_req(resource_quant_scale_req),
        .quant_scale_rsp_valid(resource_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(resource_quant_scale_rsp_ready),
        .quant_scale_rsp(resource_quant_scale_rsp),
        .quant_values_req_valid(resource_quant_values_req_valid),
        .quant_values_req_ready(resource_quant_values_req_ready),
        .quant_values_req(resource_quant_values_req),
        .quant_values_rsp_valid(resource_quant_values_rsp_valid),
        .quant_values_rsp_ready(resource_quant_values_rsp_ready),
        .quant_values_rsp(resource_quant_values_rsp),
        .writer_abort_request(resource_writer_abort_request),
        .writer_cfg_valid(resource_writer_cfg_valid),
        .writer_cfg_ready(resource_writer_cfg_ready),
        .writer_cfg(resource_writer_cfg),
        .writer_scale_valid(resource_writer_scale_valid),
        .writer_scale_ready(resource_writer_scale_ready),
        .writer_scale(resource_writer_scale),
        .writer_quantized_valid(resource_writer_values_valid),
        .writer_quantized_ready(resource_writer_values_ready),
        .writer_quantized(resource_writer_values),
        .writer_done_pulse(resource_writer_done_pulse),
        .writer_error(resource_writer_error),
        .activation_scale_valid(shared_activation_scale_valid),
        .activation_scale_ready(shared_activation_scale_ready),
        .activation_scale_row_base(shared_activation_scale_row_base),
        .activation_scale_row_mask(shared_activation_scale_row_mask),
        .activation_scale_values(shared_activation_scale_values),
        .activation_write_valid(matmul_activation_write_valid),
        .activation_write_ready(matmul_activation_write_ready),
        .activation_write_slot_valid(matmul_activation_write.slot_valid),
        .activation_write_address(matmul_activation_write.byte_address),
        .activation_write_data(matmul_activation_write.data),
        .activation_write_byte_enable(matmul_activation_write.byte_enable),
        .activation_write_tag(matmul_activation_write_tag)
    );

    always_comb begin : matmul_command_select
        matmul_command = hardware_types_pkg::MATMUL_COMMAND_QKV;
        if (!qkv_phase_active) begin
            case (execution_layer_operator_index)
                LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL:
                    matmul_command =
                        hardware_types_pkg::MATMUL_COMMAND_ATTENTION_OUTPUT;
                LAYER_OPERATOR_FFN_GATE_MATMUL:
                    matmul_command =
                        hardware_types_pkg::MATMUL_COMMAND_FFN_GATE;
                LAYER_OPERATOR_FFN_UP_MATMUL:
                    matmul_command =
                        hardware_types_pkg::MATMUL_COMMAND_FFN_UP;
                LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL:
                    matmul_command =
                        hardware_types_pkg::MATMUL_COMMAND_FFN_DOWN;
                default: begin end
            endcase
        end

        matmul_start_operator_tag = {operator_command_layer[4:0],
            qkv_phase_active ? LAYER_OPERATOR_QKV_PREPARATION :
                execution_layer_operator_index,
            qkv_phase_active ? {qkv_matmul_qkv_select, qkv_matmul_head[4:0]} :
                7'd0};
    end

    always_comb begin : matmul_batch_group_select
        matmul_ffn_batch_pair_config = ffn_batch_pair_config;
        if (qkv_phase_active && ffn_batch_pair_config.qkvo_group)
            matmul_ffn_batch_pair_config.qkv_staging_base =
                qkv_matmul_group_staging_base;
    end

    always_comb begin : matmul_prefetch_select
        matmul_prefetch_valid = 1'b0;
        matmul_prefetch_cancel = command_child_done_valid &&
            command_child_done_error && !matmul_prefetch_empty;
        matmul_prefetch_operator_tag = 16'd0;
        matmul_prefetch_command = hardware_types_pkg::MATMUL_COMMAND_QKV;
        matmul_prefetch_qkv_select = qkv_matmul_prefetch_qkv_select;
        matmul_prefetch_qkv_head = qkv_matmul_prefetch_head;
        qkv_matmul_prefetch_ready = 1'b0;

        matmul_prefetch_source_valid = 5'd0;
        if (execution_block_schedule_active && layer_entry_valid &&
            layer_entry_index == operator_command_layer[4:0] &&
            !local_layout_update_valid &&
            !(command_child_done_valid && command_child_done_error)) begin
            matmul_prefetch_source_valid[MATMUL_PREFETCH_RMS_QKV] =
                execution_layer_operator_index == LAYER_OPERATOR_ATTENTION_RMSNORM &&
                !attention_normalized_capture &&
                !active_command_rms_qkv_prefetch_q &&
                rms_gamma_loaded &&
                current_layout.panel_layout == LOCAL_PANEL_LAYOUT_QKV;
            matmul_prefetch_source_valid[MATMUL_PREFETCH_QKV_NEXT] =
                execution_layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION &&
                qkv_matmul_prefetch_valid &&
                current_layout.panel_layout == LOCAL_PANEL_LAYOUT_QKV &&
                !current_layout.qkv_head_staging;
            matmul_prefetch_source_valid[MATMUL_PREFETCH_FFN_GATE] =
                !ffn_batch_pair_config.enable &&
                execution_layer_operator_index == LAYER_OPERATOR_FFN_RESIDUAL_SPILL &&
                current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED;
            matmul_prefetch_source_valid[MATMUL_PREFETCH_FFN_UP] =
                !ffn_batch_pair_config.fused_product &&
                execution_layer_operator_index == LAYER_OPERATOR_FFN_GATE_MATMUL &&
                (current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED ||
                 current_layout.panel_layout == LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL);
            matmul_prefetch_source_valid[MATMUL_PREFETCH_FFN_DOWN] =
                execution_layer_operator_index ==
                    LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE &&
                elementwise_busy &&
                elementwise_down_prefetch_ready &&
                current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED;
        end

        matmul_prefetch_valid = |matmul_prefetch_source_valid;
        case (matmul_prefetch_source_valid)
            5'b00001: begin
                matmul_prefetch_valid = 1'b1;
                matmul_prefetch_qkv_select =
                    active_command_rms_qkv_prefetch_q ? 2'd0 : 2'd2;
                matmul_prefetch_qkv_head = 6'd0;
                matmul_prefetch_operator_tag = {operator_command_layer[4:0],
                    LAYER_OPERATOR_QKV_PREPARATION,
                    active_command_rms_qkv_prefetch_q ? 2'd0 : 2'd2, 5'd0};
            end
            5'b00010: begin
                qkv_matmul_prefetch_ready = matmul_prefetch_ready;
                matmul_prefetch_operator_tag = {operator_command_layer[4:0],
                    LAYER_OPERATOR_QKV_PREPARATION,
                    qkv_matmul_prefetch_qkv_select,
                    qkv_matmul_prefetch_head[4:0]};
            end
            5'b00100: begin
                matmul_prefetch_command =
                    hardware_types_pkg::MATMUL_COMMAND_FFN_GATE;
                matmul_prefetch_operator_tag = {operator_command_layer[4:0],
                    LAYER_OPERATOR_FFN_GATE_MATMUL, 7'd0};
            end
            5'b01000: begin
                matmul_prefetch_command =
                    hardware_types_pkg::MATMUL_COMMAND_FFN_UP;
                matmul_prefetch_operator_tag = {operator_command_layer[4:0],
                    LAYER_OPERATOR_FFN_UP_MATMUL, 7'd0};
            end
            5'b10000: begin
                matmul_prefetch_command =
                    hardware_types_pkg::MATMUL_COMMAND_FFN_DOWN;
                matmul_prefetch_operator_tag = {operator_command_layer[4:0],
                    LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL, 7'd0};
            end
            default: begin end
        endcase
    end

    assign qkv_abort_request = attention_q_head_abort_request ||
        (active_command_valid &&
         active_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION &&
         qkv_residual_spill_complete && qkv_residual_spill_error);
    assign attention_q_head_abort_ack = qkv_abort_ack;
    assign qkv_start_valid = operator_command_valid &&
        pending_start_enable && pending_start_select[COMMAND_START_QKV];
    assign qkv_phase_active =
        (active_command_valid &&
         active_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION) ||
        attention_q_head_active;
    assign attention_abort_request = 1'b0;
    assign attention_start_valid = operator_command_valid &&
        pending_start_enable &&
        pending_start_select[COMMAND_START_ATTENTION];
    assign cache_config_release = qkv_cache_config_release ||
        active_command_valid &&
        active_command.cache_release &&
        operator_command_done_valid && operator_command_done_ready;
    assign block_matmul_start_valid = operator_command_valid &&
        pending_start_enable && pending_start_select[COMMAND_START_MATMUL];
    assign qkv_matmul_start_enable = qkv_phase_active &&
        (!attention_q_head_active || qkv_head_staging_layout_ready) &&
        (!qkvo_kv_requires_expanded_layout || !local_layout_update_valid) &&
        (attention_q_head_active ||
         (qkv_residual_spill_complete && !qkv_residual_spill_error));
    assign qkv_matmul_start_ready = matmul_start_ready &&
        qkv_matmul_start_enable;
    assign matmul_start_valid = qkv_phase_active ?
        qkv_matmul_start_valid && qkv_matmul_start_enable :
        block_matmul_start_valid;
    assign matmul_abort_request = qkv_phase_active &&
        qkv_matmul_abort_request;
    assign qkv_matmul_abort_ack = qkv_phase_active && matmul_abort_ack;
    assign qkv_matmul_done_valid = qkv_phase_active && matmul_done_valid;
    assign qkv_matmul_error = qkv_phase_active && matmul_error;

    always_comb begin : accepted_command_done_ready
        block_rms_done_ready = 1'b0;
        qkv_done_ready = 1'b0;
        attention_done_ready = 1'b0;
        matmul_done_ready = 1'b0;
        elementwise_done_ready = 1'b0;
        if (active_command_valid) begin
            case (active_command.layer_operator_index)
                LAYER_OPERATOR_ATTENTION_RMSNORM,
                LAYER_OPERATOR_FFN_RMSNORM:
                    block_rms_done_ready = operator_command_done_ready &&
                        command_done_can_complete;
                LAYER_OPERATOR_QKV_PREPARATION: begin
                    qkv_done_ready = qkv_residual_spill_complete &&
                        operator_command_done_ready && command_done_can_complete;
                    matmul_done_ready = 1'b1;
                end
                LAYER_OPERATOR_ATTENTION: begin
                    attention_done_ready = operator_command_done_ready &&
                        command_done_can_complete;
                    matmul_done_ready = qkv_phase_active;
                end
                LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
                LAYER_OPERATOR_FFN_GATE_MATMUL,
                LAYER_OPERATOR_FFN_UP_MATMUL,
                LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL:
                    matmul_done_ready = operator_command_done_ready &&
                        command_done_can_complete;
                LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE:
                    elementwise_done_ready = operator_command_done_ready &&
                        command_done_can_complete;
                default: begin end
            endcase
        end
    end

    assign qkv_cache_mapping_commit_ready = cache_config_valid;
    assign qkv_cache_mapping_commit_error = cache_error;
    assign matmul_dma_reader_abort_ack = 1'b0;
    assign qkv_quantized_commit_ready = 1'b1;
    assign qkv_cache_write_complete_valid = 1'b0;
    assign qkv_cache_write_complete_error = 1'b0;

    always_comb begin : pending_command_decode
        pending_start_select = 6'd0;
        pending_initial_layout = '0;
        pending_initial_layout.phase = LOCAL_MEMORY_PHASE_IDLE;
        pending_initial_layout.active_panel = 1'b0;
        pending_initial_layout.qkv_head_staging = 1'b0;
        pending_initial_layout.q_head_active = 1'b0;
        pending_initial_layout.q_head_slot = 1'b0;
        pending_initial_layout.panel_layout = LOCAL_PANEL_LAYOUT_NONE;
        case (pending_command.layer_operator_index)
            LAYER_OPERATOR_ATTENTION_RMSNORM: begin
                pending_start_select[pending_command.qkv_residual_spill ? COMMAND_START_HIDDEN : COMMAND_START_RMS] = 1'b1;
                pending_initial_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                pending_initial_layout.panel_layout = LOCAL_PANEL_LAYOUT_QKV;
            end
            LAYER_OPERATOR_QKV_PREPARATION: begin
                pending_start_select[COMMAND_START_QKV] = 1'b1;
                pending_initial_layout.kv_pair_enable = kv_pair_enable;
                pending_initial_layout.kv_first_batch_tokens = kv_pair_enable ? row_active_rows : 6'd0;
                pending_initial_layout.phase =
                    LOCAL_MEMORY_PHASE_QKV_PREPARATION;
                pending_initial_layout.panel_layout = LOCAL_PANEL_LAYOUT_QKV;
            end
            LAYER_OPERATOR_ATTENTION: begin
                pending_start_select[COMMAND_START_ATTENTION] = 1'b1;
                pending_initial_layout.phase =
                    LOCAL_MEMORY_PHASE_ATTENTION_QK;
                // Grouped Q first reads the expanded activation image. The
                // Attention pair mapping may claim those SRAM ports only
                // after all grouped Q Matmul outputs have been staged.
                pending_initial_layout.attention_pair_enable =
                    attention_pair_enable && !qkvo_query_group_precompute;
                pending_initial_layout.attention_score_words =
                    pending_initial_layout.attention_pair_enable ?
                    8'(local_memory_layout_pkg::attention_pair_score_stride(row_sequence_length)) : 8'd0;
            end
            LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
            LAYER_OPERATOR_FFN_GATE_MATMUL,
            LAYER_OPERATOR_FFN_UP_MATMUL: begin
                pending_start_select[COMMAND_START_MATMUL] = 1'b1;
                pending_initial_layout.phase =
                    ffn_batch_pair_config.enable ? LOCAL_MEMORY_PHASE_MATMUL :
                    LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN;
                pending_initial_layout.panel_layout =
                    ffn_pair_requires_expanded_layout ?
                        LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL :
                        LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_RESIDUAL_SPILL: begin
                pending_start_select[COMMAND_START_HIDDEN] = 1'b1;
                pending_initial_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                pending_initial_layout.panel_layout = LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_HIDDEN_REFILL: begin
                pending_start_select[COMMAND_START_HIDDEN] = 1'b1;
                pending_initial_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
            end
            LAYER_OPERATOR_FFN_RMSNORM: begin
                pending_start_select[COMMAND_START_RMS] = 1'b1;
                pending_initial_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                pending_initial_layout.panel_layout = LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE: begin
                pending_start_select[COMMAND_START_ELEMENTWISE] = 1'b1;
                pending_initial_layout.phase = ffn_batch_pair_config.enable ?
                    LOCAL_MEMORY_PHASE_R4_TWO_TOKENS : matmul_row_config.mixed_group_enable ?
                    LOCAL_MEMORY_PHASE_R4 : LOCAL_MEMORY_PHASE_ELEMENTWISE;
                pending_initial_layout.panel_layout = ffn_batch_pair_config.enable ?
                    LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL : LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL: begin
                pending_start_select[COMMAND_START_MATMUL] = 1'b1;
                pending_initial_layout.phase = LOCAL_MEMORY_PHASE_MATMUL;
                pending_initial_layout.panel_layout = ffn_batch_pair_config.enable ?
                    LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL : LOCAL_PANEL_LAYOUT_MIXED;
            end
            default: begin end
        endcase
    end

    always_comb begin : accepted_command_layout
        accepted_layout = '0;
        accepted_layout.phase = LOCAL_MEMORY_PHASE_IDLE;
        accepted_layout.active_panel = 1'b0;
        accepted_layout.qkv_head_staging = 1'b0;
        accepted_layout.q_head_active = 1'b0;
        accepted_layout.q_head_slot = 1'b0;
        accepted_layout.panel_layout = LOCAL_PANEL_LAYOUT_NONE;
        case (active_command.layer_operator_index)
            LAYER_OPERATOR_ATTENTION_RMSNORM: begin
                accepted_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                accepted_layout.panel_layout = LOCAL_PANEL_LAYOUT_QKV;
            end
            LAYER_OPERATOR_QKV_PREPARATION: begin
                if (qkv_done_valid && !qkv_error) begin
                    accepted_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                end else begin
                    accepted_layout.phase =
                        LOCAL_MEMORY_PHASE_QKV_PREPARATION;
                    accepted_layout.kv_pair_enable = qkv_pair_active &&
                        !qkvo_kv_expanded_matmul;
                    accepted_layout.kv_second_batch = qkv_pair_active && qkv_second_batch;
                    accepted_layout.kv_first_batch_tokens = qkv_pair_active ? row_active_rows : 6'd0;
                    accepted_layout.qkv_head_staging = qkv_head_staging_layout;
                    accepted_layout.panel_layout = qkvo_kv_expanded_matmul ?
                        LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL : LOCAL_PANEL_LAYOUT_QKV;
                end
            end
            LAYER_OPERATOR_ATTENTION: begin
                accepted_layout.phase = attention_local_phase;
                accepted_layout.attention_pair_enable = attention_pair_active &&
                    !attention_q_head_precompute_only;
                accepted_layout.attention_second_batch =
                    accepted_layout.attention_pair_enable &&
                    attention_active_second_batch;
                accepted_layout.attention_batch_index =
                    accepted_layout.attention_pair_enable ?
                    attention_active_batch_index : 2'd0;
                accepted_layout.attention_score_words =
                    accepted_layout.attention_pair_enable ?
                    attention_pair_score_words : 8'd0;
                accepted_layout.active_panel = attention_operand_req.active_panel;
                accepted_layout.qkv_head_staging = qkv_head_staging_layout;
                accepted_layout.q_head_active = attention_q_head_active;
                accepted_layout.q_head_slot = attention_q_head_slot;
                accepted_layout.panel_layout =
                    qkvo_query_requires_expanded_layout ?
                        LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL :
                        LOCAL_PANEL_LAYOUT_QKV;
            end
            LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
            LAYER_OPERATOR_FFN_GATE_MATMUL,
            LAYER_OPERATOR_FFN_UP_MATMUL: begin
                accepted_layout.phase =
                    ffn_batch_pair_config.enable ? LOCAL_MEMORY_PHASE_MATMUL :
                    LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN;
                accepted_layout.panel_layout =
                    ffn_pair_requires_expanded_layout ?
                        LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL :
                        LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_RESIDUAL_SPILL,
            LAYER_OPERATOR_FFN_RMSNORM: begin
                accepted_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
                accepted_layout.panel_layout = LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_HIDDEN_REFILL:
                accepted_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
            LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE: begin
                accepted_layout.phase = ffn_batch_pair_config.enable ?
                    LOCAL_MEMORY_PHASE_R4_TWO_TOKENS : matmul_row_config.mixed_group_enable ?
                    LOCAL_MEMORY_PHASE_R4 : LOCAL_MEMORY_PHASE_ELEMENTWISE;
                accepted_layout.panel_layout = ffn_batch_pair_config.enable ?
                    LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL : LOCAL_PANEL_LAYOUT_MIXED;
            end
            LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL: begin
                accepted_layout.phase = LOCAL_MEMORY_PHASE_MATMUL;
                accepted_layout.panel_layout = ffn_batch_pair_config.enable ?
                    LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL : LOCAL_PANEL_LAYOUT_MIXED;
            end
            default: begin end
        endcase
    end

    always_comb begin : current_layout_requirement
        requested_layout = '0;
        requested_layout.phase = LOCAL_MEMORY_PHASE_IDLE;
        requested_layout.active_panel = 1'b0;
        requested_layout.qkv_head_staging = 1'b0;
        requested_layout.q_head_active = 1'b0;
        requested_layout.q_head_slot = 1'b0;
        requested_layout.panel_layout = LOCAL_PANEL_LAYOUT_NONE;
        if (execution_memory_stage == EXEC_MEMORY_FORWARD_POSTPROCESS) begin
            requested_layout.phase = LOCAL_MEMORY_PHASE_FINAL_OUTPUT;
        end else if (execution_hidden_source_selected) begin
            requested_layout.phase = LOCAL_MEMORY_PHASE_RMSNORM;
        end else if (active_command_valid) begin
            requested_layout = accepted_layout;
        end else if (execution_block_schedule_active) begin
            requested_layout = pending_initial_layout;
        end
    end

    assign local_layout_update_valid = current_layout != requested_layout;
    assign attention_second_batch = active_command_valid &&
        active_command.layer_operator_index == LAYER_OPERATOR_ATTENTION && attention_pair_enable &&
        attention_active_second_batch;
    assign attention_batch_index = active_command_valid &&
        active_command.layer_operator_index == LAYER_OPERATOR_ATTENTION && attention_pair_enable ?
            attention_active_batch_index : 2'd0;
    assign memory_abort_request = 1'b0;
    assign pending_layout_matches = current_layout == pending_initial_layout;
    assign pending_start_enable = !active_command_valid &&
        execution_block_schedule_active && layer_entry_valid &&
        layer_entry_index == pending_command.layer[4:0] &&
        pending_layout_matches;
    assign pending_start_ready[COMMAND_START_RMS] = rms_start_ready;
    assign pending_start_ready[COMMAND_START_QKV] = qkv_start_ready;
    assign pending_start_ready[COMMAND_START_ATTENTION] = attention_start_ready;
    assign pending_start_ready[COMMAND_START_MATMUL] = matmul_start_ready;
    assign pending_start_ready[COMMAND_START_HIDDEN] = hidden_start_ready;
    assign pending_start_ready[COMMAND_START_ELEMENTWISE] =
        elementwise_start_ready;
    assign operator_command_ready = pending_start_enable &&
        |(pending_start_select & pending_start_ready);

    assign command_child_start_fire[COMMAND_START_RMS] =
        block_rms_start_valid && rms_start_ready && !rms_after_spill_pending;
    assign command_child_start_fire[COMMAND_START_QKV] =
        qkv_start_valid && qkv_start_ready;
    assign command_child_start_fire[COMMAND_START_ATTENTION] =
        attention_start_valid && attention_start_ready;
    assign command_child_start_fire[COMMAND_START_MATMUL] =
        block_matmul_start_valid && matmul_start_ready;
    assign command_child_start_fire[COMMAND_START_HIDDEN] =
        ffn_hidden_start_valid && hidden_start_ready;
    assign command_child_start_fire[COMMAND_START_ELEMENTWISE] =
        elementwise_start_valid && elementwise_start_ready;

    always_comb begin : accepted_command_completion
        command_child_done_valid = 1'b0;
        command_child_done_error = 1'b0;
        command_child_done_error_id = 16'd0;
        if (active_command_valid) begin
            case (active_command.layer_operator_index)
                LAYER_OPERATOR_ATTENTION_RMSNORM,
                LAYER_OPERATOR_FFN_RMSNORM: begin
                    command_child_done_valid = rms_done_valid ||
                        (rms_after_spill_pending && qkv_residual_spill_complete && qkv_residual_spill_error);
                    command_child_done_error = rms_error ||
                        (rms_after_spill_pending && qkv_residual_spill_error);
                    command_child_done_error_id = rms_after_spill_pending && qkv_residual_spill_error ?
                        {12'h150, qkv_residual_spill_error_id} : rms_error ?
                        {12'h110, rms_error_id} : 16'd0;
                end
                LAYER_OPERATOR_QKV_PREPARATION: begin
                    command_child_done_valid =
                        (qkv_residual_spill_error && qkv_abort_ack) ||
                        (qkv_done_valid && qkv_residual_spill_complete);
                    command_child_done_error = qkv_error ||
                        qkv_residual_spill_error;
                    command_child_done_error_id = qkv_error ?
                        {8'h12, qkv_error_id} :
                        qkv_residual_spill_error ?
                            {12'h150, qkv_residual_spill_error_id} : 16'd0;
                end
                LAYER_OPERATOR_ATTENTION: begin
                    command_child_done_valid = attention_done_valid;
                    command_child_done_error = attention_error;
                    command_child_done_error_id =
                        {8'h13, attention_error_id};
                end
                LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
                LAYER_OPERATOR_FFN_GATE_MATMUL,
                LAYER_OPERATOR_FFN_UP_MATMUL,
                LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL: begin
                    command_child_done_valid = matmul_done_valid;
                    command_child_done_error = matmul_error;
                    command_child_done_error_id = matmul_error ?
                        16'h1601 : 16'd0;
                end
                LAYER_OPERATOR_FFN_RESIDUAL_SPILL,
                LAYER_OPERATOR_FFN_HIDDEN_REFILL: begin
                    command_child_done_valid = hidden_done_valid &&
                        hidden_done_for_ffn;
                    command_child_done_error = hidden_error;
                    command_child_done_error_id = hidden_error ?
                        {12'h150, hidden_error_id} : 16'd0;
                end
                LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE: begin
                    command_child_done_valid = elementwise_done_valid;
                    command_child_done_error = elementwise_error;
                    command_child_done_error_id = elementwise_error ?
                        {12'h170, elementwise_error_id} : 16'd0;
                end
                default: begin end
            endcase
        end
    end

    assign command_done_can_complete = !command_child_done_valid ||
        !command_child_done_error || matmul_prefetch_empty;
    assign operator_command_done_valid = command_child_done_valid &&
        command_done_can_complete;
    assign operator_command_done_error = command_child_done_error;
    assign operator_command_done_error_id = command_child_done_error_id;

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(active_command_valid && operator_command_ready))
                else $error("operator_subsystem accepted a second command while a command was active");
            assert (!operator_command_done_valid || active_command_valid)
                else $error("operator_subsystem reported completion without an active command");
            assert ($onehot0(matmul_prefetch_source_valid))
                else $error("operator_subsystem selected multiple Matmul prefetch sources");
            assert ($onehot0({execution_hidden_source_selected,
                              ffn_hidden_source_selected,
                              qkv_residual_spill_start_valid,
                              post_hidden_source_selected}))
                else $error("operator_subsystem selected multiple hidden command sources");
            if (matmul_start_valid)
                assert ($onehot({qkv_phase_active && qkv_matmul_start_valid,
                                 !qkv_phase_active &&
                                     block_matmul_start_valid}))
                    else $error("operator_subsystem Matmul start source was not one-hot");
            if (qkv_phase_active && matmul_start_valid && matmul_start_ready)
                assert (qkv_matmul_start_valid && qkv_matmul_start_ready)
                    else $error("operator_subsystem QKV Matmul start handshake disagreed across the layout boundary");
            if (qkvo_kv_requires_expanded_layout && matmul_start_valid && matmul_start_ready)
                assert (current_layout.panel_layout == LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL &&
                        !current_layout.kv_pair_enable && !local_layout_update_valid)
                    else $error("operator_subsystem grouped K/V Matmul started before expanded activation layout was installed");
            if (active_command_valid &&
                active_command.layer_operator_index == LAYER_OPERATOR_QKV_PREPARATION &&
                qkv_done_valid && !qkv_residual_spill_complete)
                assert (!qkv_done_ready)
                    else $error("operator_subsystem consumed QKV completion before residual spill drained");
            if (hidden_start_valid)
                assert ($onehot({qkv_residual_spill_start_valid,
                                 execution_hidden_source_selected &&
                                     execution_hidden_start_valid,
                                 ffn_hidden_source_selected &&
                                     ffn_hidden_start_valid,
                                 post_hidden_source_selected &&
                                     post_hidden_start_valid}))
                    else $error("operator_subsystem hidden start source was not one-hot");
            if (operator_command_valid && operator_command_ready) begin
                assert ($onehot(pending_start_select))
                    else $error("operator_subsystem accepted a command without one child selection");
                assert ($onehot(command_child_start_fire))
                    else $error("operator_subsystem command did not start exactly one child");
            end
            if (active_command_valid && rms_after_spill_pending && block_rms_start_valid)
                assert (qkv_residual_spill_complete && !qkv_residual_spill_active && !qkv_residual_spill_error)
                    else $error("operator_subsystem started RMSNorm before original hidden was saved");
            assert ((|command_child_start_fire) ==
                    (operator_command_valid && operator_command_ready))
                else $error("operator_subsystem child start did not match command acceptance");
            if ($past(operator_command_done_valid &&
                      !operator_command_done_ready))
                assert (operator_command_done_valid &&
                        execution_completion == stalled_operator_completion)
                    else $error("operator_subsystem changed a stalled completion");
            if (operator_command_done_valid && !operator_command_done_ready)
                stalled_operator_completion <= execution_completion;
            if (command_child_done_valid && command_child_done_error &&
                !matmul_prefetch_empty) begin
                assert (matmul_prefetch_cancel)
                    else $error("operator_subsystem did not cancel Matmul prefetch after child error");
                assert (!matmul_prefetch_valid)
                    else $error("operator_subsystem issued Matmul prefetch while draining child error");
                assert (!operator_command_done_valid)
                    else $error("operator_subsystem reported child error before Matmul prefetch drained");
            end
            if (operator_command_done_valid && operator_command_done_error)
                assert (matmul_prefetch_empty)
                    else $error("operator_subsystem reported error with Matmul prefetch still active");
        end
    end
`endif

endmodule

`default_nettype wire
