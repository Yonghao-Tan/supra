`timescale 1ns/1ps
`default_nettype none

// Verification boundary for supra_top. The DDR model and all observation
// signals stay here and are excluded from production.f.
module supra_top_tb #(
    // Production weight regions extend beyond the small zero-filled reference data.
    // Keep the logical DDR address range separate from the allocated backing.
    parameter integer DDR_MEMORY_BYTES = 512 * 1024 * 1024,
    parameter integer DDR_BACKING_BYTES = 128 * 1024 * 1024,
    parameter integer ATTENTION_HEAD_COUNT = 32,
    parameter integer HIDDEN_FEATURES = 4096,
    parameter integer FFN_FEATURES = 12288,
    parameter bit USE_DRAMSIM3 = 1'b0,
    parameter integer DRAMSIM3_CORE_FREQUENCY_MHZ = 500,
    parameter logic [63:0] DDR_BASE_ADDRESS = 64'h0000_0000_1000_0000
) (
    input logic clk, input logic rst,
    input logic run_valid, output logic run_ready,
    input logic [1:0] run_scope,
    input logic [63:0] run_config_address, input logic [31:0] run_id,
    input logic random_stall, input logic performance_config,
    input logic inject_read_response_error, input logic inject_write_response_error,
    output logic completion_valid, input logic completion_ready,
    output logic [31:0] completion_id, output logic completion_error,
    output logic [15:0] completion_error_id,
    output logic post_completion_observe_valid,
    output logic post_completion_observe_block_complete,
    output logic [3:0] observed_command, output logic [5:0] observed_layer,
    output logic operator_command_observe_valid,
    output logic operator_command_observe_ready,
    output logic observed_command_cache_release,
    output logic observed_command_qkv_residual_spill,
    output logic observed_command_rms_qkv_prefetch_q,
    output logic observed_command_q_head_preprocess,
    output logic debug_qkv_residual_spill_active,
    output logic [5:0] debug_row_token_batch_index,
    output logic [5:0] debug_row_active_rows,
    output logic [11:0] debug_row_total_rows,
    output logic debug_row_last_round,
    output logic [4:0] debug_execution_state,
    output logic debug_execution_hidden_start_valid,
    output logic debug_execution_hidden_start_ready,
    output logic debug_execution_hidden_start_write,
    output logic debug_execution_hidden_start_residual,
    output logic [63:0] debug_execution_hidden_start_address,
    output logic [31:0] debug_execution_hidden_start_bytes,
    output logic [3:0] debug_execution_config_loader_state,
    output logic [4:0] debug_token_metadata_state,
    output logic debug_token_metadata_done_valid,
    output logic debug_token_metadata_done_error,
    output logic [15:0] debug_token_metadata_done_error_id,
    output logic debug_token_metadata_request_valid,
    output logic debug_token_metadata_request_ready,
    output logic debug_token_metadata_data_valid,
    output logic debug_token_metadata_data_ready,
    output logic debug_token_metadata_completion_done,
    output logic debug_token_metadata_completion_error,
    output logic [1023:0] debug_matmul_frontend_source_rsp_values,
    output logic [63:0] accepted_command_count, completed_command_count,
    output logic [63:0] accepted_read_bytes, accepted_write_bytes,
    output logic [63:0] accepted_read_bursts, accepted_write_bursts,
    output logic [63:0] rms_gamma_dma_accepted_count,
    output logic [63:0] rms_gamma_vector_request_count,
    output logic [4:0] read_outstanding_high_water, write_outstanding_high_water,
    output logic [4:0] debug_read_outstanding, debug_write_outstanding,
    output logic debug_ddr_paths_drained,
    output logic debug_ddr_read_requested, debug_ddr_write_requested,
    output logic [63:0] post_weight_bytes, post_scale_bytes,
    output logic [63:0] post_pe_accepted_count,
    output logic [63:0] post_candidate_wait_cycles,
    output logic [13:0] post_lm_panel_count, post_candidate_panel_count,
    output logic post_writer_running, post_lm_waiting_for_panel,
    output logic post_active,
    output logic [6:0] post_norm_group_base,
    output logic post_quant_observe_valid, post_scale_observe_valid,
    output logic [6:0] post_quant_observe_row, post_scale_observe_row,
    output logic [15:0] post_quant_observe_element,
    output logic [511:0] post_quant_observe_values,
    output logic [63:0] post_quant_observe_mask,
    output logic [127:0] post_scale_observe_values,
    output logic [7:0] post_scale_observe_mask,
    output logic post_int32_observe_valid, post_logit_observe_valid,
    output logic [6:0] post_int32_observe_row, post_logit_observe_row,
    output logic [16:0] post_int32_observe_vocab, post_logit_observe_vocab,
    output logic [2047:0] post_int32_observe_values,
    output logic [1023:0] post_logit_observe_values,
    output logic [63:0] post_int32_observe_mask, post_logit_observe_mask,
    output logic post_candidate_observe_valid,
    output logic [6:0] post_candidate_observe_row,
    output logic [16:0] post_candidate_observe_top1,
    output logic [15:0] post_candidate_observe_logit, post_candidate_observe_confidence,
    output logic [15:0] post_candidate_observe_probability, post_candidate_observe_action,
    output logic [31:0] qkv_accepted_matmul_count, qkv_accepted_rope_count,
    output logic [31:0] qkv_accepted_head_read_count, qkv_accepted_quant_count,
    output logic [31:0] qkv_accepted_quantized_commit_count,
    output logic [31:0] qkv_completed_cache_write_count,
    output logic [15:0] qkv_outstanding_cache_writes,
    output logic [63:0] matmul_accepted_issue_count, matmul_accepted_result_count,
    output logic [63:0] matmul_base_request_count, matmul_enhancement_request_count,
    output logic [63:0] matmul_scale_request_count, matmul_base_request_bytes,
    output logic [63:0] matmul_enhancement_request_bytes, matmul_scale_request_bytes,
    output logic [3:0] debug_matmul_state, output logic [5:0] debug_matmul_current_row,
    output logic debug_matmul_frontend_done,
    output logic debug_matmul_source_req_valid,
    output logic debug_matmul_source_req_ready,
    output logic debug_matmul_source_rsp_valid,
    output logic debug_matmul_source_rsp_ready,
    output logic debug_matmul_activation_writer_done,
    output logic debug_matmul_output_controller_done,
    output logic [1:0] debug_matmul_panel_valid, debug_matmul_panel_loading,
    output logic [1:0] debug_matmul_panel_computing,
    output logic [15:0] debug_matmul_next_load_stripe,
    output logic [15:0] debug_matmul_current_stripe,
    output logic debug_matmul_current_panel_valid,
    output logic debug_matmul_current_panel,
    output logic [31:0] debug_matmul_panel_stripe,
    output logic [31:0] debug_matmul_panel_operator,
    output logic debug_matmul_acquire_valid,
    output logic debug_matmul_acquire_ready,
    output logic debug_matmul_acquire_panel,
    output logic [15:0] debug_matmul_acquire_stripe,
    output logic debug_matmul_release_valid,
    output logic debug_matmul_release_ready,
    output logic debug_matmul_pending_release,
    output logic [2:0] debug_matmul_schedule_state,
    output logic [15:0] debug_matmul_schedule_n_base,
    output logic [15:0] debug_matmul_schedule_k_base,
    output logic debug_matmul_issue_valid,
    output logic debug_matmul_issue_ready,
    output logic [15:0] debug_matmul_issue_stripe,
    output logic debug_matmul_reader_scale_valid,
    output logic debug_matmul_reader_scale_match,
    output logic [1:0] debug_matmul_reader_response_pending,
    output logic [2:0] debug_matmul_reader_fifo_occupancy,
    output logic [2:0] debug_attention_operand_fifo_occupancy,
    output logic [1:0] debug_attention_score_fifo_occupancy,
    output logic [2:0] debug_qkv_head_tile_fifo_occupancy,
    output logic [2:0] debug_local_attention_scratch_fifo_occupancy,
    output logic [2:0] debug_local_rms_tile_fifo_occupancy,
    output logic [2:0] debug_local_rms_scratch_fifo_occupancy,
    output logic debug_matmul_reader_address_valid,
    output logic debug_matmul_read_bundle_ready,
    output logic [1:0] debug_shared_pe_accum_result_occupancy,
    output logic [2:0] debug_matmul_descriptor_count,
    output logic debug_matmul_result_valid,
    output logic debug_matmul_result_ready,
    output logic debug_matmul_fragment_valid,
    output logic debug_matmul_fragment_ready,
    output logic debug_matmul_tile_req_valid,
    output logic debug_matmul_tile_req_ready,
    output logic [4:0] debug_matmul_output_controller_state,
    output logic [15:0] debug_matmul_output_controller_stripe,
    output logic [5:0] debug_matmul_output_controller_input_row,
    output logic [5:0] debug_matmul_output_controller_row,
    output logic debug_matmul_residual_read_active,
    output logic debug_matmul_residual_read_is_prefetch,
    output logic debug_matmul_residual_prefetch_pending,
    output logic debug_matmul_residual_prefetched_valid,
    output logic debug_matmul_residual_request_valid,
    output logic debug_matmul_residual_request_ready,
    output logic debug_matmul_residual_data_valid,
    output logic debug_matmul_residual_data_ready,
    output logic debug_matmul_residual_request_done,
    output logic debug_matmul_residual_local_read_issue,
    output logic [1:0] debug_matmul_residual_local_read_pending,
    output logic debug_matmul_residual_local_response_pending,
    output logic [1:0] debug_matmul_memory_stage,
    output logic debug_matmul_aux_start_valid,
    output logic debug_matmul_aux_start_ready,
    output logic debug_matmul_aux_residual_active,
    output logic debug_matmul_aux_request_valid,
    output logic debug_matmul_aux_request_ready,
    output logic [31:0] debug_matmul_aux_request_bytes,
    output logic debug_matmul_aux_read_valid,
    output logic debug_matmul_aux_read_ready,
    output logic debug_matmul_aux_data_valid,
    output logic debug_matmul_aux_data_ready,
    output logic debug_matmul_aux_done,
    output logic debug_matmul_aux_error,
    output logic [2:0] debug_matmul_writer_state,
    output logic debug_matmul_writer_active,
    output logic debug_matmul_writer_request_valid,
    output logic debug_matmul_writer_request_ready,
    output logic debug_matmul_writer_data_valid,
    output logic debug_matmul_writer_data_ready,
    output logic debug_matmul_writer_write_valid,
    output logic debug_matmul_writer_write_ready,
    output logic debug_matmul_writer_done,
    output logic [1:0] debug_matmul_loader_plane,
    output logic debug_matmul_dma_request_outstanding,
    output logic debug_matmul_dma_payload_complete,
    output logic [31:0] debug_matmul_dma_response_remaining,
    output logic debug_matmul_dma_req_valid, debug_matmul_dma_req_done,
    output logic debug_matmul_dma_req_error,
    output logic [5:0] integration_head_count,
    output logic [15:0] integration_ffn_features,
    output logic [63:0] attention_completed_head_count_debug,
    output logic [63:0] attention_qk_count_debug, attention_pv_count_debug,
    output logic [3:0] current_local_phase_debug, requested_local_phase_debug,
    output logic [3:0] debug_read_active_client, debug_selected_read_active_client,
    output logic [3:0] debug_qkv_state, debug_rms_state,
    output logic [5:0] qkv_state,
    output logic [3:0] elementwise_prefetch_state,
    output logic [3:0] r4_controller_state,
    output logic [2:0] r4_h1024_state,
    output logic [3:0] r4_h12_state,
    output logic r4_product_req_valid, r4_product_req_ready,
    output logic r4_product_req_write,
    output logic r4_arithmetic_req_valid, r4_arithmetic_req_ready,
    output logic elementwise_read_request_valid,
    output logic elementwise_read_request_ready,
    output logic elementwise_read_data_valid,
    output logic elementwise_read_data_ready,
    output logic [5:0] debug_qkv_current_head,
    output logic [1:0] debug_qkv_select,
    output logic [1:0] elementwise_prefetch_ready_count,
    output logic [3:0] qkv_commit_state,
    output logic [63:0] debug_read_address, output logic [31:0] debug_read_bytes,
    output logic [7:0] debug_read_tag,
    output logic debug_read_second_span_valid,
    output logic [63:0] debug_read_second_span_address,
    output logic [31:0] debug_read_pair_stride,
    output logic [10:0] debug_read_pair_count,
    output logic debug_memory_read_request_ready,
    output logic debug_width_read_active, debug_width_read_done_pending,
    output logic debug_wide_memory_read_request_ready,
    output logic debug_read_alignment_ready,
    output logic [2:0] debug_read_alignment_state,
    output logic debug_axi_read_active, debug_axi_read_idle,
    output logic debug_layer_entry_valid, debug_layer_done_valid,
    output logic [4:0] debug_layer_entry_index,
    output logic debug_layer_done_error,
    output logic [15:0] debug_layer_done_error_id,
    output logic [63:0] debug_compute_progress_count,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        bank_port_overflow,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_MACRO_COUNT-1:0]
        bank_same_address_conflict,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        bank_role_mismatch,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_compute_a_req_valid,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_compute_a_write,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_compute_b_req_valid,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_compute_b_write,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_local_dma_req_valid,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_local_dma_write,
    output logic rms_trace_sample_valid, output logic [5:0] rms_trace_sample_row,
    output logic [15:0] rms_trace_sample_element, output logic [1023:0] rms_trace_sample_data,
    output logic [127:0] rms_trace_sample_byte_enable,
    output logic matmul_int32_observe_valid,
    output logic [2:0] matmul_quantization_second_batch,
    output logic [2047:0] matmul_int32_observe_accumulators,
    output logic [63:0] matmul_int32_observe_mask,
    output logic [1:0] matmul_int32_observe_mode,
    output logic matmul_int32_observe_last_k_step,
    output logic [15:0] matmul_int32_observe_tag,
    output logic matmul_rescale_observe_valid,
    output logic [1023:0] matmul_rescale_observe_values,
    output logic [63:0] matmul_rescale_observe_mask,
    output logic [15:0] matmul_rescale_observe_tag,
    output logic matmul_residual_add_observe_valid,
    output logic [255:0] matmul_residual_add_observe_values,
    output logic [255:0] matmul_residual_add_observe_paired_values,
    output logic [15:0] matmul_residual_add_observe_lane_mask,
    output logic [15:0] matmul_residual_add_observe_tag,
    output logic matmul_bf16_observe_valid,
    output logic [31:0] matmul_bf16_observe_address,
    output logic [127:0] matmul_bf16_observe_data,
    output logic [15:0] matmul_bf16_observe_byte_enable,
    output logic matmul_ddr_observe_valid,
    output logic [63:0] matmul_ddr_observe_address,
    output logic [127:0] matmul_ddr_observe_data,
    output logic [15:0] matmul_ddr_observe_byte_enable,
    output logic matmul_ddr_observe_first, matmul_ddr_observe_last,
    output logic ffn_fragment_valid,
    output logic [2:0] ffn_fragment_batch,
    output logic [5:0] ffn_fragment_row,
    output logic [15:0] ffn_fragment_column,
    output logic ffn_fragment_up,
    output logic [127:0] ffn_fragment_data,
    output logic ffn_silu_valid,
    output logic [127:0] ffn_silu_data,
    output logic elementwise_trace_sample_valid, elementwise_trace_sample_pass,
    output logic [10:0] elementwise_trace_sample_stripe,
    output logic [5:0] elementwise_trace_sample_row_base,
    output logic [63:0] elementwise_trace_sample_lane_mask,
    output logic [1023:0] elementwise_trace_sample_silu,
    output logic [1023:0] elementwise_trace_sample_product,
    output logic [5:0] elementwise_debug_state,
    output logic [127:0] elementwise_debug_gate_word,
    output logic [127:0] elementwise_debug_up_word,
    output logic [127:0] elementwise_debug_read_word,
    output logic elementwise_debug_bf16_req_fire,
    output logic [2:0] elementwise_debug_bf16_req_operation,
    output logic [15:0] elementwise_debug_bf16_req_value0,
    output logic [15:0] elementwise_debug_bf16_req_factor0,
    output logic [15:0] elementwise_debug_bf16_req_factor1,
    output logic elementwise_debug_bf16_rsp_fire,
    output logic [15:0] elementwise_debug_bf16_rsp_value0,
    output logic [2:0] elementwise_debug_bf16_active_client,
    output logic [7:0] elementwise_debug_bf16_outstanding,
    output logic elementwise_quantized_observe_valid,
    output logic [5:0] elementwise_quantized_observe_row,
    output logic [15:0] elementwise_quantized_observe_element,
    output logic [511:0] elementwise_quantized_observe_values,
    output logic [63:0] elementwise_quantized_observe_lane_mask,
    output logic elementwise_scale_observe_valid,
    output logic [5:0] elementwise_scale_observe_row,
    output logic [7:0] elementwise_scale_observe_row_mask,
    output logic [127:0] elementwise_scale_observe_values,
    output logic hidden_ddr_observe_valid, hidden_ddr_observe_qkv,
    output logic hidden_ddr_observe_preserve,
    output logic [5:0] hidden_ddr_observe_preserve_layer,
    output logic hidden_ddr_observe_ffn,
    output logic [127:0] hidden_ddr_observe_data,
    output logic [15:0] hidden_ddr_observe_byte_enable,
    output logic hidden_ddr_observe_last,
    output logic hidden_local_write_observe_valid,
    output logic hidden_local_write_observe_preserved_refill,
    output logic [5:0] hidden_local_write_observe_physical_row,
    output logic [8:0] hidden_local_write_observe_channel_word,
    output logic [127:0] hidden_local_write_observe_data,
    output logic [15:0] hidden_local_write_observe_byte_enable,
    output logic rope_bf16_observe_valid, rope_bf16_observe_is_k,
    output logic [15:0] rope_bf16_observe_lane_mask,
    output logic [287:0] rope_bf16_observe_lane_address,
    output logic [255:0] rope_bf16_observe_lane_data,
    output logic qkv_quantized_observe_valid,
    output logic [1:0] qkv_quantized_observe_qkv_select,
    output logic [5:0] qkv_quantized_observe_head, qkv_quantized_observe_row,
    output logic [10:0] qkv_quantized_observe_token_position,
    output logic qkv_quantized_observe_half,
    output logic [511:0] qkv_quantized_observe_values,
    output logic [63:0] qkv_quantized_observe_lane_mask,
    output logic [15:0] qkv_quantized_observe_scale,
    output logic qkv_cache_map_write_accepted,
    output logic [5:0] qkv_cache_map_write_head,
    output logic [5:0] qkv_cache_map_write_physical_row,
    output logic [10:0] qkv_cache_map_write_logical_slot,
    output logic [4:0] qkv_cache_map_write_chunk,
    output logic [63:0] qkv_cache_map_write_k_address,
    output logic [63:0] qkv_cache_map_write_v_address,
    output logic [63:0] qkv_cache_map_write_k_scale_address,
    output logic attention_int32_observe_valid,
    output logic [2047:0] attention_int32_observe_accumulators,
    output logic [63:0] attention_int32_observe_mask,
    output logic attention_int32_observe_last_k_step,
    output logic [15:0] attention_int32_observe_tag,
    output logic [1023:0] attention_int32_observe_activation_scales,
    output logic [1023:0] attention_int32_observe_weight_scales,
    output logic attention_score_observe_valid,
    output logic [5:0] attention_score_observe_head,
    output logic [5:0] attention_score_observe_query_base,
    output logic [11:0] attention_score_observe_key_base,
    output logic [1023:0] attention_score_observe_data,
    output logic [127:0] attention_score_observe_byte_enable,
    output logic softmax_bf16_observe_valid,
    output logic [5:0] softmax_bf16_observe_head, softmax_bf16_observe_row,
    output logic [11:0] softmax_bf16_observe_key,
    output logic [1023:0] softmax_bf16_observe_data,
    output logic [127:0] softmax_bf16_observe_byte_enable,
    output logic softmax_quantized_observe_valid,
    output logic [5:0] softmax_quantized_observe_head, softmax_quantized_observe_row,
    output logic [11:0] softmax_quantized_observe_key,
    output logic [511:0] softmax_quantized_observe_data,
    output logic [63:0] softmax_quantized_observe_byte_enable,
    output logic softmax_scale_observe_valid,
    output logic [5:0] softmax_scale_observe_head, softmax_scale_observe_row,
    output logic [127:0] softmax_scale_observe_value,
    output logic [7:0] softmax_scale_observe_row_mask,
    output logic context_bf16_observe_valid,
    output logic [9:0] context_bf16_observe_address,
    output logic [127:0] context_bf16_observe_data,
    output logic [15:0] context_bf16_observe_byte_enable,
    output logic context_ddr_observe_valid,
    output logic [127:0] context_ddr_observe_data,
    output logic [15:0] context_ddr_observe_byte_enable,
    output logic context_ddr_observe_last,
    output logic rms_write_valid, rms_write_ready,
    output logic [5:0] rms_write_row_base,
    output logic [15:0] rms_write_element,
    output logic [1023:0] rms_write_data,
    output logic [63:0] rms_write_lane_mask,
    output logic rms_source_valid, rms_source_ready,
    output logic [5:0] rms_source_row_base,
    output logic [1023:0] rms_source_values,
    output logic [127:0] rms_source_gamma,
    output logic [63:0] rms_source_lane_mask,
    output logic [15:0] rms_source_tag,
    output logic rms_scratch_write_valid,
    output logic rms_scratch_write_ready,
    output logic rms_scratch_write_bank,
    output logic [9:0] rms_scratch_write_address,
    output logic [127:0] rms_scratch_write_data,
    output logic [15:0] rms_scratch_write_byte_enable,
    output logic rms_scratch_read_valid,
    output logic rms_scratch_read_ready,
    output logic rms_scratch_read_bank,
    output logic [9:0] rms_scratch_read_left_address,
    output logic [9:0] rms_scratch_read_right_address,
    output logic [15:0] rms_scratch_read_tag,
    output logic rms_scratch_rsp_valid,
    output logic rms_scratch_rsp_ready,
    output logic [127:0] rms_scratch_rsp_left_data,
    output logic [127:0] rms_scratch_rsp_right_data,
    output logic [15:0] rms_scratch_rsp_tag,
    output logic matmul_scale_valid, matmul_scale_ready,
    output logic [5:0] matmul_scale_row_base,
    output logic [7:0] matmul_scale_row_mask,
    output logic [127:0] matmul_scale_values,
    output logic matmul_activation_observe_valid,
    output logic [5:0] matmul_activation_observe_row,
    output logic [15:0] matmul_activation_observe_element,
    output logic [511:0] matmul_activation_observe_values,
    output logic [63:0] matmul_activation_observe_lane_mask,
    output logic qkv_quant_req_valid, qkv_quant_req_ready,
    output logic [1023:0] qkv_quant_req_values,
    output logic [4:0] attention_controller_state,
    output logic shared_pe_req_valid, shared_pe_req_ready,
    output logic [1:0] shared_pe_req_mode,
    output logic shared_pe_req_first,
    output logic shared_pe_req_last,
    output logic [7:0] shared_pe_req_row_mask,
    output logic [31:0] shared_pe_req_k_mask,
    output logic [7:0] shared_pe_req_col_mask,
    output logic [15:0] shared_pe_req_tag,
    output logic [1023:0] shared_pe_req_activation_payload,
    output logic [1023:0] shared_pe_req_weight_payload,
    output logic [63:0] shared_pe_req_row0_activation,
    output logic [255:0] shared_pe_req_weight_payload_low,
    output logic bf16_request_observe_accepted,
    output logic [4:0] debug_bf16_request_clients,
    output logic [2:0] bf16_request_observe_operation,
    output logic clip_observe_valid, clip_scale_observe_valid,
    output logic [5:0] clip_observe_row,
    output logic [2:0] clip_observe_batch_index,
    output logic [15:0] clip_observe_tag,
    output logic [15:0] clip_observe_element,
    output logic [1023:0] clip_observe_input, clip_observe_output,
    output logic [63:0] clip_observe_lane_mask,
    output logic [127:0] clip_observe_maximum, clip_observe_limit,
    output logic [7:0] clip_observe_row_mask,
    output logic shared_pe_accum_result_valid,
    output logic shared_pe_accum_result_ready,
    output logic [1:0] matmul_issue_mode,
    output logic [5:0] matmul_issue_row_base,
    output logic [15:0] matmul_issue_k_base,
    output logic [15:0] matmul_issue_n_base,
    output logic matmul_issue_first, matmul_issue_last,
    output logic [15:0] matmul_issue_tag,
    output logic [15:0] matmul_issue_row_batch,
    output logic [31:0] matmul_issue_activation_address,
    output logic [15:0] matmul_activation_word_index,
    output logic [3:0] matmul_activation_word_count,
    output logic matmul_panel_read_select,
    output logic [11:0] matmul_panel_read_mask,
    output logic [119:0] matmul_panel_read_address,
    output logic matmul_read_response_valid,
    output logic matmul_accum_result_last,
    output logic [1:0] matmul_accum_result_mode,
    output logic [15:0] matmul_accum_result_tag,
    output logic matmul_rescale_req_valid,
    output logic matmul_rescale_req_ready,
    output logic [15:0] matmul_rescale_req_tag,
    output logic matmul_rescale_rsp_valid,
    output logic matmul_rescale_rsp_ready,
    output logic [15:0] matmul_rescale_rsp_tag,
    output logic shared_rescale_req_valid,
    output logic shared_rescale_req_ready,
    output logic [15:0] shared_rescale_req_tag,
    output logic shared_rescale_rsp_valid,
    output logic shared_rescale_rsp_ready,
    output logic [15:0] shared_rescale_rsp_tag,
    output logic [15:0] matmul_fragment_tag,
    output logic [5:0] matmul_fragment_row,
    output logic [15:0] matmul_fragment_channel,
    output logic matmul_combine_write_valid,
    output logic matmul_combine_write_ready,
    output logic [9:0] matmul_combine_write_address,
    output logic matmul_combine_read_valid,
    output logic matmul_combine_read_ready,
    output logic [9:0] matmul_combine_read_address,
    output logic matmul_ddr_write_valid,
    output logic matmul_ddr_write_ready,
    output logic [63:0] matmul_ddr_write_address,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        local_sram_port_accept,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        local_sram_client_accept,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*6-1:0]
        local_sram_port_demand_count,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*42-1:0]
        local_sram_port_requester_mask,
    output logic [41:0] local_sram_requester_coverage,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        local_sram_port_write,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        local_sram_port_address,
    output logic matmul_source_req_valid,
    output logic matmul_source_req_ready,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        matmul_source_port_mask,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        matmul_source_port_address,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        matmul_operand_port_mask,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        matmul_compute_port_address,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        matmul_aux_port_mask,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        matmul_aux_port_address,
    output logic [5:0] matmul_combine_write_port,
    output logic [5:0] matmul_combine_read_port,
    output logic matmul_output_controller_cfg_valid,
    output logic matmul_output_controller_cfg_ready,
    output logic matmul_local_output_write_valid,
    output logic matmul_local_output_write_ready,
    output logic [31:0] matmul_local_output_write_address,
    output logic matmul_fragment_active,
    output logic [3:0] shared_rescale_state,
    output logic matmul_panel_memory_write_valid, matmul_panel_memory_write_ready,
    output logic matmul_dma_rsp_valid, matmul_dma_rsp_ready,
    output logic matmul_read_bundle_valid,
    output logic [1:0] matmul_read_response_pending,
    output logic attention_operand_req_valid,
    output logic attention_operand_req_ready,
    output logic attention_operand_rsp_valid,
    output logic attention_operand_rsp_ready,
    output logic attention_panel_write_valid,
    output logic attention_panel_write_ready,
    output logic attention_panel_write_target,
    output logic attention_panel_write_quarter,
    output logic attention_panel_write_bank,
    output logic [9:0] attention_panel_write_address,
    output logic [31:0] attention_panel_write_byte_enable,
    output logic [15:0] attention_panel_write_tag,
    output logic attention_scale_write_valid,
    output logic attention_scale_write_ready,
    output logic [10:0] attention_scale_write_token_base,
    output logic [15:0] attention_scale_write_byte_enable,
    output logic [15:0] attention_scale_write_tag,
    output logic attention_operand_req_is_pv,
    output logic [5:0] attention_operand_req_query_base,
    output logic [11:0] attention_operand_req_output_base,
    output logic [11:0] attention_operand_req_k_base,
    output logic [7:0] attention_operand_req_row_mask,
    output logic [15:0] attention_operand_req_tag,
    output logic [5:0] attention_active_head,
    output logic attention_active_panel,
    output logic attention_active_quarter,
    output logic attention_loader_start_pending,
    output logic attention_loaded_valid,
    output logic [1:0] attention_loader_select,
    output logic [1:0] attention_loaded_select,
    output logic attention_cache_dma_owned,
    output logic attention_cache_dma_done_seen,
    output logic [31:0] attention_cache_dma_expected_bytes,
    output logic [31:0] attention_cache_dma_received_bytes,
    output logic [1:0] attention_cache_buffer_valid,
    output logic attention_cache_drain_active,
    output logic attention_cache_drain_word_pending,
    output logic attention_cache_drain_scale_pending,
    output logic attention_cache_lookup_outstanding,
    output logic attention_cache_terminal_error,
    output logic [3:0] attention_cache_terminal_error_id,
    output logic attention_dma_start_valid,
    output logic attention_dma_start_ready,
    output logic attention_dma_request_valid,
    output logic attention_dma_request_ready,
    output logic attention_dma_data_valid,
    output logic attention_dma_data_ready,
    output logic attention_dma_done,
    output logic attention_dma_error,
    output logic [3:0] attention_cache_state,
    output logic [2:0] attention_matmul_state,
    output logic [2:0] softmax_quant_state,
    output logic [4:0] softmax_row_state,
    output logic softmax_source_req_valid, softmax_source_req_ready,
    output logic [1:0] softmax_source_req_pass,
    output logic [11:0] softmax_source_req_key,
    output logic [5:0] softmax_source_req_head,
    output logic [5:0] softmax_source_req_row_base,
    output logic [15:0] softmax_source_req_tag,
    output logic softmax_source_rsp_valid, softmax_source_rsp_ready,
    output logic [15:0] softmax_source_rsp_tag,
    output logic softmax_vector_req_valid, softmax_vector_req_ready,
    output logic [15:0] softmax_vector_req_tag,
    output logic softmax_vector_rsp_valid, softmax_vector_rsp_ready,
    output logic [15:0] softmax_vector_rsp_tag,
    output logic softmax_max_req_valid, softmax_max_req_ready,
    output logic [15:0] softmax_max_req_tag,
    output logic softmax_max_rsp_valid, softmax_max_rsp_ready,
    output logic [15:0] softmax_max_rsp_tag,
    output logic softmax_bf16_write_valid,
    output logic softmax_bf16_write_ready,
    output logic softmax_bf16_write_probability,
    output logic [15:0] softmax_bf16_write_tag,
    output logic [15:0] softmax_quantized_write_tag,
    output logic softmax_scratch_read_valid,
    output logic softmax_scratch_read_ready,
    output logic softmax_scratch_read_bank,
    output logic [9:0] softmax_scratch_read_left_address,
    output logic [9:0] softmax_scratch_read_right_address,
    output logic [15:0] softmax_scratch_read_tag,
    output logic softmax_scratch_write_valid,
    output logic softmax_scratch_write_ready,
    output logic softmax_scratch_write_bank,
    output logic [9:0] softmax_scratch_write_address,
    output logic [15:0] softmax_scratch_write_byte_enable,
    output logic attention_context_read_valid,
    output logic attention_context_read_ready,
    output logic [9:0] attention_context_read_address,
    output logic [15:0] attention_context_read_tag,
    output logic attention_loader_inflight, attention_matmul_busy,
    output logic attention_softmax_busy, attention_writer_busy,
    output logic attention_q_head_active,
    output logic attention_q_head_matmul_done,
    output logic attention_rescale_outstanding,
    output logic attention_result_pending,
    output logic axi_ar_fire, axi_r_fire,
    output logic [63:0] axi_araddr,
    output logic [7:0] axi_arlen,
    output logic axi_aw_fire, axi_w_fire, axi_b_fire,
    output logic [63:0] axi_awaddr,
    output logic [7:0] axi_awlen,
    output logic [31:0] axi_wstrb,
    output logic axi_wlast,
    output logic [1:0] axi_bresp,
    output logic memory_write_request_fire,
    output logic [63:0] memory_write_request_address,
    output logic [31:0] memory_write_request_bytes,
    output logic [7:0] memory_write_request_tag,
    output logic [2:0] selected_write_source,
    output logic attention_context_write_request_valid,
    output logic attention_context_write_request_ready,
    output logic attention_context_write_done,
    output logic attention_context_write_error,
    output logic matmul_panel_write_valid,
    output logic matmul_panel_write_ready,
    output logic matmul_panel_write_panel,
    output logic [1:0] matmul_panel_write_plane,
    output logic [15:0] matmul_panel_write_stripe,
    output logic [2:0] matmul_panel_write_start_bank,
    output logic [255:0] matmul_panel_write_data,
    output logic [31:0] matmul_panel_write_byte_enable,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        matmul_panel_write_port_mask,
    output logic [3:0] qkv_panel_prefetch_blocked,
    output logic [63:0] read_credit_stall_cycles,
    output logic [63:0] write_credit_stall_cycles,
    output logic [63:0] cache_lookup_requests,
    output logic [63:0] cache_dma_requests,
    output logic [63:0] cache_dma_bytes,
    output logic [63:0] cache_panel_writes,
    output logic [63:0] cache_compute_overlap_cycles,
    output logic [63:0] attention_operand_requests,
    output logic [63:0] attention_operand_responses,
    output logic [63:0] attention_pe_requests,
    output logic [63:0] attention_pe_accum_results,
    output logic [63:0] attention_output_tiles,
    output logic [9:0] debug_read_request_vector,
    output logic [9:0] debug_read_allowed_mask,
    output logic [9:0] debug_read_start_vector,
    output logic debug_memory_read_request_valid,
    output logic debug_memory_read_completion_done,
    output logic debug_memory_read_completion_error,
    output logic debug_memory_read_data_valid,
    output logic debug_memory_read_data_ready,
    output logic debug_memory_wide_read_data_valid,
    output logic debug_memory_wide_read_data_ready,
    output logic debug_axi_ar_valid,
    output logic debug_axi_ar_ready,
    output logic debug_axi_read_more_bursts,
    output logic debug_axi_read_credit_full,
    output logic debug_axi_r_valid,
    output logic debug_axi_r_ready,
    output logic debug_axi_r_last,
    output logic [3:0] debug_ddr_read_slots_used,
    output logic [3:0] debug_ddr_read_bursts_not_submitted,
    output logic debug_ddr_read_backend_can_accept,
    output logic [2:0] quant_active_client,
    output logic quant_idle,
    output logic quant_abort_ack,
    output logic quant_req_valid, quant_req_ready,
    output logic quant_rsp_valid, quant_rsp_ready,
    output logic image_check_done, image_check_pass,
    output logic [31:0] image_check_mismatch_count,
    input logic image_check_request,
    output logic ddr_initial_image_loaded,
    output logic [63:0] ddr_initial_image_bytes,
    output logic ddr_backend_idle,
    output logic ddr_protocol_error,
    output logic [3:0] ddr_read_id
);
    logic [3:0] awid, bid, arid, rid;
    logic [63:0] awaddr, araddr;
    logic [7:0] awlen, arlen;
    logic [2:0] awsize, arsize;
    logic [1:0] awburst, arburst, bresp, rresp;
    logic awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic arvalid, arready, rlast, rvalid, rready;
    logic [255:0] wdata, rdata;
    logic [31:0] wstrb;
    logic [4:0] unused_ddr_max_read_queued;
    logic [63:0] unused_expected_image_bytes;

    supra_top #(
        .ATTENTION_HEAD_COUNT(ATTENTION_HEAD_COUNT),
        .HIDDEN_FEATURES(HIDDEN_FEATURES),
        .FFN_FEATURES(FFN_FEATURES)
    ) dut (
        .core_clk(clk), .core_rst(rst), .launch_valid(run_valid),
        .launch_ready(run_ready), .launch_config_address(run_config_address),
        .launch_id(run_id), .completion_valid(completion_valid),
        .completion_ready(completion_ready), .completion_id(completion_id),
        .completion_error(completion_error),
        .completion_error_id(completion_error_id),
        .m_axi_awid(awid), .m_axi_awaddr(awaddr), .m_axi_awlen(awlen),
        .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready), .m_axi_bid(bid),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .m_axi_arid(arid), .m_axi_araddr(araddr), .m_axi_arlen(arlen),
        .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_rid(rid),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast),
        .m_axi_rvalid(rvalid), .m_axi_rready(rready)
    );

    generate
        if (USE_DRAMSIM3) begin : dramsim3_memory
            axi4_lpddr4_dramsim3_adapter #(
                .CORE_FREQUENCY_MHZ(DRAMSIM3_CORE_FREQUENCY_MHZ)
            ) ddr (
                .clk(clk), .rst(rst),
                .inject_read_response_error(inject_read_response_error),
                .inject_write_response_error(inject_write_response_error),
                .s_axi_awid(awid), .s_axi_awaddr(awaddr), .s_axi_awlen(awlen),
                .s_axi_awsize(awsize), .s_axi_awburst(awburst),
                .s_axi_awvalid(awvalid), .s_axi_awready(awready),
                .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wlast(wlast),
                .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bid(bid),
                .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
                .s_axi_arid(arid), .s_axi_araddr(araddr), .s_axi_arlen(arlen),
                .s_axi_arsize(arsize), .s_axi_arburst(arburst),
                .s_axi_arvalid(arvalid), .s_axi_arready(arready), .s_axi_rid(rid),
                .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rlast(rlast),
                .s_axi_rvalid(rvalid), .s_axi_rready(rready),
                .max_read_queued(unused_ddr_max_read_queued),
                .debug_read_slots_used(debug_ddr_read_slots_used),
                .debug_read_bursts_not_submitted(
                    debug_ddr_read_bursts_not_submitted),
                .debug_read_backend_can_accept(
                    debug_ddr_read_backend_can_accept),
                .initial_image_loaded(ddr_initial_image_loaded),
                .initial_image_bytes(ddr_initial_image_bytes),
                .backend_idle(ddr_backend_idle),
                .protocol_error(ddr_protocol_error)
            );
            assign image_check_done = 1'b0;
            assign image_check_pass = 1'b0;
            assign image_check_mismatch_count = 32'd0;
            assign unused_expected_image_bytes = 64'd0;
            wire unused_behavioral_controls = random_stall ^ performance_config ^
                image_check_request;
        end else begin : behavioral_memory
            axi4_ddr_model #(
                .DATA_WIDTH(256),
                .MEMORY_BYTES(DDR_MEMORY_BYTES), .BACKING_BYTES(DDR_BACKING_BYTES),
                .BASE_ADDRESS(DDR_BASE_ADDRESS)
            ) ddr (
                .clk(clk), .rst(rst), .random_stall(random_stall),
                .performance_config(performance_config), .inject_read_id_error(1'b0),
                .inject_read_last_error(1'b0),
                .inject_read_response_error(inject_read_response_error),
                .inject_write_id_error(1'b0),
                .inject_write_response_error(inject_write_response_error),
                .s_axi_awid(awid), .s_axi_awaddr(awaddr), .s_axi_awlen(awlen),
                .s_axi_awsize(awsize), .s_axi_awburst(awburst),
                .s_axi_awvalid(awvalid), .s_axi_awready(awready),
                .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wlast(wlast),
                .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bid(bid),
                .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
                .s_axi_arid(arid), .s_axi_araddr(araddr), .s_axi_arlen(arlen),
                .s_axi_arsize(arsize), .s_axi_arburst(arburst),
                .s_axi_arvalid(arvalid), .s_axi_arready(arready), .s_axi_rid(rid),
                .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rlast(rlast),
                .s_axi_rvalid(rvalid), .s_axi_rready(rready),
                .max_read_queued(unused_ddr_max_read_queued),
                .image_check_request(image_check_request),
                .initial_image_loaded(ddr_initial_image_loaded),
                .initial_image_bytes(ddr_initial_image_bytes),
                .image_check_done(image_check_done), .image_check_pass(image_check_pass),
                .image_check_mismatch_count(image_check_mismatch_count),
                .expected_image_bytes(unused_expected_image_bytes)
            );
            assign ddr_backend_idle = 1'b1;
            assign ddr_protocol_error = 1'b0;
            assign debug_ddr_read_slots_used = 4'd0;
            assign debug_ddr_read_bursts_not_submitted = 4'd0;
            assign debug_ddr_read_backend_can_accept = 1'b0;
        end
    endgenerate

    wire unused_run_scope = ^run_scope;
    assign ddr_read_id = rid;
    localparam logic [3:0] ATTENTION_COMMAND = 4'd2;
`define OBSERVE(name) assign name = dut.operators.name
    assign observed_command = dut.execution_layer_operator_index;
    assign post_completion_observe_valid =
        dut.operators.forward_postprocess.done_valid &&
        dut.operators.forward_postprocess.done_ready;
    assign post_completion_observe_block_complete =
        dut.execution_forward_postprocess_block_complete;
    assign observed_layer = dut.operator_command_layer;
    assign operator_command_observe_valid = dut.operator_command_valid;
    assign operator_command_observe_ready = dut.operator_command_ready;
    assign observed_command_cache_release = dut.operator_command_cache_release;
    assign observed_command_qkv_residual_spill =
        dut.operator_command_qkv_residual_spill;
    assign observed_command_rms_qkv_prefetch_q =
        dut.operator_command_rms_qkv_prefetch_q;
    assign observed_command_q_head_preprocess =
        dut.operator_command_q_head_preprocess;
    assign debug_qkv_residual_spill_active = dut.qkv_residual_spill_active;
    assign debug_row_token_batch_index = dut.execution.row_token_batch_index;
    assign debug_row_active_rows = dut.row_active_rows;
    assign debug_row_total_rows = dut.execution.row_total_rows;
    assign debug_row_last_round = dut.execution.row_last_round;
    assign debug_execution_state = dut.execution.state;
    assign debug_execution_hidden_start_valid = dut.execution_hidden_start_valid;
    assign debug_execution_hidden_start_ready = dut.execution_hidden_start_ready;
    assign debug_execution_hidden_start_write = dut.execution_hidden_start_write;
    assign debug_execution_hidden_start_residual =
        dut.execution_hidden_start_residual;
    assign debug_execution_hidden_start_address = dut.execution_hidden_start_address;
    assign debug_execution_hidden_start_bytes = dut.execution_hidden_start_bytes;
    assign debug_execution_config_loader_state =
        dut.execution.execution_config_loader.state;
    assign debug_token_metadata_state = dut.execution.v4_row_configuration.state;
    assign debug_token_metadata_done_valid = dut.execution.metadata_done_valid;
    assign debug_token_metadata_done_error = dut.execution.metadata_done_error;
    assign debug_token_metadata_done_error_id =
        dut.execution.metadata_done_error_id;
    assign debug_token_metadata_request_valid =
        dut.execution.dma_read_request_valid[1];
    assign debug_token_metadata_request_ready =
        dut.execution.dma_read_request_ready[1];
    assign debug_token_metadata_data_valid =
        dut.execution.dma_read_data_valid[1];
    assign debug_token_metadata_data_ready =
        dut.execution.dma_read_data_ready[1];
    assign debug_token_metadata_completion_done =
        dut.execution.dma_read_completion[1].done_pulse;
    assign debug_token_metadata_completion_error =
        dut.execution.dma_read_completion[1].error;
    assign debug_matmul_frontend_source_rsp_values =
        dut.operators.shared_matmul.frontend_source_rsp_values;
    assign accepted_read_bytes = dut.memory_useful_read_bytes;
    assign accepted_write_bytes = dut.memory_useful_write_bytes;
    assign accepted_read_bursts = dut.memory_read_burst_count;
    assign accepted_write_bursts = dut.memory_write_burst_count;
    assign debug_read_outstanding = dut.memory_read_outstanding;
    assign debug_write_outstanding = dut.memory_write_outstanding;
    assign debug_ddr_read_requested =
        (|dut.execution_dma_read_request_valid) ||
        (|dut.operator_dma_read_request_valid) || dut.memory_read_request_valid || arvalid;
    assign debug_ddr_write_requested =
        (|dut.operator_dma_write_request_valid) || dut.memory_write_request_valid ||
        dut.memory_write_data_valid || awvalid || wvalid;
    assign debug_ddr_paths_drained = ddr_backend_idle &&
        !dut.memory_read_busy && !dut.memory_write_busy && dut.dma_read_stream_idle &&
        dut.memory_read_outstanding == 0 && dut.memory_write_outstanding == 0 &&
        !arvalid && !rvalid && !awvalid && !wvalid && !bvalid;
    assign post_weight_bytes = dut.operators.post_weight_bytes;
    assign post_scale_bytes = dut.operators.post_scale_bytes;
    assign post_pe_accepted_count = dut.operators.post_pe_count;
    assign post_candidate_wait_cycles =
        dut.operators.post_candidate_wait_cycles;
    assign post_lm_panel_count =
        dut.operators.forward_postprocess.lm_head.saved_panel_count;
    assign post_candidate_panel_count =
        dut.operators.forward_postprocess.candidate.panel_count;
    assign post_writer_running = dut.operators.forward_postprocess.writer_running;
    assign post_lm_waiting_for_panel =
        dut.operators.forward_postprocess.lm_head.compute_finished &&
        !dut.operators.forward_postprocess.lm_head.load_finished;
    `OBSERVE(qkv_accepted_matmul_count); `OBSERVE(qkv_accepted_rope_count);
    `OBSERVE(qkv_accepted_head_read_count); `OBSERVE(qkv_accepted_quant_count);
    `OBSERVE(qkv_accepted_quantized_commit_count); `OBSERVE(qkv_completed_cache_write_count);
    `OBSERVE(qkv_outstanding_cache_writes); `OBSERVE(matmul_accepted_issue_count);
    `OBSERVE(matmul_accepted_result_count); `OBSERVE(matmul_base_request_count);
    `OBSERVE(matmul_enhancement_request_count); `OBSERVE(matmul_scale_request_count);
    `OBSERVE(matmul_base_request_bytes); `OBSERVE(matmul_enhancement_request_bytes);
    `OBSERVE(matmul_scale_request_bytes);
    assign debug_matmul_state = dut.operators.shared_matmul.state;
    assign debug_matmul_current_row = dut.operators.shared_matmul.current_row;
    assign debug_matmul_frontend_done =
        dut.operators.shared_matmul.frontend_done_pulse;
    assign debug_matmul_source_req_valid =
        dut.operators.shared_matmul.frontend_source_req_valid;
    assign debug_matmul_source_req_ready =
        dut.operators.shared_matmul.frontend_source_req_ready;
    assign debug_matmul_source_rsp_valid =
        dut.operators.shared_matmul.frontend_source_rsp_valid;
    assign debug_matmul_source_rsp_ready =
        dut.operators.shared_matmul.frontend_source_rsp_ready;
    assign debug_matmul_activation_writer_done =
        dut.operators.shared_matmul.activation_writer_done_pulse;
    assign debug_matmul_output_controller_done =
        dut.operators.shared_matmul.output_path_done_pulse;
    assign debug_matmul_panel_valid = dut.operators.shared_matmul.panel_valid;
    assign debug_matmul_panel_loading = dut.operators.shared_matmul.panel_loading;
    assign debug_matmul_panel_computing = dut.operators.shared_matmul.panel_computing;
    assign debug_matmul_next_load_stripe = dut.operators.shared_matmul.next_load_stripe;
    assign debug_matmul_current_stripe = dut.operators.shared_matmul.current_stripe;
    assign debug_matmul_current_panel_valid = dut.operators.shared_matmul.current_panel_valid;
    assign debug_matmul_current_panel = dut.operators.shared_matmul.current_panel;
    assign debug_matmul_panel_stripe = dut.operators.shared_matmul.loader.panel_stripe_id;
    assign debug_matmul_panel_operator = dut.operators.shared_matmul.loader.panel_operator_id;
    assign debug_matmul_acquire_valid = dut.operators.shared_matmul.acquire_valid;
    assign debug_matmul_acquire_ready = dut.operators.shared_matmul.acquire_ready;
    assign debug_matmul_acquire_panel = dut.operators.shared_matmul.acquire_panel;
    assign debug_matmul_acquire_stripe = dut.operators.shared_matmul.acquire_stripe;
    assign debug_matmul_release_valid = dut.operators.shared_matmul.release_valid;
    assign debug_matmul_release_ready = dut.operators.shared_matmul.release_ready;
    assign debug_matmul_pending_release = dut.operators.shared_matmul.pending_release;
    assign debug_matmul_schedule_state = dut.operators.shared_matmul.scheduler.state;
    assign debug_matmul_schedule_n_base = dut.operators.shared_matmul.scheduler.n_base;
    assign debug_matmul_schedule_k_base = dut.operators.shared_matmul.scheduler.k_base;
    assign debug_matmul_issue_valid = dut.operators.shared_matmul.issue_valid;
    assign debug_matmul_issue_ready = dut.operators.shared_matmul.issue_ready;
    assign debug_matmul_issue_stripe = dut.operators.shared_matmul.issue_stripe;
    assign debug_matmul_reader_scale_valid =
        dut.operators.shared_matmul.reader.scale_cache_valid;
    assign debug_matmul_reader_scale_match = dut.operators.shared_matmul.reader.scale_match;
    assign debug_matmul_reader_response_pending =
        dut.operators.shared_matmul.reader.response_pending;
    assign debug_matmul_reader_fifo_occupancy =
        dut.operators.shared_matmul.reader.response_fifo_occupancy;
    assign debug_attention_operand_fifo_occupancy =
        dut.operators.attention.matmul.fifo_occupancy;
    assign debug_attention_score_fifo_occupancy =
        dut.operators.attention.softmax_score_fifo_occupancy;
    assign debug_qkv_head_tile_fifo_occupancy =
        dut.operators.qkv_preparation.head_tile_fifo_occupancy;
    assign debug_local_attention_scratch_fifo_occupancy =
        dut.local_sram.attention_scratch_fifo_occupancy;
    assign debug_local_rms_tile_fifo_occupancy =
        dut.local_sram.rms_tile_fifo_occupancy;
    assign debug_local_rms_scratch_fifo_occupancy =
        dut.local_sram.rms_scratch_fifo_occupancy;
    assign debug_matmul_reader_address_valid = dut.operators.shared_matmul.reader.address_valid;
    assign debug_matmul_read_bundle_ready = dut.matmul_read_bundle_ready;
    assign debug_shared_pe_accum_result_occupancy =
        {1'b0, dut.operators.shared_compute.shared_pe.accum_result_fifo_valid};
    assign debug_matmul_descriptor_count = dut.operators.shared_matmul.scheduler.descriptor_count;
    assign debug_matmul_result_valid = dut.operators.shared_matmul.result_valid;
    assign debug_matmul_result_ready = dut.operators.shared_matmul.result_ready;
    assign debug_matmul_fragment_valid = dut.operators.shared_matmul.fragment_valid;
    assign debug_matmul_fragment_ready = dut.operators.shared_matmul.fragment_ready;
    assign debug_matmul_tile_req_valid = dut.operators.shared_matmul.tile_req_valid;
    assign debug_matmul_tile_req_ready = dut.operators.shared_matmul.tile_req_ready;
    assign debug_matmul_output_controller_state = dut.operators.shared_matmul.output_writer.state;
    assign debug_matmul_output_controller_stripe = dut.operators.shared_matmul.output_writer.stripe_index;
    assign debug_matmul_output_controller_input_row = dut.operators.shared_matmul.output_writer.input_row;
    assign debug_matmul_output_controller_row = dut.operators.shared_matmul.output_writer.output_row;
    assign debug_matmul_residual_read_active =
        dut.operators.shared_matmul.output_writer.residual_read_active;
    assign debug_matmul_residual_read_is_prefetch =
        dut.operators.shared_matmul.output_writer.residual_read_is_prefetch;
    assign debug_matmul_residual_prefetch_pending =
        dut.operators.shared_matmul.output_writer.residual_prefetch_pending;
    assign debug_matmul_residual_prefetched_valid =
        dut.operators.shared_matmul.output_writer.residual_prefetched_valid;
    assign debug_matmul_residual_request_valid =
        dut.operators.shared_matmul.residual_read_request_valid;
    assign debug_matmul_residual_request_ready =
        dut.operators.shared_matmul.residual_read_request_ready;
    assign debug_matmul_residual_data_valid =
        dut.operators.shared_matmul.residual_read_data_valid;
    assign debug_matmul_residual_data_ready =
        dut.operators.shared_matmul.residual_read_data_ready;
    assign debug_matmul_residual_request_done =
        dut.operators.shared_matmul.residual_read_request_done;
    assign debug_matmul_residual_local_read_issue =
        dut.operators.shared_matmul.residual_local_read_valid;
    assign debug_matmul_residual_local_read_pending =
        dut.operators.shared_matmul.residual_local_read_pending;
    assign debug_matmul_residual_local_response_pending =
        dut.operators.shared_matmul.residual_local_response_pending;
    assign debug_matmul_memory_stage = dut.matmul_memory_stage;
    assign debug_matmul_aux_start_valid = dut.operators.shared_matmul.aux_start_valid;
    assign debug_matmul_aux_start_ready = dut.operators.shared_matmul.aux_start_ready;
    assign debug_matmul_aux_residual_active = dut.operators.shared_matmul.aux_residual_active;
    assign debug_matmul_aux_request_valid =
        dut.operator_dma_read_request_valid[4];
    assign debug_matmul_aux_request_ready =
        dut.operator_dma_read_request_ready[4];
    assign debug_matmul_aux_request_bytes =
        dut.operator_dma_read_request[4].byte_count;
    assign debug_matmul_aux_read_valid = dut.operator_dma_read_data_valid[4];
    assign debug_matmul_aux_read_ready = dut.operator_dma_read_data_ready[4];
    assign debug_matmul_aux_data_valid = dut.operators.shared_matmul.aux_stream_valid;
    assign debug_matmul_aux_data_ready = dut.operators.shared_matmul.aux_stream_ready;
    assign debug_matmul_aux_done = dut.operators.shared_matmul.aux_stream_done;
    assign debug_matmul_aux_error =
        dut.operators.shared_matmul.aux_stream_error;
    assign debug_matmul_writer_state =
        dut.operators.shared_matmul.output_writer_dma.state;
    assign debug_matmul_writer_active = dut.operators.shared_matmul.writer_active;
    assign debug_matmul_writer_request_valid =
        dut.operator_dma_write_request_valid[3];
    assign debug_matmul_writer_request_ready =
        dut.operator_dma_write_request_ready[3];
    assign debug_matmul_writer_data_valid = dut.operators.shared_matmul.writer_data_valid;
    assign debug_matmul_writer_data_ready = dut.operators.shared_matmul.writer_data_ready;
    assign debug_matmul_writer_write_valid = dut.operator_dma_write_data_valid[3];
    assign debug_matmul_writer_write_ready = dut.operator_dma_write_data_ready[3];
    assign debug_matmul_writer_done = dut.operators.shared_matmul.writer_done;
    assign integration_head_count = 6'(ATTENTION_HEAD_COUNT);
    assign integration_ffn_features = 16'(FFN_FEATURES);
    assign debug_matmul_loader_plane = dut.operators.shared_matmul.loader.active_plane;
    assign debug_matmul_dma_request_outstanding =
        dut.operators.shared_matmul.loader.dma_request_outstanding;
    assign debug_matmul_dma_payload_complete =
        dut.operators.shared_matmul.loader.dma_payload_complete;
    assign debug_matmul_dma_response_remaining =
        dut.operators.shared_matmul.loader.response_remaining;
    assign debug_matmul_dma_req_valid = dut.operator_dma_read_request_valid[3];
    assign debug_matmul_dma_req_done =
        dut.operator_dma_read_completion[3].done_pulse;
    assign debug_matmul_dma_req_error = dut.operator_dma_read_completion[3].error;
    assign attention_completed_head_count_debug = dut.operators.attention_completed_head_count;
    assign attention_qk_count_debug = dut.operators.attention_qk_count;
    assign attention_pv_count_debug = dut.operators.attention_pv_count;
    assign current_local_phase_debug = dut.current_local_phase;
    assign requested_local_phase_debug = dut.requested_local_phase;
    assign debug_read_active_client = dut.dma_router.read_active_client;
    assign debug_selected_read_active_client = dut.dma_router.selected_read_active_client;
    assign debug_qkv_state = {3'd0, dut.operators.qkv_phase_active};
    assign debug_rms_state = {3'd0, dut.operators.rms_active};
    assign qkv_state = dut.operators.qkv_preparation.state;
    assign debug_qkv_current_head =
        dut.operators.qkv_preparation.head_index;
    assign debug_qkv_select = dut.operators.qkv_preparation.qkv_select;
    assign qkv_commit_state = dut.operators.qkv_preparation.commit_state;
    assign elementwise_prefetch_state =
        dut.operators.ffn_elementwise.prefetch_state;
    assign r4_controller_state =
        dut.operators.ffn_elementwise.r4_controller.state;
    assign r4_h1024_state =
        dut.operators.ffn_elementwise.r4_controller.h1024_pipeline.state;
    assign r4_h12_state =
        dut.operators.ffn_elementwise.r4_controller.h12_pipeline.state;
    assign r4_product_req_valid =
        dut.operators.ffn_elementwise.r4_product_req_valid;
    assign r4_product_req_ready =
        dut.operators.ffn_elementwise.r4_product_req_ready;
    assign r4_product_req_write =
        dut.operators.ffn_elementwise.r4_product_req.write;
    assign r4_arithmetic_req_valid =
        dut.operators.ffn_elementwise.r4_arithmetic_req_valid;
    assign r4_arithmetic_req_ready =
        dut.operators.ffn_elementwise.r4_arithmetic_req_ready;
    assign elementwise_read_request_valid =
        dut.operators.ffn_elementwise.read_request_valid;
    assign elementwise_read_request_ready =
        dut.operators.ffn_elementwise.read_request_ready;
    assign elementwise_read_data_valid =
        dut.operators.ffn_elementwise.read_data_valid;
    assign elementwise_read_data_ready =
        dut.operators.ffn_elementwise.read_data_ready;
    assign elementwise_prefetch_ready_count =
        dut.operators.ffn_elementwise.prefetch_ready_count;
    assign debug_read_address = dut.memory_read_request.byte_address;
    assign debug_read_bytes = dut.memory_read_request.byte_count;
    assign debug_read_tag = dut.memory_read_request.tag;
    assign debug_read_second_span_valid = dut.memory_read_second_span_valid;
    assign debug_read_second_span_address =
        dut.memory_read_second_span_address;
    assign debug_read_pair_stride = dut.memory_read_pair_stride;
    assign debug_read_pair_count = dut.memory_read_pair_count;
    assign debug_memory_read_request_ready = dut.memory_read_request_ready;
    assign debug_width_read_active = dut.memory.width_adapter.read_active;
    assign debug_width_read_done_pending =
        dut.memory.width_adapter.read_done_pending;
    assign debug_wide_memory_read_request_ready =
        dut.memory.wide_read_request_ready;
    assign debug_read_alignment_ready =
        dut.memory.axi_transport.read_alignment_ready;
    assign debug_read_alignment_state =
        dut.memory.axi_transport.read_alignment.state;
    assign debug_axi_read_active = dut.memory.axi_transport.axi.read_active;
    assign debug_axi_read_idle = dut.memory.axi_transport.master_read_idle;
    assign debug_layer_entry_valid = dut.layer_entry_valid;
    assign debug_layer_entry_index = dut.layer_entry_index;
    assign debug_layer_done_valid = dut.execution.layer_done_valid;
    assign debug_layer_done_error = dut.execution.layer_done_error;
    assign debug_layer_done_error_id = dut.execution.layer_done_error_id;
    assign bank_port_overflow = dut.local_sram.port_double_drive;
    assign bank_same_address_conflict =
        dut.local_sram.macro_same_address_conflict;
    assign bank_role_mismatch = '0;
    assign debug_compute_a_req_valid = dut.local_sram.logical_req_valid;
    assign debug_compute_a_write = dut.local_sram.port_write;
    assign debug_compute_b_req_valid = '0;
    assign debug_compute_b_write = '0;
    assign debug_local_dma_req_valid = '0;
    assign debug_local_dma_write = '0;
    `OBSERVE(rms_trace_sample_valid); `OBSERVE(rms_trace_sample_row); `OBSERVE(rms_trace_sample_element);
    `OBSERVE(rms_trace_sample_data); `OBSERVE(rms_trace_sample_byte_enable);
    assign matmul_int32_observe_valid = dut.operators.matmul_accum_result_valid;
    assign matmul_quantization_second_batch = dut.operators.shared_matmul.quantization_batch_index;
    assign matmul_int32_observe_accumulators = dut.operators.matmul_accum_result_accumulators;
    assign matmul_int32_observe_mask = dut.operators.matmul_accum_result_mask;
    assign matmul_int32_observe_mode = dut.operators.matmul_accum_result_mode;
    assign matmul_int32_observe_last_k_step = dut.operators.matmul_accum_result_last_k_step;
    assign matmul_int32_observe_tag = dut.operators.matmul_accum_result_tag;
    assign matmul_rescale_observe_valid =
        dut.operators.matmul_rescale_rsp_valid &&
        dut.operators.matmul_rescale_rsp_ready;
    assign matmul_rescale_observe_values =
        dut.operators.rescale_client_rsp[0].values;
    assign matmul_rescale_observe_mask =
        dut.operators.rescale_client_rsp[0].lane_mask;
    assign matmul_rescale_observe_tag =
        dut.operators.rescale_client_rsp[0].tag;
    assign matmul_residual_add_observe_valid =
        dut.operators.shared_matmul.output_writer.arithmetic_req_valid &&
        dut.operators.shared_matmul.output_writer.arithmetic_req_ready;
    assign matmul_residual_add_observe_values =
        dut.operators.shared_matmul.output_writer.arithmetic_req_values;
    assign matmul_residual_add_observe_paired_values =
        dut.operators.shared_matmul.output_writer.arithmetic_req_paired_values;
    assign matmul_residual_add_observe_lane_mask =
        dut.operators.shared_matmul.output_writer.arithmetic_req_lane_mask;
    assign matmul_residual_add_observe_tag =
        dut.operators.shared_matmul.output_writer.arithmetic_req_tag;
    assign matmul_bf16_observe_valid = dut.matmul_local_output_write_valid &&
        dut.matmul_local_output_write_ready;
    assign matmul_bf16_observe_address =
        dut.matmul_local_output_write.byte_address;
    assign matmul_bf16_observe_data = dut.matmul_local_output_write.data;
    assign matmul_bf16_observe_byte_enable =
        dut.matmul_local_output_write.byte_enable;
    assign matmul_ddr_observe_valid = dut.operators.shared_matmul.output_write_valid &&
        dut.operators.shared_matmul.output_write_ready;
    assign matmul_ddr_observe_address = dut.operators.shared_matmul.output_write_address;
    assign matmul_ddr_observe_data = dut.operators.shared_matmul.output_write_data;
    assign matmul_ddr_observe_byte_enable =
        dut.operators.shared_matmul.output_write_byte_enable;
    assign matmul_ddr_observe_first = dut.operators.shared_matmul.output_write_first;
    assign matmul_ddr_observe_last = dut.operators.shared_matmul.output_write_last;
    logic ffn_product_start_pending, ffn_product_start_expected;
    always_ff @(posedge clk) begin
        if (rst) begin
            ffn_product_start_pending <= 1'b0;
            ffn_product_start_expected <= 1'b0;
        end else begin
            if (ffn_product_start_pending)
                assert (dut.operators.ffn_elementwise.precomputed_product ==
                        ffn_product_start_expected)
                    else $error("FFN consumer did not latch the configured product source");
            ffn_product_start_pending <= dut.operators.elementwise_start_valid &&
                dut.operators.elementwise_start_ready;
            if (dut.operators.elementwise_start_valid &&
                dut.operators.elementwise_start_ready)
                ffn_product_start_expected <=
                    dut.operators.ffn_batch_pair_config.fused_product;
        end
    end

    assign ffn_fragment_valid = dut.operators.shared_matmul.fused_product &&
        dut.operators.shared_matmul.fragment_valid &&
        dut.operators.shared_matmul.fragment_ready;
    assign ffn_fragment_batch = dut.operators.shared_matmul.fragment_batch_index;
    assign ffn_fragment_row = dut.operators.shared_matmul.fragment_row;
    assign ffn_fragment_column = dut.operators.shared_matmul.fragment_channel;
    assign ffn_fragment_up = dut.operators.shared_matmul.output_writer.stripe_index[0];
    assign ffn_fragment_data = dut.operators.shared_matmul.fragment_data;
    assign ffn_silu_valid = dut.operators.shared_matmul.fused_product &&
        dut.operators.shared_matmul.output_writer.arithmetic_req_valid &&
        dut.operators.shared_matmul.output_writer.arithmetic_req_ready &&
        dut.operators.shared_matmul.output_writer.arithmetic_req_operation == 3'd1;
    assign ffn_silu_data = dut.operators.shared_matmul.output_writer.saved_silu;
    `OBSERVE(elementwise_trace_sample_valid); `OBSERVE(elementwise_trace_sample_pass);
    `OBSERVE(elementwise_trace_sample_stripe); `OBSERVE(elementwise_trace_sample_row_base);
    `OBSERVE(elementwise_trace_sample_lane_mask); `OBSERVE(elementwise_trace_sample_silu);
    `OBSERVE(elementwise_trace_sample_product);
    assign elementwise_debug_state = dut.operators.ffn_elementwise.state;
    assign elementwise_debug_gate_word =
        dut.operators.ffn_elementwise.tile_values[127:0];
    assign elementwise_debug_up_word =
        dut.operators.ffn_elementwise.up_values[127:0];
    assign elementwise_debug_read_word =
        dut.operators.dma_read_data[6].data;
    assign elementwise_debug_bf16_req_fire =
        dut.operators.elementwise_bf16_req_valid &&
        dut.operators.elementwise_bf16_req_ready;
    assign elementwise_debug_bf16_req_operation =
        dut.operators.bf16_client_req[4].operation;
    assign elementwise_debug_bf16_req_value0 =
        dut.operators.bf16_client_req[4].values[15:0];
    assign elementwise_debug_bf16_req_factor0 =
        dut.operators.bf16_client_req[4].factor0_values[15:0];
    assign elementwise_debug_bf16_req_factor1 =
        dut.operators.bf16_client_req[4].factor1_values[15:0];
    assign elementwise_debug_bf16_rsp_fire =
        dut.operators.elementwise_bf16_rsp_valid &&
        dut.operators.elementwise_bf16_rsp_ready;
    assign elementwise_debug_bf16_rsp_value0 =
        dut.operators.bf16_client_rsp[4].values[15:0];
    assign elementwise_debug_bf16_active_client =
        dut.operators.shared_compute.bf16_active_client;
    assign elementwise_debug_bf16_outstanding =
        dut.operators.shared_compute.bf16_outstanding;
    assign elementwise_quantized_observe_valid = dut.operators.elementwise_writer_quantized_valid &&
        dut.operators.elementwise_writer_quantized_ready;
    assign elementwise_quantized_observe_row =
        dut.operators.writer_client_values[1].physical_row_base;
    assign elementwise_quantized_observe_element =
        dut.operators.writer_client_values[1].element_base;
    assign elementwise_quantized_observe_values =
        dut.operators.writer_client_values[1].values;
    assign elementwise_quantized_observe_lane_mask =
        dut.operators.writer_client_values[1].lane_mask;
    assign elementwise_scale_observe_valid =
        dut.operators.elementwise_writer_scale_valid && dut.operators.elementwise_writer_scale_ready;
    assign elementwise_scale_observe_row =
        dut.operators.writer_client_scale[1].row_base;
    assign elementwise_scale_observe_row_mask =
        dut.operators.writer_client_scale[1].row_mask;
    assign elementwise_scale_observe_values =
        dut.operators.writer_client_scale[1].values_bf16;
    assign hidden_ddr_observe_valid = dut.operator_dma_write_data_valid[0] &&
        dut.operator_dma_write_data_ready[0];
    assign hidden_ddr_observe_qkv = dut.qkv_residual_spill_active ||
        dut.execution.state == 5'd17;
    assign hidden_ddr_observe_preserve = dut.execution.state == 5'd15;
    // Initial hidden preservation precedes the first operator command.
    assign hidden_ddr_observe_preserve_layer = dut.execution.v4_layer_index;
    assign hidden_ddr_observe_ffn = dut.execution_layer_operator_index == 4'd4;
    assign hidden_ddr_observe_data = dut.operator_dma_write_data[0].data;
    assign hidden_ddr_observe_byte_enable =
        dut.operator_dma_write_data[0].byte_enable;
    assign hidden_ddr_observe_last = dut.operator_dma_write_data[0].last;
    assign hidden_local_write_observe_valid = dut.hidden_local_write_valid &&
        dut.hidden_local_write_ready;
    assign hidden_local_write_observe_preserved_refill =
        dut.execution.v4_read_preserved_input &&
        !dut.execution_hidden_start_write;
    assign hidden_local_write_observe_physical_row =
        dut.hidden_local_write.physical_row;
    assign hidden_local_write_observe_channel_word =
        dut.hidden_local_write.channel_word;
    assign hidden_local_write_observe_data = dut.hidden_local_write.data;
    assign hidden_local_write_observe_byte_enable =
        dut.hidden_local_write.byte_enable;
    assign rope_bf16_observe_valid = dut.rope_destination_write_valid &&
        dut.rope_destination_write_ready;
    assign rope_bf16_observe_is_k = dut.operators.rope_destination_write_is_k;
    assign rope_bf16_observe_lane_mask = dut.operators.rope_destination_write_lane_mask;
    assign rope_bf16_observe_lane_address =
        dut.rope_destination_write.lane_address;
    assign rope_bf16_observe_lane_data = dut.rope_destination_write.lane_data;
    assign qkv_quantized_observe_valid = dut.operators.qkv_quantized_commit_valid &&
        dut.operators.qkv_quantized_commit_ready;
    assign qkv_quantized_observe_qkv_select = dut.operators.qkv_quantized_commit_qkv_select;
    assign qkv_quantized_observe_head = dut.operators.qkv_quantized_commit_head;
    assign qkv_quantized_observe_row = dut.operators.qkv_quantized_commit_physical_row;
    assign qkv_quantized_observe_token_position = dut.operators.qkv_quantized_commit_token_position;
    assign qkv_quantized_observe_half = dut.operators.qkv_quantized_commit_half;
    assign qkv_quantized_observe_values = dut.operators.qkv_quantized_commit_values;
    assign qkv_quantized_observe_lane_mask = dut.operators.qkv_quantized_commit_lane_mask;
    assign qkv_quantized_observe_scale = dut.operators.qkv_quantized_commit_scale;
    assign qkv_cache_map_write_accepted =
        dut.operators.cache_current_write_accepted;
    assign qkv_cache_map_write_head = dut.operators.qkv_current_write_head;
    assign qkv_cache_map_write_physical_row =
        dut.operators.qkv_current_write_row;
    assign qkv_cache_map_write_logical_slot =
        dut.operators.qkv_current_write_slot;
    assign qkv_cache_map_write_chunk = dut.operators.qkv_current_write_chunk;
    assign qkv_cache_map_write_k_address =
        dut.operators.cache_current_k_write_address;
    assign qkv_cache_map_write_v_address =
        dut.operators.cache_current_v_write_address;
    assign qkv_cache_map_write_k_scale_address =
        dut.operators.cache_current_k_scale_write_address;
    assign attention_int32_observe_valid = dut.operators.attention_pe_accum_result_valid &&
        dut.operators.attention_pe_accum_result_ready &&
        dut.execution_layer_operator_index == ATTENTION_COMMAND;
    assign attention_int32_observe_accumulators =
        dut.operators.shared_compute.shared_pe_accum_result.accumulators;
    assign attention_int32_observe_mask =
        dut.operators.shared_compute.shared_pe_accum_result.accumulator_mask;
    assign attention_int32_observe_last_k_step =
        dut.operators.shared_compute.shared_pe_accum_result.last_k_step;
    assign attention_int32_observe_tag =
        dut.operators.shared_compute.shared_pe_accum_result.tag;
    generate
        for (genvar attention_lane = 0; attention_lane < 64;
             attention_lane = attention_lane + 1) begin : expand_attention_accum_result_scales
            assign attention_int32_observe_activation_scales[
                attention_lane*16 +: 16] =
                dut.operators.shared_compute.shared_pe_accum_result.activation_scales[
                    (attention_lane/8)*16 +: 16];
            assign attention_int32_observe_weight_scales[
                attention_lane*16 +: 16] =
                dut.operators.shared_compute.shared_pe_accum_result.weight_scales[
                    (attention_lane%8)*16 +: 16];
        end
    endgenerate
    assign attention_score_observe_valid = dut.attention_score_write_valid &&
        dut.attention_score_write_ready;
    assign attention_score_observe_head = dut.operators.attention_score_write_head;
    assign attention_score_observe_query_base =
        dut.attention_score_write.row_base;
    assign attention_score_observe_key_base = dut.attention_score_write.key_base;
    assign attention_score_observe_data = dut.attention_score_write.values;
    for (genvar score_lane = 0; score_lane < 64; score_lane++) begin
        assign attention_score_observe_byte_enable[score_lane*2 +: 2] =
            {2{dut.attention_score_write.lane_mask[score_lane]}};
    end
    assign softmax_bf16_observe_valid = dut.attention_probability_write_valid &&
        dut.attention_probability_write_ready &&
        dut.operators.attention_probability_write_probability;
    assign softmax_bf16_observe_head = dut.operators.attention_probability_write_head;
    assign softmax_bf16_observe_row = dut.attention_probability_write.row_base;
    assign softmax_bf16_observe_key = dut.attention_probability_write.key;
    assign softmax_bf16_observe_data = dut.attention_probability_write.values;
    for (genvar observe_lane = 0; observe_lane < 64; observe_lane++) begin
        assign softmax_bf16_observe_byte_enable[observe_lane*2 +: 2] =
            {2{dut.attention_probability_write.lane_mask[observe_lane]}};
    end
    assign softmax_quantized_observe_valid =
        dut.attention_probability_quantized_write_valid &&
        dut.attention_probability_quantized_write_ready;
    assign softmax_quantized_observe_head = dut.operators.attention_probability_quantized_write_head;
    assign softmax_quantized_observe_row =
        dut.attention_probability_quantized_write.row_base;
    assign softmax_quantized_observe_key =
        dut.attention_probability_quantized_write.key;
    assign softmax_quantized_observe_data =
        dut.attention_probability_quantized_write.values;
    assign softmax_quantized_observe_byte_enable =
        dut.attention_probability_quantized_write.lane_mask;
    assign softmax_scale_observe_valid =
        dut.attention_probability_scale_write_valid &&
        dut.attention_probability_scale_write_ready;
    assign softmax_scale_observe_head =
        dut.attention_probability_scale_write.head;
    assign softmax_scale_observe_row =
        dut.attention_probability_scale_write.row_base;
    assign softmax_scale_observe_value =
        dut.attention_probability_scale_write.values;
    assign softmax_scale_observe_row_mask =
        dut.attention_probability_scale_write.row_mask;
    assign context_bf16_observe_valid = dut.attention_context_buffer_write_valid &&
        dut.attention_context_buffer_write_ready;
    assign context_bf16_observe_address =
        dut.attention_context_write.address;
    assign context_bf16_observe_data = dut.attention_context_write.data;
    assign context_bf16_observe_byte_enable =
        dut.attention_context_write.byte_enable;
    assign context_ddr_observe_valid = dut.operator_dma_write_data_valid[2] &&
        dut.operator_dma_write_data_ready[2];
    assign context_ddr_observe_data = dut.operator_dma_write_data[2].data;
    assign context_ddr_observe_byte_enable =
        dut.operator_dma_write_data[2].byte_enable;
    assign context_ddr_observe_last = dut.operator_dma_write_data[2].last;
`undef OBSERVE

    always_ff @(posedge clk) begin
        if (rst) begin
            accepted_command_count <= 64'd0;
            completed_command_count <= 64'd0;
            rms_gamma_dma_accepted_count <= 64'd0;
            rms_gamma_vector_request_count <= 64'd0;
            read_outstanding_high_water <= 5'd0;
            write_outstanding_high_water <= 5'd0;
            debug_compute_progress_count <= 64'd0;
        end else begin
            if (dut.operator_command_valid &&
                dut.operator_command_ready)
                accepted_command_count <= accepted_command_count + 64'd1;
            if (dut.operator_command_done_valid &&
                dut.operator_command_done_ready)
                completed_command_count <= completed_command_count + 64'd1;
            if (dut.operators.shared_rmsnorm.gamma_dma_request_valid &&
                dut.operators.shared_rmsnorm.gamma_dma_request_ready)
                rms_gamma_dma_accepted_count <=
                    rms_gamma_dma_accepted_count + 64'd1;
            if (dut.operators.shared_rmsnorm.vector_req_valid &&
                dut.operators.shared_rmsnorm.vector_req_ready &&
                dut.operators.shared_rmsnorm.vector_req_operation == 3'd5)
                rms_gamma_vector_request_count <=
                    rms_gamma_vector_request_count + 64'd1;
            if (dut.memory_read_outstanding > read_outstanding_high_water)
                read_outstanding_high_water <= dut.memory_read_outstanding;
            if (dut.memory_write_outstanding > write_outstanding_high_water)
                write_outstanding_high_water <= dut.memory_write_outstanding;
            if ((dut.matmul_source_req_valid && dut.matmul_local_source_req_ready) ||
                (dut.operators.shared_matmul.frontend_source_rsp_valid &&
                 dut.operators.shared_matmul.frontend_source_rsp_ready) ||
                (dut.matmul_activation_write_valid && dut.matmul_activation_write_ready) ||
                (dut.matmul_panel_write_valid && dut.matmul_panel_write_ready) ||
                (dut.matmul_read_bundle_valid && dut.matmul_read_bundle_ready) ||
                (dut.operators.matmul_tile_req_valid && dut.operators.matmul_tile_req_ready) ||
                (dut.operators.matmul_tile_accum_result_valid &&
                 dut.operators.matmul_tile_accum_result_ready) ||
                (dut.operators.matmul_rescale_req_valid && dut.operators.matmul_rescale_req_ready) ||
                (dut.operators.matmul_rescale_rsp_valid && dut.operators.matmul_rescale_rsp_ready) ||
                (dut.operators.shared_compute.shared_quant_req_valid &&
                 dut.operators.shared_compute.shared_quant_req_ready) ||
                (dut.operators.shared_compute.shared_quant_rsp_valid &&
                 dut.operators.shared_compute.shared_quant_rsp_ready) ||
                (dut.operators.rope_arithmetic_req_valid &&
                 dut.operators.rope_arithmetic_req_ready) ||
                (dut.operators.rope_arithmetic_rsp_valid &&
                 dut.operators.rope_arithmetic_rsp_ready) ||
                (dut.qkv_head_read_req_valid && dut.qkv_head_read_req_ready) ||
                (dut.qkv_head_read_rsp_valid && dut.qkv_head_read_rsp_ready))
                debug_compute_progress_count <=
                    debug_compute_progress_count + 64'd1;
        end
    end

    assign rms_write_valid = dut.rms_norm_write_valid;
    assign rms_write_ready = dut.rms_norm_write_ready;
    assign rms_write_row_base = dut.rms_norm_write.row_base;
    assign rms_write_element = dut.rms_norm_write.element;
    assign rms_write_data = dut.rms_norm_write.data;
    assign rms_write_lane_mask = dut.rms_norm_write.lane_mask;
    assign rms_source_valid = dut.rms_tile_read_rsp_valid;
    assign rms_source_ready = dut.rms_tile_read_rsp_ready;
    assign rms_source_row_base = dut.operators.shared_rmsnorm.current_row_base;
    assign rms_source_values = dut.rms_tile_read_rsp.values;
    assign rms_source_gamma = dut.rms_tile_read_rsp.gamma;
    assign rms_source_lane_mask = dut.rms_tile_read_rsp.lane_mask;
    assign rms_source_tag = dut.rms_tile_read_rsp.tag;
    assign rms_scratch_write_valid = dut.rms_scratch_write_valid;
    assign rms_scratch_write_ready = dut.rms_scratch_write_ready;
    assign rms_scratch_write_bank = dut.rms_scratch_write.bank;
    assign rms_scratch_write_address = dut.rms_scratch_write.address;
    assign rms_scratch_write_data = dut.rms_scratch_write.data;
    assign rms_scratch_write_byte_enable =
        dut.rms_scratch_write.byte_enable;
    assign rms_scratch_read_valid = dut.rms_scratch_read_valid;
    assign rms_scratch_read_ready = dut.rms_scratch_read_ready;
    assign rms_scratch_read_bank = dut.rms_scratch_read_req.bank;
    assign rms_scratch_read_left_address =
        dut.rms_scratch_read_req.left_address;
    assign rms_scratch_read_right_address =
        dut.rms_scratch_read_req.right_address;
    assign rms_scratch_read_tag = dut.rms_scratch_read_req.tag;
    assign rms_scratch_rsp_valid = dut.rms_scratch_read_rsp_valid;
    assign rms_scratch_rsp_ready = dut.rms_scratch_read_rsp_ready;
    assign rms_scratch_rsp_left_data = dut.rms_scratch_read_rsp.left_data;
    assign rms_scratch_rsp_right_data = dut.rms_scratch_read_rsp.right_data;
    assign rms_scratch_rsp_tag = dut.rms_scratch_read_rsp.tag;
    assign post_active = dut.operators.post_active;
    assign post_norm_group_base = dut.operators.forward_postprocess.sequencer.current_group_base;
    assign post_quant_observe_valid = dut.operators.forward_postprocess.writer_values_valid &&
        dut.operators.forward_postprocess.writer_values_ready;
    assign post_quant_observe_row = {dut.operators.forward_postprocess.quantizer.saved_group, 3'b000} +
        7'(dut.operators.forward_postprocess.writer_values.physical_row_base);
    assign post_quant_observe_element = dut.operators.forward_postprocess.writer_values.element_base;
    assign post_quant_observe_values = dut.operators.forward_postprocess.writer_values.values;
    assign post_quant_observe_mask = dut.operators.forward_postprocess.writer_values.lane_mask;
    assign post_scale_observe_valid = dut.operators.forward_postprocess.quant_writer_scale_valid &&
        dut.operators.forward_postprocess.quant_writer_scale_ready;
    assign post_scale_observe_row = {dut.operators.forward_postprocess.quant_start_group, 3'b000};
    assign post_scale_observe_values = dut.operators.forward_postprocess.writer_scale.values_bf16;
    assign post_scale_observe_mask = dut.operators.forward_postprocess.writer_scale.row_mask;
    assign post_int32_observe_valid = dut.operators.forward_postprocess.lm_head.rescale_req_valid &&
        dut.operators.forward_postprocess.lm_head.rescale_req_ready;
    assign post_int32_observe_row = {dut.operators.forward_postprocess.lm_head.rescale_req_tag[14:11], 3'b000};
    assign post_int32_observe_vocab = 17'(dut.operators.forward_postprocess.lm_head.compute.saved_panel_index) * 17'd8;
    assign post_int32_observe_values = dut.operators.forward_postprocess.lm_head.rescale_req_accumulators;
    assign post_int32_observe_mask = dut.operators.forward_postprocess.lm_head.rescale_req_lane_mask;
    assign post_logit_observe_valid = dut.operators.forward_postprocess.lm_logit_valid &&
        dut.operators.forward_postprocess.lm_logit_ready;
    assign post_logit_observe_row = dut.operators.forward_postprocess.lm_logit_row_base;
    assign post_logit_observe_vocab = dut.operators.forward_postprocess.lm_logit_vocab_base;
    assign post_logit_observe_values = dut.operators.forward_postprocess.lm_logit_values;
    assign post_logit_observe_mask = dut.operators.forward_postprocess.lm_logit_lane_mask;
    assign post_candidate_observe_valid = dut.operators.forward_postprocess.candidate_result_valid &&
        dut.operators.forward_postprocess.candidate_result_ready;
    assign post_candidate_observe_row = dut.operators.forward_postprocess.candidate_result_row;
    assign post_candidate_observe_top1 = dut.operators.forward_postprocess.candidate_result_top1;
    assign post_candidate_observe_logit = dut.operators.forward_postprocess.candidate.result_top_logit_bf16;
    assign post_candidate_observe_confidence = dut.operators.forward_postprocess.candidate.result_raw_confidence_bf16;
    assign post_candidate_observe_probability = dut.operators.forward_postprocess.candidate_result_selected_probability;
    assign post_candidate_observe_action = dut.operators.forward_postprocess.candidate_result_action_confidence;

    assign matmul_scale_valid = dut.operators.shared_activation_scale_valid;
    assign matmul_scale_ready = dut.operators.shared_activation_scale_ready;
    assign matmul_scale_row_base = dut.operators.shared_activation_scale_row_base;
    assign matmul_scale_row_mask = dut.operators.shared_activation_scale_row_mask;
    assign matmul_scale_values = dut.operators.shared_activation_scale_values;
    assign matmul_activation_observe_valid =
        dut.operators.matmul_activation_writer_quantized_valid &&
        dut.operators.matmul_activation_writer_quantized_ready;
    assign matmul_activation_observe_row =
        dut.operators.writer_client_values[0].physical_row_base;
    assign matmul_activation_observe_element =
        dut.operators.writer_client_values[0].element_base;
    assign matmul_activation_observe_values =
        dut.operators.writer_client_values[0].values;
    assign matmul_activation_observe_lane_mask =
        dut.operators.writer_client_values[0].lane_mask;
    assign qkv_quant_req_valid = dut.operators.qkv_quant_req_valid;
    assign qkv_quant_req_ready = dut.operators.qkv_quant_req_ready;
    assign qkv_quant_req_values =
        dut.operators.quant_values_client_req[0].values_bf16;
    assign attention_controller_state = dut.operators.attention.state;

    assign shared_pe_req_valid = dut.operators.shared_compute.shared_pe_req_valid;
    assign shared_pe_req_ready = dut.operators.shared_compute.shared_pe_req_ready;
    assign shared_pe_req_mode = dut.operators.shared_compute.shared_pe_req.mode;
    assign shared_pe_req_first =
        dut.operators.shared_compute.shared_pe_req.first_k_step;
    assign shared_pe_req_last =
        dut.operators.shared_compute.shared_pe_req.last_k_step;
    assign shared_pe_req_row_mask =
        dut.operators.shared_compute.shared_pe_req.row_mask;
    assign shared_pe_req_k_mask =
        dut.operators.shared_compute.shared_pe_req.k_mask;
    assign shared_pe_req_col_mask =
        dut.operators.shared_compute.shared_pe_req.column_mask;
    assign shared_pe_req_tag = dut.operators.shared_compute.shared_pe_req.tag;
    assign shared_pe_req_activation_payload =
        dut.operators.shared_compute.shared_pe_req.activation_payload;
    assign shared_pe_req_weight_payload =
        dut.operators.shared_compute.shared_pe_req.weight_payload;
    assign shared_pe_req_row0_activation =
        dut.operators.shared_compute.shared_pe_req.activation_payload[63:0];
    assign shared_pe_req_weight_payload_low =
        dut.operators.shared_compute.shared_pe_req.weight_payload[255:0];
    assign bf16_request_observe_accepted =
        dut.operators.shared_compute.bf16_request_fire;
    assign debug_bf16_request_clients =
        dut.operators.shared_compute.bf16_req_valid;
    assign bf16_request_observe_operation =
        dut.operators.shared_compute.shared_bf16_req.operation;
    always_ff @(posedge clk) begin
        if (dut.operators.shared_compute.shared_quantizer.quantized_req_valid &&
            dut.operators.shared_compute.shared_quantizer.quantized_req_ready) begin
            clip_observe_input <= dut.operators.shared_compute.shared_quantizer.quantized_req_values_bf16;
            clip_observe_element <= dut.operators.shared_compute.quant_active_client == 3 ?
                {2'd0, dut.operators.ffn_elementwise.product_stream_fifo_stripe, 3'd0} :
                (dut.operators.shared_compute.shared_quantizer.quantized_req_tag << 3);
        end
    end
    assign clip_observe_valid = dut.operators.shared_compute.shared_quantizer.config_clip_enable &&
        dut.operators.shared_compute.shared_quantizer.clip_input_valid &&
        dut.operators.shared_compute.shared_quantizer.pipeline_advance &&
        !dut.operators.shared_compute.shared_quantizer.abort_request;
    assign clip_scale_observe_valid = dut.operators.shared_compute.shared_quantizer.config_clip_enable &&
        dut.operators.shared_compute.shared_quantizer.scale_response_fire;
    assign clip_observe_row = dut.operators.shared_compute.quant_active_client == 3 ?
        dut.operators.ffn_elementwise.batch_row_base : dut.operators.shared_matmul.frontend.saved_row_base;
    assign clip_observe_batch_index = dut.operators.shared_compute.quant_active_client == 1 ?
        dut.operators.shared_matmul.quantization_batch_index : 3'd0;
    assign clip_observe_tag = dut.operators.shared_compute.shared_quantizer.clip_input_tag;
    assign clip_observe_output = dut.operators.shared_compute.shared_quantizer.clip_input_values;
    assign clip_observe_lane_mask = dut.operators.shared_compute.shared_quantizer.clip_input_lane_mask;
    assign clip_observe_maximum = dut.operators.shared_compute.shared_quantizer.config_row_max;
    assign clip_observe_limit = dut.operators.shared_compute.shared_quantizer.config_clip_limits;
    assign clip_observe_row_mask = dut.operators.shared_compute.shared_quantizer.config_row_mask;
    assign shared_pe_accum_result_valid =
        dut.operators.shared_compute.shared_pe_accum_result_valid;
    assign shared_pe_accum_result_ready =
        dut.operators.shared_compute.shared_pe_accum_result_ready;
    assign matmul_issue_mode = dut.operators.shared_matmul.issue_mode;
    assign matmul_issue_row_base = dut.operators.shared_matmul.issue_row_base;
    assign matmul_issue_k_base = dut.operators.shared_matmul.issue_k_base;
    assign matmul_issue_n_base = dut.operators.shared_matmul.issue_n_base;
    assign matmul_issue_first = dut.operators.shared_matmul.issue_first;
    assign matmul_issue_last = dut.operators.shared_matmul.issue_last;
    assign matmul_issue_tag = dut.operators.shared_matmul.issue_tag;
    assign matmul_issue_row_batch = dut.operators.shared_matmul.issue_row_batch;
    assign matmul_issue_activation_address = dut.operators.shared_matmul.issue_activation_addr;
    assign matmul_activation_word_index = dut.matmul_read_req.activation_word_index;
    assign matmul_activation_word_count = dut.matmul_read_req.activation_word_count;
    assign matmul_panel_read_select = dut.matmul_read_req.panel_select;
    assign matmul_panel_read_mask = dut.matmul_read_req.panel_port_mask;
    assign matmul_panel_read_address = dut.matmul_read_req.panel_address;
    assign matmul_read_response_valid = dut.matmul_read_response_valid;
    assign matmul_accum_result_last =
        dut.operators.shared_compute.shared_pe_accum_result.last_k_step;
    assign matmul_accum_result_mode =
        dut.operators.shared_compute.shared_pe_accum_result.mode;
    assign matmul_accum_result_tag =
        dut.operators.shared_compute.shared_pe_accum_result.tag;
    assign matmul_rescale_req_valid = dut.operators.matmul_rescale_req_valid;
    assign matmul_rescale_req_ready = dut.operators.matmul_rescale_req_ready;
    assign matmul_rescale_req_tag =
        dut.operators.rescale_client_req[0].tag;
    assign matmul_rescale_rsp_valid = dut.operators.matmul_rescale_rsp_valid;
    assign matmul_rescale_rsp_ready = dut.operators.matmul_rescale_rsp_ready;
    assign matmul_rescale_rsp_tag =
        dut.operators.rescale_client_rsp[0].tag;
    assign shared_rescale_req_valid =
        dut.operators.shared_compute.shared_rescale_req_valid;
    assign shared_rescale_req_ready =
        dut.operators.shared_compute.shared_rescale_req_ready;
    assign shared_rescale_req_tag =
        dut.operators.shared_compute.shared_rescale_req.tag;
    assign shared_rescale_rsp_valid =
        dut.operators.shared_compute.shared_rescale_rsp_valid;
    assign shared_rescale_rsp_ready =
        dut.operators.shared_compute.shared_rescale_rsp_ready;
    assign shared_rescale_rsp_tag =
        dut.operators.shared_compute.shared_rescale_rsp.tag;
    assign matmul_fragment_tag = dut.operators.shared_matmul.fragment_tag;
    assign matmul_fragment_row = dut.operators.shared_matmul.fragment_row;
    assign matmul_fragment_channel = dut.operators.shared_matmul.fragment_channel;
    assign matmul_combine_write_valid = dut.matmul_combine_write_valid;
    assign matmul_combine_write_ready = dut.matmul_combine_write_ready;
    assign matmul_combine_write_address = dut.matmul_combine_write.address;
    assign matmul_combine_read_valid = dut.matmul_combine_read_valid;
    assign matmul_combine_read_ready = dut.matmul_combine_read_ready;
    assign matmul_combine_read_address = dut.matmul_combine_read_req.address;
    assign matmul_ddr_write_valid = dut.operators.shared_matmul.output_write_valid;
    assign matmul_ddr_write_ready = dut.operators.shared_matmul.output_write_ready;
    assign matmul_ddr_write_address = dut.operators.shared_matmul.output_write_address;
    assign local_sram_port_accept = dut.local_sram.macro_req_valid &
        dut.local_sram.macro_req_ready;
    assign local_sram_client_accept = dut.local_sram.port_req_valid &
        dut.local_sram.client_req_ready;
    always_comb begin : local_sram_accepted_requester
        integer requester_id;
        local_sram_port_demand_count = '0;
        local_sram_port_requester_mask = '0;
        local_sram_requester_coverage = '0;
        requester_id = 0;
        for (integer physical_port = 0;
             physical_port < local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT;
             physical_port++) begin
            if (local_sram_client_accept[physical_port]) begin
                requester_id = 2;
                if (dut.local_sram.hidden_req_valid[physical_port])
                    requester_id = dut.local_sram.hidden_write[physical_port] ?
                        0 : 1;
                else if (dut.local_sram.context_req_valid[physical_port])
                    requester_id = dut.local_sram.context_write[physical_port] ?
                        40 : 41;
                else if (dut.local_sram.attention_scratch_req_valid[physical_port])
                    requester_id = dut.local_sram.attention_scratch_write[
                        physical_port] ? 39 : (physical_port[0] ? 38 : 37);
                else if (dut.local_sram.qkv_head_read_req[physical_port])
                    requester_id = 21;
                else if (dut.local_sram.rope_req_valid[physical_port]) begin
                    if (dut.local_sram.rope_write[physical_port])
                        requester_id = physical_port ==
                            dut.local_sram.ROPE_DESTINATION_PRIMARY_PORT ?
                            19 : 20;
                    else if (dut.local_sram.rope_cos_read_req_valid &&
                             !dut.local_sram.rope_cos_read_pending &&
                             physical_port ==
                                 dut.local_sram.ROPE_CONSTANT_PRIMARY_PORT)
                        requester_id = 17;
                    else if (dut.local_sram.rope_sin_read_req_valid &&
                             !dut.local_sram.rope_sin_read_pending &&
                             physical_port ==
                                 dut.local_sram.ROPE_CONSTANT_SECONDARY_PORT)
                        requester_id = 18;
                    else
                        requester_id = physical_port ==
                            dut.local_sram.ROPE_SOURCE_PRIMARY_PORT ? 15 : 16;
                end
                else if (dut.local_sram.qkv_stage_write_req_valid[physical_port])
                    requester_id = 22;
                else if (dut.local_sram.qkv_tile_read_req_valid[physical_port])
                    requester_id = 23;
                else if (dut.local_sram.attention_operand_req[physical_port]) begin
                    if (physical_port == dut.local_sram.ATTENTION_METADATA_PORT)
                        requester_id = 30;
                    else if (physical_port ==
                                 dut.local_sram.ATTENTION_PANEL_METADATA0_PORT ||
                             physical_port ==
                                 dut.local_sram.ATTENTION_PANEL_METADATA1_PORT)
                        requester_id = 31;
                    else if (physical_port ==
                                 dut.local_sram.ATTENTION_PANEL0_PRIMARY_PORT ||
                             physical_port ==
                                 dut.local_sram.ATTENTION_PANEL1_PRIMARY_PORT ||
                             physical_port ==
                                 dut.local_sram.ATTENTION_PANEL2_PRIMARY_PORT ||
                             physical_port ==
                                 dut.local_sram.ATTENTION_PANEL3_PRIMARY_PORT)
                        requester_id = 29;
                    else
                        requester_id = 28;
                end
                else if (dut.local_sram.attention_score_req_valid[physical_port])
                    requester_id = dut.local_sram.attention_score_port_write[
                        physical_port] ? 32 : 33;
                else if (dut.local_sram.attention_probability_req_valid[
                             physical_port]) begin
                    if (dut.attention_probability_scale_write_valid)
                        requester_id = 36;
                    else if (dut.attention_probability_quantized_write_valid)
                        requester_id = 35;
                    else
                        requester_id = 34;
                end else if (dut.local_sram.cache_write_req_valid[physical_port]) begin
                    if (dut.attention_scale_write_valid &&
                        dut.attention_scale_write_ready &&
                        ((!dut.attention_panel_scale_write.target_panel &&
                          physical_port == dut.local_sram.
                              ATTENTION_PANEL_METADATA0_WRITE_PORT) ||
                         (dut.attention_panel_scale_write.target_panel &&
                          physical_port == dut.local_sram.
                              ATTENTION_PANEL_METADATA1_WRITE_PORT)))
                        requester_id = 27;
                    else if (dut.attention_panel_write_valid &&
                             dut.attention_panel_write_ready) begin
                        requester_id = 2;
                        case ({dut.attention_panel_write.address[0],
                               dut.attention_panel_write.bank})
                            2'd0: requester_id = physical_port ==
                                dut.local_sram.ATTENTION_PANEL0_SECONDARY_PORT ?
                                25 : requester_id;
                            2'd1: requester_id = physical_port ==
                                dut.local_sram.ATTENTION_PANEL1_SECONDARY_PORT ?
                                25 : requester_id;
                            2'd2: requester_id = physical_port ==
                                dut.local_sram.ATTENTION_PANEL2_SECONDARY_PORT ?
                                25 : requester_id;
                            default: requester_id = physical_port ==
                                dut.local_sram.ATTENTION_PANEL3_SECONDARY_PORT ?
                                25 : requester_id;
                        endcase
                        if (|dut.attention_panel_write.byte_enable[31:16]) begin
                            case ({dut.attention_panel_write.address[0],
                                   !dut.attention_panel_write.bank})
                                2'd0: requester_id = physical_port ==
                                    dut.local_sram.
                                        ATTENTION_PANEL0_SECONDARY_PORT ?
                                    26 : requester_id;
                                2'd1: requester_id = physical_port ==
                                    dut.local_sram.
                                        ATTENTION_PANEL1_SECONDARY_PORT ?
                                    26 : requester_id;
                                2'd2: requester_id = physical_port ==
                                    dut.local_sram.
                                        ATTENTION_PANEL2_SECONDARY_PORT ?
                                    26 : requester_id;
                                default: requester_id = physical_port ==
                                    dut.local_sram.
                                        ATTENTION_PANEL3_SECONDARY_PORT ?
                                    26 : requester_id;
                            endcase
                        end
                        if (requester_id == 2 &&
                            dut.qkv_q_local_write_valid &&
                            dut.qkv_q_local_write_ready)
                            requester_id = 24;
                    end
                    else if (dut.qkv_q_local_write_valid &&
                             dut.qkv_q_local_write_ready)
                        requester_id = 24;
                    else
                        requester_id = 2;
                end else if (dut.local_sram.matmul_req_valid[physical_port]) begin
                    requester_id = 3;
                    if (dut.matmul_activation_write_valid &&
                        dut.matmul_activation_write_ready) begin
                        for (integer lane = 0; lane < 8; lane++)
                            if (dut.matmul_activation_write.slot_valid[lane] &&
                                dut.local_sram.matmul_activation_write_port[lane] ==
                                    physical_port)
                                requester_id = 4;
                    end
                    if (dut.matmul_read_bundle_valid &&
                        dut.matmul_read_bundle_ready) begin
                        for (integer lane = 0; lane < 8; lane++)
                            if (lane < dut.matmul_read_req.activation_word_count &&
                                dut.local_sram.matmul_activation_read_port[lane] ==
                                    physical_port)
                                requester_id = 5;
                        for (integer panel_lane = 0; panel_lane < 12;
                             panel_lane++)
                            if (dut.matmul_read_req.panel_port_mask[panel_lane] &&
                                dut.local_sram.matmul_panel_read_port[panel_lane] ==
                                    physical_port)
                                requester_id = panel_lane < 6 ? 6 : 7;
                    end
                    if (dut.matmul_panel_write_valid &&
                        dut.matmul_panel_write_ready) begin
                        if (dut.local_sram.matmul_panel_write_port[0] == physical_port)
                            requester_id = 8;
                        if (dut.local_sram.matmul_panel_write_port[1] == physical_port)
                            requester_id = 9;
                    end
                    if (dut.matmul_combine_write_valid &&
                        dut.matmul_combine_write_ready &&
                        dut.local_sram.matmul_combine_write_port == physical_port)
                        requester_id = 11;
                    if (dut.matmul_combine_read_valid &&
                        dut.matmul_combine_read_ready &&
                        dut.local_sram.matmul_combine_read_port == physical_port)
                        requester_id = 12;
                    if (dut.matmul_local_output_write_valid &&
                        dut.matmul_local_output_write_ready &&
                        dut.local_sram.matmul_local_output_port == physical_port)
                        requester_id = 13;
                    if (dut.matmul_residual_local_read_issue &&
                        dut.matmul_residual_memory_read_ready &&
                        dut.local_sram.matmul_residual_port == physical_port)
                        requester_id = 14;
                end
                local_sram_port_demand_count[physical_port*6 +: 6] = 6'd1;
                local_sram_port_requester_mask[
                    physical_port*42 + requester_id] = 1'b1;
                local_sram_requester_coverage[requester_id] = 1'b1;
            end
        end
    end
    assign local_sram_port_write = dut.local_sram.macro_write;
    assign local_sram_port_address = dut.local_sram.macro_address;
    assign matmul_source_req_valid = dut.matmul_source_req_valid;
    assign matmul_source_req_ready = dut.matmul_local_source_req_ready;
    assign matmul_source_port_mask = dut.local_sram.matmul_source_port_mask;
    assign matmul_source_port_address = dut.local_sram.matmul_source_address;
    assign matmul_operand_port_mask = dut.local_sram.matmul_operand_port_mask;
    assign matmul_compute_port_address = dut.local_sram.matmul_operand_address;
    assign matmul_aux_port_mask = dut.local_sram.matmul_other_port_mask;
    assign matmul_aux_port_address = dut.local_sram.matmul_other_address;
    assign matmul_combine_write_port =
        dut.operators.matmul_local_layout ? 6'h3f :
            local_memory_layout_pkg::local_region_port_index(
                local_memory_layout_pkg::qkv_constant_region(), 1'b0);
    assign matmul_combine_read_port =
        dut.operators.matmul_local_layout ? 6'h3f :
            local_memory_layout_pkg::local_region_port_index(
                local_memory_layout_pkg::qkv_constant_region(), 1'b1);
    assign matmul_output_controller_cfg_valid = dut.operators.shared_matmul.output_cfg_valid;
    assign matmul_output_controller_cfg_ready = dut.operators.shared_matmul.output_cfg_ready;
    assign matmul_local_output_write_valid =
        dut.matmul_local_output_write_valid;
    assign matmul_local_output_write_ready =
        dut.matmul_local_output_write_ready;
    assign matmul_local_output_write_address =
        dut.matmul_local_output_write.byte_address;
    assign matmul_fragment_active = dut.operators.shared_matmul.fragmenter.active;
    assign shared_rescale_state = dut.operators.shared_compute.shared_rescale.state;
    assign matmul_panel_memory_write_valid =
        dut.operators.shared_matmul.panel_memory_write_valid;
    assign matmul_panel_memory_write_ready =
        dut.operators.shared_matmul.panel_memory_write_ready;
    assign matmul_dma_rsp_valid = dut.operators.shared_matmul.dma_rsp_valid;
    assign matmul_dma_rsp_ready = dut.operators.shared_matmul.dma_rsp_ready;
    assign matmul_read_bundle_valid = dut.matmul_read_bundle_valid;
    assign matmul_read_response_pending =
        dut.local_sram.matmul_operand_read_pending;
    assign attention_operand_req_valid = dut.attention_operand_req_valid;
    assign attention_operand_req_ready = dut.attention_operand_req_ready;
    assign attention_operand_rsp_valid = dut.attention_operand_rsp_valid;
    assign attention_operand_rsp_ready =
        dut.operators.attention.matmul.operand_rsp_ready;
    assign attention_panel_write_valid = dut.attention_panel_write_valid;
    assign attention_panel_write_ready = dut.attention_panel_write_ready;
    assign attention_panel_write_target =
        dut.attention_panel_write.target_panel;
    assign attention_panel_write_quarter =
        dut.attention_panel_write.target_quarter;
    assign attention_panel_write_bank = dut.attention_panel_write.bank;
    assign attention_panel_write_address = dut.attention_panel_write.address;
    assign attention_panel_write_byte_enable =
        dut.attention_panel_write.byte_enable;
    assign attention_panel_write_tag = dut.operators.attention_panel_write_tag;
    assign attention_scale_write_valid = dut.attention_scale_write_valid;
    assign attention_scale_write_ready = dut.attention_scale_write_ready;
    assign attention_scale_write_token_base =
        dut.attention_panel_scale_write.token_base;
    assign attention_scale_write_byte_enable =
        dut.attention_panel_scale_write.byte_enable;
    assign attention_scale_write_tag = dut.operators.attention_scale_write_tag;
    assign attention_operand_req_is_pv = dut.attention_operand_req.is_pv;
    assign attention_operand_req_query_base =
        dut.attention_operand_req.query_base;
    assign attention_operand_req_output_base =
        dut.attention_operand_req.output_base;
    assign attention_operand_req_k_base = dut.attention_operand_req.k_base;
    assign attention_operand_req_row_mask =
        dut.operators.attention.matmul.operand_req_row_mask;
    assign attention_operand_req_tag = dut.operators.attention.matmul.operand_req_tag;
    assign attention_active_head = dut.attention_operand_req.active_head;
    assign attention_active_panel = dut.attention_operand_req.active_panel;
    assign attention_active_quarter =
        dut.attention_operand_req.active_quarter;
    assign attention_loader_start_pending =
        dut.operators.attention.loader_start_pending;
    assign attention_loaded_valid = dut.operators.attention.loaded_valid;
    assign attention_loader_select = dut.operators.attention.loader_select;
    assign attention_loaded_select = dut.operators.attention.loaded_select;
    assign attention_cache_dma_owned = dut.operators.attention.cache_loader.dma_owned;
    assign attention_cache_dma_done_seen =
        dut.operators.attention.cache_loader.dma_done_seen;
    assign attention_cache_dma_expected_bytes =
        dut.operators.attention.cache_loader.dma_expected_bytes;
    assign attention_cache_dma_received_bytes =
        dut.operators.attention.cache_loader.dma_received_bytes;
    assign attention_cache_buffer_valid[0] =
        dut.operators.attention.cache_loader.buffer_valid[0];
    assign attention_cache_buffer_valid[1] =
        dut.operators.attention.cache_loader.buffer_valid[1];
    assign attention_cache_drain_active =
        dut.operators.attention.cache_loader.drain_active;
    assign attention_cache_drain_word_pending =
        dut.operators.attention.cache_loader.drain_word_pending;
    assign attention_cache_drain_scale_pending =
        dut.operators.attention.cache_loader.drain_scale_pending;
    assign attention_cache_lookup_outstanding =
        dut.operators.attention.cache_loader.cache_lookup_outstanding;
    assign attention_cache_terminal_error =
        dut.operators.attention.cache_loader.terminal_error;
    assign attention_cache_terminal_error_id =
        dut.operators.attention.cache_loader.terminal_error_id;
    assign attention_dma_start_valid =
        dut.operators.attention.cache_dma_start_valid;
    assign attention_dma_start_ready =
        dut.operators.attention.cache_dma_start_ready;
    assign attention_dma_request_valid =
        dut.operator_dma_read_request_valid[5];
    assign attention_dma_request_ready =
        dut.operator_dma_read_request_ready[5];
    assign attention_dma_data_valid =
        dut.operators.attention.cache_dma_data_valid;
    assign attention_dma_data_ready =
        dut.operators.attention.cache_dma_data_ready;
    assign attention_dma_done = dut.operators.attention.cache_dma_done;
    assign attention_dma_error = dut.operators.attention.cache_dma_error;
    assign attention_cache_state = dut.operators.attention.cache_loader.fill_state;
    assign attention_matmul_state = dut.operators.attention.matmul.state;
    assign softmax_quant_state = dut.operators.attention.softmax.state;
    assign softmax_row_state = {1'b0, dut.operators.attention.softmax.row_pipeline.state};
    assign softmax_source_req_valid = dut.attention_softmax_score_req_valid;
    assign softmax_source_req_ready = dut.attention_softmax_score_req_ready;
    assign softmax_source_req_pass = dut.operators.attention_softmax_score_pass;
    assign softmax_source_req_key = dut.attention_score_read_req.key;
    assign softmax_source_req_head = dut.operators.attention_softmax_score_head;
    assign softmax_source_req_row_base =
        dut.attention_score_read_req.row_base;
    assign softmax_source_req_tag = dut.operators.attention_softmax_score_tag;
    assign softmax_source_rsp_valid =
        dut.operators.attention.softmax_score_read_rsp_valid;
    assign softmax_source_rsp_ready =
        dut.operators.attention.softmax_score_read_rsp_ready;
    assign softmax_source_rsp_tag =
        dut.operators.attention.softmax_score_read_rsp_tag;
    assign softmax_vector_req_valid = dut.operators.attention_softmax_bf16_req_valid;
    assign softmax_vector_req_ready = dut.operators.attention_softmax_bf16_req_ready;
    assign softmax_vector_req_tag =
        dut.operators.bf16_client_req[3].tag;
    assign softmax_vector_rsp_valid = dut.operators.attention_softmax_bf16_rsp_valid;
    assign softmax_vector_rsp_ready = dut.operators.attention_softmax_bf16_rsp_ready;
    assign softmax_vector_rsp_tag =
        dut.operators.shared_compute.shared_bf16_rsp.tag;
    assign softmax_max_req_valid = dut.operators.attention_softmax_max_req_valid;
    assign softmax_max_req_ready = dut.operators.attention_softmax_max_req_ready;
    assign softmax_max_req_tag =
        dut.operators.maximum_client_req[2].tag;
    assign softmax_max_rsp_valid = dut.operators.attention_softmax_max_rsp_valid;
    assign softmax_max_rsp_ready = dut.operators.attention_softmax_max_rsp_ready;
    assign softmax_max_rsp_tag =
        dut.operators.maximum_client_rsp[2].tag;
    assign softmax_bf16_write_valid = dut.attention_probability_write_valid;
    assign softmax_bf16_write_ready = dut.attention_probability_write_ready;
    assign softmax_bf16_write_probability =
        dut.operators.attention_probability_write_probability;
    assign softmax_bf16_write_tag = dut.operators.attention_probability_write_tag;
    assign softmax_quantized_write_tag = dut.operators.attention_probability_quantized_write_tag;
    assign softmax_scratch_read_valid =
        dut.attention_softmax_scratch_read_valid;
    assign softmax_scratch_read_ready =
        dut.attention_softmax_scratch_read_ready;
    assign softmax_scratch_read_bank =
        dut.attention_softmax_scratch_read_req.bank;
    assign softmax_scratch_read_left_address =
        dut.attention_softmax_scratch_read_req.left_address;
    assign softmax_scratch_read_right_address =
        dut.attention_softmax_scratch_read_req.right_address;
    assign softmax_scratch_read_tag =
        dut.attention_softmax_scratch_read_req.tag;
    assign softmax_scratch_write_valid =
        dut.attention_softmax_scratch_write_valid;
    assign softmax_scratch_write_ready =
        dut.attention_softmax_scratch_write_ready;
    assign softmax_scratch_write_bank =
        dut.attention_softmax_scratch_write.bank;
    assign softmax_scratch_write_address =
        dut.attention_softmax_scratch_write.address;
    assign softmax_scratch_write_byte_enable =
        dut.attention_softmax_scratch_write.byte_enable;
    assign attention_context_read_valid =
        dut.attention_context_read_req_valid;
    assign attention_context_read_ready =
        dut.attention_context_read_req_ready;
    assign attention_context_read_address =
        dut.attention_context_read_req.address;
    assign attention_context_read_tag = dut.attention_context_read_req.tag;
    assign attention_loader_inflight = dut.operators.attention.loader_inflight;
    assign attention_matmul_busy = dut.operators.attention.matmul_busy;
    assign attention_softmax_busy = dut.operators.attention.softmax_busy;
    assign attention_writer_busy = dut.operators.attention.writer_busy;
    assign attention_q_head_active = dut.operators.attention.q_head_inflight;
    assign attention_q_head_matmul_done =
        dut.operators.attention.q_head_matmul_done;
    assign attention_rescale_outstanding = dut.operators.attention.rescale_outstanding;
    assign attention_result_pending = dut.operators.attention.result_pending;
    assign axi_ar_fire = arvalid && arready;
    assign axi_r_fire = rvalid && rready;
    assign axi_araddr = araddr;
    assign axi_arlen = arlen;
    assign axi_aw_fire = awvalid && awready;
    assign axi_w_fire = wvalid && wready;
    assign axi_b_fire = bvalid && bready;
    assign axi_awaddr = awaddr;
    assign axi_awlen = awlen;
    assign axi_wstrb = wstrb;
    assign axi_wlast = wlast;
    assign axi_bresp = bresp;
    assign memory_write_request_fire = dut.memory_write_request_valid &&
        dut.memory_write_request_ready;
    assign memory_write_request_address =
        dut.memory_write_request.byte_address;
    assign memory_write_request_bytes = dut.memory_write_request.byte_count;
    assign memory_write_request_tag = dut.memory_write_request.tag;
    assign selected_write_source = dut.dma_router.selected_write_source;
    assign attention_context_write_request_valid =
        dut.operator_dma_write_request_valid[2];
    assign attention_context_write_request_ready =
        dut.operator_dma_write_request_ready[2];
    assign attention_context_write_done =
        dut.operator_dma_write_completion[2].done_pulse;
    assign attention_context_write_error =
        dut.operator_dma_write_completion[2].error;
    assign matmul_panel_write_valid = dut.matmul_panel_write_valid;
    assign matmul_panel_write_ready = dut.matmul_panel_write_ready;
    assign matmul_panel_write_panel = dut.matmul_panel_write.panel;
    assign matmul_panel_write_plane = dut.matmul_panel_write.plane;
    assign matmul_panel_write_stripe =
        dut.operators.shared_matmul.loader.active_stripe_id;
    assign matmul_panel_write_start_bank = dut.matmul_panel_write.start_bank;
    assign matmul_panel_write_data = dut.matmul_panel_write.data;
    assign matmul_panel_write_byte_enable = dut.matmul_panel_write.byte_enable;
    always_comb begin : observe_matmul_panel_write_ports
        matmul_panel_write_port_mask = '0;
        if (dut.matmul_panel_write_valid && dut.matmul_panel_write_ready) begin
            matmul_panel_write_port_mask[
                dut.local_sram.matmul_panel_write_port[0]] = 1'b1;
            if (|dut.matmul_panel_write.byte_enable[31:16])
                matmul_panel_write_port_mask[
                    dut.local_sram.matmul_panel_write_port[1]] = 1'b1;
        end
    end
    always_comb begin : observe_qkv_panel_prefetch_blocked
        local_memory_layout_pkg::local_sram_region_t panel_region;
        logic [5:0] physical_port;

        qkv_panel_prefetch_blocked = '0;
        for (integer panel_slot = 0; panel_slot < 4; panel_slot++) begin
            panel_region = dut.matmul_panel_write.panel ?
                local_memory_layout_pkg::q_work_panel_region(2'(panel_slot)) :
                local_memory_layout_pkg::qkv_panel_region(2'(panel_slot));
            physical_port = local_memory_layout_pkg::local_region_port_index(
                panel_region, 1'b1);
            qkv_panel_prefetch_blocked[panel_slot] =
                !dut.local_sram.client_req_ready[physical_port];
        end
    end
    assign read_credit_stall_cycles = dut.memory_read_credit_stall_cycles;
    assign write_credit_stall_cycles = dut.memory_write_credit_stall_cycles;
    assign cache_lookup_requests =
        dut.operators.attention.cache_loader.accepted_cache_lookup_request_count;
    assign cache_dma_requests =
        dut.operators.attention.cache_loader.accepted_dma_request_count;
    assign cache_dma_bytes =
        dut.operators.attention.cache_loader.accepted_dma_byte_count;
    assign cache_panel_writes =
        dut.operators.attention.cache_loader.accepted_panel_write_count;
    assign cache_compute_overlap_cycles =
        dut.operators.attention.cache_compute_overlap_cycle_count;
    assign attention_operand_requests =
        dut.operators.attention.matmul.accepted_operand_request_count;
    assign attention_operand_responses =
        dut.operators.attention.matmul.completed_operand_response_count;
    assign attention_pe_requests =
        dut.operators.attention.matmul.accepted_pe_request_count;
    assign attention_pe_accum_results =
        dut.operators.attention.matmul.completed_pe_accum_result_count;
    assign attention_output_tiles =
        dut.operators.attention.matmul.completed_output_tile_count;
    assign debug_read_request_vector = dut.dma_router.read_request_vector;
    assign debug_read_allowed_mask = dut.dma_router.read_allowed_mask;
    assign debug_read_start_vector = dut.dma_router.read_start_vector;
    assign debug_memory_read_request_valid = dut.memory_read_request_valid;
    assign debug_memory_read_completion_done =
        dut.memory_read_completion.done_pulse;
    assign debug_memory_read_completion_error =
        dut.memory_read_completion.error;
    assign debug_memory_read_data_valid = dut.memory_read_data_valid;
    assign debug_memory_read_data_ready = dut.memory_read_data_ready;
    assign debug_memory_wide_read_data_valid =
        dut.memory_wide_read_data_valid;
    assign debug_memory_wide_read_data_ready =
        dut.memory_wide_read_data_ready;
    assign debug_axi_ar_valid = arvalid;
    assign debug_axi_ar_ready = arready;
    assign debug_axi_read_more_bursts =
        dut.memory.axi_transport.axi.read_issued_bytes <
        dut.memory.axi_transport.axi.read_total_bytes;
    assign debug_axi_read_credit_full =
        dut.memory.axi_transport.axi.read_active &&
        debug_axi_read_more_bursts &&
        dut.memory.axi_transport.axi.read_outstanding_wide >=
        dut.memory.axi_transport.axi.READ_CREDIT_LIMIT;
    assign debug_axi_r_valid = rvalid;
    assign debug_axi_r_ready = rready;
    assign debug_axi_r_last = rlast;
    assign quant_active_client = dut.operators.shared_compute.quant_route_client;
    assign quant_idle = dut.operators.shared_compute.shared_quant_idle;
    assign quant_abort_ack =
        dut.operators.shared_compute.shared_quant_abort_ack;
    assign quant_req_valid = dut.operators.shared_compute.shared_quant_req_valid;
    assign quant_req_ready = dut.operators.shared_compute.shared_quant_req_ready;
    assign quant_rsp_valid = dut.operators.shared_compute.shared_quant_rsp_valid;
    assign quant_rsp_ready = dut.operators.shared_compute.shared_quant_rsp_ready;
endmodule

`default_nettype wire
