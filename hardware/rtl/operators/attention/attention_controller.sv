`default_nettype none

// Executes one complete head-local Attention command. Compute is serial;
// the inactive cache panel read and the previous head context write may run in
// parallel with the current compute phase.
module attention_controller #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     abort_request,
    output logic                     abort_ack,
    input  logic                     start_valid,
    output logic                     start_ready,
    input  logic [5:0]               start_query_count,
    input  logic                     start_output_subset_enable,
    input  logic [5:0]               start_output_query_count,
    input  logic                     start_pair_enable,
    input  logic [5:0]               start_second_query_count,
    input  logic                     start_third_enable,
    input  logic [5:0]               start_third_query_count,
    input  logic                     layout_ready,
    output logic                     active_second_batch,
    output logic [1:0]               active_batch_index,
    output logic                     pair_active,
    output logic [7:0]               pair_score_words,
    input  logic [527:0]             start_token_position,
    input  logic [11:0]              start_sequence_length,
    input  logic [5:0]               start_head_count,
    input  logic                     start_q_group_precompute,
    input  logic                     start_q_group_use_staged,
    input  logic                     start_context_max_enable,
    input  hardware_types_pkg::attention_probability_config_t start_probability_config,
    input  logic [5:0]               start_token_batch_index,
    input  logic [5:0]               start_layer,
    input  logic [ADDR_WIDTH-1:0]    start_context_base,
    input  logic [31:0]              start_context_query_stride,
    input  logic [TAG_WIDTH-1:0]     start_tag,
    output logic                     busy,
    output logic                     done_valid,
    input  logic                     done_ready,
    output logic                     error,
    output logic [7:0]               error_id,
    output logic [47:0]              context_row_max_valid,
    output logic [48*16-1:0]         context_row_max_values,
    output local_memory_layout_pkg::local_memory_phase_t local_memory_phase,
    output logic                     q_head_start_valid,
    input  logic                     q_head_start_ready,
    output logic [5:0]               q_head_start_head,
    output logic                     q_head_start_slot,
    output logic                     q_head_postprocess_enable,
    output logic                     q_head_precompute_only,
    output logic                     q_head_use_staged,
    input  logic                     q_head_matmul_done,
    input  logic                     q_head_done_valid,
    output logic                     q_head_done_ready,
    input  logic                     q_head_error,
    input  logic [7:0]               q_head_error_id,
    output logic                     q_head_abort_request,
    input  logic                     q_head_abort_ack,
    output logic                     q_head_active,
    output logic                     q_head_slot,
    output hardware_types_pkg::attention_read_source_t attention_read_source,
    input  logic                     dma_read_stream_idle,

    output logic                     cache_lookup_req_valid,
    input  logic                     cache_lookup_req_ready,
    output logic [5:0]               cache_lookup_req_head,
    output logic [10:0]              cache_lookup_req_key_group,
    output logic [4:0]               cache_lookup_req_chunk,
    output logic [TAG_WIDTH-1:0]     cache_lookup_req_tag,
    input  logic                     cache_lookup_rsp_valid,
    output logic                     cache_lookup_rsp_ready,
    input  logic [10:0]              cache_lookup_rsp_key_group,
    input  logic [7:0]               cache_lookup_rsp_current_mask,
    input  logic [7:0]               cache_lookup_rsp_retained_mask,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_k_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_v_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_k_scale_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_k_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_v_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_k_scale_address,
    input  logic [TAG_WIDTH-1:0]     cache_lookup_rsp_tag,
    output logic                     cache_lookup_abort_request,
    input  logic                     cache_lookup_abort_ack,

    output logic                     cache_dma_request_valid,
    input  logic                     cache_dma_request_ready,
    output logic [ADDR_WIDTH-1:0]    cache_dma_request_address,
    output logic [31:0]              cache_dma_request_bytes,
    output logic [7:0]               cache_dma_request_tag,
    input  logic                     cache_dma_request_error,
    input  logic                     cache_dma_read_valid,
    output logic                     cache_dma_read_ready,
    input  logic [255:0]             cache_dma_read_data,
    input  logic [31:0]              cache_dma_read_byte_enable,
    input  logic                     cache_dma_read_last,
    input  logic [7:0]               cache_dma_read_tag,
    output logic                     cache_panel_write_valid,
    input  logic                     cache_panel_write_ready,
    output logic                     cache_panel_write_target_panel,
    output logic                     cache_panel_write_target_quarter,
    output logic                     cache_panel_write_bank,
    output logic [9:0]               cache_panel_write_address,
    output logic [255:0]             cache_panel_write_data,
    output logic [31:0]              cache_panel_write_byte_enable,
    output logic [TAG_WIDTH-1:0]     cache_panel_write_tag,
    output logic                     cache_scale_write_valid,
    input  logic                     cache_scale_write_ready,
    output logic                     cache_scale_write_target_panel,
    output logic [10:0]              cache_scale_write_token_base,
    output logic [127:0]             cache_scale_write_data,
    output logic [15:0]              cache_scale_write_byte_enable,
    output logic [TAG_WIDTH-1:0]     cache_scale_write_tag,

    output logic                     operand_req_valid,
    input  logic                     operand_req_ready,
    output logic                     operand_req_is_pv,
    output logic [5:0]               operand_req_query_base,
    output logic [11:0]              operand_req_output_base,
    output logic [11:0]              operand_req_k_base,
    output logic [5:0]               operand_active_head,
    output logic                     operand_active_q_slot,
    output logic                     operand_active_panel,
    output logic                     operand_active_quarter,
    input  logic [15:0]              operand_v_static_scale,
    input  logic                     operand_rsp_valid,
    input  logic [1023:0]            operand_rsp_activation_rows,
    input  logic [511:0]             operand_rsp_panel_rows,
    input  logic [127:0]             operand_rsp_metadata,
    input  logic [127:0]             operand_rsp_panel_metadata,
    output logic                     operand_abort_request,
    input  logic                     operand_abort_ack,

    output logic                     pe_req_valid,
    input  logic                     pe_req_ready,
    output logic [1:0]               pe_req_mode,
    output logic [1023:0]            pe_req_activation_payload,
    output logic [1023:0]            pe_req_weight_payload,
    output logic [7:0]               pe_req_row_mask,
    output logic [31:0]              pe_req_k_mask,
    output logic [7:0]               pe_req_col_mask,
    output logic                     pe_req_first_k_step,
    output logic                     pe_req_last_k_step,
    output logic [127:0]             pe_req_activation_scales,
    output logic [127:0]             pe_req_weight_scales,
    output logic [TAG_WIDTH-1:0]     pe_req_tag,
    input  logic                     pe_accum_result_valid,
    output logic                     pe_accum_result_ready,
    input  logic [2047:0]            pe_accum_result_accumulators,
    input  logic [63:0]              pe_accum_result_mask,
    input  logic [1:0]               pe_accum_result_mode,
    input  logic                     pe_accum_result_last_k_step,
    input  logic [127:0]             pe_accum_result_activation_scales,
    input  logic [127:0]             pe_accum_result_weight_scales,
    input  logic [TAG_WIDTH-1:0]     pe_accum_result_tag,
    output logic                     pe_abort_request,
    input  logic                     pe_abort_ack,

    output logic                     rescale_req_valid,
    input  logic                     rescale_req_ready,
    output logic [2047:0]            rescale_req_accumulators,
    output logic [127:0]             rescale_req_activation_scales,
    output logic [127:0]             rescale_req_weight_scales,
    output logic [15:0]              rescale_req_qk_scale_bf16,
    output logic [1:0]               rescale_req_rescale_mode,
    output logic [63:0]              rescale_req_lane_mask,
    output logic [TAG_WIDTH-1:0]     rescale_req_tag,
    input  logic                     rescale_rsp_valid,
    output logic                     rescale_rsp_ready,
    input  logic [1023:0]            rescale_rsp_values,
    input  logic [63:0]              rescale_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]     rescale_rsp_tag,
    output logic                     rescale_abort_request,
    input  logic                     rescale_abort_ack,

    output logic                     score_write_valid,
    input  logic                     score_write_ready,
    output logic [5:0]               score_write_head,
    output logic [5:0]               score_write_query_base,
    output logic [11:0]              score_write_key_base,
    output logic [1023:0]            score_write_values,
    output logic [63:0]              score_write_lane_mask,
    output logic [TAG_WIDTH-1:0]     score_write_tag,
    output logic                     context_buffer_write_valid,
    input  logic                     context_buffer_write_ready,
    output logic [9:0]               context_buffer_write_address,
    output logic [127:0]             context_buffer_write_data,
    output logic [15:0]              context_buffer_write_byte_enable,
    output logic [TAG_WIDTH-1:0]     context_buffer_write_tag,

    output logic                     softmax_score_read_req_valid,
    input  logic                     softmax_score_read_req_ready,
    output logic [1:0]               softmax_score_read_pass,
    output logic [23:0]              softmax_score_read_group_base,
    output logic [5:0]               softmax_score_read_head,
    output logic [5:0]               softmax_score_read_row_base,
    output logic [11:0]              softmax_score_read_key,
    output logic [15:0]              softmax_score_read_tag,
    output logic                     softmax_score_read_slot_ready,
    input  logic                     softmax_score_memory_rsp_valid,
    input  logic [1023:0]            softmax_score_memory_rsp_values,
    output logic                     softmax_probability_write_valid,
    input  logic                     softmax_probability_write_ready,
    output logic                     softmax_probability_write_probability,
    output logic [23:0]              softmax_probability_write_group_base,
    output logic [5:0]               softmax_probability_write_head,
    output logic [5:0]               softmax_probability_write_row_base,
    output logic [11:0]              softmax_probability_write_key,
    output logic [63:0]              softmax_probability_write_lane_mask,
    output logic [1023:0]            softmax_probability_write_values,
    output logic [15:0]              softmax_probability_write_tag,
    output logic                     softmax_quantized_write_valid,
    input  logic                     softmax_quantized_write_ready,
    output logic [5:0]               softmax_quantized_write_head,
    output logic [5:0]               softmax_quantized_write_row_base,
    output logic [11:0]              softmax_quantized_write_key,
    output logic [511:0]             softmax_quantized_write_values,
    output logic [63:0]              softmax_quantized_write_lane_mask,
    output logic [15:0]              softmax_quantized_write_tag,
    output logic                     softmax_scale_write_valid,
    input  logic                     softmax_scale_write_ready,
    output logic [5:0]               softmax_scale_write_head,
    output logic [5:0]               softmax_scale_write_row_base,
    output logic [127:0]             softmax_scale_write_values,
    output logic [7:0]               softmax_scale_write_row_mask,
    output logic                     softmax_arithmetic_req_valid,
    input  logic                     softmax_arithmetic_req_ready,
    output logic [2:0]               softmax_arithmetic_req_operation,
    output logic [1023:0]            softmax_arithmetic_req_values,
    output logic [1023:0]            softmax_arithmetic_req_paired_values,
    output logic [1023:0]            softmax_arithmetic_req_factor0_values,
    output logic [1023:0]            softmax_arithmetic_req_factor1_values,
    output logic [63:0]              softmax_arithmetic_req_lane_mask,
    output logic [15:0]              softmax_arithmetic_req_tag,
    input  logic                     softmax_arithmetic_rsp_valid,
    output logic                     softmax_arithmetic_rsp_ready,
    input  logic [1023:0]            softmax_arithmetic_rsp_values,
    input  logic [63:0]              softmax_arithmetic_rsp_lane_mask,
    input  logic [15:0]              softmax_arithmetic_rsp_tag,
    output logic                     softmax_max_req_valid,
    input  logic                     softmax_max_req_ready,
    output logic                     softmax_max_req_magnitude,
    output logic [1023:0]            softmax_max_req_values,
    output logic [63:0]              softmax_max_req_lane_mask,
    output logic [15:0]              softmax_max_req_tag,
    input  logic                     softmax_max_rsp_valid,
    output logic                     softmax_max_rsp_ready,
    input  logic [127:0]             softmax_max_rsp_values,
    input  logic [7:0]               softmax_max_rsp_row_mask,
    input  logic [15:0]              softmax_max_rsp_tag,
    output logic                     softmax_reduction_req_valid,
    input  logic                     softmax_reduction_req_ready,
    output logic [1023:0]            softmax_reduction_req_values,
    output logic [63:0]              softmax_reduction_req_lane_mask,
    output logic [15:0]              softmax_reduction_req_tag,
    input  logic                     softmax_reduction_rsp_valid,
    output logic                     softmax_reduction_rsp_ready,
    input  logic [127:0]             softmax_reduction_rsp_values,
    input  logic [7:0]               softmax_reduction_rsp_row_mask,
    input  logic [15:0]              softmax_reduction_rsp_tag,
    output logic                     softmax_exp_req_valid,
    input  logic                     softmax_exp_req_ready,
    output logic [1023:0]            softmax_exp_req_delta_values,
    output logic [63:0]              softmax_exp_req_lane_mask,
    output logic [15:0]              softmax_exp_req_tag,
    input  logic                     softmax_exp_rsp_valid,
    output logic                     softmax_exp_rsp_ready,
    input  logic [1023:0]            softmax_exp_rsp_values,
    input  logic [63:0]              softmax_exp_rsp_lane_mask,
    input  logic [15:0]              softmax_exp_rsp_tag,
    output logic                     softmax_reciprocal_req_valid,
    input  logic                     softmax_reciprocal_req_ready,
    output logic [127:0]             softmax_reciprocal_req_sum_values,
    output logic [7:0]               softmax_reciprocal_req_row_mask,
    output logic [15:0]              softmax_reciprocal_req_tag,
    input  logic                     softmax_reciprocal_rsp_valid,
    output logic                     softmax_reciprocal_rsp_ready,
    input  logic [127:0]             softmax_reciprocal_rsp_values,
    input  logic [7:0]               softmax_reciprocal_rsp_row_mask,
    input  logic [15:0]              softmax_reciprocal_rsp_tag,
    output logic                     softmax_quant_scale_req_valid,
    input  logic                     softmax_quant_scale_req_ready,
    output logic [127:0]             softmax_quant_scale_req_row_max_abs,
    output logic [7:0]               softmax_quant_scale_req_row_mask,
    input  logic                     softmax_quant_scale_rsp_valid,
    output logic                     softmax_quant_scale_rsp_ready,
    input  logic [127:0]             softmax_quant_scale_rsp_values,
    output logic                     softmax_quant_values_req_valid,
    input  logic                     softmax_quant_values_req_ready,
    output logic [1023:0]            softmax_quant_values_req_values,
    output logic [63:0]              softmax_quant_values_req_lane_mask,
    output logic [15:0]              softmax_quant_values_req_tag,
    input  logic                     softmax_quant_values_rsp_valid,
    output logic                     softmax_quant_values_rsp_ready,
    input  logic [511:0]             softmax_quant_values_rsp_values,
    input  logic [63:0]              softmax_quant_values_rsp_lane_mask,
    input  logic [15:0]              softmax_quant_values_rsp_tag,
    output logic                     softmax_scratch_read_valid,
    input  logic                     softmax_scratch_read_ready,
    output logic                     softmax_scratch_read_bank,
    output logic [9:0]               softmax_scratch_read_left_address,
    output logic [9:0]               softmax_scratch_read_right_address,
    output logic [15:0]              softmax_scratch_read_tag,
    input  logic                     softmax_scratch_rsp_valid,
    output logic                     softmax_scratch_rsp_ready,
    input  logic [127:0]             softmax_scratch_rsp_left_data,
    input  logic [127:0]             softmax_scratch_rsp_right_data,
    input  logic [15:0]              softmax_scratch_rsp_tag,
    output logic                     softmax_scratch_write_valid,
    input  logic                     softmax_scratch_write_ready,
    output logic                     softmax_scratch_write_bank,
    output logic [9:0]               softmax_scratch_write_address,
    output logic [127:0]             softmax_scratch_write_data,
    output logic [15:0]              softmax_scratch_write_byte_enable,

    output logic                     context_read_req_valid,
    input  logic                     context_read_req_ready,
    output logic [9:0]               context_read_req_address,
    output logic [TAG_WIDTH-1:0]     context_read_req_tag,
    input  logic                     context_read_rsp_valid,
    output logic                     context_read_rsp_ready,
    input  logic [127:0]             context_read_rsp_data,
    input  logic [TAG_WIDTH-1:0]     context_read_rsp_tag,
    output logic                     context_write_request_valid,
    input  logic                     context_write_request_ready,
    output logic [ADDR_WIDTH-1:0]    context_write_request_address,
    output logic [31:0]              context_write_request_bytes,
    output logic [7:0]               context_write_request_tag,
    input  logic                     context_write_request_done,
    input  logic                     context_write_request_error,
    output logic                     context_write_data_valid,
    input  logic                     context_write_data_ready,
    output logic [127:0]             context_write_data,
    output logic [15:0]              context_write_byte_enable,
    output logic                     context_write_data_last,

    output logic [63:0]              completed_head_count,
    output logic [63:0]              qk_command_count,
    output logic [63:0]              pv_command_count,
    output logic [63:0]              cache_compute_overlap_cycle_count,
    output logic [63:0]              context_compute_overlap_cycle_count
);
`ifdef SYNTHESIS
    always_comb begin
        completed_head_count = '0;
        qk_command_count = '0;
        pv_command_count = '0;
        cache_compute_overlap_cycle_count = '0;
        context_compute_overlap_cycle_count = '0;
    end
`endif
    localparam logic [7:0] ERROR_CONFIGURATION = 8'h01;
    localparam logic [7:0] ERROR_CACHE = 8'h02;
    localparam logic [7:0] ERROR_MATMUL = 8'h03;
    localparam logic [7:0] ERROR_RESCALE = 8'h04;
    localparam logic [7:0] ERROR_SOFTMAX = 8'h05;
    localparam logic [7:0] ERROR_CONTEXT = 8'h06;
    localparam logic [7:0] ERROR_PROBABILITY_EXPORT = 8'h07;
    localparam logic [1:0] RESCALE_PV = 2'd1;
    localparam logic [1:0] RESCALE_QK = 2'd2;

    typedef enum logic [4:0] {
        IDLE, WAIT_Q0, WAIT_K, START_QK, RUN_QK,
        START_SOFTMAX, RUN_SOFTMAX, WAIT_Q_MATMUL,
        WAIT_V, START_PV, RUN_PV, START_CONTEXT_WRITE,
        WAIT_NEXT_HEAD, WAIT_FINAL_WRITES, COMPLETE, ERROR_DRAIN, ABORT_DRAIN,
        WAIT_PAIR_CONTEXT_WRITE
    } state_t;
    typedef enum logic [1:0] {LOAD_NONE, LOAD_K, LOAD_V} cache_load_select_t;

    state_t state;
    logic [5:0] query_count, head_count, head_index;
    logic [5:0] second_query_count, third_query_count, active_query_count;
    logic [1:0] batch_count;
    logic [5:0] output_query_count, active_output_query_count;
    logic output_subset_active;
    logic [527:0] active_token_position;
    logic [ADDR_WIDTH-1:0] active_context_base;
    logic pair_layout_ready;
    logic [527:0] token_position;
    logic [11:0] sequence_length, k_panel_base, k_panel_count;
    logic [4:0] v_stripe;
    logic active_panel;
    logic active_quarter;
    logic [ADDR_WIDTH-1:0] context_base;
    logic [31:0] context_query_stride;
    logic [TAG_WIDTH-1:0] operation_tag;
    logic terminal_error;
    logic [7:0] terminal_error_id;

    logic loader_start_pending, loader_inflight, loaded_valid;
    cache_load_select_t loader_select, loaded_select;
    logic [5:0] loader_head, loaded_head;
    logic [11:0] loader_token_base, loader_token_count;
    logic [11:0] loaded_token_base, loaded_token_count;
    logic [3:0] loader_channel_tile, loaded_channel_tile;
    logic loader_target_panel, loader_target_quarter, loaded_target_panel;
    logic [3:0] v_slot_valid;
    logic [1:0] v_slot_base;
    logic [4:0] next_v_load_stripe;
    logic v_prefetch_started;
    logic [1:0] next_v_load_slot;
    logic [1:0] current_v_slot;
    logic next_v_load_slot_available;
    logic cache_start_ready, cache_busy, cache_done_pulse, cache_error;
    logic [3:0] cache_error_id;
    logic cache_abort_ack;

    logic matmul_start_valid, matmul_start_ready, matmul_busy;
    logic matmul_done_pulse;
    logic matmul_error, matmul_abort_ack;
    logic [3:0] matmul_error_id;
    logic matmul_done_seen;
    logic [7:0] matmul_operand_req_row_mask;
    logic [7:0] matmul_operand_req_k_mask;
    logic [7:0] matmul_operand_req_col_mask;
    logic matmul_operand_req_first_k_step;
    logic matmul_operand_req_last_k_step;
    logic [TAG_WIDTH-1:0] matmul_operand_req_tag;
    logic matmul_tile_result_valid, matmul_tile_result_ready;
    logic matmul_tile_result_is_pv;
    logic [5:0] matmul_tile_result_query_base;
    logic [11:0] matmul_tile_result_output_base;
    logic [2047:0] matmul_tile_result_accumulators;
    logic [63:0] matmul_tile_result_mask;
    logic [127:0] matmul_tile_result_activation_scales;
    logic [127:0] matmul_tile_result_weight_scales;
    logic [TAG_WIDTH-1:0] matmul_tile_result_tag;

    logic rescale_outstanding, result_pending;
    logic result_is_pv;
    logic [5:0] result_query_base;
    logic [11:0] result_output_base;
    logic [1023:0] result_values;
    logic [63:0] result_mask;
    logic [TAG_WIDTH-1:0] result_tag;
    logic [3:0] result_row;
    logic [7:0] result_row_mask;
    logic result_write_valid, result_write_ready, result_write_fire;
    logic score_write_fire;

    logic softmax_start_valid, softmax_start_ready, softmax_busy;
    logic softmax_done_pulse, softmax_error, softmax_abort_ack;
    logic [3:0] softmax_error_id;
    logic softmax_score_read_rsp_valid;
    logic softmax_score_read_rsp_ready;
    logic [1023:0] softmax_score_read_rsp_values;
    logic [63:0] softmax_score_read_rsp_lane_mask;
    logic [15:0] softmax_score_read_rsp_tag;
    logic [1:0] softmax_score_read_pending;
    logic [63:0] softmax_score_read_pending_mask [0:1];
    logic [15:0] softmax_score_read_pending_tag [0:1];
    logic [5:0] softmax_engine_probability_write_head;
    logic [5:0] softmax_engine_quantized_write_head;
    logic [5:0] softmax_engine_scale_write_head;
    logic [1103:0] softmax_score_fifo_input_data;
    logic [1103:0] softmax_score_fifo_output_data;
    logic softmax_score_fifo_input_ready;
    logic [1:0] softmax_score_fifo_occupancy;
    logic [2:0] softmax_score_reserved_count;
    logic softmax_score_response_pop;
    logic writer_start_valid, writer_start_ready, writer_busy;
    logic writer_done_pulse;
    logic writer_error, writer_abort_ack, writer_source_released;
    logic [3:0] writer_error_id;
    logic [TAG_WIDTH-1:0] writer_write_request_tag;
    logic writer_inflight, context_buffer_available;

    hardware_types_pkg::attention_probability_config_t probability_config;
    logic capture_valid, capture_ready, capture_active, capture_start_ready;
    logic capture_start, capture_finish, capture_p8_required, capture_stream_ready;
    logic matmul_pe_valid, matmul_pe_ready, capture_stream_valid;
    logic capture_softmax_p8, capture_reads_score, capture_softmax_required;
    logic softmax_p8_valid, softmax_p8_ready, softmax_p8_export_ready;
    // One 8-row scale write precedes its P8 tiles.
    logic [127:0] capture_softmax_scales;
    logic [5:0] capture_softmax_scale_row;
    logic [7:0] capture_softmax_row_mask;
    logic [11:0] capture_stream_key;
    logic [1023:0] capture_stream_values;
    logic [5:0] softmax_capture_row_base;
    logic [7:0] softmax_capture_row_mask;
    logic capture_done, capture_error, capture_abort_ack;
    logic [5:0] capture_row_base;
    logic [7:0] capture_row_mask;
    logic [47:0] capture_query_mask;
    logic [5:0] capture_rounds_remaining;
    logic [64:0] capture_head_base, capture_batch_base;
    logic [64:0] capture_batch_limit;
    logic [64:0] active_batch_round_offset;
    logic [63:0] capture_write_address;
    logic [31:0] capture_write_bytes;
    logic [7:0] capture_write_tag;
    logic capture_write_request_valid, capture_write_valid, capture_write_last;
    logic [127:0] capture_write_data;
    logic [15:0] capture_write_enable;
    logic capture_read_valid;
    logic [11:0] capture_read_key;
    logic [15:0] capture_read_tag;
    logic engine_read_valid, engine_read_rsp_ready;
    logic [1:0] engine_read_pass;
    logic [23:0] engine_read_group_base;
    logic [5:0] engine_read_head, engine_read_row_base;
    logic [11:0] engine_read_key;
    logic [15:0] engine_read_tag;
    logic writer_request_valid, writer_data_valid, writer_data_last;
    logic [ADDR_WIDTH-1:0] writer_request_address;
    logic [31:0] writer_request_bytes;
    logic [127:0] writer_data;
    logic [15:0] writer_byte_enable;

    logic context_max_enable;
    logic context_max_req_valid;
    logic context_max_req_ready;
    logic context_max_rsp_valid;
    logic [15:0] context_max_rsp_value;
    logic [5:0] context_max_rsp_row;
    logic context_max_request_fire;
    logic context_max_response_fire;
    logic context_max_response_error;
    logic [7:0] context_max_outstanding;
`ifndef SYNTHESIS
    logic [31:0] context_max_request_count;
    logic [31:0] context_max_response_count;
`endif
    logic [15:0] context_row_max [0:47];

    logic [63:0] unused_cache_lookup_count, unused_cache_dma_request_count;
    logic [63:0] unused_cache_dma_byte_count, unused_cache_panel_write_count;
    logic [63:0] unused_cache_scale_write_count, unused_cache_overlap_count;
    logic [63:0] unused_matmul_operand_req_count, unused_matmul_operand_rsp_count;
    logic [63:0] unused_matmul_pe_req_count, unused_matmul_pe_rsp_count;
    logic [63:0] unused_matmul_tile_count;
    logic [63:0] unused_softmax_counter [0:9];
    logic [63:0] unused_writer_counter [0:6];

    logic start_configuration_valid;
    logic cache_start_fire, matmul_start_fire, rescale_req_fire;
    logic rescale_rsp_fire, result_final_row;
    logic softmax_start_fire, writer_start_fire;
    logic matmul_path_drained, children_drained;
    logic child_error_event, stop_issue;
    logic [7:0] child_error_id;
    logic [11:0] next_k_base;
    logic [11:0] next_k_count;
    logic q_head_start_pending;
    logic q_head_inflight;
    logic [5:0] q_head_request_head;
    logic q_head_request_slot;
    logic [1:0] q_slot_valid;
    logic q_head_postprocess_allowed;
    logic q_group_precompute_enabled;
    logic q_group_precompute_phase;
    logic q_head_start_fire;
    logic q_head_done_fire;

    logic cache_dma_start_valid;
    logic cache_dma_start_ready;
    logic [ADDR_WIDTH-1:0] cache_dma_start_address;
    logic [31:0] cache_dma_start_bytes;
    logic cache_dma_data_valid;
    logic cache_dma_data_ready;
    logic [255:0] cache_dma_data;
    logic [31:0] cache_dma_data_byte_enable;
    logic [31:0] cache_dma_data_offset;
    logic cache_dma_done;
    logic cache_dma_error;
    logic cache_dma_abort_request;
    logic cache_dma_abort_ack;

    import local_memory_layout_pkg::*;

    function automatic [15:0] duplicate_byte_mask(input [7:0] lane_mask);
        begin
            for (integer lane = 0; lane < 8; lane = lane + 1)
                duplicate_byte_mask[lane*2 +: 2] = {2{lane_mask[lane]}};
        end
    endfunction

    assign start_configuration_valid = start_query_count != 0 &&
        start_query_count <= 6'd48 && start_sequence_length != 0 &&
        start_sequence_length <= 12'd2048 && start_head_count != 0 &&
        start_head_count <= 6'd32 && start_context_query_stride >= 32'd8192 &&
        (!start_q_group_precompute || start_q_group_use_staged) &&
        (!start_output_subset_enable ||
            (!start_pair_enable && start_output_query_count <= start_query_count)) &&
        (!start_pair_enable || (attention_pair_sequence_supported(start_sequence_length) &&
            start_second_query_count != 0 && start_second_query_count <= 6'd48 &&
            start_context_query_stride == 32'd8192)) &&
        (!start_third_enable || (start_pair_enable &&
            attention_pair_sequence_supported(start_sequence_length) &&
            start_third_query_count != 0 && start_third_query_count <= 6'd48 &&
            start_context_query_stride == 32'd8192)) &&
        (!start_probability_config.enable ||
            (start_probability_config.output_base[3:0] == 0 &&
             start_probability_config.batch_stride_bytes[3:0] == 0 &&
             start_probability_config.batch_stride_bytes != 0 &&
             start_probability_config.head_stride_bytes[3:0] == 0 &&
             start_probability_config.query_begin <= start_probability_config.query_end &&
             start_probability_config.query_end <= start_sequence_length &&
             start_probability_config.round_stride_bytes[3:0] == 0 &&
             {5'd0, start_probability_config.round_stride_bytes} >=
                {start_probability_config.head_stride_bytes, 5'd0} &&
             {3'd0, start_probability_config.head_stride_bytes} >=
                35'(start_probability_config.batch_stride_bytes)*35'd6));
    assign start_ready = state == IDLE && !abort_request;
    assign busy = state != IDLE;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign pair_layout_ready = !pair_active || layout_ready;
    assign active_second_batch = active_batch_index != 2'd0;
    assign active_query_count = active_batch_index == 2'd2 ? third_query_count :
        active_batch_index == 2'd1 ? second_query_count : query_count;
    assign active_output_query_count = pair_active ? active_query_count : output_query_count;
    assign active_token_position = pair_active ? start_token_position : token_position;
    assign active_context_base = context_base +
        (active_batch_index == 2'd2 ?
            (ADDR_WIDTH'(query_count + second_query_count) << 13) :
         active_batch_index == 2'd1 ? (ADDR_WIDTH'(query_count) << 13) : '0);
    assign active_batch_round_offset = active_batch_index == 2'd2 ?
        ({33'd0, probability_config.round_stride_bytes} << 1) :
        active_batch_index == 2'd1 ?
            {33'd0, probability_config.round_stride_bytes} : 65'd0;
    assign operand_active_head = head_index;
    assign operand_active_q_slot = pair_active ? active_batch_index[0] : head_index[0];
    assign operand_active_panel = active_panel;
    assign operand_active_quarter = active_quarter;
    assign softmax_probability_write_head = head_index;
    assign softmax_quantized_write_head = head_index;
    assign softmax_scale_write_head = head_index;
    assign next_k_base = k_panel_base + k_panel_count;
    assign next_k_count = sequence_length - next_k_base >= 12'd256 ?
        12'd256 : sequence_length - next_k_base;
    assign next_v_load_slot = v_slot_base + next_v_load_stripe[1:0];
    assign current_v_slot = pair_active ? 2'd0 : v_slot_base + v_stripe[1:0];
    assign next_v_load_slot_available =
        !v_slot_valid[next_v_load_slot] &&
        !((state == START_PV || state == RUN_PV) &&
          next_v_load_slot == {active_panel, active_quarter}) &&
        !((state == WAIT_K || state == START_QK || state == RUN_QK) &&
          next_v_load_slot[1] == active_panel);
    assign q_head_start_valid = q_head_start_pending && !stop_issue && pair_layout_ready &&
        (!pair_active || local_memory_phase == LOCAL_MEMORY_PHASE_ATTENTION_QK);
    assign q_head_start_head = q_head_request_head;
    assign q_head_start_slot = q_head_request_slot;
    assign q_head_postprocess_enable = !q_group_precompute_phase &&
        (pair_active || q_head_request_head == 6'd0 ||
         q_head_postprocess_allowed);
    assign q_head_precompute_only = q_group_precompute_phase;
    assign q_head_use_staged = q_group_precompute_enabled &&
        !q_group_precompute_phase;
    assign q_head_start_fire = q_head_start_valid && q_head_start_ready;
    assign q_head_done_ready = q_head_inflight;
    assign q_head_done_fire = q_head_done_valid && q_head_done_ready;
    assign q_head_abort_request = stop_issue && q_head_inflight;
    assign q_head_active = q_head_inflight;
    assign q_head_slot = q_head_request_slot;

    always_ff @(posedge clk) begin
        if (rst || start_valid && start_ready) begin
            local_memory_phase <= LOCAL_MEMORY_PHASE_ATTENTION_QK;
        end else begin
            case (state)
                IDLE, COMPLETE:
                    local_memory_phase <= LOCAL_MEMORY_PHASE_ATTENTION_QK;
                WAIT_Q0, WAIT_K, START_QK, RUN_QK, WAIT_NEXT_HEAD:
                    local_memory_phase <= LOCAL_MEMORY_PHASE_ATTENTION_QK;
                START_SOFTMAX, RUN_SOFTMAX:
                    local_memory_phase <=
                        LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX;
                WAIT_Q_MATMUL, WAIT_V, START_PV, RUN_PV, START_CONTEXT_WRITE,
                WAIT_FINAL_WRITES, WAIT_PAIR_CONTEXT_WRITE:
                    local_memory_phase <= LOCAL_MEMORY_PHASE_ATTENTION_PV;
                default: local_memory_phase <= local_memory_phase;
            endcase
        end
    end

    assign child_error_event = (cache_done_pulse && cache_error) ||
        (q_head_done_fire && q_head_error) ||
        (matmul_done_pulse && matmul_error) ||
        (softmax_done_pulse && softmax_error) ||
        (capture_done && capture_error) ||
        (writer_done_pulse && writer_error) ||
        context_max_response_error ||
        (rescale_rsp_fire && (rescale_rsp_tag != result_tag ||
            rescale_rsp_lane_mask != result_mask));
    assign child_error_id =
        (capture_done && capture_error) ? ERROR_PROBABILITY_EXPORT :
        (rescale_rsp_fire && (rescale_rsp_tag != result_tag ||
            rescale_rsp_lane_mask != result_mask)) ? ERROR_RESCALE :
        (q_head_done_fire && q_head_error) ? q_head_error_id :
        ((writer_done_pulse && writer_error) || context_max_response_error) ?
            ERROR_CONTEXT :
        (softmax_done_pulse && softmax_error) ? ERROR_SOFTMAX :
        (matmul_done_pulse && matmul_error) ? ERROR_MATMUL : ERROR_CACHE;
    assign stop_issue = abort_request || child_error_event ||
        state == ERROR_DRAIN || state == ABORT_DRAIN;

    assign cache_start_fire = loader_start_pending && cache_start_ready && !stop_issue && pair_layout_ready;
    assign matmul_start_valid = !stop_issue && pair_layout_ready &&
        ((state == START_QK && (!pair_active || local_memory_phase == LOCAL_MEMORY_PHASE_ATTENTION_QK)) ||
        (state == START_PV && (!pair_active || local_memory_phase == LOCAL_MEMORY_PHASE_ATTENTION_PV) &&
         (v_stripe != 0 || context_buffer_available ||
         (pair_active && active_second_batch))));
    assign matmul_start_fire = matmul_start_valid && matmul_start_ready;
    assign matmul_tile_result_ready = rescale_req_ready && !rescale_outstanding &&
        !result_pending && !stop_issue;
    assign rescale_req_valid = matmul_tile_result_valid &&
        !rescale_outstanding && !result_pending && !stop_issue;
    assign rescale_req_accumulators = matmul_tile_result_accumulators;
    assign rescale_req_activation_scales = matmul_tile_result_activation_scales;
    assign rescale_req_weight_scales = matmul_tile_result_weight_scales;
    // QK applies the fixed head-dimension scale after Q and K scales. PV has
    // only the probability and V scales, so its third multiplier remains 1.0.
    assign rescale_req_qk_scale_bf16 = matmul_tile_result_is_pv ?
        16'h3f80 : 16'h3db5;
    assign rescale_req_rescale_mode = matmul_tile_result_is_pv ? RESCALE_PV : RESCALE_QK;
    assign rescale_req_lane_mask = matmul_tile_result_mask;
    assign rescale_req_tag = matmul_tile_result_tag;
    assign rescale_req_fire = rescale_req_valid && rescale_req_ready;
    assign rescale_rsp_ready = rescale_outstanding && !result_pending;
    assign rescale_rsp_fire = rescale_rsp_valid && rescale_rsp_ready;
    assign rescale_abort_request = stop_issue && rescale_outstanding;

    assign result_row_mask = result_mask[result_row*8 +: 8];
    assign score_write_valid = result_pending && !result_is_pv &&
        result_mask != 0 && !stop_issue;
    assign score_write_fire = score_write_valid && score_write_ready;
    assign result_write_valid = result_pending && result_is_pv &&
        result_row_mask != 0 && !stop_issue;
    assign context_buffer_write_valid = result_write_valid &&
        (!context_max_enable || context_max_req_ready);
    assign result_write_ready = context_buffer_write_ready &&
        (!context_max_enable || context_max_req_ready);
    assign result_write_fire = result_write_valid && result_write_ready;
    assign result_final_row = result_row == 4'd7;
    assign score_write_head = head_index;
    assign score_write_query_base = result_query_base;
    assign score_write_key_base = result_output_base;
    assign score_write_values = result_values;
    assign score_write_lane_mask = result_mask;
    assign score_write_tag = result_tag;
    assign context_buffer_write_address =
        ((10'(result_query_base) + 10'(result_row)) << 4) +
        10'(result_output_base >> 3);
    assign context_buffer_write_data = result_values[result_row*128 +: 128];
    assign context_buffer_write_byte_enable = duplicate_byte_mask(result_row_mask);
    assign context_buffer_write_tag = result_tag;

    // The dedicated 8-value pipe avoids contention with the next Q head, which
    // can use the shared max resource while the current PV result is written.
    assign context_max_req_valid = context_max_enable && result_write_valid &&
        context_buffer_write_ready;
    assign context_max_request_fire = context_max_req_valid &&
        context_max_req_ready;
    assign context_max_response_fire = context_max_rsp_valid;
    assign context_max_response_error = context_max_response_fire &&
        context_max_rsp_row >= output_query_count;

    bf16_vector8_max_pipe #(.TAG_WIDTH(6)) context_max_pipe (
        .clk(clk), .rst(rst), .req_valid(context_max_req_valid),
        .req_ready(context_max_req_ready),
        .req_values(context_buffer_write_data),
        .req_lane_mask(result_row_mask),
        .req_tag(6'(result_query_base + result_row)),
        .rsp_valid(context_max_rsp_valid), .rsp_ready(1'b1),
        .rsp_value(context_max_rsp_value), .rsp_tag(context_max_rsp_row));

    for (genvar max_row = 0; max_row < 48; max_row++) begin : pack_context_max
        assign context_row_max_values[max_row*16 +: 16] =
            context_row_max[max_row];
    end

    assign softmax_start_valid = state == START_SOFTMAX && !stop_issue &&
        capture_rounds_remaining == 0 && pair_layout_ready &&
        (!pair_active || local_memory_phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX);
    assign softmax_start_fire = softmax_start_valid && softmax_start_ready;
    assign writer_start_valid = state == START_CONTEXT_WRITE && !writer_inflight &&
        !stop_issue && pair_layout_ready &&
        (!pair_active || local_memory_phase == LOCAL_MEMORY_PHASE_ATTENTION_PV);
    assign writer_start_fire = writer_start_valid && writer_start_ready;
    assign context_write_request_valid = capture_active ? capture_write_request_valid : writer_request_valid;
    assign context_write_request_address = capture_active ? capture_write_address : writer_request_address;
    assign context_write_request_bytes = capture_active ? capture_write_bytes : writer_request_bytes;
    assign context_write_request_tag = capture_active ? capture_write_tag : writer_write_request_tag[7:0];
    assign context_write_data_valid = capture_active ? capture_write_valid : writer_data_valid;
    assign context_write_data = capture_active ? capture_write_data : writer_data;
    assign context_write_byte_enable = capture_active ? capture_write_enable : writer_byte_enable;
    assign context_write_data_last = capture_active ? capture_write_last : writer_data_last;
    assign matmul_path_drained = matmul_done_seen && !rescale_outstanding &&
        !result_pending;
    assign children_drained = !cache_busy && !loader_start_pending &&
        !loader_inflight && !loaded_valid && v_slot_valid == 4'b0000 &&
        !matmul_busy && !softmax_busy &&
        !writer_busy && !writer_inflight && !capture_active && !rescale_outstanding &&
        !result_pending && !q_head_inflight && !q_head_start_pending &&
        context_max_outstanding == 0 && dma_read_stream_idle;

    operator_dma_reader #(.DATA_WIDTH(256), .REQUEST_TAG(8'h81)) cache_dma_reader (
        .clk(clk),
        .rst(rst),
        .abort_request(cache_dma_abort_request),
        .start_valid(cache_dma_start_valid),
        .start_ready(cache_dma_start_ready),
        .start_address(cache_dma_start_address),
        .start_bytes(cache_dma_start_bytes),
        .start_total_bytes(cache_dma_start_bytes),
        .request_valid(cache_dma_request_valid),
        .request_ready(cache_dma_request_ready),
        .request_address(cache_dma_request_address),
        .request_bytes(cache_dma_request_bytes),
        .request_tag(cache_dma_request_tag),
        .read_valid(cache_dma_read_valid),
        .read_ready(cache_dma_read_ready),
        .read_data(cache_dma_read_data),
        .read_byte_enable(cache_dma_read_byte_enable),
        .read_last(cache_dma_read_last),
        .read_tag(cache_dma_read_tag),
        .response_error(cache_dma_request_error),
        .upstream_abort_ack(1'b0),
        .data_valid(cache_dma_data_valid),
        .data_ready(cache_dma_data_ready),
        .data(cache_dma_data),
        .data_byte_enable(cache_dma_data_byte_enable),
        .data_last(),
        .data_byte_offset(cache_dma_data_offset),
        .done_pulse(cache_dma_done),
        .error(cache_dma_error),
        .abort_ack(cache_dma_abort_ack)
    );

    attention_cache_loader #(.ADDR_WIDTH(ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH)) cache_loader (
        .clk(clk), .rst(rst), .abort_request(stop_issue), .abort_ack(cache_abort_ack),
        .start_valid(loader_start_pending && !stop_issue && pair_layout_ready),
        .start_ready(cache_start_ready),
        .start_is_v(loader_select == LOAD_V), .start_head(loader_head),
        .start_token_base(loader_token_base[10:0]),
        .start_token_count(loader_token_count),
        .start_channel_tile(loader_channel_tile),
        .start_target_panel(loader_target_panel),
        .start_target_quarter(loader_target_quarter), .start_tag(operation_tag),
        .busy(cache_busy), .done_pulse(cache_done_pulse), .error(cache_error),
        .error_id(cache_error_id),
        .cache_lookup_req_valid(cache_lookup_req_valid),
        .cache_lookup_req_ready(cache_lookup_req_ready),
        .cache_lookup_req_head(cache_lookup_req_head),
        .cache_lookup_req_key_group(cache_lookup_req_key_group),
        .cache_lookup_req_chunk(cache_lookup_req_chunk),
        .cache_lookup_req_tag(cache_lookup_req_tag),
        .cache_lookup_rsp_valid(cache_lookup_rsp_valid),
        .cache_lookup_rsp_ready(cache_lookup_rsp_ready),
        .cache_lookup_rsp_key_group(cache_lookup_rsp_key_group),
        .cache_lookup_rsp_current_mask(cache_lookup_rsp_current_mask),
        .cache_lookup_rsp_retained_mask(cache_lookup_rsp_retained_mask),
        .cache_lookup_rsp_current_k_address(cache_lookup_rsp_current_k_address),
        .cache_lookup_rsp_current_v_address(cache_lookup_rsp_current_v_address),
        .cache_lookup_rsp_current_k_scale_address(
            cache_lookup_rsp_current_k_scale_address),
        .cache_lookup_rsp_retained_k_address(cache_lookup_rsp_retained_k_address),
        .cache_lookup_rsp_retained_v_address(cache_lookup_rsp_retained_v_address),
        .cache_lookup_rsp_retained_k_scale_address(
            cache_lookup_rsp_retained_k_scale_address),
        .cache_lookup_rsp_tag(cache_lookup_rsp_tag),
        .cache_lookup_abort_request(cache_lookup_abort_request),
        .cache_lookup_abort_ack(cache_lookup_abort_ack),
        .dma_start_valid(cache_dma_start_valid), .dma_start_ready(cache_dma_start_ready),
        .dma_start_address(cache_dma_start_address), .dma_start_bytes(cache_dma_start_bytes),
        .dma_data_valid(cache_dma_data_valid), .dma_data_ready(cache_dma_data_ready),
        .dma_data(cache_dma_data), .dma_data_byte_enable(cache_dma_data_byte_enable),
        .dma_data_offset(cache_dma_data_offset), .dma_done(cache_dma_done),
        .dma_error(cache_dma_error), .dma_abort_request(cache_dma_abort_request),
        .dma_abort_ack(cache_dma_abort_ack),
        .panel_write_valid(cache_panel_write_valid),
        .panel_write_ready(cache_panel_write_ready),
        .panel_write_target_panel(cache_panel_write_target_panel),
        .panel_write_target_quarter(cache_panel_write_target_quarter),
        .panel_write_bank(cache_panel_write_bank),
        .panel_write_address(cache_panel_write_address),
        .panel_write_data(cache_panel_write_data),
        .panel_write_byte_enable(cache_panel_write_byte_enable),
        .panel_write_tag(cache_panel_write_tag),
        .scale_write_valid(cache_scale_write_valid),
        .scale_write_ready(cache_scale_write_ready),
        .scale_write_target_panel(cache_scale_write_target_panel),
        .scale_write_token_base(cache_scale_write_token_base),
        .scale_write_data(cache_scale_write_data),
        .scale_write_byte_enable(cache_scale_write_byte_enable),
        .scale_write_tag(cache_scale_write_tag),
        .accepted_cache_lookup_request_count(unused_cache_lookup_count),
        .accepted_dma_request_count(unused_cache_dma_request_count),
        .accepted_dma_byte_count(unused_cache_dma_byte_count),
        .accepted_panel_write_count(unused_cache_panel_write_count),
        .accepted_scale_write_count(unused_cache_scale_write_count),
        .load_compute_overlap_cycle_count(unused_cache_overlap_count)
    );

    attention_matmul_engine #(.TAG_WIDTH(TAG_WIDTH)) matmul (
        .clk(clk), .rst(rst), .abort_request(stop_issue),
        .abort_ack(matmul_abort_ack),
        .start_valid(matmul_start_valid), .start_ready(matmul_start_ready),
        .start_is_pv(state == START_PV),
        .start_query_count(state == START_PV ? active_output_query_count : active_query_count),
        .start_output_base(state == START_PV ? {4'd0, v_stripe, 3'd0} : k_panel_base),
        .start_output_count(state == START_PV ? 12'd8 : k_panel_count),
        .start_reduction_count(state == START_PV ? sequence_length : 12'd128),
        .busy(matmul_busy), .done_pulse(matmul_done_pulse),
        .error(matmul_error),
        .error_id(matmul_error_id),
        .operand_req_valid(operand_req_valid), .operand_req_ready(operand_req_ready),
        .operand_req_is_pv(operand_req_is_pv),
        .operand_req_query_base(operand_req_query_base),
        .operand_req_output_base(operand_req_output_base),
        .operand_req_k_base(operand_req_k_base),
        .operand_req_row_mask(matmul_operand_req_row_mask),
        .operand_req_k_mask(matmul_operand_req_k_mask),
        .operand_req_col_mask(matmul_operand_req_col_mask),
        .operand_req_first_k_step(matmul_operand_req_first_k_step),
        .operand_req_last_k_step(matmul_operand_req_last_k_step),
        .operand_req_tag(matmul_operand_req_tag),
        .operand_v_static_scale(operand_v_static_scale),
        .operand_rsp_valid(operand_rsp_valid),
        .operand_rsp_activation_rows(operand_rsp_activation_rows),
        .operand_rsp_panel_rows(operand_rsp_panel_rows),
        .operand_rsp_metadata(operand_rsp_metadata),
        .operand_rsp_panel_metadata(operand_rsp_panel_metadata),
        .operand_abort_request(operand_abort_request),
        .operand_abort_ack(operand_abort_ack),
        .pe_req_valid(matmul_pe_valid), .pe_req_ready(matmul_pe_ready),
        .pe_req_mode(pe_req_mode),
        .pe_req_activation_payload(pe_req_activation_payload),
        .pe_req_weight_payload(pe_req_weight_payload),
        .pe_req_row_mask(pe_req_row_mask), .pe_req_k_mask(pe_req_k_mask),
        .pe_req_col_mask(pe_req_col_mask),
        .pe_req_first_k_step(pe_req_first_k_step),
        .pe_req_last_k_step(pe_req_last_k_step),
        .pe_req_activation_scales(pe_req_activation_scales),
        .pe_req_weight_scales(pe_req_weight_scales), .pe_req_tag(pe_req_tag),
        .pe_accum_result_valid(pe_accum_result_valid),
        .pe_accum_result_ready(pe_accum_result_ready),
        .pe_accum_result_accumulators(pe_accum_result_accumulators),
        .pe_accum_result_mask(pe_accum_result_mask),
        .pe_accum_result_mode(pe_accum_result_mode),
        .pe_accum_result_last_k_step(pe_accum_result_last_k_step),
        .pe_accum_result_activation_scales(pe_accum_result_activation_scales),
        .pe_accum_result_weight_scales(pe_accum_result_weight_scales),
        .pe_accum_result_tag(pe_accum_result_tag),
        .pe_abort_request(pe_abort_request), .pe_abort_ack(pe_abort_ack),
        .tile_result_valid(matmul_tile_result_valid),
        .tile_result_ready(matmul_tile_result_ready),
        .tile_result_is_pv(matmul_tile_result_is_pv),
        .tile_result_query_base(matmul_tile_result_query_base),
        .tile_result_output_base(matmul_tile_result_output_base),
        .tile_result_accumulators(matmul_tile_result_accumulators),
        .tile_result_mask(matmul_tile_result_mask),
        .tile_result_activation_scales(matmul_tile_result_activation_scales),
        .tile_result_weight_scales(matmul_tile_result_weight_scales),
        .tile_result_tag(matmul_tile_result_tag),
        .accepted_operand_request_count(unused_matmul_operand_req_count),
        .completed_operand_response_count(unused_matmul_operand_rsp_count),
        .accepted_pe_request_count(unused_matmul_pe_req_count),
        .completed_pe_accum_result_count(unused_matmul_pe_rsp_count),
        .completed_output_tile_count(unused_matmul_tile_count)
    );

    assign softmax_score_fifo_input_data = {
        softmax_score_read_pending_tag[1],
        softmax_score_read_pending_mask[1],
        softmax_score_memory_rsp_values
    };
    assign softmax_score_response_pop = softmax_score_read_rsp_valid &&
        softmax_score_read_rsp_ready;
    assign capture_reads_score = capture_active && !probability_config.capture_p8;
    assign softmax_score_read_rsp_ready = capture_reads_score || engine_read_rsp_ready;
    assign softmax_score_read_req_valid = capture_reads_score ? capture_read_valid : engine_read_valid;
    assign softmax_score_read_pass = capture_reads_score ? 2'd3 : engine_read_pass;
    assign softmax_score_read_group_base = capture_reads_score ? 24'd0 : engine_read_group_base;
    assign softmax_score_read_head = capture_reads_score ? 6'd0 : engine_read_head;
    assign softmax_score_read_row_base = capture_reads_score ? capture_row_base : engine_read_row_base;
    assign softmax_score_read_key = capture_reads_score ? capture_read_key : engine_read_key;
    assign softmax_score_read_tag = capture_reads_score ? capture_read_tag : engine_read_tag;
    assign softmax_score_reserved_count =
        {1'b0, softmax_score_fifo_occupancy} +
        3'($countones(softmax_score_read_pending));
    assign softmax_score_read_slot_ready =
        softmax_score_reserved_count < 3'd2 || softmax_score_response_pop;
    assign {
        softmax_score_read_rsp_tag,
        softmax_score_read_rsp_lane_mask,
        softmax_score_read_rsp_values
    } = softmax_score_fifo_output_data;

    ready_valid_fifo #(.DATA_WIDTH(1104), .DEPTH(2))
        softmax_score_response_fifo (
            .clk(clk), .rst(rst),
            .input_valid(softmax_score_memory_rsp_valid),
            .input_ready(softmax_score_fifo_input_ready),
            .input_data(softmax_score_fifo_input_data),
            .output_valid(softmax_score_read_rsp_valid),
            .output_ready(softmax_score_read_rsp_ready),
            .output_data(softmax_score_fifo_output_data),
            .occupancy(softmax_score_fifo_occupancy)
        );

    always_ff @(posedge clk) begin
        if (rst) begin
            softmax_score_read_pending <= '0;
            softmax_score_read_pending_mask[0] <= '0;
            softmax_score_read_pending_mask[1] <= '0;
            softmax_score_read_pending_tag[0] <= '0;
            softmax_score_read_pending_tag[1] <= '0;
        end else begin
            softmax_score_read_pending <= {
                softmax_score_read_pending[0],
                softmax_score_read_req_valid &&
                    softmax_score_read_req_ready};
            softmax_score_read_pending_mask[1] <=
                softmax_score_read_pending_mask[0];
            softmax_score_read_pending_tag[1] <=
                softmax_score_read_pending_tag[0];
            if (softmax_score_read_req_valid &&
                softmax_score_read_req_ready) begin
                softmax_score_read_pending_tag[0] <=
                    softmax_score_read_tag;
                for (integer score_row = 0; score_row < 8; score_row++)
                    for (integer score_lane = 0; score_lane < 8;
                         score_lane++)
                        softmax_score_read_pending_mask[0][
                            score_row*8 + score_lane] <=
                            integer'(softmax_score_read_row_base) + score_row <
                                integer'(active_query_count) &&
                            integer'(softmax_score_read_key) + score_lane <
                                integer'(sequence_length);
            end
        end
    end

    softmax_engine #(.ADDR_WIDTH(24), .TAG_WIDTH(16)) softmax (
        .start_capture_probability_enable(probability_config.enable && !probability_config.capture_p8), .capture_probability_ready(capture_ready),
        .capture_probability_valid(capture_valid), .capture_probability_head(),
        .capture_probability_row_base(softmax_capture_row_base), .capture_probability_row_mask(softmax_capture_row_mask),
        .clk(clk), .rst(rst), .abort_request(stop_issue),
        .abort_ack(softmax_abort_ack),
        .start_valid(softmax_start_valid), .start_ready(softmax_start_ready),
        .start_row_count(active_query_count), .start_head_count(6'd1),
        .start_row_length(sequence_length), .start_score_base(24'd0),
        .start_score_head_stride(24'd0),
        .start_score_row_stride(24'(sequence_length)),
        .start_probability_base(24'd0),
        .start_probability_head_stride(24'd0),
        .start_probability_row_stride(24'(sequence_length)),
        .busy(softmax_busy), .done_pulse(softmax_done_pulse),
        .error(softmax_error),
        .error_id(softmax_error_id),
        .tile_read_req_valid(engine_read_valid),
        .tile_read_req_ready(softmax_score_read_req_ready && !capture_reads_score),
        .tile_read_pass(engine_read_pass),
        .tile_read_group_base(engine_read_group_base),
        .tile_read_head(engine_read_head),
        .tile_read_row_base(engine_read_row_base),
        .tile_read_key(engine_read_key),
        .tile_read_tag(engine_read_tag),
        .tile_read_rsp_valid(softmax_score_read_rsp_valid && !capture_reads_score),
        .tile_read_rsp_ready(engine_read_rsp_ready),
        .tile_read_rsp_values(softmax_score_read_rsp_values),
        .tile_read_rsp_lane_mask(softmax_score_read_rsp_lane_mask),
        .tile_read_rsp_tag(softmax_score_read_rsp_tag),
        .bf16_write_valid(softmax_probability_write_valid),
        .bf16_write_ready(softmax_probability_write_ready),
        .bf16_write_probability(softmax_probability_write_probability),
        .bf16_write_group_base(softmax_probability_write_group_base),
        .bf16_write_head(softmax_engine_probability_write_head),
        .bf16_write_row_base(softmax_probability_write_row_base),
        .bf16_write_key(softmax_probability_write_key),
        .bf16_write_values(softmax_probability_write_values),
        .bf16_write_lane_mask(softmax_probability_write_lane_mask),
        .bf16_write_tag(softmax_probability_write_tag),
        .quantized_write_valid(softmax_p8_valid),
        .quantized_write_ready(softmax_p8_ready),
        .quantized_write_head(softmax_engine_quantized_write_head),
        .quantized_write_row_base(softmax_quantized_write_row_base),
        .quantized_write_key(softmax_quantized_write_key),
        .quantized_write_values(softmax_quantized_write_values),
        .quantized_write_lane_mask(softmax_quantized_write_lane_mask),
        .quantized_write_tag(softmax_quantized_write_tag),
        .scale_write_valid(softmax_scale_write_valid),
        .scale_write_ready(softmax_scale_write_ready),
        .scale_write_head(softmax_engine_scale_write_head),
        .scale_write_row_base(softmax_scale_write_row_base),
        .scale_write_values(softmax_scale_write_values),
        .scale_write_row_mask(softmax_scale_write_row_mask),
        .scratch_read_valid(softmax_scratch_read_valid),
        .scratch_read_ready(softmax_scratch_read_ready),
        .scratch_read_bank(softmax_scratch_read_bank),
        .scratch_read_left_address(softmax_scratch_read_left_address),
        .scratch_read_right_address(softmax_scratch_read_right_address),
        .scratch_read_tag(softmax_scratch_read_tag),
        .scratch_rsp_valid(softmax_scratch_rsp_valid),
        .scratch_rsp_ready(softmax_scratch_rsp_ready),
        .scratch_rsp_left_data(softmax_scratch_rsp_left_data),
        .scratch_rsp_right_data(softmax_scratch_rsp_right_data),
        .scratch_rsp_tag(softmax_scratch_rsp_tag),
        .scratch_write_valid(softmax_scratch_write_valid),
        .scratch_write_ready(softmax_scratch_write_ready),
        .scratch_write_bank(softmax_scratch_write_bank),
        .scratch_write_address(softmax_scratch_write_address),
        .scratch_write_data(softmax_scratch_write_data),
        .scratch_write_byte_enable(softmax_scratch_write_byte_enable),
        .vector_req_valid(softmax_arithmetic_req_valid),
        .vector_req_ready(softmax_arithmetic_req_ready),
        .vector_req_operation(softmax_arithmetic_req_operation),
        .vector_req_values(softmax_arithmetic_req_values),
        .vector_req_paired_values(softmax_arithmetic_req_paired_values),
        .vector_req_factor0_values(softmax_arithmetic_req_factor0_values),
        .vector_req_factor1_values(softmax_arithmetic_req_factor1_values),
        .vector_req_lane_mask(softmax_arithmetic_req_lane_mask),
        .vector_req_tag(softmax_arithmetic_req_tag),
        .vector_rsp_valid(softmax_arithmetic_rsp_valid),
        .vector_rsp_ready(softmax_arithmetic_rsp_ready),
        .vector_rsp_values(softmax_arithmetic_rsp_values),
        .vector_rsp_lane_mask(softmax_arithmetic_rsp_lane_mask),
        .vector_rsp_tag(softmax_arithmetic_rsp_tag),
        .max_req_valid(softmax_max_req_valid),
        .max_req_ready(softmax_max_req_ready),
        .max_req_magnitude(softmax_max_req_magnitude),
        .max_req_values(softmax_max_req_values),
        .max_req_lane_mask(softmax_max_req_lane_mask),
        .max_req_tag(softmax_max_req_tag),
        .max_rsp_valid(softmax_max_rsp_valid),
        .max_rsp_ready(softmax_max_rsp_ready),
        .max_rsp_values(softmax_max_rsp_values),
        .max_rsp_row_mask(softmax_max_rsp_row_mask),
        .max_rsp_tag(softmax_max_rsp_tag),
        .reduction_req_valid(softmax_reduction_req_valid),
        .reduction_req_ready(softmax_reduction_req_ready),
        .reduction_req_values(softmax_reduction_req_values),
        .reduction_req_lane_mask(softmax_reduction_req_lane_mask),
        .reduction_req_tag(softmax_reduction_req_tag),
        .reduction_rsp_valid(softmax_reduction_rsp_valid),
        .reduction_rsp_ready(softmax_reduction_rsp_ready),
        .reduction_rsp_values(softmax_reduction_rsp_values),
        .reduction_rsp_row_mask(softmax_reduction_rsp_row_mask),
        .reduction_rsp_tag(softmax_reduction_rsp_tag),
        .exp_req_valid(softmax_exp_req_valid),
        .exp_req_ready(softmax_exp_req_ready),
        .exp_req_delta_values(softmax_exp_req_delta_values),
        .exp_req_lane_mask(softmax_exp_req_lane_mask),
        .exp_req_tag(softmax_exp_req_tag),
        .exp_rsp_valid(softmax_exp_rsp_valid),
        .exp_rsp_ready(softmax_exp_rsp_ready),
        .exp_rsp_values(softmax_exp_rsp_values),
        .exp_rsp_lane_mask(softmax_exp_rsp_lane_mask),
        .exp_rsp_tag(softmax_exp_rsp_tag),
        .reciprocal_req_valid(softmax_reciprocal_req_valid),
        .reciprocal_req_ready(softmax_reciprocal_req_ready),
        .reciprocal_req_sum_values(softmax_reciprocal_req_sum_values),
        .reciprocal_req_row_mask(softmax_reciprocal_req_row_mask),
        .reciprocal_req_tag(softmax_reciprocal_req_tag),
        .reciprocal_rsp_valid(softmax_reciprocal_rsp_valid),
        .reciprocal_rsp_ready(softmax_reciprocal_rsp_ready),
        .reciprocal_rsp_values(softmax_reciprocal_rsp_values),
        .reciprocal_rsp_row_mask(softmax_reciprocal_rsp_row_mask),
        .reciprocal_rsp_tag(softmax_reciprocal_rsp_tag),
        .quant_scale_req_valid(softmax_quant_scale_req_valid),
        .quant_scale_req_ready(softmax_quant_scale_req_ready),
        .quant_scale_req_row_max_abs(softmax_quant_scale_req_row_max_abs),
        .quant_scale_req_row_mask(softmax_quant_scale_req_row_mask),
        .quant_scale_rsp_valid(softmax_quant_scale_rsp_valid),
        .quant_scale_rsp_ready(softmax_quant_scale_rsp_ready),
        .quant_scale_rsp_values(softmax_quant_scale_rsp_values),
        .quant_values_req_valid(softmax_quant_values_req_valid),
        .quant_values_req_ready(softmax_quant_values_req_ready),
        .quant_values_req_values(softmax_quant_values_req_values),
        .quant_values_req_lane_mask(softmax_quant_values_req_lane_mask),
        .quant_values_req_tag(softmax_quant_values_req_tag),
        .quant_values_rsp_valid(softmax_quant_values_rsp_valid),
        .quant_values_rsp_ready(softmax_quant_values_rsp_ready),
        .quant_values_rsp_values(softmax_quant_values_rsp_values),
        .quant_values_rsp_lane_mask(softmax_quant_values_rsp_lane_mask),
        .quant_values_rsp_tag(softmax_quant_values_rsp_tag),
        .accepted_source_tile_count(unused_softmax_counter[0]),
        .accepted_bf16_write_count(unused_softmax_counter[1]),
        .accepted_quantized_write_count(unused_softmax_counter[2]),
        .accepted_scratch_read_count(unused_softmax_counter[3]),
        .accepted_scratch_write_count(unused_softmax_counter[4]),
        .accepted_vector_request_count(unused_softmax_counter[5]),
        .accepted_max_request_count(unused_softmax_counter[6]),
        .accepted_reduction_request_count(unused_softmax_counter[7]),
        .accepted_quantized_value_count(unused_softmax_counter[8]),
        .completed_command_count()
    );

    assign capture_softmax_p8 = output_subset_active && probability_config.enable &&
        probability_config.capture_p8;
    for (genvar capture_row = 0; capture_row < 8; capture_row++) begin : p8_row_mask
        assign capture_softmax_row_mask[capture_row] =
            |softmax_quantized_write_lane_mask[capture_row*8 +: 8];
    end
    assign capture_row_base = capture_softmax_p8 ? softmax_quantized_write_row_base :
        probability_config.capture_p8 ? {pe_req_tag[14:12], 3'd0} : softmax_capture_row_base;
    assign capture_row_mask = capture_softmax_p8 ? capture_softmax_row_mask :
        probability_config.capture_p8 ? pe_req_row_mask : softmax_capture_row_mask;
    assign capture_softmax_required = capture_softmax_p8 && |probability_config.key_groups &&
        |(capture_softmax_row_mask & 8'(capture_query_mask >> capture_row_base));
    assign capture_p8_required = probability_config.enable && probability_config.capture_p8 &&
        !capture_softmax_p8 && state == RUN_PV && v_stripe == 0 && |probability_config.key_groups &&
        |(pe_req_row_mask & 8'(capture_query_mask >> capture_row_base));
    // The SRAM and exporter accept each selected P8 tile on the same cycle.
    assign softmax_p8_export_ready = !capture_softmax_p8 ||
        (!capture_softmax_required && !capture_active) ||
        (capture_softmax_required && capture_active && capture_stream_ready);
    assign softmax_quantized_write_valid = softmax_p8_valid && softmax_p8_export_ready;
    assign softmax_p8_ready = softmax_quantized_write_ready && softmax_p8_export_ready;
    assign pe_req_valid = matmul_pe_valid &&
        (capture_softmax_p8 || !(capture_p8_required || (probability_config.capture_p8 && capture_active)) || (capture_active && capture_stream_ready));
    assign matmul_pe_ready = pe_req_ready &&
        (capture_softmax_p8 || !(capture_p8_required || (probability_config.capture_p8 && capture_active)) || (capture_active && capture_stream_ready));
    assign capture_stream_valid = capture_softmax_p8 ?
        softmax_p8_valid && softmax_quantized_write_ready && capture_softmax_required && capture_active :
        matmul_pe_valid && pe_req_ready && capture_p8_required && capture_active;
    always_comb begin
        capture_stream_values = '0;
        for (integer row = 0; row < 8; row++) begin
            capture_stream_values[row*128 +: 64] = capture_softmax_p8 ?
                softmax_quantized_write_values[row*64 +: 64] : pe_req_activation_payload[row*64 +: 64];
            capture_stream_values[row*128+64 +: 16] = capture_softmax_p8 ?
                capture_softmax_scales[row*16 +: 16] : pe_req_activation_scales[row*16 +: 16];
        end
    end
    assign capture_start = (capture_softmax_p8 ? (softmax_p8_valid && capture_softmax_required) :
        probability_config.capture_p8 ? (matmul_pe_valid && capture_p8_required) : capture_valid) &&
        !capture_active && !writer_busy && !writer_inflight &&
        (capture_softmax_p8 || softmax_score_reserved_count == 0) && !stop_issue;
    assign capture_finish = capture_active && capture_done && !capture_error && !stop_issue;
    assign capture_ready = capture_finish && !probability_config.capture_p8;
    always_ff @(posedge clk) begin
        if (rst) begin
            probability_config <= '0;
            capture_active <= 1'b0;
            capture_head_base <= '0;
            capture_batch_base <= '0;
            capture_batch_limit <= '0;
            capture_query_mask <= '0;
            capture_rounds_remaining <= '0;
            capture_stream_key <= '0;
            capture_softmax_scales <= '0;
            capture_softmax_scale_row <= '0;
        end else begin
            if (capture_softmax_p8 && softmax_scale_write_valid && softmax_scale_write_ready) begin
                capture_softmax_scales <= softmax_scale_write_values;
                capture_softmax_scale_row <= softmax_scale_write_row_base;
            end
            if (start_valid && start_ready) begin
                probability_config <= start_probability_config;
                probability_config.enable <= start_probability_config.enable && start_layer < 32 &&
                    start_probability_config.layer_mask[start_layer[4:0]];
                capture_rounds_remaining <= start_probability_config.enable ? start_token_batch_index : 6'd0;
                for (integer row = 0; row < 48; row++)
                    capture_query_mask[row] <= {1'b0, start_token_position[row*11 +: 11]} >=
                        start_probability_config.query_begin &&
                        {1'b0, start_token_position[row*11 +: 11]} < start_probability_config.query_end;
                capture_head_base <= {1'b0, start_probability_config.output_base};
                capture_batch_base <= {1'b0, start_probability_config.output_base};
                capture_batch_limit <= {1'b0, start_probability_config.output_base} +
                    {33'd0, start_probability_config.batch_stride_bytes};
            end else if (capture_rounds_remaining != 0) begin
                capture_rounds_remaining <= capture_rounds_remaining - 6'd1;
                capture_head_base <= capture_head_base + {33'd0, probability_config.round_stride_bytes};
                capture_batch_base <= capture_batch_base + {33'd0, probability_config.round_stride_bytes};
            end
            if (state == WAIT_NEXT_HEAD && loaded_valid && loaded_select == LOAD_K &&
                loaded_head == head_index + 6'd1 && loaded_token_base == 0 &&
                q_slot_valid[~head_index[0]] && (!output_subset_active || !capture_active) && !stop_issue) begin
                capture_head_base <= capture_head_base + {33'd0, probability_config.head_stride_bytes};
                capture_batch_base <= capture_head_base + {33'd0, probability_config.head_stride_bytes};
            end
            if (pair_active && state == WAIT_PAIR_CONTEXT_WRITE && writer_done_pulse &&
                !writer_error && active_batch_index + 2'd1 >= batch_count &&
                head_index + 6'd1 < head_count)
                capture_head_base <= capture_head_base + {33'd0, probability_config.head_stride_bytes};
            if (softmax_start_fire)
                capture_batch_limit <= capture_batch_base + {33'd0, probability_config.batch_stride_bytes};
            if (pair_active && (softmax_start_fire ||
                (matmul_start_fire && state == START_PV && v_stripe == 0))) begin
                capture_batch_base <= capture_head_base +
                    active_batch_round_offset;
                capture_batch_limit <= capture_head_base +
                    active_batch_round_offset +
                    {33'd0, probability_config.batch_stride_bytes};
                for (integer row = 0; row < 48; row++)
                    capture_query_mask[row] <= row < integer'(active_query_count) &&
                        {1'b0, active_token_position[row*11 +: 11]} >= probability_config.query_begin &&
                        {1'b0, active_token_position[row*11 +: 11]} < probability_config.query_end;
            end
            if (capture_start && capture_start_ready) begin
                capture_active <= 1'b1;
                capture_stream_key <= '0;
            end
            if (capture_stream_valid && capture_stream_ready)
                capture_stream_key <= capture_stream_key + 12'd8;
            if (capture_finish) begin
                capture_active <= 1'b0;
                capture_batch_base <= capture_batch_base + {33'd0, probability_config.batch_stride_bytes};
                capture_batch_limit <= capture_batch_limit + {33'd0, probability_config.batch_stride_bytes};
            end
            if (probability_config.enable && probability_config.capture_p8 && !capture_softmax_p8 && state == RUN_PV && v_stripe == 0 &&
                !capture_p8_required && pe_req_valid && pe_req_ready && pe_req_first_k_step) begin
                capture_batch_base <= capture_batch_base + {33'd0, probability_config.batch_stride_bytes};
                capture_batch_limit <= capture_batch_limit + {33'd0, probability_config.batch_stride_bytes};
            end
            if (capture_softmax_p8 && !capture_softmax_required &&
                softmax_quantized_write_valid && softmax_quantized_write_ready &&
                softmax_quantized_write_key == 0) begin
                capture_batch_base <= capture_batch_base + {33'd0, probability_config.batch_stride_bytes};
                capture_batch_limit <= capture_batch_limit + {33'd0, probability_config.batch_stride_bytes};
            end
            if (stop_issue && (capture_abort_ack || capture_done || capture_start_ready))
                capture_active <= 1'b0;
        end
    end
    attention_probability_export probability_export (
        .clk, .rst, .abort_request(stop_issue), .abort_ack(capture_abort_ack),
        .start_valid(capture_start), .start_ready(capture_start_ready),
        .start_p8_stream(probability_config.capture_p8),
        .stream_valid(capture_stream_valid), .stream_ready(capture_stream_ready),
        .stream_key(capture_stream_key), .stream_values(capture_stream_values),
        .start_sequence_length(sequence_length),
        .start_row_mask(capture_row_mask & 8'(capture_query_mask >> capture_row_base)),
        .start_key_groups(probability_config.key_groups),
        .start_output_address(capture_batch_base[63:0]),
        .start_output_limit(capture_batch_base[64] || capture_batch_limit[64] ? 64'd0 :
            capture_batch_limit[63:0] < probability_config.output_limit ?
                capture_batch_limit[63:0] : probability_config.output_limit),
        .done_valid(capture_done), .done_ready(1'b1), .error(capture_error), .error_id(), .output_bytes(),
        .source_request_valid(capture_read_valid), .source_request_ready(softmax_score_read_req_ready && capture_reads_score),
        .source_request_key(capture_read_key), .source_request_tag(capture_read_tag),
        .source_response_valid(softmax_score_read_rsp_valid && capture_reads_score),
        .source_response_values(softmax_score_read_rsp_values),
        .source_response_lane_mask(softmax_score_read_rsp_lane_mask), .source_response_tag(softmax_score_read_rsp_tag),
        .write_request_valid(capture_write_request_valid),
        .write_request_ready(context_write_request_ready && capture_active),
        .write_request_address(capture_write_address), .write_request_bytes(capture_write_bytes),
        .write_request_tag(capture_write_tag), .write_valid(capture_write_valid),
        .write_ready(context_write_data_ready && capture_active), .write_data(capture_write_data),
        .write_byte_enable(capture_write_enable), .write_last(capture_write_last),
        .write_done(context_write_request_done && capture_active), .write_error(context_write_request_error && capture_active)
    );

    attention_context_writer #(.ADDR_WIDTH(ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH)) writer (
        .clk(clk), .rst(rst), .abort_request(stop_issue),
        .abort_ack(writer_abort_ack),
        .start_valid(writer_start_valid), .start_ready(writer_start_ready),
        .start_query_count(active_output_query_count), .start_head(head_index),
        .start_context_base(active_context_base),
        .start_context_query_stride(context_query_stride),
        .start_tag(operation_tag), .busy(writer_busy),
        .done_pulse(writer_done_pulse),
        .error(writer_error), .error_id(writer_error_id),
        .source_buffer_released(writer_source_released),
        .context_read_req_valid(context_read_req_valid),
        .context_read_req_ready(context_read_req_ready),
        .context_read_req_address(context_read_req_address),
        .context_read_req_tag(context_read_req_tag),
        .context_read_rsp_valid(context_read_rsp_valid),
        .context_read_rsp_ready(context_read_rsp_ready),
        .context_read_rsp_data(context_read_rsp_data),
        .context_read_rsp_tag(context_read_rsp_tag),
        .write_request_valid(writer_request_valid),
        .write_request_ready(context_write_request_ready && !capture_active),
        .write_request_address(writer_request_address),
        .write_request_bytes(writer_request_bytes),
        .write_request_tag(writer_write_request_tag),
        .write_request_done(context_write_request_done && !capture_active),
        .write_request_error(context_write_request_error && !capture_active),
        .write_data_valid(writer_data_valid),
        .write_data_ready(context_write_data_ready && !capture_active),
        .write_data(writer_data),
        .write_byte_enable(writer_byte_enable),
        .write_data_last(writer_data_last),
        .accepted_read_request_count(unused_writer_counter[0]),
        .completed_read_response_count(unused_writer_counter[1]),
        .accepted_fragment_request_count(unused_writer_counter[2]),
        .accepted_write_data_count(unused_writer_counter[3]),
        .completed_fragment_response_count(unused_writer_counter[4]),
        .accepted_write_byte_count(unused_writer_counter[5]),
        .completed_command_count(unused_writer_counter[6])
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            query_count <= '0;
            output_query_count <= '0;
            output_subset_active <= 1'b0;
            second_query_count <= '0;
            third_query_count <= '0;
            batch_count <= 2'd1;
            pair_active <= 1'b0;
            active_batch_index <= 2'd0;
            pair_score_words <= '0;
            token_position <= '0;
            head_count <= '0;
            head_index <= '0;
            sequence_length <= '0;
            k_panel_base <= '0;
            k_panel_count <= '0;
            v_stripe <= '0;
            active_panel <= 1'b0;
            active_quarter <= 1'b0;
            context_base <= '0;
            context_query_stride <= '0;
            operation_tag <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            loader_start_pending <= 1'b0;
            loader_inflight <= 1'b0;
            loaded_valid <= 1'b0;
            loader_select <= LOAD_NONE;
            loaded_select <= LOAD_NONE;
            loader_head <= '0;
            loaded_head <= '0;
            loader_token_base <= '0;
            loader_token_count <= '0;
            loaded_token_base <= '0;
            loaded_token_count <= '0;
            loader_channel_tile <= '0;
            loaded_channel_tile <= '0;
            loader_target_panel <= 1'b0;
            loader_target_quarter <= 1'b0;
            loaded_target_panel <= 1'b0;
            v_slot_valid <= '0;
            v_slot_base <= '0;
            next_v_load_stripe <= '0;
            v_prefetch_started <= 1'b0;
            matmul_done_seen <= 1'b0;
            rescale_outstanding <= 1'b0;
            result_pending <= 1'b0;
            result_is_pv <= 1'b0;
            result_query_base <= '0;
            result_output_base <= '0;
            result_values <= '0;
            result_mask <= '0;
            result_tag <= '0;
            result_row <= '0;
            writer_inflight <= 1'b0;
            context_buffer_available <= 1'b1;
`ifndef SYNTHESIS
            completed_head_count <= '0;
            qk_command_count <= '0;
            pv_command_count <= '0;
            cache_compute_overlap_cycle_count <= '0;
            context_compute_overlap_cycle_count <= '0;
`endif
            q_head_start_pending <= 1'b0;
            q_head_inflight <= 1'b0;
            q_head_request_head <= '0;
            q_head_request_slot <= 1'b0;
            q_slot_valid <= '0;
            q_head_postprocess_allowed <= 1'b0;
            q_group_precompute_enabled <= 1'b0;
            q_group_precompute_phase <= 1'b0;
            attention_read_source <= hardware_types_pkg::ATTENTION_READ_NONE;
            context_max_enable <= 1'b0;
            context_row_max_valid <= '0;
            context_max_outstanding <= '0;
`ifndef SYNTHESIS
            context_max_request_count <= '0;
            context_max_response_count <= '0;
`endif
            for (integer max_row = 0; max_row < 48; max_row++)
                context_row_max[max_row] <= '0;
        end else begin
            abort_ack <= 1'b0;

            case ({context_max_request_fire, context_max_response_fire})
                2'b10: context_max_outstanding <=
                    context_max_outstanding + 8'd1;
                2'b01: context_max_outstanding <=
                    context_max_outstanding - 8'd1;
                default: context_max_outstanding <= context_max_outstanding;
            endcase
`ifndef SYNTHESIS
            if (context_max_request_fire)
                context_max_request_count <= context_max_request_count + 32'd1;
            if (context_max_response_fire)
                context_max_response_count <= context_max_response_count + 32'd1;
`endif
            if (context_max_response_fire && !context_max_response_error) begin
                for (integer max_row = 0; max_row < 48; max_row++) begin
                    if (context_max_rsp_row == 6'(max_row)) begin
                        context_row_max_valid[max_row] <= 1'b1;
                        if (!context_row_max_valid[max_row] ||
                            context_max_rsp_value[14:0] >
                                context_row_max[max_row][14:0])
                            context_row_max[max_row] <=
                                {1'b0, context_max_rsp_value[14:0]};
                    end
                end
            end

            if (start_valid && start_ready) begin
                query_count <= start_query_count;
                output_query_count <= start_output_subset_enable ? start_output_query_count : start_query_count;
                output_subset_active <= start_output_subset_enable && start_output_query_count < start_query_count;
                second_query_count <= start_second_query_count;
                third_query_count <= start_third_query_count;
                batch_count <= start_third_enable ? 2'd3 :
                    start_pair_enable ? 2'd2 : 2'd1;
                pair_active <= start_pair_enable;
                active_batch_index <= 2'd0;
                pair_score_words <= 8'(attention_pair_score_stride(start_sequence_length));
                token_position <= start_token_position;
                sequence_length <= start_sequence_length;
                head_count <= start_head_count;
                context_max_enable <= start_context_max_enable && !start_pair_enable &&
                    (!start_output_subset_enable || start_output_query_count != 0);
                head_index <= 6'd0;
                context_base <= start_context_base;
                context_query_stride <= start_context_query_stride;
                operation_tag <= start_tag;
                terminal_error <= !start_configuration_valid;
                terminal_error_id <= start_configuration_valid ?
                    8'd0 : ERROR_CONFIGURATION;
                k_panel_base <= 12'd0;
                k_panel_count <= start_sequence_length >= 12'd256 ?
                    12'd256 : start_sequence_length;
                v_stripe <= 5'd0;
                active_panel <= 1'b0;
                active_quarter <= 1'b0;
                loader_start_pending <= 1'b0;
                loader_select <= LOAD_K;
                loader_head <= 6'd0;
                loader_token_base <= 12'd0;
                loader_token_count <= start_sequence_length >= 12'd256 ?
                    12'd256 : start_sequence_length;
                loader_channel_tile <= 4'd0;
                loader_target_panel <= 1'b0;
                loader_target_quarter <= 1'b0;
                loaded_valid <= 1'b0;
                v_slot_valid <= '0;
                v_slot_base <= '0;
                next_v_load_stripe <= '0;
                v_prefetch_started <= 1'b0;
                matmul_done_seen <= 1'b0;
                writer_inflight <= 1'b0;
                context_buffer_available <= 1'b1;
                q_head_start_pending <= start_configuration_valid;
                q_head_request_head <= 6'd0;
                q_head_request_slot <= 1'b0;
                q_slot_valid <= '0;
                q_head_postprocess_allowed <= 1'b0;
                q_group_precompute_enabled <= start_q_group_use_staged;
                q_group_precompute_phase <= start_q_group_precompute;
                context_row_max_valid <= '0;
                context_max_outstanding <= '0;
`ifndef SYNTHESIS
                context_max_request_count <= '0;
                context_max_response_count <= '0;
`endif
                for (integer max_row = 0; max_row < 48; max_row++)
                    context_row_max[max_row] <= '0;
                attention_read_source <= start_configuration_valid ?
                    hardware_types_pkg::ATTENTION_READ_Q_MATMUL :
                    hardware_types_pkg::ATTENTION_READ_NONE;
                state <= start_configuration_valid ? WAIT_Q0 : COMPLETE;
            end

            if (q_head_start_fire) begin
                q_head_start_pending <= 1'b0;
                q_head_inflight <= 1'b1;
            end
            if (q_head_done_fire) begin
                q_head_inflight <= 1'b0;
                q_head_postprocess_allowed <= 1'b0;
                if (!q_head_error && !q_group_precompute_phase)
                    q_slot_valid[q_head_request_slot] <= 1'b1;
            end
            if (q_head_abort_ack)
                q_head_inflight <= 1'b0;
            if (q_head_matmul_done && !q_group_precompute_phase &&
                state != ERROR_DRAIN && state != ABORT_DRAIN)
                attention_read_source <=
                    hardware_types_pkg::ATTENTION_READ_CACHE;

            if (cache_start_fire) begin
                loader_start_pending <= 1'b0;
                loader_inflight <= 1'b1;
            end
            if (cache_done_pulse) begin
                loader_inflight <= 1'b0;
                if (!cache_error) begin
                    if (loader_select == LOAD_V) begin
                        v_slot_valid[{loader_target_panel,
                            loader_target_quarter}] <= 1'b1;
                    end else begin
                        loaded_valid <= 1'b1;
                        loaded_select <= loader_select;
                        loaded_head <= loader_head;
                        loaded_token_base <= loader_token_base;
                        loaded_token_count <= loader_token_count;
                        loaded_channel_tile <= loader_channel_tile;
                        loaded_target_panel <= loader_target_panel;
                    end
                end
            end
            if (cache_abort_ack)
                loader_inflight <= 1'b0;

            if (!pair_active && output_query_count != 0 && v_prefetch_started && next_v_load_stripe < 5'd16 &&
                !loader_start_pending && !loader_inflight && !loaded_valid &&
                next_v_load_slot_available && !stop_issue) begin
                loader_start_pending <= 1'b1;
                loader_select <= LOAD_V;
                loader_head <= head_index;
                loader_token_base <= 12'd0;
                loader_token_count <= sequence_length;
                loader_channel_tile <= next_v_load_stripe[3:0];
                loader_target_panel <= next_v_load_slot[1];
                loader_target_quarter <= next_v_load_slot[0];
                next_v_load_stripe <= next_v_load_stripe + 5'd1;
            end

            if (matmul_start_fire) begin
                matmul_done_seen <= 1'b0;
                if (state == START_QK) begin
`ifndef SYNTHESIS
                    qk_command_count <= qk_command_count + 64'd1;
`endif
                    if (!pair_active && (next_k_base < sequence_length || output_query_count != 0) &&
                        !loader_start_pending && !loader_inflight && !loaded_valid) begin
                        loader_start_pending <= 1'b1;
                        loader_target_panel <= !active_panel;
                        loader_target_quarter <= 1'b0;
                        loader_head <= head_index;
                        if (next_k_base < sequence_length) begin
                            loader_select <= LOAD_K;
                            loader_token_base <= next_k_base;
                            loader_token_count <= next_k_count;
                            loader_channel_tile <= 4'd0;
                        end else begin
                            loader_select <= LOAD_V;
                            loader_token_base <= 12'd0;
                            loader_token_count <= sequence_length;
                            loader_channel_tile <= 4'd0;
                            v_slot_base <= {!active_panel, 1'b0};
                            next_v_load_stripe <= 5'd1;
                            v_prefetch_started <= 1'b1;
                        end
                    end
                    state <= RUN_QK;
                end else begin
`ifndef SYNTHESIS
                    pv_command_count <= pv_command_count + 64'd1;
`endif
                    if (v_stripe == 0)
                        context_buffer_available <= 1'b0;
                    // V stripes 14 and 15 use the current panel. The opposite
                    // panel is free after stripe 13 drains, so the next K panel
                    // can start one PV stripe earlier.
                    if (!pair_active && v_stripe >= 5'd14 && head_index + 6'd1 < head_count &&
                        !loader_start_pending && !loader_inflight && !loaded_valid) begin
                        loader_start_pending <= 1'b1;
                        loader_target_panel <= !active_panel;
                        loader_target_quarter <= 1'b0;
                        loader_token_base <= 12'd0;
                        loader_select <= LOAD_K;
                        loader_head <= head_index + 6'd1;
                        loader_token_count <= sequence_length >= 12'd256 ?
                            12'd256 : sequence_length;
                        loader_channel_tile <= 4'd0;
                    end
                    state <= RUN_PV;
                end
            end

            if (matmul_done_pulse) begin
                matmul_done_seen <= 1'b1;
            end

            if (rescale_req_fire) begin
                rescale_outstanding <= 1'b1;
                result_is_pv <= matmul_tile_result_is_pv;
                result_query_base <= matmul_tile_result_query_base;
                result_output_base <= matmul_tile_result_output_base;
                result_mask <= matmul_tile_result_mask;
                result_tag <= matmul_tile_result_tag;
            end
            if (rescale_rsp_fire) begin
                rescale_outstanding <= 1'b0;
                if (rescale_rsp_tag == result_tag &&
                    rescale_rsp_lane_mask == result_mask) begin
                    result_values <= rescale_rsp_values;
                    result_mask <= rescale_rsp_lane_mask;
                    result_row <= 4'd0;
                    result_pending <= 1'b1;
                end
            end
            if (result_pending && !result_is_pv) begin
                if (result_mask == 0 || score_write_fire)
                    result_pending <= 1'b0;
            end else if (result_pending) begin
                if (result_row_mask == 0) begin
                    if (result_final_row)
                        result_pending <= 1'b0;
                    else
                        result_row <= result_row + 4'd1;
                end else if (result_write_fire) begin
                    if (result_final_row)
                        result_pending <= 1'b0;
                    else
                        result_row <= result_row + 4'd1;
                end
            end
            if ((state == ERROR_DRAIN || state == ABORT_DRAIN) &&
                rescale_abort_ack)
                rescale_outstanding <= 1'b0;

            if (writer_start_fire) begin
                writer_inflight <= 1'b1;
                state <= pair_active ? WAIT_PAIR_CONTEXT_WRITE : head_index + 6'd1 >= head_count ?
                    WAIT_FINAL_WRITES : WAIT_NEXT_HEAD;
            end
            if (writer_source_released)
                context_buffer_available <= 1'b1;
            if (writer_done_pulse) begin
                writer_inflight <= 1'b0;
            end
            if (writer_abort_ack)
                writer_inflight <= 1'b0;

            if (!child_error_event && (!abort_request || state == ABORT_DRAIN)) case (state)
                IDLE: begin end
                WAIT_Q0: if (q_head_done_fire && !q_head_error) begin
                    if (q_group_precompute_phase) begin
                        if (q_head_request_head + 6'd1 < head_count) begin
                            q_head_request_head <= q_head_request_head + 6'd1;
                            q_head_request_slot <= 1'b0;
                            q_head_start_pending <= 1'b1;
                        end else begin
                            q_group_precompute_phase <= 1'b0;
                            q_head_request_head <= 6'd0;
                            q_head_request_slot <= 1'b0;
                            q_head_start_pending <= 1'b1;
                        end
                    end else if (pair_active && active_batch_index + 2'd1 < batch_count) begin
                        active_batch_index <= active_batch_index + 2'd1;
                        q_head_request_slot <= ~active_batch_index[0];
                        q_head_start_pending <= 1'b1;
                        attention_read_source <= hardware_types_pkg::ATTENTION_READ_Q_MATMUL;
                    end else begin
                        active_batch_index <= 2'd0;
                        loader_start_pending <= 1'b1;
                        state <= WAIT_K;
                    end
                end
                WAIT_K: if (loaded_valid && loaded_select == LOAD_K &&
                    loaded_head == head_index && loaded_token_base == k_panel_base) begin
                    active_panel <= loaded_target_panel;
                    active_quarter <= 1'b0;
                    k_panel_count <= loaded_token_count;
                    loaded_valid <= 1'b0;
                    state <= START_QK;
                end
                START_QK: begin end
                RUN_QK: if (matmul_path_drained) begin
                    matmul_done_seen <= 1'b0;
                    if (pair_active && active_batch_index + 2'd1 < batch_count) begin
                        active_batch_index <= active_batch_index + 2'd1;
                        state <= START_QK;
                    end else if (next_k_base < sequence_length) begin
                        active_batch_index <= 2'd0;
                        k_panel_base <= next_k_base;
                        k_panel_count <= next_k_count;
                        if (pair_active) begin
                            loader_start_pending <= 1'b1;
                            loader_select <= LOAD_K;
                            loader_head <= head_index;
                            loader_token_base <= next_k_base;
                            loader_token_count <= next_k_count;
                            loader_target_panel <= 1'b0;
                            loader_target_quarter <= 1'b0;
                        end
                        state <= WAIT_K;
                    end else begin
                        active_batch_index <= 2'd0;
                        if (pair_active) begin
                            loader_start_pending <= 1'b1;
                            loader_select <= LOAD_V;
                            loader_head <= head_index;
                            loader_token_base <= '0;
                            loader_token_count <= sequence_length;
                            loader_channel_tile <= '0;
                            loader_target_panel <= 1'b0;
                            loader_target_quarter <= 1'b0;
                        end
                        state <= START_SOFTMAX;
                    end
                    if (!pair_active) q_slot_valid[head_index[0]] <= 1'b0;
                end
                START_SOFTMAX: if (softmax_start_fire) begin
                    if (!pair_active && head_index + 6'd1 < head_count) begin
                        q_head_start_pending <= 1'b1;
                        q_head_request_head <= head_index + 6'd1;
                        q_head_request_slot <= ~head_index[0];
                        attention_read_source <=
                            hardware_types_pkg::ATTENTION_READ_Q_MATMUL;
                    end
                    state <= RUN_SOFTMAX;
                end
                RUN_SOFTMAX: if (softmax_done_pulse && !softmax_error) begin
                    q_head_postprocess_allowed <= 1'b1;
                    if (output_query_count == 0) begin
`ifndef SYNTHESIS
                        completed_head_count <= completed_head_count + 64'd1;
`endif
                        state <= head_index + 6'd1 < head_count ? WAIT_NEXT_HEAD : WAIT_FINAL_WRITES;
                    end else if (pair_active && active_batch_index + 2'd1 < batch_count) begin
                        active_batch_index <= active_batch_index + 2'd1;
                        state <= START_SOFTMAX;
                    end else begin
                        active_batch_index <= 2'd0;
                        state <= !pair_active && head_index + 6'd1 < head_count &&
                            !q_head_matmul_done ? WAIT_Q_MATMUL : WAIT_V;
                    end
                end
                WAIT_Q_MATMUL: if (q_head_matmul_done)
                    state <= WAIT_V;
                WAIT_V: if (v_slot_valid[current_v_slot]) begin
                    active_panel <= current_v_slot[1];
                    active_quarter <= current_v_slot[0];
                    v_slot_valid[current_v_slot] <= 1'b0;
                    state <= START_PV;
                end
                START_PV: begin end
                RUN_PV: if (matmul_path_drained && (!pair_active || !capture_active)) begin
                    matmul_done_seen <= 1'b0;
                    if (pair_active && active_batch_index + 2'd1 < batch_count) begin
                        active_batch_index <= active_batch_index + 2'd1;
                        state <= START_PV;
                    end else if (v_stripe < 5'd15) begin
                        active_batch_index <= 2'd0;
                        v_stripe <= v_stripe + 5'd1;
                        if (pair_active) begin
                            loader_start_pending <= 1'b1;
                            loader_select <= LOAD_V;
                            loader_head <= head_index;
                            loader_token_base <= '0;
                            loader_token_count <= sequence_length;
                            loader_channel_tile <= v_stripe[3:0] + 4'd1;
                            loader_target_panel <= 1'b0;
                            loader_target_quarter <= 1'b0;
                        end
                        state <= WAIT_V;
                    end else begin
                        active_batch_index <= 2'd0;
`ifndef SYNTHESIS
                        completed_head_count <= completed_head_count + 64'd1;
`endif
                        state <= START_CONTEXT_WRITE;
                    end
                end
                START_CONTEXT_WRITE: begin end
                WAIT_PAIR_CONTEXT_WRITE: if (writer_done_pulse && !writer_error) begin
                    if (active_batch_index + 2'd1 < batch_count) begin
                        active_batch_index <= active_batch_index + 2'd1;
                        state <= START_CONTEXT_WRITE;
                    end else if (head_index + 6'd1 < head_count) begin
                        active_batch_index <= 2'd0;
                        head_index <= head_index + 6'd1;
                        k_panel_base <= '0;
                        k_panel_count <= sequence_length >= 12'd256 ? 12'd256 : sequence_length;
                        v_stripe <= '0;
                        q_head_request_head <= head_index + 6'd1;
                        q_head_request_slot <= 1'b0;
                        q_head_start_pending <= 1'b1;
                        q_slot_valid <= '0;
                        loader_select <= LOAD_K;
                        loader_head <= head_index + 6'd1;
                        loader_token_base <= '0;
                        loader_token_count <= sequence_length >= 12'd256 ? 12'd256 : sequence_length;
                        loader_channel_tile <= '0;
                        attention_read_source <= hardware_types_pkg::ATTENTION_READ_Q_MATMUL;
                        state <= WAIT_Q0;
                    end else begin
                        state <= WAIT_FINAL_WRITES;
                    end
                end
                WAIT_NEXT_HEAD: if (output_query_count == 0 && !capture_active &&
                    q_slot_valid[~head_index[0]] && !loader_start_pending &&
                    !loader_inflight && !loaded_valid) begin
                    loader_start_pending <= 1'b1;
                    loader_select <= LOAD_K;
                    loader_head <= head_index + 6'd1;
                    loader_token_base <= '0;
                    loader_token_count <= sequence_length >= 12'd256 ? 12'd256 : sequence_length;
                    loader_channel_tile <= '0;
                    loader_target_panel <= !active_panel;
                    loader_target_quarter <= 1'b0;
                end else if (loaded_valid && loaded_select == LOAD_K &&
                    loaded_head == head_index + 6'd1 && loaded_token_base == 0 &&
                    q_slot_valid[~head_index[0]] && (!output_subset_active || !capture_active)) begin
                    head_index <= head_index + 6'd1;
                    k_panel_base <= 12'd0;
                    k_panel_count <= loaded_token_count;
                    v_stripe <= 5'd0;
                    active_panel <= loaded_target_panel;
                    active_quarter <= 1'b0;
                    loaded_valid <= 1'b0;
                    v_slot_valid <= '0;
                    next_v_load_stripe <= '0;
                    v_prefetch_started <= 1'b0;
                    state <= START_QK;
                end
                WAIT_FINAL_WRITES: if (children_drained)
                    state <= COMPLETE;
                COMPLETE: if (done_ready) begin
                    attention_read_source <=
                        hardware_types_pkg::ATTENTION_READ_NONE;
                    state <= IDLE;
                end
                ERROR_DRAIN: if (children_drained)
                    state <= COMPLETE;
                ABORT_DRAIN: if (children_drained) begin
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase

            if (child_error_event && state != ABORT_DRAIN) begin
                terminal_error <= 1'b1;
                terminal_error_id <= child_error_id;
                loader_start_pending <= 1'b0;
                loaded_valid <= 1'b0;
                v_slot_valid <= '0;
                v_prefetch_started <= 1'b0;
                q_head_start_pending <= 1'b0;
                q_head_postprocess_allowed <= 1'b0;
                attention_read_source <= hardware_types_pkg::ATTENTION_READ_NONE;
                q_slot_valid <= '0;
                result_pending <= 1'b0;
                matmul_done_seen <= 1'b0;
                state <= ERROR_DRAIN;
            end

`ifndef SYNTHESIS
            if (loader_inflight && matmul_busy)
                cache_compute_overlap_cycle_count <=
                    cache_compute_overlap_cycle_count + 64'd1;
            if (writer_busy && (matmul_busy || softmax_busy))
                context_compute_overlap_cycle_count <=
                    context_compute_overlap_cycle_count + 64'd1;
`endif

            if (abort_request && state != IDLE && state != ABORT_DRAIN) begin
                loader_start_pending <= 1'b0;
                q_head_postprocess_allowed <= 1'b0;
                attention_read_source <= hardware_types_pkg::ATTENTION_READ_NONE;
                loaded_valid <= 1'b0;
                v_slot_valid <= '0;
                v_prefetch_started <= 1'b0;
                result_pending <= 1'b0;
                matmul_done_seen <= 1'b0;
                state <= ABORT_DRAIN;
            end
        end
    end

    initial begin
        if (ADDR_WIDTH != 64 || TAG_WIDTH < 16)
            $error("attention_controller requires 64-bit addresses and TAG_WIDTH >= 16");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(score_write_valid && context_buffer_write_valid))
                else $error("attention controller selected score and context writes together");
            assert (!(loader_start_pending && loader_inflight))
                else $error("attention controller duplicated a cache-loader task");
            if (matmul_start_fire && state == START_PV && v_stripe == 0)
                assert (context_buffer_available || (pair_active && active_second_batch))
                    else $error("attention controller started a head before context bank release");
            if (state == RUN_PV)
                assert (!softmax_busy)
                    else $error("attention controller ran Softmax and PV compute together");
            assert (context_max_request_fire ==
                    (result_write_fire && context_max_enable))
                else $error("attention context write and row-max request were not accepted atomically");
            assert (context_max_response_count <= context_max_request_count)
                else $error("attention controller accepted an unmatched context max response");
            if (done_valid && !error && context_max_enable) begin
                assert ((context_row_max_valid &
                         ((48'd1 << output_query_count) - 48'd1)) ==
                        ((48'd1 << output_query_count) - 48'd1))
                    else $error("attention completed without a max value for every query token");
                assert (context_max_request_count ==
                        32'(head_count) * 32'(output_query_count) * 32'd16 &&
                        context_max_response_count == context_max_request_count)
                    else $error("attention context max request/response count is incomplete");
            end
            if (softmax_score_memory_rsp_valid) begin
                assert (softmax_score_read_pending[1])
                    else $error("attention controller received a score response without matching metadata");
                assert (softmax_score_fifo_input_ready)
                    else $error("attention score SRAM response buffer had no reserved slot");
            end
            if (softmax_score_read_pending[1])
                assert (softmax_score_memory_rsp_valid)
                    else $error("attention score response missed the fixed SRAM latency");
            if (softmax_probability_write_valid)
                assert (softmax_engine_probability_write_head == 6'd0)
                    else $error("attention single-head Softmax returned a nonzero BF16 write head");
            if (softmax_quantized_write_valid)
                assert (softmax_engine_quantized_write_head == 6'd0)
                    else $error("attention single-head Softmax returned a nonzero quantized write head");
            if (capture_softmax_p8 && capture_stream_valid && capture_stream_ready) begin
                assert (capture_softmax_scale_row == softmax_quantized_write_row_base)
                    else $error("attention P8 export scale row does not match quantized row");
                assert (capture_stream_key == softmax_quantized_write_key)
                    else $error("attention P8 export key does not match quantized key");
            end
            if (softmax_scale_write_valid)
                assert (softmax_engine_scale_write_head == 6'd0)
                    else $error("attention single-head Softmax returned a nonzero scale write head");
            if (cache_done_pulse)
                assert (loader_inflight)
                    else $error("Attention cache completion pulse had no accepted loader task");
            if (cache_start_fire && loader_select == LOAD_V)
                assert (!v_slot_valid[{loader_target_panel,
                            loader_target_quarter}] &&
                        !((state == START_PV || state == RUN_PV) &&
                          {loader_target_panel, loader_target_quarter} ==
                              {active_panel, active_quarter}))
                    else $error("attention V load targeted an occupied panel quarter");
            if (cache_start_fire && loader_select == LOAD_K) begin
                assert (!loader_target_quarter)
                    else $error("attention K load selected a panel quarter");
                if (v_prefetch_started)
                    assert (v_slot_valid[{loader_target_panel, 1'b0}] == 1'b0 &&
                            v_slot_valid[{loader_target_panel, 1'b1}] == 1'b0 &&
                            !((state == START_PV || state == RUN_PV) &&
                              loader_target_panel == active_panel))
                        else $error("attention K load targeted a half with live V data");
            end
            if (matmul_done_pulse)
                assert (state == RUN_QK || state == RUN_PV ||
                        state == ERROR_DRAIN)
                    else $error("Attention Matmul completion pulse had no active compute task");
            if (softmax_done_pulse)
                assert (state == RUN_SOFTMAX || state == ERROR_DRAIN)
                    else $error("Attention Softmax completion pulse had no active Softmax task");
            if (writer_done_pulse)
                assert (writer_inflight)
                    else $error("Attention context completion pulse had no accepted writer task");
            if ($past(done_valid && !done_ready))
                assert (done_valid && $stable({error, error_id}))
                    else $error("attention controller changed a stalled completion response");
        end
    end
`endif
endmodule

`default_nettype wire
