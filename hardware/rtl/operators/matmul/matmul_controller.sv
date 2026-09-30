`default_nettype none

// Controls mixed-precision Matmul commands, activation preparation, weight
// panels and output writes. PE and rescale requests use shared resources
// selected by the current serial compute phase.
module matmul_controller #(
    parameter bit FIXED_SCHEDULE_INGRESS = 1'b0,
    parameter bit TRUSTED_FIXED_CONFIGURATION = 1'b0,
    parameter integer ATTENTION_HEAD_COUNT = 32,
    parameter integer HIDDEN_FEATURES = 4096,
    parameter integer FFN_FEATURES = 12288
) (
    input logic clk, input logic rst, input logic abort_request,
    output logic abort_ack,
    input logic start_valid, output logic start_ready,
    input logic [15:0] start_operator_tag,
    input logic [5:0] start_rows, input logic [15:0] start_input_features,
    input logic [15:0] start_output_features, input logic [1:0] start_segment_count,
    input logic [5:0] start_segment_mode, input logic [17:0] start_segment_row_base,
    input logic [17:0] start_segment_row_count,
    input logic [95:0] start_segment_activation_base_byte_offset,
    input logic [47:0] start_row_enable,
    input logic start_source_ddr,
    input logic [63:0] start_source_base_addr,
    input logic start_use_resident_activation, input logic start_qkv_w8_layout,
    input logic [2:0] start_output_mode,
    input logic start_operator_has_w8,
    input logic [63:0] start_weight_base_addr,
    input logic [63:0] start_weight_enhancement_addr,
    input logic [63:0] start_weight_scale_addr,
    input logic [63:0] start_output_base, input logic [63:0] start_output_limit,
    input logic [31:0] start_output_row_stride,

    input hardware_types_pkg::matmul_command_t fixed_command,
    input logic [5:0] fixed_command_rows,
    input logic [5:0] fixed_command_residual_storage_rows,
    input hardware_types_pkg::matmul_row_config_t fixed_command_row_config,
    input logic [1:0] fixed_command_qkv_select,
    input logic [5:0] fixed_command_qkv_head,
    input logic fixed_command_qkv_use_resident_activation,
    input logic fixed_command_qkv_w8_layout,
    input hardware_types_pkg::matmul_weight_config_t fixed_weight_config,
    input hardware_types_pkg::attention_context_config_t
        fixed_attention_context_config,
    input logic [47:0] fixed_attention_context_max_valid,
    input logic fixed_attention_context_precomputed_max_disable,
    input logic [48*16-1:0] fixed_attention_context_max_values,
    input hardware_types_pkg::ffn_workspace_config_t
        fixed_ffn_workspace_config,
    input hardware_types_pkg::ffn_batch_pair_config_t fixed_ffn_batch_pair,

    input logic prefetch_valid, output logic prefetch_ready,
    output logic prefetch_empty,
    input logic prefetch_cancel,
    input logic [15:0] prefetch_operator_tag,
    input logic [15:0] prefetch_input_features,
    input logic [63:0] prefetch_weight_base_addr,
    input logic [63:0] prefetch_weight_enhancement_addr,
    input logic [63:0] prefetch_weight_scale_addr,
    input logic prefetch_operator_has_w8,
    input logic prefetch_qkv_w8_layout,
    input hardware_types_pkg::matmul_command_t fixed_prefetch_command,
    input logic [1:0] fixed_prefetch_qkv_select,
    input logic [5:0] fixed_prefetch_qkv_head,

    output logic source_req_valid, input logic source_req_ready,
    output logic [5:0] source_req_row,
    output logic [3:0] source_req_rows,
    output logic [15:0] source_req_element,
    output logic [4:0] source_req_bank_base,
    input logic source_rsp_valid, output logic source_rsp_ready,
    input logic [1023:0] source_rsp_values,
    input logic [63:0] source_rsp_lane_mask,

    output logic aux_read_request_valid,
    input logic aux_read_request_ready,
    output logic [63:0] aux_read_request_address,
    output logic [31:0] aux_read_request_bytes,
    output logic [7:0] aux_read_request_tag,
    output logic aux_read_second_span_valid,
    output logic [63:0] aux_read_second_span_address,
    output logic [31:0] aux_read_pair_stride,
    output logic [10:0] aux_read_pair_count,
    input logic aux_read_request_error,
    input logic aux_read_data_valid,
    output logic aux_read_data_ready,
    input logic [127:0] aux_read_data,
    input logic [15:0] aux_read_byte_enable,
    input logic aux_read_data_last,
    input logic [7:0] aux_read_data_tag,

    output logic output_dma_request_valid,
    input logic output_dma_request_ready,
    output logic [63:0] output_dma_request_address,
    output logic [31:0] output_dma_request_bytes,
    output logic [7:0] output_dma_request_tag,
    input logic output_dma_request_done,
    input logic output_dma_request_error,
    output logic output_dma_data_valid,
    input logic output_dma_data_ready,
    output logic [127:0] output_dma_data,
    output logic [15:0] output_dma_byte_enable,
    output logic output_dma_data_last,

    output logic residual_local_read_valid,
    input logic residual_local_read_ready,
    output logic [5:0] residual_local_read_row,
    output logic [8:0] residual_local_read_channel_chunk,
    input logic residual_local_read_rsp_valid,
    input logic [127:0] residual_local_read_rsp_data,

    output logic dma_req_valid, input logic dma_req_ready,
    output logic [63:0] dma_req_addr, output logic [31:0] dma_req_bytes,
    output logic [1:0] dma_req_plane, output logic dma_req_panel,
    output logic dma_req_last_for_plane,
    input logic dma_req_done, input logic dma_req_error,
    input logic dma_rsp_valid, output logic dma_rsp_ready,
    input logic [255:0] dma_rsp_data, input logic [31:0] dma_rsp_byte_enable,
    input logic dma_rsp_error, input logic dma_rsp_last,
    input logic dma_abort_ack,


    output logic activation_writer_abort_request,
    output logic activation_writer_cfg_valid,
    input logic activation_writer_cfg_ready,
    output logic [1:0] activation_writer_cfg_mode,
    output logic [5:0] activation_writer_cfg_physical_row_base,
    output logic [5:0] activation_writer_cfg_row_count,
    output logic [15:0] activation_writer_cfg_elements_per_row,
    output logic [31:0] activation_writer_cfg_activation_base_byte_offset,
    output logic [31:0] activation_writer_cfg_activation_limit_byte_offset,
    output logic activation_writer_cfg_chunk_major,
    output logic activation_writer_cfg_row_partial,
    output logic [1:0] activation_writer_cfg_segment_count,
    output logic [5:0] activation_writer_cfg_segment_mode,
    output logic [17:0] activation_writer_cfg_segment_row_base,
    output logic [17:0] activation_writer_cfg_segment_row_count,
    output logic activation_writer_cfg_mixed_group_enable,
    output logic [5:0] activation_writer_cfg_compute_group_count,
    output logic [47:0] activation_writer_cfg_row_precision_a8,
    output logic [287:0] activation_writer_cfg_row_compute_group,
    output logic [143:0] activation_writer_cfg_row_pe_slot,
    output logic [95:0] activation_writer_cfg_row_phase_mask,
    output logic activation_writer_quantized_valid,
    input logic activation_writer_quantized_ready,
    output logic [5:0] activation_writer_quantized_row,
    output logic [15:0] activation_writer_quantized_element,
    output logic [511:0] activation_writer_quantized_values,
    output logic [63:0] activation_writer_quantized_lane_mask,
    output logic [15:0] activation_writer_quantized_scale,
    output logic activation_writer_scale_valid,
    input logic activation_writer_scale_ready,
    output logic [5:0] activation_writer_scale_row_base,
    output logic [7:0] activation_writer_scale_row_mask,
    output logic [127:0] activation_writer_scale_values,
    input logic activation_writer_done_pulse,
    input logic activation_writer_error,
    input logic activation_scale_valid,
    output logic activation_scale_ready,
    input logic [5:0] activation_scale_row_base,
    input logic [7:0] activation_scale_row_mask,
    input logic [127:0] activation_scale_values,
    input logic resident_activation_commit_valid,
    input logic [5:0] resident_activation_commit_rows,
    input logic [15:0] resident_activation_commit_features,
    input logic [1:0] resident_activation_commit_segment_count,
    input logic [5:0] resident_activation_commit_segment_mode,
    input logic [17:0] resident_activation_commit_segment_row_base,
    input logic [17:0] resident_activation_commit_segment_row_count,
    input logic [95:0] resident_activation_commit_segment_activation_base_byte_offset,
    input logic [47:0] resident_activation_commit_row_enable,
    output logic panel_write_accepted, output logic panel_write_panel,
    output logic [1:0] panel_write_plane,
    output logic [2:0] panel_write_start_bank,
    output logic [9:0] panel_write_word_row,
    output logic [255:0] panel_write_data,
    output logic [31:0] panel_write_byte_enable,
    output logic accum_result_valid, input logic accum_result_ready,
    output logic [2047:0] accum_result_accumulators,
    output logic [63:0] accum_result_mask, output logic [1:0] accum_result_mode,
    output logic accum_result_last_k_step, output logic [15:0] accum_result_tag,
    output logic output_write_spill,
    output logic [5:0] output_write_physical_row,

    output logic shared_bf16_abort_request,
    input logic shared_bf16_abort_ack,
    output logic shared_bf16_req_valid,
    input logic shared_bf16_req_ready,
    output logic [2:0] shared_bf16_req_operation,
    output logic [255:0] shared_bf16_req_values,
    output logic [255:0] shared_bf16_req_paired_values,
    output logic [255:0] shared_bf16_req_factor0_values,
    output logic [255:0] shared_bf16_req_factor1_values,
    output logic [15:0] shared_bf16_req_lane_mask,
    output logic [15:0] shared_bf16_req_tag,
    input logic shared_bf16_rsp_valid,
    output logic shared_bf16_rsp_ready,
    input logic [255:0] shared_bf16_rsp_values,
    input logic [15:0] shared_bf16_rsp_lane_mask,
    input logic [15:0] shared_bf16_rsp_tag,
    output logic done_valid, input logic done_ready,
    output logic error,
    output logic [63:0] accepted_issue_count,
    output logic [63:0] accepted_result_count,
    output logic [63:0] accepted_base_request_count,
    output logic [63:0] accepted_enhancement_request_count,
    output logic [63:0] accepted_scale_request_count,
    output logic [63:0] accepted_base_request_bytes,
    output logic [63:0] accepted_enhancement_request_bytes,
    output logic [63:0] accepted_scale_request_bytes,
    output logic [1:0] memory_stage,

    output logic shared_tile_abort_request,
    input logic shared_tile_abort_ack,
    output logic shared_tile_req_valid, input logic shared_tile_req_ready,
    output logic [1:0] shared_tile_req_mode,
    output logic [1023:0] shared_tile_req_activation_payload,
    output logic [1023:0] shared_tile_req_weight_payload,
    output logic [7:0] shared_tile_req_row_mask,
    output logic [31:0] shared_tile_req_k_mask,
    output logic [7:0] shared_tile_req_col_mask,
    output logic shared_tile_req_first_k_step,
    output logic shared_tile_req_last_k_step,
    output logic shared_tile_req_mixed_phase,
    output logic shared_tile_req_mixed_phase_first,
    output logic [7:0] shared_tile_req_mixed_a8_rows,
    output logic [127:0] shared_tile_req_activation_scales,
    output logic [127:0] shared_tile_req_weight_scales,
    output logic [15:0] shared_tile_req_tag,
    input logic shared_tile_accum_result_valid,
    output logic shared_tile_accum_result_ready,
    input logic [2047:0] shared_tile_accum_result_accumulators,
    input logic [127:0] shared_tile_accum_result_activation_scales,
    input logic [127:0] shared_tile_accum_result_weight_scales,
    input logic [63:0] shared_tile_accum_result_mask,
    input logic [1:0] shared_tile_accum_result_mode,
    input logic shared_tile_accum_result_last_k_step,
    input logic [15:0] shared_tile_accum_result_tag,

    output logic shared_rescale_claim_valid,
    input logic shared_rescale_claim_ready,
    output logic shared_rescale_abort_request,
    input logic shared_rescale_abort_ack,
    output logic shared_rescale_req_valid,
    input logic shared_rescale_req_ready,
    output logic [2047:0] shared_rescale_req_accumulators,
    output logic [127:0] shared_rescale_req_activation_scales,
    output logic [127:0] shared_rescale_req_weight_scales,
    output logic [15:0] shared_rescale_req_qk_scale_bf16,
    output logic [1:0] shared_rescale_req_rescale_mode,
    output logic [63:0] shared_rescale_req_lane_mask,
    output logic [15:0] shared_rescale_req_tag,
    input logic shared_rescale_rsp_valid,
    output logic shared_rescale_rsp_ready,
    input logic [1023:0] shared_rescale_rsp_values,
    input logic [63:0] shared_rescale_rsp_lane_mask,
    input logic [15:0] shared_rescale_rsp_tag,

    output logic max_req_valid, input logic max_req_ready,
    output logic [1023:0] max_req_values,
    output logic [63:0] max_req_lane_mask,
    output logic [15:0] max_req_tag,
    input logic max_rsp_valid, output logic max_rsp_ready,
    input logic [127:0] max_rsp_values,
    input logic [7:0] max_rsp_row_mask,
    input logic [15:0] max_rsp_tag,
    output logic quant_scale_req_valid,
    input logic quant_scale_req_ready,
    output logic [7:0] quant_scale_req_a4_row_mask,
    output logic [15:0] quant_scale_req_clip_ratio_bf16,
    output logic [127:0] quant_scale_req_row_max_abs,
    output logic [7:0] quant_scale_req_row_mask,
    input logic quant_scale_rsp_valid,
    output logic quant_scale_rsp_ready,
    input logic [127:0] quant_scale_rsp_values,
    output logic quant_values_req_valid,
    input logic quant_values_req_ready,
    output logic [1023:0] quant_values_req_values,
    output logic [63:0] quant_values_req_lane_mask,
    output logic [15:0] quant_values_req_tag,
    input logic quant_values_rsp_valid,
    output logic quant_values_rsp_ready,
    input logic [511:0] quant_values_rsp_values,
    input logic [63:0] quant_values_rsp_lane_mask,
    input logic [15:0] quant_values_rsp_tag,

    output logic panel_memory_write_valid, input logic panel_memory_write_ready,
    output logic panel_memory_write_panel, output logic [1:0] panel_memory_write_plane,
    output logic [2:0] panel_memory_write_start_bank,
    output logic [9:0] panel_memory_write_word_row,
    output logic [255:0] panel_memory_write_data,
    output logic [31:0] panel_memory_write_byte_enable,
    output logic read_bundle_valid, input logic read_bundle_ready,
    output logic [15:0] activation_read_word_index,
    output logic [3:0] activation_read_word_count,
    output logic activation_read_qkv_padded_layout,
    output logic [5:0] activation_read_physical_row_base,
    output logic activation_read_explicit_rows,
    output logic [47:0] activation_read_physical_rows,
    output logic [127:0] activation_read_lane_word_index,
    output logic panel_read_select,
    output logic [11:0] panel_read_port_mask, output logic [119:0] panel_read_address,
    input logic read_response_valid, input logic [1023:0] activation_read_data,
    input logic [1535:0] panel_read_data,
    output logic combine_write_valid, input logic combine_write_ready,
    output logic [9:0] combine_write_address, output logic [127:0] combine_write_data,
    output logic [15:0] combine_write_byte_enable,
    output logic combine_read_valid, input logic combine_read_ready,
    output logic [9:0] combine_read_address,
    input logic combine_read_response_valid, input logic [127:0] combine_read_data,
    output logic local_output_write_valid, input logic local_output_write_ready,
    output logic [31:0] local_output_write_address,
    output logic [127:0] local_output_write_data,
    output logic [15:0] local_output_write_byte_enable
);
    import hardware_types_pkg::attention_context_config_t;
    import hardware_types_pkg::ffn_workspace_config_t;
    import hardware_types_pkg::matmul_command_t;
    import hardware_types_pkg::matmul_row_config_t;
    import hardware_types_pkg::matmul_weight_config_t;
    import hardware_types_pkg::MATMUL_COMMAND_ATTENTION_OUTPUT;
    import hardware_types_pkg::MATMUL_COMMAND_FFN_DOWN;
    import hardware_types_pkg::MATMUL_COMMAND_FFN_GATE;
    import hardware_types_pkg::MATMUL_COMMAND_FFN_UP;
    import hardware_types_pkg::MATMUL_COMMAND_QKV;
    localparam logic [1:0] MODE_W4A4=0, MODE_W4A8=1, MODE_W8A8=2;
    localparam logic [1:0] MEMORY_STAGE_NONE = 2'd0;
    localparam logic [1:0] MEMORY_STAGE_WEIGHT = 2'd1;
    localparam logic [1:0] MEMORY_STAGE_AUXILIARY = 2'd2;
    typedef enum logic [3:0] {IDLE, CONFIG_DERIVE, CONFIG_RANGE, CONFIG_CHECK,
        CONFIG_DISPATCH, SEG_CFG, ROW_START, ROW_WAIT, SEG_WAIT, OUTPUT_CFG,
        COMPUTE_START, COMPUTE, FINISH, ABORT_WAIT, ERROR_WAIT} state_t;
    state_t state;

    logic [5:0] command_rows;
    logic [15:0] command_input_features;
    logic [15:0] command_output_features;
    logic [1:0] command_segment_count;
    logic [5:0] command_segment_mode;
    logic [17:0] command_segment_row_base;
    logic [17:0] command_segment_row_count;
    logic [95:0] command_segment_activation_base_byte_offset;
    logic [47:0] command_row_enable;
    logic command_mixed_group_enable;
    logic [5:0] command_compute_group_count;
    logic [47:0] command_row_precision_a8;
    logic [287:0] command_row_compute_group;
    logic [143:0] command_row_pe_slot;
    logic [95:0] command_row_phase_mask;
    logic command_source_ddr;
    logic [63:0] command_source_base;
    logic command_use_resident_activation;
    logic command_use_precomputed_max;
    logic command_attention_pair_query;
    logic command_kv_batch_pair;
    logic command_qkvo_group;
    assign command_attention_pair_query = FIXED_SCHEDULE_INGRESS &&
        fixed_command == hardware_types_pkg::MATMUL_COMMAND_QKV &&
        fixed_command_qkv_select == 2'd0 && fixed_ffn_batch_pair.enable &&
        !fixed_ffn_batch_pair.qkvo_group;
    assign command_kv_batch_pair = FIXED_SCHEDULE_INGRESS &&
        fixed_command == hardware_types_pkg::MATMUL_COMMAND_QKV &&
        fixed_command_qkv_select != 2'd0 && fixed_ffn_batch_pair.enable;
    assign command_qkvo_group = FIXED_SCHEDULE_INGRESS &&
        fixed_command == hardware_types_pkg::MATMUL_COMMAND_QKV &&
        fixed_ffn_batch_pair.qkvo_group;
    logic [15:0] command_clip_ratio_bf16;
    logic command_qkv_w8_layout;
    logic command_fused_product;
    logic [2:0] command_output_mode;
    logic command_operator_has_w8;
    logic [63:0] command_weight_base;
    logic [63:0] command_weight_enhancement;
    logic [63:0] command_weight_scale;
    logic [63:0] command_output_base;
    logic [63:0] command_output_limit;
    logic [31:0] command_output_row_stride;
    logic [5:0] command_residual_storage_rows;
    logic [15:0] command_prefetch_input_features;
    logic [63:0] command_prefetch_weight_base;
    logic [63:0] command_prefetch_weight_enhancement;
    logic [63:0] command_prefetch_weight_scale;
    logic command_prefetch_operator_has_w8;
    logic command_prefetch_qkv_w8_layout;

    always_comb begin : fixed_schedule_command_derive
        integer compact_segment;
        integer next_activation_base_byte_offset;
        integer segment_row_bytes;

        command_rows = start_rows;
        command_input_features = start_input_features;
        command_output_features = start_output_features;
        command_segment_count = start_segment_count;
        command_segment_mode = start_segment_mode;
        command_segment_row_base = start_segment_row_base;
        command_segment_row_count = start_segment_row_count;
        command_segment_activation_base_byte_offset = start_segment_activation_base_byte_offset;
        command_row_enable = start_row_enable;
        command_mixed_group_enable = 1'b0;
        command_compute_group_count = 6'd0;
        command_row_precision_a8 = 48'd0;
        command_row_compute_group = 288'd0;
        command_row_pe_slot = 144'd0;
        command_row_phase_mask = 96'd0;
        command_source_ddr = start_source_ddr;
        command_source_base = start_source_base_addr;
        command_use_resident_activation = start_use_resident_activation;
        command_use_precomputed_max = 1'b0;
        command_clip_ratio_bf16 = 16'd0;
        command_qkv_w8_layout = start_qkv_w8_layout;
        command_fused_product = 1'b0;
        command_output_mode = start_output_mode;
        command_operator_has_w8 = start_operator_has_w8;
        command_weight_base = start_weight_base_addr;
        command_weight_enhancement = start_weight_enhancement_addr;
        command_weight_scale = start_weight_scale_addr;
        command_output_base = start_output_base;
        command_output_limit = start_output_limit;
        command_output_row_stride = start_output_row_stride;
        command_residual_storage_rows = 6'd0;

        command_prefetch_input_features = prefetch_input_features;
        command_prefetch_weight_base = prefetch_weight_base_addr;
        command_prefetch_weight_enhancement =
            prefetch_weight_enhancement_addr;
        command_prefetch_weight_scale = prefetch_weight_scale_addr;
        command_prefetch_operator_has_w8 = prefetch_operator_has_w8;
        command_prefetch_qkv_w8_layout = prefetch_qkv_w8_layout;

        if (FIXED_SCHEDULE_INGRESS) begin
            command_clip_ratio_bf16 = fixed_weight_config.query_clip_ratio_bf16;
            command_rows = fixed_command_rows;
            command_input_features = 16'd4096;
            command_output_features = 16'd128;
            command_segment_count = 2'd0;
            command_segment_mode = 6'd0;
            command_segment_row_base = 18'd0;
            command_segment_row_count = 18'd0;
            command_segment_activation_base_byte_offset = 96'd0;
            command_row_enable = fixed_command_row_config.row_enable;
            command_mixed_group_enable =
                fixed_command_row_config.mixed_group_enable;
            command_compute_group_count =
                fixed_command_row_config.compute_group_count;
            command_row_precision_a8 =
                fixed_command_row_config.row_precision_a8;
            command_row_compute_group =
                fixed_command_row_config.row_compute_group;
            command_row_pe_slot = fixed_command_row_config.row_pe_slot;
            command_row_phase_mask = fixed_command_row_config.row_phase_mask;
            command_source_ddr = 1'b0;
            command_source_base = 64'd0;
            if (command_attention_pair_query || command_kv_batch_pair ||
                command_qkvo_group) begin
                command_source_ddr = 1'b1;
                command_source_base = fixed_ffn_batch_pair.normalized_base;
            end
            command_use_resident_activation =
                fixed_command_qkv_use_resident_activation;
            command_qkv_w8_layout = 1'b0;
            command_fused_product = fixed_ffn_batch_pair.fused_product &&
                fixed_command == MATMUL_COMMAND_FFN_GATE;
            command_output_mode = command_qkvo_group ? 3'd6 : 3'd0;
            command_operator_has_w8 = 1'b0;
            command_output_base = command_qkvo_group ?
                fixed_ffn_batch_pair.qkv_staging_base : 64'd0;
            command_output_limit = command_qkvo_group ?
                fixed_ffn_batch_pair.qkv_staging_limit :
                (command_kv_batch_pair ? 64'd24576 : 64'd12288);
            command_output_row_stride = 32'd256;

            case (fixed_command_qkv_select)
                2'd0: begin
                    command_weight_base = fixed_weight_config.query_base;
                    command_weight_enhancement =
                        fixed_weight_config.query_enhancement;
                    command_weight_scale = fixed_weight_config.query_scale;
                end
                2'd1: begin
                    command_weight_base = fixed_weight_config.key_base;
                    command_weight_enhancement =
                        fixed_weight_config.key_enhancement;
                    command_weight_scale = fixed_weight_config.key_scale;
                end
                default: begin
                    command_weight_base = fixed_weight_config.value_base;
                    command_weight_enhancement =
                        fixed_weight_config.value_enhancement;
                    command_weight_scale = fixed_weight_config.value_scale;
                end
            endcase
            command_weight_base = command_weight_base +
                64'(fixed_command_qkv_head) * 64'd262144;
            command_weight_enhancement = command_weight_enhancement +
                64'(fixed_command_qkv_head) * 64'd262144;
            command_weight_scale = command_weight_scale +
                64'(fixed_command_qkv_head) * 64'd256;

            if (fixed_command == MATMUL_COMMAND_ATTENTION_OUTPUT) begin
                command_residual_storage_rows =
                    fixed_command_residual_storage_rows;
                command_clip_ratio_bf16 = fixed_weight_config.attention_output_clip_ratio_bf16;
                command_input_features = 16'(ATTENTION_HEAD_COUNT * 128);
                command_output_features = 16'(HIDDEN_FEATURES);
                command_source_ddr = 1'b1;
                command_source_base = fixed_attention_context_config.context_base;
                command_use_resident_activation = 1'b0;
                command_use_precomputed_max =
                    fixed_command_row_config.mixed_group_enable &&
                    !fixed_attention_context_precomputed_max_disable &&
                    !fixed_ffn_batch_pair.enable;
                command_qkv_w8_layout = 1'b0;
                command_output_mode = fixed_ffn_batch_pair.enable ? 3'd4 : 3'd3;
                command_operator_has_w8 =
                    fixed_command_row_config.operator_has_w8;
                command_output_base = fixed_ffn_batch_pair.enable ?
                    fixed_ffn_workspace_config.residual_base : 64'd0;
                command_output_limit = fixed_ffn_batch_pair.enable ?
                    fixed_ffn_workspace_config.residual_limit : 64'd393216;
                command_output_row_stride = 32'd8192;
                command_weight_base =
                    fixed_weight_config.attention_output_base;
                command_weight_enhancement =
                    fixed_weight_config.attention_output_enhancement;
                command_weight_scale =
                    fixed_weight_config.attention_output_scale;
            end else if (fixed_command != MATMUL_COMMAND_QKV) begin
                command_input_features =
                    fixed_command == MATMUL_COMMAND_FFN_DOWN ?
                    16'(FFN_FEATURES) : 16'd4096;
                command_output_features =
                    fixed_command == MATMUL_COMMAND_FFN_DOWN ?
                    16'(HIDDEN_FEATURES) : 16'(FFN_FEATURES);
                command_source_ddr = 1'b0;
                command_source_base = 64'd0;
                if (fixed_ffn_batch_pair.enable &&
                    (fixed_command == MATMUL_COMMAND_FFN_GATE || fixed_command == MATMUL_COMMAND_FFN_UP)) begin
                    command_source_ddr = fixed_command == MATMUL_COMMAND_FFN_GATE;
                    command_source_base = fixed_ffn_batch_pair.normalized_base;
                end
                command_use_resident_activation =
                    fixed_command != MATMUL_COMMAND_FFN_GATE;
                command_qkv_w8_layout = 1'b0;
                command_output_mode =
                    fixed_command == MATMUL_COMMAND_FFN_GATE &&
                        command_fused_product ? 3'd5 :
                    fixed_command == MATMUL_COMMAND_FFN_GATE ? 3'd1 :
                    fixed_command == MATMUL_COMMAND_FFN_UP ? 3'd2 : 3'd4;
                command_operator_has_w8 =
                    fixed_command_row_config.operator_has_w8;
                command_output_base =
                    fixed_command == MATMUL_COMMAND_FFN_DOWN ?
                    fixed_ffn_workspace_config.residual_base :
                    fixed_ffn_workspace_config.gate_up_base;
                command_output_limit =
                    fixed_command == MATMUL_COMMAND_FFN_DOWN ?
                    fixed_ffn_workspace_config.residual_limit :
                    fixed_ffn_workspace_config.gate_up_limit;
                command_output_row_stride = 32'(HIDDEN_FEATURES * 2);
                if (fixed_command == MATMUL_COMMAND_FFN_GATE) begin
                    command_clip_ratio_bf16 = fixed_weight_config.ffn_gate_clip_ratio_bf16;
                    command_weight_base = fixed_weight_config.ffn_gate_base;
                    command_weight_enhancement =
                        fixed_weight_config.ffn_gate_enhancement;
                    command_weight_scale = fixed_weight_config.ffn_gate_scale;
                end else if (fixed_command == MATMUL_COMMAND_FFN_UP) begin
                    command_clip_ratio_bf16 = fixed_weight_config.ffn_up_clip_ratio_bf16;
                    command_weight_base = fixed_weight_config.ffn_up_base;
                    command_weight_enhancement =
                        fixed_weight_config.ffn_up_enhancement;
                    command_weight_scale = fixed_weight_config.ffn_up_scale;
                end else begin
                    command_clip_ratio_bf16 = fixed_weight_config.ffn_down_clip_ratio_bf16;
                    command_weight_base = fixed_weight_config.ffn_down_base;
                    command_weight_enhancement =
                        fixed_weight_config.ffn_down_enhancement;
                    command_weight_scale = fixed_weight_config.ffn_down_scale;
                end
            end

            compact_segment = 0;
            next_activation_base_byte_offset = command_attention_pair_query &&
                fixed_ffn_batch_pair.prepare_second ? 98304 : 0;
            command_segment_activation_base_byte_offset[31:0] = 32'(next_activation_base_byte_offset);
            for (integer source_segment = 0; source_segment < 3;
                 source_segment = source_segment + 1) begin
                if (fixed_command_row_config.segment_row_count[
                        source_segment*6 +: 6] != 0) begin
                    command_segment_mode[compact_segment*2 +: 2] =
                        fixed_command_row_config.segment_mode[
                            source_segment*2 +: 2];
                    command_segment_row_base[compact_segment*6 +: 6] =
                        fixed_command_row_config.segment_row_base[
                            source_segment*6 +: 6];
                    command_segment_row_count[compact_segment*6 +: 6] =
                        fixed_command_row_config.segment_row_count[
                            source_segment*6 +: 6];
                    command_segment_activation_base_byte_offset[compact_segment*32 +: 32] =
                        32'(next_activation_base_byte_offset);
                    if (fixed_command == MATMUL_COMMAND_ATTENTION_OUTPUT)
                        segment_row_bytes =
                            fixed_command_row_config.segment_mode[
                                source_segment*2 +: 2] == 2'd0 ?
                            ATTENTION_HEAD_COUNT * 64 : ATTENTION_HEAD_COUNT * 128;
                    else
                        segment_row_bytes =
                            fixed_command_row_config.segment_mode[
                                source_segment*2 +: 2] == 2'd0 ?
                            integer'(command_input_features) / 2 :
                            integer'(command_input_features);
                    next_activation_base_byte_offset = next_activation_base_byte_offset +
                        integer'(fixed_command_row_config.segment_row_count[
                            source_segment*6 +: 6]) * segment_row_bytes;
                    compact_segment = compact_segment + 1;
                end
            end
            command_segment_count = 2'(compact_segment);

            command_prefetch_input_features =
                fixed_prefetch_command == MATMUL_COMMAND_FFN_DOWN ?
                16'(FFN_FEATURES) : 16'd4096;
            command_prefetch_operator_has_w8 =
                fixed_prefetch_command == MATMUL_COMMAND_QKV ? 1'b0 :
                fixed_command_row_config.operator_has_w8;
            command_prefetch_qkv_w8_layout = 1'b0;
            case (fixed_prefetch_command)
                MATMUL_COMMAND_FFN_GATE: begin
                    command_prefetch_weight_base =
                        fixed_weight_config.ffn_gate_base;
                    command_prefetch_weight_enhancement =
                        fixed_weight_config.ffn_gate_enhancement;
                    command_prefetch_weight_scale =
                        fixed_weight_config.ffn_gate_scale;
                end
                MATMUL_COMMAND_FFN_UP: begin
                    command_prefetch_weight_base = fixed_weight_config.ffn_up_base;
                    command_prefetch_weight_enhancement =
                        fixed_weight_config.ffn_up_enhancement;
                    command_prefetch_weight_scale = fixed_weight_config.ffn_up_scale;
                end
                MATMUL_COMMAND_FFN_DOWN: begin
                    command_prefetch_weight_base =
                        fixed_weight_config.ffn_down_base;
                    command_prefetch_weight_enhancement =
                        fixed_weight_config.ffn_down_enhancement;
                    command_prefetch_weight_scale =
                        fixed_weight_config.ffn_down_scale;
                end
                default: begin
                    case (fixed_prefetch_qkv_select)
                        2'd0: begin
                            command_prefetch_weight_base =
                                fixed_weight_config.query_base;
                            command_prefetch_weight_enhancement =
                                fixed_weight_config.query_enhancement;
                            command_prefetch_weight_scale =
                                fixed_weight_config.query_scale;
                        end
                        2'd1: begin
                            command_prefetch_weight_base =
                                fixed_weight_config.key_base;
                            command_prefetch_weight_enhancement =
                                fixed_weight_config.key_enhancement;
                            command_prefetch_weight_scale =
                                fixed_weight_config.key_scale;
                        end
                        default: begin
                            command_prefetch_weight_base =
                                fixed_weight_config.value_base;
                            command_prefetch_weight_enhancement =
                                fixed_weight_config.value_enhancement;
                            command_prefetch_weight_scale =
                                fixed_weight_config.value_scale;
                        end
                    endcase
                    command_prefetch_weight_base =
                        command_prefetch_weight_base +
                        64'(fixed_prefetch_qkv_head) * 64'd262144;
                    command_prefetch_weight_enhancement =
                        command_prefetch_weight_enhancement +
                        64'(fixed_prefetch_qkv_head) * 64'd262144;
                    command_prefetch_weight_scale =
                        command_prefetch_weight_scale +
                        64'(fixed_prefetch_qkv_head) * 64'd256;
                end
            endcase
        end
    end
    logic [5:0] rows; logic [15:0] k_size,n_size; logic [1:0] seg_count,seg_index;
    logic [5:0] seg_mode; logic [17:0] seg_base; logic [17:0] seg_rows;
    logic [95:0] seg_activation; logic [47:0] row_enable;
    logic mixed_groups;
    logic [5:0] mixed_compute_group_count;
    logic [47:0] mixed_row_precision_a8;
    logic [287:0] mixed_row_compute_group;
    logic [143:0] mixed_row_pe_slot;
    logic [95:0] mixed_row_phase_mask;
    logic source_ddr,has_w8,use_resident_activation,qkv_w8_layout;
    logic fused_product;
    logic qkv_query_command;
    logic attention_pair_query, attention_pair_query_second;
    logic kv_batch_pair;
    logic [1:0] attention_pair_query_prepared;
    logic attention_pair_resident_matches;
    logic activation_source_live;
    logic use_precomputed_max;
    logic [63:0] source_base;
    logic [2:0] output_mode;
    logic [63:0] weight_base,weight_enh,weight_scale;
    logic [63:0] output_base,output_limit;
    logic [31:0] output_stride;
    logic [5:0] residual_storage_rows;
    logic [5:0] current_row;
    logic [15:0] activation_scales[0:47];
    logic [15:0] second_activation_scales[0:47];
    logic [15:0] third_activation_scales[0:47];
    logic [15:0] fourth_activation_scales[0:47];
    logic [15:0] fifth_activation_scales[0:47];
    logic [15:0] sixth_activation_scales[0:47];
    wire hardware_types_pkg::ffn_batch_pair_config_t batch_pair =
        FIXED_SCHEDULE_INGRESS && fixed_ffn_batch_pair.enable &&
        (fixed_command == hardware_types_pkg::MATMUL_COMMAND_FFN_GATE ||
         fixed_command == hardware_types_pkg::MATMUL_COMMAND_FFN_UP ||
         fixed_command == hardware_types_pkg::MATMUL_COMMAND_FFN_DOWN ||
         fixed_command == hardware_types_pkg::MATMUL_COMMAND_ATTENTION_OUTPUT ||
         command_kv_batch_pair || command_qkvo_group) ?
            fixed_ffn_batch_pair : '0;
    hardware_types_pkg::ffn_batch_pair_config_t resident_batch_pair;
    logic expanded_activation_layout;
    logic [2:0] scale_write_batch_index;
    logic [2:0] scale_read_batch_index;
    logic [2:0] quantization_batch_index;
    logic [2:0] issue_batch_index, result_batch_index, fragment_batch_index;
    logic [6:0] total_activation_group_count;
    assign scale_read_batch_index = attention_pair_query && attention_pair_query_second ?
        3'd1 : issue_batch_index;
    assign expanded_activation_layout = batch_pair.enable &&
        batch_pair.compact_a4_layout &&
        (total_activation_group_count >
            (k_size == 16'd12288 ? 7'd4 : 7'd12) ||
         (qkv_query_command && batch_pair.qkvo_group &&
          total_activation_group_count > 7'd4));
    assign scale_write_batch_index = fixed_ffn_batch_pair.prepare_second ?
        3'd1 : quantization_batch_index;
    logic [5:0] quantization_rows;
    logic [7:0] quantization_source_row_offset;
    logic [8:0] total_output_rows;
    hardware_types_pkg::matmul_row_config_t quantization_row_config;
    logic [64:0] normalized_source_end;
    logic [3:0] current_batch_rows;
    logic [7:0] current_batch_a4_rows;
    logic [7:0] current_batch_row_mask;
    logic current_batch_precomputed_max_valid;
    logic [127:0] current_batch_precomputed_max;
    logic [7:0] current_batch_precomputed_valid_mask;
    logic frontend_start,frontend_ready,frontend_done_pulse,frontend_error;
    logic frontend_abort_ack;
    logic quantized_valid,quantized_ready; logic [5:0] quantized_row;
    logic [15:0] quantized_element;
    logic [511:0] quantized_values; logic [63:0] quantized_mask;
    logic frontend_source_req_valid;
    logic frontend_source_req_ready;
    logic [15:0] frontend_source_req_row;
    logic [3:0] frontend_source_req_rows;
    logic [15:0] frontend_source_req_element;
    logic frontend_source_rsp_valid;
    logic frontend_source_rsp_ready;
    logic [1023:0] frontend_source_rsp_values;
    logic [63:0] frontend_source_rsp_lane_mask;

    logic aux_start_valid;
    logic aux_start_ready;
    logic [63:0] aux_start_address;
    logic [31:0] aux_start_bytes;
    logic [31:0] aux_start_total_bytes;
    logic aux_stream_valid;
    logic aux_stream_ready;
    logic [127:0] aux_stream_data;
    logic [15:0] aux_stream_byte_enable;
    logic aux_stream_last;
    logic aux_stream_done;
    logic aux_stream_error;
    logic aux_stream_abort_ack;
    logic aux_source_active;
    logic aux_source_gather_active;
    logic aux_source_pair_active;
    logic [3:0] aux_source_row_index;
    logic [5:0] aux_source_row_base;
    logic [3:0] aux_source_row_count;
    logic [15:0] aux_source_element;
    logic aux_residual_active;
    logic [7:0] aux_residual_tag;
    logic [1023:0] aux_source_values;
    logic [1023:0] aux_source_next_values;
    logic [1023:0] aux_source_third_values;
    logic [1023:0] aux_source_fourth_values;
    logic [1023:0] aux_source_fifth_values;
    logic [1023:0] aux_source_sixth_values;
    logic [1023:0] aux_source_seventh_values;
    logic [1023:0] aux_source_eighth_values;
    logic [7:0] aux_source_beat_count;
    logic aux_source_response_pending;
    logic [3:0] aux_source_fetch_tile_count;
    logic [2:0] aux_source_cached_tile_count;
    logic [5:0] aux_source_cached_row_base;
    logic [3:0] aux_source_cached_row_count;
    logic [15:0] aux_source_cached_element;
    logic aux_source_cache_match;
    logic [15:0] aux_source_remaining_elements;

    logic loader_req,loader_req_ready,loader_done,loader_done_ready,loader_done_panel;
    logic [3:0] loader_error; logic acquire_valid,acquire_ready,acquire_panel;
    logic [15:0] acquire_operator_id;
    logic [15:0] acquire_stripe; logic acquire_has_enh;
    logic release_valid,release_ready,release_panel,loader_abort_ack;
    logic [1:0] panel_valid,panel_loading,panel_computing;
    logic [15:0] next_load_stripe,current_stripe,stripe_count;
    logic [63:0] next_load_weight_base;
    logic [63:0] next_load_weight_enhancement;
    logic [63:0] next_load_weight_scale;
    logic [63:0] next_load_up_weight_base;
    logic [63:0] next_load_up_weight_enhancement;
    logic [63:0] next_load_up_weight_scale;
    logic current_panel_valid,current_panel;
    logic [15:0] active_operator_tag;
    logic current_loader_req;
    logic prefetch_loader_req;
    logic prefetch_descriptor_valid;
    logic prefetch_submitted;
    logic prefetch_failed;
    logic prefetch_cancel_pending;
    logic [15:0] saved_prefetch_operator_tag;
    logic [15:0] saved_prefetch_input_features;
    logic [63:0] saved_prefetch_weight_base;
    logic [63:0] saved_prefetch_weight_enhancement;
    logic [63:0] saved_prefetch_weight_scale;
    logic saved_prefetch_has_w8;
    logic saved_prefetch_qkv_layout;
    logic start_prefetch_match;
    logic loader_req_prefetch;
    logic [15:0] loader_req_operator_id;
    logic [15:0] loader_req_stripe;
    logic [15:0] loader_req_input_features;
    logic [63:0] loader_req_weight_base;
    logic [63:0] loader_req_weight_enhancement;
    logic [63:0] loader_req_weight_scale;
    logic loader_req_has_w8;
    logic loader_req_qkv_layout;
    logic [127:0] panel0_scale_base,panel1_scale_base;
    logic panel0_scale_valid,panel1_scale_valid;
    logic [2:0] panel_bank_count;
    logic [9:0] panel_plane_rows;

    logic schedule_start,schedule_ready,schedule_done,schedule_done_ready;
    logic schedule_abort_ack,datapath_abort,datapath_abort_ack;
    logic issue_valid,issue_ready; logic [1:0] issue_mode;
    logic [5:0] issue_row_base; logic [3:0] issue_active_rows;
    logic [15:0] issue_k_base,issue_n_base;
    logic [7:0] issue_row_mask,issue_col_mask; logic [31:0] issue_k_mask;
    logic issue_first,issue_last,issue_last_stripe; logic [15:0] issue_tag;
    logic issue_mixed_phase, issue_mixed_phase_first, issue_reuse_panel;
    logic [7:0] issue_mixed_a8_rows;
    logic issue_explicit_activation_rows;
    logic [47:0] issue_activation_physical_rows;
    logic [127:0] issue_activation_lane_word_index;
    logic [15:0] issue_row_batch,issue_stripe;
    logic [31:0] issue_activation_addr;
    logic issue_qkv_padded_layout;
    logic [2:0] issue_panel_start_bank;
    logic [9:0] issue_panel_word_row;
    logic [2:0] issue_panel_bank_count;
    logic [9:0] issue_panel_plane_rows;
    logic issue_tag_start_ready;
    logic rescale_claimed;
    logic reader_scale_valid,reader_scale_ready; logic [127:0] reader_act_scales,
          reader_weight_scales; logic reader_issue_ready,reader_abort_ack;
    logic tile_req_valid,tile_req_ready; logic [1:0] tile_req_mode;
    logic [1023:0] tile_activation_payload, tile_weight_payload;
    logic [7:0] tile_row_mask,tile_col_mask; logic [31:0] tile_k_mask;
    logic tile_first,tile_last;
    logic tile_mixed_phase, tile_mixed_phase_first;
    logic [7:0] tile_mixed_a8_rows;
    logic [127:0] tile_ascale,tile_wscale;
    logic [15:0] tile_req_tag; logic tile_result_valid,tile_result_ready;
    logic [1023:0] tile_result; logic [63:0] tile_result_mask;
    logic [15:0] tile_result_tag;
    logic rescale_req_valid,rescale_req_ready,rescale_rsp_valid,rescale_rsp_ready;
    logic [1023:0] rescale_rsp_values;
    logic [63:0] rescale_rsp_mask;
    logic [15:0] rescale_rsp_tag;
    logic result_valid,result_ready; logic [1023:0] result_bf16;
    logic [63:0] result_mask; logic [5:0] result_row_base;
    logic [15:0] result_n_base, fragment_n_base;
    logic [7:0] result_row_mask,result_col_mask;
    logic [15:0] result_tag;
    logic fragment_valid,fragment_ready,fragment_abort_ack;
    logic [5:0] fragment_row; logic [15:0] fragment_channel;
    logic [63:0] fragment_row_base; logic [127:0] fragment_data;
    logic [15:0] fragment_be; logic [15:0] fragment_tag;
    logic output_cfg_valid,output_cfg_ready,output_abort_ack,output_error;
    logic output_path_done_pulse;
    logic output_residual_ddr_stage;
    logic output_write_valid;
    logic output_write_ready;
    logic [63:0] output_write_address;
    logic [31:0] output_write_transaction_bytes;
    logic [127:0] output_write_data;
    logic [15:0] output_write_byte_enable;
    logic output_write_first;
    logic output_write_last;
    logic output_write_done;
    logic output_write_error;
    logic writer_start_valid;
    logic writer_start_ready;
    logic writer_data_valid;
    logic writer_data_ready;
    logic writer_done;
    logic writer_error;
    logic writer_abort_ack;
    logic writer_active;
    logic residual_read_request_valid;
    logic residual_read_request_ready;
    logic [63:0] residual_read_request_address;
    logic [31:0] residual_read_request_bytes;
    logic [7:0] residual_read_request_tag;
    logic residual_read_local;
    logic residual_read_request_done;
    logic residual_read_request_error;
    logic residual_read_data_valid;
    logic residual_read_data_ready;
    logic [127:0] residual_read_data;
    logic [15:0] residual_read_byte_enable;
    logic residual_read_data_last;
    logic [7:0] residual_read_data_tag;
    logic residual_local_active;
    logic [5:0] residual_local_row;
    logic [7:0] residual_local_tag;
    logic [1:0] residual_local_read_pending;
    logic residual_local_response_pending;
    logic [127:0] residual_local_response_data;
    logic residual_local_done_pulse;
    logic local_valid,local_ready; logic [63:0] local_addr; logic [127:0] local_data;
    logic [15:0] local_be;
    logic [9:0] combine_write_address_i, combine_read_address_i;
    logic [8:0] completed_rows;
    logic output_row_commit;
    logic pending_release; logic [15:0] release_tag;
    localparam integer ABORT_FRONTEND = 0;
    localparam integer ABORT_READER = 1;
    localparam integer ABORT_TILE = 2;
    localparam integer ABORT_FRAGMENT = 3;
    localparam integer ABORT_OUTPUT = 4;
    localparam integer ABORT_LOADER = 5;
    localparam integer ABORT_RESCALE = 6;
    logic [6:0] abort_client_active;
    logic [6:0] abort_client_pending;
    logic [6:0] abort_client_start_event;
    logic [6:0] abort_client_clear_event;
    logic [6:0] abort_client_active_next;
    logic abort_started;
    logic schedule_abort_complete;
    logic child_abort_request;
    logic derived_segment_valid;
    logic [6:0] derived_row_end;
    logic [32:0] derived_activation_end;
    logic [13:0] derived_output_stripes;
    logic [8:0] derived_output_rows;
    logic [24:0] derived_output_bytes;
    logic configuration_valid;
    logic configuration_valid_reg;
    logic accepted_prefetch_present;
    logic accepted_prefetch_match;
    logic [21:0] local_stride_product_low;
    logic [21:0] local_stride_product_high;
    logic [6:0] configured_row_end;
    logic [64:0] configured_output_end;
    logic [24:0] configured_output_bytes;
    logic [13:0] configured_output_stripes;
    logic [32:0] configured_activation_end;
    logic [32:0] mixed_activation_end;
    logic [38:0] configured_local_output_end;
    logic [24:0] mixed_input_bytes;
    logic [6:0] quantization_group_base;
    logic configured_segment_valid;
    logic configured_output_within_limit;
    logic configured_local_output_within_limit;
    logic command_error;
    logic completion_error;
    logic rescale_inflight;
    logic current_panel_scale_valid;
    logic panel_acquire_active;
    logic resident_activation_valid;
    logic [5:0] resident_rows;
    logic [15:0] resident_input_features;
    logic [1:0] resident_segment_count;
    logic [5:0] resident_segment_mode;
    logic [17:0] resident_segment_row_base;
    logic [17:0] resident_segment_row_count;
    logic [95:0] resident_segment_activation_base_byte_offset;
    logic [47:0] resident_row_enable;
    logic resident_mixed_groups;
    logic [5:0] resident_compute_group_count;
    logic [47:0] resident_row_precision_a8;
    logic [287:0] resident_row_compute_group;
    logic [143:0] resident_row_pe_slot;
    logic [95:0] resident_row_phase_mask;
    logic mixed_activation_rows;
    logic writer_all_rows;

    function automatic [5:0] get_seg_base(input [1:0] index);
        get_seg_base=seg_base[index*6+:6]; endfunction
    function automatic [5:0] get_seg_rows(input [1:0] index);
        get_seg_rows=seg_rows[index*6+:6]; endfunction
    function automatic [3:0] rows_up_to_eight(input [5:0] count);
        rows_up_to_eight = count > 6'd8 ? 4'd8 : 4'(count);
    endfunction

    always_comb begin : current_batch_precision
        total_activation_group_count = {1'b0, mixed_compute_group_count};
        if (batch_pair.enable && batch_pair.batch_count >= 3'd2) begin
            total_activation_group_count = total_activation_group_count +
                {1'b0, batch_pair.second_row_config.compute_group_count};
            if (batch_pair.batch_count >= 3'd3)
                total_activation_group_count = total_activation_group_count +
                    {1'b0, batch_pair.third_row_config.compute_group_count};
            if (batch_pair.batch_count >= 3'd4)
                total_activation_group_count = total_activation_group_count +
                    {1'b0, batch_pair.fourth_row_config.compute_group_count};
            if (batch_pair.batch_count >= 3'd5)
                total_activation_group_count = total_activation_group_count +
                    {1'b0, batch_pair.fifth_row_config.compute_group_count};
            if (batch_pair.batch_count >= 3'd6)
                total_activation_group_count = total_activation_group_count +
                    {1'b0, batch_pair.sixth_row_config.compute_group_count};
        end

        quantization_group_base = 7'd0;
        if (quantization_batch_index >= 3'd1)
            quantization_group_base = quantization_group_base +
                {1'b0, mixed_compute_group_count};
        if (quantization_batch_index >= 3'd2)
            quantization_group_base = quantization_group_base +
                {1'b0, batch_pair.second_row_config.compute_group_count};
        if (quantization_batch_index >= 3'd3)
            quantization_group_base = quantization_group_base +
                {1'b0, batch_pair.third_row_config.compute_group_count};
        if (quantization_batch_index >= 3'd4)
            quantization_group_base = quantization_group_base +
                {1'b0, batch_pair.fourth_row_config.compute_group_count};
        if (quantization_batch_index >= 3'd5)
            quantization_group_base = quantization_group_base +
                {1'b0, batch_pair.fifth_row_config.compute_group_count};

        current_batch_a4_rows = '0;
        for (integer batch_row = 0; batch_row < 8; batch_row = batch_row + 1) begin
            if (mixed_groups && batch_row < current_batch_rows) begin
                current_batch_a4_rows[batch_row] =
                    !quantization_row_config.row_precision_a8[
                        current_row + 6'(batch_row)];
            end else begin
                for (integer segment = 0; segment < 3; segment = segment + 1) begin
                    if (batch_row < current_batch_rows && segment < seg_count &&
                        current_row + 6'(batch_row) >= get_seg_base(2'(segment)) &&
                        current_row + 6'(batch_row) <
                            get_seg_base(2'(segment)) + get_seg_rows(2'(segment)) &&
                        seg_mode[segment*2 +: 2] == MODE_W4A4)
                    current_batch_a4_rows[batch_row] = 1'b1;
                end
            end
        end
    end

    always_comb begin : current_batch_context_max
        current_batch_precomputed_max = '0;
        current_batch_precomputed_valid_mask = '0;
        case (current_row[5:3])
            3'd0: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[0*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[0*8 +: 8];
            end
            3'd1: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[1*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[1*8 +: 8];
            end
            3'd2: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[2*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[2*8 +: 8];
            end
            3'd3: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[3*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[3*8 +: 8];
            end
            3'd4: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[4*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[4*8 +: 8];
            end
            3'd5: begin
                current_batch_precomputed_max =
                    fixed_attention_context_max_values[5*128 +: 128];
                current_batch_precomputed_valid_mask =
                    fixed_attention_context_max_valid[5*8 +: 8];
            end
            default: begin end
        endcase
        current_batch_row_mask = current_batch_rows == 4'd8 ? 8'hff :
            8'((9'd1 << current_batch_rows) - 9'd1);
        current_batch_precomputed_max_valid = use_precomputed_max &&
            current_row[2:0] == 0 &&
            (current_batch_precomputed_valid_mask & current_batch_row_mask) ==
                current_batch_row_mask;
    end

    function automatic logic start_has_w8_segment;
        input logic [1:0] count;
        input logic [5:0] modes;
        logic found;
        begin
            found = 1'b0;
            for (integer index = 0; index < 3; index = index + 1)
                if (index < count && modes[index*2 +: 2] == MODE_W8A8)
                    found = 1'b1;
            start_has_w8_segment = found;
        end
    endfunction

    function automatic [15:0] bf16_div16_rne(input [15:0] value);
        logic sign;
        logic [7:0] exponent;
        logic [6:0] fraction;
        logic [7:0] significand;
        logic [7:0] quotient;
        logic [7:0] remainder;
        logic [7:0] halfway;
        integer shift;
        begin
            sign = value[15];
            exponent = value[14:7];
            fraction = value[6:0];
            if (exponent == 8'hff) begin
                bf16_div16_rne = value;
            end else if (exponent >= 8'd5) begin
                bf16_div16_rne = {sign, exponent - 8'd4, fraction};
            end else begin
                significand = exponent == 8'd0 ? {1'b0, fraction} :
                    {1'b1, fraction};
                shift = exponent == 8'd0 ? 4 : 5 - integer'(exponent);
                quotient = significand >> shift;
                remainder = significand & (8'hff >> (8 - shift));
                halfway = 8'd1 << (shift - 1);
                if (remainder > halfway ||
                    (remainder == halfway && quotient[0]))
                    quotient = quotient + 8'd1;
                bf16_div16_rne = {sign, 7'd0, quotient};
            end
        end
    endfunction

    always_comb begin
        logic [15:0] segment_row_bytes;

        derived_segment_valid = 1'b1;
        mixed_input_bytes = k_size == 16'd4096 ?
            (25'(total_activation_group_count) << 15) :
            ((25'(total_activation_group_count) << 16) +
             (25'(total_activation_group_count) << 15));
        derived_row_end = 7'd0;
        derived_activation_end = {1'b0, seg_activation[31:0]};
        segment_row_bytes = 16'd0;
        for (integer segment = 0; segment < 3; segment = segment + 1) begin
            if (segment < seg_count) begin
                segment_row_bytes =
                    seg_mode[segment*2 +: 2] == MODE_W4A4 ?
                    k_size >> 1 : k_size;
                if (seg_mode[segment*2 +: 2] > MODE_W8A8 ||
                    ((seg_mode[segment*2 +: 2] == MODE_W8A8) !=
                     qkv_w8_layout) ||
                    seg_rows[segment*6 +: 6] == 0 ||
                    {1'b0, seg_base[segment*6 +: 6]} != derived_row_end ||
                    {1'b0, seg_activation[segment*32 +: 32]} != derived_activation_end)
                    derived_segment_valid = 1'b0;
                derived_row_end =
                    {1'b0, seg_base[segment*6 +: 6]} +
                    {1'b0, seg_rows[segment*6 +: 6]};
                derived_activation_end = derived_activation_end +
                    33'(seg_rows[segment*6 +: 6]) * 33'(segment_row_bytes);
            end
        end
        case (({1'b0, rows} + 7'd7) >> 3)
            7'd1: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 3);
            7'd2: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 4);
            7'd3: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 4) + (33'(k_size) << 3);
            7'd4: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 5);
            7'd5: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 5) + (33'(k_size) << 3);
            7'd6: mixed_activation_end = {1'b0, seg_activation[31:0]} +
                (33'(k_size) << 5) + (33'(k_size) << 4);
            default: mixed_activation_end = {1'b0, seg_activation[31:0]};
        endcase
        derived_output_stripes =
            14'(({1'b0, n_size} + 17'd7) >> 3);
        derived_output_rows = {3'b000, rows};
        if ((output_mode == 3'd1 || output_mode == 3'd2 ||
             output_mode == 3'd5) && rows[0])
            derived_output_rows = {3'b000, rows} + 9'd1;
        if (batch_pair.enable && batch_pair.batch_count >= 3'd2) begin
            derived_output_rows = derived_output_rows +
                {3'b000, batch_pair.second_rows} +
                {8'd0, batch_pair.second_rows[0] &&
                    !batch_pair.qkvo_group};
            if (batch_pair.batch_count >= 3'd3)
                derived_output_rows = derived_output_rows +
                    {3'b000, batch_pair.third_rows} +
                    {8'd0, batch_pair.third_rows[0] &&
                        !batch_pair.qkvo_group};
            if (batch_pair.batch_count >= 3'd4)
                derived_output_rows = derived_output_rows +
                    {3'b000, batch_pair.fourth_rows} +
                    {8'd0, batch_pair.fourth_rows[0] &&
                        !batch_pair.qkvo_group};
            if (batch_pair.batch_count >= 3'd5)
                derived_output_rows = derived_output_rows +
                    {3'b000, batch_pair.fifth_rows} +
                    {8'd0, batch_pair.fifth_rows[0] &&
                        !batch_pair.qkvo_group};
            if (batch_pair.batch_count >= 3'd6)
                derived_output_rows = derived_output_rows +
                    {3'b000, batch_pair.sixth_rows} +
                    {8'd0, batch_pair.sixth_rows[0] &&
                        !batch_pair.qkvo_group};
        end
        derived_output_bytes =
            25'(derived_output_rows) * 25'(derived_output_stripes) * 25'd16;
        if (output_mode == 3'd1 || output_mode == 3'd2)
            derived_output_bytes = derived_output_bytes << 1;
    end

    always_comb begin
        attention_pair_resident_matches = attention_pair_query_prepared == 2'b11 &&
            resident_input_features == 16'd4096 && resident_mixed_groups &&
            (attention_pair_query_second ?
             (rows == resident_batch_pair.second_rows &&
              row_enable == resident_batch_pair.second_row_config.row_enable &&
              mixed_compute_group_count == resident_batch_pair.second_row_config.compute_group_count &&
              mixed_row_precision_a8 == resident_batch_pair.second_row_config.row_precision_a8 &&
              mixed_row_compute_group == resident_batch_pair.second_row_config.row_compute_group &&
              mixed_row_pe_slot == resident_batch_pair.second_row_config.row_pe_slot &&
              mixed_row_phase_mask == resident_batch_pair.second_row_config.row_phase_mask) :
             (rows == resident_rows && row_enable == resident_row_enable &&
              mixed_compute_group_count == resident_compute_group_count &&
              mixed_row_precision_a8 == resident_row_precision_a8 &&
              mixed_row_compute_group == resident_row_compute_group &&
              mixed_row_pe_slot == resident_row_pe_slot &&
              mixed_row_phase_mask == resident_row_phase_mask));
        configured_output_within_limit = !configured_output_end[64] &&
            (configured_output_end[63:32] < output_limit[63:32] ||
             (configured_output_end[63:32] == output_limit[63:32] &&
              configured_output_end[31:0] <= output_limit[31:0]));
        configured_local_output_within_limit =
            output_base[63:32] == 0 && output_limit[63:32] == 0 &&
            configured_local_output_end[38:32] == 0 &&
            configured_local_output_end[31:0] <= output_limit[31:0];
        configuration_valid = rows != 0 && rows <= 48 && k_size != 0 &&
            (!attention_pair_query ||
             (mixed_groups && mixed_row_precision_a8 == 0 &&
              mixed_compute_group_count <= 6'd3 && k_size == 16'd4096 &&
              seg_activation[31:0] == (attention_pair_query_second ? 32'd98304 : 32'd0) &&
              {1'b0, source_base} + (65'(rows) << 13) <=
                {1'b0, fixed_ffn_batch_pair.normalized_limit} &&
              (use_resident_activation || (!attention_pair_query_second ||
               attention_pair_query_prepared[0])))) &&
            (!batch_pair.enable ||
             ((((output_mode == 3'd1 || output_mode == 3'd2 ||
                 output_mode == 3'd5) && k_size == 16'd4096) ||
               (output_mode == 3'd0 && k_size == 16'd4096 && n_size == 16'd128 &&
                batch_pair.batch_count == 3'd2 && batch_pair.compact_a4_layout &&
                mixed_compute_group_count <= 6'd3 &&
                batch_pair.second_row_config.compute_group_count <= 6'd3 &&
                mixed_row_precision_a8 == 0 &&
                batch_pair.second_row_config.row_precision_a8 == 0) ||
               (output_mode == 3'd6 && k_size == 16'd4096 &&
                batch_pair.qkvo_group && batch_pair.batch_count >= 3'd2) ||
               (output_mode == 3'd4 &&
                (k_size == 16'd12288 || k_size == 16'd4096) &&
                batch_pair.compact_a4_layout &&
                (batch_pair.qkvo_group ||
                 (batch_pair.batch_count == 3'd2 &&
                  mixed_compute_group_count <= 6'd3 &&
                  batch_pair.second_row_config.compute_group_count <=
                    6'd3 && mixed_row_precision_a8 == 0 &&
                  batch_pair.second_row_config.row_precision_a8 == 0)))) &&
              mixed_groups &&
              $countones(row_enable) == rows &&
              batch_pair.batch_count >= 3'd2 &&
              batch_pair.batch_count <= 3'd6 &&
              (batch_pair.compact_a4_layout ||
               batch_pair.batch_count == 3'd2) &&
              batch_pair.second_rows != 0 && batch_pair.second_rows <= 6'd48 &&
              batch_pair.second_row_config.mixed_group_enable &&
              !batch_pair.second_row_config.operator_has_w8 &&
              batch_pair.second_row_config.compute_group_count != 0 &&
              batch_pair.second_row_config.compute_group_count <= 6'd4 &&
              $countones(batch_pair.second_row_config.row_enable) ==
                  batch_pair.second_rows &&
              (batch_pair.batch_count < 3'd3 ||
               (batch_pair.third_rows != 0 && batch_pair.third_rows <= 6'd48 &&
                batch_pair.third_row_config.mixed_group_enable &&
                !batch_pair.third_row_config.operator_has_w8 &&
                batch_pair.third_row_config.compute_group_count != 0 &&
                batch_pair.third_row_config.compute_group_count <= 6'd4 &&
                $countones(batch_pair.third_row_config.row_enable) ==
                    batch_pair.third_rows)) &&
              (batch_pair.batch_count < 3'd4 ||
               (batch_pair.fourth_rows != 0 && batch_pair.fourth_rows <= 6'd48 &&
                batch_pair.fourth_row_config.mixed_group_enable &&
                !batch_pair.fourth_row_config.operator_has_w8 &&
                batch_pair.fourth_row_config.compute_group_count != 0 &&
                batch_pair.fourth_row_config.compute_group_count <= 6'd4 &&
                $countones(batch_pair.fourth_row_config.row_enable) ==
                    batch_pair.fourth_rows)) &&
              (batch_pair.batch_count < 3'd5 ||
               (batch_pair.fifth_rows != 0 && batch_pair.fifth_rows <= 6'd48 &&
                batch_pair.fifth_row_config.mixed_group_enable &&
                !batch_pair.fifth_row_config.operator_has_w8 &&
                batch_pair.fifth_row_config.compute_group_count != 0 &&
                batch_pair.fifth_row_config.compute_group_count <= 6'd4 &&
                $countones(batch_pair.fifth_row_config.row_enable) ==
                    batch_pair.fifth_rows)) &&
              (batch_pair.batch_count < 3'd6 ||
               (batch_pair.sixth_rows != 0 && batch_pair.sixth_rows <= 6'd48 &&
                batch_pair.sixth_row_config.mixed_group_enable &&
                !batch_pair.sixth_row_config.operator_has_w8 &&
                batch_pair.sixth_row_config.compute_group_count != 0 &&
                batch_pair.sixth_row_config.compute_group_count <= 6'd4 &&
                $countones(batch_pair.sixth_row_config.row_enable) ==
                    batch_pair.sixth_rows)) &&
              ((output_mode == 3'd4 && !source_ddr) ||
               (source_base == batch_pair.normalized_base && !normalized_source_end[64] &&
               normalized_source_end <= {1'b0,batch_pair.normalized_limit})))) &&
            k_size <= 12288 && n_size != 0 &&
            output_mode <= 3'd6 &&
            (fused_product == (output_mode == 3'd5)) &&
            (!fused_product ||
             ((!batch_pair.enable || batch_pair.compact_a4_layout) &&
              n_size[2:0] == 0 && n_size <= 16'd32760)) &&
            (mixed_groups ?
                (seg_count == 0 && mixed_compute_group_count != 0 &&
                 mixed_compute_group_count <= rows &&
                 (k_size == 16'd4096 || k_size == 16'd12288) &&
                 mixed_input_bytes <=
                    (expanded_activation_layout ?
                        25'd589824 : 25'd393216)) :
                (seg_count != 0 && configured_segment_valid &&
                 configured_row_end == {1'b0, rows})) &&
            (!has_w8 || weight_enh != 0) &&
            has_w8 == start_has_w8_segment(seg_count, seg_mode) &&
            has_w8 == qkv_w8_layout &&
            (!qkv_w8_layout || (has_w8 && k_size == 16'd4096)) &&
            (!use_resident_activation ||
             (resident_activation_valid &&
              (attention_pair_query ? attention_pair_resident_matches : (
              hardware_types_pkg::same_matmul_activation_group(
                  batch_pair, resident_batch_pair) &&
              rows == resident_rows &&
              k_size == resident_input_features &&
              seg_count == resident_segment_count &&
              seg_mode == resident_segment_mode &&
              seg_base == resident_segment_row_base &&
              seg_rows == resident_segment_row_count &&
              seg_activation == resident_segment_activation_base_byte_offset &&
              row_enable == resident_row_enable &&
              mixed_groups == resident_mixed_groups &&
              (!mixed_groups ||
               (mixed_compute_group_count == resident_compute_group_count &&
                mixed_row_precision_a8 == resident_row_precision_a8 &&
                mixed_row_compute_group == resident_row_compute_group &&
                mixed_row_pe_slot == resident_row_pe_slot &&
               mixed_row_phase_mask == resident_row_phase_mask)))))) &&
            (!use_precomputed_max ||
             ((fixed_attention_context_max_valid & row_enable) == row_enable)) &&
            configured_activation_end <=
                ((attention_pair_query || kv_batch_pair) ? 33'd196608 : expanded_activation_layout ?
                    33'd589824 : 33'd393216) &&
            (!mixed_activation_rows || mixed_activation_end <= 33'd393216) &&
            ((output_mode == 3'd0 || output_mode == 3'd3) ?
             (output_limit > output_base &&
              output_stride >= ({16'd0, n_size} << 1) &&
              configured_local_output_end <= 39'd393216 &&
              configured_local_output_within_limit) :
             (output_limit > output_base && configured_output_within_limit));
    end
    assign start_prefetch_match = start_operator_tag == saved_prefetch_operator_tag &&
        command_input_features == saved_prefetch_input_features &&
        command_weight_base == saved_prefetch_weight_base &&
        command_weight_enhancement == saved_prefetch_weight_enhancement &&
        command_weight_scale == saved_prefetch_weight_scale &&
        command_operator_has_w8 == saved_prefetch_has_w8 &&
        command_qkv_w8_layout == saved_prefetch_qkv_layout;
    assign start_ready = state == IDLE && !abort_request &&
        !prefetch_cancel_pending &&
        (!prefetch_descriptor_valid || prefetch_submitted);
    assign prefetch_ready = !prefetch_descriptor_valid &&
        !prefetch_cancel_pending && !abort_request &&
        state != ABORT_WAIT && state != ERROR_WAIT;
    assign prefetch_empty = !prefetch_descriptor_valid &&
        !prefetch_cancel_pending;
    assign panel_bank_count = qkv_w8_layout ? 3'd2 :
        has_w8 && k_size == 16'd12288 ? 3'd6 : 3'd4;
    assign panel_plane_rows = panel_bank_count == 3'd6 ?
        10'd512 : panel_bank_count == 3'd2 ?
        10'(k_size >> 3) : 10'(k_size >> 4);
    assign child_abort_request = abort_request || datapath_abort ||
        state == ABORT_WAIT || state == ERROR_WAIT;
    assign shared_tile_abort_request = child_abort_request;
    assign shared_rescale_abort_request = child_abort_request;
    assign shared_bf16_abort_request = child_abort_request;
    assign frontend_start = state == ROW_START && !use_resident_activation;
    assign activation_writer_abort_request = child_abort_request;
    assign activation_writer_cfg_valid = state == SEG_CFG &&
        !use_resident_activation;
    assign mixed_activation_rows = output_mode == 3'd0 && !qkv_w8_layout &&
        !mixed_groups;
    assign writer_all_rows = mixed_activation_rows || mixed_groups;
    always_comb begin
        quantization_rows = rows;
        quantization_row_config = '0;
        quantization_row_config.mixed_group_enable = mixed_groups;
        quantization_row_config.compute_group_count = mixed_compute_group_count;
        quantization_row_config.row_precision_a8 = mixed_row_precision_a8;
        quantization_row_config.row_compute_group = mixed_row_compute_group;
        quantization_row_config.row_pe_slot = mixed_row_pe_slot;
        quantization_row_config.row_phase_mask = mixed_row_phase_mask;
        quantization_source_row_offset = 8'd0;
        case (quantization_batch_index)
            3'd1: begin
                quantization_rows = batch_pair.second_rows;
                quantization_row_config = batch_pair.second_row_config;
                quantization_source_row_offset = {2'd0, rows};
            end
            3'd2: begin
                quantization_rows = batch_pair.third_rows;
                quantization_row_config = batch_pair.third_row_config;
                quantization_source_row_offset = {2'd0, rows} +
                    {2'd0, batch_pair.second_rows};
            end
            3'd3: begin
                quantization_rows = batch_pair.fourth_rows;
                quantization_row_config = batch_pair.fourth_row_config;
                quantization_source_row_offset = {2'd0, rows} +
                    {2'd0, batch_pair.second_rows} +
                    {2'd0, batch_pair.third_rows};
            end
            3'd4: begin
                quantization_rows = batch_pair.fifth_rows;
                quantization_row_config = batch_pair.fifth_row_config;
                quantization_source_row_offset = {2'd0, rows} +
                    {2'd0, batch_pair.second_rows} +
                    {2'd0, batch_pair.third_rows} +
                    {2'd0, batch_pair.fourth_rows};
            end
            3'd5: begin
                quantization_rows = batch_pair.sixth_rows;
                quantization_row_config = batch_pair.sixth_row_config;
                quantization_source_row_offset = {2'd0, rows} +
                    {2'd0, batch_pair.second_rows} +
                    {2'd0, batch_pair.third_rows} +
                    {2'd0, batch_pair.fourth_rows} +
                    {2'd0, batch_pair.fifth_rows};
            end
            default: begin end
        endcase
    end
    assign total_output_rows = {3'd0,rows} +
        (batch_pair.enable && batch_pair.batch_count >= 3'd2 ?
            {3'd0,batch_pair.second_rows} : 9'd0) +
        (batch_pair.enable && batch_pair.batch_count >= 3'd3 ?
            {3'd0,batch_pair.third_rows} : 9'd0) +
        (batch_pair.enable && batch_pair.batch_count >= 3'd4 ?
            {3'd0,batch_pair.fourth_rows} : 9'd0) +
        (batch_pair.enable && batch_pair.batch_count >= 3'd5 ?
            {3'd0,batch_pair.fifth_rows} : 9'd0) +
        (batch_pair.enable && batch_pair.batch_count >= 3'd6 ?
            {3'd0,batch_pair.sixth_rows} : 9'd0);
    always_comb begin
        activation_writer_cfg_mode = MODE_W4A4;
        activation_writer_cfg_physical_row_base = 6'd0;
        activation_writer_cfg_row_count = 6'd0;
        case (seg_index)
            2'd0: begin
                activation_writer_cfg_mode = seg_mode[1:0];
                activation_writer_cfg_physical_row_base = seg_base[5:0];
                activation_writer_cfg_row_count = seg_rows[5:0];
            end
            2'd1: begin
                activation_writer_cfg_mode = seg_mode[3:2];
                activation_writer_cfg_physical_row_base = seg_base[11:6];
                activation_writer_cfg_row_count = seg_rows[11:6];
            end
            2'd2: begin
                activation_writer_cfg_mode = seg_mode[5:4];
                activation_writer_cfg_physical_row_base = seg_base[17:12];
                activation_writer_cfg_row_count = seg_rows[17:12];
            end
            default: begin end
        endcase
        if (writer_all_rows) begin
            activation_writer_cfg_mode = mixed_groups ? MODE_W4A4 :
                seg_mode[1:0];
            activation_writer_cfg_physical_row_base = 6'd0;
            activation_writer_cfg_row_count = quantization_rows;
        end
    end
    assign activation_writer_cfg_elements_per_row = k_size;
    assign activation_writer_cfg_activation_base_byte_offset = quantization_batch_index == 0 ?
        (writer_all_rows ? seg_activation[31:0] :
         seg_activation[seg_index*32 +: 32]) :
        batch_pair.compact_a4_layout ?
            (k_size == 16'd12288 ?
                ((32'(quantization_group_base) << 16) +
                 (32'(quantization_group_base) << 15)) :
                (32'(quantization_group_base) << 15)) :
            32'd131072;
    assign activation_writer_cfg_activation_limit_byte_offset = mixed_groups ?
        (expanded_activation_layout ?
            32'd589824 : 32'd393216) : mixed_activation_rows ? mixed_activation_end[31:0] :
        (seg_index + 1'b1 < seg_count ?
         seg_activation[(seg_index+1'b1)*32 +: 32] : 32'd393216);
    assign activation_writer_cfg_chunk_major = writer_all_rows;
    assign activation_writer_cfg_row_partial = 1'b0;
    assign activation_writer_cfg_segment_count = seg_count;
    assign activation_writer_cfg_segment_mode = seg_mode;
    assign activation_writer_cfg_segment_row_base = seg_base;
    assign activation_writer_cfg_segment_row_count = seg_rows;
    assign activation_writer_cfg_mixed_group_enable = mixed_groups;
    assign activation_writer_cfg_compute_group_count =
        quantization_row_config.compute_group_count;
    assign activation_writer_cfg_row_precision_a8 =
        quantization_row_config.row_precision_a8;
    assign activation_writer_cfg_row_compute_group =
        quantization_row_config.row_compute_group;
    assign activation_writer_cfg_row_pe_slot = quantization_row_config.row_pe_slot;
    assign activation_writer_cfg_row_phase_mask =
        quantization_row_config.row_phase_mask;
    assign activation_writer_quantized_valid = quantized_valid;
    assign quantized_ready = activation_writer_quantized_ready;
    assign activation_writer_quantized_row = quantized_row;
    assign activation_writer_quantized_element = quantized_element;
    assign activation_writer_quantized_values = quantized_values;
    assign activation_writer_quantized_lane_mask = quantized_mask;
    assign activation_writer_quantized_scale = 16'd0;
    assign activation_scale_ready = 1'b1;
    assign output_cfg_valid = state == OUTPUT_CFG;
    assign schedule_start = state == COMPUTE_START;
    assign schedule_done_ready=state==COMPUTE && completed_rows==total_output_rows &&
        output_path_done_pulse;
    assign done_valid=state==FINISH;
    assign error=completion_error;
    assign current_panel_scale_valid = current_panel ? panel1_scale_valid :
        panel0_scale_valid;

    always_comb begin
        abort_client_start_event = '0;
        abort_client_start_event[ABORT_FRONTEND] = frontend_start && frontend_ready;
        abort_client_start_event[ABORT_READER] = issue_valid && issue_ready;
        abort_client_start_event[ABORT_TILE] = tile_req_valid && tile_req_ready;
        abort_client_start_event[ABORT_FRAGMENT] = result_valid && result_ready;
        abort_client_start_event[ABORT_OUTPUT] = output_cfg_valid && output_cfg_ready;
        abort_client_start_event[ABORT_LOADER] = loader_req && loader_req_ready;
        abort_client_start_event[ABORT_RESCALE] = rescale_req_valid && rescale_req_ready;

        abort_client_clear_event = '0;
        abort_client_clear_event[ABORT_FRONTEND] = frontend_done_pulse ||
            frontend_abort_ack;
        abort_client_clear_event[ABORT_READER] = reader_abort_ack;
        abort_client_clear_event[ABORT_TILE] = shared_tile_abort_ack ||
            (shared_tile_accum_result_valid && shared_tile_accum_result_ready &&
             shared_tile_accum_result_last_k_step);
        abort_client_clear_event[ABORT_FRAGMENT] = fragment_abort_ack;
        abort_client_clear_event[ABORT_OUTPUT] = output_abort_ack;
        abort_client_clear_event[ABORT_LOADER] = loader_abort_ack;
        abort_client_clear_event[ABORT_RESCALE] =
            (rescale_rsp_valid && rescale_rsp_ready) || shared_rescale_abort_ack;

        abort_client_active_next = abort_client_active;
        if (start_valid && start_ready) begin
            abort_client_active_next = '0;
            if (prefetch_descriptor_valid && prefetch_submitted)
                abort_client_active_next[ABORT_LOADER] =
                    abort_client_active[ABORT_LOADER];
        end
        abort_client_active_next =
            (abort_client_active_next | abort_client_start_event) &
            ~abort_client_clear_event;
    end

    matmul_activation_controller #(.MAX_ELEMENTS(12288),.ADDR_WIDTH(14),.TAG_WIDTH(16)) frontend(
        .clk,.rst,.abort_request(child_abort_request),.abort_ack(frontend_abort_ack),
        .start_valid(frontend_start),.start_ready(frontend_ready),
        .activation_a4_rows(current_batch_a4_rows),
        .precomputed_max_valid(current_batch_precomputed_max_valid),
        .precomputed_row_max_abs(current_batch_precomputed_max),
        .row_base(current_row),.row_count(current_batch_rows),
        .element_count(k_size[13:0]),
        .source_request_valid(frontend_source_req_valid),
        .source_request_ready(frontend_source_req_ready),
        .source_request_row_base(frontend_source_req_row[5:0]),
        .source_request_row_count(frontend_source_req_rows),
        .source_request_element(frontend_source_req_element[13:0]),
        .source_response_valid(frontend_source_rsp_valid),
        .source_response_ready(frontend_source_rsp_ready),
        .source_response_values(frontend_source_rsp_values),
        .source_response_lane_mask(frontend_source_rsp_lane_mask),
        .max_req_valid,.max_req_ready,.max_req_values,.max_req_lane_mask,
        .max_req_tag,.max_rsp_valid,.max_rsp_ready,.max_rsp_values,
        .max_rsp_row_mask,.max_rsp_tag,
        .quant_scale_req_valid,.quant_scale_req_ready,.quant_scale_req_a4_row_mask,
        .quant_scale_req_row_max_abs,.quant_scale_req_row_mask,
        .quant_scale_rsp_valid,.quant_scale_rsp_ready,
        .quant_scale_rsp_values_bf16(quant_scale_rsp_values),
        .quant_values_req_valid,.quant_values_req_ready,
        .quant_values_req_values_bf16(quant_values_req_values),
        .quant_values_req_lane_mask,.quant_values_req_tag,
        .quant_values_rsp_valid,.quant_values_rsp_ready,.quant_values_rsp_values,
        .quant_values_rsp_lane_mask,.quant_values_rsp_tag,
        .scale_valid(activation_writer_scale_valid),
        .scale_ready(activation_writer_scale_ready),
        .scale_row_base(activation_writer_scale_row_base),
        .scale_row_mask(activation_writer_scale_row_mask),
        .scale_values_bf16(activation_writer_scale_values),
        .quantized_valid,.quantized_ready,.quantized_row_base(quantized_row),
        .quantized_element(quantized_element[13:0]),.quantized_values,
        .quantized_lane_mask(quantized_mask),.quantized_tag(),
        .done_pulse(frontend_done_pulse),.error(frontend_error),
        .accepted_source_tiles(),.accepted_max_tiles(),.accepted_quantized_tiles());

    assign source_req_valid = frontend_source_req_valid && !source_ddr;
    assign source_req_row = frontend_source_req_row[5:0];
    assign source_req_rows = frontend_source_req_rows;
    assign source_req_element = frontend_source_req_element;
    assign source_req_bank_base =
        output_mode == 3'd1 || output_mode == 3'd5 ? 5'd0 : 5'd16;
    assign frontend_source_req_ready = source_ddr ?
        (!aux_source_gather_active && !aux_source_active &&
         !aux_residual_active && !aux_source_response_pending) :
        source_req_ready;
    assign frontend_source_rsp_valid = source_ddr ?
        aux_source_response_pending : source_rsp_valid;
    assign frontend_source_rsp_values = source_ddr ?
        aux_source_values : source_rsp_values;
    assign frontend_source_rsp_lane_mask = source_ddr ?
        ({64{1'b1}} >> (64 - 8 * aux_source_row_count)) :
        source_rsp_lane_mask;
    assign source_rsp_ready = !source_ddr && frontend_source_rsp_ready;

    assign aux_source_cache_match = aux_source_cached_tile_count != 0 &&
        frontend_source_req_row[5:0] == aux_source_cached_row_base &&
        frontend_source_req_rows == aux_source_cached_row_count &&
        frontend_source_req_element == aux_source_cached_element;

    assign aux_source_remaining_elements =
        k_size - aux_source_element;

    always_comb begin
        aux_start_valid = 1'b0;
        aux_start_address = 64'd0;
        aux_start_bytes = 32'd0;
        aux_start_total_bytes = 32'd0;
        if (!child_abort_request && aux_source_gather_active &&
            !aux_source_active && !aux_source_response_pending) begin
            aux_start_valid = 1'b1;
            aux_start_address = source_base +
                (64'(quantization_source_row_offset) << 13) +
                64'(aux_source_row_base + aux_source_row_index) * 64'd8192 +
                64'(aux_source_element) * 64'd2;
            aux_start_bytes = aux_source_remaining_elements >= 16'd64 ?
                32'd128 : {15'd0, aux_source_remaining_elements, 1'b0};
            aux_start_total_bytes = aux_source_pair_active ?
                (32'(aux_source_row_count) << 7) : aux_start_bytes;
        end else if (!child_abort_request && residual_read_request_valid &&
                     !residual_read_local && !aux_residual_active) begin
            aux_start_valid = 1'b1;
            aux_start_address = residual_read_request_address;
            aux_start_bytes = residual_read_request_bytes;
            aux_start_total_bytes = residual_read_request_bytes;
        end

        aux_stream_ready = aux_source_active ? 1'b1 :
            aux_residual_active ? residual_read_data_ready : 1'b0;
        residual_read_request_ready = residual_read_local ?
            (!residual_local_active && !residual_local_response_pending) :
            (aux_start_ready && !aux_source_active && !aux_residual_active &&
             !aux_source_gather_active);
        residual_read_request_done = residual_read_local ?
            residual_local_done_pulse :
            (aux_residual_active &&
             ((aux_stream_done && !aux_stream_error) || aux_stream_abort_ack));
        residual_read_request_error = !residual_read_local &&
            aux_residual_active &&
            ((aux_stream_done && aux_stream_error) || aux_stream_abort_ack);
        residual_read_data_valid = residual_read_local ?
            residual_local_response_pending :
            (aux_residual_active && aux_stream_valid);
        residual_read_data = residual_read_local ?
            residual_local_response_data : aux_stream_data;
        residual_read_byte_enable = residual_read_local ?
            16'hffff : aux_stream_byte_enable;
        residual_read_data_last = residual_read_local ?
            residual_local_row + 1'b1 == rows : aux_stream_last;
        residual_read_data_tag = residual_read_local ?
            residual_local_tag : aux_residual_tag;

        writer_start_valid = output_write_valid && output_write_first &&
            !writer_active;
        writer_data_valid = output_write_valid && writer_active;
        output_write_ready = writer_active && writer_data_ready;
        output_write_done = writer_done;
        output_write_error = writer_error || (writer_abort_ack && writer_active);
    end

    assign residual_local_read_valid = residual_local_active &&
        !(|residual_local_read_pending) &&
        !residual_local_response_pending;
    assign residual_local_read_row = residual_local_row;
    assign residual_local_read_channel_chunk =
        residual_read_request_address[12:4];

    operator_dma_reader #(.REQUEST_TAG(8'h82)) auxiliary_reader (
        .clk(clk), .rst(rst), .abort_request(child_abort_request),
        .start_valid(aux_start_valid), .start_ready(aux_start_ready),
        .start_address(aux_start_address), .start_bytes(aux_start_bytes),
        .start_total_bytes(aux_start_total_bytes),
        .request_valid(aux_read_request_valid),
        .request_ready(aux_read_request_ready),
        .request_address(aux_read_request_address),
        .request_bytes(aux_read_request_bytes),
        .request_tag(aux_read_request_tag), .read_valid(aux_read_data_valid),
        .read_ready(aux_read_data_ready), .read_data(aux_read_data),
        .read_byte_enable(aux_read_byte_enable),
        .read_last(aux_read_data_last), .read_tag(aux_read_data_tag),
        .response_error(aux_read_request_error), .upstream_abort_ack(1'b0),
        .data_valid(aux_stream_valid), .data_ready(aux_stream_ready),
        .data(aux_stream_data),
        .data_byte_enable(aux_stream_byte_enable),
        .data_last(aux_stream_last), .data_byte_offset(),
        .done_pulse(aux_stream_done), .error(aux_stream_error),
        .abort_ack(aux_stream_abort_ack)
    );

    assign aux_read_second_span_valid = aux_read_request_valid &&
        aux_source_pair_active;
    assign aux_read_second_span_address = aux_source_pair_active ?
        source_base + 64'(aux_source_row_base + 6'd1) * 64'd8192 +
            (64'(quantization_source_row_offset) << 13) +
            64'(aux_source_element) * 64'd2 : 64'd0;
    assign aux_read_pair_stride = aux_source_pair_active &&
        aux_source_row_count > 4'd2 ? 32'd16384 : 32'd0;
    assign aux_read_pair_count = aux_source_pair_active ?
        {8'd0, aux_source_row_count[3:1]} : 11'd1;

    operator_dma_writer #(.REQUEST_TAG(8'h93)) output_writer_dma (
        .clk(clk), .rst(rst), .abort_request(child_abort_request),
        .start_valid(writer_start_valid), .start_ready(writer_start_ready),
        .start_address(output_write_address),
        .start_bytes(output_write_transaction_bytes),
        .request_valid(output_dma_request_valid),
        .request_ready(output_dma_request_ready),
        .request_address(output_dma_request_address),
        .request_bytes(output_dma_request_bytes),
        .request_tag(output_dma_request_tag),
        .data_valid(writer_data_valid), .data_ready(writer_data_ready),
        .data(output_write_data),
        .data_byte_enable(output_write_byte_enable),
        .data_last(output_write_last),
        .write_valid(output_dma_data_valid),
        .write_ready(output_dma_data_ready), .write_data(output_dma_data),
        .write_byte_enable(output_dma_byte_enable),
        .write_last(output_dma_data_last),
        .transaction_done(output_dma_request_done),
        .transaction_error(output_dma_request_error),
        .done_pulse(writer_done), .error(writer_error),
        .abort_ack(writer_abort_ack)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            aux_source_active <= 1'b0;
            aux_source_gather_active <= 1'b0;
            aux_source_pair_active <= 1'b0;
            aux_source_row_index <= '0;
            aux_source_row_base <= '0;
            aux_source_row_count <= '0;
            aux_source_element <= '0;
            aux_residual_active <= 1'b0;
            aux_residual_tag <= '0;
            aux_source_values <= '0;
            aux_source_next_values <= '0;
            aux_source_third_values <= '0;
            aux_source_fourth_values <= '0;
            aux_source_fifth_values <= '0;
            aux_source_sixth_values <= '0;
            aux_source_seventh_values <= '0;
            aux_source_eighth_values <= '0;
            aux_source_beat_count <= '0;
            aux_source_response_pending <= 1'b0;
            aux_source_fetch_tile_count <= '0;
            aux_source_cached_tile_count <= '0;
            aux_source_cached_row_base <= '0;
            aux_source_cached_row_count <= '0;
            aux_source_cached_element <= '0;
            residual_local_active <= 1'b0;
            residual_local_row <= '0;
            residual_local_tag <= '0;
            residual_local_read_pending <= '0;
            residual_local_response_pending <= 1'b0;
            residual_local_response_data <= '0;
            residual_local_done_pulse <= 1'b0;
            writer_active <= 1'b0;
        end else begin
            residual_local_done_pulse <= 1'b0;

            if ((start_valid && start_ready) ||
                (state == SEG_WAIT && activation_writer_done_pulse &&
                 batch_pair.enable && quantization_batch_index == 2'd0))
                aux_source_cached_tile_count <= '0;

            if (frontend_source_req_valid && frontend_source_req_ready &&
                source_ddr) begin
                aux_source_row_base <= frontend_source_req_row[5:0];
                aux_source_row_count <= frontend_source_req_rows;
                aux_source_element <= frontend_source_req_element;
                if (aux_source_cache_match) begin
                    aux_source_values <= aux_source_next_values;
                    aux_source_next_values <= aux_source_third_values;
                    aux_source_third_values <= aux_source_fourth_values;
                    aux_source_fourth_values <= aux_source_fifth_values;
                    aux_source_fifth_values <= aux_source_sixth_values;
                    aux_source_sixth_values <= aux_source_seventh_values;
                    aux_source_seventh_values <= aux_source_eighth_values;
                    aux_source_eighth_values <= '0;
                    aux_source_response_pending <= 1'b1;
                    aux_source_cached_tile_count <=
                        aux_source_cached_tile_count - 1'b1;
                    aux_source_cached_element <=
                        aux_source_cached_element + 16'd8;
                end else begin
                    aux_source_gather_active <= 1'b1;
                    // The same row spans serve max scanning and quantization.
                    aux_source_pair_active <= frontend_source_req_rows > 1 &&
                        !frontend_source_req_rows[0] &&
                        k_size - frontend_source_req_element >= 16'd64;
                    aux_source_row_index <= '0;
                    aux_source_values <= '0;
                    aux_source_next_values <= '0;
                    aux_source_third_values <= '0;
                    aux_source_fourth_values <= '0;
                    aux_source_fifth_values <= '0;
                    aux_source_sixth_values <= '0;
                    aux_source_seventh_values <= '0;
                    aux_source_eighth_values <= '0;
                    if (k_size - frontend_source_req_element >= 16'd64)
                        aux_source_fetch_tile_count <= 4'd8;
                    else
                        aux_source_fetch_tile_count <=
                            4'((k_size - frontend_source_req_element + 16'd7) >> 3);
                    aux_source_cached_tile_count <= '0;
                end
            end

            if (aux_start_valid && aux_start_ready) begin
                if (aux_source_gather_active) begin
                    aux_source_active <= 1'b1;
                    aux_source_beat_count <= '0;
                end else begin
                    aux_residual_active <= 1'b1;
                    aux_residual_tag <= residual_read_request_tag;
                end
            end
            if (aux_source_active && aux_stream_valid && aux_stream_ready) begin
                for (integer source_row = 0; source_row < 8; source_row++) begin
                    if ((aux_source_pair_active ?
                            {1'b0, aux_source_beat_count[5:3]} :
                            aux_source_row_index) == 4'(source_row)) begin
                        case (aux_source_beat_count[2:0])
                            3'd0: aux_source_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd1: aux_source_next_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd2: aux_source_third_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd3: aux_source_fourth_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd4: aux_source_fifth_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd5: aux_source_sixth_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd6: aux_source_seventh_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            3'd7: aux_source_eighth_values[
                                source_row*128 +: 128] <= aux_stream_data;
                            default: begin end
                        endcase
                    end
                end
                aux_source_beat_count <= aux_source_beat_count + 1'b1;
            end
            if (frontend_source_rsp_valid && frontend_source_rsp_ready &&
                source_ddr)
                aux_source_response_pending <= 1'b0;
            if (aux_stream_done && aux_source_active) begin
                aux_source_active <= 1'b0;
                if (aux_stream_error) begin
                    aux_source_gather_active <= 1'b0;
                    aux_source_response_pending <= 1'b1;
                    aux_source_cached_tile_count <= '0;
                end else if (!aux_source_pair_active &&
                             aux_source_row_index + 1'b1 <
                             aux_source_row_count) begin
                    aux_source_row_index <= aux_source_row_index + 1'b1;
                end else begin
                    aux_source_gather_active <= 1'b0;
                    aux_source_pair_active <= 1'b0;
                    aux_source_response_pending <= 1'b1;
                    aux_source_cached_tile_count <=
                        3'(aux_source_fetch_tile_count - 1'b1);
                    aux_source_cached_row_base <= aux_source_row_base;
                    aux_source_cached_row_count <= aux_source_row_count;
                    aux_source_cached_element <= aux_source_element + 16'd8;
                end
            end else if (aux_stream_done && aux_residual_active) begin
                aux_residual_active <= 1'b0;
            end
            if (aux_stream_abort_ack) begin
                if (aux_source_active || aux_source_gather_active)
                    aux_source_response_pending <= 1'b1;
                aux_source_active <= 1'b0;
                aux_source_gather_active <= 1'b0;
                aux_source_pair_active <= 1'b0;
                aux_residual_active <= 1'b0;
                aux_source_cached_tile_count <= '0;
            end

            if (done_valid || error)
                aux_source_cached_tile_count <= '0;

            if (residual_read_request_valid &&
                residual_read_request_ready && residual_read_local) begin
                residual_local_active <= 1'b1;
                residual_local_row <= 6'd0;
                residual_local_tag <= residual_read_request_tag;
            end
            residual_local_read_pending <= {
                residual_local_read_pending[0],
                residual_local_read_valid && residual_local_read_ready};
            if (residual_local_read_rsp_valid) begin
                residual_local_response_pending <= 1'b1;
                residual_local_response_data <= residual_local_read_rsp_data;
            end
            if (residual_local_response_pending &&
                residual_read_data_ready) begin
                residual_local_response_pending <= 1'b0;
                if (residual_local_row + 1'b1 == rows) begin
                    residual_local_active <= 1'b0;
                    residual_local_done_pulse <= 1'b1;
                end else begin
                    residual_local_row <= residual_local_row + 1'b1;
                end
            end
            if (child_abort_request) begin
                residual_local_active <= 1'b0;
                residual_local_response_pending <= 1'b0;
                residual_local_read_pending <= '0;
                residual_local_done_pulse <= 1'b1;
            end

            if (writer_start_valid && writer_start_ready)
                writer_active <= 1'b1;
            if (writer_done || writer_error || writer_abort_ack)
                writer_active <= 1'b0;
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(aux_source_gather_active &&
                      residual_read_request_valid && !residual_read_local))
                else $error("matmul_controller issued source and residual auxiliary reads together");
            if (residual_local_read_rsp_valid)
                assert (|residual_local_read_pending)
                    else $error("matmul_controller received an unowned local residual response");
        end
    end
`endif

    assign frontend_source_req_row[15:6] = '0;
    assign frontend_source_req_element[15:14] = 2'b00;
    assign quantized_element[15:14] = 2'b00;
    assign panel_acquire_active =
        (state == SEG_CFG || state == ROW_START || state == ROW_WAIT ||
         state == SEG_WAIT || state == OUTPUT_CFG ||
         state == COMPUTE_START || state == COMPUTE);
    assign activation_source_live = output_mode == 3'd0 && !source_ddr &&
        (state == SEG_CFG || state == ROW_START || state == ROW_WAIT ||
         state == SEG_WAIT);
    assign current_loader_req = panel_acquire_active &&
        next_load_stripe < stripe_count &&
        !(qkv_query_command && activation_source_live);
    assign prefetch_loader_req = prefetch_descriptor_valid &&
        !prefetch_submitted && !prefetch_cancel_pending && !prefetch_cancel &&
        (state == IDLE || state == FINISH ||
         (panel_acquire_active && next_load_stripe >= stripe_count));
    assign loader_req = current_loader_req ||
        (!current_loader_req && prefetch_loader_req);
    assign loader_req_prefetch = !current_loader_req && prefetch_loader_req;
    assign loader_req_operator_id = loader_req_prefetch ? saved_prefetch_operator_tag :
        active_operator_tag;
    assign loader_req_stripe = loader_req_prefetch ? 16'd0 : next_load_stripe;
    assign loader_req_input_features = loader_req_prefetch ?
        saved_prefetch_input_features : k_size;
    assign loader_req_weight_base = loader_req_prefetch ?
        saved_prefetch_weight_base :
        fused_product && next_load_stripe[0] ?
            next_load_up_weight_base : next_load_weight_base;
    assign loader_req_weight_enhancement = loader_req_prefetch ?
        saved_prefetch_weight_enhancement :
        fused_product && next_load_stripe[0] ?
            next_load_up_weight_enhancement : next_load_weight_enhancement;
    assign loader_req_weight_scale = loader_req_prefetch ?
        saved_prefetch_weight_scale :
        fused_product && next_load_stripe[0] ?
            next_load_up_weight_scale : next_load_weight_scale;
    assign loader_req_has_w8 = loader_req_prefetch ?
        saved_prefetch_has_w8 : has_w8;
    assign loader_req_qkv_layout = loader_req_prefetch ?
        saved_prefetch_qkv_layout : qkv_w8_layout;
    matmul_weight_panel_loader #(.OPERATOR_ID_WIDTH(16)) loader(
        .clk,.rst,.abort_request(child_abort_request || prefetch_cancel_pending),
        .abort_ack(loader_abort_ack),.load_req_valid(loader_req),.load_req_ready(loader_req_ready),
        .load_req_operator_id(loader_req_operator_id),.load_req_stripe_id(loader_req_stripe),
        .load_req_input_features({16'd0,loader_req_input_features}),
        .load_req_base_addr(loader_req_weight_base),
        .load_req_enhancement_addr(loader_req_weight_enhancement),
        .load_req_scale_addr(loader_req_weight_scale),
        .load_req_operator_has_w8(loader_req_has_w8),
        .load_req_qkv_layout(loader_req_qkv_layout),
        .load_req_allow_deferred(current_loader_req && state == COMPUTE &&
            (output_mode == 3'd1 || output_mode == 3'd2 ||
             output_mode == 3'd5)),
        // Panel 1 overlaps live activation storage in the paired projection
        // layouts. A paired Down panel also occupies rows 0..767 of macros
        // 22..25, so its next stripe must wait and reuse panel 0.
        .load_req_block_panel1((rows > 6'd8 && activation_source_live) ||
            attention_pair_query || kv_batch_pair ||
            (batch_pair.enable && k_size == 16'd12288)),
        .load_done_valid(loader_done),
        .load_done_ready(loader_done_ready),.load_done_panel(loader_done_panel),
        .load_done_error(loader_error),.dma_req_valid,.dma_req_ready,.dma_req_addr,
        .dma_req_bytes,.dma_req_plane,.dma_req_panel,.dma_req_last_for_plane,
        .dma_req_done,.dma_req_error,
        .dma_rsp_valid,.dma_rsp_ready,.dma_rsp_data,.dma_rsp_byte_enable,.dma_rsp_error,
        .dma_rsp_last,.dma_abort_ack,
        .panel_write_valid(panel_memory_write_valid),.panel_write_ready(panel_memory_write_ready),
        .panel_write_panel(panel_memory_write_panel),.panel_write_plane(panel_memory_write_plane),
        .panel_write_start_bank(panel_memory_write_start_bank),
        .panel_write_word_row(panel_memory_write_word_row),.panel_write_data(panel_memory_write_data),
        .panel_write_byte_enable(panel_memory_write_byte_enable),
        .panel0_scale_base,.panel0_scale_valid,.panel1_scale_base,.panel1_scale_valid,
        .compute_request_operator_id(active_operator_tag),
        .compute_request_stripe_id(current_stripe),
        .compute_acquire_valid(acquire_valid),.compute_acquire_ready(acquire_ready),
        .compute_acquire_panel(acquire_panel),.compute_acquire_operator_id(acquire_operator_id),
        .compute_acquire_stripe_id(acquire_stripe),.compute_acquire_has_enhancement(acquire_has_enh),
        .compute_release_valid(release_valid),.compute_release_ready(release_ready),
        .compute_release_panel(release_panel),.panel_valid,.panel_loading,.panel_computing,
        .panel_operator_id(),.panel_stripe_id(),.panel_has_enhancement(),.accepted_load_count(),
        .accepted_dma_request_count(),.accepted_base_request_count,
        .accepted_enhancement_request_count,.accepted_base_request_bytes,
        .accepted_enhancement_request_bytes,.accepted_scale_request_count,
        .accepted_scale_request_bytes,.accepted_dma_response_count(),
        .accepted_dma_response_bytes(),.discarded_dma_response_bytes(),
        .four_kib_split_count(),.dma_error_count());
    assign loader_done_ready=1'b1;
    assign panel_write_accepted=panel_memory_write_valid&&panel_memory_write_ready;
    assign panel_write_panel=panel_memory_write_panel; assign panel_write_plane=panel_memory_write_plane;
    assign panel_write_start_bank=panel_memory_write_start_bank;
    assign panel_write_word_row=panel_memory_write_word_row;
    assign panel_write_data=panel_memory_write_data;
    assign panel_write_byte_enable=panel_memory_write_byte_enable;
    assign acquire_ready=panel_acquire_active&&!current_panel_valid&&
        acquire_operator_id==active_operator_tag && acquire_stripe==current_stripe &&
        (acquire_panel ? panel1_scale_valid : panel0_scale_valid);
    assign release_valid=(pending_release&&shared_tile_req_valid&&shared_tile_req_ready&&
        tile_req_tag==release_tag&&tile_last)||(child_abort_request&&current_panel_valid);
    assign release_panel=current_panel;

    phase_shared_compute_scheduler #(
        .TRUSTED_START_CONFIGURATION(1'b1)
    ) scheduler(.clk,.rst,.abort_request(child_abort_request),.abort_ack(schedule_abort_ack),
        .datapath_abort_request(datapath_abort),.datapath_abort_ack,
        .start_valid(schedule_start),.start_ready(schedule_ready),
        .start_input_features(k_size),
        .start_output_features(fused_product ? n_size << 1 : n_size),
        .start_segment_count(seg_count),
        .start_segment_row_base(seg_base),.start_segment_row_count(seg_rows),
        .start_segment_mode(seg_mode),.start_segment_activation_base_byte_offset(seg_activation),
        .start_row_enable(row_enable),
        .start_mixed_group_enable(mixed_groups),
        .start_compute_group_count(mixed_compute_group_count),
        .start_row_precision_a8(mixed_row_precision_a8),
        .start_row_compute_group(mixed_row_compute_group),
        .start_row_pe_slot(mixed_row_pe_slot),
        .start_row_phase_mask(mixed_row_phase_mask),
        .start_batch_group(batch_pair),
        .issue_batch_index, .result_batch_index,
        .start_panel_bank_count(panel_bank_count),
        .start_panel_plane_rows(panel_plane_rows),.configuration_error(),
        .start_qkv_padded_layout(mixed_activation_rows),
        .issue_valid,.issue_ready,.issue_mode,.issue_physical_row_base(issue_row_base),
        .issue_active_rows,
        .issue_input_feature_base(issue_k_base),.issue_output_feature_base(issue_n_base),
        .issue_row_mask,.issue_k_mask,.issue_column_mask(issue_col_mask),
        .issue_first_k_step(issue_first),.issue_last_k_step(issue_last),
        .issue_last_for_stripe(issue_last_stripe),
        .issue_mixed_phase, .issue_mixed_phase_first,
        .issue_mixed_a8_rows, .issue_reuse_panel,
        .issue_explicit_activation_rows, .issue_activation_physical_rows,
        .issue_activation_lane_word_index, .issue_tag,
        .issue_row_batch_id(issue_row_batch),.issue_stripe_id(issue_stripe),
        .issue_activation_byte_address(issue_activation_addr),
        .issue_qkv_padded_layout,
        .issue_panel_start_bank(issue_panel_start_bank),
        .issue_panel_word_row(issue_panel_word_row),
        .issue_panel_bank_count,.issue_panel_plane_rows,
        .tile_result_valid,.tile_result_ready,.tile_result_bf16(tile_result),
        .tile_result_mask,.tile_result_tag,.result_valid,.result_ready,
        .result_bf16,.result_mask,.result_physical_row_base(result_row_base),
        .result_output_feature_base(result_n_base),.result_row_mask,.result_column_mask(result_col_mask),
        .result_tag,.done_valid(schedule_done),.done_ready(schedule_done_ready),
        .accepted_issue_count,.accepted_result_count,.completed_tile_count());

    always_comb begin
        for(integer i=0;i<8;i=i+1) begin
            case (scale_read_batch_index)
                3'd1: reader_act_scales[i*16+:16] = second_activation_scales[
                    issue_activation_physical_rows[i*6 +: 6]];
                3'd2: reader_act_scales[i*16+:16] = third_activation_scales[
                    issue_activation_physical_rows[i*6 +: 6]];
                3'd3: reader_act_scales[i*16+:16] = fourth_activation_scales[
                    issue_activation_physical_rows[i*6 +: 6]];
                3'd4: reader_act_scales[i*16+:16] = fifth_activation_scales[
                    issue_activation_physical_rows[i*6 +: 6]];
                3'd5: reader_act_scales[i*16+:16] = sixth_activation_scales[
                    issue_activation_physical_rows[i*6 +: 6]];
                default: reader_act_scales[i*16+:16] = activation_scales[
                    issue_explicit_activation_rows ?
                        issue_activation_physical_rows[i*6 +: 6] :
                        issue_row_base+i];
            endcase
            reader_weight_scales[i*16+:16] = issue_mode == MODE_W8A8 ?
                bf16_div16_rne(current_panel ? panel1_scale_base[i*16+:16] :
                    panel0_scale_base[i*16+:16]) :
                (current_panel ? panel1_scale_base[i*16+:16] :
                    panel0_scale_base[i*16+:16]);
        end
    end
    assign reader_scale_valid=issue_valid&&issue_first&&current_panel_valid&&
        current_panel_scale_valid&&
        issue_stripe==current_stripe;
    // Preserve one-cycle K-bundle acceptance after a tag starts. The next tag
    // waits at its boundary while rescale still owns the preceding result.
    assign shared_rescale_claim_valid = issue_valid && issue_first &&
        current_panel_valid && current_panel_scale_valid &&
        issue_stripe == current_stripe && !rescale_claimed;
    assign issue_tag_start_ready = !issue_first || rescale_claimed ||
        shared_rescale_claim_ready;
    assign issue_ready=reader_issue_ready&&current_panel_valid&&current_panel_scale_valid&&
        issue_stripe==current_stripe&&issue_tag_start_ready;
    matmul_operand_path reader(.clk,.rst,.abort_request(child_abort_request),
        .abort_ack(reader_abort_ack),.scale_cfg_valid(reader_scale_valid),
        .scale_cfg_ready(reader_scale_ready),.scale_cfg_activation(reader_act_scales),
        .scale_cfg_weight(reader_weight_scales),.scale_cfg_row_batch_id(issue_row_batch),
        .scale_cfg_stripe_id(issue_stripe),.issue_valid(issue_valid&&current_panel_valid&&
        issue_stripe==current_stripe&&issue_tag_start_ready),
        .issue_ready(reader_issue_ready),.issue_mode,
        .issue_active_rows,.issue_panel(current_panel),.issue_row_mask,.issue_k_mask,
        .issue_column_mask(issue_col_mask),.issue_first_k_step(issue_first),
        .issue_last_k_step(issue_last),.issue_tag,.issue_row_batch_id(issue_row_batch),
        .issue_mixed_phase, .issue_mixed_phase_first,
        .issue_mixed_a8_rows, .issue_reuse_panel,
        .issue_explicit_activation_rows, .issue_activation_physical_rows,
        .issue_activation_lane_word_index,
        .issue_expanded_activation_layout(expanded_activation_layout),
        .issue_activation_scales(reader_act_scales),
        .issue_weight_scales(reader_weight_scales),
        .issue_stripe_id(issue_stripe),.issue_activation_byte_address(issue_activation_addr),
        .issue_qkv_padded_layout,
        .issue_physical_row_base(issue_row_base),
        .issue_panel_start_bank(issue_panel_start_bank),
        .issue_panel_word_row(issue_panel_word_row),
        .issue_panel_bank_count,.issue_panel_plane_rows,
        .read_bundle_valid,.read_bundle_ready,.activation_read_word_index,
        .activation_read_word_count,.activation_read_qkv_padded_layout,
        .activation_read_physical_row_base,
        .activation_read_explicit_rows,
        .activation_read_physical_rows,
        .activation_read_lane_word_index,
        .panel_read_panel(panel_read_select),.panel_read_port_mask,
        .panel_read_address,.read_response_valid,.activation_read_data,.panel_read_data,
        .req_valid(tile_req_valid),.req_ready(tile_req_ready),.req_mode(tile_req_mode),
        .req_activation_payload(tile_activation_payload),
        .req_weight_payload(tile_weight_payload),.req_row_mask(tile_row_mask),
        .req_k_mask(tile_k_mask),.req_col_mask(tile_col_mask),
        .req_first_k_step(tile_first),.req_last_k_step(tile_last),
        .req_mixed_phase(tile_mixed_phase),
        .req_mixed_phase_first(tile_mixed_phase_first),
        .req_mixed_a8_rows(tile_mixed_a8_rows),
        .req_activation_scales(tile_ascale),.req_weight_scales(tile_wscale),
        .req_tag(tile_req_tag),.accepted_issue_count(),.accepted_request_count());
    assign shared_tile_req_valid = tile_req_valid;
    assign tile_req_ready = shared_tile_req_ready;
    assign shared_tile_req_mode = tile_req_mode;
    assign shared_tile_req_activation_payload = tile_activation_payload;
    assign shared_tile_req_weight_payload = tile_weight_payload;
    assign shared_tile_req_row_mask = tile_row_mask;
    assign shared_tile_req_k_mask = tile_k_mask;
    assign shared_tile_req_col_mask = tile_col_mask;
    assign shared_tile_req_first_k_step = tile_first;
    assign shared_tile_req_last_k_step = tile_last;
    assign shared_tile_req_mixed_phase = tile_mixed_phase;
    assign shared_tile_req_mixed_phase_first = tile_mixed_phase_first;
    assign shared_tile_req_mixed_a8_rows = tile_mixed_a8_rows;
    assign shared_tile_req_activation_scales = tile_ascale;
    assign shared_tile_req_weight_scales = tile_wscale;
    assign shared_tile_req_tag = tile_req_tag;
    assign accum_result_valid = shared_tile_accum_result_valid &&
        (!shared_tile_accum_result_last_k_step || rescale_req_ready);
    assign shared_tile_accum_result_ready = accum_result_ready &&
        (!shared_tile_accum_result_last_k_step || rescale_req_ready);
    assign accum_result_accumulators = shared_tile_accum_result_accumulators;
    assign accum_result_mask = shared_tile_accum_result_mask;
    assign accum_result_mode = shared_tile_accum_result_mode;
    assign accum_result_last_k_step = shared_tile_accum_result_last_k_step;
    assign accum_result_tag = shared_tile_accum_result_tag;
    assign rescale_req_valid = shared_tile_accum_result_valid && accum_result_ready &&
        shared_tile_accum_result_last_k_step;
    assign shared_rescale_req_valid = rescale_req_valid;
    assign rescale_req_ready = shared_rescale_req_ready;
    assign shared_rescale_req_accumulators = shared_tile_accum_result_accumulators;
    assign shared_rescale_req_activation_scales =
        shared_tile_accum_result_activation_scales;
    assign shared_rescale_req_weight_scales =
        shared_tile_accum_result_weight_scales;
    assign shared_rescale_req_qk_scale_bf16 = '0;
    assign shared_rescale_req_rescale_mode = 2'd0;
    assign shared_rescale_req_lane_mask = shared_tile_accum_result_mask;
    assign shared_rescale_req_tag = shared_tile_accum_result_tag;
    assign rescale_rsp_valid = shared_rescale_rsp_valid;
    assign shared_rescale_rsp_ready = rescale_rsp_ready;
    assign rescale_rsp_values = shared_rescale_rsp_values;
    assign rescale_rsp_mask = shared_rescale_rsp_lane_mask;
    assign rescale_rsp_tag = shared_rescale_rsp_tag;
    assign tile_result_valid = rescale_rsp_valid;
    assign rescale_rsp_ready = tile_result_ready || child_abort_request;
    assign tile_result = rescale_rsp_values;
    assign tile_result_mask = rescale_rsp_mask;
    assign tile_result_tag = rescale_rsp_tag;

    assign fragment_n_base = fused_product ?
        {1'b0, result_n_base[15:4], 3'b000} : result_n_base;
    matmul_output_fragmenter #(.ADDR_WIDTH(64)) fragmenter(
        .clk,.rst,.abort_request(child_abort_request),
        .abort_ack(fragment_abort_ack),.tile_valid(result_valid),.tile_ready(result_ready),
        .tile_bf16(result_bf16),.tile_mask(result_mask),.tile_physical_row_base(result_row_base),
        .tile_output_channel_base(fragment_n_base),.tile_row_mask(result_row_mask),
        .tile_column_mask(result_col_mask),
        .tile_row_byte_base(output_base +
            64'({26'd0, result_row_base} * output_stride)),
        .tile_row_stride(output_stride),.tile_tag(result_tag),.fragment_valid,.fragment_ready,
        .fragment_physical_row(fragment_row),.fragment_output_channel(fragment_channel),
        .fragment_row_byte_base(fragment_row_base),.fragment_data,.fragment_byte_enable(fragment_be),
        .fragment_tag,.error());
    matmul_output_controller #(.ADDR_WIDTH(64), .MAX_ROWS(48)) output_writer(
        .clk,.rst,.abort_request(child_abort_request),
        .abort_ack(output_abort_ack),.cfg_valid(output_cfg_valid),.cfg_ready(output_cfg_ready),
        .cfg_mode(output_mode),.cfg_physical_row_base(6'd0),.cfg_row_count(rows),
        .cfg_fused_product(fused_product),
        .cfg_batch_count(batch_pair.enable ? batch_pair.batch_count : 3'd1),
        .cfg_second_batch_rows(batch_pair.enable && batch_pair.batch_count >= 3'd2 ?
            batch_pair.second_rows : 6'd0),
        .cfg_third_batch_rows(batch_pair.enable && batch_pair.batch_count >= 3'd3 ?
            batch_pair.third_rows : 6'd0),
        .cfg_fourth_batch_rows(batch_pair.enable && batch_pair.batch_count >= 3'd4 ?
            batch_pair.fourth_rows : 6'd0),
        .cfg_fifth_batch_rows(batch_pair.enable && batch_pair.batch_count >= 3'd5 ?
            batch_pair.fifth_rows : 6'd0),
        .cfg_sixth_batch_rows(batch_pair.enable && batch_pair.batch_count >= 3'd6 ?
            batch_pair.sixth_rows : 6'd0),
        .in_batch_index(fragment_batch_index),
        .cfg_output_channels(n_size),.cfg_output_base(output_base),
        .cfg_output_limit(output_limit),.cfg_local_row_stride(output_stride),
        .cfg_residual_from_ddr(output_mode == 3'd3),
        .cfg_residual_base(fixed_ffn_workspace_config.residual_base),
        .cfg_residual_limit(fixed_ffn_workspace_config.residual_limit),
        .cfg_residual_storage_rows(residual_storage_rows),
        .in_valid(fragment_valid),.in_ready(fragment_ready),
        .in_physical_row(fragment_row),.in_output_channel(fragment_channel),
        .in_row_byte_base(fragment_row_base),.in_data(fragment_data),.in_byte_enable(fragment_be),
        .residual_read_request_valid,
        .residual_ddr_stage(output_residual_ddr_stage),
        .residual_read_request_ready,
        .residual_read_request_address,.residual_read_request_bytes,
        .residual_read_request_tag,.residual_read_local,
        .residual_read_request_done,.residual_read_request_error,
        .residual_read_data_valid,.residual_read_data_ready,.residual_read_data,
        .residual_read_byte_enable,.residual_read_data_last,.residual_read_data_tag,
        .arithmetic_req_valid(shared_bf16_req_valid),
        .arithmetic_req_ready(shared_bf16_req_ready),
        .arithmetic_req_operation(shared_bf16_req_operation),
        .arithmetic_req_values(shared_bf16_req_values),
        .arithmetic_req_paired_values(shared_bf16_req_paired_values),
        .arithmetic_req_factor0_values(shared_bf16_req_factor0_values),
        .arithmetic_req_factor1_values(shared_bf16_req_factor1_values),
        .arithmetic_req_lane_mask(shared_bf16_req_lane_mask),
        .arithmetic_req_tag(shared_bf16_req_tag),
        .arithmetic_rsp_valid(shared_bf16_rsp_valid),
        .arithmetic_rsp_ready(shared_bf16_rsp_ready),
        .arithmetic_rsp_values(shared_bf16_rsp_values),
        .arithmetic_rsp_lane_mask(shared_bf16_rsp_lane_mask),
        .arithmetic_rsp_tag(shared_bf16_rsp_tag),
        .arithmetic_abort_ack(shared_bf16_abort_ack),
        .local_write_valid(local_valid),.local_write_ready(local_ready),
        .local_write_byte_address(local_addr),.local_write_data(local_data),
        .local_write_byte_enable(local_be),
        .output_write_valid,.output_write_ready,
        .output_write_byte_address(output_write_address),
        .output_write_transaction_bytes,.output_write_data,
        .output_write_byte_enable,.output_write_first,.output_write_last,
        .output_write_done,.output_write_error,
        .staging_write_valid(combine_write_valid),
        .staging_write_ready(combine_write_ready),
        .staging_write_address(combine_write_address_i),
        .staging_write_data(combine_write_data),
        .staging_write_byte_enable(combine_write_byte_enable),
        .staging_read_request_valid(combine_read_valid),
        .staging_read_request_ready(combine_read_ready),
        .staging_read_address(combine_read_address_i),
        .staging_read_response_valid(combine_read_response_valid),
        .staging_read_data(combine_read_data),
        .row_commit(output_row_commit),
        .done_pulse(output_path_done_pulse),.busy(),
        .error(output_error),.error_id());
    assign combine_write_address = combine_write_address_i;
    assign combine_read_address = combine_read_address_i;
    assign local_output_write_valid=local_valid;
    assign local_output_write_address=local_addr[31:0];
    assign local_output_write_data=local_data; assign local_output_write_byte_enable=local_be;
    assign local_ready=local_output_write_ready;
    assign output_write_spill = output_mode == 3'd1 ||
        output_mode == 3'd2 || output_mode == 3'd4;
    assign output_write_physical_row = 6'd0;

    // A retained prefetch descriptor may outlive its reads. Only an actual
    // weight request takes priority over activation or residual source reads;
    // the router retains any already accepted transaction until it drains.
    always_comb begin
        memory_stage = MEMORY_STAGE_NONE;
        if (dma_req_valid) begin
            memory_stage = MEMORY_STAGE_WEIGHT;
        end else if (state != IDLE && state != FINISH) begin
            if ((source_ddr &&
                 (state == SEG_CFG || state == ROW_START ||
                  state == ROW_WAIT || state == SEG_WAIT)) ||
                output_residual_ddr_stage)
                memory_stage = MEMORY_STAGE_AUXILIARY;
            else
                memory_stage = MEMORY_STAGE_WEIGHT;
        end
    end

    always_ff @(posedge clk) begin
      if(rst) begin state<=IDLE; abort_ack<=0; seg_index<=0; current_row<=0;
        current_batch_rows<='0;
        next_load_stripe<=0;current_stripe<=0;current_panel_valid<=0;current_panel<=0;
        next_load_weight_base<=0;next_load_weight_enhancement<=0;
        next_load_weight_scale<=0;
        next_load_up_weight_base<=0;next_load_up_weight_enhancement<=0;
        next_load_up_weight_scale<=0;fused_product<=1'b0;
        pending_release<=0;completed_rows<=0;abort_client_active<=0;
        abort_client_pending<=0;abort_started<=0;schedule_abort_complete<=0;
        command_error<=0;completion_error<=0;rescale_inflight<=0;rescale_claimed<=0;
        configuration_valid_reg<=1'b0;
        accepted_prefetch_present<=1'b0;accepted_prefetch_match<=1'b0;
        local_stride_product_low<='0;local_stride_product_high<='0;
        configured_row_end<='0;configured_output_end<='0;
        configured_output_bytes<='0;configured_output_stripes<='0;
        configured_activation_end<='0;configured_local_output_end<='0;
        configured_segment_valid<=1'b0;
        active_operator_tag<=16'd0;
        prefetch_descriptor_valid<=1'b0;prefetch_submitted<=1'b0;
        prefetch_failed<=1'b0;prefetch_cancel_pending<=1'b0;
        saved_prefetch_operator_tag<=16'd0;saved_prefetch_input_features<=16'd0;
        saved_prefetch_weight_base<=64'd0;saved_prefetch_weight_enhancement<=64'd0;
        saved_prefetch_weight_scale<=64'd0;saved_prefetch_has_w8<=1'b0;
        saved_prefetch_qkv_layout<=1'b0;
        output_mode<=3'd0;source_base<=64'd0;residual_storage_rows<=6'd0;
        use_resident_activation<=1'b0;use_precomputed_max<=1'b0;
        attention_pair_query<=1'b0;attention_pair_query_second<=1'b0;
        kv_batch_pair<=1'b0;
        attention_pair_query_prepared<='0;
        quant_scale_req_clip_ratio_bf16 <= 16'd0;
        qkv_w8_layout<=1'b0;
        qkv_query_command<=1'b0;
        resident_activation_valid<=1'b0;
        quantization_batch_index<=3'd0;
        fragment_batch_index<=3'd0;normalized_source_end<='0;
      end else begin
        abort_ack<=0;
        if(activation_scale_valid && activation_scale_ready)
          for(integer scale_row=0;scale_row<8;scale_row=scale_row+1)
            if(activation_scale_row_mask[scale_row]) begin
              case (scale_write_batch_index)
                3'd1: second_activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
                3'd2: third_activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
                3'd3: fourth_activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
                3'd4: fifth_activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
                3'd5: sixth_activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
                default: activation_scales[activation_scale_row_base+scale_row]
                    <= activation_scale_values[scale_row*16+:16];
              endcase
            end
        if(result_valid && result_ready) fragment_batch_index<=result_batch_index;
        abort_client_active<=abort_client_active_next;
        if(state==IDLE || (start_valid&&start_ready) || child_abort_request)
          rescale_claimed<=1'b0;
        else begin
          if(shared_rescale_claim_valid&&shared_rescale_claim_ready)
            rescale_claimed<=1'b1;
          if(issue_valid&&issue_ready&&issue_first)
            rescale_claimed<=1'b0;
        end
        if(schedule_abort_ack) schedule_abort_complete<=1'b1;
        if(rescale_req_valid&&rescale_req_ready) begin
          rescale_inflight<=1'b1;end
        if((rescale_rsp_valid&&rescale_rsp_ready) || shared_rescale_abort_ack) begin
          rescale_inflight<=1'b0;end
        if(prefetch_valid&&prefetch_ready) begin
          prefetch_descriptor_valid<=1'b1;prefetch_submitted<=1'b0;
          prefetch_failed<=1'b0;
          saved_prefetch_operator_tag<=prefetch_operator_tag;
          saved_prefetch_input_features<=command_prefetch_input_features;
          saved_prefetch_weight_base<=command_prefetch_weight_base;
          saved_prefetch_weight_enhancement<=command_prefetch_weight_enhancement;
          saved_prefetch_weight_scale<=command_prefetch_weight_scale;
          saved_prefetch_has_w8<=command_prefetch_operator_has_w8;
          saved_prefetch_qkv_layout<=command_prefetch_qkv_w8_layout;
        end
        if(prefetch_cancel&&prefetch_descriptor_valid) begin
          if(prefetch_submitted) prefetch_cancel_pending<=1'b1;
          else begin
            prefetch_descriptor_valid<=1'b0;prefetch_submitted<=1'b0;
            prefetch_failed<=1'b0;
          end
        end
        if(loader_req&&loader_req_ready) begin
          if(loader_req_prefetch) prefetch_submitted<=1'b1;
          else begin
            next_load_stripe<=next_load_stripe+1;
            if(fused_product && next_load_stripe[0]) begin
              next_load_up_weight_base<=next_load_up_weight_base+
                {46'd0,k_size,2'b00};
              next_load_up_weight_enhancement<=next_load_up_weight_enhancement+
                {46'd0,k_size,2'b00};
              next_load_up_weight_scale<=next_load_up_weight_scale+64'd16;
            end else begin
              next_load_weight_base<=next_load_weight_base+{46'd0,k_size,2'b00};
              next_load_weight_enhancement<=next_load_weight_enhancement+
                {46'd0,k_size,2'b00};
              next_load_weight_scale<=next_load_weight_scale+64'd16;
            end
          end
        end
        if(loader_done&&loader_error!=0&&prefetch_descriptor_valid&&
            prefetch_submitted) prefetch_failed<=1'b1;
        if(prefetch_cancel_pending&&loader_abort_ack) begin
          prefetch_cancel_pending<=1'b0;prefetch_descriptor_valid<=1'b0;
          prefetch_submitted<=1'b0;prefetch_failed<=1'b0;
        end
        if(acquire_valid&&acquire_ready) begin current_panel_valid<=1;current_panel<=acquire_panel;end
        if(issue_valid&&issue_ready&&issue_last_stripe) begin pending_release<=1;release_tag<=issue_tag;end
        if(release_valid&&release_ready) begin current_panel_valid<=0;pending_release<=0;
          if(!(abort_request|datapath_abort)) current_stripe<=current_stripe+1; end
        if(output_row_commit) completed_rows<=completed_rows+1;
        case(state)
          IDLE: if(start_valid&&start_ready) begin
            attention_pair_query<=command_attention_pair_query;
            kv_batch_pair<=command_kv_batch_pair;
            attention_pair_query_second<=command_attention_pair_query &&
                fixed_ffn_batch_pair.prepare_second;
            if (!command_attention_pair_query ||
                (!fixed_ffn_batch_pair.prepare_second && !command_use_resident_activation))
                attention_pair_query_prepared<='0;
            command_error<=1'b0;
            completion_error<=1'b0;
            schedule_abort_complete<=1'b0;
            rows<=command_rows;k_size<=command_input_features;
            n_size<=command_output_features;
            active_operator_tag<=start_operator_tag;
            seg_count<=command_segment_count;seg_mode<=command_segment_mode;
            seg_base<=command_segment_row_base;seg_rows<=command_segment_row_count;
            seg_activation<=command_segment_activation_base_byte_offset;row_enable<=command_row_enable;
            mixed_groups<=command_mixed_group_enable;
            mixed_compute_group_count<=command_compute_group_count;
            mixed_row_precision_a8<=command_row_precision_a8;
            mixed_row_compute_group<=command_row_compute_group;
            mixed_row_pe_slot<=command_row_pe_slot;
            mixed_row_phase_mask<=command_row_phase_mask;
            source_ddr<=command_source_ddr;has_w8<=command_operator_has_w8;
            source_base<=command_source_base;
            fused_product<=command_fused_product;
            quantization_batch_index<=3'd0;
            output_mode<=command_output_mode;
            use_resident_activation<=command_use_resident_activation;
            use_precomputed_max<=command_use_precomputed_max;
            quant_scale_req_clip_ratio_bf16 <= command_clip_ratio_bf16;
            qkv_w8_layout<=command_qkv_w8_layout;
            qkv_query_command<=FIXED_SCHEDULE_INGRESS &&
                fixed_command == hardware_types_pkg::MATMUL_COMMAND_QKV &&
                fixed_command_qkv_select == 2'd0;
            weight_base<=command_weight_base;
            weight_enh<=command_weight_enhancement;
            weight_scale<=command_weight_scale;output_base<=command_output_base;
            next_load_up_weight_base<=fixed_weight_config.ffn_up_base;
            next_load_up_weight_enhancement<=
              fixed_weight_config.ffn_up_enhancement;
            next_load_up_weight_scale<=fixed_weight_config.ffn_up_scale;
            output_limit<=command_output_limit;
            output_stride<=command_output_row_stride;
            residual_storage_rows<=command_residual_storage_rows;
            accepted_prefetch_present<=prefetch_descriptor_valid;
            accepted_prefetch_match<=start_prefetch_match;
            state<=CONFIG_DERIVE;
          end
          CONFIG_DERIVE: begin
            normalized_source_end <= {1'b0,batch_pair.normalized_base} + (65'(total_output_rows) << 13);
            configured_segment_valid<=derived_segment_valid;
            configured_row_end<=derived_row_end;
            configured_activation_end<=derived_activation_end;
            configured_output_stripes<=derived_output_stripes;
            configured_output_bytes<=derived_output_bytes;
            local_stride_product_low<=
              (22'(rows) + ((output_mode == 3'd0 && batch_pair.enable) ?
                22'(batch_pair.second_rows) : 22'd0) - 22'd1) * 22'(output_stride[15:0]);
            local_stride_product_high<=
              (22'(rows) + ((output_mode == 3'd0 && batch_pair.enable) ?
                22'(batch_pair.second_rows) : 22'd0) - 22'd1) * 22'(output_stride[31:16]);
            state<=CONFIG_RANGE;
          end
          CONFIG_RANGE: begin
            configured_local_output_end<=
              {7'd0,output_base[31:0]}+
              {1'b0,local_stride_product_high,16'd0}+
              {{17{1'b0}},local_stride_product_low}+
              (39'(n_size)<<1);
            configured_output_end<=
              {1'b0,output_base}+65'(configured_output_bytes);
            state<=CONFIG_CHECK;
          end
          CONFIG_CHECK: begin
            configuration_valid_reg<=
              TRUSTED_FIXED_CONFIGURATION && FIXED_SCHEDULE_INGRESS ?
                1'b1 : configuration_valid;
            state<=CONFIG_DISPATCH;
          end
          CONFIG_DISPATCH: begin
            command_error<=!configuration_valid_reg ||
              (accepted_prefetch_present &&
               (!accepted_prefetch_match || !prefetch_descriptor_valid ||
                prefetch_failed || prefetch_cancel_pending));
            if(!configuration_valid_reg) begin
              state<=FINISH;
            end else if(accepted_prefetch_present &&
                    (!accepted_prefetch_match || !prefetch_descriptor_valid ||
                     prefetch_failed || prefetch_cancel_pending)) begin
              if(prefetch_descriptor_valid) prefetch_cancel_pending<=1'b1;
              state<=ERROR_WAIT;
            end else begin
            seg_index<=0;current_row<=seg_base[5:0];
            current_batch_rows<=rows_up_to_eight(seg_rows[5:0]);
            next_load_stripe<=accepted_prefetch_present ? 16'd1 : 16'd0;
            next_load_weight_base<=weight_base+
              (accepted_prefetch_present ? {46'd0,k_size,2'b00} : 64'd0);
            next_load_weight_enhancement<=weight_enh+
              (accepted_prefetch_present ? {46'd0,k_size,2'b00} : 64'd0);
            next_load_weight_scale<=weight_scale+
              (accepted_prefetch_present ? 64'd16 : 64'd0);
            current_stripe<=0;current_panel_valid<=0;
            if(accepted_prefetch_present) begin
              prefetch_descriptor_valid<=1'b0;prefetch_submitted<=1'b0;
              prefetch_failed<=1'b0;
            end
            stripe_count<=fused_product ?
              {1'b0,configured_output_stripes,1'b0} :
              {2'b00,configured_output_stripes};completed_rows<=0;
            if(use_resident_activation) state<=OUTPUT_CFG;
            else begin resident_activation_valid<=1'b0;state<=SEG_CFG;end
            end
          end
          SEG_CFG: if(activation_writer_cfg_valid&&activation_writer_cfg_ready) begin
            current_row<=writer_all_rows ? 6'd0 : get_seg_base(seg_index);
            current_batch_rows<=rows_up_to_eight(writer_all_rows ?
              quantization_rows : get_seg_rows(seg_index));
            state<=ROW_START;
          end
          ROW_START: if(frontend_start&&frontend_ready) state<=ROW_WAIT;
          ROW_WAIT: if(frontend_done_pulse) begin
            if(current_row+current_batch_rows>=
               (writer_all_rows ? quantization_rows :
                get_seg_base(seg_index)+get_seg_rows(seg_index))) state<=SEG_WAIT;
            else begin
              current_row<=current_row+current_batch_rows;
              current_batch_rows<=rows_up_to_eight(
                (writer_all_rows ? quantization_rows :
                 get_seg_base(seg_index)+get_seg_rows(seg_index))-
                (current_row+current_batch_rows));
              state<=ROW_START;
            end
          end
          SEG_WAIT: if(activation_writer_done_pulse) begin if(!writer_all_rows &&
              seg_index+1<seg_count) begin
            seg_index<=seg_index+1;current_row<=get_seg_base(seg_index+1);state<=SEG_CFG;
            end else if (batch_pair.enable &&
                         {1'b0, quantization_batch_index} + 3'd1 <
                         batch_pair.batch_count) begin
              quantization_batch_index<=quantization_batch_index+3'd1;
              state<=SEG_CFG;
            end else begin
              resident_activation_valid<=1'b1;
              if (attention_pair_query)
                attention_pair_query_prepared[attention_pair_query_second]<=1'b1;
              if (!attention_pair_query || !attention_pair_query_second) begin
              resident_rows<=rows;
              resident_batch_pair<=attention_pair_query ? fixed_ffn_batch_pair : batch_pair;
              resident_input_features<=k_size;
              resident_segment_count<=seg_count;
              resident_segment_mode<=seg_mode;
              resident_segment_row_base<=seg_base;
              resident_segment_row_count<=seg_rows;
              resident_segment_activation_base_byte_offset<=seg_activation;
              resident_row_enable<=row_enable;
              resident_mixed_groups<=mixed_groups;
              resident_compute_group_count<=mixed_compute_group_count;
              resident_row_precision_a8<=mixed_row_precision_a8;
              resident_row_compute_group<=mixed_row_compute_group;
              resident_row_pe_slot<=mixed_row_pe_slot;
              resident_row_phase_mask<=mixed_row_phase_mask;
              end
              state<=OUTPUT_CFG;
            end end
          OUTPUT_CFG: if(output_cfg_valid&&output_cfg_ready) state<=COMPUTE_START;
          COMPUTE_START: if(schedule_start&&schedule_ready) state<=COMPUTE;
          COMPUTE: if(schedule_done&&schedule_done_ready) state<=FINISH;
          FINISH: if(done_ready) begin
            // The following elementwise command publishes Down scales to
            // batch zero, including when fused Gate/Up skips the Up command.
            quantization_batch_index<=3'd0;
            state<=IDLE;
          end
          ABORT_WAIT: if((schedule_abort_ack || schedule_abort_complete) &&
              abort_client_pending==0) begin
            abort_ack<=1;abort_started<=0;schedule_abort_complete<=0;
            abort_client_active<=0;
            prefetch_descriptor_valid<=1'b0;prefetch_submitted<=1'b0;
            prefetch_failed<=1'b0;prefetch_cancel_pending<=1'b0;
            state<=IDLE;end
          ERROR_WAIT: if((schedule_abort_ack || schedule_abort_complete) &&
              abort_client_pending==0) begin
            abort_started<=0;schedule_abort_complete<=0;
            abort_client_active<=0;
            prefetch_descriptor_valid<=1'b0;prefetch_submitted<=1'b0;
            prefetch_failed<=1'b0;prefetch_cancel_pending<=1'b0;
            state<=FINISH;end
          default: state<=IDLE;
        endcase
        if(child_abort_request) begin
          if(!abort_started) begin
            abort_started<=1'b1;
            abort_client_pending<=abort_client_active_next & ~abort_client_clear_event;
          end else begin
            abort_client_pending<=abort_client_pending & ~abort_client_clear_event;
          end
        end
        if(abort_request&&state!=IDLE&&state!=ABORT_WAIT) state<=ABORT_WAIT;
        else if(loader_done&&loader_error!=0&&
            !(prefetch_descriptor_valid&&prefetch_submitted)&&
            state!=IDLE&&state!=FINISH&&
            state!=ABORT_WAIT&&state!=ERROR_WAIT) begin
          command_error<=1'b1;state<=ERROR_WAIT;end
        else if(aux_stream_error&&state!=IDLE&&state!=FINISH&&
            state!=ABORT_WAIT&&state!=ERROR_WAIT) begin
          command_error<=1'b1;state<=ERROR_WAIT;end
        else if(output_error&&state!=IDLE&&state!=FINISH&&
            state!=ABORT_WAIT&&state!=ERROR_WAIT) begin
          command_error<=1'b1;state<=ERROR_WAIT;end
        if(state!=IDLE&&state!=FINISH&&
            (command_error||frontend_error||activation_writer_error||
             output_error||(loader_done&&loader_error!=0)||
             (state==CONFIG_DISPATCH&&!configuration_valid_reg)))
          completion_error<=1'b1;
        if(resident_activation_commit_valid) begin
          attention_pair_query_prepared<='0;
          resident_batch_pair<=fixed_ffn_batch_pair.enable ? fixed_ffn_batch_pair : '0;
          resident_batch_pair.prepare_second<=1'b0;
          resident_activation_valid<=!fixed_ffn_batch_pair.enable ||
              fixed_ffn_batch_pair.prepare_second;
          if (!fixed_ffn_batch_pair.prepare_second) begin
          resident_rows<=resident_activation_commit_rows;
          resident_input_features<=resident_activation_commit_features;
          resident_segment_count<=resident_activation_commit_segment_count;
          resident_segment_mode<=resident_activation_commit_segment_mode;
          resident_segment_row_base<=resident_activation_commit_segment_row_base;
          resident_segment_row_count<=resident_activation_commit_segment_row_count;
          resident_segment_activation_base_byte_offset<=resident_activation_commit_segment_activation_base_byte_offset;
          resident_row_enable<=resident_activation_commit_row_enable;
          // Elementwise can publish a different batch from the last Gate/Up.
          resident_mixed_groups<=FIXED_SCHEDULE_INGRESS && fixed_command_row_config.mixed_group_enable;
          resident_compute_group_count<=fixed_command_row_config.compute_group_count;
          resident_row_precision_a8<=fixed_command_row_config.row_precision_a8;
          resident_row_compute_group<=fixed_command_row_config.row_compute_group;
          resident_row_pe_slot<=fixed_command_row_config.row_pe_slot;
          resident_row_phase_mask<=fixed_command_row_config.row_phase_mask;
          end
        end
      end
    end
    assign datapath_abort_ack=abort_started && abort_client_pending==0;

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(prefetch_cancel && loader_req_prefetch && loader_req_ready))
                else $error("matmul_controller accepted a prefetched panel load while canceling it");
            if (batch_pair.enable && k_size == 16'd12288) begin
                if (panel_memory_write_valid)
                    assert (!panel_memory_write_panel)
                        else $error("paired Down wrote the overlapping weight panel 1");
                if (acquire_valid)
                    assert (!acquire_panel)
                        else $error("paired Down acquired the overlapping weight panel 1");
            end
            if (activation_scale_valid && activation_scale_ready)
                assert (activation_scale_row_base < 6'd48 &&
                        activation_scale_row_base + $countones(
                            activation_scale_row_mask) <= 48)
                    else $error("Matmul activation scale batch exceeds the 48-row table");
            if (done_valid && !error)
                assert (completed_rows == total_output_rows && !output_write_valid && !local_valid &&
                    !fragment_valid && !result_valid)
                    else $error("mixed Matmul completed before all output fragments drained");
            if (state == FINISH && !command_error)
                assert (completed_rows == total_output_rows)
                    else $error("mixed Matmul entered FINISH before every output row was accepted");
            if (frontend_done_pulse)
                assert (state == ROW_WAIT || abort_started)
                    else $error("Matmul activation completion pulse had no active parent wait state");
            if (output_path_done_pulse)
                assert (state == COMPUTE && completed_rows == total_output_rows)
                    else $error("Matmul output completion pulse was not consumed by schedule completion");
            if (state == COMPUTE && $past(state) != COMPUTE)
                assert ($past(state) == COMPUTE_START)
                    else $error("Matmul entered compute before schedule start was accepted");
            if ($past(done_valid && !done_ready))
                assert (done_valid && $stable(error))
                    else $error("matmul_controller changed a stalled completion response");
        end
    end
`endif
endmodule

`default_nettype wire
