`default_nettype none

// Boundary selection scans one position-ordered table. Each scan is an ordered
// stream supplied by the shared DMA path; only the selected-position bitmap is
// retained here. Stable ties use the original eligible/required list indices.
module attention_guided_token_selector (
    input logic clk,
    input logic rst,
    input logic abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    output logic packer_abort_request,
    input logic packer_abort_ack,
    output logic packer_start_valid,
    input logic packer_start_ready,
    output hardware_types_pkg::token_metadata_config_t packer_config,
    output logic packer_row_valid,
    input logic packer_row_ready,
    output hardware_types_pkg::token_metadata_row_t packer_row,
    input logic packer_beat_valid,
    output logic packer_beat_ready,
    input logic [127:0] packer_beat_data,
    input logic packer_beat_last,
    input logic packer_done_valid,
    output logic packer_done_ready,
    input logic packer_error,
    input logic [7:0] packer_error_id,
    input logic [5:0] packer_compute_groups,
    input logic [5:0] packer_semantic_groups,
    input logic [63:0] start_descriptor_address,
    input logic [63:0] start_joint_descriptor_address,
    input logic start_prepare,
    input logic start_relation_only,
    input logic start_relation_l31,
    input logic start_closeout,
    output logic prepared_for_forward,
    input logic loaded_token_valid, loaded_token_kv_write,
    input logic [10:0] loaded_token_position,
    input logic consume_source_a,
    output logic source_a_valid,
    output logic [31:0] source_a_mask,
    output logic [10:0] source_a_block_start,
    output logic [31:0] source_a_capture_index,
    output hardware_types_pkg::attention_probability_config_t probability_config,
    output logic [31:0] relation_layer_mask,
    output logic bf16_abort_request,
    input logic bf16_abort_ack,
    output logic bf16_req_valid,
    input logic bf16_req_ready,
    output hardware_types_pkg::bf16_request_t bf16_req,
    input logic bf16_rsp_valid,
    output logic bf16_rsp_ready,
    input hardware_types_pkg::bf16_response_t bf16_rsp,
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

    output logic metadata_scratch_req_valid,
    input logic metadata_scratch_req_ready,
    output logic metadata_scratch_write,
    output logic [8:0] metadata_scratch_address,
    output logic [58:0] metadata_scratch_write_data,
    output logic [58:0] metadata_scratch_write_enable,
    input logic metadata_scratch_rsp_valid,
    input logic [58:0] metadata_scratch_rsp_data,
    output logic metadata_scratch_aux_write_valid,
    input logic metadata_scratch_aux_write_ready,
    output logic [8:0] metadata_scratch_aux_write_address,
    output logic [19:0] metadata_scratch_aux_write_data,
    output logic [19:0] metadata_scratch_aux_write_enable,

    output logic selected_valid,
    input logic selected_ready,
    output logic [10:0] selected_position,
    output logic selected_last,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id
);
    import token_refresh_config_pkg::*;
    import attention_probability_config_pkg::*;
    import uaps_config_pkg::*;
    typedef enum logic [6:0] {
        IDLE, CLEAR, SCAN_START, SCAN_ROWS, SCAN_DRAIN, CHECK_INITIAL,
        CHOOSE, OUTPUT_LOAD, OUTPUT_ROWS, COMPLETE, ABORT_DRAIN, ABORT_LOW,
        COMMIT_START, COMMIT_WAIT, METADATA_WAIT, CONFIG_START, CONFIG_WAIT, CONFIG_DECODE,
        WAIT_FORWARD, RELATION_START, RELATION_WAIT, PROBABILITY_RANGE, PROBABILITY_START, PROBABILITY_WAIT, PROBABILITY_CHECK,
        JOINT_CONFIG_START, JOINT_CONFIG_WAIT, JOINT_RANGE, JOINT_CHECK, JOINT_START,
        JOINT_BASE_START, JOINT_BASE_WAIT, JOINT_FUTURE_START, JOINT_FUTURE_WAIT,
        JOINT_WAIT, JOINT_MARK, PENDING_RANGE, PENDING_CONFIG_START, PENDING_CONFIG_WAIT,
        PENDING_CHECK, PENDING_START, PENDING_WAIT, BUDGET_RANGE, BUDGET_START, BUDGET_WAIT, BUDGET_CHECK,
        REGULAR_LIMIT, REGULAR_FINISH, JOINT_SELECTED_LOAD, JOINT_SELECTED_FIND,
        JOINT_SELECTED_START, JOINT_SELECTED_WAIT,
        SHORTLIST_CONFIG_START, SHORTLIST_CONFIG_WAIT, SHORTLIST_CHECK, SHORTLIST_START, SHORTLIST_WAIT,
        PRECISION_BEGIN, PRECISION_READ, PRECISION_SCAN, PRECISION_PICK, PRECISION_DONE, JOINT_FUTURE_STATE,
        JOINT_EXTENSION_START, JOINT_EXTENSION_WAIT, ATTEMPT_START, ATTEMPT_WAIT, ATTEMPT_CHECK,
        JOINT_STATE_BEGIN, JOINT_CURRENT_COUNT, JOINT_HANDOFF_COUNT, CONSUMED_CLEAR,
        PENDING_SCOPE_START, PENDING_SCOPE_WAIT, PENDING_NEXT,
        EXECUTION_EXTENSION_START, EXECUTION_EXTENSION_WAIT, EXECUTION_EXTENSION_CHECK
    } state_t;
    state_t state;
    logic [15:0] probability_offset;
    logic [64:0] probability_address;
    logic probability_reserved_error;
    logic [63:0] shortlist_descriptor_address;
    logic [255:0] shortlist_configuration;
    logic shortlist_active, shortlist_start_ready, shortlist_done, shortlist_error, shortlist_abort_ack;
    logic [7:0] shortlist_error_id;
    logic short_multiply_valid, short_multiply_result_ready;
    logic [15:0] short_multiply_lhs, short_multiply_rhs;
    logic [11:0] short_target_token_count;
    logic short_request_valid, short_read_ready, short_write_request_valid, short_write_valid, short_write_last;
    logic [63:0] short_request_address, short_write_request_address;
    logic [31:0] short_request_bytes, short_write_request_bytes;
    logic [7:0] short_request_tag, short_write_request_tag;
    logic [127:0] short_write_data;
    logic [15:0] short_write_byte_enable;
    logic [63:0] joint_descriptor_address;
    logic [511:0] joint_configuration;
    logic [255:0] joint_extension;
    // DDR state: 16-byte header + 32 x 32-bit counters. One 16-byte beat is
    // loaded per cycle; selection reads one counter, marking updates one.
    logic [127:0] attempt_header;
    logic [31:0] attempt_counts [0:31];
    logic [63:0] attempt_base;
    logic [64:0] attempt_end;
    logic attempt_enabled, attempt_same_block, attempt_replay, attempt_record_error;
    logic [4:0] attempt_future_index;
    logic [31:0] attempt_count, attempt_limit;
    logic [15:0] retry_min_confidence, future_last_confidence;
    logic future_last_confidence_valid, attempt_selected_replay, future_admission_allowed;
    logic pending_source_a_pending;
    logic capture_consumed, live_consumed_enabled;
    // One cross-block update followed by one in-block update, using the same
    // pending arithmetic and DMA. Only the second update drives selection.
    logic pending_regular_phase;
    logic paired_pending;
    logic pending_scope_loaded, pending_scope_invalid;
    logic [11:0] consume_current_begin;
    logic [12:0] consume_current_end;
    logic [1151:0] attempt_write_payload;
    logic [64:0] joint_base_end, joint_future_end, joint_result_end;
    logic [63:0] joint_base_address, joint_future_address, joint_result_address;
    logic [5:0] joint_base_rows, joint_future_rows, joint_max_next;
    logic [5:0] regular_base_rows, joint_base_index;
    logic [10:0] joint_next_start;
    logic joint_running, joint_start_ready, joint_base_ready, joint_future_ready;
    logic joint_state_input, pending_state_valid;
    logic joint_state_control, joint_current_allowed, joint_allow_new, joint_deferred;
    logic [5:0] current_unresolved_count, future_tentative_count;
    logic joint_base_a8, joint_base_forecast_a8;
    logic [2:0] pending_state_value;
    logic [15:0] pending_state_confidence;
    logic pending_state_confidence_valid;
    logic joint_done, joint_error, joint_abort_ack;
    logic joint_add_valid, joint_add_abort;
    logic [15:0] joint_add_lhs, joint_add_rhs;
    logic [31:0] joint_progress_mask, joint_added_mask;
    logic [5:0] joint_progress_count, joint_added_count, joint_mark_index;
    logic [7:0] joint_base_units, joint_units, joint_forecast_units;
    logic joint_record_error, joint_bf16_error;
    logic metadata_embedding_source;
    logic [15:0] pending_offset;
    logic [64:0] pending_configuration_address;
    logic [639:0] pending_configuration;
    logic pending_active, pending_start_ready, pending_done, pending_error, pending_abort_ack;
    logic advance_prepared;
    logic [7:0] pending_error_id;
    logic pend_request_valid, pend_read_ready, pend_write_request_valid, pend_write_valid, pend_write_last;
    logic [63:0] pend_request_address, pend_write_request_address;
    logic [31:0] pend_request_bytes, pend_write_request_bytes;
    logic [7:0] pend_request_tag, pend_write_request_tag;
    logic [127:0] pend_write_data;
    logic [15:0] pend_write_byte_enable;
    logic pend_arithmetic_valid, pend_arithmetic_ready, pend_multiply, pend_arithmetic_abort, pending_bf16_error;
    logic [15:0] pend_lhs, pend_rhs;
    logic pending_score_valid, pending_score_mandatory, regular_enabled;
    logic [11:0] pending_score_row;
    logic [15:0] pending_score_bf16;
    // 17 bits x 96 = 204 B; captures each produced score once and serves the
    // repeated stable-selection scans without rereading the pending vector.
    logic [16:0] regular_scores [0:95];
    logic [16:0] regular_record;
    logic precision_enabled, all_selected_tokens_a8, deep_precision_enabled;
    logic [6:0] precision_quota, token_precision_count, token_precision_index, selected_token_precision_index;
    // Three 96-bit masks share the serial score scan; no score payload is copied.
    logic [95:0] prediction_token_mask, a8_token_mask, selected_token_mask;
    logic selected_token_precision_valid, selected_token_a8;
    logic [127:0] regular_budget;
    logic signed [32:0] regular_available_half;
    logic signed [32:0] regular_final_credit;
    logic [11:0] regular_begin, regular_end;
    logic [64:0] regular_budget_end;
    logic [63:0] scan_address;
    logic [31:0] scan_request_bytes;
    logic [11:0] sequence_length, target_token_count, required_quota;
    logic initial_scan, required_scan;
    logic [11:0] input_index;
    logic [5:0] clear_word, output_word;
    logic [4:0] output_bit;
    logic [31:0] output_bits;
    logic [11:0] output_count, selected_count;
    logic [11:0] current_count, eligible_count, required_count, mandatory_required;
    logic [11:0] remaining_required;
    logic [31:0] selected_words [0:7][0:7];
    logic [31:0] read_bank_word [0:7];
    logic [5:0] read_word;
    logic [31:0] read_selected_word;
    logic mark_valid;
    logic [10:0] mark_position;
    logic row_pending, pending_last, pending_current, pending_eligible;
    logic pending_mandatory, pending_required, pending_selected;
    logic [15:0] pending_score;
    logic [10:0] pending_position, pending_order;
    logic best_valid;
    logic [15:0] best_selection_key;
    logic [10:0] best_position, best_order;
    logic candidate_valid, candidate_better;
    logic terminal_error, aborting;
    logic [7:0] terminal_error_id;
    logic [63:0] table_base;
    logic [64:0] table_end;
    logic scan_valid, scan_ready, scan_error, scan_abort_ack, scan_abort_request;
    logic row_valid, row_ready, row_last;
    logic [63:0] row_record;
    logic [15:0] row_score;
    logic row_current, row_eligible, row_mandatory, row_required;
    logic [10:0] row_eligible_order, row_required_order;
    logic reader_valid, reader_ready, reader_last, reader_done, reader_error, reader_abort_ack;
    logic [127:0] reader_data;
    logic [15:0] reader_byte_enable;
    logic [31:0] reader_offset;
    logic [31:0] scan_bytes;
    logic commit_cache, commit_active, commit_start_ready;
    logic commit_done, commit_error, commit_abort_ack;
    logic [7:0] commit_error_id;
    logic [383:0] cache_bases, cache_limits;
    logic [31:0] head_stride, scale_head_stride;
    logic table_request_valid, table_read_ready;
    logic [63:0] table_request_address;
    logic [31:0] table_request_bytes;
    logic [7:0] table_request_tag;
    logic commit_request_valid, commit_read_ready;
    logic [63:0] commit_request_address;
    logic [31:0] commit_request_bytes;
    logic [7:0] commit_request_tag;
    logic output_row_valid, output_row_ready;
    logic output_selected_accepted, output_metadata_accepted;
    logic write_metadata, metadata_active, metadata_start_ready, metadata_position_ready;
    logic metadata_position_table_ready;
    logic commit_position_read_req_valid, commit_position_read_req_ready;
    logic [8:0] commit_position_read_req_index;
    logic commit_position_read_rsp_valid;
    logic [19:0] commit_position_read_rsp_data;
    logic short_scratch_req_valid, short_scratch_write;
    logic [8:0] short_scratch_address;
    logic [58:0] short_scratch_write_data, short_scratch_write_enable;
    logic writer_scratch_req_valid, writer_scratch_req_ready, writer_scratch_write;
    logic [8:0] writer_scratch_address;
    logic [58:0] writer_scratch_write_data, writer_scratch_write_enable;
    logic writer_scratch_rsp_valid;
    logic [58:0] writer_scratch_rsp_data;
    logic metadata_store_required;
    logic metadata_done, metadata_error, metadata_abort_ack;
    logic [7:0] metadata_error_id;
    logic [63:0] metadata_base, metadata_limit, table_limit;
    logic [31:0] metadata_version, capture_index;
    logic meta_request_valid, meta_read_ready;
    logic [63:0] meta_request_address;
    logic [31:0] meta_request_bytes;
    logic [7:0] meta_request_tag;
    logic meta_write_request_valid, meta_write_valid, meta_write_last;
    logic [63:0] meta_write_request_address;
    logic [31:0] meta_write_request_bytes;
    logic [7:0] meta_write_request_tag;
    logic [127:0] meta_write_data;
    logic [15:0] meta_write_byte_enable;
    logic commit_write_request_valid, commit_write_valid, commit_write_last;
    logic [63:0] commit_write_request_address;
    logic [31:0] commit_write_request_bytes;
    logic [7:0] commit_write_request_tag;
    logic [127:0] commit_write_data;
    logic [15:0] commit_write_byte_enable;
    logic [TOKEN_REFRESH_CONFIG_BYTES*8-1:0] configuration;
    logic [63:0] descriptor_address;
    logic [64:0] descriptor_end;
    logic [15:0] start_sequence_length, start_target_token_count, start_required_quota;
    logic [63:0] start_table_base, start_table_limit, start_metadata_base, start_metadata_limit;
    logic [383:0] start_cache_bases, start_cache_limits;
    logic [31:0] start_head_stride, start_scale_head_stride, start_metadata_version, start_capture_index;
    logic start_commit_cache, start_write_metadata;
    logic configuration_invalid;
    logic prepare_only;
    logic [255:0] execution_extension;
    logic [64:0] l31_relation_job_end;
    logic [15:0] relation_job_count;
    logic [63:0] relation_job_base;
    logic [64:0] relation_job_end, configured_job_end;
    logic relation_active, relation_start_ready, relation_done, relation_error, relation_abort_ack;
    logic relation_only, relation_seen, relation_per_layer;
    logic saved_closeout, regular_closed;
    logic [31:0] writer_metadata_bytes;
    logic [3:0] writer_metadata_rounds;
    assign metadata_bytes = regular_closed ? 32'd0 : writer_metadata_bytes;
    assign metadata_rounds = regular_closed ? 4'd0 : writer_metadata_rounds;
    assign paired_pending = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+7];
    assign consume_current_end = {1'b0, consume_current_begin} + 13'd32;
    assign live_consumed_enabled = |pending_configuration[127:126];
    // Initial-key boundary consumption is selected_optional, including selected
    // future tokens, irrespective of K/V suppression. Regular initial-key steps
    // leave both bits clear and do not consume a query-derived set.
    assign capture_consumed = prepared_for_forward && loaded_token_valid &&
        (pending_configuration[126] ? loaded_token_kv_write :
         pending_configuration[127] && ({1'b0, loaded_token_position} < consume_current_begin ||
            {2'b0, loaded_token_position} >= consume_current_end));

    assign relation_layer_mask = prepared_for_forward && probability_config.enable &&
        relation_per_layer && relation_job_count != 0 ? probability_config.layer_mask : 32'd0;
    logic [7:0] relation_error_id;
    logic rel_request_valid, rel_read_ready, rel_write_request_valid, rel_write_valid, rel_write_last;
    logic [63:0] rel_request_address, rel_write_request_address;
    logic [31:0] rel_request_bytes, rel_write_request_bytes;
    logic [7:0] rel_request_tag, rel_write_request_tag;
    logic [127:0] rel_write_data;
    logic [15:0] rel_write_byte_enable;
    logic rel_dequant_valid, rel_dequant_ready, rel_dequant_abort, rel_dequant_error;
    logic [127:0] rel_dequant_values;
    logic [15:0] rel_dequant_scale;
    logic [7:0] rel_dequant_mask;
    assign configured_job_end = {1'b0, configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET*8 +: 64]} +
        {43'd0, configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET*8 +: 16], 6'd0};
    assign l31_relation_job_end = {1'b0, execution_extension[63:0]} + {43'd0, execution_extension[79:64], 6'd0};
    assign relation_job_end = {1'b0, relation_job_base} + {43'd0, relation_job_count, 6'd0};
    assign descriptor_end = {1'b0, start_descriptor_address} + 65'(TOKEN_REFRESH_CONFIG_BYTES);
    assign start_sequence_length = configuration[TOKEN_REFRESH_CONFIG_SEQUENCE_LENGTH_OFFSET*8 +: 16];
    assign start_target_token_count = configuration[TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET*8 +: 16];
    assign start_required_quota = configuration[TOKEN_REFRESH_CONFIG_REQUIRED_QUOTA_OFFSET*8 +: 16];
    assign start_table_base = configuration[TOKEN_REFRESH_CONFIG_TABLE_BASE_OFFSET*8 +: 64];
    assign start_table_limit = configuration[TOKEN_REFRESH_CONFIG_TABLE_LIMIT_OFFSET*8 +: 64];
    assign start_metadata_base = configuration[TOKEN_REFRESH_CONFIG_METADATA_BASE_OFFSET*8 +: 64];
    assign start_metadata_limit = configuration[TOKEN_REFRESH_CONFIG_METADATA_LIMIT_OFFSET*8 +: 64];
    assign start_metadata_version = configuration[TOKEN_REFRESH_CONFIG_METADATA_VERSION_OFFSET*8 +: 32];
    assign start_capture_index = configuration[TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET*8 +: 32];
    assign start_cache_bases = configuration[TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET*8 +: 384];
    assign start_cache_limits = configuration[TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET*8 +: 384];
    assign start_head_stride = configuration[TOKEN_REFRESH_CONFIG_HEAD_STRIDE_OFFSET*8 +: 32];
    assign start_scale_head_stride = configuration[TOKEN_REFRESH_CONFIG_SCALE_HEAD_STRIDE_OFFSET*8 +: 32];
    assign start_commit_cache = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8];
    assign start_write_metadata = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+1];
    assign configuration_invalid = configuration[TOKEN_REFRESH_CONFIG_MAGIC_OFFSET*8 +: 32] != 32'h31465241 ||
        configuration[TOKEN_REFRESH_CONFIG_VERSION_OFFSET*8 +: 16] != 16'd1 ||
        (configuration[TOKEN_REFRESH_CONFIG_BYTES_OFFSET*8 +: 16] != 16'(TOKEN_REFRESH_CONFIG_BYTES) &&
         configuration[TOKEN_REFRESH_CONFIG_BYTES_OFFSET*8 +: 16] != 16'd208) ||
        (paired_pending && (pending_offset == 0 || !start_write_metadata ||
            !configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+2] || start_commit_cache)) ||
        (configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+6] && !start_write_metadata) ||
        (configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+15] && !start_write_metadata) || precision_quota > 96 ||
        (!precision_enabled && (all_selected_tokens_a8 || (!deep_precision_enabled && precision_quota != 0))) ||
        (deep_precision_enabled && (precision_enabled || pending_offset != 0 || !start_write_metadata || configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+2])) ||
        (precision_enabled && pending_offset == 0) ||
        (pending_offset != 0 && (pending_offset < configuration[TOKEN_REFRESH_CONFIG_BYTES_OFFSET*8 +: 16] || pending_offset[3:0] != 0)) ||
        (probability_offset != 0 && (probability_offset < configuration[TOKEN_REFRESH_CONFIG_BYTES_OFFSET*8 +: 16] ||
            probability_offset[3:0] != 0 || !prepare_only)) ||
        configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET*8+16 +: 16] != 0 ||
        configured_job_end[64] || configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET*8 +: 4] != 0;

    assign table_end = {1'b0, start_table_base} + {46'd0, start_sequence_length, 3'd0};
    assign scan_bytes = {17'd0, sequence_length, 3'd0};
    assign row_record = input_index[0] ? reader_data[127:64] : reader_data[63:0];
    assign regular_begin = regular_budget[27:16];
    assign regular_final_credit = regular_available_half - $signed({20'd0, selected_count, 1'b0});
    assign regular_end = regular_begin + pending_configuration[75:64];
    assign precision_enabled = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+3];
    assign deep_precision_enabled = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+5];
    assign all_selected_tokens_a8 = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+4];
    assign precision_quota = configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+8 +: 7];
    assign token_precision_index = 7'(input_index-regular_begin);
    assign selected_token_precision_index = 7'({1'b0, selected_position}-regular_begin);
    assign selected_token_precision_valid = precision_enabled && {1'b0, selected_position} >= regular_begin &&
        {1'b0, selected_position} < regular_end && selected_token_mask[selected_token_precision_index];
    assign selected_token_a8 = a8_token_mask[selected_token_precision_index];
    assign regular_record = input_index >= regular_begin && input_index < regular_end ?
        regular_scores[7'(input_index-regular_begin)] : 17'd0;
    assign row_score = regular_enabled ? regular_record[15:0] : {8'd0, row_record[7:0]};
    assign row_current = regular_enabled ? regular_record[16] : row_record[8];
    assign row_eligible = regular_enabled ? input_index >= regular_begin && input_index < regular_end && !row_current : row_record[9];
    assign row_mandatory = !regular_enabled && row_record[10];
    assign row_required = !regular_enabled && row_record[11];
    assign row_eligible_order = regular_enabled ? input_index[10:0] : row_record[22:12];
    assign row_required_order = row_record[33:23];
    assign row_valid = reader_valid;
    assign row_last = input_index + 12'd1 == sequence_length;
    assign probability_offset = configuration[TOKEN_REFRESH_CONFIG_PROBABILITY_CONFIGURATION_OFFSET_OFFSET*8 +: 16];
    assign pending_offset = configuration[TOKEN_REFRESH_CONFIG_PENDING_CONFIGURATION_OFFSET_OFFSET*8 +: 16];
    assign reader_ready = state == JOINT_BASE_WAIT || state == JOINT_SELECTED_WAIT ? joint_base_ready :
        state == JOINT_FUTURE_WAIT ? joint_future_ready :
        state == EXECUTION_EXTENSION_WAIT || state == CONFIG_WAIT || state == PROBABILITY_WAIT || state == SHORTLIST_CONFIG_WAIT || state == JOINT_CONFIG_WAIT ||
        state == JOINT_EXTENSION_WAIT || state == ATTEMPT_WAIT || state == PENDING_CONFIG_WAIT || state == PENDING_SCOPE_WAIT || state == BUDGET_WAIT || state == ABORT_DRAIN ||
            (row_ready && (input_index[0] || row_last));
    assign scan_error = reader_error;
    assign scan_abort_ack = (reader_abort_ack || scan_ready) &&
        (!commit_active || commit_abort_ack) && (!metadata_active || metadata_abort_ack) &&
        (!relation_active || relation_abort_ack) && (!pending_active || pending_abort_ack) && (!shortlist_active || shortlist_abort_ack) && (!joint_running || joint_abort_ack);
    assign dma_request_valid = shortlist_active ? short_request_valid : pending_active ? pend_request_valid : relation_active ? rel_request_valid : metadata_active ? meta_request_valid : commit_active ? commit_request_valid : table_request_valid;
    assign dma_request_address = shortlist_active ? short_request_address : pending_active ? pend_request_address : relation_active ? rel_request_address : metadata_active ? meta_request_address : commit_active ? commit_request_address : table_request_address;
    assign dma_request_bytes = shortlist_active ? short_request_bytes : pending_active ? pend_request_bytes : relation_active ? rel_request_bytes : metadata_active ? meta_request_bytes : commit_active ? commit_request_bytes : table_request_bytes;
    assign dma_request_tag = shortlist_active ? short_request_tag : pending_active ? pend_request_tag : relation_active ? rel_request_tag : metadata_active ? meta_request_tag : commit_active ? commit_request_tag : table_request_tag;
    assign dma_read_ready = shortlist_active ? short_read_ready : pending_active ? pend_read_ready : relation_active ? rel_read_ready : metadata_active ? meta_read_ready : commit_active ? commit_read_ready : table_read_ready;
    assign dma_write_request_valid = shortlist_active ? short_write_request_valid : pending_active ? pend_write_request_valid : relation_active ? rel_write_request_valid : metadata_active ? meta_write_request_valid : commit_write_request_valid;
    assign dma_write_request_address = shortlist_active ? short_write_request_address : pending_active ? pend_write_request_address : relation_active ? rel_write_request_address : metadata_active ? meta_write_request_address : commit_write_request_address;
    assign dma_write_request_bytes = shortlist_active ? short_write_request_bytes : pending_active ? pend_write_request_bytes : relation_active ? rel_write_request_bytes : metadata_active ? meta_write_request_bytes : commit_write_request_bytes;
    assign dma_write_request_tag = shortlist_active ? short_write_request_tag : pending_active ? pend_write_request_tag : relation_active ? rel_write_request_tag : metadata_active ? meta_write_request_tag : commit_write_request_tag;
    assign dma_write_valid = shortlist_active ? short_write_valid : pending_active ? pend_write_valid : relation_active ? rel_write_valid : metadata_active ? meta_write_valid : commit_write_valid;
    assign dma_write_data = shortlist_active ? short_write_data : pending_active ? pend_write_data : relation_active ? rel_write_data : metadata_active ? meta_write_data : commit_write_data;
    assign dma_write_byte_enable = shortlist_active ? short_write_byte_enable : pending_active ? pend_write_byte_enable : relation_active ? rel_write_byte_enable : metadata_active ? meta_write_byte_enable : commit_write_byte_enable;
    assign dma_write_last = shortlist_active ? short_write_last : pending_active ? pend_write_last : relation_active ? rel_write_last : metadata_active ? meta_write_last : commit_write_last;
    assign output_row_valid = state == OUTPUT_ROWS && output_bits[0] && !abort_request;
    // Each consumer accepts a token once. A SRAM stall must not withdraw valid
    // from a selected-position consumer that is independently backpressured.
    assign output_row_ready = (output_selected_accepted || selected_ready) &&
        (!metadata_store_required || output_metadata_accepted || metadata_position_ready);
    assign metadata_store_required = write_metadata ||
        (commit_cache && target_token_count != 0);
    assign metadata_scratch_req_valid = shortlist_active ? short_scratch_req_valid : metadata_active ? writer_scratch_req_valid :
        commit_active ? commit_position_read_req_valid : 1'b0;
    assign metadata_scratch_write = shortlist_active ? short_scratch_write : metadata_active && writer_scratch_write;
    assign metadata_scratch_address = shortlist_active ? short_scratch_address : metadata_active ? writer_scratch_address :
        commit_position_read_req_index;
    assign metadata_scratch_write_data = shortlist_active ? short_scratch_write_data : writer_scratch_write_data;
    assign metadata_scratch_write_enable = shortlist_active ? short_scratch_write_enable : writer_scratch_write_enable;
    assign writer_scratch_req_ready = metadata_active && metadata_scratch_req_ready;
    assign commit_position_read_req_ready = commit_active && !metadata_active &&
        metadata_scratch_req_ready;
    assign writer_scratch_rsp_valid = metadata_active && metadata_scratch_rsp_valid;
    assign writer_scratch_rsp_data = metadata_scratch_rsp_data;
    assign commit_position_read_rsp_valid = commit_active && !metadata_active &&
        metadata_scratch_rsp_valid;
    assign commit_position_read_rsp_data = metadata_scratch_rsp_data[58:39];

    always_comb begin
        scan_address = table_base;
        scan_request_bytes = scan_bytes;
        case (state)
            CONFIG_START: begin scan_address = descriptor_address; scan_request_bytes = 32'(TOKEN_REFRESH_CONFIG_BYTES); end
            EXECUTION_EXTENSION_START: begin scan_address = descriptor_address + 64'd176; scan_request_bytes = 32'd32; end
            PROBABILITY_START: begin scan_address = probability_address[63:0]; scan_request_bytes = 32'(ATTENTION_PROBABILITY_CONFIG_BYTES); end
            SHORTLIST_CONFIG_START: begin scan_address = shortlist_descriptor_address; scan_request_bytes = 32'd32; end
            JOINT_CONFIG_START: begin scan_address = joint_descriptor_address; scan_request_bytes = 32'd64; end
            JOINT_EXTENSION_START: begin scan_address = joint_descriptor_address + 64'd64; scan_request_bytes = 32'd32; end
            ATTEMPT_START: begin scan_address = attempt_base; scan_request_bytes = 32'd144; end
            PENDING_CONFIG_START: begin scan_address = pending_configuration_address[63:0]; scan_request_bytes = 32'd80; end
            PENDING_SCOPE_START: begin scan_address = pending_configuration_address[63:0]+64'd80; scan_request_bytes = 32'd16; end
            BUDGET_START: begin scan_address = pending_configuration[575:512]; scan_request_bytes = 32'd16; end
            JOINT_BASE_START: begin scan_address = joint_base_address; scan_request_bytes = {22'd0, joint_base_rows, 4'd0}; end
            JOINT_SELECTED_START: begin scan_address = table_base + {50'd0, output_word, output_bit, 3'd0}; scan_request_bytes = 32'd8; end
            JOINT_FUTURE_START: begin scan_address = joint_future_address; scan_request_bytes = {22'd0, joint_future_rows, 4'd0}; end
            default: begin end
        endcase
    end
    assign joint_base_address = joint_configuration[UAPS_CONFIG_BASE_TABLE_BASE_OFFSET*8 +: 64];
    assign joint_future_address = joint_configuration[UAPS_CONFIG_FUTURE_TABLE_BASE_OFFSET*8 +: 64];
    assign joint_result_address = joint_configuration[UAPS_CONFIG_RESULT_BASE_OFFSET*8 +: 64];
    assign joint_base_rows = regular_enabled ? regular_base_rows : joint_configuration[UAPS_CONFIG_BASE_TOKEN_COUNT_OFFSET*8 +: 6];
    assign joint_future_rows = joint_configuration[UAPS_CONFIG_FUTURE_TOKEN_COUNT_OFFSET*8 +: 6];
    assign joint_next_start = joint_configuration[UAPS_CONFIG_NEXT_BLOCK_START_OFFSET*8 +: 11];
    assign joint_max_next = joint_configuration[UAPS_CONFIG_MAX_NEXT_TOKENS_OFFSET*8 +: 6];
    assign attempt_base = joint_extension[63:0];
    assign attempt_limit = joint_extension[159:128];
    assign attempt_enabled = attempt_base != 0;
    assign attempt_end = {1'b0, attempt_base} + 65'd144;
    assign attempt_same_block = attempt_header[31:0] == 32'h31425441 && attempt_header[47:32] == {5'd0, joint_next_start};
    assign attempt_replay = attempt_same_block && attempt_header[95:64] == capture_index;
    assign attempt_future_index = joint_state_input ? joint_mark_index[4:0] : reader_offset[8:4];
    assign attempt_count = attempt_counts[attempt_future_index];
    assign retry_min_confidence = joint_extension[175:160];
    assign future_last_confidence = joint_state_input ? pending_state_confidence : reader_data[79:64];
    assign future_last_confidence_valid = joint_state_input ? pending_state_confidence_valid : reader_data[80];
    assign attempt_selected_replay = attempt_replay && attempt_header[7'd96+{2'd0, attempt_future_index}];
    assign future_admission_allowed = (!attempt_enabled || attempt_count < attempt_limit ||
        (attempt_selected_replay && attempt_count == attempt_limit)) &&
        (retry_min_confidence == 0 || attempt_count == 0 || (attempt_selected_replay && attempt_count == 1) ||
         (future_last_confidence_valid && future_last_confidence >= retry_min_confidence));
    assign attempt_write_payload[127:0] = {joint_added_mask, capture_index, 16'd0, 5'd0, joint_next_start, 32'h31425441};
    for (genvar counter = 0; counter < 32; counter++) begin : attempt_payload
        assign attempt_write_payload[128+counter*32 +: 32] = attempt_counts[counter];
    end
    assign joint_state_control = joint_extension[232];
    assign joint_deferred = joint_extension[233];
    assign joint_current_allowed = !joint_state_control || {2'd0, current_unresolved_count} <= joint_extension[215:208];
    assign joint_allow_new = joint_extension[231:224] != 0 && {2'd0, future_tentative_count} < joint_extension[223:216];
    assign joint_base_a8 = selected_token_precision_valid ? selected_token_a8 : regular_enabled ? reader_data[34] : reader_data[16];
    // Published bit52 identifies a current MASKED token; optional context A8
    // allocation changes actual precision but not that next-pass obligation.
    assign joint_base_forecast_a8 = joint_state_control ?
        joint_base_a8 || (joint_current_allowed && !joint_deferred && reader_data[52]) :
        selected_token_precision_valid ? selected_token_a8 || reader_data[52] : regular_enabled ? reader_data[52] : reader_data[17];
    assign joint_state_input = regular_enabled && joint_future_rows != 0 && joint_future_address == 0 && joint_configuration[383:320] == 0;
    assign joint_record_error = (state == JOINT_FUTURE_STATE && !pending_state_valid) || (reader_valid && reader_ready &&
        ((state == JOINT_BASE_WAIT && (reader_data[127:19] != 0 || reader_data[15:11] != 0 ||
            {1'b0, reader_data[10:0]} >= sequence_length)) ||
         (state == JOINT_FUTURE_WAIT && (reader_data[127:112] != 0 || reader_data[95:81] != 0 ||
             (!joint_extension[184] && reader_data[111:96] != 0) || reader_data[31:19] != 0 ||
             (reader_data[80] && reader_data[79:64] > 16'h3f80))) ||
         (state == JOINT_SELECTED_WAIT && reader_data[63:55] != 0)));
    assign joint_bf16_error = joint_running && bf16_rsp_valid &&
        (bf16_rsp.tag != 16'hb900 || bf16_rsp.lane_mask != 64'd1 || bf16_rsp.values[15:0] > 16'h4200);
    assign pending_bf16_error = pending_active && bf16_rsp_valid &&
        (bf16_rsp.tag != 16'hba00 || bf16_rsp.lane_mask != 64'd1);
    always_comb begin
        bf16_req = '0;
        bf16_req.operation = pending_active && pend_multiply ? hardware_types_pkg::BF16_VECTOR_MULTIPLY : hardware_types_pkg::BF16_VECTOR_ADD;
        bf16_req.values[15:0] = pending_active ? pend_lhs : joint_add_lhs;
        bf16_req.paired_values[15:0] = pending_active ? pend_rhs : joint_add_rhs;
        bf16_req.factor0_values[15:0] = pend_rhs;
        bf16_req.lane_mask = 64'd1;
        bf16_req.tag = pending_active ? 16'hba00 : 16'hb900;
        if (relation_active) begin
            bf16_req = '0;
            bf16_req.operation = hardware_types_pkg::BF16_VECTOR_MULTIPLY;
            bf16_req.values[127:0] = rel_dequant_values;
            for (integer lane = 0; lane < 8; lane++)
                bf16_req.factor0_values[lane*16 +: 16] = rel_dequant_scale;
            bf16_req.lane_mask = {56'd0, rel_dequant_mask};
            bf16_req.tag = 16'hbb00;
        end
        if (shortlist_active) begin
            bf16_req = '0;
            bf16_req.operation = hardware_types_pkg::BF16_VECTOR_MULTIPLY;
            bf16_req.values[15:0] = short_multiply_lhs;
            bf16_req.factor0_values[15:0] = short_multiply_rhs;
            bf16_req.lane_mask = 64'd1;
            bf16_req.tag = 16'hbc00;
        end
    end
    assign bf16_req_valid = shortlist_active ? short_multiply_valid : relation_active ? rel_dequant_valid : pending_active ? pend_arithmetic_valid : joint_add_valid;
    assign bf16_rsp_ready = shortlist_active ? short_multiply_result_ready : relation_active ? rel_dequant_ready : pending_active ? pend_arithmetic_ready : joint_running;
    // The shortlist drains its single accepted multiply, including on abort.
    assign bf16_abort_request = shortlist_active ? 1'b0 : relation_active ? rel_dequant_abort : pending_active ? pend_arithmetic_abort : joint_add_abort;
    assign rel_dequant_error = bf16_rsp.tag != 16'hbb00 || bf16_rsp.lane_mask != {56'd0, rel_dequant_mask};
    utilization_aware_prefetch_scheduler joint (
        .clk, .rst, .abort_request(abort_request || state == ABORT_DRAIN), .abort_ack(joint_abort_ack),
        .start_valid(state == JOINT_START && !abort_request), .start_ready(joint_start_ready),
        .start_base_token_count(joint_base_rows), .start_future_token_count(joint_future_rows), .start_target_token_count(target_token_count[5:0]),
        .start_max_next_tokens(joint_max_next), .start_next_block_start(joint_next_start),
        .start_prediction_target(joint_extension[182:176]),
        .start_min_reuse_score(joint_extension[207:192]),
        .start_state_eligibility(joint_state_control), .start_allow_new_admission(joint_allow_new),
        .start_rank_priorities(joint_configuration[127]),
        .start_source_b_dependency_tie_rank(joint_extension[184]),
        .start_source_b_a4_only(joint_extension[185]),
        .start_block_step_index(joint_extension[255:240]),
        .start_low_dependency(joint_configuration[126]),
        .start_available_rows(joint_configuration[125:120]),
        .base_valid((state == JOINT_BASE_WAIT || state == JOINT_SELECTED_WAIT) && reader_valid), .base_ready(joint_base_ready),
        .base_index(regular_enabled ? joint_base_index : reader_offset[9:4]),
        .base_position(regular_enabled ? selected_position : reader_data[10:0]),
        .base_a8(joint_base_a8),
        .base_next_pass_a8(joint_base_forecast_a8),
        .base_prediction(regular_enabled ? reader_data[53] : reader_data[18]),
        .future_valid(state == JOINT_FUTURE_STATE ? pending_state_valid : state == JOINT_FUTURE_WAIT && reader_valid),
        .future_ready(joint_future_ready), .future_index(joint_state_input ? joint_mark_index[4:0] : reader_offset[8:4]),
        .future_a8(joint_state_input ? pending_state_value[2] : reader_data[18]),
        .future_tentative(joint_state_input ? (pending_state_value[1:0] == 1 && !pending_source_a_pending) : reader_data[17]),
        .future_eligible(joint_state_input ? (joint_current_allowed && pending_state_value[1:0] != 2 && !pending_source_a_pending) : reader_data[16]),
        .future_admission_allowed(future_admission_allowed),
        .future_dependency_bf16(joint_state_input ? regular_scores[input_index[6:0]][15:0] :
            joint_configuration[127] ? reader_data[15:0] : reader_data[111:96]),
        .future_priority_bf16(joint_state_input ? regular_scores[input_index[6:0]][15:0] : reader_data[15:0]),
        .future_service_count(joint_state_input ? 32'd0 : reader_data[63:32]),
        .add_valid(joint_add_valid), .add_ready(bf16_req_ready), .add_lhs_bf16(joint_add_lhs), .add_rhs_bf16(joint_add_rhs),
        .add_result_valid(bf16_rsp_valid && joint_running && !joint_bf16_error), .add_result_bf16(bf16_rsp.values[15:0]),
        .add_abort_request(joint_add_abort), .add_abort_ack(bf16_abort_ack),
        .done_valid(joint_done), .done_ready(1'b1), .error(joint_error),
        .future_prediction_mask(joint_progress_mask), .added_future_token_mask(joint_added_mask),
        .future_prediction_count(joint_progress_count), .added_future_token_count(joint_added_count),
        .base_activation_slots(joint_base_units), .joint_activation_slots(joint_units), .next_pass_activation_slots(joint_forecast_units)
    );
    operator_dma_reader #(.REQUEST_TAG(8'hb1)) table_reader (
        .clk, .rst, .abort_request(scan_abort_request),
        .start_valid(scan_valid), .start_ready(scan_ready),
        .start_address(scan_address), .start_bytes(scan_request_bytes), .start_total_bytes(scan_request_bytes),
        .request_valid(table_request_valid), .request_ready(dma_request_ready && !commit_active && !metadata_active && !relation_active && !pending_active && !shortlist_active),
        .request_address(table_request_address), .request_bytes(table_request_bytes),
        .request_tag(table_request_tag), .read_valid(dma_read_valid && !commit_active && !metadata_active && !relation_active && !pending_active && !shortlist_active), .read_ready(table_read_ready),
        .read_data(dma_read_data), .read_byte_enable(dma_read_byte_enable),
        .read_last(dma_read_last), .read_tag(dma_read_tag), .response_error(dma_error && !commit_active && !metadata_active && !relation_active && !pending_active && !shortlist_active),
        .upstream_abort_ack(dma_abort_ack && !commit_active && !metadata_active && !relation_active && !pending_active && !shortlist_active), .data_valid(reader_valid), .data_ready(reader_ready),
        .data(reader_data), .data_byte_enable(reader_byte_enable), .data_last(reader_last),
        .data_byte_offset(reader_offset), .done_pulse(reader_done), .error(reader_error),
        .abort_ack(reader_abort_ack)
    );

    cross_block_token_shortlist shortlist (
        .clk, .rst, .abort_request(shortlist_active && (abort_request || state == ABORT_DRAIN)),
        .abort_ack(shortlist_abort_ack), .start_valid(state == SHORTLIST_START && !abort_request),
        .start_ready(shortlist_start_ready), .start_sequence_length(sequence_length),
        .start_shortlist_token_count(shortlist_configuration[137:128]),
        .start_flags(shortlist_configuration[147:144]),
        .start_target_token_count(target_token_count[9:0]),
        .start_relative_score_floor(shortlist_configuration[175:160]),
        .start_protected_begin(shortlist_configuration[187:176]),
        .start_protected_end(shortlist_configuration[203:192]),
        .selected_target_token_count(short_target_token_count),
        .multiply_valid(short_multiply_valid), .multiply_ready(bf16_req_ready && shortlist_active),
        .multiply_lhs(short_multiply_lhs), .multiply_rhs(short_multiply_rhs),
        .multiply_result_valid(bf16_rsp_valid && shortlist_active), .multiply_result_ready(short_multiply_result_ready),
        .multiply_result(bf16_rsp.values[15:0]),
        .multiply_error(bf16_rsp.tag != 16'hbc00 || bf16_rsp.lane_mask != 64'd1),
        .scratch_req_valid(short_scratch_req_valid),
        .scratch_req_ready(shortlist_active && metadata_scratch_req_ready),
        .scratch_write(short_scratch_write), .scratch_address(short_scratch_address),
        .scratch_write_data(short_scratch_write_data), .scratch_write_enable(short_scratch_write_enable),
        .scratch_rsp_valid(shortlist_active && metadata_scratch_rsp_valid),
        .scratch_rsp_data(metadata_scratch_rsp_data),
        .start_pending_base(shortlist_configuration[63:0]), .start_pending_limit(shortlist_configuration[127:64]),
        .start_table_base(table_base), .start_table_limit(table_limit),
        .done_valid(shortlist_done), .done_ready(1'b1), .error(shortlist_error), .error_id(shortlist_error_id),
        .read_request_valid(short_request_valid), .read_request_ready(dma_request_ready && shortlist_active),
        .read_request_address(short_request_address), .read_request_bytes(short_request_bytes), .read_request_tag(short_request_tag),
        .read_valid(dma_read_valid && shortlist_active), .read_ready(short_read_ready), .read_data(dma_read_data),
        .read_byte_enable(dma_read_byte_enable), .read_last(dma_read_last), .read_tag(dma_read_tag),
        .read_error(dma_error && shortlist_active), .read_abort_ack(dma_abort_ack && shortlist_active),
        .write_request_valid(short_write_request_valid), .write_request_ready(dma_write_request_ready && shortlist_active),
        .write_request_address(short_write_request_address), .write_request_bytes(short_write_request_bytes), .write_request_tag(short_write_request_tag),
        .write_valid(short_write_valid), .write_ready(dma_write_ready && shortlist_active), .write_data(short_write_data),
        .write_byte_enable(short_write_byte_enable), .write_last(short_write_last),
        .write_done(dma_write_done && shortlist_active), .write_error(dma_write_error && shortlist_active)
    );

    attention_cache_commit cache_commit (
        .clk, .rst, .abort_request(commit_active && (abort_request || state == ABORT_DRAIN)),
        .abort_ack(commit_abort_ack), .start_valid(state == COMMIT_START && commit_cache && !commit_active && !abort_request),
        .start_ready(commit_start_ready), .start_sequence_length(sequence_length),
        .start_selected_count(target_token_count[8:0]), .start_bases(cache_bases), .start_limits(cache_limits),
        .start_head_stride(head_stride), .start_scale_head_stride(scale_head_stride),
        .position_table_ready(metadata_position_table_ready),
        .position_read_req_valid(commit_position_read_req_valid),
        .position_read_req_ready(commit_position_read_req_ready),
        .position_read_req_index(commit_position_read_req_index),
        .position_read_rsp_valid(commit_position_read_rsp_valid),
        .position_read_rsp_data(commit_position_read_rsp_data),
        .copy_enable(!metadata_active),
        .done_valid(commit_done), .done_ready(1'b1), .error(commit_error), .error_id(commit_error_id),
        .read_request_valid(commit_request_valid), .read_request_ready(dma_request_ready && commit_active && !metadata_active),
        .read_request_address(commit_request_address), .read_request_bytes(commit_request_bytes),
        .read_request_tag(commit_request_tag), .read_valid(dma_read_valid && commit_active && !metadata_active),
        .read_ready(commit_read_ready), .read_data(dma_read_data), .read_byte_enable(dma_read_byte_enable),
        .read_last(dma_read_last), .read_tag(dma_read_tag), .read_error(dma_error && commit_active && !metadata_active),
        .read_abort_ack(dma_abort_ack && commit_active && !metadata_active),
        .write_request_valid(commit_write_request_valid), .write_request_ready(dma_write_request_ready && !metadata_active),
        .write_request_address(commit_write_request_address), .write_request_bytes(commit_write_request_bytes),
        .write_request_tag(commit_write_request_tag), .write_valid(commit_write_valid), .write_ready(dma_write_ready && !metadata_active),
        .write_data(commit_write_data), .write_byte_enable(commit_write_byte_enable), .write_last(commit_write_last),
        .write_done(dma_write_done && commit_active && !metadata_active), .write_error(dma_write_error && commit_active && !metadata_active)
    );

    attention_dependency_update relation (
        .dequant_valid(rel_dequant_valid), .dequant_ready(bf16_req_ready && relation_active),
        .dequant_values(rel_dequant_values), .dequant_scale(rel_dequant_scale), .dequant_lane_mask(rel_dequant_mask),
        .dequant_result_valid(bf16_rsp_valid && relation_active), .dequant_result_ready(rel_dequant_ready),
        .dequant_result_values(bf16_rsp.values[127:0]), .dequant_error(rel_dequant_error),
        .dequant_abort(rel_dequant_abort), .dequant_abort_ack(bf16_abort_ack && relation_active),
        .clk, .rst, .abort_request(relation_active && (abort_request || state == ABORT_DRAIN)),
        .abort_ack(relation_abort_ack), .start_valid(state == RELATION_START && !abort_request),
        .start_ready(relation_start_ready), .start_job_base(relation_job_base), .start_job_limit(relation_job_end[63:0]),
        .start_job_count(relation_job_count), .done_valid(relation_done), .done_ready(1'b1),
        .start_layer_update(relation_only), .start_first_layer(!relation_seen),
        .error(relation_error), .error_id(relation_error_id),
        .read_request_valid(rel_request_valid), .read_request_ready(dma_request_ready && relation_active),
        .read_request_address(rel_request_address), .read_request_bytes(rel_request_bytes), .read_request_tag(rel_request_tag),
        .read_valid(dma_read_valid && relation_active), .read_ready(rel_read_ready), .read_data(dma_read_data),
        .read_byte_enable(dma_read_byte_enable), .read_last(dma_read_last), .read_tag(dma_read_tag),
        .read_error(dma_error && relation_active), .read_abort_ack(dma_abort_ack && relation_active),
        .write_request_valid(rel_write_request_valid), .write_request_ready(dma_write_request_ready && relation_active),
        .write_request_address(rel_write_request_address), .write_request_bytes(rel_write_request_bytes), .write_request_tag(rel_write_request_tag),
        .write_valid(rel_write_valid), .write_ready(dma_write_ready && relation_active), .write_data(rel_write_data),
        .write_byte_enable(rel_write_byte_enable), .write_last(rel_write_last),
        .write_done(dma_write_done && relation_active), .write_error(dma_write_error && relation_active)
    );

    refresh_score_update pending_update (
        .state_read_index(input_index[6:0]), .state_read_valid(pending_state_valid), .state_read_value(pending_state_value),
        .state_read_source_a_pending(pending_source_a_pending),
        .state_read_confidence(pending_state_confidence), .state_read_confidence_valid(pending_state_confidence_valid),
        .start_update_row_table(pending_configuration[122]), .start_precision_enabled(precision_enabled),
        .start_row_table_base(table_base), .start_row_table_limit(table_limit),
        .start_state_changes(pending_configuration[121]),
        .start_change_position_begin(regular_enabled ? regular_begin : pending_configuration[107:96]-12'd32),
        .start_change_capture_index(configuration[TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET*8 +: 32]),
        .clk, .rst, .abort_request(pending_active && (abort_request || state == ABORT_DRAIN)),
        .abort_ack(pending_abort_ack), .start_valid(state == PENDING_START && !abort_request), .start_ready(pending_start_ready),
        .start_rows(pending_configuration[75:64]), .start_keys(pending_configuration[86:80]),
        .start_block_end(pending_configuration[107:96]), .start_relation_row_shift(pending_configuration[115:112]),
        .start_all_changes(pending_configuration[120]),
        .start_live_consumed(live_consumed_enabled), .live_consumed(read_selected_word[pending_score_row[4:0]]),
        .start_clear_relation(prepare_only), .start_advance_block(pending_configuration[125]),
        .start_confidence_mode(pending_configuration[124:123]),
        .start_relation_base(pending_configuration[191:128]), .start_relation_limit(pending_configuration[255:192]),
        .start_change_base(pending_configuration[319:256]), .start_change_limit(pending_configuration[383:320]),
        .start_pending_base(pending_configuration[447:384]), .start_pending_limit(pending_configuration[511:448]),
        .done_valid(pending_done), .done_ready(1'b1), .error(pending_error), .error_id(pending_error_id),
        .score_valid(pending_score_valid), .score_row(pending_score_row), .score_bf16(pending_score_bf16),
        .score_mandatory(pending_score_mandatory),
        .read_request_valid(pend_request_valid), .read_request_ready(dma_request_ready && pending_active),
        .read_request_address(pend_request_address), .read_request_bytes(pend_request_bytes), .read_request_tag(pend_request_tag),
        .read_valid(dma_read_valid && pending_active), .read_ready(pend_read_ready), .read_data(dma_read_data),
        .read_byte_enable(dma_read_byte_enable), .read_last(dma_read_last), .read_tag(dma_read_tag),
        .read_error(dma_error && pending_active), .read_abort_ack(dma_abort_ack && pending_active),
        .write_request_valid(pend_write_request_valid), .write_request_ready(dma_write_request_ready && pending_active),
        .write_request_address(pend_write_request_address), .write_request_bytes(pend_write_request_bytes), .write_request_tag(pend_write_request_tag),
        .write_valid(pend_write_valid), .write_ready(dma_write_ready && pending_active), .write_data(pend_write_data),
        .write_byte_enable(pend_write_byte_enable), .write_last(pend_write_last),
        .write_done(dma_write_done && pending_active), .write_error(dma_write_error && pending_active),
        .arithmetic_valid(pend_arithmetic_valid), .arithmetic_ready(bf16_req_ready && pending_active),
        .arithmetic_multiply(pend_multiply), .arithmetic_lhs(pend_lhs), .arithmetic_rhs(pend_rhs),
        .arithmetic_result_valid(bf16_rsp_valid && pending_active && !pending_bf16_error),
        .arithmetic_result_ready(pend_arithmetic_ready), .arithmetic_result(bf16_rsp.values[15:0]),
        .arithmetic_abort_request(pend_arithmetic_abort), .arithmetic_abort_ack(bf16_abort_ack && pending_active)
    );

    refresh_token_metadata_writer metadata_writer (
        .start_publish_result(configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+6]),
        .start_attempt_base(attempt_enabled ? attempt_base : 64'd0), .start_attempt_payload(attempt_write_payload),
        .start_deep_precision_enabled(deep_precision_enabled), .start_deep_a8_limit(precision_quota),
        .start_pending_base(regular_enabled && pending_configuration[122] ? pending_configuration[447:384] : 64'd0),
        .start_pending_limit(pending_configuration[511:448]),
        .start_pending_rows(pending_configuration[70:64]), .start_pending_selected_mask(selected_token_mask),
        .position_precision_valid(selected_token_precision_valid), .position_a8(selected_token_a8),
        .start_write_output(write_metadata),
        .position_table_ready(metadata_position_table_ready),
        .scratch_req_valid(writer_scratch_req_valid),
        .scratch_req_ready(writer_scratch_req_ready),
        .scratch_write(writer_scratch_write),
        .scratch_address(writer_scratch_address),
        .scratch_write_data(writer_scratch_write_data),
        .scratch_write_enable(writer_scratch_write_enable),
        .scratch_rsp_valid(writer_scratch_rsp_valid),
        .scratch_rsp_data(writer_scratch_rsp_data),
        .scratch_aux_write_valid(metadata_scratch_aux_write_valid),
        .scratch_aux_write_ready(metadata_scratch_aux_write_ready),
        .scratch_aux_write_address(metadata_scratch_aux_write_address),
        .scratch_aux_write_data(metadata_scratch_aux_write_data),
        .scratch_aux_write_enable(metadata_scratch_aux_write_enable),
        .start_budget_base(regular_enabled ? pending_configuration[575:512] : 64'd0),
        .start_budget_limit(pending_configuration[639:576]), .start_budget_record(regular_budget),
        .start_embedding_source(metadata_embedding_source),
        .start_joint_result_base(joint_descriptor_address == 0 ? 64'd0 : joint_result_address),
        .start_joint_result_limit(joint_configuration[511:448]),
        .start_joint_result({24'd0, joint_forecast_units, joint_units, joint_base_units,
            2'd0, joint_added_count, 2'd0, joint_progress_count, joint_added_mask, joint_progress_mask}),
        .clk, .rst, .abort_request(metadata_active && (abort_request || state == ABORT_DRAIN)),
        .packer_abort_request, .packer_abort_ack, .packer_start_valid, .packer_start_ready, .packer_config,
        .packer_row_valid, .packer_row_ready, .packer_row, .packer_beat_valid, .packer_beat_ready,
        .packer_beat_data, .packer_beat_last, .packer_done_valid, .packer_done_ready,
        .packer_error, .packer_error_id, .packer_compute_groups, .packer_semantic_groups,
        .abort_ack(metadata_abort_ack),
        .start_valid(state == COMMIT_START && metadata_store_required && !metadata_active && !abort_request),
        .start_ready(metadata_start_ready), .start_selected_count(target_token_count[8:0]),
        .start_sequence_length(sequence_length), .start_table_base(table_base), .start_table_limit(table_limit),
        .start_output_base(metadata_base), .start_output_limit(metadata_limit),
        .start_metadata_version(metadata_version), .start_capture_index(capture_index),
        .start_qkvo_group(configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+15]),
        .position_valid(output_row_valid && !output_metadata_accepted && metadata_store_required),
        .position_ready(metadata_position_ready), .position(selected_position), .position_last(selected_last),
        .done_valid(metadata_done), .done_ready(1'b1), .error(metadata_error), .error_id(metadata_error_id),
        .output_bytes(writer_metadata_bytes), .output_rounds(writer_metadata_rounds),
        .read_request_valid(meta_request_valid), .read_request_ready(dma_request_ready && metadata_active),
        .read_request_address(meta_request_address), .read_request_bytes(meta_request_bytes), .read_request_tag(meta_request_tag),
        .read_valid(dma_read_valid && metadata_active), .read_ready(meta_read_ready), .read_data(dma_read_data),
        .read_byte_enable(dma_read_byte_enable), .read_last(dma_read_last), .read_tag(dma_read_tag),
        .read_error(dma_error && metadata_active), .read_abort_ack(dma_abort_ack && metadata_active),
        .write_request_valid(meta_write_request_valid), .write_request_ready(dma_write_request_ready && metadata_active),
        .write_request_address(meta_write_request_address), .write_request_bytes(meta_write_request_bytes), .write_request_tag(meta_write_request_tag),
        .write_valid(meta_write_valid), .write_ready(dma_write_ready && metadata_active), .write_data(meta_write_data),
        .write_byte_enable(meta_write_byte_enable), .write_last(meta_write_last),
        .write_done(dma_write_done && metadata_active), .write_error(dma_write_error && metadata_active)
    );

    assign start_ready = (state == IDLE || state == WAIT_FORWARD) && !abort_request;
    assign scan_valid = (state == SCAN_START || state == CONFIG_START || state == EXECUTION_EXTENSION_START || state == PROBABILITY_START ||
        state == SHORTLIST_CONFIG_START || state == JOINT_CONFIG_START || state == JOINT_EXTENSION_START || state == ATTEMPT_START ||
        state == JOINT_BASE_START || state == JOINT_SELECTED_START || state == JOINT_FUTURE_START || state == PENDING_CONFIG_START || state == PENDING_SCOPE_START || state == BUDGET_START) && !abort_request;
    assign row_ready = state == SCAN_ROWS && !abort_request && !scan_error;
    assign selected_position = {output_word, output_bit};
    assign selected_valid = output_row_valid && !output_selected_accepted;
    assign selected_last = output_count + 12'd1 == target_token_count;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign scan_abort_request = state == ABORT_DRAIN;
    assign read_word = pending_active && live_consumed_enabled ? pending_score_row[10:5] : state == OUTPUT_LOAD || state == JOINT_SELECTED_LOAD ? output_word : input_index[10:5];
    for (genvar bank = 0; bank < 8; bank++) begin : bitmap_bank
        assign read_bank_word[bank] = selected_words[bank][read_word[2:0]];
        for (genvar word_index = 0; word_index < 8; word_index++) begin : bitmap_word
            always_ff @(posedge clk) begin
                if ((state == CLEAR || state == CONSUMED_CLEAR) && clear_word == 6'(bank*8 + word_index))
                    selected_words[bank][word_index] <= '0;
                else if (mark_valid && mark_position[10:5] == 6'(bank*8 + word_index))
                    selected_words[bank][word_index] <= selected_words[bank][word_index] |
                        (32'd1 << mark_position[4:0]);
            end
        end
    end
    assign read_selected_word = read_bank_word[read_word[5:3]];
    assign mark_valid = !rst && !abort_request && !scan_error &&
        ((capture_consumed) || (row_pending && initial_scan && (pending_current || pending_mandatory)) ||
         (state == CHOOSE && best_valid && (!regular_enabled || best_selection_key != 0)) ||
         (state == JOINT_BASE_WAIT && reader_valid && reader_ready && !joint_record_error) ||
         (state == JOINT_MARK && joint_mark_index < joint_future_rows && joint_added_mask[joint_mark_index[4:0]]));
    assign mark_position = capture_consumed ? loaded_token_position : state == JOINT_BASE_WAIT ? reader_data[10:0] :
        state == JOINT_MARK ? joint_next_start + {5'd0, joint_mark_index} : initial_scan ? pending_position : best_position;
    assign candidate_valid = row_pending && !initial_scan && pending_eligible &&
        !pending_selected && (!required_scan || pending_required);
    assign candidate_better = !best_valid || pending_score > best_selection_key ||
        (pending_score == best_selection_key && pending_order < best_order);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            sequence_length <= '0;
            target_token_count <= '0;
            required_quota <= '0;
            initial_scan <= 1'b0;
            required_scan <= 1'b0;
            input_index <= '0;
            clear_word <= '0;
            output_word <= '0;
            output_bit <= '0;
            output_bits <= '0;
            output_count <= '0;
            selected_count <= '0;
            current_count <= '0;
            eligible_count <= '0;
            required_count <= '0;
            mandatory_required <= '0;
            remaining_required <= '0;
            row_pending <= 1'b0;
            pending_last <= 1'b0;
            pending_current <= 1'b0;
            pending_eligible <= 1'b0;
            pending_mandatory <= 1'b0;
            pending_required <= 1'b0;
            pending_selected <= 1'b0;
            pending_score <= '0;
            pending_position <= '0;
            pending_order <= '0;
            best_valid <= 1'b0;
            best_selection_key <= '0;
            best_position <= '0;
            best_order <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            advance_prepared <= 1'b0;
            source_a_valid <= 1'b0; source_a_mask <= '0; source_a_block_start <= '0; source_a_capture_index <= '0;
            abort_ack <= 1'b0;
            aborting <= 1'b0;
            table_base <= '0;
            commit_cache <= 1'b0;
            commit_active <= 1'b0;
            cache_bases <= '0;
            cache_limits <= '0;
            configuration <= '0;
            probability_config <= '0;
            joint_configuration <= '0;
            joint_extension <= '0;
            current_unresolved_count <= '0; future_tentative_count <= '0;
            attempt_header <= '0;
            pending_configuration <= '0; pending_regular_phase <= 1'b0;
            pending_scope_loaded <= 1'b0; pending_scope_invalid <= 1'b0; consume_current_begin <= '0;
            regular_budget <= '0;
            shortlist_configuration <= '0;
            head_stride <= '0;
            scale_head_stride <= '0;
            write_metadata <= 1'b0; metadata_active <= 1'b0;
            metadata_base <= '0; metadata_limit <= '0; table_limit <= '0;
            metadata_version <= '0; capture_index <= '0;
            descriptor_address <= '0;
            prepare_only <= 1'b0; prepared_for_forward <= 1'b0;
            saved_closeout <= 1'b0; regular_closed <= 1'b0;
            execution_extension <= '0;
            relation_job_count <= '0; relation_job_base <= '0; relation_active <= 1'b0;
            relation_only <= 1'b0; relation_seen <= 1'b0; relation_per_layer <= 1'b0;
                        advance_prepared <= 1'b0;
            probability_address <= '0; probability_reserved_error <= 1'b0;
            joint_descriptor_address <= '0; joint_running <= 1'b0; joint_mark_index <= '0;
            attempt_record_error <= 1'b0;
            joint_base_end <= '0; joint_future_end <= '0; joint_result_end <= '0;
            metadata_embedding_source <= 1'b0;
            pending_active <= 1'b0; pending_configuration_address <= '0;
            regular_enabled <= 1'b0; regular_available_half <= '0; regular_budget_end <= '0;
            prediction_token_mask <= '0; a8_token_mask <= '0; selected_token_mask <= '0; token_precision_count <= '0;
            regular_base_rows <= '0; joint_base_index <= '0;
            shortlist_active <= 1'b0; shortlist_descriptor_address <= '0;
            output_selected_accepted <= 1'b0; output_metadata_accepted <= 1'b0;
        end else begin
            // A successful selection publishes membership once for the next
            // postprocess. PREPARE and layer callbacks must retain this result.
            if (consume_source_a || abort_request) source_a_valid <= 1'b0;
            if (state == COMPLETE && done_ready) begin
                if (terminal_error || regular_closed) source_a_valid <= 1'b0;
                else if (!prepare_only && !relation_only) begin
                    source_a_valid <= joint_descriptor_address != 0;
                    source_a_mask <= joint_descriptor_address != 0 ? joint_progress_mask & ~joint_added_mask : 32'd0;
                    source_a_block_start <= joint_next_start;
                    source_a_capture_index <= capture_index;
                end
            end
            abort_ack <= 1'b0;
            row_pending <= 1'b0;
            if (state != OUTPUT_ROWS || !output_row_valid || output_row_ready || abort_request) begin
                output_selected_accepted <= 1'b0;
                output_metadata_accepted <= 1'b0;
            end else begin
                if (selected_valid && selected_ready) output_selected_accepted <= 1'b1;
                if (metadata_store_required && metadata_position_ready) output_metadata_accepted <= 1'b1;
            end
            if (state == EXECUTION_EXTENSION_WAIT && reader_valid && reader_ready)
                execution_extension[reader_offset[4]*128 +: 128] <= reader_data;
            if (state == CONFIG_WAIT && reader_valid && reader_ready)
                configuration[reader_offset[7:4]*128 +: 128] <= reader_data;
            if (state == JOINT_CONFIG_WAIT && reader_valid && reader_ready)
                joint_configuration[reader_offset[5:4]*128 +: 128] <= reader_data;
            if (state == JOINT_EXTENSION_WAIT && reader_valid && reader_ready)
                joint_extension[reader_offset[4]*128 +: 128] <= reader_data;
            if (state == ATTEMPT_WAIT && reader_valid && reader_ready) begin
                if (reader_offset == 0) attempt_header <= reader_data;
                else begin
                    for (integer lane = 0; lane < 4; lane++)
                        attempt_counts[5'((reader_offset-16)/4)+5'(lane)] <= reader_data[lane*32 +: 32];
                    if (reader_data[31] || reader_data[63] || reader_data[95] || reader_data[127]) attempt_record_error <= 1'b1;
                end
            end
            if (state == PENDING_CONFIG_WAIT && reader_valid && reader_ready)
                pending_configuration[reader_offset[6:4]*128 +: 128] <= reader_data;
            if (state == PENDING_SCOPE_WAIT && reader_valid && reader_ready) begin
                consume_current_begin <= reader_data[11:0];
                pending_scope_invalid <= reader_data[127:12] != 0;
                pending_scope_loaded <= 1'b1;
            end
            if (state == BUDGET_WAIT && reader_valid && reader_ready)
                regular_budget <= reader_data;
            if (state == SHORTLIST_CONFIG_WAIT && reader_valid && reader_ready)
                shortlist_configuration[reader_offset[4]*128 +: 128] <= reader_data;
            for (integer entry = 0; entry < 96; entry++)
                if (regular_enabled && pending_score_valid && pending_score_row[6:0] == 7'(entry))
                    regular_scores[entry] <= {pending_score_mandatory, pending_score_bf16};
            if (state == PROBABILITY_WAIT && reader_valid && reader_ready) begin
                case (reader_offset[6:4])
                    3'd0: begin
                        probability_config.output_base <= reader_data[63:0];
                        probability_config.output_limit <= reader_data[127:64];
                    end
                    3'd1: begin
                        probability_config.batch_stride_bytes <= reader_data[31:0];
                        probability_config.head_stride_bytes <= reader_data[63:32];
                        probability_config.round_stride_bytes <= reader_data[95:64];
                        probability_config.layer_mask <= reader_data[127:96];
                    end
                    3'd2: probability_config.key_groups[127:0] <= reader_data;
                    3'd3: probability_config.key_groups[255:128] <= reader_data;
                    default: begin
                        probability_config.query_begin <= reader_data[11:0];
                        probability_config.query_end <= reader_data[27:16];
                        probability_config.capture_p8 <= reader_data[32];
                        shortlist_descriptor_address <= reader_data[127:64];
                        probability_reserved_error <= reader_data[63:33] != 0 ||
                            reader_data[15:12] != 0 || reader_data[31:28] != 0;
                    end
                endcase
            end
            if (state == ABORT_DRAIN) begin
                if (commit_abort_ack) commit_active <= 1'b0;
                if (metadata_abort_ack) metadata_active <= 1'b0;
                if (relation_abort_ack) relation_active <= 1'b0;
                if (joint_abort_ack) joint_running <= 1'b0;
                if (pending_abort_ack) pending_active <= 1'b0;
                if (shortlist_abort_ack) shortlist_active <= 1'b0;
            end
            if (abort_request && state != IDLE && state != ABORT_LOW &&
                    state != ABORT_DRAIN) begin
                aborting <= 1'b1;
                prepared_for_forward <= 1'b0;
                state <= state == EXECUTION_EXTENSION_WAIT || state == CONFIG_WAIT || state == PROBABILITY_WAIT || state == SCAN_ROWS || state == SCAN_DRAIN ||
                    state == JOINT_CONFIG_WAIT || state == JOINT_EXTENSION_WAIT || state == ATTEMPT_WAIT || state == JOINT_BASE_WAIT || state == JOINT_SELECTED_WAIT || state == JOINT_FUTURE_WAIT || state == PENDING_CONFIG_WAIT || state == PENDING_SCOPE_WAIT || state == BUDGET_WAIT ||
                    state == SHORTLIST_CONFIG_WAIT || commit_active || metadata_active || relation_active || pending_active || shortlist_active || joint_running ? ABORT_DRAIN : ABORT_LOW;
                if (state != CONFIG_WAIT && state != PROBABILITY_WAIT && state != SCAN_ROWS && state != SCAN_DRAIN &&
                    state != JOINT_CONFIG_WAIT && state != JOINT_EXTENSION_WAIT && state != ATTEMPT_WAIT && state != JOINT_BASE_WAIT && state != JOINT_SELECTED_WAIT && state != JOINT_FUTURE_WAIT && state != PENDING_CONFIG_WAIT && state != PENDING_SCOPE_WAIT && state != BUDGET_WAIT &&
                    state != SHORTLIST_CONFIG_WAIT && !commit_active && !metadata_active && !relation_active && !pending_active && !shortlist_active && !joint_running) abort_ack <= 1'b1;
            end else if (pending_bf16_error && state != ABORT_DRAIN) begin
                terminal_error <= 1'b1; terminal_error_id <= 8'h45; aborting <= 1'b0; state <= ABORT_DRAIN;
            end else if ((joint_record_error || joint_bf16_error || (joint_done && joint_error)) && state != ABORT_DRAIN) begin
                terminal_error <= 1'b1; terminal_error_id <= 8'h08; aborting <= 1'b0; state <= ABORT_DRAIN;
                if (joint_done) joint_running <= 1'b0;
            end else if (commit_active && commit_done && commit_error && state != ABORT_DRAIN) begin
                terminal_error <= 1'b1;
                terminal_error_id <= 8'h10 | commit_error_id;
                commit_active <= 1'b0;
                aborting <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (metadata_active && metadata_done && metadata_error && state != ABORT_DRAIN) begin
                terminal_error <= 1'b1;
                terminal_error_id <= 8'h20 | metadata_error_id;
                metadata_active <= 1'b0;
                aborting <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (scan_error && (state == EXECUTION_EXTENSION_WAIT || state == CONFIG_WAIT || state == PROBABILITY_WAIT || state == SCAN_ROWS || state == SCAN_DRAIN ||
                state == SHORTLIST_CONFIG_WAIT || state == JOINT_CONFIG_WAIT || state == JOINT_EXTENSION_WAIT || state == ATTEMPT_WAIT || state == JOINT_BASE_WAIT || state == JOINT_SELECTED_WAIT || state == JOINT_FUTURE_WAIT || state == PENDING_CONFIG_WAIT || state == PENDING_SCOPE_WAIT || state == BUDGET_WAIT)) begin
                terminal_error <= 1'b1;
                terminal_error_id <= 8'h03;
                state <= ABORT_DRAIN;
                aborting <= 1'b0;
            end else begin
                if (row_valid && row_ready) begin
                    if (initial_scan && precision_enabled && input_index >= regular_begin && input_index < regular_end) begin
                        prediction_token_mask[token_precision_index] <= row_record[53];
                        a8_token_mask[token_precision_index] <= all_selected_tokens_a8 || (row_record[53] && row_record[34]);
                    end
                    row_pending <= 1'b1;
                    pending_last <= row_last;
                    pending_current <= row_current;
                    pending_eligible <= row_eligible;
                    pending_mandatory <= row_mandatory;
                    pending_required <= row_required;
                    pending_score <= row_score;
                    pending_position <= input_index[10:0];
                    pending_order <= required_scan ? row_required_order : row_eligible_order;
                    pending_selected <= read_selected_word[input_index[4:0]];
                    input_index <= input_index + 12'd1;
                    if (row_record[63:55] != 0 ||
                            (row_current && row_eligible) ||
                            ((row_mandatory || row_required) && !row_eligible)) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= 8'h02;
                        row_pending <= 1'b0;
                        state <= ABORT_DRAIN;
                        aborting <= 1'b0;
                    end else if (row_last) state <= SCAN_DRAIN;
                end
                if (row_pending) begin
                    if (initial_scan) begin
                        current_count <= current_count + {11'd0, pending_current};
                        eligible_count <= eligible_count + {11'd0, pending_eligible};
                        required_count <= required_count + {11'd0, pending_required};
                        mandatory_required <= mandatory_required +
                            {11'd0, pending_required && pending_mandatory};
                        selected_count <= selected_count +
                            {11'd0, pending_current || pending_mandatory};
                    end else if (candidate_valid && candidate_better) begin
                        best_valid <= 1'b1;
                        best_position <= pending_position;
                        best_order <= pending_order;
                        best_selection_key <= pending_score;
                    end
                end
                case (state)
                    IDLE, WAIT_FORWARD: if (start_valid && start_ready) begin
                        saved_closeout <= start_closeout;
                        regular_closed <= 1'b0;
                        if (state == WAIT_FORWARD && !start_prepare) begin
                            prepared_for_forward <= start_relation_only;
                            relation_only <= start_relation_only;
                            prepare_only <= 1'b0;
                            if (start_descriptor_address != descriptor_address || start_joint_descriptor_address != joint_descriptor_address) begin
                                terminal_error <= 1'b1;
                                terminal_error_id <= 8'h01;
                                state <= COMPLETE;
                            end else if (start_relation_only) begin
                                if (!relation_per_layer || relation_job_count == 0) begin
                                    terminal_error <= 1'b1;
                                    terminal_error_id <= 8'h01;
                                    state <= COMPLETE;
                                end else begin
                                    if (start_relation_l31 && execution_extension[79:64] != 0) begin
                                        relation_job_base <= execution_extension[63:0];
                                        relation_job_count <= execution_extension[79:64];
                                    end
                                    state <= RELATION_START;
                                end
                            end else state <= relation_job_count != 0 && !relation_seen ? RELATION_START : pending_offset != 0 ? PENDING_RANGE : joint_descriptor_address != 0 ? JOINT_CONFIG_START : selected_count == target_token_count ?
                                (commit_cache || write_metadata ? COMMIT_START : OUTPUT_LOAD) : SCAN_START;
                        end else begin
                        // A new execution can replace a drained preparation left
                        // by an intervening Transformer or forward-postprocess failure.
                        relation_only <= 1'b0; relation_seen <= 1'b0; relation_per_layer <= 1'b0;
                        regular_enabled <= 1'b0; pending_regular_phase <= 1'b0;
                        joint_descriptor_address <= start_joint_descriptor_address;
                        joint_extension <= '0;
                        execution_extension <= '0;
                        probability_config <= '0;
                        probability_reserved_error <= 1'b0;
                        shortlist_descriptor_address <= '0;
                        descriptor_address <= start_descriptor_address;
                        prepare_only <= start_prepare;
                        prepared_for_forward <= 1'b0;
                        terminal_error <= 1'b0;
                        terminal_error_id <= '0;
                        if (start_relation_only || start_descriptor_address[3:0] != 0 || descriptor_end[64] || start_joint_descriptor_address[3:0] != 0 ||
                            start_joint_descriptor_address > 64'hffffffffffffffbf) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h01;
                            state <= COMPLETE;
                        end else state <= CONFIG_START;
                        end
                    end
                    CONFIG_START: if (scan_ready) state <= CONFIG_WAIT;
                    CONFIG_WAIT: if (reader_done && !reader_error) state <= CONFIG_DECODE;
                    CONFIG_DECODE: begin
                        metadata_embedding_source <= configuration[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET*8+2];
                        probability_address <= {1'b0, descriptor_address} + {49'd0, probability_offset};
                        pending_configuration_address <= {1'b0, descriptor_address} + {49'd0, pending_offset};
                        sequence_length <= start_sequence_length[11:0];
                        target_token_count <= start_target_token_count[11:0];
                        required_quota <= start_required_quota[11:0];
                        table_base <= start_table_base;
                        commit_cache <= start_commit_cache;
                        cache_bases <= start_cache_bases;
                        cache_limits <= start_cache_limits;
                        head_stride <= start_head_stride;
                        scale_head_stride <= start_scale_head_stride;
                        write_metadata <= start_write_metadata;
                        metadata_base <= start_metadata_base;
                        metadata_limit <= start_metadata_limit;
                        metadata_version <= start_metadata_version;
                        capture_index <= start_capture_index;
                        table_limit <= start_table_limit;
                        relation_job_count <= configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET*8 +: 16];
                        relation_job_base <= configuration[TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET*8 +: 64];
                        clear_word <= '0;
                        output_word <= '0;
                        output_count <= '0;
                        selected_count <= '0;
                        current_count <= '0;
                        eligible_count <= '0;
                        required_count <= '0;
                        mandatory_required <= '0;
                        remaining_required <= '0;
                        initial_scan <= 1'b1;
                        required_scan <= 1'b0;
                        terminal_error <= 1'b0;
                        terminal_error_id <= '0;
                        aborting <= 1'b0;
                        if (configuration_invalid || start_sequence_length == 0 || start_sequence_length > 16'd2048 ||
                                start_target_token_count == 0 || start_target_token_count > start_sequence_length ||
                                start_required_quota > start_target_token_count || start_table_base[3:0] != 0 ||
                                table_end[64] || table_end[63:0] > start_table_limit ||
                                start_target_token_count > 16'd432) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h01;
                            state <= COMPLETE;
                        end else if (configuration[TOKEN_REFRESH_CONFIG_BYTES_OFFSET*8 +: 16] == 16'd208) begin
                            if (descriptor_address > 64'hffffffffffffff2f) begin
                                terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                            end else state <= EXECUTION_EXTENSION_START;
                        end else state <= probability_offset == 0 ? CLEAR : PROBABILITY_RANGE;
                    end
                    EXECUTION_EXTENSION_START: if (scan_ready) state <= EXECUTION_EXTENSION_WAIT;
                    EXECUTION_EXTENSION_WAIT: if (reader_done && !reader_error) state <= EXECUTION_EXTENSION_CHECK;
                    EXECUTION_EXTENSION_CHECK: begin
                        if (execution_extension[95:80] != 0 || execution_extension[127:98] != 0 ||
                            execution_extension[175:171] != 0 || execution_extension[191:176] != 0 ||
                            execution_extension[255:224] != 0 || execution_extension[3:0] != 0 ||
                            l31_relation_job_end[64] ||
                            (execution_extension[79:64] != 0 && execution_extension[63:0] == 0)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                        end else begin
                            if (prepare_only && execution_extension[96]) begin
                                source_a_valid <= execution_extension[97];
                                source_a_mask <= execution_extension[159:128];
                                source_a_block_start <= execution_extension[170:160];
                                source_a_capture_index <= execution_extension[223:192];
                            end
                            state <= probability_offset == 0 ? CLEAR : PROBABILITY_RANGE;
                        end
                    end
                    PROBABILITY_RANGE: begin
                        if (probability_address[64] || probability_address > 65'h0ffffffffffffffaf) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                        end else state <= PROBABILITY_START;
                    end
                    PROBABILITY_START: if (scan_ready) state <= PROBABILITY_WAIT;
                    PROBABILITY_WAIT: if (reader_done && !reader_error) state <= PROBABILITY_CHECK;
                    PROBABILITY_CHECK: begin
                        relation_per_layer <= probability_config.capture_p8 ||
                            $countones(probability_config.layer_mask) > 1;
                        if (probability_reserved_error || probability_config.layer_mask == 0 ||
                            (relation_job_count == 0 && $countones(probability_config.layer_mask) != 1) ||
                            probability_config.query_begin > probability_config.query_end ||
                            probability_config.query_end > sequence_length ||
                            probability_config.output_base[3:0] != 0 ||
                            probability_config.output_base >= probability_config.output_limit ||
                            probability_config.batch_stride_bytes == 0 || probability_config.batch_stride_bytes[3:0] != 0 ||
                            probability_config.head_stride_bytes[3:0] != 0 || probability_config.round_stride_bytes[3:0] != 0 ||
                            {3'd0, probability_config.head_stride_bytes} < 35'(probability_config.batch_stride_bytes)*35'd6 ||
                            {5'd0, probability_config.round_stride_bytes} < {probability_config.head_stride_bytes, 5'd0}) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                        end else if (shortlist_descriptor_address != 0 &&
                            (shortlist_descriptor_address[3:0] != 0 || shortlist_descriptor_address > 64'hffffffffffffffdf)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                        end else begin
                            probability_config.enable <= 1'b1;
                            state <= shortlist_descriptor_address != 0 ? SHORTLIST_CONFIG_START : CLEAR;
                        end
                    end
                    SHORTLIST_CONFIG_START: if (scan_ready) state <= SHORTLIST_CONFIG_WAIT;
                    SHORTLIST_CONFIG_WAIT: if (reader_done && !reader_error) state <= SHORTLIST_CHECK;
                    SHORTLIST_CHECK: begin
                        if (shortlist_configuration[143:138] != 0 || shortlist_configuration[159:148] != 0 ||
                            shortlist_configuration[191:188] != 0 || shortlist_configuration[255:204] != 0 ||
                            (shortlist_configuration[145] && required_quota != 0)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h51; state <= COMPLETE;
                        end else state <= SHORTLIST_START;
                    end
                    SHORTLIST_START: if (shortlist_start_ready) begin shortlist_active <= 1'b1; state <= SHORTLIST_WAIT; end
                    SHORTLIST_WAIT: if (shortlist_done) begin
                        shortlist_active <= 1'b0;
                        if (shortlist_error) begin terminal_error <= 1'b1; terminal_error_id <= 8'h50 | shortlist_error_id; state <= COMPLETE; end
                        else begin
                            if (shortlist_configuration[146]) target_token_count <= short_target_token_count;
                            state <= CLEAR;
                        end
                    end
                    CLEAR: begin
                        selected_token_mask <= '0;
                        token_precision_count <= '0;
                        clear_word <= clear_word + 6'd1;
                        if (clear_word == 6'd63) begin
                            if (prepare_only && (joint_descriptor_address != 0 || pending_offset != 0)) begin
                                if (pending_offset != 0) state <= PENDING_RANGE;
                                else begin prepared_for_forward <= 1'b1; state <= COMPLETE; end
                            end else state <= !prepare_only && relation_job_count != 0 ? RELATION_START :
                                !prepare_only && pending_offset != 0 ? PENDING_RANGE :
                                joint_descriptor_address != 0 ? JOINT_CONFIG_START : SCAN_START;
                        end
                    end
                    RELATION_START: if (relation_start_ready) begin relation_active <= 1'b1; state <= RELATION_WAIT; end
                    RELATION_WAIT: if (relation_done) begin
                        relation_active <= 1'b0;
                        if (relation_error) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h30 | relation_error_id; state <= COMPLETE;
                        end else if (relation_only) begin
                            relation_seen <= 1'b1;
                            state <= COMPLETE;
                        end else state <= pending_offset != 0 ? PENDING_RANGE : joint_descriptor_address != 0 ? JOINT_CONFIG_START : initial_scan ? SCAN_START : selected_count == target_token_count ?
                            (commit_cache || write_metadata ? COMMIT_START : OUTPUT_LOAD) : SCAN_START;
                    end
                    SCAN_START: if (scan_ready) begin
                        input_index <= '0;
                        best_valid <= 1'b0;
                        state <= SCAN_ROWS;
                    end
                    SCAN_DRAIN: if (row_pending && pending_last)
                        state <= initial_scan ? CHECK_INITIAL : CHOOSE;
                    CHECK_INITIAL: begin
                        initial_scan <= 1'b0;
                        remaining_required <= required_quota > mandatory_required ?
                            required_quota - mandatory_required : 12'd0;
                        required_scan <= required_quota > mandatory_required;
                        if (regular_enabled) begin
                            target_token_count <= regular_available_half < 0 ? selected_count :
                                (regular_available_half >>> 1) < $signed({21'd0, selected_count}) ? selected_count :
                                (regular_available_half >>> 1) > $signed({21'd0, pending_configuration[75:64]}) ? pending_configuration[75:64] :
                                regular_available_half[12:1];
                            required_scan <= 1'b0;
                            state <= REGULAR_LIMIT;
                        end else if (current_count == 0 || selected_count > target_token_count ||
                                current_count + eligible_count < target_token_count ||
                                required_quota > required_count ||
                                (required_quota > mandatory_required &&
                                 required_quota - mandatory_required > target_token_count - selected_count)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h04;
                            state <= COMPLETE;
                        end else if (prepare_only) begin
                            prepared_for_forward <= 1'b1;
                            state <= COMPLETE;
                        end else state <= selected_count == target_token_count ?
                            (commit_cache || write_metadata ? COMMIT_START : OUTPUT_LOAD) : SCAN_START;
                    end
                    CHOOSE: begin
                        if (regular_enabled && (!best_valid || best_selection_key == 0)) begin
                            state <= REGULAR_FINISH;
                        end else if (!best_valid) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h05;
                            state <= COMPLETE;
                        end else begin
                            selected_count <= selected_count + 12'd1;
                            if (required_scan) begin
                                remaining_required <= remaining_required - 12'd1;
                                if (remaining_required == 12'd1) required_scan <= 1'b0;
                            end
                            state <= selected_count + 12'd1 == target_token_count ?
                                (regular_enabled ? REGULAR_FINISH : commit_cache || write_metadata ? COMMIT_START : OUTPUT_LOAD) : SCAN_START;
                        end
                    end
                    OUTPUT_LOAD: begin
                        output_bits <= read_selected_word;
                        output_bit <= '0;
                        state <= OUTPUT_ROWS;
                    end
                    OUTPUT_ROWS: if (!output_bits[0] || output_row_ready) begin
                        output_bits <= {1'b0, output_bits[31:1]};
                        output_bit <= output_bit + 5'd1;
                        if (output_bits[0]) output_count <= output_count + 12'd1;
                        if (output_bits[0] && selected_last)
                            state <= metadata_store_required ? METADATA_WAIT : COMPLETE;
                        else if (selected_position + 11'd1 == sequence_length[10:0]) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h06;
                            state <= COMPLETE;
                        end else if (output_bit == 5'd31) begin
                            output_word <= output_word + 6'd1;
                            state <= OUTPUT_LOAD;
                        end
                    end
                    COMPLETE: if (done_ready) begin
                        if (terminal_error) prepared_for_forward <= 1'b0;
                        state <= prepared_for_forward && !terminal_error ? WAIT_FORWARD : IDLE;
                    end
                    PENDING_RANGE: begin
                        if (pending_configuration_address[64] || pending_configuration_address > 65'h0ffffffffffffffaf) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h41; state <= COMPLETE;
                        end else begin pending_scope_loaded <= 1'b0; pending_scope_invalid <= 1'b0; state <= PENDING_CONFIG_START; end
                    end
                    PENDING_CONFIG_START: if (scan_ready) state <= PENDING_CONFIG_WAIT;
                    PENDING_CONFIG_WAIT: if (reader_done && !reader_error) state <= PENDING_CHECK;
                    PENDING_SCOPE_START: if (scan_ready) state <= PENDING_SCOPE_WAIT;
                    PENDING_SCOPE_WAIT: if (reader_done && !reader_error) state <= PENDING_CHECK;
                    PENDING_CHECK: begin
                        regular_budget_end <= {1'b0, pending_configuration[575:512]} + 65'd16;
                        if (pending_configuration[31:0] != 32'h314e5041 || pending_configuration[47:32] != 16'd1 ||
                            pending_configuration[63:48] != (pending_configuration[127] ? 16'd96 : 16'd80) ||
                            pending_configuration[79:76] != 0 || pending_configuration[95:87] != 0 ||
                            pending_configuration[111:108] != 0 || pending_configuration[119:116] != 0 ||
                            (paired_pending && (!pending_regular_phase ?
                                (!pending_configuration[120] || pending_configuration[127] ||
                                 pending_configuration[575:512] != 0) :
                                (pending_configuration[120] || pending_configuration[575:512] == 0))) ||
                            (live_consumed_enabled && !pending_configuration[120]) ||
                            (pending_configuration[127] && (pending_configuration[126] || pending_configuration[125] ||
                                pending_configuration[575:512] != 0 || joint_descriptor_address != 0 ||
                                pending_configuration_address > 65'h0ffffffffffffff9f ||
                                (pending_scope_loaded && (pending_scope_invalid || consume_current_end > {1'b0, sequence_length})) )) ||
                            (pending_configuration[125] && (!pending_configuration[120] ||
                                (!prepare_only && !advance_prepared))) || pending_configuration[124:123] == 3 ||
                            (!pending_configuration[120] && pending_configuration[124:123] != 0) ||
                            (pending_configuration[122] && (!pending_configuration[121] || pending_configuration[120] ||
                                pending_configuration[575:512] == 0 || !metadata_embedding_source)) ||
                            (pending_configuration[121] && pending_configuration[575:512] == 0 && !pending_configuration[120])) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h41; state <= COMPLETE;
                        end else if (pending_configuration[127] && !pending_scope_loaded) begin
                            state <= PENDING_SCOPE_START;
                        end else if (prepare_only) begin
                            if (pending_configuration[125]) state <= PENDING_START;
                            else begin prepared_for_forward <= 1'b1; state <= COMPLETE; end
                        end else if (precision_enabled && pending_configuration[575:512] == 0 &&
                                !(paired_pending && !pending_regular_phase)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h46; state <= COMPLETE;
                        end else if (pending_configuration[575:512] != 0) begin regular_enabled <= 1'b1; state <= BUDGET_RANGE; end
                        else state <= PENDING_START;
                    end
                    BUDGET_RANGE: begin
                        if (pending_configuration[515:512] != 0 || regular_budget_end[64] ||
                            regular_budget_end[63:0] > pending_configuration[639:576]) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h46; state <= COMPLETE;
                        end else state <= BUDGET_START;
                    end
                    BUDGET_START: if (scan_ready) state <= BUDGET_WAIT;
                    BUDGET_WAIT: if (reader_done && !reader_error) state <= BUDGET_CHECK;
                    BUDGET_CHECK: begin
                        regular_available_half <= $signed({regular_budget[63], regular_budget[63:32]}) + $signed({25'd0, regular_budget[7:0]});
                        if (regular_budget[7:0] < 2 || regular_budget[7:0] > 192 || regular_budget[31:28] != 0 ||
                            regular_budget[7:0] > {3'd0, pending_configuration[75:64], 1'b0} ||
                            regular_budget[15:8] != 0 ||
                            regular_end > sequence_length || pending_configuration[75:64] > 96 ||
                            pending_configuration[86:80] != pending_configuration[70:64] || pending_configuration[120] ||
                            !write_metadata || commit_cache || required_quota != 0 ||
                            regular_budget[95:64] == 32'hffffffff || regular_budget[127:96] > 32'hffffff9f) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h46; state <= COMPLETE;
                        end else if (saved_closeout) begin
                            // Cross-block pending has already been updated by
                            // the first paired phase. Block-local pending and
                            // budget are rebuilt at the next block; no next
                            // regular selection may publish UAPS state here.
                            regular_closed <= 1'b1;
                            prepared_for_forward <= 1'b0;
                            state <= COMPLETE;
                        end else state <= PENDING_START;
                    end
                    REGULAR_LIMIT: state <= selected_count >= target_token_count ? REGULAR_FINISH : SCAN_START;
                    REGULAR_FINISH: begin
                        regular_budget[63:32] <= regular_final_credit[31:0];
                        regular_budget[95:64] <= regular_budget[95:64] + 32'd1;
                        regular_budget[127:96] <= regular_budget[127:96] + {20'd0, selected_count};
                        target_token_count <= selected_count;
                        regular_base_rows <= selected_count[5:0];
                        if (regular_final_credit[32] != regular_final_credit[31] || (joint_descriptor_address != 0 && selected_count > 48)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h47; state <= COMPLETE;
                        end else if (precision_enabled || pending_configuration[122]) state <= PRECISION_BEGIN;
                        else if (joint_descriptor_address != 0) begin
                            target_token_count <= configuration[TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET*8 +: 12];
                            state <= JOINT_CONFIG_START;
                        end else state <= COMMIT_START;
                    end
                    PRECISION_BEGIN: begin
                        input_index <= regular_begin;
                        best_valid <= 1'b0;
                        state <= PRECISION_READ;
                    end
                    PRECISION_READ: begin
                        pending_score <= regular_record[15:0];
                        pending_selected <= read_selected_word[input_index[4:0]];
                        pending_mandatory <= prediction_token_mask[token_precision_index] || a8_token_mask[token_precision_index];
                        state <= PRECISION_SCAN;
                    end
                    PRECISION_SCAN: begin
                        selected_token_mask[token_precision_index] <= pending_selected;
                        if (pending_selected && !pending_mandatory &&
                                (!best_valid || pending_score > best_selection_key)) begin
                            best_valid <= 1'b1;
                            best_selection_key <= pending_score;
                            best_position <= input_index[10:0];
                        end
                        input_index <= input_index + 12'd1;
                        state <= input_index + 12'd1 == regular_end ? PRECISION_PICK : PRECISION_READ;
                    end
                    PRECISION_PICK: begin
                        if (precision_enabled && best_valid && token_precision_count < precision_quota) begin
                            a8_token_mask[7'({1'b0, best_position}-regular_begin)] <= 1'b1;
                            token_precision_count <= token_precision_count + 7'd1;
                            state <= token_precision_count + 7'd1 < precision_quota ? PRECISION_BEGIN : PRECISION_DONE;
                        end else state <= PRECISION_DONE;
                    end
                    PRECISION_DONE: begin
                        if (joint_descriptor_address != 0) begin
                            target_token_count <= configuration[TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET*8 +: 12];
                            state <= JOINT_CONFIG_START;
                        end else state <= COMMIT_START;
                    end
                    PENDING_START: if (pending_start_ready) begin pending_active <= 1'b1; state <= PENDING_WAIT; end
                    PENDING_WAIT: if (pending_done) begin
                        pending_active <= 1'b0;
                        if (pending_error) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h40 | pending_error_id; state <= COMPLETE;
                        end else if (prepare_only) begin
                            advance_prepared <= 1'b1; prepared_for_forward <= 1'b1; state <= COMPLETE;
                        end else if (live_consumed_enabled) begin
                            clear_word <= '0; state <= CONSUMED_CLEAR;
                        end else if (paired_pending && !pending_regular_phase) state <= PENDING_NEXT;
                        else state <= !regular_enabled && joint_descriptor_address != 0 ? JOINT_CONFIG_START : initial_scan ? SCAN_START :
                            selected_count == target_token_count ? (commit_cache || write_metadata ? COMMIT_START : OUTPUT_LOAD) : SCAN_START;
                    end
                    CONSUMED_CLEAR: begin
                        clear_word <= clear_word + 6'd1;
                        if (clear_word == 63) state <= paired_pending && !pending_regular_phase ? PENDING_NEXT :
                            joint_descriptor_address != 0 ? JOINT_CONFIG_START : SCAN_START;
                    end
                    PENDING_NEXT: begin
                        pending_regular_phase <= 1'b1;
                        pending_configuration_address <= pending_configuration_address + 65'd80;
                        state <= PENDING_RANGE;
                    end
                    JOINT_CONFIG_START: if (scan_ready) begin joint_extension <= '0; state <= JOINT_CONFIG_WAIT; end
                    JOINT_CONFIG_WAIT: if (reader_done && !reader_error) begin
                        if (joint_configuration[63:48] == 16'd96 && joint_descriptor_address > 64'hffffffffffffff9f) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h08; state <= COMPLETE;
                        end else state <= joint_configuration[63:48] == 16'd96 ? JOINT_EXTENSION_START : JOINT_RANGE;
                    end
                    JOINT_EXTENSION_START: if (scan_ready) state <= JOINT_EXTENSION_WAIT;
                    JOINT_EXTENSION_WAIT: if (reader_done && !reader_error) state <= JOINT_RANGE;
                    JOINT_RANGE: begin
                        joint_base_end <= {1'b0, joint_base_address} + {55'd0, joint_base_rows, 4'd0};
                        joint_future_end <= {1'b0, joint_future_address} + {55'd0, joint_future_rows, 4'd0};
                        joint_result_end <= {1'b0, joint_result_address} + 65'd16;
                        state <= JOINT_CHECK;
                    end
                    JOINT_CHECK: begin
                        if (joint_configuration[31:0] != 32'h314a4449 || joint_configuration[47:32] != 16'd1 ||
                            (joint_configuration[63:48] != 16'd64 && joint_configuration[63:48] != 16'd96) ||
                            (joint_configuration[63:48] == 16'd96 && joint_descriptor_address > 64'hffffffffffffff9f) ||
                            joint_extension[239:234] != 0 || joint_extension[191:186] != 0 || joint_extension[183] ||
                            (joint_state_control ? !joint_state_input || joint_extension[215:208] == 0 || joint_extension[215:208] > 32 ||
                                joint_extension[223:216] == 0 || joint_extension[223:216] > 32 || joint_extension[231:224] > 32 ||
                                pending_configuration[107:96] < regular_begin + 12'd32 ||
                                pending_configuration[107:96] > regular_end ||
                                {1'b0, joint_next_start} != pending_configuration[107:96] : joint_extension[233:208] != 0) ||
                            joint_extension[207:192] > 16'h3f80 || joint_extension[182:176] > 64 ||
                            retry_min_confidence > 16'h3f80 || (retry_min_confidence != 0 && !attempt_enabled) ||
                            (attempt_enabled && (!write_metadata || attempt_base[3:0] != 0 || attempt_limit[31] || attempt_end[64] ||
                                attempt_end[63:0] > joint_extension[127:64] ||
                                (attempt_base < table_limit && table_base < attempt_end[63:0]) ||
                                (attempt_base < metadata_limit && metadata_base < attempt_end[63:0]) ||
                                (attempt_base < joint_result_end[63:0] && joint_result_address < attempt_end[63:0]) ||
                                (attempt_base < joint_descriptor_address+64'd96 && joint_descriptor_address < attempt_end[63:0]) ||
                                (attempt_base < descriptor_address+64'd176 && descriptor_address < attempt_end[63:0]) ||
                                (!regular_enabled && attempt_base < joint_base_end[63:0] && joint_base_address < attempt_end[63:0]) ||
                                (!joint_state_input && attempt_base < joint_future_end[63:0] && joint_future_address < attempt_end[63:0]) ||
                                (pending_offset != 0 && attempt_base < pending_configuration_address[63:0]+64'd80 && pending_configuration_address[63:0] < attempt_end[63:0]) ||
                                (probability_offset != 0 && attempt_base < probability_address[63:0]+64'd80 && probability_address[63:0] < attempt_end[63:0]) ||
                                (shortlist_descriptor_address != 0 && attempt_base < shortlist_descriptor_address+64'd32 && shortlist_descriptor_address < attempt_end[63:0]) ||
                                (relation_job_count != 0 && attempt_base < configured_job_end[63:0] && relation_job_base < attempt_end[63:0]) ||
                                (regular_enabled && ((attempt_base < pending_configuration[511:448] && pending_configuration[447:384] < attempt_end[63:0]) ||
                                    (attempt_base < pending_configuration[639:576] && pending_configuration[575:512] < attempt_end[63:0]) ||
                                    (attempt_base < pending_configuration[255:192] && pending_configuration[191:128] < attempt_end[63:0]) ||
                                    (attempt_base < pending_configuration[383:320] && pending_configuration[319:256] < attempt_end[63:0]))))) ||
                            (regular_enabled ? joint_configuration[79:64] != 0 : joint_configuration[79:64] == 0 || joint_configuration[79:64] > 48) ||
                            joint_configuration[95:80] > 32 || joint_configuration[111:96] > 2047 ||
                            joint_configuration[119:112] == 0 || joint_configuration[119:112] > 32 ||
                            (!joint_configuration[127] && joint_configuration[126:120] != 0) ||
                            (joint_configuration[127] && joint_configuration[125:120] > joint_future_rows) ||
                            {1'b0, joint_next_start} + {6'd0, joint_future_rows} > sequence_length ||
                            target_token_count > 48 || !write_metadata || commit_cache ||
                            (regular_enabled ? joint_base_address != 0 || joint_configuration[255:192] != 0 :
                                joint_base_address[3:0] != 0 || joint_base_end[64] || joint_base_end[63:0] > joint_configuration[255:192]) ||
                            (joint_state_input ? !pending_configuration[122] || !joint_configuration[127] ||
                                {1'b0, joint_next_start} < regular_begin ||
                                {1'b0, joint_next_start}+{6'd0, joint_future_rows} > regular_end :
                                joint_future_address[3:0] != 0 || joint_future_end[64] || joint_future_end[63:0] > joint_configuration[383:320]) ||
                            joint_result_address == 0 || joint_result_address[3:0] != 0 || joint_result_end[64] ||
                            joint_result_end[63:0] > joint_configuration[511:448]) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h08; state <= COMPLETE;
                        end else state <= attempt_enabled ? ATTEMPT_START : joint_state_control ? JOINT_STATE_BEGIN : JOINT_START;
                    end
                    ATTEMPT_START: if (scan_ready) begin attempt_record_error <= 1'b0; state <= ATTEMPT_WAIT; end
                    ATTEMPT_WAIT: if (reader_done && !reader_error) state <= ATTEMPT_CHECK;
                    ATTEMPT_CHECK: begin
                        if (attempt_record_error || (attempt_header[31:0] != 0 && attempt_header[31:0] != 32'h31425441) || attempt_header[63:48] != 0) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h09; state <= COMPLETE;
                        end else begin
                            if (!attempt_same_block) for (integer counter = 0; counter < 32; counter++) attempt_counts[counter] <= '0;
                            state <= joint_state_control ? JOINT_STATE_BEGIN : JOINT_START;
                        end
                    end
                    JOINT_STATE_BEGIN: begin
                        input_index <= pending_configuration[107:96] - 12'd32 - regular_begin;
                        current_unresolved_count <= '0; future_tentative_count <= '0;
                        joint_mark_index <= '0; state <= JOINT_CURRENT_COUNT;
                    end
                    JOINT_CURRENT_COUNT: begin
                        if (!pending_state_valid) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h08; state <= COMPLETE;
                        end else begin
                            if (pending_state_value[1:0] != 2) current_unresolved_count <= current_unresolved_count + 6'd1;
                            input_index <= input_index + 12'd1; joint_mark_index <= joint_mark_index + 6'd1;
                            if (joint_mark_index == 31) begin
                                joint_mark_index <= '0; state <= JOINT_HANDOFF_COUNT;
                            end
                        end
                    end
                    JOINT_HANDOFF_COUNT: begin
                        if (joint_mark_index == joint_future_rows) state <= JOINT_START;
                        else if (!pending_state_valid) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h08; state <= COMPLETE;
                        end else begin
                            if (pending_state_value[1:0] == 1) future_tentative_count <= future_tentative_count + 6'd1;
                            input_index <= input_index + 12'd1; joint_mark_index <= joint_mark_index + 6'd1;
                        end
                    end
                    JOINT_START: if (joint_start_ready) begin
                        joint_running <= 1'b1; joint_base_index <= '0; output_word <= '0;
                        joint_mark_index <= '0;
                        input_index <= {1'b0, joint_next_start}-regular_begin;
                        state <= regular_enabled ? (regular_base_rows == 0 ?
                            (joint_future_rows == 0 ? JOINT_WAIT : joint_state_input ? JOINT_FUTURE_STATE : JOINT_FUTURE_START) : JOINT_SELECTED_LOAD) : JOINT_BASE_START;
                    end
                    JOINT_SELECTED_LOAD: begin output_bits <= read_selected_word; output_bit <= '0; state <= JOINT_SELECTED_FIND; end
                    JOINT_SELECTED_FIND: begin
                        if (output_bits[0]) state <= JOINT_SELECTED_START;
                        else begin
                            output_bits <= {1'b0, output_bits[31:1]}; output_bit <= output_bit + 5'd1;
                            if (output_bit == 31) begin output_word <= output_word + 6'd1; state <= JOINT_SELECTED_LOAD; end
                        end
                    end
                    JOINT_SELECTED_START: if (scan_ready) state <= JOINT_SELECTED_WAIT;
                    JOINT_SELECTED_WAIT: begin
                        if (reader_valid && reader_ready) joint_base_index <= joint_base_index + 6'd1;
                        if (reader_done && !reader_error) begin
                            output_bits <= {1'b0, output_bits[31:1]}; output_bit <= output_bit + 5'd1;
                            if (joint_base_index == regular_base_rows) state <= joint_future_rows == 0 ? JOINT_WAIT : joint_state_input ? JOINT_FUTURE_STATE : JOINT_FUTURE_START;
                            else if (output_bit == 31) begin output_word <= output_word + 6'd1; state <= JOINT_SELECTED_LOAD; end
                            else state <= JOINT_SELECTED_FIND;
                        end
                    end
                    JOINT_BASE_START: if (scan_ready) state <= JOINT_BASE_WAIT;
                    JOINT_BASE_WAIT: if (reader_done && !reader_error) state <= joint_future_rows == 0 ? JOINT_WAIT : JOINT_FUTURE_START;
                    JOINT_FUTURE_START: if (scan_ready) state <= JOINT_FUTURE_WAIT;
                    JOINT_FUTURE_WAIT: if (reader_done && !reader_error) state <= JOINT_WAIT;
                    JOINT_FUTURE_STATE: if (pending_state_valid && joint_future_ready) begin
                        input_index <= input_index + 12'd1;
                        joint_mark_index <= joint_mark_index + 6'd1;
                        if (joint_mark_index + 6'd1 == joint_future_rows) state <= JOINT_WAIT;
                    end
                    JOINT_WAIT: if (joint_done) begin
                        joint_running <= 1'b0; joint_mark_index <= '0;
                        output_word <= '0; output_count <= '0;
                        selected_count <= {6'd0, joint_base_rows} + {6'd0, joint_added_count};
                        target_token_count <= {6'd0, joint_base_rows} + {6'd0, joint_added_count};
                        state <= JOINT_MARK;
                    end
                    JOINT_MARK: begin
                        if (attempt_enabled && joint_mark_index < joint_future_rows && joint_added_mask[joint_mark_index[4:0]] && !attempt_replay)
                            attempt_counts[joint_mark_index[4:0]] <= attempt_counts[joint_mark_index[4:0]] + 32'd1;
                        joint_mark_index <= joint_mark_index + 6'd1;
                        if (joint_mark_index + 6'd1 >= joint_future_rows) state <= COMMIT_START;
                        if (attempt_enabled && joint_mark_index < joint_future_rows && joint_added_mask[joint_mark_index[4:0]] &&
                            !attempt_replay && attempt_counts[joint_mark_index[4:0]] == 32'h7fffffff) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h09; state <= COMPLETE;
                        end
                    end
                    COMMIT_START: begin
                        if (commit_cache && commit_start_ready) commit_active <= 1'b1;
                        if (metadata_store_required && metadata_start_ready) metadata_active <= 1'b1;
                        if ((!commit_cache || commit_active || commit_start_ready) &&
                                (!metadata_store_required || metadata_active || metadata_start_ready))
                            state <= target_token_count == 0 ?
                                (metadata_store_required ? METADATA_WAIT :
                                    commit_cache ? COMMIT_WAIT : COMPLETE) : OUTPUT_LOAD;
                    end
                    METADATA_WAIT: if (metadata_done) begin
                        metadata_active <= 1'b0;
                        state <= commit_cache ? COMMIT_WAIT : COMPLETE;
                    end
                    COMMIT_WAIT: if (commit_done) begin
                        commit_active <= 1'b0;
                        metadata_active <= 1'b0;
                        state <= COMPLETE;
                    end
                    ABORT_DRAIN: if (scan_abort_ack) begin
                        commit_active <= 1'b0;
                        state <= aborting ? ABORT_LOW : COMPLETE;
                        abort_ack <= aborting;
                    end
                    ABORT_LOW: if (!abort_request) state <= IDLE;
                    default: ;
                endcase
            end
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) if (!rst)
        assert (!(shortlist_active && (metadata_active || commit_active)))
            else $error("refresh scratch has overlapping shortlist/metadata/commit owners");

    always_ff @(posedge clk) begin
        if (!rst && row_valid && row_ready) begin
            assert (reader_offset == ({20'd0, input_index} & 32'hfffffffe) * 32'd8)
                else $error("attention_guided_token_selector table offset mismatch");
            assert (input_index[0] ? &reader_byte_enable[15:8] : &reader_byte_enable[7:0])
                else $error("attention_guided_token_selector incomplete table record");
        end
        if (!rst && reader_done && !reader_error)
            assert (reader_last || state == EXECUTION_EXTENSION_WAIT || state == CONFIG_WAIT || state == PROBABILITY_WAIT || state == SCAN_DRAIN || state == CHECK_INITIAL || state == CHOOSE ||
                state == JOINT_CONFIG_WAIT || state == JOINT_EXTENSION_WAIT || state == ATTEMPT_WAIT ||
                state == JOINT_BASE_WAIT || state == JOINT_SELECTED_WAIT || state == JOINT_FUTURE_WAIT)
                else $error("attention_guided_token_selector table completion before final row");
    end
`endif
endmodule

`default_nettype wire
