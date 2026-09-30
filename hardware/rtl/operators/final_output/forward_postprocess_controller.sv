`default_nettype none

module forward_postprocess_controller #(
    parameter integer LM_PANEL_COUNT = 15808
) (
    input  logic clk,
    input  logic rst,

    input  logic          start_valid,
    output logic          start_ready,
    input  logic [2559:0] start_configuration_bits,
    input logic start_source_a_valid,
    input logic [31:0] start_source_a_mask,
    input logic [10:0] start_source_a_block_start,
    input logic [31:0] start_source_a_capture_index,
    output logic          done_valid,
    output logic          current_block_complete,
    input  logic          done_ready,
    output logic          error,
    output logic [7:0]    error_id,
    output logic          active,
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

    output logic          meta_request_valid,
    input  logic          meta_request_ready,
    output hardware_types_pkg::dma_read_request_t meta_request,
    input  logic          meta_data_valid,
    output logic          meta_data_ready,
    input  hardware_types_pkg::dma_read_beat_t meta_data,
    input  hardware_types_pkg::dma_completion_t meta_completion,

    output logic          lm_request_valid,
    input  logic          lm_request_ready,
    output hardware_types_pkg::dma_read_request_t lm_request,
    input  logic          lm_data_valid,
    output logic          lm_data_ready,
    input  hardware_types_pkg::dma_read_beat_t lm_data,
    input  hardware_types_pkg::dma_completion_t lm_completion,

    output logic          write_request_valid,
    input  logic          write_request_ready,
    output hardware_types_pkg::dma_write_request_t write_request,
    output logic          write_data_valid,
    input  logic          write_data_ready,
    output hardware_types_pkg::dma_write_beat_t write_data,
    input  hardware_types_pkg::dma_completion_t write_completion,

    output logic          hidden_start_valid,
    input  logic          hidden_start_ready,
    output logic [63:0]   hidden_start_address,
    output logic [63:0]   hidden_start_limit,
    output logic [31:0]   hidden_start_bytes,
    output logic [5:0]    hidden_start_rows,
    output logic [527:0]  hidden_start_ddr_row_index,
    output logic [55:0]   hidden_start_source_round,
    output logic [47:0]   hidden_start_source_row,
    input  logic          hidden_done_valid,
    output logic          hidden_done_ready,
    input  logic          hidden_error,

    output logic          rms_start_valid,
    input  logic          rms_start_ready,
    output logic [5:0]    rms_start_rows,
    output logic [15:0]   rms_start_elements,
    output logic [15:0]   rms_start_epsilon_bf16,
    output logic          rms_start_gamma_bypass,
    input  logic          rms_done_valid,
    output logic          rms_done_ready,
    input  logic          rms_error,

    output logic          rms_source_req_valid,
    input  logic          rms_source_req_ready,
    output hardware_types_pkg::rms_tile_read_request_t rms_source_req,
    input  logic          rms_source_rsp_valid,
    output logic          rms_source_rsp_ready,
    input  hardware_types_pkg::rms_tile_read_response_t rms_source_rsp,

    output logic          max_req_valid,
    input  logic          max_req_ready,
    output hardware_types_pkg::maximum_request_t max_req,
    input  logic          max_rsp_valid,
    output logic          max_rsp_ready,
    input  hardware_types_pkg::maximum_response_t max_rsp,
    output logic          quant_scale_req_valid,
    input  logic          quant_scale_req_ready,
    output hardware_types_pkg::quant_scale_request_t quant_scale_req,
    input  logic          quant_scale_rsp_valid,
    output logic          quant_scale_rsp_ready,
    input  hardware_types_pkg::quant_scale_response_t quant_scale_rsp,
    output logic          quant_values_req_valid,
    input  logic          quant_values_req_ready,
    output hardware_types_pkg::quant_values_request_t quant_values_req,
    input  logic          quant_values_rsp_valid,
    output logic          quant_values_rsp_ready,
    input  hardware_types_pkg::quant_values_response_t quant_values_rsp,
    output logic          writer_cfg_valid,
    input  logic          writer_cfg_ready,
    output hardware_types_pkg::activation_writer_config_t writer_cfg,
    output logic          writer_scale_valid,
    input  logic          writer_scale_ready,
    output hardware_types_pkg::activation_writer_scale_t writer_scale,
    output logic          writer_values_valid,
    input  logic          writer_values_ready,
    output hardware_types_pkg::activation_writer_values_t writer_values,
    input  logic          writer_done_pulse,
    input  logic          writer_error,

    output logic          panel_write_valid,
    input  logic          panel_write_ready,
    output hardware_types_pkg::matmul_panel_write_t panel_write,
    output logic          operand_read_valid,
    input  logic          operand_read_ready,
    output hardware_types_pkg::matmul_read_request_t operand_read,
    input  logic          operand_rsp_valid,
    input  hardware_types_pkg::matmul_read_response_t operand_rsp,
    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [6:0]    scale_write_address,
    output logic [127:0]  scale_write_data,
    output logic          scale_read_valid,
    input  logic          scale_read_ready,
    output logic [6:0]    scale_read_address,
    input  logic          scale_read_rsp_valid,
    input  logic [127:0]  scale_read_rsp_data,

    output logic          candidate_state_read_valid,
    input  logic          candidate_state_read_ready,
    output logic [3:0]    candidate_state_read_row_group,
    output logic [2:0]    candidate_state_read_lane_block,
    output logic          candidate_state_read_row_half,
    output logic [1:0]    candidate_state_read_word,
    output logic [3:0]    candidate_state_read_row_mask,
    output logic [15:0]   candidate_state_read_tag,
    input  logic          candidate_state_read_rsp_valid,
    output logic          candidate_state_read_rsp_ready,
    input  logic [511:0]  candidate_state_read_rsp_data,
    input  logic [3:0]    candidate_state_read_rsp_row_mask,
    input  logic [15:0]   candidate_state_read_rsp_tag,
    output logic          candidate_state_write_valid,
    input  logic          candidate_state_write_ready,
    output logic [3:0]    candidate_state_write_row_group,
    output logic [2:0]    candidate_state_write_lane_block,
    output logic          candidate_state_write_row_half,
    output logic [1:0]    candidate_state_write_word,
    output logic [3:0]    candidate_state_write_row_mask,
    output logic [511:0]  candidate_state_write_data,

    output logic          post_table_access_valid,
    input  logic          post_table_access_ready,
    output logic          post_table_access_enable,
    output logic          post_state_write_valid,
    input  logic          post_state_write_ready,
    output logic [9:0]    post_state_write_word_address,
    output logic [127:0]  post_state_write_data,
    output logic          post_state_read_valid,
    input  logic          post_state_read_ready,
    output logic [9:0]    post_state_read_word_address,
    output logic [15:0]   post_state_read_tag,
    input  logic          post_state_read_rsp_valid,
    input  logic [127:0]  post_state_read_rsp_data,
    input  logic [15:0]   post_state_read_rsp_tag,
    output logic          post_suppressed_write_valid,
    input  logic          post_suppressed_write_ready,
    output logic [5:0]    post_suppressed_write_address,
    output logic [127:0]  post_suppressed_write_data,
    output logic          post_suppressed_read_valid,
    input  logic          post_suppressed_read_ready,
    output logic [7:0]    post_suppressed_read_index,
    input  logic          post_suppressed_read_rsp_valid,
    input  logic [16:0]   post_suppressed_read_rsp_token_id,

    output logic          pe_abort_request,
    input  logic          pe_abort_ack,
    output logic          pe_req_valid,
    input  logic          pe_req_ready,
    output hardware_types_pkg::pe_request_t pe_req,
    input  logic          pe_rsp_valid,
    output logic          pe_rsp_ready,
    input  hardware_types_pkg::pe_response_t pe_rsp,
    input  logic          pe_idle,
    output logic          rescale_abort_request,
    input  logic          rescale_abort_ack,
    output logic          rescale_req_valid,
    input  logic          rescale_req_ready,
    output hardware_types_pkg::rescale_request_t rescale_req,
    input  logic          rescale_rsp_valid,
    output logic          rescale_rsp_ready,
    input  hardware_types_pkg::rescale_response_t rescale_rsp,
    input  logic          rescale_idle,
    output logic          bf16_abort_request,
    input  logic          bf16_abort_ack,
    output logic          bf16_req_valid,
    input  logic          bf16_req_ready,
    output hardware_types_pkg::bf16_request_t bf16_req,
    input  logic          bf16_rsp_valid,
    output logic          bf16_rsp_ready,
    input  hardware_types_pkg::bf16_response_t bf16_rsp,
    output logic          reduction_req_valid,
    input  logic          reduction_req_ready,
    output hardware_types_pkg::reduction_request_t reduction_req,
    input  logic          reduction_rsp_valid,
    output logic          reduction_rsp_ready,
    input  hardware_types_pkg::reduction_response_t reduction_rsp,
    output logic          exp_req_valid,
    input  logic          exp_req_ready,
    output hardware_types_pkg::softmax_exp_request_t exp_req,
    input  logic          exp_rsp_valid,
    output logic          exp_rsp_ready,
    input  hardware_types_pkg::softmax_exp_response_t exp_rsp,
    output logic          reciprocal_req_valid,
    input  logic          reciprocal_req_ready,
    output hardware_types_pkg::softmax_reciprocal_request_t reciprocal_req,
    input  logic          reciprocal_rsp_valid,
    output logic          reciprocal_rsp_ready,
    input  hardware_types_pkg::softmax_reciprocal_response_t reciprocal_rsp,

    output logic [63:0]   accepted_weight_bytes,
    output logic [63:0]   accepted_scale_bytes,
    output logic [63:0]   accepted_pe_count,
    output logic [63:0]   candidate_wait_cycles
);
    import execution_config_pkg::*;
    import prediction_record_pkg::*;
    import forward_postprocess_config_pkg::*;
    import draft_verify_block_config_pkg::*;

    localparam logic [9:0] DESCRIPTOR_WORD_BASE = 10'd128;
    localparam logic [9:0] EXPECTED_DESCRIPTOR_WORD_BASE = 10'd320;
    localparam logic [9:0] EVENT_WORD_BASE = 10'd448;
    localparam logic [9:0] NEXT_METADATA_WORD_BASE = 10'd512;
    import token_state_entry_pkg::*;

    localparam logic [7:0] ERROR_SEQUENCE = 8'h10;
    localparam logic [7:0] ERROR_BLOCK_DMA = 8'h20;
    localparam logic [7:0] ERROR_BLOCK_FORMAT = 8'h21;
    localparam logic [7:0] ERROR_STATE_DMA = 8'h22;
    localparam logic [7:0] ERROR_SUPPRESSED_DMA = 8'h23;
    localparam logic [7:0] ERROR_LM_HEAD = 8'h30;
    localparam logic [7:0] ERROR_CANDIDATE = 8'h40;
    localparam logic [7:0] ERROR_STATE_ENTRY = 8'h50;
    localparam logic [7:0] ERROR_DRAFT_VERIFY = 8'h60;
    localparam logic [7:0] ERROR_WRITEBACK = 8'h70;

    typedef enum logic [5:0] {
        IDLE,
        SEQUENCE_START,
        SEQUENCE_WAIT,
        LM_CANDIDATE_START,
        LM_WAIT,
        TABLE_ENABLE,
        BLOCK_LOAD_START,
        BLOCK_LOAD_WAIT,
        BLOCK_VALIDATE,
        RANGE_VALIDATE,
        DESCRIPTOR_VALIDATE,
        DESCRIPTOR_READ_LOW,
        DESCRIPTOR_WAIT_LOW,
        DESCRIPTOR_READ_HIGH,
        DESCRIPTOR_WAIT_HIGH,
        DESCRIPTOR_EXPECTED_WRITE,
        STATE_LOAD_START,
        STATE_LOAD_WAIT,
        SUPPRESSED_LOAD_START,
        SUPPRESSED_LOAD_WAIT,
        CANDIDATE_WAIT,
        OUTPUT_WRITER_START,
        BLOCK_SELECT,
        FEATURE_START,
        FEATURE_EXPECTED_READ,
        FEATURE_EXPECTED_WAIT,
        FEATURE_READ_LOW,
        FEATURE_WAIT_LOW,
        FEATURE_READ_HIGH,
        FEATURE_WAIT_HIGH,
        FEATURE_SEND_ROW,
        FEATURE_WAIT,
        EVENT_STORE,
        OUTPUT_WRITER_WAIT,
        ERROR_DRAIN,
        TABLE_DISABLE,
        COMPLETE
    } state_t;

    state_t state;
    logic current_state_seen, current_unresolved;
    assign current_block_complete = current_state_seen && !current_unresolved;
    typedef struct packed {
        logic canonical_future;
        logic source_a_handoff;
        logic transfer_only;
        logic tail_bypass_all;
        logic tail_bypass_stable_only;
        logic [1:0] closeout_kind;
        logic [31:0] observed_mask;
        logic [6:0] next_entry;
        logic [6:0] current_entry;
        logic [31:0] capture_index;
        logic [31:0] input_capture_index;
        logic [5:0] position_count;
        logic [15:0] maturity_age;
        logic [15:0] tail_after_step;
        logic [15:0] step_index;
        logic [15:0] remaining_forwards;
        logic [15:0] scheduled_quota;
        logic [5:0] max_handoff_tokens;
        logic tail_threshold_enable;
        logic [5:0] global_block_id;
        logic [1:0] block_slot;
        logic [1:0] input_block_slot;
    } block_config_t;
    logic [63:0] saved_post_config_base;
    logic saved_source_a_valid;
    logic [31:0] saved_source_a_mask, saved_source_a_capture_index;
    logic [10:0] saved_source_a_block_start;
    logic [11:0] source_a_local;
    logic [63:0] saved_post_config_limit;
    logic [63:0] saved_prediction_base;
    logic [63:0] saved_prediction_limit;
    logic [63:0] saved_final_hidden_base;
    logic [63:0] saved_final_hidden_limit;
    logic [63:0] saved_next_metadata_base;
    logic [63:0] saved_next_metadata_limit;
    logic [11:0] saved_sequence_length;
    logic [1535:0] post_config_bits;
    logic [7:0] saved_prediction_count;
    logic [2:0] saved_block_count;
    logic [7:0] saved_suppressed_count;
    logic [7:0] total_state_entries;
    logic terminal_error;
    logic [7:0] terminal_error_id;

    block_config_t block_config [0:3];
    logic [3:0] block_valid;
    logic [31:0] block_masked [0:3];
    logic [31:0] block_tentative [0:3];
    logic [6:0] descriptor_current_entry [0:95];
    logic [7:0] descriptor_query_group [0:95];
    logic [7:0] descriptor_cache_group [0:95];
    logic [127:0] state_descriptor_valid;
    logic [135:0] selected_token_group [0:11];
    logic quant_writer_scale_valid, quant_writer_scale_ready;
    logic lm_activation_scale_ready;

    logic [7:0] descriptor_count;
    logic descriptor_store_high;
    logic [7:0] block_load_beat;
    logic [127:0] block_first_half;
    logic [2:0] block_validate_index;
    logic [7:0] descriptor_validate_index;
    logic [6:0] descriptor_expected_entry;
    logic [53:0] expected_descriptor_data;
    logic descriptor_format_error;
    logic block_load_format_error;
    logic [7:0] state_load_beat;
    logic [5:0] suppressed_load_beat;
    logic [7:0] output_state_entry;
    logic [2:0] active_block_slot;
    logic [5:0] active_block_rows;
    logic [5:0] active_block_row;
    logic [6:0] active_current_entry;
    logic [127:0] state_entry_low;
    logic [127:0] state_entry_high;
    logic [255:0] next_state_entry;
    logic next_state_pending;
    logic next_state_half;
    logic [2:0] event_slot;
    logic [1:0] event_word;
    logic event_read_pending;
    logic event_data_valid;
    logic [127:0] event_data;
    logic candidate_done_seen;
    logic candidate_error_seen;
    logic [7:0] candidate_error_saved;
    logic [7:0] candidate_result_count;
    logic candidate_result_format_error;
    logic lm_started, lm_finished;
    logic candidate_started, candidate_finished;
    logic feature_running, writer_running, table_enabled;

    logic sequence_start_valid, sequence_start_ready;
    logic sequence_done_valid, sequence_done_ready;
    logic sequence_error;
    logic [7:0] sequence_error_id;
    logic [7:0] sequence_prediction_count;
    logic descriptor_valid, descriptor_ready;
    logic [255:0] descriptor_data;
    logic descriptor_lookup_valid, descriptor_lookup_ready;
    logic [6:0] descriptor_lookup_index;
    logic descriptor_lookup_rsp_valid, descriptor_lookup_rsp_ready;
    logic [255:0] descriptor_lookup_rsp_data;
    logic sequence_abort_ack;
    logic sequence_meta_request_valid, sequence_meta_request_ready;
    logic [63:0] sequence_meta_request_address;
    logic [31:0] sequence_meta_request_bytes;
    logic [7:0] sequence_meta_request_tag;
    logic sequence_meta_data_valid, sequence_meta_data_ready;
    logic sequence_meta_done_pulse, sequence_meta_error;

    logic quant_start_valid, quant_start_ready;
    logic [6:0] quant_start_prediction_base;
    logic [3:0] quant_start_group;
    logic [3:0] quant_start_rows;
    logic quant_done_valid, quant_done_ready;
    logic quant_error;
    logic [3:0] quant_error_id;

    logic lm_start_valid, lm_start_ready;
    logic lm_done_valid, lm_done_ready;
    logic lm_error;
    logic [7:0] lm_error_id;
    logic lm_logit_valid, lm_logit_ready;
    logic [6:0] lm_logit_row_base;
    logic [16:0] lm_logit_vocab_base;
    logic [1023:0] lm_logit_values;
    logic [63:0] lm_logit_lane_mask;
    logic candidate_logit_valid, candidate_logit_ready;
    logic [6:0] candidate_logit_row_base;
    logic [16:0] candidate_logit_vocab_base;
    logic [1023:0] candidate_logit_values;
    logic [63:0] candidate_logit_lane_mask;
    logic [135:0] candidate_logit_selected_token_ids;
    logic [63:0] unused_operand_count;
    logic [63:0] unused_completed_logit_count;
    logic [63:0] unused_operand_stall_cycles;
    logic [63:0] unused_local_request_count;
    logic [63:0] unused_local_response_count;
    logic lm_abort_ack;

    logic candidate_start_valid, candidate_start_ready;
    logic candidate_done_valid, candidate_done_ready;
    logic candidate_error;
    logic [7:0] candidate_error_id;
    logic candidate_result_valid, candidate_result_ready;
    logic [6:0] candidate_result_row;
    logic [16:0] candidate_result_top1;
    logic [15:0] candidate_result_selected_probability;
    logic [15:0] candidate_result_action_confidence;
    logic candidate_result_suppressed;
    logic [63:0] unused_candidate_logit_count;
    logic candidate_abort_ack;

    logic candidate_bf16_abort, candidate_bf16_abort_ack;
    logic candidate_bf16_req_valid, candidate_bf16_req_ready;
    hardware_types_pkg::bf16_request_t candidate_bf16_req;
    logic candidate_bf16_rsp_valid, candidate_bf16_rsp_ready;
    hardware_types_pkg::bf16_response_t candidate_bf16_rsp;
    logic candidate_max_req_valid, candidate_max_req_ready;
    hardware_types_pkg::maximum_request_t candidate_max_req;
    logic candidate_max_rsp_valid, candidate_max_rsp_ready;
    hardware_types_pkg::maximum_response_t candidate_max_rsp;
    logic quant_max_req_valid, quant_max_req_ready;
    hardware_types_pkg::maximum_request_t quant_max_req;
    logic quant_max_rsp_valid, quant_max_rsp_ready;
    hardware_types_pkg::maximum_response_t quant_max_rsp;

    logic feature_start_valid, feature_start_ready;
    logic feature_row_valid, feature_row_ready;
    logic psme_next_valid, psme_next_ready;
    logic feature_done_valid, feature_done_ready;
    logic feature_error;
    logic [7:0] feature_error_id;
    logic feature_bf16_abort, feature_bf16_abort_ack;
    logic feature_bf16_req_valid, feature_bf16_req_ready;
    hardware_types_pkg::bf16_request_t feature_bf16_req;
    logic feature_bf16_rsp_valid, feature_bf16_rsp_ready;
    hardware_types_pkg::bf16_response_t feature_bf16_rsp;
    logic [4:0] psme_next_index;
    logic [31:0] psme_next_token_id, psme_next_last_top1;
    logic signed [15:0] psme_next_precision_age;
    logic [10:0] psme_next_token_position;
    logic [1:0] psme_next_state, psme_next_origin;
    logic [3:0] psme_next_bits;
    logic psme_next_cache_valid, psme_next_refresh_required;
    logic psme_next_prediction_flag;
    logic feature_row_prediction_flag;
    logic psme_next_source_a_pending;
    logic [1:0] psme_next_change_flags;
    logic [15:0] psme_next_action_confidence;
    logic psme_next_action_confidence_valid;
    logic [15:0] psme_next_change_confidence;
    logic [31:0] feature_event_mask [0:11];
    logic feature_abort_ack;

    logic aux_start_valid, aux_start_ready;
    logic [63:0] aux_start_address;
    logic [31:0] aux_start_bytes;
    logic aux_data_valid, aux_data_ready, aux_data_last;
    logic [127:0] aux_data;
    logic [15:0] aux_data_byte_enable;
    logic [31:0] aux_data_offset;
    logic aux_done_pulse, aux_error, aux_abort_ack;
    logic aux_request_valid, aux_request_ready;
    logic [63:0] aux_request_address;
    logic [31:0] aux_request_bytes;
    logic [7:0] aux_request_tag;
    logic aux_read_valid, aux_read_ready;
    logic auxiliary_reader_selected;

    logic state_writer_start_valid, state_writer_start_ready;
    logic state_writer_state_valid, state_writer_state_ready;
    logic [127:0] state_writer_state_data;
    logic state_writer_event_valid, state_writer_event_ready;
    logic [127:0] state_writer_event_data;
    logic state_writer_done_valid, state_writer_done_ready;
    logic state_writer_error;
    logic writer_abort_ack;
    logic completion_published;
    logic [63:0] accepted_state_beat_count;
    logic [63:0] accepted_event_beat_count;
    logic [63:0] accepted_metadata_beat_count;
    logic configuration_is_v4;
    logic saved_emit_metadata;

    logic [1:0] packer_round;
    logic packer_start_pending;
    logic packer_all_done;
    logic [7:0] metadata_store_count;
    logic [7:0] metadata_read_count;
    logic metadata_read_pending;
    logic metadata_data_valid;
    logic [127:0] metadata_data;
    logic metadata_data_ready;
    logic [15:0] next_metadata_bytes;
    logic [7:0] next_metadata_beats;
    logic metadata_read_request;
    logic next_token_id_valid;

    logic candidate_result_read_pending;
    logic candidate_result_data_valid;
    logic [6:0] candidate_result_entry;
    logic [49:0] candidate_result_payload;

    typedef enum logic [2:0] {
        LOOKUP_IDLE,
        LOOKUP_READ_LOW,
        LOOKUP_WAIT_LOW,
        LOOKUP_READ_HIGH,
        LOOKUP_WAIT_HIGH,
        LOOKUP_RESPONSE
    } lookup_state_t;

    function automatic logic [15:0] metadata_token_batch_bytes(
        input logic [6:0] rows);
        logic [6:0] inverse_bytes;
        begin
            inverse_bytes = ((rows + 7'd15) >> 4) << 4;
            metadata_token_batch_bytes = 16'd32 + {5'd0, rows, 4'd0} +
                {9'd0, inverse_bytes};
        end
    endfunction

    assign configuration_is_v4 =
        start_configuration_bits[EXECUTION_CONFIG_MAGIC_OFFSET*8 +: 32] ==
            32'h344e4c44 &&
        start_configuration_bits[EXECUTION_CONFIG_VERSION_OFFSET*8 +: 16] == 16'd4;
    assign next_token_id_valid = psme_next_token_id < 32'd126464;
    always_comb begin
        if (!saved_emit_metadata)
            next_metadata_bytes = 16'd0;
        else if (saved_prediction_count <= 8'd48)
            next_metadata_bytes = metadata_token_batch_bytes(
                7'(saved_prediction_count));
        else
            next_metadata_bytes = metadata_token_batch_bytes(7'd48) +
                metadata_token_batch_bytes(7'(saved_prediction_count - 8'd48));
        next_metadata_beats = 8'(next_metadata_bytes >> 4);
    end
    lookup_state_t lookup_state;
    logic [6:0] lookup_index;
    logic [127:0] lookup_low_word;

    function automatic logic [31:0] position_mask(input logic [5:0] count);
        position_mask = count == 6'd32 ? 32'hffff_ffff :
            (32'd1 << count) - 32'd1;
    endfunction

    function automatic logic [5:0] block_position_count(input logic [1:0] slot);
        block_position_count = block_config[slot].position_count;
    endfunction

    function automatic logic [6:0] block_current_entry(input logic [1:0] slot);
        block_current_entry = block_config[slot].current_entry;
    endfunction

    function automatic logic [6:0] block_next_entry(input logic [1:0] slot);
        block_next_entry = block_config[slot].next_entry;
    endfunction

    assign start_ready = state == IDLE;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign active = state != IDLE && state != COMPLETE;

    assign sequence_start_valid = state == SEQUENCE_START;
    assign sequence_done_ready = state == SEQUENCE_WAIT;
    assign descriptor_ready = descriptor_count < saved_prediction_count &&
        descriptor_store_high && post_state_write_ready;
    assign descriptor_lookup_ready = lookup_state == LOOKUP_IDLE;

    final_output_sequencer sequencer (
        .clk, .rst, .abort_request(state == ERROR_DRAIN),
        .abort_ack(sequence_abort_ack),
        .abort_wait_status(),
        .start_valid(sequence_start_valid), .start_ready(sequence_start_ready),
        .start_post_config_base(saved_post_config_base),
        .start_post_config_limit(saved_post_config_limit),
        .start_prediction_base(saved_prediction_base),
        .start_prediction_limit(saved_prediction_limit),
        .start_prediction_count(saved_prediction_count),
        .start_final_hidden_base(saved_final_hidden_base),
        .start_final_hidden_limit(saved_final_hidden_limit),
        .dma_request_valid(sequence_meta_request_valid),
        .dma_request_ready(sequence_meta_request_ready),
        .dma_request_address(sequence_meta_request_address),
        .dma_request_bytes(sequence_meta_request_bytes),
        .dma_request_tag(sequence_meta_request_tag),
        .dma_response_valid(sequence_meta_data_valid),
        .dma_response_ready(sequence_meta_data_ready),
        .dma_response_data(meta_data.data),
        .dma_response_byte_enable(meta_data.byte_enable),
        .dma_response_last(meta_data.last),
        .dma_done_pulse(sequence_meta_done_pulse),
        .dma_error(sequence_meta_error),
        .hidden_start_valid, .hidden_start_ready, .hidden_start_address,
        .hidden_start_limit, .hidden_start_bytes, .hidden_start_rows,
        .hidden_start_ddr_row_index, .hidden_start_source_round,
        .hidden_start_source_row, .hidden_done_valid, .hidden_done_ready,
        .hidden_error, .rms_start_valid, .rms_start_ready, .rms_start_rows,
        .rms_start_elements, .rms_start_epsilon_bf16,
        .rms_start_gamma_bypass, .rms_done_valid, .rms_done_ready,
        .rms_error, .quant_start_valid, .quant_start_ready,
        .quant_start_prediction_base, .quant_start_group,
        .quant_start_rows, .quant_done_valid, .quant_done_ready,
        .quant_error, .descriptor_valid, .descriptor_ready,
        .descriptor_data, .descriptor_lookup_valid,
        .descriptor_lookup_ready, .descriptor_lookup_index,
        .descriptor_lookup_rsp_valid, .descriptor_lookup_rsp_ready,
        .descriptor_lookup_rsp_data, .done_valid(sequence_done_valid),
        .done_ready(sequence_done_ready), .error(sequence_error),
        .error_id(sequence_error_id),
        .prediction_count(sequence_prediction_count), .post_config_bits);

    final_output_quantizer quantizer (
        .clk, .rst, .start_valid(quant_start_valid),
        .start_ready(quant_start_ready), .start_group(quant_start_group),
        .start_rows(quant_start_rows), .source_req_valid(rms_source_req_valid),
        .source_req_ready(rms_source_req_ready), .source_req(rms_source_req),
        .source_rsp_valid(rms_source_rsp_valid),
        .source_rsp_ready(rms_source_rsp_ready), .source_rsp(rms_source_rsp),
        .max_req_valid(quant_max_req_valid),
        .max_req_ready(quant_max_req_ready), .max_req(quant_max_req),
        .max_rsp_valid(quant_max_rsp_valid),
        .max_rsp_ready(quant_max_rsp_ready), .max_rsp(quant_max_rsp),
        .quant_scale_req_valid, .quant_scale_req_ready, .quant_scale_req,
        .quant_scale_rsp_valid, .quant_scale_rsp_ready, .quant_scale_rsp,
        .quant_values_req_valid, .quant_values_req_ready, .quant_values_req,
        .quant_values_rsp_valid, .quant_values_rsp_ready, .quant_values_rsp,
        .writer_cfg_valid, .writer_cfg_ready, .writer_cfg,
        .writer_scale_valid(quant_writer_scale_valid),
        .writer_scale_ready(quant_writer_scale_ready), .writer_scale,
        .writer_values_valid, .writer_values_ready, .writer_values,
        .writer_done_pulse, .writer_error, .done_valid(quant_done_valid),
        .done_ready(quant_done_ready), .error(quant_error),
        .error_id(quant_error_id));

    assign lm_start_valid = state == LM_CANDIDATE_START &&
        candidate_start_ready;
    assign candidate_start_valid = state == LM_CANDIDATE_START &&
        lm_start_ready;
    assign lm_done_ready = state == LM_WAIT;
    assign candidate_done_ready = state == CANDIDATE_WAIT ||
        (state == LM_WAIT && candidate_done_valid && candidate_error);
    assign pe_req.mixed_phase = 1'b0;
    assign pe_req.mixed_phase_first = 1'b0;
    assign pe_req.mixed_a8_rows = 8'd0;

    lm_head_controller #(.PANEL_COUNT(LM_PANEL_COUNT)) lm_head (
        .clk, .rst, .abort_request(
            state == ERROR_DRAIN && lm_started && !lm_finished),
        .abort_ack(lm_abort_ack),
        .pe_abort_request, .pe_abort_ack, .rescale_abort_request,
        .rescale_abort_ack, .start_valid(lm_start_valid),
        .start_ready(lm_start_ready), .start_rows(saved_prediction_count[6:0]),
        .start_panel_count(14'((post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 17] +
            17'd7) >> 3)),
        .start_weight_base(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_RAW_W8_BASE_OFFSET*8 +: 64]),
        .start_weight_limit(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_RAW_W8_LIMIT_OFFSET*8 +: 64]),
        .start_scale_base(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_WEIGHT_SCALE_BASE_OFFSET*8 +: 64]),
        .start_scale_limit(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_WEIGHT_SCALE_LIMIT_OFFSET*8 +: 64]),
        .activation_scale_load_valid(
            quant_writer_scale_valid && writer_scale_ready),
        .activation_scale_load_ready(lm_activation_scale_ready),
        .activation_scale_load_group(quant_start_group),
        .activation_scale_load_data(writer_scale.values_bf16),
        .dma_request_valid(lm_request_valid),
        .dma_request_ready(lm_request_ready),
        .dma_request_address(lm_request.byte_address),
        .dma_request_bytes(lm_request.byte_count),
        .dma_request_tag(lm_request.tag), .dma_response_valid(lm_data_valid),
        .dma_response_ready(lm_data_ready), .dma_response_data(lm_data.data),
        .dma_response_byte_enable(lm_data.byte_enable),
        .dma_response_last(lm_data.last),
        .dma_done_pulse(lm_completion.done_pulse),
        .dma_error(lm_completion.error),
        .local_scale_write_valid(scale_write_valid),
        .local_scale_write_ready(scale_write_ready),
        .local_scale_write_address(scale_write_address),
        .local_scale_write_data(scale_write_data),
        .local_scale_read_valid(scale_read_valid),
        .local_scale_read_ready(scale_read_ready),
        .local_scale_read_address(scale_read_address),
        .local_scale_read_rsp_valid(scale_read_rsp_valid),
        .local_scale_read_rsp_data(scale_read_rsp_data),
        .local_panel_write_valid(panel_write_valid),
        .local_panel_write_ready(panel_write_ready),
        .local_panel_write(panel_write), .local_read_valid(operand_read_valid),
        .local_read_ready(operand_read_ready), .local_read(operand_read),
        .local_rsp_valid(operand_rsp_valid), .local_rsp(operand_rsp),
        .pe_req_valid, .pe_req_ready, .pe_req_mode(pe_req.mode),
        .pe_req_activation(pe_req.activation_payload),
        .pe_req_weight(pe_req.weight_payload),
        .pe_req_row_mask(pe_req.row_mask), .pe_req_k_mask(pe_req.k_mask),
        .pe_req_column_mask(pe_req.column_mask),
        .pe_req_first_k_step(pe_req.first_k_step),
        .pe_req_last_k_step(pe_req.last_k_step),
        .pe_req_activation_scales_bf16(pe_req.activation_scales),
        .pe_req_weight_scales_bf16(pe_req.weight_scales),
        .pe_req_tag(pe_req.tag), .pe_accum_valid(pe_rsp_valid),
        .pe_accum_ready(pe_rsp_ready),
        .pe_accumulators(pe_rsp.accumulators),
        .pe_accum_mask(pe_rsp.accumulator_mask),
        .pe_accum_mode(pe_rsp.mode),
        .pe_accum_last_k_step(pe_rsp.last_k_step),
        .pe_accum_activation_scales_bf16(pe_rsp.activation_scales),
        .pe_accum_weight_scales_bf16(pe_rsp.weight_scales),
        .pe_accum_tag(pe_rsp.tag), .pe_idle,
        .rescale_req_valid, .rescale_req_ready,
        .rescale_req_accumulators(rescale_req.accumulators),
        .rescale_req_activation_scales_bf16(rescale_req.activation_scales),
        .rescale_req_weight_scales_bf16(rescale_req.weight_scales),
        .rescale_req_lane_mask(rescale_req.lane_mask),
        .rescale_req_tag(rescale_req.tag), .rescale_rsp_valid,
        .rescale_rsp_ready, .rescale_rsp_values(rescale_rsp.values),
        .rescale_rsp_lane_mask(rescale_rsp.lane_mask),
        .rescale_rsp_tag(rescale_rsp.tag), .rescale_idle,
        .logit_valid(lm_logit_valid), .logit_ready(lm_logit_ready),
        .logit_row_base(lm_logit_row_base),
        .logit_vocab_base(lm_logit_vocab_base),
        .logit_values_bf16(lm_logit_values),
        .logit_lane_mask(lm_logit_lane_mask), .done_valid(lm_done_valid),
        .done_ready(lm_done_ready), .error(lm_error), .error_id(lm_error_id),
        .accepted_weight_bytes, .accepted_scale_bytes,
        .accepted_operand_count(unused_operand_count), .accepted_pe_count,
        .completed_logit_count(unused_completed_logit_count),
        .operand_stall_cycles(unused_operand_stall_cycles),
        .local_request_count(unused_local_request_count),
        .local_response_count(unused_local_response_count));

    assign lm_logit_ready = candidate_logit_ready;
    assign writer_scale_valid = quant_writer_scale_valid &&
        lm_activation_scale_ready;
    assign quant_writer_scale_ready = writer_scale_ready &&
        lm_activation_scale_ready;
    assign candidate_logit_valid = lm_logit_valid;
    assign candidate_logit_row_base = lm_logit_row_base;
    assign candidate_logit_vocab_base = lm_logit_vocab_base;
    assign candidate_logit_values = lm_logit_values;
    assign candidate_logit_lane_mask = lm_logit_lane_mask;
    assign candidate_logit_selected_token_ids =
        selected_token_group[lm_logit_row_base[6:3]];

    candidate_reducer candidate (
        .clk, .rst, .abort_request(
            state == ERROR_DRAIN && candidate_started && !candidate_finished),
        .abort_ack(candidate_abort_ack),
        .start_valid(candidate_start_valid),
        .start_ready(candidate_start_ready),
        .start_row_count(saved_prediction_count),
        .start_vocabulary_size(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 17]),
        .start_suppressed_count(saved_suppressed_count),
        .logit_valid(candidate_logit_valid),
        .logit_ready(candidate_logit_ready),
        .logit_row_base(candidate_logit_row_base),
        .logit_vocab_base(candidate_logit_vocab_base),
        .logit_values(candidate_logit_values),
        .logit_lane_mask(candidate_logit_lane_mask),
        .logit_selected_token_ids(candidate_logit_selected_token_ids),
        .candidate_state_read_valid, .candidate_state_read_ready,
        .candidate_state_read_row_group,
        .candidate_state_read_lane_block, .candidate_state_read_row_half,
        .candidate_state_read_word, .candidate_state_read_row_mask,
        .candidate_state_read_tag, .candidate_state_read_rsp_valid,
        .candidate_state_read_rsp_ready, .candidate_state_read_rsp_data,
        .candidate_state_read_rsp_row_mask, .candidate_state_read_rsp_tag,
        .candidate_state_write_valid, .candidate_state_write_ready,
        .candidate_state_write_row_group,
        .candidate_state_write_lane_block, .candidate_state_write_row_half,
        .candidate_state_write_word, .candidate_state_write_row_mask,
        .candidate_state_write_data, .bf16_abort_request(candidate_bf16_abort),
        .bf16_abort_ack(candidate_bf16_abort_ack),
        .bf16_req_valid(candidate_bf16_req_valid),
        .bf16_req_ready(candidate_bf16_req_ready),
        .bf16_req(candidate_bf16_req),
        .bf16_rsp_valid(candidate_bf16_rsp_valid),
        .bf16_rsp_ready(candidate_bf16_rsp_ready),
        .bf16_rsp(candidate_bf16_rsp), .max_req_valid(candidate_max_req_valid),
        .max_req_ready(candidate_max_req_ready), .max_req(candidate_max_req),
        .max_rsp_valid(candidate_max_rsp_valid),
        .max_rsp_ready(candidate_max_rsp_ready), .max_rsp(candidate_max_rsp),
        .reduction_req_valid, .reduction_req_ready, .reduction_req,
        .reduction_rsp_valid, .reduction_rsp_ready, .reduction_rsp,
        .exp_req_valid, .exp_req_ready, .exp_req, .exp_rsp_valid,
        .exp_rsp_ready, .exp_rsp, .reciprocal_req_valid,
        .reciprocal_req_ready, .reciprocal_req, .reciprocal_rsp_valid,
        .reciprocal_rsp_ready, .reciprocal_rsp,
        .suppressed_read_valid(post_suppressed_read_valid),
        .suppressed_read_ready(post_suppressed_read_ready),
        .suppressed_read_index(post_suppressed_read_index),
        .suppressed_rsp_valid(post_suppressed_read_rsp_valid),
        .suppressed_rsp_ready(),
        .suppressed_rsp_token_id(post_suppressed_read_rsp_token_id),
        .result_valid(candidate_result_valid),
        .result_ready(candidate_result_ready),
        .result_row_index(candidate_result_row),
        .result_top1_token_id(candidate_result_top1),
        .result_top_logit_bf16(), .result_raw_confidence_bf16(),
        .result_selected_token_logit_bf16(),
        .result_selected_probability_bf16(
            candidate_result_selected_probability),
        .result_suppressed_winner(candidate_result_suppressed),
        .result_action_confidence_bf16(candidate_result_action_confidence),
        .done_valid(candidate_done_valid), .done_ready(candidate_done_ready),
        .error(candidate_error), .error_id(candidate_error_id),
        .accepted_logit_count(unused_candidate_logit_count),
        .logit_wait_cycle_count(candidate_wait_cycles));

    operator_dma_reader #(.REQUEST_TAG(8'hb4)) metadata_reader (
        .clk, .rst, .abort_request(1'b0), .start_valid(aux_start_valid),
        .start_ready(aux_start_ready), .start_address(aux_start_address),
        .start_bytes(aux_start_bytes), .start_total_bytes(aux_start_bytes),
        .request_valid(aux_request_valid),
        .request_ready(aux_request_ready),
        .request_address(aux_request_address),
        .request_bytes(aux_request_bytes), .request_tag(aux_request_tag),
        .read_valid(aux_read_valid), .read_ready(aux_read_ready),
        .read_data(meta_data.data),
        .read_byte_enable(meta_data.byte_enable),
        .read_last(meta_data.last), .read_tag(meta_data.tag),
        .response_error(meta_completion.error),
        .upstream_abort_ack(1'b0),
        .data_valid(aux_data_valid), .data_ready(aux_data_ready),
        .data(aux_data), .data_byte_enable(aux_data_byte_enable),
        .data_last(aux_data_last), .data_byte_offset(aux_data_offset),
        .done_pulse(aux_done_pulse), .error(aux_error),
        .abort_ack(aux_abort_ack));

    draft_verify_state_controller draft_verify (
        .start_canonical_future(block_config[active_block_slot[1:0]].canonical_future),
        .start_source_a_handoff(block_config[active_block_slot[1:0]].source_a_handoff),
        .row_source_a(block_config[active_block_slot[1:0]].canonical_future && saved_source_a_valid &&
            source_a_local < 12'd32 && saved_source_a_mask[source_a_local[4:0]]),
        .row_source_a_pending(state_entry_high[123]),
        .next_token_source_a_pending(psme_next_source_a_pending),
        .start_transfer_only(block_config[active_block_slot[1:0]].transfer_only),
        .start_observed_mask(block_config[active_block_slot[1:0]].observed_mask),
        .clk, .rst, .abort_request(state == ERROR_DRAIN && feature_running),
        .abort_ack(feature_abort_ack),
        .start_valid(feature_start_valid), .start_ready(feature_start_ready),
        .start_row_count(active_block_rows),
        .start_masked_mask(block_masked[active_block_slot]),
        .start_tentative_mask(block_tentative[active_block_slot]),
        .start_locked_mask(position_mask(active_block_rows) &
            ~(block_masked[active_block_slot] |
              block_tentative[active_block_slot])),
        .start_mask_token_id(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_MASK_TOKEN_ID_OFFSET*8 +: 32]),
        .start_vocabulary_size(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 17]),
        .start_high_confidence_threshold_bf16(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_HIGH_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]),
        .start_tail_high_confidence_threshold_bf16(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_TAIL_HIGH_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]),
        .start_low_confidence_threshold_bf16(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_LOW_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]),
        .start_verify_threshold_bf16(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_VERIFY_THRESHOLD_BF16_OFFSET*8 +: 16]),
        .start_stability_bonus_bf16(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_STABILITY_BONUS_BF16_OFFSET*8 +: 16]),
        .start_budget_scale_bf16(block_config[active_block_slot[1:0]].canonical_future ?
            16'h3f80 : post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_BUDGET_SCALE_BF16_OFFSET*8 +: 16]),
        .start_max_handoff_tokens(block_config[active_block_slot[1:0]].max_handoff_tokens),
        .start_scheduled_quota(
            block_config[active_block_slot[1:0]].scheduled_quota),
        .start_remaining_forwards(
            block_config[active_block_slot[1:0]].remaining_forwards),
        .start_step_index(block_config[active_block_slot[1:0]].step_index),
        .start_tail_after_step(
            block_config[active_block_slot[1:0]].tail_after_step),
        .start_maturity_age(
            block_config[active_block_slot[1:0]].maturity_age),
        .start_tail_threshold_enable(
            block_config[active_block_slot[1:0]].tail_threshold_enable),
        .start_tail_all(1'b0),
        .start_tail_bypass_all(block_config[active_block_slot[1:0]].tail_bypass_all),
        .start_tail_bypass_stable_only(block_config[active_block_slot[1:0]].tail_bypass_stable_only),
        .start_closeout_kind(block_config[active_block_slot[1:0]].closeout_kind),
        .row_valid(feature_row_valid), .row_ready(feature_row_ready),
        .row_index(active_block_row[4:0]),
        .row_token_id(state_entry_low[95:64]),
        .row_last_top1(state_entry_low[127:96]),
        .row_precision_age(state_entry_high[15:0]),
        .row_token_position(state_entry_low[10:0]),
        .row_origin(state_entry_low[49:48]),
        .activation_bits(state_descriptor_valid[active_current_entry] ?
            expected_descriptor_data[20:17] : state_entry_low[59:56]),
        .row_cache_valid(state_entry_high[16]),
        .row_refresh_required(state_entry_high[24]),
        .row_prediction_flag(feature_row_prediction_flag),
        .row_candidate_top1({15'd0, state_entry_high[121:105]}),
        .row_selected_probability_bf16(state_entry_high[72 +: 16]),
        .row_action_confidence_bf16(state_entry_high[88 +: 16]),
        .row_action_confidence_valid(state_entry_high[122]),
        .row_suppressed_winner(state_entry_high[104]),
        .bf16_abort_request(feature_bf16_abort),
        .bf16_abort_ack(feature_bf16_abort_ack),
        .bf16_req_valid(feature_bf16_req_valid),
        .bf16_req_ready(feature_bf16_req_ready), .bf16_req(feature_bf16_req),
        .bf16_rsp_valid(feature_bf16_rsp_valid),
        .bf16_rsp_ready(feature_bf16_rsp_ready), .bf16_rsp(feature_bf16_rsp),
        .next_token_valid(psme_next_valid), .next_token_ready(psme_next_ready),
        .next_token_index(psme_next_index),
        .next_token_token_id(psme_next_token_id),
        .next_token_last_top1(psme_next_last_top1),
        .next_token_precision_age(psme_next_precision_age),
        .next_token_token_position(psme_next_token_position),
        .next_token_state(psme_next_state), .next_token_origin(psme_next_origin),
        .next_activation_bits(psme_next_bits),
        .next_token_cache_valid(psme_next_cache_valid),
        .next_token_refresh_required(psme_next_refresh_required),
        .next_token_prediction_flag(psme_next_prediction_flag),
        .next_token_change_flags(psme_next_change_flags),
        .next_token_change_confidence_bf16(psme_next_change_confidence),
        .next_token_action_confidence_bf16(psme_next_action_confidence),
        .next_token_action_confidence_valid(psme_next_action_confidence_valid),
        .done_valid(feature_done_valid), .done_ready(feature_done_ready),
        .error(feature_error), .error_id(feature_error_id),
        .confirmed_mask(feature_event_mask[0]),
        .remasked_mask(feature_event_mask[1]),
        .selected_mask(feature_event_mask[2]),
        .direct_locked_mask(feature_event_mask[3]),
        .stable_tentative_mask(feature_event_mask[4]),
        .fallback_tentative_mask(feature_event_mask[5]),
        .mandatory_refresh_mask(feature_event_mask[6]),
        .cache_commit_mask(feature_event_mask[7]),
        .cache_invalidate_mask(feature_event_mask[8]),
        .cache_keep_mask(feature_event_mask[9]),
        .token_changed_mask(feature_event_mask[10]),
        .tail_closed_mask(feature_event_mask[11]));

    assign state_writer_start_valid = state == OUTPUT_WRITER_START;
    assign state_writer_done_ready = writer_running;

    token_state_writer state_writer (
        .clk, .rst, .abort_request(state == ERROR_DRAIN && writer_running),
        .abort_ack(writer_abort_ack),
        .start_valid(state_writer_start_valid),
        .start_ready(state_writer_start_ready),
        .start_state_base(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_NEXT_STATE_BASE_OFFSET*8 +: 64]),
        .start_state_limit(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_NEXT_STATE_LIMIT_OFFSET*8 +: 64]),
        .start_state_entry_count(total_state_entries),
        .start_metadata_base(saved_emit_metadata ?
            saved_next_metadata_base : 64'd0),
        .start_metadata_limit(saved_emit_metadata ?
            saved_next_metadata_limit : 64'd0),
        .start_metadata_bytes(next_metadata_bytes),
        .start_event_base(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_FORWARD_EVENT_BASE_OFFSET*8 +: 64]),
        .start_event_limit(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_FORWARD_EVENT_LIMIT_OFFSET*8 +: 64]),
        .start_block_count(saved_block_count),
        .start_completion_base(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_COMPLETION_BASE_OFFSET*8 +: 64]),
        .start_completion_limit(post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_COMPLETION_LIMIT_OFFSET*8 +: 64]),
        .start_completion_data({16'd0, 8'd0, {5'd0, saved_block_count},
            block_config[0].capture_index,
            post_config_bits[FORWARD_POSTPROCESS_CONFIG_COMPLETION_VERSION_OFFSET*8 +: 32],
            32'h31444250}),
        .state_data_valid(state_writer_state_valid),
        .state_data_ready(state_writer_state_ready),
        .state_data(state_writer_state_data),
        .metadata_data_valid(metadata_data_valid),
        .metadata_data_ready(metadata_data_ready),
        .metadata_data(metadata_data),
        .event_data_valid(state_writer_event_valid),
        .event_data_ready(state_writer_event_ready),
        .event_data(state_writer_event_data),
        .memory_request_valid(write_request_valid),
        .memory_request_ready(write_request_ready),
        .memory_request_address(write_request.byte_address),
        .memory_request_bytes(write_request.byte_count),
        .memory_request_tag(write_request.tag),
        .memory_data_valid(write_data_valid),
        .memory_data_ready(write_data_ready),
        .memory_data(write_data.data),
        .memory_data_byte_enable(write_data.byte_enable),
        .memory_data_last(write_data.last),
        .memory_done_pulse(write_completion.done_pulse),
        .memory_error(write_completion.error),
        .done_valid(state_writer_done_valid),
        .done_ready(state_writer_done_ready), .error(state_writer_error),
        .completion_published, .accepted_state_beat_count,
        .accepted_metadata_beat_count,
        .accepted_event_beat_count);

    assign source_a_local = {1'b0, state_entry_low[10:0]} - {1'b0, saved_source_a_block_start};

    assign feature_row_prediction_flag = state_entry_low[41:40] != 2 &&
        (block_config[active_block_slot[1:0]].closeout_kind == 0 ||
         (block_config[active_block_slot[1:0]].closeout_kind == 1 && state_entry_low[41:40] == 1) ||
         (block_config[active_block_slot[1:0]].closeout_kind == 2 && state_entry_low[41:40] == 0)) &&
        (!block_config[active_block_slot[1:0]].canonical_future ||
         block_config[active_block_slot[1:0]].observed_mask[active_block_row[4:0]]);

    assign packer_abort_request = state == ERROR_DRAIN;
    assign packer_config.row_count = packer_round == 0 ?
        (saved_prediction_count > 8'd48 ? 6'd48 : 6'(saved_prediction_count)) :
        6'(saved_prediction_count - 8'd48);
    assign packer_config.sequence_length = saved_sequence_length;
    assign packer_config.token_batch_index = {14'd0, packer_round};
    assign packer_config.first_token_ordinal = packer_round == 0 ? 16'd0 : 16'd48;
    assign packer_config.metadata_version = post_config_bits[
        FORWARD_POSTPROCESS_CONFIG_COMPLETION_VERSION_OFFSET*8 +: 32];
    assign packer_config.capture_index = block_config[0].capture_index;
    assign packer_row.index = 6'(output_state_entry + psme_next_index) - (packer_round == 0 ? 6'd0 : 6'd48);
    assign packer_row.source_index = psme_next_token_id[16:0];
    assign packer_row.token_position = psme_next_token_position;
    assign packer_row.kv_index = psme_next_token_position;
    assign packer_row.embedding_source = 1'b1;
    assign packer_row.kv_write_disable = psme_next_source_a_pending;
    assign packer_row.bits = psme_next_bits;
    assign packer_row.query_group = descriptor_query_group[7'(output_state_entry + 8'(psme_next_index))];
    assign packer_row.cache_group = descriptor_cache_group[7'(output_state_entry + 8'(psme_next_index))];

    assign rescale_req.qk_scale_bf16 = 16'h3f80;
    assign rescale_req.rescale_mode = 2'd0;

    always_comb begin : shared_arithmetic_select
        bf16_abort_request = candidate_bf16_abort || feature_bf16_abort;
        candidate_bf16_abort_ack = bf16_abort_ack && candidate_bf16_abort;
        feature_bf16_abort_ack = bf16_abort_ack && feature_bf16_abort;
        bf16_req_valid = feature_bf16_req_valid || candidate_bf16_req_valid;
        bf16_req = feature_bf16_req_valid ? feature_bf16_req : candidate_bf16_req;
        feature_bf16_req_ready = bf16_req_ready && feature_bf16_req_valid;
        candidate_bf16_req_ready = bf16_req_ready &&
            !feature_bf16_req_valid;
        feature_bf16_rsp_valid = bf16_rsp_valid &&
            (state >= FEATURE_START && state <= FEATURE_WAIT);
        candidate_bf16_rsp_valid = bf16_rsp_valid && !feature_bf16_rsp_valid;
        feature_bf16_rsp = bf16_rsp;
        candidate_bf16_rsp = bf16_rsp;
        bf16_rsp_ready = feature_bf16_rsp_valid ? feature_bf16_rsp_ready :
            candidate_bf16_rsp_ready;

        max_req_valid = quant_max_req_valid || candidate_max_req_valid;
        max_req = quant_max_req_valid ? quant_max_req : candidate_max_req;
        quant_max_req_ready = max_req_ready && quant_max_req_valid;
        candidate_max_req_ready = max_req_ready && !quant_max_req_valid;
        quant_max_rsp_valid = max_rsp_valid &&
            (state == SEQUENCE_START || state == SEQUENCE_WAIT);
        candidate_max_rsp_valid = max_rsp_valid && !quant_max_rsp_valid;
        quant_max_rsp = max_rsp;
        candidate_max_rsp = max_rsp;
        max_rsp_ready = quant_max_rsp_valid ? quant_max_rsp_ready :
            candidate_max_rsp_ready;
    end

    assign auxiliary_reader_selected = state >= BLOCK_LOAD_START &&
        state <= SUPPRESSED_LOAD_WAIT;
    assign meta_request_valid = auxiliary_reader_selected ?
        aux_request_valid : sequence_meta_request_valid;
    assign meta_request.byte_address = auxiliary_reader_selected ?
        aux_request_address : sequence_meta_request_address;
    assign meta_request.byte_count = auxiliary_reader_selected ?
        aux_request_bytes : sequence_meta_request_bytes;
    assign meta_request.tag = auxiliary_reader_selected ? aux_request_tag :
        sequence_meta_request_tag;
    assign meta_request.wide_data = 1'b0;
    assign aux_request_ready = auxiliary_reader_selected &&
        meta_request_ready;
    assign sequence_meta_request_ready = !auxiliary_reader_selected &&
        meta_request_ready;
    assign aux_read_valid = auxiliary_reader_selected && meta_data_valid;
    assign sequence_meta_data_valid = !auxiliary_reader_selected &&
        meta_data_valid;
    assign meta_data_ready = auxiliary_reader_selected ? aux_read_ready :
        sequence_meta_data_ready;
    assign sequence_meta_done_pulse = !auxiliary_reader_selected &&
        meta_completion.done_pulse;
    assign sequence_meta_error = !auxiliary_reader_selected &&
        meta_completion.error;

    assign lm_request.wide_data = 1'b0;

    always_comb begin : auxiliary_reader_routing
        aux_start_valid = state == BLOCK_LOAD_START ||
            state == STATE_LOAD_START || state == SUPPRESSED_LOAD_START;
        aux_start_address = state == BLOCK_LOAD_START ? post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_BLOCK_CONFIGURATION_BASE_OFFSET*8 +: 64] :
            state == STATE_LOAD_START ? post_config_bits[
                FORWARD_POSTPROCESS_CONFIG_CURRENT_STATE_BASE_OFFSET*8 +: 64] :
            post_config_bits[
                FORWARD_POSTPROCESS_CONFIG_SUPPRESSED_TOKEN_BASE_OFFSET*8 +: 64];
        aux_start_bytes = state == BLOCK_LOAD_START ?
            32'(saved_block_count) * 32'd32 :
            state == STATE_LOAD_START ?
                32'(total_state_entries) * 32'd32 :
                32'(saved_suppressed_count) * 32'd4;
        aux_data_ready = state == BLOCK_LOAD_WAIT ? 1'b1 :
            state == STATE_LOAD_WAIT ? post_state_write_ready :
            state == SUPPRESSED_LOAD_WAIT ? post_suppressed_write_ready :
            1'b0;
    end

    always_comb begin : state_and_event_streams
        post_state_write_valid = state == SEQUENCE_WAIT && descriptor_valid;
        post_state_write_word_address = DESCRIPTOR_WORD_BASE +
            10'(descriptor_count) * 10'd2 + descriptor_store_high;
        post_state_write_data = descriptor_store_high ?
            descriptor_data[255:128] : descriptor_data[127:0];
        if (state == STATE_LOAD_WAIT) begin
            post_state_write_valid = aux_data_valid;
            post_state_write_word_address = {3'd0, state_load_beat};
            post_state_write_data = aux_data;
            // Candidate membership belongs to this forward, not the saved state.
            if (state_load_beat[0]) begin
                post_state_write_data[64] = 1'b0;
                post_state_write_data[127:72] = '0;
                post_state_write_data[123] = aux_data[120];
                // Reuse the candidate confidence slot for unobserved history.
                if (post_config_bits[FORWARD_POSTPROCESS_CONFIG_FLAGS_OFFSET*8]) begin
                    post_state_write_data[103:88] = aux_data[111:96];
                    post_state_write_data[122] = aux_data[112];
                end
            end
        end
        post_suppressed_write_valid = state == SUPPRESSED_LOAD_WAIT &&
            aux_data_valid;
        post_suppressed_write_address = suppressed_load_beat;
        post_suppressed_write_data = aux_data;

        metadata_read_request = saved_emit_metadata && packer_all_done &&
            metadata_read_count < next_metadata_beats &&
            !metadata_read_pending && !metadata_data_valid;
        post_state_read_valid = metadata_read_request ||
            (state == SEQUENCE_WAIT &&
             (lookup_state == LOOKUP_READ_LOW ||
              lookup_state == LOOKUP_READ_HIGH)) ||
            state == DESCRIPTOR_READ_LOW ||
            state == DESCRIPTOR_READ_HIGH ||
            state == FEATURE_EXPECTED_READ ||
            state == FEATURE_READ_LOW ||
            state == FEATURE_READ_HIGH ||
            (state == OUTPUT_WRITER_WAIT &&
             event_slot < saved_block_count && !event_read_pending &&
             !event_data_valid && !packer_beat_valid) ||
            (state == CANDIDATE_WAIT && candidate_result_valid &&
             !candidate_result_read_pending && !candidate_result_data_valid);
        post_state_read_word_address = metadata_read_request ?
            NEXT_METADATA_WORD_BASE + {2'd0, metadata_read_count} :
            state == SEQUENCE_WAIT ?
            DESCRIPTOR_WORD_BASE + {2'd0, lookup_index, 1'b0} +
                (lookup_state == LOOKUP_READ_HIGH) :
            state == DESCRIPTOR_READ_LOW ?
            DESCRIPTOR_WORD_BASE + 10'(descriptor_validate_index) * 10'd2 :
            state == DESCRIPTOR_READ_HIGH ?
                DESCRIPTOR_WORD_BASE + 10'(descriptor_validate_index) * 10'd2 + 10'd1 :
            state == FEATURE_EXPECTED_READ ?
                EXPECTED_DESCRIPTOR_WORD_BASE + {3'd0, active_current_entry} :
            state == FEATURE_READ_LOW ? {2'd0, active_current_entry, 1'b0} :
            state == FEATURE_READ_HIGH ? {2'd0, active_current_entry, 1'b1} :
            state == OUTPUT_WRITER_WAIT ? EVENT_WORD_BASE +
                {5'd0, event_slot, event_word} :
            {2'd0, descriptor_current_entry[candidate_result_row], 1'b1};
        post_state_read_tag = metadata_read_request ?
            {8'hf1, metadata_read_count} : state == SEQUENCE_WAIT ?
            {8'hb2, lookup_index, lookup_state == LOOKUP_READ_HIGH} :
            state == DESCRIPTOR_READ_LOW ?
            {8'he0, descriptor_validate_index} :
            state == DESCRIPTOR_READ_HIGH ?
                {8'he1, descriptor_validate_index} : state == CANDIDATE_WAIT ?
                {8'hc1, candidate_result_row, 1'b0} :
            state == OUTPUT_WRITER_WAIT ?
                {8'hf0, 3'd0, event_slot, event_word} :
            {8'hd0, active_block_row,
             state == FEATURE_READ_HIGH, state == FEATURE_EXPECTED_READ};

        candidate_result_ready = state == CANDIDATE_WAIT &&
            candidate_result_data_valid && post_state_write_ready;
        if (state == CANDIDATE_WAIT && candidate_result_data_valid) begin
            post_state_write_valid = 1'b1;
            post_state_write_word_address = {candidate_result_entry, 1'b1};
            post_state_write_data = state_entry_high;
            post_state_write_data[64] = 1'b1;
            post_state_write_data[121:72] = candidate_result_payload;
            post_state_write_data[122] = post_config_bits[FORWARD_POSTPROCESS_CONFIG_FLAGS_OFFSET*8];
        end
        if (state == DESCRIPTOR_EXPECTED_WRITE) begin
            post_state_write_valid = 1'b1;
            post_state_write_word_address = EXPECTED_DESCRIPTOR_WORD_BASE +
                {3'd0, descriptor_expected_entry};
            post_state_write_data = {73'd0, expected_descriptor_data};
        end
        if (state == EVENT_STORE) begin
            post_state_write_valid = feature_done_valid;
            post_state_write_word_address = EVENT_WORD_BASE +
                {5'd0, active_block_slot[1:0], event_word};
            case (event_word)
                2'd0: post_state_write_data = {
                    feature_event_mask[1], feature_event_mask[0],
                    10'd0,
                    block_config[active_block_slot[1:0]].position_count,
                    2'd0,
                    block_config[active_block_slot[1:0]].global_block_id,
                    6'd0,
                    block_config[active_block_slot[1:0]].block_slot,
                    block_config[active_block_slot[1:0]].capture_index};
                2'd1: post_state_write_data = {
                    feature_event_mask[5], feature_event_mask[4],
                    feature_event_mask[3], feature_event_mask[2]};
                2'd2: post_state_write_data = {
                    feature_event_mask[9], feature_event_mask[8],
                    feature_event_mask[7], feature_event_mask[6]};
                default: post_state_write_data = {
                    64'd0, feature_event_mask[11], feature_event_mask[10]};
            endcase
        end
        if (packer_beat_valid && state != EVENT_STORE) begin
            post_state_write_valid = 1'b1;
            post_state_write_word_address = NEXT_METADATA_WORD_BASE +
                {2'd0, metadata_store_count};
            post_state_write_data = packer_beat_data;
        end

        feature_row_valid = state == FEATURE_SEND_ROW;
        packer_start_valid = packer_start_pending;
        packer_row_valid = psme_next_valid && saved_emit_metadata &&
            next_token_id_valid && state == FEATURE_WAIT && !next_state_pending &&
            !packer_start_pending;
        packer_beat_ready = post_state_write_ready && state != EVENT_STORE;
        packer_done_ready = 1'b1;
        psme_next_ready = state == FEATURE_WAIT && !next_state_pending &&
            (!saved_emit_metadata ||
             (next_token_id_valid && packer_row_ready && !packer_start_pending));
        state_writer_state_valid = next_state_pending;
        state_writer_state_data = next_state_half ?
            next_state_entry[255:128] : next_state_entry[127:0];
        state_writer_event_valid = event_data_valid;
        state_writer_event_data = event_data;
    end

    assign post_table_access_valid = state == TABLE_ENABLE ||
        state == TABLE_DISABLE;
    assign post_table_access_enable = state == TABLE_ENABLE;

    assign feature_start_valid = state == FEATURE_START;
    assign feature_done_ready =
        (feature_running && feature_error) ||
        (state == FEATURE_WAIT && (feature_error || terminal_error)) ||
        (state == EVENT_STORE && event_word == 2'd3 &&
         post_state_write_ready);

    always_ff @(posedge clk) begin : descriptor_lookup
        if (rst || state == IDLE || state == COMPLETE) begin
            lookup_state <= LOOKUP_IDLE;
            lookup_index <= '0;
            lookup_low_word <= '0;
            descriptor_lookup_rsp_valid <= 1'b0;
            descriptor_lookup_rsp_data <= '0;
        end else begin
            case (lookup_state)
                LOOKUP_IDLE: begin
                    descriptor_lookup_rsp_valid <= 1'b0;
                    if (descriptor_lookup_valid && descriptor_lookup_ready) begin
                        lookup_index <= descriptor_lookup_index;
                        lookup_state <= LOOKUP_READ_LOW;
                    end
                end
                LOOKUP_READ_LOW:
                    if (post_state_read_valid && post_state_read_ready)
                        lookup_state <= LOOKUP_WAIT_LOW;
                LOOKUP_WAIT_LOW:
                    if (post_state_read_rsp_valid) begin
                        lookup_low_word <= post_state_read_rsp_data;
                        lookup_state <= LOOKUP_READ_HIGH;
                    end
                LOOKUP_READ_HIGH:
                    if (post_state_read_valid && post_state_read_ready)
                        lookup_state <= LOOKUP_WAIT_HIGH;
                LOOKUP_WAIT_HIGH:
                    if (post_state_read_rsp_valid) begin
                        descriptor_lookup_rsp_data <= {
                            post_state_read_rsp_data, lookup_low_word};
                        descriptor_lookup_rsp_valid <= 1'b1;
                        lookup_state <= LOOKUP_RESPONSE;
                    end
                LOOKUP_RESPONSE:
                    if (descriptor_lookup_rsp_valid &&
                        descriptor_lookup_rsp_ready) begin
                        descriptor_lookup_rsp_valid <= 1'b0;
                        lookup_state <= LOOKUP_IDLE;
                    end
                default: lookup_state <= LOOKUP_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk) begin : forward_postprocess_sequence
        logic [1:0] descriptor_slot;
        logic [6:0] descriptor_current;
        logic [6:0] descriptor_index;
        logic [1:0] loaded_slot;
        logic [5:0] loaded_rows;
        logic [8:0] new_total;
        logic found_block;
        logic [1:0] found_slot;
        logic state_entry_error;

        if (rst) begin
            state <= IDLE;
            current_state_seen <= 1'b0;
            current_unresolved <= 1'b0;
            saved_post_config_base <= '0;
            saved_post_config_limit <= '0;
            saved_prediction_base <= '0;
            saved_prediction_limit <= '0;
            saved_final_hidden_base <= '0;
            saved_final_hidden_limit <= '0;
            saved_next_metadata_base <= '0;
            saved_next_metadata_limit <= '0;
            saved_sequence_length <= '0;
            saved_prediction_count <= '0;
            saved_block_count <= '0;
            saved_suppressed_count <= '0;
            total_state_entries <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            saved_source_a_valid <= 1'b0; saved_source_a_mask <= '0; saved_source_a_capture_index <= '0; saved_source_a_block_start <= '0;
            descriptor_count <= '0;
            descriptor_store_high <= 1'b0;
            block_load_beat <= '0;
            block_first_half <= '0;
            block_validate_index <= '0;
            descriptor_validate_index <= '0;
            descriptor_expected_entry <= '0;
            expected_descriptor_data <= '0;
            descriptor_format_error <= 1'b0;
            block_load_format_error <= 1'b0;
            state_load_beat <= '0;
            suppressed_load_beat <= '0;
            output_state_entry <= '0;
            active_block_slot <= '0;
            active_block_rows <= '0;
            active_block_row <= '0;
            active_current_entry <= '0;
            state_entry_low <= '0;
            state_entry_high <= '0;
            next_state_entry <= '0;
            next_state_pending <= 1'b0;
            next_state_half <= 1'b0;
            event_slot <= '0;
            event_word <= '0;
            event_read_pending <= 1'b0;
            event_data_valid <= 1'b0;
            event_data <= '0;
            candidate_done_seen <= 1'b0;
            candidate_error_seen <= 1'b0;
            candidate_error_saved <= '0;
            candidate_result_count <= '0;
            candidate_result_format_error <= 1'b0;
            lm_started <= 1'b0;
            lm_finished <= 1'b0;
            candidate_started <= 1'b0;
            candidate_finished <= 1'b0;
            feature_running <= 1'b0;
            writer_running <= 1'b0;
            table_enabled <= 1'b0;
            candidate_result_read_pending <= 1'b0;
            candidate_result_data_valid <= 1'b0;
            candidate_result_entry <= '0;
            candidate_result_payload <= '0;
            block_valid <= '0;
            saved_emit_metadata <= 1'b0;
            packer_round <= '0;
            packer_start_pending <= 1'b0;
            packer_all_done <= 1'b0;
            metadata_store_count <= '0;
            metadata_read_count <= '0;
            metadata_read_pending <= 1'b0;
            metadata_data_valid <= 1'b0;
            metadata_data <= '0;
            for (integer slot = 0; slot < 4; slot++) begin
                block_masked[slot] <= '0;
                block_tentative[slot] <= '0;
            end
            state_descriptor_valid <= '0;
        end else begin
            if (packer_start_valid && packer_start_ready)
                packer_start_pending <= 1'b0;
            if (packer_beat_valid && packer_beat_ready)
                metadata_store_count <= metadata_store_count + 8'd1;
            if (metadata_read_request && post_state_read_valid &&
                post_state_read_ready)
                metadata_read_pending <= 1'b1;
            if (post_state_read_rsp_valid &&
                post_state_read_rsp_tag[15:8] == 8'hf1) begin
                metadata_read_pending <= 1'b0;
                if (post_state_read_rsp_tag[7:0] != metadata_read_count) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_WRITEBACK;
                    state <= ERROR_DRAIN;
                end else begin
                    metadata_data_valid <= 1'b1;
                    metadata_data <= post_state_read_rsp_data;
                end
            end
            if (metadata_data_valid && metadata_data_ready) begin
                metadata_data_valid <= 1'b0;
                metadata_read_count <= metadata_read_count + 8'd1;
            end
            if (packer_done_valid && packer_done_ready) begin
                if (packer_error) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_WRITEBACK |
                        (packer_error_id & 8'h0f);
                    state <= ERROR_DRAIN;
                end else if (packer_round == 0 &&
                             saved_prediction_count > 8'd48) begin
                    if (metadata_store_count != 8'd53) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_WRITEBACK;
                        state <= ERROR_DRAIN;
                    end else begin
                        packer_round <= 2'd1;
                        packer_start_pending <= 1'b1;
                    end
                end else begin
                    if (metadata_store_count != next_metadata_beats) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_WRITEBACK;
                        state <= ERROR_DRAIN;
                    end else begin
                        packer_all_done <= 1'b1;
                    end
                end
            end
            if (descriptor_valid && descriptor_ready) begin
                descriptor_index = descriptor_data[23:16];
                descriptor_slot = descriptor_data[25:24];
                descriptor_current = descriptor_data[223:192];
                if (descriptor_index != descriptor_count ||
                    descriptor_slot >= saved_block_count ||
                    descriptor_data[223:199] != 0 ||
                    descriptor_data[255:231] != 0 ||
                    state_descriptor_valid[descriptor_current])
                    descriptor_format_error <= 1'b1;
                descriptor_current_entry[descriptor_index] <=
                    descriptor_current;
                state_descriptor_valid[descriptor_current] <= 1'b1;
                case (descriptor_index[6:3])
                    4'd0: selected_token_group[0][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd1: selected_token_group[1][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd2: selected_token_group[2][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd3: selected_token_group[3][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd4: selected_token_group[4][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd5: selected_token_group[5][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd6: selected_token_group[6][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd7: selected_token_group[7][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd8: selected_token_group[8][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd9: selected_token_group[9][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    4'd10: selected_token_group[10][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                    default: selected_token_group[11][descriptor_index[2:0]*17 +: 17] <= descriptor_data[64 +: 17];
                endcase
                descriptor_count <= descriptor_count + 8'd1;
                descriptor_store_high <= 1'b0;
            end
            if (state == SEQUENCE_WAIT && descriptor_valid &&
                post_state_write_valid && post_state_write_ready &&
                !descriptor_store_high)
                descriptor_store_high <= 1'b1;

            if (candidate_done_valid) begin
                candidate_done_seen <= 1'b1;
                candidate_error_seen <= candidate_error;
                candidate_error_saved <= candidate_error_id;
            end
            if (state == OUTPUT_WRITER_WAIT && post_state_read_valid &&
                post_state_read_ready && !metadata_read_request)
                event_read_pending <= 1'b1;
            if (post_state_read_rsp_valid &&
                post_state_read_rsp_tag[15:8] == 8'hf0) begin
                event_read_pending <= 1'b0;
                if (post_state_read_rsp_tag !=
                    {8'hf0, 3'd0, event_slot, event_word}) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_WRITEBACK;
                    state <= ERROR_DRAIN;
                end else begin
                    event_data_valid <= 1'b1;
                    event_data <= post_state_read_rsp_data;
                end
            end else if (state_writer_event_valid &&
                         state_writer_event_ready) begin
                event_data_valid <= 1'b0;
            end
            if (candidate_done_valid && candidate_done_ready)
                candidate_finished <= 1'b1;
            if (candidate_abort_ack)
                candidate_finished <= 1'b1;
            if (lm_done_valid && lm_done_ready)
                lm_finished <= 1'b1;
            if (lm_abort_ack)
                lm_finished <= 1'b1;
            if (feature_abort_ack)
                feature_running <= 1'b0;
            if (writer_abort_ack)
                writer_running <= 1'b0;

            if (psme_next_valid && psme_next_ready) begin
                // Both fixed-k and mixed-precision decoding close the current
                // block when all emitted current states are locked.
                if (!block_config[active_block_slot[1:0]].canonical_future) begin
                    current_state_seen <= 1'b1;
                    if (psme_next_state != 2'd2) current_unresolved <= 1'b1;
                end
                next_state_entry <= '0;
                next_state_entry[TOKEN_STATE_ENTRY_SOURCE_A_PENDING_OFFSET*8] <= psme_next_source_a_pending;
                if (post_config_bits[FORWARD_POSTPROCESS_CONFIG_FLAGS_OFFSET*8]) begin
                    next_state_entry[TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_BF16_OFFSET*8 +: 16] <= psme_next_action_confidence;
                    next_state_entry[TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_VALID_OFFSET*8] <= psme_next_action_confidence_valid;
                end
                next_state_entry[TOKEN_STATE_ENTRY_CHANGE_FLAGS_OFFSET*8 +: 8] <= {6'd0, psme_next_change_flags};
                next_state_entry[TOKEN_STATE_ENTRY_CHANGE_CONFIDENCE_BF16_OFFSET*8 +: 16] <= psme_next_change_confidence;
                next_state_entry[
                    TOKEN_STATE_ENTRY_TOKEN_POSITION_OFFSET*8 +: 16] <=
                    {5'd0, psme_next_token_position};
                next_state_entry[
                    TOKEN_STATE_ENTRY_BLOCK_LOCAL_POSITION_OFFSET*8 +: 8] <=
                    {3'd0, psme_next_index};
                next_state_entry[
                    TOKEN_STATE_ENTRY_BLOCK_SLOT_OFFSET*8 +: 8] <=
                    {6'd0, active_block_slot};
                next_state_entry[
                    TOKEN_STATE_ENTRY_GLOBAL_BLOCK_ID_OFFSET*8 +: 8] <=
                    {2'd0,
                     block_config[active_block_slot[1:0]].global_block_id};
                next_state_entry[
                    TOKEN_STATE_ENTRY_STATE_OFFSET*8 +: 8] <=
                    {6'd0, psme_next_state};
                next_state_entry[
                    TOKEN_STATE_ENTRY_ORIGIN_OFFSET*8 +: 8] <=
                    {6'd0, psme_next_origin};
                next_state_entry[
                    TOKEN_STATE_ENTRY_ACTIVATION_BITS_OFFSET*8 +: 8] <=
                    {4'd0, psme_next_bits};
                next_state_entry[
                    TOKEN_STATE_ENTRY_TOKEN_ID_OFFSET*8 +: 32] <=
                    psme_next_token_id;
                next_state_entry[
                    TOKEN_STATE_ENTRY_LAST_TOP1_OFFSET*8 +: 32] <=
                    psme_next_last_top1;
                next_state_entry[
                    TOKEN_STATE_ENTRY_PRECISION_AGE_OFFSET*8 +: 16] <=
                    psme_next_precision_age;
                next_state_entry[
                    TOKEN_STATE_ENTRY_CACHE_VALID_OFFSET*8 +: 8] <=
                    {7'd0, psme_next_cache_valid};
                next_state_entry[
                    TOKEN_STATE_ENTRY_REFRESH_REQUIRED_OFFSET*8 +: 8] <=
                    {7'd0, psme_next_refresh_required};
                next_state_entry[
                    TOKEN_STATE_ENTRY_CAPTURE_INDEX_OFFSET*8 +: 32] <=
                    block_config[active_block_slot[1:0]].capture_index;
                next_state_entry[
                    TOKEN_STATE_ENTRY_PREDICTION_FLAG_OFFSET*8 +: 8] <=
                    {7'd0, psme_next_prediction_flag};
                next_state_pending <= 1'b1;
                next_state_half <= 1'b0;
            end
            if (state_writer_state_valid && state_writer_state_ready) begin
                if (!next_state_half)
                    next_state_half <= 1'b1;
                else begin
                    next_state_pending <= 1'b0;
                    next_state_half <= 1'b0;
                    event_data_valid <= 1'b0;
                end
            end

            if (feature_running && feature_done_valid && feature_error) begin
                feature_running <= 1'b0;
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_DRAFT_VERIFY | (feature_error_id & 8'h0f);
                state <= ERROR_DRAIN;
            end else if (saved_emit_metadata && state == FEATURE_WAIT &&
                psme_next_valid && !next_token_id_valid) begin
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_STATE_ENTRY;
                state <= ERROR_DRAIN;
            end else if (state_writer_done_valid && state_writer_done_ready &&
                state != OUTPUT_WRITER_WAIT) begin
                writer_running <= 1'b0;
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_WRITEBACK;
                state <= ERROR_DRAIN;
            end else if (feature_done_valid && feature_done_ready &&
                         state != FEATURE_WAIT && state != EVENT_STORE) begin
                feature_running <= 1'b0;
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_DRAFT_VERIFY |
                    (feature_error_id & 8'h0f);
                state <= ERROR_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    current_state_seen <= 1'b0;
                    current_unresolved <= 1'b0;
                    saved_source_a_valid <= start_source_a_valid;
                    saved_source_a_mask <= start_source_a_mask;
                    saved_source_a_block_start <= start_source_a_block_start;
                    saved_source_a_capture_index <= start_source_a_capture_index;
                    saved_emit_metadata <= configuration_is_v4 && start_configuration_bits[
                        EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET*8 +: 16] == 0;
                    saved_post_config_base <= start_configuration_bits[
                        EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET*8 +: 64];
                    saved_post_config_limit <= start_configuration_bits[
                        EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET*8 +: 64];
                    saved_prediction_base <= start_configuration_bits[
                        EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET*8 +: 64];
                    saved_prediction_limit <= start_configuration_bits[
                        EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET*8 +: 64];
                    saved_final_hidden_base <= start_configuration_bits[
                        EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET*8 +: 64];
                    saved_final_hidden_limit <= start_configuration_bits[
                        EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET*8 +: 64];
                    saved_next_metadata_base <= start_configuration_bits[
                        EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET*8 +: 64];
                    saved_next_metadata_limit <= start_configuration_bits[
                        EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET*8 +: 64];
                    saved_sequence_length <= start_configuration_bits[
                        EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET*8 +: 12];
                    saved_prediction_count <= start_configuration_bits[
                        EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET*8 +: 8];
                    saved_block_count <= start_configuration_bits[
                        EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET*8 +: 3];
                    saved_suppressed_count <= start_configuration_bits[
                        EXECUTION_CONFIG_SUPPRESSED_TOKEN_COUNT_OFFSET*8 +: 8];
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    descriptor_count <= '0;
                    descriptor_store_high <= 1'b0;
                    state_descriptor_valid <= '0;
                    descriptor_format_error <= 1'b0;
                    block_load_format_error <= 1'b0;
                    total_state_entries <= '0;
                    block_valid <= '0;
                    candidate_done_seen <= 1'b0;
                    candidate_error_seen <= 1'b0;
                    candidate_result_count <= '0;
                    candidate_result_format_error <= 1'b0;
                    packer_round <= '0;
                    packer_start_pending <= 1'b0;
                    packer_all_done <= 1'b0;
                    metadata_store_count <= '0;
                    metadata_read_count <= '0;
                    metadata_read_pending <= 1'b0;
                    metadata_data_valid <= 1'b0;
                    lm_started <= 1'b0;
                    lm_finished <= 1'b0;
                    candidate_started <= 1'b0;
                    candidate_finished <= 1'b0;
                    feature_running <= 1'b0;
                    writer_running <= 1'b0;
                    table_enabled <= 1'b0;
                    candidate_result_read_pending <= 1'b0;
                    candidate_result_data_valid <= 1'b0;
                    next_state_pending <= 1'b0;
                    next_state_half <= 1'b0;
                    for (integer slot = 0; slot < 4; slot++) begin
                        block_masked[slot] <= '0;
                        block_tentative[slot] <= '0;
                    end
                    if (start_configuration_bits[
                            EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET*8 +: 8] == 0 ||
                        start_configuration_bits[
                            EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET*8 +: 8] > 4) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_BLOCK_FORMAT;
                        state <= COMPLETE;
                    end else begin
                        state <= SEQUENCE_START;
                    end
                end

                SEQUENCE_START:
                    if (sequence_start_valid && sequence_start_ready)
                        state <= SEQUENCE_WAIT;

                SEQUENCE_WAIT:
                    if (sequence_done_valid && sequence_done_ready) begin
                        if (sequence_error || descriptor_format_error ||
                            descriptor_count != saved_prediction_count) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SEQUENCE |
                                (sequence_error_id & 8'h0f);
                            state <= COMPLETE;
                        end else begin
                            state <= LM_CANDIDATE_START;
                        end
                    end

                LM_CANDIDATE_START:
                    if (lm_start_ready && candidate_start_ready) begin
                        lm_started <= 1'b1;
                        lm_finished <= 1'b0;
                        candidate_started <= 1'b1;
                        state <= LM_WAIT;
                    end

                LM_WAIT:
                    if (candidate_done_valid && candidate_done_ready) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_CANDIDATE |
                            (candidate_error_id & 8'h0f);
                        state <= ERROR_DRAIN;
                    end else if (lm_done_valid && lm_done_ready) begin
                        if (lm_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_LM_HEAD |
                                (lm_error_id & 8'h0f);
                            state <= ERROR_DRAIN;
                        end else begin
                            state <= TABLE_ENABLE;
                        end
                    end

                TABLE_ENABLE:
                    if (post_table_access_valid && post_table_access_ready) begin
                        table_enabled <= 1'b1;
                        state <= BLOCK_LOAD_START;
                    end

                BLOCK_LOAD_START:
                    if (aux_start_valid && aux_start_ready) begin
                        block_load_beat <= '0;
                        state <= BLOCK_LOAD_WAIT;
                    end

                BLOCK_LOAD_WAIT: begin
                    if (aux_data_valid && aux_data_ready) begin
                        if (!block_load_beat[0])
                            block_first_half <= aux_data;
                        else begin
                            loaded_slot = block_first_half[1:0];
                            if (block_first_half[7:2] != 0 ||
                                loaded_slot >= saved_block_count ||
                                block_valid[loaded_slot] ||
                                |block_first_half[15:14] ||
                                |block_first_half[31:30] || block_first_half[29:24] > 32 ||
                                (block_first_half[29:24] != 0 && !block_first_half[17]) || (block_first_half[23] && !block_first_half[17]) || block_first_half[20:19] == 2'd3 ||
                                (block_first_half[21] && (|block_first_half[20:16] || block_first_half[22])) ||
                                (block_first_half[22] && |block_first_half[21:17]) ||
                                (block_first_half[20:19] != 0 && (block_first_half[17] || block_first_half[18])) ||
                                (block_first_half[17] && (block_first_half[16] || block_first_half[18])) ||
                                (!block_first_half[17] && !block_first_half[21] && block_first_half[20:19] == 0 && block_first_half[47:32] == 0) ||
                                block_first_half[47:32] > 16'd32 ||
                                block_first_half[63:48] == 0 ||
                                block_first_half[111:96] == 0 ||
                                |block_first_half[127:118] ||
                                aux_data[39] || aux_data[47:40] > 8'd4 || |aux_data[63:55] ||
                                (!block_first_half[17] && |aux_data[127:96]) ||
                                (aux_data[127:96] & ~position_mask(block_first_half[117:112])) != 0)
                                block_load_format_error <= 1'b1;
                            block_config[loaded_slot] <= '{
                                canonical_future: block_first_half[17],
                                source_a_handoff: block_first_half[23],
                                transfer_only: block_first_half[21],
                                tail_bypass_all: block_first_half[18],
                                tail_bypass_stable_only: block_first_half[22],
                                closeout_kind: block_first_half[20:19],
                                observed_mask: aux_data[127:96],
                                next_entry: aux_data[54:48],
                                current_entry: aux_data[38:32],
                                capture_index: aux_data[31:0],
                                input_capture_index: aux_data[95:64],
                                position_count: block_first_half[117:112],
                                maturity_age: block_first_half[111:96],
                                tail_after_step: block_first_half[95:80],
                                step_index: block_first_half[79:64],
                                remaining_forwards: block_first_half[63:48],
                                scheduled_quota: block_first_half[47:32],
                                max_handoff_tokens: block_first_half[29:24],
                                tail_threshold_enable: block_first_half[16],
                                global_block_id: block_first_half[13:8],
                                block_slot: loaded_slot,
                                input_block_slot: aux_data[47:40] == 0 ? loaded_slot : aux_data[41:40] - 2'd1};
                            block_valid[loaded_slot] <= 1'b1;
                        end
                        block_load_beat <= block_load_beat + 8'd1;
                    end
                    if (aux_done_pulse) begin
                        if (aux_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BLOCK_DMA;
                            state <= ERROR_DRAIN;
                        end else begin
                            block_validate_index <= '0;
                            total_state_entries <= '0;
                            state <= BLOCK_VALIDATE;
                        end
                    end
                end

                BLOCK_VALIDATE: begin
                    loaded_slot = block_validate_index[1:0];
                    loaded_rows = block_position_count(loaded_slot);
                    new_total = {1'b0, total_state_entries} + loaded_rows;
                    if (block_validate_index >= saved_block_count) begin
                        if (block_valid != (4'd1 << saved_block_count)-1'b1 ||
                            total_state_entries == 0 ||
                            block_load_format_error ||
                            (saved_emit_metadata &&
                             total_state_entries != saved_prediction_count)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BLOCK_FORMAT;
                            state <= ERROR_DRAIN;
                        end else begin
                            block_validate_index <= '0;
                            state <= RANGE_VALIDATE;
                        end
                    end else if (!block_valid[loaded_slot] ||
                        loaded_rows == 0 || loaded_rows > 32 ||
                        new_total > 128) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_BLOCK_FORMAT;
                        state <= ERROR_DRAIN;
                    end else begin
                        total_state_entries <= new_total[7:0];
                        block_validate_index <= block_validate_index + 3'd1;
                    end
                end

                RANGE_VALIDATE: begin
                    logic range_error;
                    logic [6:0] current_base;
                    logic [6:0] next_base;
                    logic [6:0] row_count;
                    range_error = 1'b0;
                    current_base = block_current_entry(
                        block_validate_index[1:0]);
                    next_base = block_next_entry(
                        block_validate_index[1:0]);
                    row_count = block_position_count(
                        block_validate_index[1:0]);
                    if (block_validate_index >= saved_block_count) begin
                        descriptor_validate_index <= '0;
                        state <= DESCRIPTOR_VALIDATE;
                    end else begin
                        range_error =
                            block_config[block_validate_index[1:0]].block_slot !=
                                block_validate_index[1:0] ||
                            {1'b0, current_base} + {1'b0, row_count} >
                                {1'b0, total_state_entries} ||
                            {1'b0, next_base} + {1'b0, row_count} >
                                {1'b0, total_state_entries} ||
                            block_config[block_validate_index[1:0]].capture_index !=
                                block_config[0].capture_index;
                        for (integer other = 0; other < 4; other++) begin
                            if (other < saved_block_count &&
                                other != block_validate_index) begin
                                if ({1'b0, current_base} <
                                        {1'b0, block_current_entry(2'(other))} +
                                        {1'b0, block_position_count(2'(other))} &&
                                    block_current_entry(2'(other)) <
                                        {1'b0, current_base} +
                                        {1'b0, row_count})
                                    range_error = 1'b1;
                                if ({1'b0, next_base} <
                                        {1'b0, block_next_entry(2'(other))} +
                                        {1'b0, block_position_count(2'(other))} &&
                                    block_next_entry(2'(other)) <
                                        {1'b0, next_base} +
                                        {1'b0, row_count})
                                    range_error = 1'b1;
                            end
                        end
                        if (range_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BLOCK_FORMAT;
                            state <= ERROR_DRAIN;
                        end else begin
                            block_validate_index <=
                                block_validate_index + 3'd1;
                        end
                    end
                end

                DESCRIPTOR_VALIDATE: begin
                    if (descriptor_validate_index >= saved_prediction_count) begin
                        state <= STATE_LOAD_START;
                    end else begin
                        state <= DESCRIPTOR_READ_LOW;
                    end
                end

                DESCRIPTOR_READ_LOW:
                    if (post_state_read_valid && post_state_read_ready)
                        state <= DESCRIPTOR_WAIT_LOW;

                DESCRIPTOR_WAIT_LOW:
                    if (post_state_read_rsp_valid) begin
                        if (post_state_read_rsp_tag !=
                            {8'he0, descriptor_validate_index}) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BLOCK_FORMAT;
                            state <= ERROR_DRAIN;
                        end else begin
                            block_first_half <= post_state_read_rsp_data;
                            state <= DESCRIPTOR_READ_HIGH;
                        end
                    end

                DESCRIPTOR_READ_HIGH:
                    if (post_state_read_valid && post_state_read_ready)
                        state <= DESCRIPTOR_WAIT_HIGH;

                DESCRIPTOR_WAIT_HIGH: if (post_state_read_rsp_valid) begin
                    logic descriptor_error;
                    logic [1:0] slot;
                    logic [4:0] local_position;
                    logic [6:0] current_entry;
                    logic [6:0] next_entry;
                    slot = block_first_half[25:24];
                    local_position = block_first_half[47:40];
                    current_entry = post_state_read_rsp_data[95:64];
                    next_entry = post_state_read_rsp_data[127:96];
                    descriptor_error = post_state_read_rsp_tag !=
                            {8'he1, descriptor_validate_index} ||
                        slot >= saved_block_count ||
                        local_position >= block_position_count(slot) ||
                        block_first_half[37:32] !=
                            block_config[slot].global_block_id ||
                        current_entry !=
                            block_current_entry(slot) + local_position ||
                        next_entry != block_next_entry(slot) + local_position ||
                        (saved_emit_metadata && next_entry >= 7'd96) ||
                        (block_config[slot].canonical_future &&
                         !block_config[slot].observed_mask[local_position]) ||
                        (block_config[slot].source_a_handoff &&
                         (!saved_source_a_valid || saved_source_a_capture_index != block_config[slot].input_capture_index ||
                          {1'b0, block_first_half[10:0]} < {1'b0, saved_source_a_block_start} ||
                          {1'b0, block_first_half[10:0]} >= {1'b0, saved_source_a_block_start}+12'd32));
                    if (descriptor_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_BLOCK_FORMAT;
                        state <= ERROR_DRAIN;
                    end else begin
                        descriptor_expected_entry <= current_entry;
                        descriptor_query_group[next_entry] <=
                            post_state_read_rsp_data[55:48];
                        descriptor_cache_group[next_entry] <=
                            post_state_read_rsp_data[63:56];
                        expected_descriptor_data <= {
                            descriptor_validate_index[6:0],
                            block_first_half[10:0], slot,
                            block_first_half[37:32], local_position,
                            block_first_half[57:56],
                            block_first_half[51:48],
                            block_first_half[80:64]};
                        state <= DESCRIPTOR_EXPECTED_WRITE;
                    end
                end

                DESCRIPTOR_EXPECTED_WRITE:
                    if (post_state_write_valid && post_state_write_ready) begin
                        descriptor_validate_index <=
                            descriptor_validate_index + 8'd1;
                        state <= DESCRIPTOR_VALIDATE;
                    end

                STATE_LOAD_START:
                    if (aux_start_valid && aux_start_ready) begin
                        state_load_beat <= '0;
                        state <= STATE_LOAD_WAIT;
                    end

                STATE_LOAD_WAIT: begin
                    if (post_state_write_valid && post_state_write_ready) begin
                        state_load_beat <= state_load_beat + 8'd1;
                        if (!state_load_beat[0]) begin
                            for (integer slot = 0; slot < 4; slot++) begin
                                if (slot < saved_block_count &&
                                    state_load_beat[7:1] >= block_current_entry(2'(slot)) &&
                                    {1'b0, state_load_beat[7:1]} <
                                        {1'b0, block_current_entry(2'(slot))} +
                                        {2'd0, block_position_count(2'(slot))}) begin
                                    block_masked[slot][aux_data[20:16]] <= aux_data[47:40] == 0;
                                    block_tentative[slot][aux_data[20:16]] <= aux_data[47:40] == 1;
                                end
                            end
                        end
                    end
                    if (aux_done_pulse) begin
                        if (aux_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_STATE_DMA;
                            state <= ERROR_DRAIN;
                        end else if (saved_suppressed_count != 0) begin
                            state <= SUPPRESSED_LOAD_START;
                        end else begin
                            state <= CANDIDATE_WAIT;
                        end
                    end
                end

                SUPPRESSED_LOAD_START:
                    if (aux_start_valid && aux_start_ready) begin
                        suppressed_load_beat <= '0;
                        state <= SUPPRESSED_LOAD_WAIT;
                    end

                SUPPRESSED_LOAD_WAIT: begin
                    if (post_suppressed_write_valid &&
                        post_suppressed_write_ready)
                        suppressed_load_beat <= suppressed_load_beat + 6'd1;
                    if (aux_done_pulse) begin
                        if (aux_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SUPPRESSED_DMA;
                            state <= ERROR_DRAIN;
                        end else begin
                            state <= CANDIDATE_WAIT;
                        end
                    end
                end

                CANDIDATE_WAIT: begin
                    if (candidate_result_valid &&
                        !candidate_result_read_pending &&
                        post_state_read_valid && post_state_read_ready) begin
                        if (candidate_result_row != candidate_result_count ||
                            candidate_result_row >= saved_prediction_count)
                            candidate_result_format_error <= 1'b1;
                        candidate_result_read_pending <= 1'b1;
                        candidate_result_entry <=
                            descriptor_current_entry[candidate_result_row];
                        candidate_result_payload <= {
                            candidate_result_top1,
                            candidate_result_suppressed,
                            candidate_result_action_confidence,
                            candidate_result_selected_probability};
                    end
                    if (candidate_result_read_pending &&
                        post_state_read_rsp_valid) begin
                        state_entry_high <= post_state_read_rsp_data;
                        candidate_result_read_pending <= 1'b0;
                        candidate_result_data_valid <= 1'b1;
                    end
                    if (candidate_result_ready && candidate_result_valid) begin
                        candidate_result_data_valid <= 1'b0;
                        candidate_result_count <= candidate_result_count + 8'd1;
                    end
                    if (candidate_done_seen && !candidate_result_read_pending &&
                        !candidate_result_data_valid) begin
                        if (candidate_error_seen ||
                            candidate_result_format_error ||
                            candidate_result_count != saved_prediction_count) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_CANDIDATE |
                                (candidate_error_saved & 8'h0f);
                            state <= ERROR_DRAIN;
                        end else begin
                            state <= OUTPUT_WRITER_START;
                        end
                    end
                end

                OUTPUT_WRITER_START:
                    if (state_writer_start_valid && state_writer_start_ready) begin
                        writer_running <= 1'b1;
                        if (saved_emit_metadata)
                            packer_start_pending <= 1'b1;
                        output_state_entry <= '0;
                        event_slot <= '0;
                        event_word <= '0;
                        state <= BLOCK_SELECT;
                    end

                BLOCK_SELECT: begin
                    found_block = 1'b0;
                    found_slot = '0;
                    for (integer slot = 0; slot < 4; slot++) begin
                        if (slot < saved_block_count &&
                            block_next_entry(2'(slot)) == output_state_entry) begin
                            found_block = 1'b1;
                            found_slot = 2'(slot);
                        end
                    end
                    if (output_state_entry >= total_state_entries) begin
                        state <= OUTPUT_WRITER_WAIT;
                    end else if (!found_block) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_BLOCK_FORMAT;
                        state <= ERROR_DRAIN;
                    end else begin
                        active_block_slot <= found_slot;
                        active_block_rows <= block_position_count(found_slot);
                        active_block_row <= '0;
                        active_current_entry <= block_current_entry(found_slot);
                        state <= FEATURE_START;
                    end
                end

                FEATURE_START:
                    if (feature_start_valid && feature_start_ready) begin
                        feature_running <= 1'b1;
                        state <= FEATURE_EXPECTED_READ;
                    end

                FEATURE_EXPECTED_READ:
                    if (post_state_read_valid && post_state_read_ready)
                        state <= FEATURE_EXPECTED_WAIT;

                FEATURE_EXPECTED_WAIT:
                    if (post_state_read_rsp_valid) begin
                        if (post_state_read_rsp_tag !=
                            {8'hd0, active_block_row, 2'b01}) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_STATE_ENTRY;
                            state <= ERROR_DRAIN;
                        end else begin
                            expected_descriptor_data <=
                                post_state_read_rsp_data[53:0];
                            state <= FEATURE_READ_LOW;
                        end
                    end

                FEATURE_READ_LOW:
                    if (post_state_read_valid && post_state_read_ready)
                        state <= FEATURE_WAIT_LOW;

                FEATURE_WAIT_LOW:
                    if (post_state_read_rsp_valid) begin
                        state_entry_low <= post_state_read_rsp_data;
                        state <= FEATURE_READ_HIGH;
                    end

                FEATURE_READ_HIGH:
                    if (post_state_read_valid && post_state_read_ready)
                        state <= FEATURE_WAIT_HIGH;

                FEATURE_WAIT_HIGH:
                    if (post_state_read_rsp_valid) begin
                        state_entry_high <= post_state_read_rsp_data;
                        state_entry_error =
                            state_entry_low[23:16] !=
                                {3'd0, active_block_row[4:0]} ||
                            state_entry_low[31:24] !=
                                {6'd0, block_config[active_block_slot[1:0]].input_block_slot} ||
                            state_entry_low[39:32] != {2'd0,
                                block_config[active_block_slot[1:0]].global_block_id} ||
                            post_state_read_rsp_data[63:32] !=
                                block_config[active_block_slot[1:0]].input_capture_index ||
                            state_entry_low[41:40] > 2 ||
                            |state_entry_low[47:42] ||
                            |state_entry_low[55:50] ||
                            |state_entry_low[63:60] ||
                            |state_entry_low[95:81] ||
                            state_entry_low[80:64] >= post_config_bits[
                                FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 17] ||
                            (state_entry_low[59:56] != 4 &&
                             state_entry_low[59:56] != 8) ||
                            |post_state_read_rsp_data[23:17] ||
                            |post_state_read_rsp_data[31:25] ||
                            |post_state_read_rsp_data[71:65] ||
                            (!state_descriptor_valid[active_current_entry] &&
                             feature_row_prediction_flag) ||
                            (state_descriptor_valid[active_current_entry] &&
                            (state_entry_low[10:0] !=
                                expected_descriptor_data[46:36] ||
                             expected_descriptor_data[35:34] != active_block_slot[1:0] ||
                             state_entry_low[39:32] !=
                                {2'd0, expected_descriptor_data[33:28]} ||
                             state_entry_low[23:16] !=
                                {3'd0, expected_descriptor_data[27:23]} ||
                             state_entry_low[41:40] !=
                                expected_descriptor_data[22:21] ||
                             state_entry_low[64 +: 17] !=
                                expected_descriptor_data[16:0]));
                        if (state_entry_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_STATE_ENTRY;
                            state <= ERROR_DRAIN;
                        end else begin
                            state <= FEATURE_SEND_ROW;
                        end
                    end

                FEATURE_SEND_ROW:
                    if (feature_row_valid && feature_row_ready) begin
                        if (active_block_row + 6'd1 == active_block_rows) begin
                            state <= FEATURE_WAIT;
                        end else begin
                            active_block_row <= active_block_row + 6'd1;
                            active_current_entry <= active_current_entry + 7'd1;
                            state <= FEATURE_EXPECTED_READ;
                        end
                    end

                FEATURE_WAIT:
                    if (feature_done_valid) begin
                        if (feature_error || terminal_error) begin
                            feature_running <= 1'b0;
                            terminal_error <= 1'b1;
                            terminal_error_id <= terminal_error ?
                                terminal_error_id : ERROR_DRAFT_VERIFY |
                                (feature_error_id & 8'h0f);
                            state <= ERROR_DRAIN;
                        end else begin
                            event_word <= '0;
                            state <= EVENT_STORE;
                        end
                    end

                EVENT_STORE:
                    if (post_state_write_valid && post_state_write_ready) begin
                        if (event_word == 2'd3) begin
                            feature_running <= 1'b0;
                            event_word <= '0;
                            output_state_entry <= output_state_entry +
                                active_block_rows;
                            state <= BLOCK_SELECT;
                        end else begin
                            event_word <= event_word + 2'd1;
                        end
                    end

                OUTPUT_WRITER_WAIT: begin
                    if (state_writer_event_valid && state_writer_event_ready) begin
                        if (event_word == 2'd3) begin
                            event_word <= '0;
                            event_slot <= event_slot + 3'd1;
                        end else begin
                            event_word <= event_word + 2'd1;
                        end
                    end
                    if (state_writer_done_valid && state_writer_done_ready) begin
                        writer_running <= 1'b0;
                        if (state_writer_error || !completion_published) begin
                            terminal_error <= 1'b1;
                            if (!terminal_error)
                                terminal_error_id <= ERROR_WRITEBACK;
                        end
                        state <= TABLE_DISABLE;
                    end
                end

                ERROR_DRAIN:
                    if ((!lm_started || lm_finished) &&
                        (!candidate_started || candidate_finished) &&
                        !feature_running && !writer_running) begin
                        state <= table_enabled ? TABLE_DISABLE : COMPLETE;
                    end

                TABLE_DISABLE:
                    if (post_table_access_valid && post_table_access_ready) begin
                        table_enabled <= 1'b0;
                        state <= COMPLETE;
                    end

                COMPLETE:
                    if (done_valid && done_ready)
                        state <= IDLE;

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (LM_PANEL_COUNT < 1 || LM_PANEL_COUNT > 15808)
            $error("forward_postprocess_controller LM panel count is invalid");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(lm_start_valid && !candidate_start_ready))
                else $error("forward-postprocess LM head started without Candidate readiness");
            assert (!(candidate_start_valid && !lm_start_ready))
                else $error("forward-postprocess Candidate started without LM-head readiness");
            if (candidate_result_valid)
                assert (candidate_result_row < saved_prediction_count)
                    else $error("Candidate result row exceeds prediction count");
            if (post_state_read_rsp_valid)
                assert (post_state_read_rsp_tag[15:8] == 8'hc1 ||
                        post_state_read_rsp_tag[15:8] == 8'hb2 ||
                        post_state_read_rsp_tag[15:8] == 8'hd0 ||
                        post_state_read_rsp_tag[15:8] == 8'he0 ||
                        post_state_read_rsp_tag[15:8] == 8'he1 ||
                        post_state_read_rsp_tag[15:8] == 8'hf0 ||
                        post_state_read_rsp_tag[15:8] == 8'hf1)
                    else $error("forward-postprocess state table returned an unknown tag");
        end
    end
`endif
endmodule

`default_nettype wire
