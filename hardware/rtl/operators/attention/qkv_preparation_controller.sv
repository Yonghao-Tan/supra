`default_nettype none

// Sequences V, K, and Q preparation through Matmul, RoPE, the shared quantizer,
// local SRAM and KV cache interfaces, and controls the cache commit DMA adapter.
// The first V projection creates resident W4A4/W4A8
// activations from the row configuration. A split Attention phase may refresh
// them on Q head 0; later projections in the same phase reuse those values.
module qkv_preparation_controller #(
    parameter integer MAX_ROWS = 48,
    parameter integer MAX_HEADS = 32,
    parameter integer HEAD_DIM = 128,
    parameter integer TAG_WIDTH = 16,
    parameter bit INTEGRATED_CACHE_COMMIT = 1'b0,
    parameter bit TRUSTED_START_CONFIGURATION = 1'b0
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    output logic                         abort_ack,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [5:0]                   start_row_count,
    input  logic [5:0]                   start_head_count,
    input  logic [8:0]                   start_head_dim,
    input  logic [47:0]                  start_row_enable,
    input  logic [47:0]                  start_kv_write_disable,
    input  logic [MAX_ROWS*11-1:0]       start_token_position,
    input  logic                         start_kv_pair_enable,
    input  logic [5:0]                   start_second_row_count,
    input  logic [47:0]                  start_second_row_enable,
    input  logic [47:0]                  start_second_kv_write_disable,
    input  logic [MAX_ROWS*11-1:0]       start_second_token_position,
    input  hardware_types_pkg::attention_batch_group_config_t start_batch_group,
    input  logic                         kv_pair_layout_ready,
    output logic                         kv_pair_active,
    output logic                         kv_second_batch,
    output logic [2:0]                   kv_batch_index,
    output logic [5:0]                   active_token_count,
    output logic [47:0]                  active_token_enable,
    input  logic [15:0]                  start_v_static_scale_bf16,
    input  logic                         start_v_scale_per_head,
    input  logic [63:0]                  start_rope_cos_base,
    input  logic [63:0]                  start_rope_sin_base,
    input  logic [63:0]                  start_v_scale_base,
    input  logic                         q_head_start_valid,
    output logic                         q_head_start_ready,
    input  logic [5:0]                   q_head_start_head,
    input  logic                         q_head_start_slot,
    input  logic                         q_head_postprocess_enable,
    input  logic                         q_head_preprocess_input,
    input  logic                         q_head_start_precompute_only,
    input  logic                         q_head_start_use_staged,
    output logic                         q_head_matmul_done,
    output logic                         q_head_done_valid,
    input  logic                         q_head_done_ready,
    output logic                         q_head_error,
    output logic [7:0]                   q_head_error_id,
    output logic                         done_valid,
    input  logic                         done_ready,
    output logic                         error,
    output logic [7:0]                   error_id,
    input  logic [5:0]                   v_scale_select_head,
    output logic [15:0]                  loaded_v_static_scale_bf16,
    output logic [1:0]                   compute_stage,
    output logic [1:0]                   memory_stage,

    output logic                         matmul_start_valid,
    input  logic                         matmul_start_ready,
    output logic [1:0]                   matmul_qkv_select,
    output logic [5:0]                   matmul_head,
    output logic                         matmul_preprocess_input,
    output logic                         matmul_use_resident_a8,
    output logic                         kv_group_activation_reuse,
    output logic                         matmul_qkv_w8_layout,
    output logic [63:0]                  matmul_group_staging_base,
    output logic                         matmul_prefetch_valid,
    input  logic                         matmul_prefetch_ready,
    output logic [1:0]                   matmul_prefetch_qkv_select,
    output logic [5:0]                   matmul_prefetch_head,
    input  logic                         matmul_done_valid,
    input  logic                         matmul_error,
    output logic                         matmul_abort_request,
    input  logic                         matmul_abort_ack,

    output logic                         rope_start_valid,
    input  logic                         rope_start_ready,
    output logic                         rope_source_is_k,
    output logic [5:0]                   rope_head,
    input  logic                         rope_done_pulse,
    input  logic                         rope_error,
    output logic                         rope_abort_request,
    input  logic                         rope_abort_ack,

    output logic                         head_staging_layout,
    input  logic                         head_staging_layout_ready,
    output logic                         matmul_overlap_active,

    output logic                         head_read_req_valid,
    input  logic                         head_read_req_ready,
    output logic                         head_read_rope_destination,
    output logic [5:0]                   head_read_physical_row,
    output logic [3:0]                   head_read_word,
    input  logic                         head_read_rsp_valid,
    output logic                         head_read_rsp_ready,
    input  logic [127:0]                 head_read_rsp_data,
    input  logic [7:0]                   head_read_rsp_lane_mask,
    output logic                         head_stage_write_valid,
    input  logic                         head_stage_write_ready,
    output logic                         head_stage_write_source,
    output logic [1:0]                   head_stage_write_bank,
    output logic                         head_stage_write_port,
    output logic [9:0]                   head_stage_write_word,
    output logic [127:0]                 head_stage_write_data,
    output logic [15:0]                  head_stage_write_byte_enable,

    output logic                         head_tile_memory_req_valid,
    input  logic                         head_tile_memory_req_ready,
    output logic [5:0]                   head_tile_memory_row_base,
    output logic [3:0]                   head_tile_memory_word,
    output logic [7:0]                   head_tile_memory_row_mask,
    input  logic                         head_tile_memory_rsp_valid,
    input  logic [1023:0]                head_tile_memory_rsp_data,
    input  logic [63:0]                  head_tile_memory_rsp_lane_mask,

    output logic                         max_req_valid,
    input  logic                         max_req_ready,
    output logic [1023:0]                max_req_values,
    output logic [63:0]                  max_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         max_req_tag,
    input  logic                         max_rsp_valid,
    output logic                         max_rsp_ready,
    input  logic [127:0]                 max_rsp_values,
    input  logic [7:0]                   max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0]         max_rsp_tag,

    output logic                         quant_scale_req_valid,
    input  logic                         quant_scale_req_ready,
    output logic                         quant_scale_req_use_static_scale,
    output logic [127:0]                 quant_scale_req_static_scales_bf16,
    output logic [127:0]                 quant_scale_req_row_max_abs,
    output logic [7:0]                   quant_scale_req_row_mask,
    input  logic                         quant_scale_rsp_valid,
    output logic                         quant_scale_rsp_ready,
    input  logic [127:0]                 quant_scale_rsp_values_bf16,

    output logic                         quant_req_valid,
    input  logic                         quant_req_ready,
    output logic [1023:0]                quant_req_values_bf16,
    output logic [63:0]                  quant_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         quant_req_tag,
    input  logic                         quant_rsp_valid,
    output logic                         quant_rsp_ready,
    input  logic [511:0]                 quant_rsp_values,
    input  logic [63:0]                  quant_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         quant_rsp_tag,
    output logic                         quant_abort_request,
    input  logic                         quant_abort_ack,

    output logic                         quantized_commit_valid,
    input  logic                         quantized_commit_ready,
    output logic [1:0]                   quantized_commit_qkv_select,
    output logic [5:0]                   quantized_commit_head,
    output logic [5:0]                   quantized_commit_physical_row,
    output logic [10:0]                  quantized_commit_token_position,
    output logic                         quantized_commit_half,
    output logic [511:0]                 quantized_commit_values,
    output logic [63:0]                  quantized_commit_lane_mask,
    output logic [15:0]                  quantized_commit_scale_bf16,
    output logic [TAG_WIDTH-1:0]         quantized_commit_tag,
    output logic                         quantized_commit_requires_response,
    input  logic                         cache_write_complete_valid,
    output logic                         cache_write_complete_ready,
    input  logic                         cache_write_complete_error,
    output logic                         cache_abort_request,
    input  logic                         cache_abort_ack,

    output logic                         cache_mapping_commit_valid,
    input  logic                         cache_mapping_commit_ready,
    input  logic                         cache_mapping_commit_error,
    output logic                         cache_config_start_valid,
    input  logic                         cache_config_start_ready,
    input  logic                         cache_config_done_pulse,
    input  logic                         cache_config_error,
    output logic                         cache_config_release,

    output logic                         metadata_read_request_valid,
    input  logic                         metadata_read_request_ready,
    output logic [63:0]                  metadata_read_request_address,
    output logic [31:0]                  metadata_read_request_bytes,
    input  logic                         metadata_read_request_done,
    input  logic                         metadata_read_request_error,
    input  logic                         metadata_read_data_valid,
    output logic                         metadata_read_data_ready,
    input  logic [127:0]                 metadata_read_data,
    input  logic [15:0]                  metadata_read_byte_enable,
    output logic                         constant_stage_write_valid,
    input  logic                         constant_stage_write_ready,
    output logic [9:0]                   constant_stage_write_word,
    output logic [127:0]                 constant_stage_write_data,
    output logic [15:0]                  constant_stage_write_byte_enable,

    output logic                         q_local_write_valid,
    input  logic                         q_local_write_ready,
    output logic                         q_local_write_scale,
    output logic                         q_local_write_slot,
    output logic [5:0]                   q_local_write_physical_row,
    output logic [9:0]                   q_local_write_word,
    output logic [127:0]                 q_local_write_data,
    output logic [15:0]                  q_local_write_byte_enable,

    output logic                         current_write_valid,
    input  logic                         current_write_ready,
    output logic [5:0]                   current_write_head,
    output logic [5:0]                   current_write_physical_row,
    output logic [10:0]                  current_write_logical_slot,
    output logic [4:0]                   current_write_chunk,
    input  logic                         current_write_accepted,
    input  logic [63:0]                  current_k_write_address,
    input  logic [63:0]                  current_v_write_address,
    input  logic [63:0]                  current_k_scale_write_address,

    output logic                         cache_writer_request_valid,
    input  logic                         cache_writer_request_ready,
    output logic [63:0]                  cache_writer_request_address,
    output logic [31:0]                  cache_writer_request_bytes,
    output logic [7:0]                   cache_writer_request_tag,
    input  logic                         cache_writer_request_done,
    input  logic                         cache_writer_request_error,
    output logic                         cache_writer_write_valid,
    input  logic                         cache_writer_write_ready,
    output logic [127:0]                 cache_writer_write_data,
    output logic [15:0]                  cache_writer_write_byte_enable,
    output logic                         cache_writer_write_last,

    output logic [31:0]                  accepted_matmul_count,
    output logic [31:0]                  accepted_rope_count,
    output logic [31:0]                  accepted_head_read_count,
    output logic [31:0]                  accepted_quant_count,
    output logic [31:0]                  accepted_quantized_commit_count,
    output logic [31:0]                  completed_cache_write_count,
    output logic [15:0]                  outstanding_cache_writes
);
    localparam logic [1:0] QKV_Q = 2'd0;
    localparam logic [1:0] QKV_K = 2'd1;
    localparam logic [1:0] QKV_V = 2'd2;
    localparam logic [1:0] COMPUTE_STAGE_NONE = 2'd0;
    localparam logic [1:0] COMPUTE_STAGE_MATMUL = 2'd1;
    localparam logic [1:0] COMPUTE_STAGE_ROPE = 2'd2;
    localparam logic [1:0] COMPUTE_STAGE_QUANT = 2'd3;
    localparam logic [1:0] MEMORY_STAGE_NONE = 2'd0;
    localparam logic [1:0] MEMORY_STAGE_METADATA = 2'd1;
    localparam logic [1:0] MEMORY_STAGE_MATMUL = 2'd2;
    localparam logic [7:0] ERROR_CONFIG = 8'h01;
    localparam logic [7:0] ERROR_MATMUL = 8'h02;
    localparam logic [7:0] ERROR_ROPE = 8'h03;
    localparam logic [7:0] ERROR_HEAD_READ = 8'h04;
    localparam logic [7:0] ERROR_QUANT = 8'h05;
    localparam logic [7:0] ERROR_CACHE_WRITE = 8'h06;
    localparam logic [7:0] ERROR_CACHE_MAPPING = 8'h07;
    localparam logic [7:0] ERROR_METADATA_READ = 8'h08;

    typedef enum logic [5:0] {
        IDLE, METADATA_REQUEST, METADATA_STREAM, CACHE_CONFIG_START,
        CACHE_CONFIG_WAIT, MATMUL_START, MATMUL_WAIT, WAIT_Q_POST,
        ROPE_START, ROPE_WAIT, LAYOUT_TO_HEAD, STAGE_FIND_ROW,
        STAGE_READ_REQ, STAGE_READ_WAIT, STAGE_WRITE, BATCH_START,
        MAX_STREAM, SCALE_REQUEST, SCALE_RESPONSE, QUANTIZED_STREAM,
        QUANTIZED_COMMIT, ADVANCE_BATCH, LAYOUT_TO_PANEL,
        CACHE_MAPPING_START, CACHE_MAPPING_WAIT, COMPLETE,
        MATMUL_OVERLAP_WAIT,
        ERROR_DRAIN, ABORT_DRAIN, KV_BATCH_SWITCH, KV_BATCH_LAYOUT_WAIT,
        STAGING_READ_REQUEST, STAGING_READ_STREAM
    } state_t;
    state_t state;
    state_t kv_resume_state;

    logic [5:0] row_count, head_count;
    logic [47:0] row_enable;
    logic [47:0] kv_write_disable;
    logic [MAX_ROWS*11-1:0] token_position;
    // One extra 630-bit metadata set; numerical head data remains in SRAM.
    logic [5:0] other_row_count;
    logic [47:0] other_row_enable, other_kv_write_disable;
    logic [MAX_ROWS*11-1:0] other_token_position;
    hardware_types_pkg::attention_batch_group_config_t batch_group;
    hardware_types_pkg::attention_batch_group_config_t runtime_batch_group;
    logic batch_group_active;
    logic [2:0] batch_group_count;
    logic [63:0] group_staging_base;
    logic [31:0] group_staging_head_stride;
    logic group_metadata_refill;
    logic [63:0] staging_batch_base, staging_stripe_address;
    logic [4:0] staging_stripe;
    logic [5:0] staging_response_row;
    logic batch_group_configuration_valid;
    logic [5:0] selected_group_row_count;
    logic [47:0] selected_group_row_enable;
    logic [47:0] selected_group_kv_write_disable;
    logic [527:0] selected_group_token_position;
    logic [11:0] selected_group_row_base;
    logic [11:0] batch_group_total_rows;
    logic [63:0] q_group_head_base;
    logic [5:0] q_group_last_head;
    logic [2:0] q_group_active_batch;
    logic q_group_head_sequence_valid;
    logic [64:0] q_head_selected_head_base;
    logic [64:0] q_head_staging_base_sum;
    logic [64:0] q_head_staging_end;
    logic q_head_group_sequence_match;
    assign active_token_count = row_count;
    assign active_token_enable = row_enable;
    logic [15:0] v_static_scale;
    logic v_scale_per_head;
    logic [127:0] v_scale_table_word [0:3];
    logic v_scale_table_valid;
    logic v_scale_metadata_error;
    logic [5:0] v_scale_lookup_head;
    logic [15:0] selected_v_scale;
    logic [63:0] rope_cos_base, rope_sin_base, v_scale_base;
    logic [5:0] metadata_row;
    logic metadata_is_sin;
    logic [3:0] metadata_word;
    logic metadata_read_outstanding;
    logic [1:0] qkv_select;
    logic [5:0] head_index, physical_row, row_scan_index;
    logic [3:0] word_index;
    logic half_index;
    logic [5:0] batch_row_base;
    logic [7:0] batch_row_mask;
    logic [127:0] row_max_abs;
    logic [127:0] row_scale;
    logic [4095:0] quantized_half_buffer;
    logic [4:0] tile_issue_count;
    logic [4:0] tile_response_count;
    logic [4:0] pipeline_response_count;
    logic [2:0] commit_row_index;
    logic [2:0] head_read_outstanding;
    logic stage_copy_active;
    logic stage_metadata_input_ready;
    logic stage_metadata_output_valid;
    logic stage_metadata_output_ready;
    logic [9:0] stage_metadata_output_data;
    logic [2:0] stage_metadata_occupancy;
    logic [5:0] stage_write_physical_row;
    logic [3:0] stage_write_word;
    logic stage_response_error;
    logic stage_copy_drained;
    logic head_tile_read_outstanding;
    logic [5:0] max_outstanding;
    logic [5:0] quant_values_outstanding;
    logic head_tile_read_req_valid, head_tile_read_req_ready;
    logic [5:0] head_tile_read_row_base;
    logic [3:0] head_tile_read_word;
    logic [7:0] head_tile_read_row_mask;
    logic head_tile_read_rsp_valid, head_tile_read_rsp_ready;
    logic [1023:0] head_tile_read_rsp_data;
    logic [63:0] head_tile_read_rsp_lane_mask;
    logic [1:0] head_tile_memory_pending;
    logic [2:0] head_tile_fifo_occupancy;
    logic matmul_active, rope_active, quant_active;
    logic next_matmul_valid, next_matmul_started, next_matmul_done;
    logic [1:0] next_matmul_qkv_select;
    logic [5:0] next_matmul_head;
    logic cache_abort_seen;
    logic terminal_error;
    logic [7:0] terminal_error_id;
    logic cache_write_error_fire;
    logic next_matmul_error_fire;
    logic q_head_command;
    logic q_head_slot;
    logic q_head_matmul_done_seen;
    logic q_head_skip_group_matmul;
    logic q_head_precompute_only;
    logic kv_group_precompute;
    logic [64:0] kv_group_staging_end;
    wire start_kv_group_reuse = start_batch_group.enable &&
        start_batch_group.reuse_group_activation && !kv_group_staging_end[64] &&
        kv_group_staging_end <= {1'b0, start_batch_group.staging_limit};
    assign kv_group_staging_end = {1'b0, start_batch_group.staging_base} +
        {28'd0, start_batch_group.staging_head_stride, 5'd0};

    assign runtime_batch_group = TRUSTED_START_CONFIGURATION ?
        start_batch_group : batch_group;

    logic cache_writer_start_valid;
    logic cache_writer_start_ready;
    logic [63:0] cache_writer_start_address;
    logic [31:0] cache_writer_start_bytes;
    logic cache_writer_data_valid;
    logic cache_writer_data_ready;
    logic [127:0] cache_writer_data;
    logic [15:0] cache_writer_byte_enable;
    logic cache_writer_data_last;
    logic cache_writer_done;
    logic cache_writer_error;
    logic cache_writer_abort_request;
    logic cache_writer_abort_ack;
    logic cache_writer_start_fire;

    typedef enum logic [3:0] {
        COMMIT_IDLE, COMMIT_Q_QUANTIZED, COMMIT_Q_SCALE, COMMIT_ADDRESS,
        COMMIT_DMA_START, COMMIT_DMA_STREAM, COMMIT_DMA_WAIT,
        COMMIT_ADVANCE, COMMIT_COMPLETE, COMMIT_V_PAIR_WAIT,
        COMMIT_V_PAIR_ADDRESS
    } commit_state_t;
    commit_state_t commit_state;
    logic [1:0] commit_qkv_select;
    logic [5:0] commit_head, commit_row;
    logic [10:0] commit_token_position;
    logic commit_half;
    logic [2:0] commit_word;
    logic [3:0] commit_v_chunk;
    logic [63:0] commit_k_address, commit_v_address, commit_k_scale_address;
    logic [5:0] commit_pair_row;
    logic [10:0] commit_pair_token_position;
    logic commit_write_scale, commit_error, commit_v_pair;
    logic [1:0] commit_completion_count;
    logic [3:0] commit_dma_outstanding;
    logic integrated_commit_complete_valid;
    logic [511:0] cache_commit_values;
    logic [511:0] cache_commit_pair_values;
    logic [15:0] cache_commit_scale;
    logic [10:0] next_commit_token_position;
    logic v_pair_available;

    logic matmul_start_fire, rope_start_fire, head_read_req_fire;
    logic matmul_prefetch_sent;
    logic next_matmul_exists;
    logic prefetch_next_matmul_exists;
    logic [1:0] prefetch_basis_qkv_select;
    logic [5:0] prefetch_basis_head;
    logic final_head_quant_commit;
    logic final_head_quant_advance;
    logic head_read_rsp_fire, head_stage_write_fire;
    logic head_tile_read_req_fire, head_tile_read_rsp_fire;
    logic max_req_fire, max_rsp_fire;
    logic quant_scale_req_fire, quant_scale_rsp_fire;
    logic quant_req_fire, quant_rsp_fire;
    logic quantized_commit_fire, cache_complete_fire;

    function automatic [15:0] max_abs2(input [15:0] lhs, input [15:0] rhs);
        logic [14:0] lhs_magnitude, rhs_magnitude;
        begin
            lhs_magnitude = lhs[14:0];
            rhs_magnitude = rhs[14:0];
            max_abs2 = lhs_magnitude >= rhs_magnitude ?
                {1'b0, lhs_magnitude} : {1'b0, rhs_magnitude};
        end
    endfunction

    function automatic logic valid_positive_scale_beat(
        input logic [127:0] values,
        input logic [15:0] byte_enable
    );
        logic beat_valid;
        logic [15:0] value;
        begin
            beat_valid = byte_enable == 16'hffff;
            for (integer lane = 0; lane < 8; lane++) begin
                value = values[lane*16 +: 16];
                if (value[15] || value[14:7] == 8'hff || value[14:0] == 0)
                    beat_valid = 1'b0;
            end
            valid_positive_scale_beat = beat_valid;
        end
    endfunction

    function automatic logic valid_positive_scale(
        input logic [15:0] value
    );
        valid_positive_scale = !value[15] && value[14:7] != 8'hff &&
            value[14:0] != 15'd0;
    endfunction

    function automatic [15:0] max_abs8(input [127:0] values);
        logic [15:0] level0 [0:3];
        logic [15:0] level1 [0:1];
        begin
            level0[0] = max_abs2(values[15:0], values[31:16]);
            level0[1] = max_abs2(values[47:32], values[63:48]);
            level0[2] = max_abs2(values[79:64], values[95:80]);
            level0[3] = max_abs2(values[111:96], values[127:112]);
            level1[0] = max_abs2(level0[0], level0[1]);
            level1[1] = max_abs2(level0[2], level0[3]);
            max_abs8 = max_abs2(level1[0], level1[1]);
        end
    endfunction

    function automatic [TAG_WIDTH-1:0] make_tag(
        input [1:0] tag_qkv_select,
        input [5:0] tag_head,
        input [5:0] tag_row,
        input tag_half
    );
        logic [15:0] packed_tag;
        begin
            packed_tag = {tag_qkv_select, tag_head[4:0], tag_row[5:0],
                          tag_half, 2'b00};
            make_tag = TAG_WIDTH'(packed_tag);
        end
    endfunction

    function automatic [TAG_WIDTH-1:0] make_tile_tag(
        input [1:0] tag_qkv_select,
        input [5:0] tag_head,
        input [5:0] tag_row_base,
        input tag_half,
        input [3:0] tag_word
    );
        logic [15:0] packed_tag;
        begin
            packed_tag = {tag_qkv_select, tag_head[4:0], tag_row_base[5:3],
                          tag_half, tag_word, 1'b0};
            make_tile_tag = TAG_WIDTH'(packed_tag);
        end
    endfunction

    function automatic [7:0] enabled_rows_for_batch(input [5:0] row_base_value);
        integer row_offset;
        begin
            enabled_rows_for_batch = 8'd0;
            for (row_offset = 0; row_offset < 8; row_offset = row_offset + 1)
                if (integer'(row_base_value) + row_offset < MAX_ROWS)
                    enabled_rows_for_batch[row_offset] =
                        row_enable[integer'(row_base_value) + row_offset];
        end
    endfunction

    always_comb begin
        selected_group_row_count = start_batch_group.row_count[0 +: 6];
        selected_group_row_enable = start_batch_group.row_enable[0 +: 48];
        selected_group_kv_write_disable =
            start_batch_group.kv_write_disable[0 +: 48];
        selected_group_token_position =
            start_batch_group.token_position[0 +: 528];
        selected_group_row_base = 12'd0;
        batch_group_total_rows = 12'd0;
        for (integer batch = 0; batch < 6; batch++) begin
            if (batch < start_batch_group.active_batch_index)
                selected_group_row_base +=
                    {6'd0, start_batch_group.row_count[batch*6 +: 6]};
            if (batch < start_batch_group.batch_count)
                batch_group_total_rows +=
                    {6'd0, start_batch_group.row_count[batch*6 +: 6]};
        end
        case (start_batch_group.active_batch_index)
            3'd1: begin
                selected_group_row_count =
                    start_batch_group.row_count[6 +: 6];
                selected_group_row_enable =
                    start_batch_group.row_enable[48 +: 48];
                selected_group_kv_write_disable =
                    start_batch_group.kv_write_disable[48 +: 48];
                selected_group_token_position =
                    start_batch_group.token_position[528 +: 528];
            end
            3'd2: begin
                selected_group_row_count =
                    start_batch_group.row_count[12 +: 6];
                selected_group_row_enable =
                    start_batch_group.row_enable[96 +: 48];
                selected_group_kv_write_disable =
                    start_batch_group.kv_write_disable[96 +: 48];
                selected_group_token_position =
                    start_batch_group.token_position[1056 +: 528];
            end
            3'd3: begin
                selected_group_row_count =
                    start_batch_group.row_count[18 +: 6];
                selected_group_row_enable =
                    start_batch_group.row_enable[144 +: 48];
                selected_group_kv_write_disable =
                    start_batch_group.kv_write_disable[144 +: 48];
                selected_group_token_position =
                    start_batch_group.token_position[1584 +: 528];
            end
            3'd4: begin
                selected_group_row_count =
                    start_batch_group.row_count[24 +: 6];
                selected_group_row_enable =
                    start_batch_group.row_enable[192 +: 48];
                selected_group_kv_write_disable =
                    start_batch_group.kv_write_disable[192 +: 48];
                selected_group_token_position =
                    start_batch_group.token_position[2112 +: 528];
            end
            3'd5: begin
                selected_group_row_count =
                    start_batch_group.row_count[30 +: 6];
                selected_group_row_enable =
                    start_batch_group.row_enable[240 +: 48];
                selected_group_kv_write_disable =
                    start_batch_group.kv_write_disable[240 +: 48];
                selected_group_token_position =
                    start_batch_group.token_position[2640 +: 528];
            end
            default: begin end
        endcase
        q_head_group_sequence_match = q_head_start_head == 0 ||
            (q_group_head_sequence_valid &&
             group_staging_base == start_batch_group.staging_base &&
             group_staging_head_stride ==
                start_batch_group.staging_head_stride &&
             ((q_head_start_head == q_group_last_head &&
               start_batch_group.active_batch_index > q_group_active_batch) ||
              (q_head_start_head == q_group_last_head + 6'd1 &&
               start_batch_group.active_batch_index <= q_group_active_batch)));
        q_head_selected_head_base = q_head_start_head == 0 ?
            {1'b0, start_batch_group.staging_base} :
            q_head_start_head == q_group_last_head ?
                {1'b0, q_group_head_base} :
                {1'b0, q_group_head_base} +
                    {33'd0, start_batch_group.staging_head_stride};
        q_head_staging_base_sum =
            q_head_selected_head_base +
            (65'(selected_group_row_base) << 8);
        q_head_staging_end = q_head_selected_head_base +
            {45'd0, batch_group_total_rows, 8'd0};
        batch_group_configuration_valid = !start_batch_group.enable ||
            (start_batch_group.batch_count >= 3'd2 &&
             start_batch_group.batch_count <= 3'd6 &&
             start_batch_group.active_batch_index <
                start_batch_group.batch_count &&
             start_batch_group.staging_base[7:0] == 0 &&
             start_batch_group.staging_limit > start_batch_group.staging_base &&
             start_batch_group.staging_head_stride >=
                {12'd0, batch_group_total_rows, 8'd0} &&
             selected_group_row_count == start_row_count &&
             selected_group_row_enable == start_row_enable &&
             selected_group_kv_write_disable == start_kv_write_disable &&
             selected_group_token_position == start_token_position);
        for (integer batch = 0; batch < 6; batch++) begin
            if (start_batch_group.enable && batch < start_batch_group.batch_count &&
                (start_batch_group.row_count[batch*6 +: 6] == 0 ||
                 start_batch_group.row_count[batch*6 +: 6] > 6'(MAX_ROWS) ||
                 $countones(start_batch_group.row_enable[batch*48 +: 48]) !=
                    start_batch_group.row_count[batch*6 +: 6]))
                batch_group_configuration_valid = 1'b0;
        end
    end

    wire start_configuration_valid = start_row_count != 0 &&
        start_row_count <= 6'(MAX_ROWS) &&
        start_head_count != 0 && start_head_count <= 6'(MAX_HEADS) &&
        start_head_dim == 9'(HEAD_DIM) &&
        $countones(start_row_enable) == start_row_count &&
        (!start_kv_pair_enable ||
         (start_second_row_count != 0 && start_second_row_count <= 6'(MAX_ROWS) &&
          $countones(start_second_row_enable) == start_second_row_count)) &&
        batch_group_configuration_valid &&
        (INTEGRATED_CACHE_COMMIT ||
         start_v_static_scale_bf16[14:0] != 15'd0);
    wire q_head_configuration_valid = start_row_count != 0 &&
        start_row_count <= 6'(MAX_ROWS) &&
        start_head_count != 0 && start_head_count <= 6'(MAX_HEADS) &&
        start_head_dim == 9'(HEAD_DIM) &&
        $countones(start_row_enable) == start_row_count &&
        q_head_start_head < start_head_count &&
        batch_group_configuration_valid &&
        (!start_batch_group.enable ||
         (q_head_group_sequence_match && !q_head_staging_base_sum[64] &&
          !q_head_staging_end[64] &&
          q_head_staging_end <= {1'b0, start_batch_group.staging_limit}));

    assign start_ready = state == IDLE && commit_state == COMMIT_IDLE &&
        !abort_request && !q_head_start_valid;
    assign q_head_start_ready = state == IDLE && commit_state == COMMIT_IDLE &&
        !abort_request && !start_valid;
    assign done_valid = state == COMPLETE && !q_head_command;
    assign q_head_done_valid = state == COMPLETE && q_head_command;
    assign q_head_matmul_done = q_head_command && q_head_matmul_done_seen;
    assign q_head_error = q_head_done_valid && terminal_error;
    assign q_head_error_id = terminal_error_id;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign next_matmul_error_fire = next_matmul_valid &&
        next_matmul_started && matmul_error;
    always_comb begin
        // Only V quantization needs the preparation head's scale. Q for the
        // next head can enter this state while Attention still consumes PV.
        v_scale_lookup_head = state == LAYOUT_TO_HEAD && qkv_select == QKV_V ?
            head_index : v_scale_select_head;
        selected_v_scale = 16'd0;
        case (v_scale_lookup_head)
            6'd0: selected_v_scale = v_scale_table_word[0][15:0];
            6'd1: selected_v_scale = v_scale_table_word[0][31:16];
            6'd2: selected_v_scale = v_scale_table_word[0][47:32];
            6'd3: selected_v_scale = v_scale_table_word[0][63:48];
            6'd4: selected_v_scale = v_scale_table_word[0][79:64];
            6'd5: selected_v_scale = v_scale_table_word[0][95:80];
            6'd6: selected_v_scale = v_scale_table_word[0][111:96];
            6'd7: selected_v_scale = v_scale_table_word[0][127:112];
            6'd8: selected_v_scale = v_scale_table_word[1][15:0];
            6'd9: selected_v_scale = v_scale_table_word[1][31:16];
            6'd10: selected_v_scale = v_scale_table_word[1][47:32];
            6'd11: selected_v_scale = v_scale_table_word[1][63:48];
            6'd12: selected_v_scale = v_scale_table_word[1][79:64];
            6'd13: selected_v_scale = v_scale_table_word[1][95:80];
            6'd14: selected_v_scale = v_scale_table_word[1][111:96];
            6'd15: selected_v_scale = v_scale_table_word[1][127:112];
            6'd16: selected_v_scale = v_scale_table_word[2][15:0];
            6'd17: selected_v_scale = v_scale_table_word[2][31:16];
            6'd18: selected_v_scale = v_scale_table_word[2][47:32];
            6'd19: selected_v_scale = v_scale_table_word[2][63:48];
            6'd20: selected_v_scale = v_scale_table_word[2][79:64];
            6'd21: selected_v_scale = v_scale_table_word[2][95:80];
            6'd22: selected_v_scale = v_scale_table_word[2][111:96];
            6'd23: selected_v_scale = v_scale_table_word[2][127:112];
            6'd24: selected_v_scale = v_scale_table_word[3][15:0];
            6'd25: selected_v_scale = v_scale_table_word[3][31:16];
            6'd26: selected_v_scale = v_scale_table_word[3][47:32];
            6'd27: selected_v_scale = v_scale_table_word[3][63:48];
            6'd28: selected_v_scale = v_scale_table_word[3][79:64];
            6'd29: selected_v_scale = v_scale_table_word[3][95:80];
            6'd30: selected_v_scale = v_scale_table_word[3][111:96];
            6'd31: selected_v_scale = v_scale_table_word[3][127:112];
            default: selected_v_scale = 16'd0;
        endcase
    end
    assign loaded_v_static_scale_bf16 = INTEGRATED_CACHE_COMMIT &&
        v_scale_per_head ?
        (v_scale_table_valid && v_scale_select_head < head_count ?
            selected_v_scale : 16'd0) : v_static_scale;

    always_comb begin
        compute_stage = COMPUTE_STAGE_NONE;
        case (state)
            MATMUL_START, MATMUL_WAIT:
                compute_stage = COMPUTE_STAGE_MATMUL;
            WAIT_Q_POST, ROPE_START, ROPE_WAIT:
                compute_stage = COMPUTE_STAGE_ROPE;
            BATCH_START, MAX_STREAM, SCALE_REQUEST, SCALE_RESPONSE,
            QUANTIZED_STREAM, QUANTIZED_COMMIT, ADVANCE_BATCH:
                compute_stage = COMPUTE_STAGE_QUANT;
            ERROR_DRAIN, ABORT_DRAIN: begin
                if (matmul_active)
                    compute_stage = COMPUTE_STAGE_MATMUL;
                else if (rope_active)
                    compute_stage = COMPUTE_STAGE_ROPE;
                else if (quant_active)
                    compute_stage = COMPUTE_STAGE_QUANT;
            end
            default: compute_stage = COMPUTE_STAGE_NONE;
        endcase
    end

    always_comb begin
        memory_stage = MEMORY_STAGE_NONE;
        if (next_matmul_valid && next_matmul_started &&
            !next_matmul_done) begin
            memory_stage = MEMORY_STAGE_MATMUL;
        end else begin
            case (state)
                METADATA_REQUEST, METADATA_STREAM,
                STAGING_READ_REQUEST, STAGING_READ_STREAM:
                    memory_stage = MEMORY_STAGE_METADATA;
                MATMUL_START, MATMUL_WAIT, MATMUL_OVERLAP_WAIT:
                    memory_stage = MEMORY_STAGE_MATMUL;
                ERROR_DRAIN, ABORT_DRAIN: begin
                    if (metadata_read_outstanding)
                        memory_stage = MEMORY_STAGE_METADATA;
                    else if (matmul_active)
                        memory_stage = MEMORY_STAGE_MATMUL;
                end
                default: memory_stage = MEMORY_STAGE_NONE;
            endcase
        end
    end

    assign matmul_start_valid =
        (state == MATMUL_START ||
         (next_matmul_valid && !next_matmul_started &&
          state != ERROR_DRAIN && state != ABORT_DRAIN &&
          head_staging_layout_ready)) && !abort_request;
    assign matmul_qkv_select = next_matmul_valid ?
        next_matmul_qkv_select : qkv_select;
    assign matmul_head = next_matmul_valid ? next_matmul_head : head_index;
    assign matmul_preprocess_input =
        (matmul_qkv_select == QKV_V && matmul_head == 0) ||
        (kv_group_activation_reuse && matmul_qkv_select == QKV_K && matmul_head == 0) ||
        (q_head_command && q_head_preprocess_input &&
         matmul_qkv_select == QKV_Q && matmul_head == 0);
    assign matmul_use_resident_a8 = !matmul_preprocess_input;
    assign matmul_qkv_w8_layout = 1'b0;
    assign matmul_group_staging_base =
        (q_head_command && batch_group_active) || kv_group_activation_reuse ? q_group_head_base :
        group_staging_base;
    assign matmul_start_fire = matmul_start_valid && matmul_start_ready;
    assign next_matmul_exists = !q_head_command && !kv_pair_active &&
        (head_index + 6'd1 < head_count || qkv_select == QKV_V);
    assign prefetch_basis_qkv_select = next_matmul_valid ?
        next_matmul_qkv_select : qkv_select;
    assign prefetch_basis_head = next_matmul_valid ?
        next_matmul_head : head_index;
    assign prefetch_next_matmul_exists = !q_head_command && !kv_pair_active &&
        (prefetch_basis_head + 6'd1 < head_count ||
         prefetch_basis_qkv_select == QKV_V);
    assign matmul_prefetch_qkv_select =
        prefetch_basis_head + 6'd1 < head_count ?
            prefetch_basis_qkv_select :
            prefetch_basis_qkv_select == QKV_V ? QKV_K : QKV_Q;
    assign matmul_prefetch_head =
        prefetch_basis_head + 6'd1 < head_count ?
            prefetch_basis_head + 6'd1 : 6'd0;
    assign final_head_quant_commit = state == QUANTIZED_COMMIT && half_index &&
        batch_row_base + 6'd8 >= 6'(MAX_ROWS);
    assign final_head_quant_advance = state == ADVANCE_BATCH && half_index &&
        batch_row_base + 6'd8 >= 6'(MAX_ROWS);
    assign matmul_prefetch_valid = !matmul_prefetch_sent &&
        prefetch_next_matmul_exists &&
        (state == MATMUL_WAIT || next_matmul_valid ||
         final_head_quant_commit || final_head_quant_advance ||
         state == LAYOUT_TO_PANEL) && !abort_request;

    assign rope_start_valid = state == ROPE_START && !abort_request;
    assign rope_source_is_k = qkv_select == QKV_K;
    assign rope_head = head_index;
    assign rope_start_fire = rope_start_valid && rope_start_ready;

    assign matmul_overlap_active = next_matmul_valid;
    assign head_staging_layout = !matmul_overlap_active &&
        (state == LAYOUT_TO_HEAD ||
        state == STAGING_READ_REQUEST || state == STAGING_READ_STREAM ||
        state == STAGE_FIND_ROW || state == STAGE_READ_REQ ||
        state == STAGE_READ_WAIT || state == STAGE_WRITE ||
        state == BATCH_START || state == MAX_STREAM ||
        state == SCALE_REQUEST || state == SCALE_RESPONSE ||
        state == QUANTIZED_STREAM ||
        (state == QUANTIZED_COMMIT && !final_head_quant_commit) ||
        (state == ADVANCE_BATCH && !final_head_quant_advance));
    assign stage_copy_active = state == STAGE_FIND_ROW ||
        state == STAGE_READ_REQ || state == STAGE_READ_WAIT;
    assign head_read_req_valid = state == STAGE_READ_REQ &&
        stage_metadata_input_ready && !stage_response_error && !abort_request;
    assign head_read_rope_destination = qkv_select != QKV_V;
    assign head_read_physical_row = physical_row;
    assign head_read_word = word_index;
    assign head_read_req_fire = head_read_req_valid && head_read_req_ready;
    assign stage_response_error = head_read_rsp_valid &&
        head_read_rsp_lane_mask != 8'hff;
    assign head_read_rsp_ready =
        (state == ERROR_DRAIN || state == ABORT_DRAIN) ?
            stage_metadata_output_valid :
        stage_copy_active && stage_metadata_output_valid &&
            (stage_response_error || head_stage_write_ready);
    assign head_read_rsp_fire = head_read_rsp_valid && head_read_rsp_ready;

    assign {stage_write_physical_row, stage_write_word} =
        stage_metadata_output_data;
    assign head_stage_write_valid = state == STAGING_READ_STREAM ?
        metadata_read_data_valid && metadata_read_byte_enable == 16'hffff &&
            !abort_request :
        stage_copy_active && head_read_rsp_valid && stage_metadata_output_valid &&
            !stage_response_error && !abort_request;
    assign head_stage_write_source = state == STAGING_READ_STREAM;
    assign head_stage_write_bank = state == STAGING_READ_STREAM ? 2'd0 :
        stage_write_physical_row[1:0];
    assign head_stage_write_port = state == STAGING_READ_STREAM ? 1'b0 :
        stage_write_physical_row[2];
    assign head_stage_write_word = state == STAGING_READ_STREAM ?
        {staging_response_row, staging_stripe[3:0]} :
        10'((q_head_command ? 0 :
            local_memory_layout_pkg::QKV_HEAD_STAGING_WORD_BASE) +
        (integer'(stage_write_physical_row) / 4) * 16 +
        integer'(stage_write_word));
    assign head_stage_write_data = state == STAGING_READ_STREAM ?
        metadata_read_data : head_read_rsp_data;
    assign head_stage_write_byte_enable = 16'hffff;
    assign head_stage_write_fire = head_stage_write_valid && head_stage_write_ready;

    ready_valid_fifo #(.DATA_WIDTH(10), .DEPTH(3)) stage_metadata_fifo (
        .clk(clk), .rst(rst),
        .input_valid(head_read_req_fire),
        .input_ready(stage_metadata_input_ready),
        .input_data({physical_row, word_index}),
        .output_valid(stage_metadata_output_valid),
        .output_ready(stage_metadata_output_ready),
        .output_data(stage_metadata_output_data),
        .occupancy(stage_metadata_occupancy)
    );
    assign stage_metadata_output_ready = head_read_rsp_fire;
    assign stage_copy_drained = head_read_outstanding == 3'd0 ||
        (head_read_outstanding == 3'd1 && head_read_rsp_fire);

    assign head_tile_read_req_valid =
        (state == MAX_STREAM && tile_issue_count < 5'd16) ||
        (state == QUANTIZED_STREAM && tile_issue_count < 5'd8);
    assign head_tile_read_row_base = batch_row_base;
    assign head_tile_read_word = state == QUANTIZED_STREAM ?
        {half_index, 3'b000} + tile_issue_count[2:0] : tile_issue_count[3:0];
    assign head_tile_read_row_mask = batch_row_mask;
    assign head_tile_read_req_fire = head_tile_read_req_valid &&
        head_tile_read_req_ready;

    // MAX_STREAM and QUANTIZED_STREAM exclusively use their II=1 consumer.
    // The two-cycle SRAM response therefore transfers directly without a
    // second payload queue.
    assign head_tile_memory_req_valid = head_tile_read_req_valid;
    assign head_tile_read_req_ready = head_tile_memory_req_ready;
    assign head_tile_memory_row_base = head_tile_read_row_base;
    assign head_tile_memory_word = head_tile_read_word;
    assign head_tile_memory_row_mask = head_tile_read_row_mask;
    assign head_tile_read_rsp_valid = head_tile_memory_rsp_valid;
    assign head_tile_read_rsp_data = head_tile_memory_rsp_data;
    assign head_tile_read_rsp_lane_mask = head_tile_memory_rsp_lane_mask;
    assign head_tile_fifo_occupancy = 3'd0;

    assign max_req_valid = state == MAX_STREAM && head_tile_read_rsp_valid;
    assign max_req_values = head_tile_read_rsp_data;
    assign max_req_lane_mask = head_tile_read_rsp_lane_mask;
    assign max_req_tag = make_tile_tag(qkv_select, head_index, batch_row_base,
        1'b0, tile_response_count[3:0]);
    assign max_req_fire = max_req_valid && max_req_ready;
    assign max_rsp_ready = state == MAX_STREAM || state == ERROR_DRAIN ||
        state == ABORT_DRAIN;
    assign max_rsp_fire = max_rsp_valid && max_rsp_ready;

    assign quant_req_valid = state == QUANTIZED_STREAM && head_tile_read_rsp_valid;
    assign quant_req_values_bf16 = head_tile_read_rsp_data;
    assign quant_req_lane_mask = head_tile_read_rsp_lane_mask;
    assign quant_req_tag = make_tile_tag(qkv_select, head_index, batch_row_base,
        half_index, tile_response_count[3:0]);
    assign quant_req_fire = quant_req_valid && quant_req_ready;
    assign quant_rsp_ready = state == QUANTIZED_STREAM || state == ERROR_DRAIN ||
        state == ABORT_DRAIN;
    assign quant_rsp_fire = quant_rsp_valid && quant_rsp_ready;

    assign head_tile_read_rsp_ready = state == MAX_STREAM ? max_req_ready :
        state == QUANTIZED_STREAM ? quant_req_ready :
        (state == ERROR_DRAIN || state == ABORT_DRAIN);
    assign head_tile_read_rsp_fire = head_tile_read_rsp_valid &&
        head_tile_read_rsp_ready;

    assign quant_scale_req_valid = state == SCALE_REQUEST && !abort_request;
    assign quant_scale_req_use_static_scale = qkv_select == QKV_V;
    assign quant_scale_req_static_scales_bf16 = {8{v_static_scale}};
    assign quant_scale_req_row_max_abs = row_max_abs;
    assign quant_scale_req_row_mask = batch_row_mask;
    assign quant_scale_req_fire = quant_scale_req_valid && quant_scale_req_ready;
    assign quant_scale_rsp_ready = state == SCALE_RESPONSE;
    assign quant_scale_rsp_fire = quant_scale_rsp_valid && quant_scale_rsp_ready;

    // In the integrated path this interface reports the cycle on which the
    // internal commit FSM accepts the payload.  It must not repeat the same
    // payload while that FSM writes Q SRAM or drains a K/V DMA request.
    assign next_commit_token_position = token_position[
        (batch_row_base + 6'(commit_row_index) + 6'd1)*11 +: 11];
    assign v_pair_available = INTEGRATED_CACHE_COMMIT && qkv_select == QKV_V &&
        commit_row_index != 3'd7 && batch_row_mask[commit_row_index + 3'd1] &&
        !kv_write_disable[batch_row_base + 6'(commit_row_index) + 6'd1] &&
        quantized_commit_token_position != 11'h7ff &&
        next_commit_token_position == quantized_commit_token_position + 11'd1;
    assign quantized_commit_valid = state == QUANTIZED_COMMIT && !abort_request &&
        batch_row_mask[commit_row_index] &&
        (!INTEGRATED_CACHE_COMMIT || commit_state == COMMIT_IDLE ||
         (commit_state == COMMIT_V_PAIR_WAIT && qkv_select == QKV_V &&
          head_index == commit_head && half_index == commit_half &&
          quantized_commit_physical_row == commit_row + 6'd1 &&
          quantized_commit_token_position ==
              commit_token_position + 11'd1));
    assign quantized_commit_qkv_select = qkv_select;
    assign quantized_commit_head = head_index;
    assign quantized_commit_physical_row = batch_row_base + 6'(commit_row_index);
    assign quantized_commit_token_position =
        token_position[(batch_row_base + 6'(commit_row_index))*11 +: 11];
    assign quantized_commit_half = half_index;
    assign quantized_commit_values = quantized_half_buffer[commit_row_index*512 +: 512];
    assign quantized_commit_lane_mask = {64{batch_row_mask[commit_row_index]}};
    assign quantized_commit_scale_bf16 = row_scale[commit_row_index*16 +: 16];
    assign quantized_commit_tag = make_tag(qkv_select, head_index,
        batch_row_base + 6'(commit_row_index), half_index);
    assign quantized_commit_requires_response = qkv_select != QKV_Q &&
        !kv_write_disable[quantized_commit_physical_row];
    assign quantized_commit_fire = quantized_commit_valid &&
        (INTEGRATED_CACHE_COMMIT || quantized_commit_ready);
    assign cache_write_complete_ready = 1'b1;
    assign integrated_commit_complete_valid = commit_state == COMMIT_COMPLETE &&
        commit_qkv_select != QKV_Q && commit_completion_count != 2'd0;
    assign cache_complete_fire = INTEGRATED_CACHE_COMMIT ?
        integrated_commit_complete_valid :
        (cache_write_complete_valid && cache_write_complete_ready);
    assign cache_write_error_fire = cache_complete_fire &&
        (INTEGRATED_CACHE_COMMIT ? commit_error : cache_write_complete_error);

    assign cache_mapping_commit_valid = state == CACHE_MAPPING_START &&
        outstanding_cache_writes == 0 && !abort_request;
    assign cache_config_start_valid = state == CACHE_CONFIG_START && !abort_request;
    assign cache_config_release = INTEGRATED_CACHE_COMMIT &&
        (state == ERROR_DRAIN || state == ABORT_DRAIN);

    assign matmul_abort_request = (state == ABORT_DRAIN || state == ERROR_DRAIN) &&
        matmul_active;
    assign rope_abort_request = (state == ABORT_DRAIN || state == ERROR_DRAIN) &&
        rope_active;
    assign quant_abort_request = (state == ABORT_DRAIN || state == ERROR_DRAIN) &&
        quant_active;
    assign cache_abort_request = !INTEGRATED_CACHE_COMMIT &&
        (state == ABORT_DRAIN || state == ERROR_DRAIN) &&
        outstanding_cache_writes != 0 && !cache_abort_seen;
    assign cache_writer_abort_request = INTEGRATED_CACHE_COMMIT &&
        (state == ABORT_DRAIN || state == ERROR_DRAIN) &&
        commit_state != COMMIT_IDLE && commit_state != COMMIT_COMPLETE;

    assign metadata_read_request_valid = INTEGRATED_CACHE_COMMIT &&
        (state == METADATA_REQUEST || state == STAGING_READ_REQUEST) &&
        !abort_request;
    assign metadata_read_request_address = state == STAGING_READ_REQUEST ?
        staging_stripe_address : metadata_row < MAX_ROWS ?
        (metadata_is_sin ? rope_sin_base : rope_cos_base) +
            64'(token_position[metadata_row*11 +: 11]) * 64'd256 :
        v_scale_base;
    assign metadata_read_request_bytes = state == STAGING_READ_REQUEST ?
        {22'd0, row_count, 4'd0} : metadata_row < MAX_ROWS ? 32'd128 :
        v_scale_per_head ? 32'd64 : 32'd2;
    assign metadata_read_data_ready = INTEGRATED_CACHE_COMMIT &&
        metadata_read_outstanding &&
        (((state == STAGING_READ_STREAM) &&
          (metadata_read_byte_enable != 16'hffff || head_stage_write_ready)) ||
         (state == METADATA_STREAM &&
          (metadata_row >= MAX_ROWS || constant_stage_write_ready)) ||
         state == ERROR_DRAIN || state == ABORT_DRAIN);
    assign constant_stage_write_valid = INTEGRATED_CACHE_COMMIT &&
        state == METADATA_STREAM && metadata_row < MAX_ROWS &&
        metadata_read_data_valid;
    assign constant_stage_write_word = 10'(
        integer'(metadata_row) * 16 + (metadata_is_sin ? 8 : 0) +
        integer'(metadata_word));
    assign constant_stage_write_data = metadata_read_data;
    assign constant_stage_write_byte_enable = metadata_read_byte_enable;

    assign q_local_write_valid = INTEGRATED_CACHE_COMMIT &&
        commit_state == COMMIT_Q_QUANTIZED ||
        (INTEGRATED_CACHE_COMMIT && commit_state == COMMIT_Q_SCALE);
    assign q_local_write_scale = commit_state == COMMIT_Q_SCALE;
    assign q_local_write_slot = q_head_slot;
    assign q_local_write_physical_row = commit_row;
    assign q_local_write_word = commit_state == COMMIT_Q_SCALE ?
        10'((q_head_slot ? 8 : 0) + integer'(commit_row[5:3])) :
        10'((commit_half ? 4 : 0) + integer'(commit_word));
    assign q_local_write_data = commit_state == COMMIT_Q_SCALE ?
        (128'(cache_commit_scale) <<
         (integer'(commit_row[2:0]) * 16)) :
        cache_commit_values[commit_word*128 +: 128];
    assign q_local_write_byte_enable = commit_state == COMMIT_Q_SCALE ?
        (16'(16'h0003) <<
         (integer'(commit_row[2:0]) * 2)) :
        16'hffff;

    assign current_write_valid = INTEGRATED_CACHE_COMMIT &&
        (commit_state == COMMIT_ADDRESS ||
         commit_state == COMMIT_V_PAIR_ADDRESS);
    assign current_write_head = commit_head;
    assign current_write_physical_row =
        commit_state == COMMIT_V_PAIR_ADDRESS ? commit_pair_row : commit_row;
    assign current_write_logical_slot =
        commit_state == COMMIT_V_PAIR_ADDRESS ?
            commit_pair_token_position : commit_token_position;
    assign current_write_chunk = commit_qkv_select == QKV_K ?
        {1'b0, commit_half, 3'b000} : {1'b0, commit_v_chunk};
    assign cache_writer_start_valid = INTEGRATED_CACHE_COMMIT &&
        commit_state == COMMIT_DMA_START && !commit_error;
    assign cache_writer_start_fire = cache_writer_start_valid &&
        cache_writer_start_ready;
    assign cache_writer_start_address = commit_write_scale ?
        commit_k_scale_address : commit_qkv_select == QKV_K ?
        commit_k_address : commit_v_address;
    assign cache_writer_start_bytes = commit_write_scale ? 32'd2 :
        commit_qkv_select == QKV_K ? 32'd64 :
        commit_v_pair ? 32'd16 : 32'd8;
    assign cache_writer_data_valid = INTEGRATED_CACHE_COMMIT &&
        commit_state == COMMIT_DMA_STREAM;
    assign cache_writer_data = commit_write_scale ? {112'd0, cache_commit_scale} :
        commit_qkv_select == QKV_K ?
        cache_commit_values[commit_word*128 +: 128] :
        commit_v_pair ?
        {cache_commit_pair_values[commit_v_chunk[2:0]*64 +: 64],
         cache_commit_values[commit_v_chunk[2:0]*64 +: 64]} :
        {64'd0, cache_commit_values[commit_v_chunk[2:0]*64 +: 64]};
    assign cache_writer_byte_enable = commit_write_scale ? 16'h0003 :
        commit_qkv_select == QKV_K || commit_v_pair ? 16'hffff : 16'h00ff;
    assign cache_writer_data_last = commit_write_scale ||
        commit_qkv_select == QKV_V || commit_word == 3'd3;
    wire cache_abort_completion = INTEGRATED_CACHE_COMMIT ?
        cache_writer_abort_ack : cache_abort_ack;

    operator_dma_writer #(
        .REQUEST_TAG(8'h92),
        .MAX_PENDING_WRITES(8)
    ) cache_dma_writer (
        .clk(clk),
        .rst(rst),
        .abort_request(cache_writer_abort_request),
        .start_valid(cache_writer_start_valid),
        .start_ready(cache_writer_start_ready),
        .start_address(cache_writer_start_address),
        .start_bytes(cache_writer_start_bytes),
        .request_valid(cache_writer_request_valid),
        .request_ready(cache_writer_request_ready),
        .request_address(cache_writer_request_address),
        .request_bytes(cache_writer_request_bytes),
        .request_tag(cache_writer_request_tag),
        .data_valid(cache_writer_data_valid),
        .data_ready(cache_writer_data_ready),
        .data(cache_writer_data),
        .data_byte_enable(cache_writer_byte_enable),
        .data_last(cache_writer_data_last),
        .write_valid(cache_writer_write_valid),
        .write_ready(cache_writer_write_ready),
        .write_data(cache_writer_write_data),
        .write_byte_enable(cache_writer_write_byte_enable),
        .write_last(cache_writer_write_last),
        .transaction_done(cache_writer_request_done),
        .transaction_error(cache_writer_request_error),
        .done_pulse(cache_writer_done),
        .error(cache_writer_error),
        .abort_ack(cache_writer_abort_ack)
    );

`ifdef SYNTHESIS
    always_comb begin
        accepted_matmul_count = '0;
        accepted_rope_count = '0;
        accepted_head_read_count = '0;
        accepted_quant_count = '0;
        accepted_quantized_commit_count = '0;
        completed_cache_write_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            row_count <= '0;
            other_row_count <= '0;
            other_row_enable <= '0;
            other_kv_write_disable <= '0;
            other_token_position <= '0;
            batch_group <= '0;
            batch_group_active <= 1'b0;
            batch_group_count <= 3'd1;
            group_staging_base <= '0;
            group_staging_head_stride <= '0;
            kv_pair_active <= 1'b0;
            kv_second_batch <= 1'b0;
            kv_batch_index <= 3'd0;
            group_metadata_refill <= 1'b0;
            staging_batch_base <= '0;
            staging_stripe_address <= '0;
            staging_stripe <= '0;
            staging_response_row <= '0;
            kv_resume_state <= IDLE;
            head_count <= '0;
            row_enable <= '0;
            kv_write_disable <= '0;
            token_position <= '0;
            v_static_scale <= '0;
            v_scale_per_head <= 1'b0;
            for (integer scale_word = 0; scale_word < 4; scale_word++)
                v_scale_table_word[scale_word] <= '0;
            v_scale_table_valid <= 1'b0;
            v_scale_metadata_error <= 1'b0;
            rope_cos_base <= '0;
            rope_sin_base <= '0;
            v_scale_base <= '0;
            metadata_row <= '0;
            metadata_is_sin <= 1'b0;
            metadata_word <= '0;
            metadata_read_outstanding <= 1'b0;
            qkv_select <= QKV_V;
            q_head_command <= 1'b0;
            q_head_slot <= 1'b0;
            q_head_matmul_done_seen <= 1'b0;
            q_head_skip_group_matmul <= 1'b0;
            q_head_precompute_only <= 1'b0;
            kv_group_activation_reuse <= 1'b0;
            kv_group_precompute <= 1'b0;
            q_group_head_base <= 64'd0;
            q_group_last_head <= 6'd0;
            q_group_active_batch <= 3'd0;
            q_group_head_sequence_valid <= 1'b0;
            head_index <= '0;
            physical_row <= '0;
            row_scan_index <= '0;
            word_index <= '0;
            half_index <= 1'b0;
            row_max_abs <= '0;
            row_scale <= '0;
            batch_row_base <= '0;
            batch_row_mask <= '0;
            quantized_half_buffer <= '0;
            tile_issue_count <= '0;
            tile_response_count <= '0;
            pipeline_response_count <= '0;
            commit_row_index <= '0;
            head_read_outstanding <= '0;
            head_tile_read_outstanding <= 1'b0;
            head_tile_memory_pending <= '0;
            max_outstanding <= '0;
            quant_values_outstanding <= '0;
            matmul_active <= 1'b0;
            next_matmul_valid <= 1'b0;
            next_matmul_started <= 1'b0;
            next_matmul_done <= 1'b0;
            next_matmul_qkv_select <= QKV_V;
            next_matmul_head <= '0;
            rope_active <= 1'b0;
            quant_active <= 1'b0;
            matmul_prefetch_sent <= 1'b0;
            cache_abort_seen <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
`ifndef SYNTHESIS
            accepted_matmul_count <= '0;
            accepted_rope_count <= '0;
            accepted_head_read_count <= '0;
            accepted_quant_count <= '0;
            accepted_quantized_commit_count <= '0;
            completed_cache_write_count <= '0;
`endif
            outstanding_cache_writes <= '0;
            commit_state <= COMMIT_IDLE;
            commit_qkv_select <= QKV_Q;
            commit_head <= '0;
            commit_row <= '0;
            commit_token_position <= '0;
            commit_half <= 1'b0;
            commit_word <= '0;
            commit_v_chunk <= '0;
            commit_k_address <= '0;
            commit_v_address <= '0;
            commit_k_scale_address <= '0;
            commit_pair_row <= '0;
            commit_pair_token_position <= '0;
            commit_write_scale <= 1'b0;
            commit_error <= 1'b0;
            commit_v_pair <= 1'b0;
            commit_completion_count <= '0;
            commit_dma_outstanding <= '0;
            cache_commit_values <= '0;
            cache_commit_pair_values <= '0;
            cache_commit_scale <= '0;
        end else begin
            abort_ack <= 1'b0;

            if (matmul_start_fire) begin
                matmul_active <= 1'b1;
                matmul_prefetch_sent <= 1'b0;
`ifndef SYNTHESIS
                accepted_matmul_count <= accepted_matmul_count + 32'd1;
`endif
                if (next_matmul_valid)
                    next_matmul_started <= 1'b1;
            end
            if (matmul_prefetch_valid && matmul_prefetch_ready)
                matmul_prefetch_sent <= 1'b1;
            if (matmul_done_valid || matmul_error)
                matmul_active <= 1'b0;
            if (matmul_abort_ack)
                matmul_active <= 1'b0;
            if (next_matmul_valid && next_matmul_started && matmul_done_valid)
                next_matmul_done <= 1'b1;
            if (rope_start_fire) begin
                rope_active <= 1'b1;
`ifndef SYNTHESIS
                accepted_rope_count <= accepted_rope_count + 32'd1;
`endif
            end
            if (rope_done_pulse || rope_error)
                rope_active <= 1'b0;
            if (rope_abort_ack)
                rope_active <= 1'b0;
`ifndef SYNTHESIS
            if (head_read_req_fire)
                accepted_head_read_count <= accepted_head_read_count + 32'd1;
`endif
            case ({head_read_req_fire, head_read_rsp_fire})
                2'b10: head_read_outstanding <= head_read_outstanding + 3'd1;
                2'b01: head_read_outstanding <= head_read_outstanding - 3'd1;
                default: head_read_outstanding <= head_read_outstanding;
            endcase
            case ({head_tile_read_req_fire, head_tile_read_rsp_fire})
                2'b10: head_tile_read_outstanding <= 1'b1;
                2'b01: head_tile_read_outstanding <= 1'b0;
                2'b11: head_tile_read_outstanding <= 1'b1;
                default: head_tile_read_outstanding <= head_tile_read_outstanding;
            endcase
            head_tile_memory_pending <= {head_tile_memory_pending[0],
                head_tile_read_req_fire};
            case ({max_req_fire, max_rsp_fire})
                2'b10: max_outstanding <= max_outstanding + 6'd1;
                2'b01: max_outstanding <= max_outstanding - 6'd1;
                default: max_outstanding <= max_outstanding;
            endcase
            case ({quant_req_fire, quant_rsp_fire})
                2'b10: quant_values_outstanding <= quant_values_outstanding + 6'd1;
                2'b01: quant_values_outstanding <= quant_values_outstanding - 6'd1;
                default: quant_values_outstanding <= quant_values_outstanding;
            endcase
            if (quant_abort_ack)
                quant_values_outstanding <= '0;
            if (quant_scale_req_fire || quant_req_fire) begin
                quant_active <= 1'b1;
            end
`ifndef SYNTHESIS
            if (quant_req_fire) begin
                accepted_quant_count <= accepted_quant_count + 32'd1;
            end
`endif
            if (quant_scale_rsp_fire ||
                (quant_rsp_fire && quant_values_outstanding == 6'd1) ||
                quant_abort_ack)
                quant_active <= 1'b0;
`ifndef SYNTHESIS
            if (quantized_commit_fire)
                accepted_quantized_commit_count <= accepted_quantized_commit_count + 32'd1;
            if (cache_complete_fire)
                completed_cache_write_count <= completed_cache_write_count + 32'd1;
`endif
            if (metadata_read_request_valid && metadata_read_request_ready)
                metadata_read_outstanding <= 1'b1;
            if (metadata_read_request_done || metadata_read_request_error)
                metadata_read_outstanding <= 1'b0;

            if (INTEGRATED_CACHE_COMMIT) begin
                case ({cache_writer_start_fire, cache_writer_done})
                    2'b10: commit_dma_outstanding <=
                        commit_dma_outstanding + 4'd1;
                    2'b01: commit_dma_outstanding <=
                        commit_dma_outstanding - 4'd1;
                    default: commit_dma_outstanding <=
                        commit_dma_outstanding;
                endcase
                if (cache_writer_done && cache_writer_error)
                    commit_error <= 1'b1;
                case (commit_state)
                    COMMIT_IDLE: if (quantized_commit_fire &&
                                    (qkv_select == QKV_Q || quantized_commit_requires_response)) begin
                        commit_qkv_select <= qkv_select;
                        commit_head <= head_index;
                        commit_row <= quantized_commit_physical_row;
                        commit_token_position <=
                            quantized_commit_token_position;
                        commit_half <= half_index;
                        cache_commit_values <= quantized_commit_values;
                        cache_commit_scale <= quantized_commit_scale_bf16;
                        commit_word <= 3'd0;
                        commit_v_chunk <= half_index ? 4'd8 : 4'd0;
                        commit_write_scale <= 1'b0;
                        commit_error <= 1'b0;
                        commit_v_pair <= 1'b0;
                        commit_completion_count <=
                            qkv_select == QKV_Q ? 2'd0 : 2'd1;
                        commit_state <= qkv_select == QKV_Q ?
                            COMMIT_Q_QUANTIZED :
                            v_pair_available ? COMMIT_V_PAIR_WAIT :
                            COMMIT_ADDRESS;
                    end
                    COMMIT_V_PAIR_WAIT: if (quantized_commit_fire) begin
                        commit_pair_row <= quantized_commit_physical_row;
                        commit_pair_token_position <=
                            quantized_commit_token_position;
                        cache_commit_pair_values <= quantized_commit_values;
                        commit_v_pair <= 1'b1;
                        commit_completion_count <= 2'd2;
                        commit_state <= COMMIT_ADDRESS;
                    end
                    COMMIT_Q_QUANTIZED: if (q_local_write_valid && q_local_write_ready) begin
                        if (commit_word == 3'd3) begin
                            commit_word <= 3'd0;
                            commit_state <= commit_half ? COMMIT_COMPLETE :
                                COMMIT_Q_SCALE;
                        end else begin
                            commit_word <= commit_word + 3'd1;
                        end
                    end
                    COMMIT_Q_SCALE: if (q_local_write_valid && q_local_write_ready)
                        commit_state <= COMMIT_COMPLETE;
                    COMMIT_ADDRESS: if (current_write_valid && current_write_ready) begin
                        if (!current_write_accepted) begin
                            commit_error <= 1'b1;
                            commit_state <= commit_dma_outstanding == 0 ?
                                COMMIT_COMPLETE : COMMIT_DMA_WAIT;
                        end else begin
                            commit_k_address <= current_k_write_address;
                            commit_v_address <= current_v_write_address;
                            commit_k_scale_address <= current_k_scale_write_address;
                            commit_word <= 3'd0;
                            commit_state <= commit_v_pair ?
                                COMMIT_V_PAIR_ADDRESS : COMMIT_DMA_START;
                        end
                    end
                    COMMIT_V_PAIR_ADDRESS: if (current_write_valid &&
                                                   current_write_ready) begin
                        if (!current_write_accepted ||
                            current_v_write_address != commit_v_address + 64'd8) begin
                            commit_error <= 1'b1;
                            commit_state <= commit_dma_outstanding == 0 ?
                                COMMIT_COMPLETE : COMMIT_DMA_WAIT;
                        end else begin
                            commit_state <= COMMIT_DMA_START;
                        end
                    end
                    COMMIT_DMA_START: begin
                        if (commit_error)
                            commit_state <= commit_dma_outstanding == 0 ?
                                COMMIT_COMPLETE : COMMIT_DMA_WAIT;
                        else if (cache_writer_start_fire)
                            commit_state <= COMMIT_DMA_STREAM;
                    end
                    COMMIT_DMA_STREAM: if (cache_writer_data_valid &&
                                                cache_writer_data_ready) begin
                        if (cache_writer_data_last)
                            commit_state <= COMMIT_ADVANCE;
                        else
                            commit_word <= commit_word + 3'd1;
                    end
                    COMMIT_DMA_WAIT: if (commit_dma_outstanding == 0)
                        commit_state <= COMMIT_COMPLETE;
                    COMMIT_ADVANCE: begin
                        if (commit_error) begin
                            commit_state <= commit_dma_outstanding == 0 ?
                                COMMIT_COMPLETE : COMMIT_DMA_WAIT;
                        end else if (commit_qkv_select == QKV_K &&
                            !commit_write_scale && !commit_half) begin
                            commit_write_scale <= 1'b1;
                            commit_state <= COMMIT_DMA_START;
                        end else if (commit_qkv_select == QKV_V &&
                                     commit_v_chunk[2:0] != 3'd7) begin
                            commit_v_chunk <= commit_v_chunk + 4'd1;
                            commit_state <= COMMIT_ADDRESS;
                        end else begin
                            commit_state <= commit_dma_outstanding == 0 ?
                                COMMIT_COMPLETE : COMMIT_DMA_WAIT;
                        end
                    end
                    COMMIT_COMPLETE: begin
                        if (commit_qkv_select == QKV_Q ||
                            commit_completion_count <= 2'd1) begin
                            commit_completion_count <= 2'd0;
                            commit_state <= COMMIT_IDLE;
                        end else begin
                            commit_completion_count <=
                                commit_completion_count - 2'd1;
                        end
                    end
                    default: commit_state <= COMMIT_IDLE;
                endcase
                if ((state == ABORT_DRAIN || state == ERROR_DRAIN) &&
                    cache_writer_abort_ack) begin
                    commit_state <= COMMIT_IDLE;
                    commit_error <= 1'b0;
                    commit_dma_outstanding <= 4'd0;
                end
            end

            case ({quantized_commit_fire && quantized_commit_requires_response,
                   cache_complete_fire})
                2'b10: outstanding_cache_writes <= outstanding_cache_writes + 16'd1;
                2'b01: outstanding_cache_writes <= outstanding_cache_writes - 16'd1;
                default: outstanding_cache_writes <= outstanding_cache_writes;
            endcase
            if (INTEGRATED_CACHE_COMMIT && cache_writer_abort_ack)
                outstanding_cache_writes <= 16'd0;

            if (cache_write_error_fire) begin
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_CACHE_WRITE;
                state <= ERROR_DRAIN;
            end

            if (next_matmul_error_fire) begin
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_MATMUL;
                next_matmul_valid <= 1'b0;
                next_matmul_started <= 1'b0;
                next_matmul_done <= 1'b0;
                state <= ERROR_DRAIN;
            end

            if (!cache_write_error_fire && !next_matmul_error_fire) case (state)
                IDLE: begin
                    if (start_valid && start_ready) begin
                        if (!TRUSTED_START_CONFIGURATION)
                            batch_group <= start_batch_group;
                        batch_group_active <= start_batch_group.enable;
                        kv_group_activation_reuse <= start_kv_group_reuse;
                        kv_group_precompute <= start_kv_group_reuse;
                        q_group_head_base <= start_batch_group.staging_base;
                        batch_group_count <= start_batch_group.enable ?
                            start_batch_group.batch_count :
                            (start_kv_pair_enable ? 3'd2 : 3'd1);
                        group_staging_base <= start_batch_group.staging_base;
                        group_staging_head_stride <=
                            start_batch_group.staging_head_stride;
                        kv_pair_active <= start_kv_pair_enable ||
                            start_batch_group.enable;
                        kv_second_batch <= 1'b0;
                        kv_batch_index <= 3'd0;
                        group_metadata_refill <= 1'b0;
                        staging_batch_base <= start_batch_group.staging_base;
                        staging_stripe_address <= start_batch_group.staging_base;
                        staging_stripe <= 5'd0;
                        staging_response_row <= 6'd0;
                        other_row_count <= start_second_row_count;
                        other_row_enable <= start_second_row_enable;
                        other_kv_write_disable <= start_second_kv_write_disable;
                        other_token_position <= start_second_token_position;
                        q_head_command <= 1'b0;
                        q_head_skip_group_matmul <= 1'b0;
                        q_head_precompute_only <= 1'b0;
                        q_group_head_sequence_valid <= 1'b0;
                        q_head_matmul_done_seen <= 1'b0;
                        terminal_error <= !start_configuration_valid;
                        terminal_error_id <= start_configuration_valid ? 8'd0 : ERROR_CONFIG;
                        row_count <= start_row_count;
                        head_count <= start_head_count;
                        row_enable <= start_row_enable;
                        kv_write_disable <= start_kv_write_disable;
                        token_position <= start_token_position;
                        v_static_scale <= start_v_static_scale_bf16;
                        v_scale_per_head <= start_v_scale_per_head;
                        rope_cos_base <= start_rope_cos_base;
                        rope_sin_base <= start_rope_sin_base;
                        v_scale_base <= start_v_scale_base;
                        v_scale_table_valid <= !INTEGRATED_CACHE_COMMIT;
                        v_scale_metadata_error <= 1'b0;
                        for (integer scale_word = 0; scale_word < 4;
                             scale_word++)
                            v_scale_table_word[scale_word] <= '0;
                        qkv_select <= QKV_V;
                        head_index <= 6'd0;
                        next_matmul_valid <= 1'b0;
                        next_matmul_started <= 1'b0;
                        next_matmul_done <= 1'b0;
                        next_matmul_qkv_select <= QKV_V;
                        next_matmul_head <= 6'd0;
                        outstanding_cache_writes <= 16'd0;
                        cache_abort_seen <= 1'b0;
                        metadata_row <= 6'd0;
                        metadata_is_sin <= 1'b0;
                        metadata_word <= 4'd0;
                        state <= !start_configuration_valid ? COMPLETE :
                            INTEGRATED_CACHE_COMMIT ? METADATA_REQUEST :
                            MATMUL_START;
                    end else if (q_head_start_valid && q_head_start_ready) begin
                        kv_pair_active <= 1'b0;
                        if (!TRUSTED_START_CONFIGURATION)
                            batch_group <= start_batch_group;
                        batch_group_active <= start_batch_group.enable;
                        batch_group_count <= start_batch_group.enable ?
                            start_batch_group.batch_count : 3'd1;
                        group_staging_base <= start_batch_group.staging_base;
                        group_staging_head_stride <=
                            start_batch_group.staging_head_stride;
                        kv_second_batch <= start_batch_group.enable &&
                            start_batch_group.active_batch_index != 0;
                        kv_batch_index <= start_batch_group.enable ?
                            start_batch_group.active_batch_index : 3'd0;
                        terminal_error <= !q_head_configuration_valid;
                        terminal_error_id <= q_head_configuration_valid ?
                            8'd0 : ERROR_CONFIG;
                        q_head_command <= 1'b1;
                        q_head_slot <= q_head_start_slot;
                        q_head_skip_group_matmul <=
                            start_batch_group.enable &&
                            (start_batch_group.active_batch_index != 0 ||
                             q_head_start_use_staged);
                        q_head_matmul_done_seen <= start_batch_group.enable &&
                            (start_batch_group.active_batch_index != 0 ||
                             q_head_start_use_staged);
                        kv_group_activation_reuse <= 1'b0;
                        kv_group_precompute <= 1'b0;
                        q_head_precompute_only <=
                            q_head_start_precompute_only;
                        if (start_batch_group.enable) begin
                            q_group_head_base <=
                                q_head_selected_head_base[63:0];
                            q_group_last_head <= q_head_start_head;
                            q_group_active_batch <=
                                start_batch_group.active_batch_index;
                            q_group_head_sequence_valid <=
                                q_head_configuration_valid;
                            staging_batch_base <=
                                q_head_staging_base_sum[63:0];
                            staging_stripe_address <=
                                q_head_staging_base_sum[63:0];
                            staging_stripe <= 5'd0;
                            staging_response_row <= 6'd0;
                        end
                        row_count <= start_row_count;
                        head_count <= start_head_count;
                        row_enable <= start_row_enable;
                        kv_write_disable <= start_kv_write_disable;
                        token_position <= start_token_position;
                        qkv_select <= QKV_Q;
                        head_index <= q_head_start_head;
                        next_matmul_valid <= 1'b0;
                        next_matmul_started <= 1'b0;
                        next_matmul_done <= 1'b0;
                        cache_abort_seen <= 1'b0;
                        if (INTEGRATED_CACHE_COMMIT && q_head_preprocess_input &&
                            q_head_start_head == 6'd0) begin
                            rope_cos_base <= start_rope_cos_base;
                            rope_sin_base <= start_rope_sin_base;
                            metadata_row <= 6'd0;
                            metadata_is_sin <= 1'b0;
                            metadata_word <= 4'd0;
                            state <= q_head_configuration_valid ?
                                METADATA_REQUEST : COMPLETE;
                        end else begin
                            state <= !q_head_configuration_valid ? COMPLETE :
                                (start_batch_group.enable &&
                                 (start_batch_group.active_batch_index != 0 ||
                                  q_head_start_use_staged)) ?
                                    STAGING_READ_REQUEST : MATMUL_START;
                        end
                    end
                end
                METADATA_REQUEST: if (metadata_read_request_valid &&
                                         metadata_read_request_ready) begin
                    metadata_word <= 4'd0;
                    state <= METADATA_STREAM;
                end
                METADATA_STREAM: begin
                    if (metadata_read_data_valid && metadata_read_data_ready) begin
                        if (metadata_row >= MAX_ROWS && v_scale_per_head) begin
                            case (metadata_word)
                                4'd0: v_scale_table_word[0] <= metadata_read_data;
                                4'd1: v_scale_table_word[1] <= metadata_read_data;
                                4'd2: v_scale_table_word[2] <= metadata_read_data;
                                4'd3: v_scale_table_word[3] <= metadata_read_data;
                                default: begin end
                            endcase
                            if (!valid_positive_scale_beat(
                                    metadata_read_data,
                                    metadata_read_byte_enable))
                                v_scale_metadata_error <= 1'b1;
                            metadata_word <= metadata_word + 4'd1;
                        end else if (metadata_row >= MAX_ROWS) begin
                            v_static_scale <= metadata_read_data[15:0];
                        end else begin
                            metadata_word <= metadata_word + 4'd1;
                        end
                    end
                    if (metadata_read_request_done || metadata_read_request_error) begin
                        if (metadata_read_request_error ||
                            (metadata_row < MAX_ROWS &&
                             metadata_word +
                                 (metadata_read_data_valid && metadata_read_data_ready) !=
                                 4'd8) ||
                            (metadata_row >= MAX_ROWS && v_scale_per_head &&
                             (metadata_word +
                                  (metadata_read_data_valid &&
                                   metadata_read_data_ready) != 4'd4 ||
                              v_scale_metadata_error ||
                              (metadata_read_data_valid &&
                               metadata_read_data_ready &&
                               !valid_positive_scale_beat(
                                   metadata_read_data,
                                   metadata_read_byte_enable)))) ||
                            (metadata_row >= MAX_ROWS && !v_scale_per_head &&
                             (!valid_positive_scale(
                                 metadata_read_data_valid &&
                                 metadata_read_data_ready ?
                                     metadata_read_data[15:0] :
                                     v_static_scale) ||
                              (metadata_read_data_valid &&
                               metadata_read_data_ready &&
                               metadata_read_byte_enable[1:0] != 2'b11)))) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_METADATA_READ;
                            v_scale_table_valid <= 1'b0;
                            state <= ERROR_DRAIN;
                        end else if (metadata_row >= MAX_ROWS) begin
                            v_scale_table_valid <= v_scale_per_head;
                            state <= CACHE_CONFIG_START;
                        end else if (!metadata_is_sin) begin
                            metadata_is_sin <= 1'b1;
                            state <= METADATA_REQUEST;
                        end else if (metadata_row + 6'd1 < row_count) begin
                            metadata_row <= metadata_row + 6'd1;
                            metadata_is_sin <= 1'b0;
                            state <= METADATA_REQUEST;
                        end else if (batch_group_active && group_metadata_refill) begin
                            group_metadata_refill <= 1'b0;
                            staging_stripe <= 5'd0;
                            staging_response_row <= 6'd0;
                            staging_stripe_address <= staging_batch_base;
                            state <= STAGING_READ_REQUEST;
                        end else if (q_head_command) begin
                            state <= q_head_skip_group_matmul ?
                                STAGING_READ_REQUEST : MATMUL_START;
                        end else if (batch_group_active) begin
                            metadata_row <= 6'(MAX_ROWS);
                            state <= METADATA_REQUEST;
                        end else if (kv_pair_active) begin
                            metadata_row <= kv_second_batch ? 6'(MAX_ROWS) : 6'd0;
                            metadata_is_sin <= 1'b0;
                            kv_resume_state <= METADATA_REQUEST;
                            state <= KV_BATCH_SWITCH;
                        end else begin
                            metadata_row <= 6'(MAX_ROWS);
                            state <= METADATA_REQUEST;
                        end
                    end
                end
                CACHE_CONFIG_START: if (cache_config_start_valid &&
                                            cache_config_start_ready)
                    state <= CACHE_CONFIG_WAIT;
                CACHE_CONFIG_WAIT: begin
                    if (cache_config_done_pulse) begin
                        if (cache_config_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_CACHE_MAPPING;
                            state <= ERROR_DRAIN;
                        end else begin
                            state <= MATMUL_START;
                        end
                    end
                end
                MATMUL_START: if (matmul_start_fire)
                    state <= MATMUL_WAIT;
                MATMUL_WAIT: begin
                    if (matmul_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_MATMUL;
                        state <= ERROR_DRAIN;
                    end else if (matmul_done_valid) begin
                        if (q_head_command)
                            q_head_matmul_done_seen <= 1'b1;
                        if (q_head_command && q_head_precompute_only) begin
                            state <= LAYOUT_TO_PANEL;
                        end else if (kv_group_activation_reuse && kv_group_precompute) begin
                            if (head_index + 6'd1 < head_count) begin
                                head_index <= head_index + 6'd1;
                                q_group_head_base <= q_group_head_base + {32'd0, group_staging_head_stride};
                                state <= MATMUL_START;
                            end else begin
                                // No postprocessing has touched expanded activation yet.
                                // After this point head staging may overwrite those banks.
                                kv_group_precompute <= 1'b0;
                                head_index <= 6'd0;
                                q_group_head_base <= group_staging_base;
                                staging_batch_base <= group_staging_base;
                                staging_stripe_address <= group_staging_base;
                                staging_stripe <= '0;
                                staging_response_row <= '0;
                                if (qkv_select == QKV_V) state <= STAGING_READ_REQUEST;
                                else begin
                                    metadata_row <= '0; metadata_is_sin <= 1'b0;
                                    group_metadata_refill <= 1'b1;
                                    state <= METADATA_REQUEST;
                                end
                            end
                        end else if (batch_group_active) begin
                            staging_stripe <= 5'd0;
                            staging_response_row <= 6'd0;
                            staging_stripe_address <= q_head_command ?
                                q_group_head_base : group_staging_base;
                            staging_batch_base <= q_head_command ?
                                q_group_head_base : group_staging_base;
                            if (qkv_select == QKV_V) begin
                                state <= STAGING_READ_REQUEST;
                            end else begin
                                metadata_row <= 6'd0;
                                metadata_is_sin <= 1'b0;
                                group_metadata_refill <= 1'b1;
                                state <= METADATA_REQUEST;
                            end
                        end else if (qkv_select == QKV_V) begin
                            state <= LAYOUT_TO_HEAD;
                        end else if (q_head_command &&
                                     !q_head_postprocess_enable) begin
                            state <= WAIT_Q_POST;
                        end else begin
                            state <= ROPE_START;
                        end
                    end
                end
                WAIT_Q_POST: if (q_head_postprocess_enable)
                    state <= ROPE_START;
                ROPE_START: if (rope_start_fire)
                    state <= ROPE_WAIT;
                ROPE_WAIT: begin
                    if (rope_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_ROPE;
                        state <= ERROR_DRAIN;
                    end else if (rope_done_pulse) begin
                        state <= LAYOUT_TO_HEAD;
                    end
                end
                STAGING_READ_REQUEST: if (metadata_read_request_valid &&
                    metadata_read_request_ready) begin
                    staging_response_row <= 6'd0;
                    state <= STAGING_READ_STREAM;
                end
                STAGING_READ_STREAM: begin
                    if (metadata_read_data_valid && metadata_read_data_ready) begin
                        if (metadata_read_byte_enable != 16'hffff) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_METADATA_READ;
                            state <= ERROR_DRAIN;
                        end else begin
                            staging_response_row <= staging_response_row + 6'd1;
                        end
                    end
                    if (metadata_read_request_done || metadata_read_request_error) begin
                        if (metadata_read_request_error ||
                            staging_response_row +
                                (metadata_read_data_valid && metadata_read_data_ready) !=
                                    row_count) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_METADATA_READ;
                            state <= ERROR_DRAIN;
                        end else if (staging_stripe == 5'd15) begin
                            // Staged Q shares RoPE arithmetic with Softmax,
                            // just like the direct Matmul result path.
                            state <= qkv_select == QKV_V ?
                                LAYOUT_TO_HEAD :
                                q_head_command && !q_head_postprocess_enable ?
                                    WAIT_Q_POST : ROPE_START;
                        end else begin
                            staging_stripe <= staging_stripe + 5'd1;
                            staging_stripe_address <= staging_stripe_address +
                                {54'd0, row_count, 4'd0};
                            state <= STAGING_READ_REQUEST;
                        end
                    end
                end
                LAYOUT_TO_HEAD: if (head_staging_layout_ready) begin
                    // The next head may compute while this head quantizes.
                    if (INTEGRATED_CACHE_COMMIT && v_scale_per_head &&
                        qkv_select == QKV_V)
                        v_static_scale <= selected_v_scale;
                    row_scan_index <= 6'd0;
                    state <= STAGE_FIND_ROW;
                end
                STAGE_FIND_ROW: begin
                    if (head_read_rsp_fire && stage_response_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_HEAD_READ;
                        state <= ERROR_DRAIN;
                    end else if (row_scan_index < 6'(MAX_ROWS)) begin
                        if (row_enable[row_scan_index]) begin
                            physical_row <= row_scan_index;
                            word_index <= 4'd0;
                            state <= STAGE_READ_REQ;
                        end else begin
                            row_scan_index <= row_scan_index + 6'd1;
                        end
                    end else begin
                        state <= STAGE_READ_WAIT;
                    end
                end
                STAGE_READ_REQ: begin
                    if (head_read_rsp_fire && stage_response_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_HEAD_READ;
                        state <= ERROR_DRAIN;
                    end else if (head_read_req_fire) begin
                        if (word_index == 4'd15) begin
                            row_scan_index <= physical_row + 6'd1;
                            state <= STAGE_FIND_ROW;
                        end else begin
                            word_index <= word_index + 4'd1;
                        end
                    end
                end
                STAGE_READ_WAIT: begin
                    if (head_read_rsp_fire && stage_response_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_HEAD_READ;
                        state <= ERROR_DRAIN;
                    end else if (stage_copy_drained &&
                                 stage_metadata_occupancy == 3'd0) begin
                        batch_row_base <= 6'd0;
                        if (qkv_select != QKV_Q && next_matmul_exists &&
                            matmul_prefetch_sent) begin
                            next_matmul_valid <= 1'b1;
                            next_matmul_started <= 1'b0;
                            next_matmul_done <= 1'b0;
                            next_matmul_qkv_select <=
                                head_index + 6'd1 < head_count ?
                                    qkv_select :
                                    qkv_select == QKV_V ? QKV_K : QKV_Q;
                            next_matmul_head <=
                                head_index + 6'd1 < head_count ?
                                    head_index + 6'd1 : 6'd0;
                        end
                        state <= BATCH_START;
                    end
                end
                STAGE_WRITE: state <= STAGE_READ_WAIT;
                BATCH_START: begin
                    if (enabled_rows_for_batch(batch_row_base) == 8'd0) begin
                        if (batch_row_base + 6'd8 < 6'(MAX_ROWS)) begin
                            batch_row_base <= batch_row_base + 6'd8;
                        end else begin
                            state <= LAYOUT_TO_PANEL;
                        end
                    end else begin
                        batch_row_mask <= enabled_rows_for_batch(batch_row_base);
                        row_max_abs <= '0;
                        tile_issue_count <= '0;
                        tile_response_count <= '0;
                        pipeline_response_count <= '0;
                        state <= qkv_select == QKV_V ?
                            SCALE_REQUEST : MAX_STREAM;
                    end
                end
                MAX_STREAM: begin
                    if (head_tile_read_req_fire)
                        tile_issue_count <= tile_issue_count + 5'd1;
                    if (max_req_fire)
                        tile_response_count <= tile_response_count + 5'd1;
                    if (max_rsp_fire) begin
                        if (max_rsp_tag != make_tile_tag(qkv_select, head_index,
                                batch_row_base, 1'b0,
                                pipeline_response_count[3:0]) ||
                            max_rsp_row_mask != batch_row_mask) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_QUANT;
                            state <= ERROR_DRAIN;
                        end else begin
                            for (integer max_row = 0; max_row < 8;
                                 max_row = max_row + 1)
                                if (max_rsp_row_mask[max_row])
                                    row_max_abs[max_row*16 +: 16] <= max_abs2(
                                        row_max_abs[max_row*16 +: 16],
                                        max_rsp_values[max_row*16 +: 16]);
                            pipeline_response_count <= pipeline_response_count + 5'd1;
                            if (pipeline_response_count == 5'd15)
                                state <= SCALE_REQUEST;
                        end
                    end
                end
                SCALE_REQUEST: if (quant_scale_req_fire)
                    state <= SCALE_RESPONSE;
                SCALE_RESPONSE: if (quant_scale_rsp_fire) begin
                    row_scale <= quant_scale_rsp_values_bf16;
                    half_index <= 1'b0;
                    tile_issue_count <= '0;
                    tile_response_count <= '0;
                    pipeline_response_count <= '0;
                    quantized_half_buffer <= '0;
                    state <= QUANTIZED_STREAM;
                end
                QUANTIZED_STREAM: begin
                    if (head_tile_read_req_fire)
                        tile_issue_count <= tile_issue_count + 5'd1;
                    if (quant_req_fire)
                        tile_response_count <= tile_response_count + 5'd1;
                    if (quant_rsp_fire) begin
                        if (quant_rsp_tag != make_tile_tag(qkv_select, head_index,
                                batch_row_base, half_index,
                                pipeline_response_count[3:0]) ||
                            quant_rsp_lane_mask != {
                                {8{batch_row_mask[7]}}, {8{batch_row_mask[6]}},
                                {8{batch_row_mask[5]}}, {8{batch_row_mask[4]}},
                                {8{batch_row_mask[3]}}, {8{batch_row_mask[2]}},
                                {8{batch_row_mask[1]}}, {8{batch_row_mask[0]}}}) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_QUANT;
                            state <= ERROR_DRAIN;
                        end else begin
                            for (integer quantized_row = 0; quantized_row < 8;
                                 quantized_row = quantized_row + 1)
                                quantized_half_buffer[quantized_row*512 +
                                    pipeline_response_count[2:0]*64 +: 64] <=
                                    quant_rsp_values[quantized_row*64 +: 64];
                            pipeline_response_count <= pipeline_response_count + 5'd1;
                            if (pipeline_response_count == 5'd7) begin
                                commit_row_index <= 3'd0;
                                state <= QUANTIZED_COMMIT;
                            end
                        end
                    end
                end
                QUANTIZED_COMMIT: begin
                    if (!batch_row_mask[commit_row_index]) begin
                        if (commit_row_index == 3'd7)
                            state <= ADVANCE_BATCH;
                        else
                            commit_row_index <= commit_row_index + 3'd1;
                    end else if (quantized_commit_fire) begin
                        if (commit_row_index == 3'd7)
                            state <= ADVANCE_BATCH;
                        else
                            commit_row_index <= commit_row_index + 3'd1;
                    end
                end
                ADVANCE_BATCH: begin
                    if (!half_index) begin
                        half_index <= 1'b1;
                        tile_issue_count <= '0;
                        tile_response_count <= '0;
                        pipeline_response_count <= '0;
                        quantized_half_buffer <= '0;
                        state <= QUANTIZED_STREAM;
                    end else if (batch_row_base + 6'd8 < 6'(MAX_ROWS)) begin
                        batch_row_base <= batch_row_base + 6'd8;
                        state <= BATCH_START;
                    end else begin
                        state <= LAYOUT_TO_PANEL;
                    end
                end
                LAYOUT_TO_PANEL: if (head_staging_layout_ready &&
                    (next_matmul_valid || !next_matmul_exists ||
                     matmul_prefetch_sent ||
                     (matmul_prefetch_valid && matmul_prefetch_ready))) begin
                    if (q_head_command) begin
                        state <= COMPLETE;
                    end else if (batch_group_active) begin
                        if (kv_batch_index + 3'd1 < batch_group_count) begin
                            kv_resume_state <= qkv_select == QKV_V ?
                                STAGING_READ_REQUEST : METADATA_REQUEST;
                        end else if (head_index + 6'd1 < head_count) begin
                            head_index <= head_index + 6'd1;
                            if (kv_group_activation_reuse) begin
                                q_group_head_base <= q_group_head_base + {32'd0, group_staging_head_stride};
                                kv_resume_state <= qkv_select == QKV_V ? STAGING_READ_REQUEST : METADATA_REQUEST;
                            end else kv_resume_state <= MATMUL_START;
                        end else if (qkv_select == QKV_V) begin
                            qkv_select <= QKV_K;
                            head_index <= 6'd0;
                            q_group_head_base <= group_staging_base;
                            kv_group_precompute <= kv_group_activation_reuse;
                            kv_resume_state <= MATMUL_START;
                        end else begin
                            kv_resume_state <= CACHE_MAPPING_START;
                        end
                        state <= KV_BATCH_SWITCH;
                    end else if (kv_pair_active) begin
                        if (!kv_second_batch) begin
                            kv_resume_state <= qkv_select == QKV_V ?
                                LAYOUT_TO_HEAD : ROPE_START;
                        end else if (head_index + 6'd1 < head_count) begin
                            head_index <= head_index + 6'd1;
                            kv_resume_state <= MATMUL_START;
                        end else if (qkv_select == QKV_V) begin
                            qkv_select <= QKV_K;
                            head_index <= 6'd0;
                            kv_resume_state <= MATMUL_START;
                        end else begin
                            kv_resume_state <= CACHE_MAPPING_START;
                        end
                        state <= KV_BATCH_SWITCH;
                    end else if (next_matmul_valid) begin
                        if (next_matmul_done) begin
                            qkv_select <= next_matmul_qkv_select;
                            head_index <= next_matmul_head;
                            next_matmul_valid <= 1'b0;
                            next_matmul_started <= 1'b0;
                            next_matmul_done <= 1'b0;
                            state <= next_matmul_qkv_select == QKV_V ?
                                LAYOUT_TO_HEAD : ROPE_START;
                        end else begin
                            state <= MATMUL_OVERLAP_WAIT;
                        end
                    end else if (head_index + 6'd1 < head_count) begin
                        head_index <= head_index + 6'd1;
                        state <= MATMUL_START;
                    end else if (qkv_select == QKV_V) begin
                        qkv_select <= QKV_K;
                        head_index <= 6'd0;
                        state <= MATMUL_START;
                    end else begin
                        state <= CACHE_MAPPING_START;
                    end
                end
                KV_BATCH_SWITCH: if (outstanding_cache_writes == 0 &&
                    commit_state == COMMIT_IDLE && !metadata_read_outstanding &&
                    !matmul_active && !rope_active && stage_copy_drained &&
                    head_tile_fifo_occupancy == 0 && max_outstanding == 0 &&
                    quant_values_outstanding == 0) begin
                    if (batch_group_active) begin
                        if (kv_batch_index + 3'd1 < batch_group_count) begin
                            kv_batch_index <= kv_batch_index + 3'd1;
                            kv_second_batch <= 1'b1;
                            row_count <= runtime_batch_group.row_count[
                                (kv_batch_index + 3'd1)*6 +: 6];
                            row_enable <= runtime_batch_group.row_enable[
                                (kv_batch_index + 3'd1)*48 +: 48];
                            kv_write_disable <= runtime_batch_group.kv_write_disable[
                                (kv_batch_index + 3'd1)*48 +: 48];
                            token_position <= runtime_batch_group.token_position[
                                (kv_batch_index + 3'd1)*528 +: 528];
                            staging_batch_base <= staging_batch_base +
                                {50'd0, row_count, 8'd0};
                            staging_stripe_address <= staging_batch_base +
                                {50'd0, row_count, 8'd0};
                        end else begin
                            kv_batch_index <= 3'd0;
                            kv_second_batch <= 1'b0;
                            row_count <= runtime_batch_group.row_count[0 +: 6];
                            row_enable <= runtime_batch_group.row_enable[0 +: 48];
                            kv_write_disable <= runtime_batch_group.kv_write_disable[0 +: 48];
                            token_position <= runtime_batch_group.token_position[0 +: 528];
                            staging_batch_base <= kv_group_activation_reuse ? q_group_head_base : group_staging_base;
                            staging_stripe_address <= kv_group_activation_reuse ? q_group_head_base : group_staging_base;
                        end
                        staging_stripe <= 5'd0;
                        staging_response_row <= 6'd0;
                        if (kv_resume_state == METADATA_REQUEST) begin
                            metadata_row <= 6'd0;
                            metadata_is_sin <= 1'b0;
                            group_metadata_refill <= 1'b1;
                        end
                    end else begin
                        row_count <= other_row_count;
                        other_row_count <= row_count;
                        row_enable <= other_row_enable;
                        other_row_enable <= row_enable;
                        kv_write_disable <= other_kv_write_disable;
                        other_kv_write_disable <= kv_write_disable;
                        token_position <= other_token_position;
                        other_token_position <= token_position;
                        kv_second_batch <= !kv_second_batch;
                        kv_batch_index <= kv_second_batch ? 3'd0 : 3'd1;
                    end
                    state <= KV_BATCH_LAYOUT_WAIT;
                end
                KV_BATCH_LAYOUT_WAIT: if (kv_pair_layout_ready)
                    state <= kv_resume_state;
                MATMUL_OVERLAP_WAIT: if (next_matmul_done) begin
                    qkv_select <= next_matmul_qkv_select;
                    head_index <= next_matmul_head;
                    next_matmul_valid <= 1'b0;
                    next_matmul_started <= 1'b0;
                    next_matmul_done <= 1'b0;
                    state <= next_matmul_qkv_select == QKV_V ?
                        LAYOUT_TO_HEAD : ROPE_START;
                end
                CACHE_MAPPING_START: if (cache_mapping_commit_valid &&
                                         cache_mapping_commit_ready) begin
                    if (cache_mapping_commit_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_CACHE_MAPPING;
                        state <= ERROR_DRAIN;
                    end else begin
                        state <= COMPLETE;
                    end
                end
                COMPLETE: if ((!q_head_command && done_ready) ||
                              (q_head_command && q_head_done_ready)) begin
                    q_head_command <= 1'b0;
                    q_head_matmul_done_seen <= 1'b0;
                    q_head_precompute_only <= 1'b0;
                    state <= IDLE;
                end
                ERROR_DRAIN: begin
                    if ((!matmul_active || matmul_abort_ack) &&
                        (!rope_active || rope_abort_ack) &&
                        (!quant_active || quant_abort_ack) &&
                        !head_read_outstanding && !metadata_read_outstanding &&
                        !head_tile_read_outstanding && max_outstanding == 0 &&
                        outstanding_cache_writes == 0) begin
                        next_matmul_valid <= 1'b0;
                        next_matmul_started <= 1'b0;
                        next_matmul_done <= 1'b0;
                        q_group_head_sequence_valid <= 1'b0;
                        state <= COMPLETE;
                    end
                end
                ABORT_DRAIN: begin
                    if (cache_abort_completion)
                        cache_abort_seen <= 1'b1;
                    if ((!matmul_active || matmul_abort_ack) &&
                        (!rope_active || rope_abort_ack) &&
                        (!quant_active || quant_abort_ack) &&
                        !head_read_outstanding && !metadata_read_outstanding &&
                        !head_tile_read_outstanding && max_outstanding == 0 &&
                        (outstanding_cache_writes == 0 || cache_abort_seen ||
                         cache_abort_completion)) begin
                        next_matmul_valid <= 1'b0;
                        next_matmul_started <= 1'b0;
                        next_matmul_done <= 1'b0;
                        q_group_head_sequence_valid <= 1'b0;
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase

            if (abort_request && state != IDLE && state != ABORT_DRAIN)
                begin
                    v_scale_table_valid <= 1'b0;
                    state <= ABORT_DRAIN;
                end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (MAX_ROWS < 1 || MAX_ROWS > 48)
            $error("QKV preparation requires MAX_ROWS in [1,48]");
        if (HEAD_DIM != 128)
            $error("QKV preparation requires HEAD_DIM=128");
        if (local_memory_layout_pkg::QKV_HEAD_STAGING_WORD_BASE +
                ((MAX_ROWS + 3) / 4) * 16 >
                local_memory_layout_pkg::QKV_PANEL_HALF_WORD_COUNT)
            $error("QKV head staging exceeds the W4 panel-free row range");
    end

    logic held_head_read;
    logic held_head_read_destination;
    logic [5:0] held_head_read_row;
    logic [3:0] held_head_read_word;
    logic held_scale_request;
    logic held_scale_static;
    logic [127:0] held_scale_static_values, held_scale_max_values;
    logic [7:0] held_scale_row_mask;
    logic held_quant_request;
    logic [1023:0] held_quant_values;
    logic [63:0] held_quant_mask;
    logic [TAG_WIDTH-1:0] held_quant_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            held_head_read <= 1'b0;
            held_head_read_destination <= 1'b0;
            held_head_read_row <= '0;
            held_head_read_word <= '0;
            held_scale_request <= 1'b0;
            held_scale_static <= 1'b0;
            held_scale_static_values <= '0;
            held_scale_max_values <= '0;
            held_scale_row_mask <= '0;
            held_quant_request <= 1'b0;
            held_quant_values <= '0;
            held_quant_mask <= '0;
            held_quant_tag <= '0;
        end else begin
            if (TRUSTED_START_CONFIGURATION && batch_group_active &&
                !q_head_command && state != IDLE && state != COMPLETE)
                assert ($stable(start_batch_group))
                    else $error("QKV grouped configuration changed before the command completed");
            assert (!next_matmul_started || next_matmul_valid)
                else $error("QKV next Matmul started without a saved descriptor");
            assert (!next_matmul_done ||
                    (next_matmul_valid && next_matmul_started))
                else $error("QKV next Matmul completed without an active descriptor");
            if (state == MATMUL_OVERLAP_WAIT)
                assert (next_matmul_valid)
                    else $error("QKV waited without a saved next Matmul descriptor");
            if (head_tile_memory_rsp_valid) begin
                assert (head_tile_memory_pending[1])
                    else $error("QKV head tile SRAM response had no accepted request");
                assert (head_tile_read_rsp_ready)
                    else $error("QKV head tile SRAM response met a stalled consumer");
            end
            if (head_tile_memory_pending[1])
                assert (head_tile_memory_rsp_valid)
                    else $error("QKV head tile accepted request had no SRAM response");
            assert (head_read_outstanding == 3'(stage_metadata_occupancy))
                else $error("QKV head read data and metadata reservations diverged");
            assert (head_read_outstanding <= 3'd3)
                else $error("QKV head read outstanding count exceeded three");
            assert (!head_read_rsp_valid || stage_metadata_output_valid)
                else $error("QKV head read response had no accepted metadata");
            if (head_read_req_valid && !head_read_req_ready) begin
                if (held_head_read)
                    assert (head_read_rope_destination == held_head_read_destination &&
                            head_read_physical_row == held_head_read_row &&
                            head_read_word == held_head_read_word)
                        else $error("QKV changed a stalled head read request");
                held_head_read <= 1'b1;
                held_head_read_destination <= head_read_rope_destination;
                held_head_read_row <= head_read_physical_row;
                held_head_read_word <= head_read_word;
            end else begin
                held_head_read <= 1'b0;
            end
            if (quant_scale_req_valid && !quant_scale_req_ready) begin
                if (held_scale_request)
                    assert (quant_scale_req_use_static_scale == held_scale_static &&
                            quant_scale_req_static_scales_bf16 == held_scale_static_values &&
                            quant_scale_req_row_max_abs == held_scale_max_values &&
                            quant_scale_req_row_mask == held_scale_row_mask)
                        else $error("QKV changed a stalled quantizer scale request");
                held_scale_request <= 1'b1;
                held_scale_static <= quant_scale_req_use_static_scale;
                held_scale_static_values <= quant_scale_req_static_scales_bf16;
                held_scale_max_values <= quant_scale_req_row_max_abs;
                held_scale_row_mask <= quant_scale_req_row_mask;
            end else begin
                held_scale_request <= 1'b0;
            end
            if (quant_req_valid && !quant_req_ready) begin
                if (held_quant_request)
                    assert (quant_req_values_bf16 == held_quant_values &&
                            quant_req_lane_mask == held_quant_mask &&
                            quant_req_tag == held_quant_tag)
                        else $error("QKV changed a stalled quantizer request");
                held_quant_request <= 1'b1;
                held_quant_values <= quant_req_values_bf16;
                held_quant_mask <= quant_req_lane_mask;
                held_quant_tag <= quant_req_tag;
            end else begin
                held_quant_request <= 1'b0;
            end
            if (cache_complete_fire)
                assert (outstanding_cache_writes != 0 ||
                        (quantized_commit_fire && quantized_commit_requires_response))
                    else $error("QKV received an unmatched cache write completion");
            if (commit_state == COMMIT_V_PAIR_WAIT && quantized_commit_fire)
                assert (qkv_select == QKV_V && head_index == commit_head &&
                        half_index == commit_half &&
                        quantized_commit_physical_row == commit_row + 6'd1 &&
                        quantized_commit_token_position ==
                            commit_token_position + 11'd1)
                    else $error("QKV accepted mismatched metadata for a paired V row");
            if (commit_state == COMMIT_V_PAIR_ADDRESS &&
                current_write_valid && current_write_ready &&
                current_write_accepted)
                assert (current_v_write_address == commit_v_address + 64'd8)
                    else $error("QKV paired V rows did not map to adjacent addresses");
            if (commit_state == COMMIT_COMPLETE &&
                commit_qkv_select != QKV_Q) begin
                assert (commit_completion_count != 2'd0)
                    else $error("QKV cache completion count underflowed");
                assert (!commit_v_pair ||
                        outstanding_cache_writes >= commit_completion_count)
                    else $error("QKV paired completion exceeded accepted logical writes");
            end
            assert (commit_dma_outstanding <= 4'd8)
                else $error("QKV cache DMA outstanding count exceeded eight");
            if (cache_writer_done)
                assert (commit_dma_outstanding != 0)
                    else $error("QKV received a cache DMA completion without an accepted write");
            if (rope_done_pulse)
                assert (state == ROPE_WAIT)
                    else $error("QKV received a RoPE completion pulse outside ROPE_WAIT");
            if (cache_config_done_pulse)
                assert (state == CACHE_CONFIG_WAIT)
                    else $error("QKV received a cache configuration pulse outside CACHE_CONFIG_WAIT");
            if (done_valid && !error)
                assert (outstanding_cache_writes == 0 && !matmul_active &&
                        !rope_active && !quant_active && !head_read_outstanding &&
                        stage_metadata_occupancy == 3'd0)
                    else $error("QKV completed with pending compute or memory transactions");
            if ($past(done_valid && !done_ready))
                assert (done_valid && $stable({error, error_id}))
                    else $error("QKV changed a stalled completion response");
        end
    end
`endif
endmodule

`default_nettype wire
