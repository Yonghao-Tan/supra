`default_nettype none
module execution_controller #(
    parameter integer FFN_FEATURES = 12288
) (
    input  logic          clk,
    input  logic          rst,

    input  logic          launch_valid,
    output logic          launch_ready,
    input  logic [63:0]   launch_config_addr,
    input  logic [31:0]   launch_id,

    output logic          completion_valid,
    input  logic          completion_ready,
    output logic [31:0]   completion_id,
    output logic          completion_error,
    output logic [15:0]   completion_error_id,
    output logic [2:0]    memory_stage,
    output logic          block_schedule_active,

    output logic          forward_postprocess_start_valid,
    input  logic          forward_postprocess_start_ready,
    output logic [2559:0] forward_postprocess_configuration_bits,
    output logic [63:0]   refresh_configuration_address,
    output logic [63:0]   joint_configuration_address,
    output logic          refresh_loaded_token_valid,
    output logic [10:0]   refresh_loaded_token_position,
    output logic          refresh_loaded_token_kv_write,
    output logic          refresh_prepare,
    output logic          refresh_select,
    output logic          refresh_relation_only,
    output logic          refresh_relation_l31,
    output logic          refresh_closeout,
    input  logic [31:0]   refresh_relation_layer_mask,
    input  logic          forward_postprocess_done_valid,
    output logic          forward_postprocess_done_ready,
    input  logic          forward_postprocess_done_error,
    input  logic          forward_postprocess_block_complete,
    input  logic [7:0]    forward_postprocess_done_error_id,

    output logic [2:0]    dma_read_request_valid,
    input  logic [2:0]    dma_read_request_ready,
    output hardware_types_pkg::dma_read_request_t dma_read_request [0:2],
    input  logic [2:0]    dma_read_data_valid,
    output logic [2:0]    dma_read_data_ready,
    input  hardware_types_pkg::dma_read_beat_t dma_read_data [0:2],
    input  hardware_types_pkg::dma_completion_t dma_read_completion [0:2],

    output logic          hidden_start_valid,
    input  logic          hidden_start_ready,
    output logic          hidden_start_write,
    output logic          hidden_start_residual,
    output logic [63:0]   hidden_start_address,
    output logic [31:0]   hidden_start_bytes,
    output logic [5:0]    hidden_start_active_rows,
    output logic [287:0]  hidden_start_row_permutation,
    output logic          hidden_start_direct_row_index_enable,
    output logic [527:0]  hidden_start_direct_row_index,
    output logic          hidden_start_source_descriptor_enable,
    output logic [815:0]  hidden_start_source_index,
    output logic [47:0]   hidden_start_source_embedding,
    output logic [63:0]   hidden_start_embedding_base,
    output logic [63:0]   hidden_start_embedding_limit,
    output logic [63:0]   hidden_start_limit,
    input  logic          hidden_done_valid,
    output logic          hidden_done_ready,
    input  logic          hidden_error,

    output logic          command_valid,
    input  logic          command_ready,
    output logic [5:0]    command_layer,
    output logic [3:0]    layer_operator_index,
    output logic          command_cache_release,
    output logic          command_qkv_residual_spill,
    output logic          command_rms_qkv_prefetch_q,
    output logic          command_q_head_preprocess,
    input  logic          command_done_valid,
    output logic          command_done_ready,
    input  logic          command_done_error,
    input  logic [15:0]   command_done_error_id,

    output logic [5:0]    row_active_rows,
    output logic [5:0]    row_residual_storage_rows,
    output logic [11:0]   row_sequence_length,
    output hardware_types_pkg::matmul_row_config_t matmul_row_config,
    output hardware_types_pkg::hidden_row_config_t hidden_row_config,
    output hardware_types_pkg::attention_row_config_t attention_row_config,
    output logic          layer_entry_valid,
    output logic [4:0]    layer_entry_index,
    output hardware_types_pkg::matmul_weight_config_t matmul_weight_config,
    output hardware_types_pkg::rmsnorm_layer_config_t rmsnorm_layer_config,
    output hardware_types_pkg::qkv_layer_config_t qkv_layer_config,
    output hardware_types_pkg::attention_cache_config_t attention_cache_config,
    output hardware_types_pkg::attention_context_config_t attention_context_config,
    output hardware_types_pkg::ffn_workspace_config_t ffn_workspace_config,
    output hardware_types_pkg::ffn_batch_pair_config_t ffn_batch_pair_config,
    output logic attention_pair_enable,
    output logic attention_normalized_capture,
    output logic kv_pair_enable,
    output hardware_types_pkg::attention_row_config_t kv_pair_second_config,
    output hardware_types_pkg::attention_batch_group_config_t kv_batch_group_config,
    output logic [5:0] attention_pair_second_rows,
    output logic attention_third_batch_enable,
    output logic [5:0] attention_third_rows,
    output logic attention_output_subset_enable,
    output logic [5:0] attention_output_token_count,
    input logic attention_second_batch,
    input logic [1:0] attention_batch_index
);
    import execution_config_pkg::*;

    localparam logic [15:0] ERROR_HIDDEN_INPUT = 16'h0004;
    localparam logic [15:0] ERROR_HIDDEN_OUTPUT = 16'h0005;
    localparam logic [15:0] ERROR_FFN_PAIR = 16'h0006;
    localparam logic [15:0] ERROR_QKVO_GROUP = 16'h0007;
    localparam integer HIDDEN_ROW_BYTES = 4096 * 2;
    localparam integer DMA_READ_CONFIGURATION = 0;
    localparam integer DMA_READ_METADATA = 1;
    localparam integer DMA_READ_LAYER = 2;

    assign dma_read_request[DMA_READ_CONFIGURATION].tag = 8'h01;
    assign dma_read_request[DMA_READ_CONFIGURATION].wide_data = 1'b0;
    assign dma_read_request[DMA_READ_METADATA].tag = 8'h02;
    assign dma_read_request[DMA_READ_METADATA].wide_data = 1'b0;
    assign dma_read_request[DMA_READ_LAYER].tag = 8'h03;
    assign dma_read_request[DMA_READ_LAYER].wide_data = 1'b0;

    typedef enum logic [4:0] {
        IDLE,
        CONFIG_START,
        CONFIG_WAIT,
        METADATA_START,
        METADATA_WAIT,
        HIDDEN_INPUT_START,
        HIDDEN_INPUT_WAIT,
        SCHEDULE_START,
        SCHEDULE_WAIT,
        HIDDEN_OUTPUT_START,
        HIDDEN_OUTPUT_WAIT,
        FORWARD_POSTPROCESS_START,
        FORWARD_POSTPROCESS_WAIT,
        COMPLETE,
        HIDDEN_PRESERVE_START,
        HIDDEN_PRESERVE_WAIT,
        HIDDEN_RESIDUAL_START,
        HIDDEN_RESIDUAL_WAIT,
        REFRESH_PREPARE_START,
        REFRESH_PREPARE_WAIT,
        REFRESH_START,
        REFRESH_WAIT,
        REFRESH_LAYER_START,
        REFRESH_LAYER_WAIT,
        SKIP_HIDDEN_OUTPUT,
        METADATA_RELEASE
    } state_t;

    localparam logic [2:0] MEMORY_STAGE_NONE = 3'd0;
    localparam logic [2:0] MEMORY_STAGE_CONFIGURATION = 3'd1;
    localparam logic [2:0] MEMORY_STAGE_METADATA = 3'd2;
    localparam logic [2:0] MEMORY_STAGE_HIDDEN_READ = 3'd3;
    localparam logic [2:0] MEMORY_STAGE_HIDDEN_WRITE = 3'd4;
    localparam logic [2:0] MEMORY_STAGE_FORWARD_POSTPROCESS = 3'd5;

    state_t state;
    typedef enum logic [4:0] {
        LAYER_PHASE_KV,
        LAYER_PHASE_ATTENTION,
        LAYER_PHASE_FFN,
        LAYER_PHASE_FFN_GROUP_CAPTURE,
        LAYER_PHASE_FFN_GROUP_MATMUL,
        LAYER_PHASE_FFN_GROUP_DOWN,
        LAYER_PHASE_FFN_GROUP_R4,
        LAYER_PHASE_FFN_GROUP_REFILL,
        LAYER_PHASE_O_CAPTURE,
        LAYER_PHASE_O_MATMUL,
        LAYER_PHASE_O_REFILL,
        LAYER_PHASE_ATTENTION_CAPTURE,
        LAYER_PHASE_ATTENTION_PAIR,
        LAYER_PHASE_O_SINGLE,
        LAYER_PHASE_KV_CAPTURE,
        LAYER_PHASE_KV_PAIR,
        LAYER_PHASE_ATTENTION_GROUP_SCAN,
        LAYER_PHASE_ATTENTION_GROUP
    } layer_phase_t;
    layer_phase_t layer_phase;
    logic [5:0] v4_layer_index;
    logic [63:0] saved_config_address;
    logic loader_start_valid;
    logic loader_start_ready;
    logic loader_done_valid;
    logic loader_done_ready;
    logic loader_done_error;
    logic [15:0] loader_done_error_id;
    logic scheduler_start_valid;
    logic scheduler_start_ready;
    logic scheduler_command_valid;
    logic refresh_layer_failed;
    logic scheduler_done_valid;
    logic scheduler_done_ready;
    logic scheduler_done_error;
    logic [15:0] scheduler_done_error_id;
    logic [5:0] configuration_layer_count;
    logic [5:0] configuration_start_layer;
    logic [5:0] configuration_layer_end;
    logic [5:0] scheduler_layer_base;
    logic [5:0] scheduler_layer_count;
    logic [10:0] scheduler_step_mask;
    logic [10:0] scheduler_cache_release_mask;
    logic [10:0] scheduler_qkv_residual_spill_mask;
    logic [10:0] scheduler_rms_qkv_prefetch_q_mask;
    logic [10:0] scheduler_q_head_preprocess_mask;
    logic metadata_start_valid;
    logic metadata_loaded_token_valid;
    logic metadata_start_ready;
    logic metadata_done_valid;
    logic metadata_done_ready;
    logic metadata_done_error;
    logic [15:0] metadata_done_error_id;
    logic row_configuration_release;
    logic [2559:0] configuration_bits;
    logic row_configuration_valid;
    logic [5:0] row_segment_mode;
    logic [17:0] row_segment_base;
    logic [17:0] row_segment_count;
    logic [287:0] row_physical_to_local;
    logic [527:0] row_token_position;
    logic [47:0] row_matmul_valid;
    logic [3:0] row_operator_has_w8;
    logic [11:0] row_total_rows;
    logic [5:0] row_token_batch_index;
    logic row_last_round;
    logic configuration_forward_postprocess_enable;
    logic configuration_refresh_enable;
    logic [64:0] refresh_address_sum;
    logic [64:0] joint_address_sum;
    logic configuration_lm_head_only;
    logic layer_load_pending;
    logic layer_load_ready;
    logic layer_done_valid;
    logic layer_done_error;
    logic [15:0] layer_done_error_id;
    logic layer_load_failed;
    logic [63:0] v4_output_hidden_address;
    logic [11:0] v4_output_row_base;
    logic [11:0] group_output_row_base [0:5];
    logic [5:0] group_active_rows [0:5];
    logic [5:0] group_token_batch_index [0:5];
    logic group_last_round [0:5];
    logic [287:0] group_physical_to_local [0:5];
    logic [527:0] group_token_position [0:5];
    logic [47:0] group_kv_write_disable [0:5];
    hardware_types_pkg::matmul_row_config_t group_row_config [0:5];
    hardware_types_pkg::matmul_row_config_t live_matmul_row_config;
    logic v4_group_phase, v4_use_group_rows;
    logic [2:0] group_capture_index, group_down_index;
    logic [2:0] group_selected_index;
    logic [2:0] group_batch_count, group_max_batch_count;
    logic [5:0] group_compute_group_count;
    logic [7:0] ffn_pair_first_batch_plus1;
    logic [7:0] ffn_round_offset;
    logic ffn_pair_repeat, ffn_four_batch, ffn_six_batch, ffn_fused_product;
    logic ffn_down_pair, group_down_pair_active, group_prepare_second;
    logic attention_output_pair;
    logic attention_batch_pair;
    logic kv_batch_pair;
    logic qkvo_group;
    logic [1:0] attention_group_command_batches;
    logic normalized_hidden_write;
    logic o_context_address_valid;
    logic [64:0] o_context_end;
    logic ffn_pair_start;
    logic ffn_six_batch_boundary;
    logic [63:0] group_workspace_offset;
    logic v4_hidden_source_descriptor;
    logic v4_multiple_rounds;
    logic v4_preserve_input;
    logic v4_read_preserved_input;
    logic hidden_preserve_write;
    logic [63:0] v4_preserved_hidden_base, v4_preserved_hidden_limit;
    logic [63:0] v4_round_residual_base;
    logic [63:0] v4_residual_spill_base, v4_residual_spill_limit;
    logic [527:0] v4_output_direct_row_index;
    logic v4_last_layer;
    hardware_types_pkg::ffn_workspace_config_t loaded_ffn_workspace_config;
    hardware_types_pkg::attention_context_config_t loaded_attention_context_config;

    logic v4_metadata_start_ready;
    logic v4_metadata_request_valid;
    logic v4_metadata_response_ready;
    logic v4_metadata_done_valid;
    logic v4_metadata_done_error;
    logic [15:0] v4_metadata_done_error_id;
    logic v4_row_configuration_valid;
    logic [11:0] v4_row_total_rows;
    logic [5:0] v4_row_token_batch_index;
    logic v4_row_last_round;
    logic [5:0] v4_row_active_rows;
    logic [5:0] v4_compute_group_count;
    logic [11:0] v4_row_sequence_length;
    logic [287:0] v4_row_physical_to_local;
    logic [527:0] v4_row_token_position;
    logic [47:0] v4_row_matmul_valid;
    logic [815:0] v4_row_source_index;
    logic [47:0] v4_row_embedding_source;
    logic [191:0] v4_row_precision_bits;
    logic [383:0] v4_row_query_group;
    logic [383:0] v4_row_cache_group;
    logic [287:0] v4_row_compute_group;
    logic [143:0] v4_row_pe_slot;
    logic [95:0] v4_row_phase_mask;
    logic [47:0] v4_row_kv_write_disable;
    logic [63:0] v4_metadata_request_address;
    logic [31:0] v4_metadata_request_bytes;
    logic [3:0] v4_operator_has_a8;
    logic v4_output_subset_enable;
    logic separate_l31_metadata, l31_metadata_active;
    logic v4_qkvo_group_end;
    logic [5:0] v4_output_token_count, v4_output_compute_groups;
    logic v4_output_subset_active, v4_subset_compute;
    logic [47:0] v4_output_token_mask;

    assign separate_l31_metadata = configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+13];
    assign l31_metadata_active = separate_l31_metadata && v4_layer_index == 6'd31;

    assign launch_ready = state == IDLE;
    assign loader_start_valid = state == CONFIG_START;
    assign loader_done_ready = state == CONFIG_WAIT;
    assign metadata_start_valid = state == METADATA_START;
    assign metadata_done_ready = state == METADATA_WAIT;
    assign hidden_start_valid = state == HIDDEN_INPUT_START ||
        state == HIDDEN_OUTPUT_START || state == HIDDEN_PRESERVE_START ||
        state == HIDDEN_RESIDUAL_START;
    assign hidden_preserve_write = state == HIDDEN_PRESERVE_START;
    assign hidden_start_residual = state == HIDDEN_RESIDUAL_START;
    assign hidden_start_write = state == HIDDEN_OUTPUT_START ||
        hidden_preserve_write || hidden_start_residual;
    assign v4_multiple_rounds = row_total_rows != {6'd0, v4_row_active_rows} ||
        v4_output_subset_enable;
    assign v4_output_subset_active = v4_output_subset_enable && v4_layer_index == 6'd31;
    assign attention_output_subset_enable = v4_output_subset_active;
    assign attention_output_token_count = v4_output_token_count;
    assign v4_output_token_mask = (48'd1 << v4_output_token_count) - 48'd1;
    assign v4_subset_compute = v4_output_subset_active &&
        (layer_phase == LAYER_PHASE_FFN ||
         (layer_phase == LAYER_PHASE_ATTENTION &&
          ((state == SCHEDULE_WAIT && layer_operator_index == layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL) ||
           state == HIDDEN_OUTPUT_START || state == HIDDEN_OUTPUT_WAIT || state == SKIP_HIDDEN_OUTPUT)));
    assign row_residual_storage_rows =
        v4_output_subset_active && layer_phase == LAYER_PHASE_ATTENTION &&
        state == SCHEDULE_WAIT &&
        layer_operator_index == layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL ?
            v4_row_active_rows : 6'd0;
    assign ffn_pair_first_batch_plus1 = l31_metadata_active ? 8'd0 : configuration_bits[EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET*8 +: 8];
    assign ffn_pair_repeat = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+4];
    assign ffn_four_batch = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+5];
    assign ffn_six_batch = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+6];
    assign ffn_fused_product = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+7];
    assign ffn_down_pair = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+8];
    assign attention_output_pair = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+9];
    assign attention_batch_pair = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+10];
    assign kv_batch_pair = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+11];
    assign qkvo_group = !l31_metadata_active && configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+12];
    assign kv_pair_enable = layer_phase == LAYER_PHASE_KV_PAIR;
    assign kv_pair_second_config.row_enable = group_row_config[1].row_enable;
    assign kv_pair_second_config.token_position = group_token_position[1];
    assign kv_pair_second_config.token_batch_index = group_token_batch_index[1];
    assign kv_pair_second_config.kv_write_disable = group_kv_write_disable[1];
    logic [64:0] kv_all_heads_staging_base, kv_all_heads_staging_end;
    logic [31:0] kv_group_head_stride;
    assign kv_group_head_stride =
        32'((group_output_row_base[group_batch_count - 3'd1] +
             {6'd0, group_active_rows[group_batch_count - 3'd1]} -
             group_output_row_base[0]) << 8);
    // Normalized rows remain live across groups. Reserve staging after all of
    // them; 32 heads * 128 BF16 elements = 8192 bytes per group row.
    assign kv_all_heads_staging_base = {1'b0, loaded_ffn_workspace_config.gate_up_base} +
        {40'd0, row_total_rows, 13'd0};
    assign kv_all_heads_staging_end = kv_all_heads_staging_base +
        {28'd0, kv_group_head_stride, 5'd0};
    always_comb begin
        kv_batch_group_config = '0;
        kv_batch_group_config.enable = qkvo_group &&
            (kv_pair_enable || layer_phase == LAYER_PHASE_ATTENTION_GROUP);
        kv_batch_group_config.active_batch_index =
            layer_phase == LAYER_PHASE_ATTENTION_GROUP ?
                group_down_index + {1'b0, attention_batch_index} :
                group_down_index;
        kv_batch_group_config.batch_count = group_batch_count;
        for (integer batch = 0; batch < 6; batch++) begin
            kv_batch_group_config.row_count[batch*6 +: 6] =
                group_active_rows[batch];
            kv_batch_group_config.row_enable[batch*48 +: 48] =
                group_row_config[batch].row_enable;
            kv_batch_group_config.kv_write_disable[batch*48 +: 48] =
                group_kv_write_disable[batch];
            kv_batch_group_config.token_position[batch*528 +: 528] =
                group_token_position[batch];
        end
        kv_batch_group_config.staging_base =
            loaded_ffn_workspace_config.gate_up_base +
            (layer_phase == LAYER_PHASE_ATTENTION_GROUP ?
                {39'd0, row_total_rows, 13'd0} :
                {39'd0,
                 group_output_row_base[group_batch_count - 3'd1] +
                    {6'd0, group_active_rows[group_batch_count - 3'd1]},
                 13'd0});
        kv_batch_group_config.reuse_group_activation = qkvo_group && kv_pair_enable &&
            group_compute_group_count > 6'd6 && !kv_all_heads_staging_base[64] &&
            !kv_all_heads_staging_end[64] &&
            kv_all_heads_staging_end <= {1'b0, loaded_ffn_workspace_config.gate_up_limit};
        if (kv_batch_group_config.reuse_group_activation)
            kv_batch_group_config.staging_base = kv_all_heads_staging_base[63:0];
        kv_batch_group_config.staging_limit = loaded_ffn_workspace_config.gate_up_limit;
        kv_batch_group_config.staging_head_stride = kv_group_head_stride;
    end
    always_comb begin
        attention_group_command_batches = 2'd1;
        attention_pair_second_rows = group_active_rows[1];
        attention_third_rows = 6'd0;
        if (layer_phase == LAYER_PHASE_ATTENTION_GROUP &&
            row_sequence_length != 0 && row_sequence_length <= 12'd1296) begin
            if (group_down_index + 3'd3 <= group_batch_count) begin
                attention_group_command_batches = 2'd3;
                attention_pair_second_rows =
                    group_active_rows[group_down_index + 3'd1];
                attention_third_rows =
                    group_active_rows[group_down_index + 3'd2];
            end else if (group_down_index + 3'd2 <= group_batch_count) begin
                attention_group_command_batches = 2'd2;
                attention_pair_second_rows =
                    group_active_rows[group_down_index + 3'd1];
            end
        end
    end
    assign attention_pair_enable = layer_phase == LAYER_PHASE_ATTENTION_PAIR ||
        layer_phase == LAYER_PHASE_O_SINGLE ||
        (layer_phase == LAYER_PHASE_ATTENTION_GROUP &&
         attention_group_command_batches >= 2'd2);
    assign attention_third_batch_enable =
        layer_phase == LAYER_PHASE_ATTENTION_GROUP &&
        attention_group_command_batches == 2'd3;
    assign attention_normalized_capture = layer_phase == LAYER_PHASE_ATTENTION_CAPTURE ||
        layer_phase == LAYER_PHASE_KV_CAPTURE;
    assign group_max_batch_count = ffn_six_batch ? 3'd6 :
        ffn_four_batch ? 3'd4 : 3'd2;
    assign ffn_round_offset = {2'd0, row_token_batch_index} + 8'd1 -
        ffn_pair_first_batch_plus1;
    always_comb begin
        case (ffn_round_offset)
            8'd0, 8'd6, 8'd12, 8'd18, 8'd24, 8'd30,
            8'd36, 8'd42, 8'd48, 8'd54, 8'd60:
                ffn_six_batch_boundary = 1'b1;
            default: ffn_six_batch_boundary = 1'b0;
        endcase
    end
    assign ffn_pair_start = ffn_pair_first_batch_plus1 != 0 && !row_last_round &&
        (ffn_pair_repeat ?
            ({2'd0,row_token_batch_index} + 8'd1 >= ffn_pair_first_batch_plus1 &&
             (ffn_six_batch ? ffn_six_batch_boundary :
              ffn_four_batch ? ffn_round_offset[1:0] == 2'd0 :
              ffn_round_offset[0] == 1'b0)) :
            {2'd0,row_token_batch_index} + 8'd1 == ffn_pair_first_batch_plus1);
    assign v4_group_phase = layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE ||
        layer_phase == LAYER_PHASE_FFN_GROUP_MATMUL ||
        layer_phase == LAYER_PHASE_FFN_GROUP_DOWN ||
        layer_phase == LAYER_PHASE_FFN_GROUP_R4 ||
        layer_phase == LAYER_PHASE_FFN_GROUP_REFILL;
    assign v4_use_group_rows = layer_phase == LAYER_PHASE_FFN_GROUP_MATMUL ||
        layer_phase == LAYER_PHASE_FFN_GROUP_DOWN ||
        layer_phase == LAYER_PHASE_FFN_GROUP_R4 ||
        layer_phase == LAYER_PHASE_FFN_GROUP_REFILL ||
        layer_phase == LAYER_PHASE_O_MATMUL ||
        layer_phase == LAYER_PHASE_O_REFILL ||
        layer_phase == LAYER_PHASE_ATTENTION_GROUP ||
        attention_pair_enable || kv_pair_enable;
    assign group_selected_index = layer_phase == LAYER_PHASE_ATTENTION_PAIR ?
        {2'd0, attention_second_batch} :
        layer_phase == LAYER_PHASE_ATTENTION_GROUP ?
            group_down_index + {1'b0, attention_batch_index} :
        (layer_phase == LAYER_PHASE_FFN_GROUP_MATMUL || kv_pair_enable) ?
        3'd0 : group_down_index + {2'd0, group_prepare_second};
    assign v4_preserve_input = v4_multiple_rounds &&
        (v4_layer_index == configuration_start_layer || l31_metadata_active);
    assign v4_read_preserved_input = v4_preserve_input &&
        (layer_phase == LAYER_PHASE_ATTENTION ||
         layer_phase == LAYER_PHASE_O_CAPTURE ||
         layer_phase == LAYER_PHASE_ATTENTION_CAPTURE);
    assign v4_preserved_hidden_base = configuration_bits[
        EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET*8 +: 64];
    assign v4_preserved_hidden_limit = configuration_bits[
        EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET*8 +: 64];
    assign v4_round_residual_base = v4_preserved_hidden_base +
        {39'd0, row_total_rows, 13'd0} +
        {39'd0, v4_output_row_base, 13'd0};
    assign v4_residual_spill_base = v4_multiple_rounds ?
        v4_round_residual_base : loaded_ffn_workspace_config.residual_base;
    assign v4_residual_spill_limit = v4_multiple_rounds ?
        v4_preserved_hidden_limit : loaded_ffn_workspace_config.residual_limit;
    assign v4_output_hidden_address =
        configuration_bits[EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET*8 +: 64] +
        {39'd0, v4_output_row_base, 13'd0};
    assign hidden_start_address = hidden_start_residual ?
        v4_residual_spill_base :
        hidden_preserve_write ?
        v4_preserved_hidden_base + {39'd0, v4_output_row_base, 13'd0} :
        (normalized_hidden_write && layer_phase == LAYER_PHASE_KV_CAPTURE) ?
        loaded_ffn_workspace_config.gate_up_base +
            {39'd0, v4_output_row_base, 13'd0} :
        v4_read_preserved_input && !hidden_start_write ? v4_preserved_hidden_base :
        normalized_hidden_write ?
        v4_preserved_hidden_base + {39'd0,v4_output_row_base,13'd0} :
        hidden_start_write ?
        v4_output_hidden_address :
        configuration_bits[EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET*8 +: 64];
    assign hidden_start_limit = normalized_hidden_write && layer_phase == LAYER_PHASE_KV_CAPTURE ?
        loaded_ffn_workspace_config.gate_up_limit : hidden_start_residual ?
        v4_residual_spill_limit : hidden_preserve_write ||
        normalized_hidden_write ||
        (v4_read_preserved_input && !hidden_start_write) ? v4_preserved_hidden_limit :
        configuration_bits[EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET*8 +: 64];
    assign hidden_start_bytes = 32'(row_active_rows) * HIDDEN_ROW_BYTES;
    assign normalized_hidden_write = (state == HIDDEN_OUTPUT_START || state == HIDDEN_OUTPUT_WAIT) &&
        (layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE ||
         layer_phase == LAYER_PHASE_ATTENTION_CAPTURE || layer_phase == LAYER_PHASE_KV_CAPTURE);
    assign hidden_start_active_rows = row_active_rows;
    assign hidden_done_ready = state == HIDDEN_INPUT_WAIT ||
        state == HIDDEN_OUTPUT_WAIT || state == HIDDEN_PRESERVE_WAIT ||
        state == HIDDEN_RESIDUAL_WAIT;
    always_comb begin
        hidden_start_row_permutation = row_physical_to_local;
        // Hidden transfer builds its inverse map after accepting START.
        if ((layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE ||
             layer_phase == LAYER_PHASE_ATTENTION_CAPTURE || layer_phase == LAYER_PHASE_KV_CAPTURE) &&
            (state == HIDDEN_OUTPUT_START || state == HIDDEN_OUTPUT_WAIT))
            for (integer row = 0; row < 48; row++)
                hidden_start_row_permutation[row*6 +: 6] = 6'(row);
    end
    assign hidden_start_direct_row_index_enable =
        !hidden_start_write;
    assign hidden_start_direct_row_index = v4_output_direct_row_index;
    assign v4_hidden_source_descriptor = !hidden_start_write && (v4_layer_index == configuration_start_layer || l31_metadata_active) &&
        (layer_phase == LAYER_PHASE_KV || layer_phase == LAYER_PHASE_KV_CAPTURE || layer_phase == LAYER_PHASE_ATTENTION ||
         layer_phase == LAYER_PHASE_O_CAPTURE ||
         layer_phase == LAYER_PHASE_ATTENTION_CAPTURE) &&
        !v4_read_preserved_input;
    assign hidden_start_source_descriptor_enable =
        v4_hidden_source_descriptor;
    assign hidden_start_source_index = v4_row_source_index;
    assign hidden_start_source_embedding = v4_row_embedding_source;
    assign hidden_start_embedding_base = configuration_bits[
        EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET*8 +: 64];
    assign hidden_start_embedding_limit = configuration_bits[
        EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET*8 +: 64];
    assign o_context_end = {1'b0, loaded_attention_context_config.context_base} +
        ((qkvo_group && (layer_phase == LAYER_PHASE_ATTENTION_GROUP ||
                         layer_phase == LAYER_PHASE_O_MATMUL ||
                         layer_phase == LAYER_PHASE_O_REFILL)) ?
            ({53'd0,
              group_output_row_base[group_batch_count - 3'd1] +
                {6'd0, group_active_rows[group_batch_count - 3'd1]} -
                group_output_row_base[0]} << 13) : 65'd786432);
    assign o_context_address_valid = !o_context_end[64] &&
        loaded_attention_context_config.context_row_stride_bytes == 32'd8192 &&
        loaded_attention_context_config.context_base >= configuration_bits[
            EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET*8 +: 64] &&
        o_context_end <= {1'b0, configuration_bits[
            EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET*8 +: 64]};
    assign scheduler_start_valid = state == SCHEDULE_START &&
        ((layer_phase != LAYER_PHASE_O_CAPTURE &&
          layer_phase != LAYER_PHASE_ATTENTION_PAIR) || o_context_address_valid);
    assign command_valid = scheduler_command_valid && state == SCHEDULE_WAIT;
    assign refresh_layer_failed = state == REFRESH_LAYER_WAIT &&
        forward_postprocess_done_valid && forward_postprocess_done_error;
    assign layer_load_failed = layer_done_valid && layer_done_error;
    assign scheduler_done_ready = state == SCHEDULE_WAIT;
    assign block_schedule_active = state == SCHEDULE_START ||
        state == SCHEDULE_WAIT;
    assign forward_postprocess_start_valid = (state == FORWARD_POSTPROCESS_START || state == REFRESH_PREPARE_START || state == REFRESH_START || state == REFRESH_LAYER_START) &&
        !(refresh_select && (refresh_address_sum[64] || joint_address_sum[64]));
    assign forward_postprocess_configuration_bits = configuration_bits;
    assign refresh_address_sum = {1'b0, saved_config_address} +
        {49'd0, configuration_bits[EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET*8 +: 16]};
    assign refresh_configuration_address = refresh_address_sum[63:0];
    assign joint_address_sum = configuration_bits[EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET*8 +: 16] == 0 ? 65'd0 :
        {1'b0, saved_config_address} + {49'd0, configuration_bits[EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET*8 +: 16]};
    assign joint_configuration_address = joint_address_sum[63:0];
    assign refresh_prepare = state == REFRESH_PREPARE_START || state == REFRESH_PREPARE_WAIT;
    assign refresh_relation_only = state == REFRESH_LAYER_START || state == REFRESH_LAYER_WAIT;
    assign refresh_select = refresh_prepare || refresh_relation_only || state == REFRESH_START || state == REFRESH_WAIT;
    assign forward_postprocess_done_ready = state == FORWARD_POSTPROCESS_WAIT || state == REFRESH_PREPARE_WAIT || state == REFRESH_WAIT || state == REFRESH_LAYER_WAIT;
    assign completion_valid = state == COMPLETE;
    always_comb begin
        memory_stage = MEMORY_STAGE_NONE;
        case (state)
            CONFIG_START, CONFIG_WAIT:
                memory_stage = MEMORY_STAGE_CONFIGURATION;
            METADATA_START, METADATA_WAIT, METADATA_RELEASE:
                memory_stage = MEMORY_STAGE_METADATA;
            HIDDEN_INPUT_START, HIDDEN_INPUT_WAIT:
                memory_stage = MEMORY_STAGE_HIDDEN_READ;
            HIDDEN_OUTPUT_START, HIDDEN_OUTPUT_WAIT,
            HIDDEN_PRESERVE_START, HIDDEN_PRESERVE_WAIT,
            HIDDEN_RESIDUAL_START, HIDDEN_RESIDUAL_WAIT:
                memory_stage = MEMORY_STAGE_HIDDEN_WRITE;
            FORWARD_POSTPROCESS_START, FORWARD_POSTPROCESS_WAIT, REFRESH_PREPARE_START, REFRESH_PREPARE_WAIT, REFRESH_START, REFRESH_WAIT,
            REFRESH_LAYER_START, REFRESH_LAYER_WAIT:
                memory_stage = MEMORY_STAGE_FORWARD_POSTPROCESS;
            default: memory_stage = MEMORY_STAGE_NONE;
        endcase
    end
    assign v4_last_layer = v4_layer_index + 1'b1 >=
        configuration_layer_end;
    always_comb begin
        v4_output_direct_row_index = '0;
        for (integer physical_row = 0; physical_row < 48;
             physical_row = physical_row + 1) begin
            v4_output_direct_row_index[physical_row*11 +: 11] =
                11'(v4_output_row_base) +
                11'(row_physical_to_local[physical_row*6 +: 6]);
        end
    end
    assign row_configuration_release =
        state == METADATA_RELEASE ||
        (v4_multiple_rounds && state == SCHEDULE_WAIT &&
         scheduler_done_valid && scheduler_done_ready &&
         !scheduler_done_error && (layer_phase == LAYER_PHASE_KV || kv_pair_enable ||
          (layer_phase == LAYER_PHASE_O_CAPTURE && group_capture_index == 0))) ||
        (((state == HIDDEN_OUTPUT_WAIT && hidden_done_valid &&
           hidden_done_ready && !hidden_error) || state == SKIP_HIDDEN_OUTPUT) &&
         (layer_phase == LAYER_PHASE_ATTENTION_CAPTURE ? group_capture_index == 0 :
          layer_phase == LAYER_PHASE_KV_CAPTURE ?
            (!row_last_round &&
             (!qkvo_group || !v4_qkvo_group_end) &&
             group_capture_index + 3'd1 < (qkvo_group ? 3'd6 : 3'd2)) :
          layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE ?
            !(row_last_round ||
              group_capture_index + 3'd1 >= group_max_batch_count) :
          layer_phase == LAYER_PHASE_FFN_GROUP_DOWN ?
            (group_down_index + 3'd1 >= group_batch_count) :
          (layer_phase == LAYER_PHASE_O_REFILL || layer_phase == LAYER_PHASE_O_SINGLE) ?
            (qkvo_group ? group_down_index + 3'd1 >= group_batch_count :
             group_prepare_second) :
          layer_phase == LAYER_PHASE_FFN_GROUP_REFILL ?
            (group_prepare_second && group_down_index + 3'd2 >= group_batch_count) :
            (!row_last_round ||
             (separate_l31_metadata && !l31_metadata_active && !v4_multiple_rounds) ||
             (v4_multiple_rounds &&
              (layer_phase != LAYER_PHASE_FFN || !v4_last_layer))))) ||
        (completion_valid && completion_ready);
    assign configuration_refresh_enable = configuration_bits[EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET*8 +: 16] != 0;
    // Head-only executions load token metadata but never refresh K/V.
    assign refresh_loaded_token_valid = metadata_loaded_token_valid &&
        configuration_refresh_enable && !configuration_lm_head_only;
    assign configuration_forward_postprocess_enable = configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+2];
    assign configuration_lm_head_only =
        configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+3];
    assign configuration_layer_count =
        configuration_bits[EXECUTION_CONFIG_LAYER_COUNT_OFFSET*8 +: 6];
    assign configuration_start_layer = configuration_bits[EXECUTION_CONFIG_START_LAYER_OFFSET*8 +: 6];
    assign configuration_layer_end = configuration_start_layer +
        configuration_layer_count;
    always_comb begin
        attention_context_config = loaded_attention_context_config;
        if ((layer_phase == LAYER_PHASE_O_CAPTURE && group_capture_index != 0) ||
            (layer_phase == LAYER_PHASE_O_SINGLE && group_prepare_second))
            attention_context_config.context_base = loaded_attention_context_config.context_base +
                {45'd0, group_active_rows[0], 13'd0};
        if (layer_phase == LAYER_PHASE_ATTENTION_GROUP)
            attention_context_config.context_base =
                loaded_attention_context_config.context_base +
                {39'd0,
                 group_output_row_base[group_down_index] -
                    group_output_row_base[0],
                 13'd0};
        ffn_workspace_config = loaded_ffn_workspace_config;
        if (v4_multiple_rounds && (layer_phase == LAYER_PHASE_KV || layer_phase == LAYER_PHASE_KV_CAPTURE || kv_pair_enable ||
            layer_phase == LAYER_PHASE_ATTENTION || v4_group_phase ||
            layer_phase == LAYER_PHASE_O_CAPTURE ||
            layer_phase == LAYER_PHASE_O_MATMUL ||
            layer_phase == LAYER_PHASE_O_REFILL ||
            layer_phase == LAYER_PHASE_ATTENTION_CAPTURE || attention_pair_enable)) begin
            ffn_workspace_config.residual_base = v4_round_residual_base;
            ffn_workspace_config.residual_limit = v4_preserved_hidden_limit;
        end
        if (layer_phase == LAYER_PHASE_FFN_GROUP_DOWN ||
            layer_phase == LAYER_PHASE_FFN_GROUP_R4)
            ffn_workspace_config.gate_up_base =
                loaded_ffn_workspace_config.gate_up_base + group_workspace_offset;
        ffn_batch_pair_config = '0;
        ffn_batch_pair_config.fused_product =
            ffn_fused_product && v4_group_phase;
        ffn_batch_pair_config.enable =
            layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE;
        if (layer_phase == LAYER_PHASE_O_MATMUL ||
            layer_phase == LAYER_PHASE_ATTENTION_PAIR ||
            layer_phase == LAYER_PHASE_ATTENTION_GROUP || kv_pair_enable) begin
            ffn_batch_pair_config.enable = 1'b1;
            ffn_batch_pair_config.batch_count =
                qkvo_group && (kv_pair_enable ||
                    layer_phase == LAYER_PHASE_ATTENTION_GROUP ||
                    layer_phase == LAYER_PHASE_O_MATMUL) ?
                    group_batch_count : 3'd2;
            ffn_batch_pair_config.active_batch_index = group_down_index;
            ffn_batch_pair_config.compact_a4_layout = 1'b1;
            ffn_batch_pair_config.second_rows = group_active_rows[1];
            ffn_batch_pair_config.second_row_config = group_row_config[1];
            ffn_batch_pair_config.normalized_base = loaded_attention_context_config.context_base;
            ffn_batch_pair_config.normalized_limit = loaded_attention_context_config.context_base +
                ((64'(group_active_rows[0]) + 64'(group_active_rows[1])) << 13);
            if (kv_pair_enable) begin
                ffn_batch_pair_config.normalized_base =
                    loaded_ffn_workspace_config.gate_up_base +
                    {39'd0, group_output_row_base[0], 13'd0};
                ffn_batch_pair_config.normalized_limit =
                    ffn_batch_pair_config.normalized_base +
                    {39'd0,
                     group_output_row_base[group_batch_count - 3'd1] +
                        {6'd0, group_active_rows[group_batch_count - 3'd1]} -
                        group_output_row_base[0],
                     13'd0};
                if (qkvo_group) begin
                    ffn_batch_pair_config.qkvo_group = 1'b1;
                    ffn_batch_pair_config.qkv_staging_base = kv_batch_group_config.staging_base;
                    ffn_batch_pair_config.qkv_staging_limit = kv_batch_group_config.staging_limit;
                    ffn_batch_pair_config.qkv_staging_head_stride =
                        kv_batch_group_config.staging_head_stride;
                    ffn_batch_pair_config.second_rows = group_active_rows[1];
                    ffn_batch_pair_config.second_row_config = group_row_config[1];
                    ffn_batch_pair_config.third_rows = group_active_rows[2];
                    ffn_batch_pair_config.third_row_config = group_row_config[2];
                    ffn_batch_pair_config.fourth_rows = group_active_rows[3];
                    ffn_batch_pair_config.fourth_row_config = group_row_config[3];
                    ffn_batch_pair_config.fifth_rows = group_active_rows[4];
                    ffn_batch_pair_config.fifth_row_config = group_row_config[4];
                    ffn_batch_pair_config.sixth_rows = group_active_rows[5];
                    ffn_batch_pair_config.sixth_row_config = group_row_config[5];
                end
            end
        if (layer_phase == LAYER_PHASE_ATTENTION_GROUP) begin
                ffn_batch_pair_config.qkvo_group = 1'b1;
                ffn_batch_pair_config.normalized_base =
                    loaded_ffn_workspace_config.gate_up_base +
                    {39'd0, group_output_row_base[0], 13'd0};
                ffn_batch_pair_config.normalized_limit =
                    ffn_batch_pair_config.normalized_base +
                    {39'd0,
                     group_output_row_base[group_batch_count - 3'd1] +
                        {6'd0, group_active_rows[group_batch_count - 3'd1]} -
                        group_output_row_base[0],
                     13'd0};
                ffn_batch_pair_config.qkv_staging_base =
                    kv_batch_group_config.staging_base;
                ffn_batch_pair_config.qkv_staging_limit =
                    kv_batch_group_config.staging_limit;
                ffn_batch_pair_config.qkv_staging_head_stride =
                    kv_batch_group_config.staging_head_stride;
                ffn_batch_pair_config.third_rows = group_active_rows[2];
                ffn_batch_pair_config.third_row_config = group_row_config[2];
                ffn_batch_pair_config.fourth_rows = group_active_rows[3];
                ffn_batch_pair_config.fourth_row_config = group_row_config[3];
                ffn_batch_pair_config.fifth_rows = group_active_rows[4];
                ffn_batch_pair_config.fifth_row_config = group_row_config[4];
                ffn_batch_pair_config.sixth_rows = group_active_rows[5];
                ffn_batch_pair_config.sixth_row_config = group_row_config[5];
            end
            if (qkvo_group && layer_phase == LAYER_PHASE_O_MATMUL) begin
                ffn_batch_pair_config.qkvo_group = 1'b1;
                ffn_batch_pair_config.third_rows = group_active_rows[2];
                ffn_batch_pair_config.third_row_config = group_row_config[2];
                ffn_batch_pair_config.fourth_rows = group_active_rows[3];
                ffn_batch_pair_config.fourth_row_config = group_row_config[3];
                ffn_batch_pair_config.fifth_rows = group_active_rows[4];
                ffn_batch_pair_config.fifth_row_config = group_row_config[4];
                ffn_batch_pair_config.sixth_rows = group_active_rows[5];
                ffn_batch_pair_config.sixth_row_config = group_row_config[5];
                ffn_batch_pair_config.normalized_base =
                    loaded_attention_context_config.context_base;
                ffn_batch_pair_config.normalized_limit =
                    loaded_attention_context_config.context_base +
                    {39'd0,
                     group_output_row_base[group_batch_count - 3'd1] +
                        {6'd0, group_active_rows[group_batch_count - 3'd1]} -
                        group_output_row_base[0],
                     13'd0};
            end
            if (layer_phase == LAYER_PHASE_ATTENTION_PAIR) begin
                ffn_batch_pair_config.prepare_second = attention_second_batch;
                ffn_batch_pair_config.normalized_base = v4_preserved_hidden_base +
                    {39'd0, group_output_row_base[group_selected_index], 13'd0};
                ffn_batch_pair_config.normalized_limit = ffn_batch_pair_config.normalized_base +
                    {45'd0, group_active_rows[group_selected_index], 13'd0};
            end
        end
        if (group_down_pair_active &&
            (layer_phase == LAYER_PHASE_FFN_GROUP_R4 ||
             layer_phase == LAYER_PHASE_FFN_GROUP_DOWN)) begin
            ffn_batch_pair_config.enable = 1'b1;
            ffn_batch_pair_config.batch_count = 3'd2;
            ffn_batch_pair_config.compact_a4_layout = 1'b1;
            ffn_batch_pair_config.prepare_second = group_prepare_second;
            ffn_batch_pair_config.second_activation_base_byte_offset =
                32'(group_row_config[group_down_index].compute_group_count) *
                32'(FFN_FEATURES * 8);
            ffn_batch_pair_config.second_rows = group_active_rows[group_down_index + 3'd1];
            ffn_batch_pair_config.second_row_config = group_row_config[group_down_index + 3'd1];
        end
        if (layer_phase == LAYER_PHASE_FFN_GROUP_MATMUL) begin
            ffn_batch_pair_config.enable = group_batch_count >= 3'd2;
            ffn_batch_pair_config.batch_count = group_batch_count;
            ffn_batch_pair_config.compact_a4_layout =
                ffn_four_batch || ffn_six_batch;
            ffn_batch_pair_config.second_rows = group_active_rows[1];
            ffn_batch_pair_config.second_row_config = group_row_config[1];
            ffn_batch_pair_config.third_rows = group_active_rows[2];
            ffn_batch_pair_config.third_row_config = group_row_config[2];
            ffn_batch_pair_config.fourth_rows = group_active_rows[3];
            ffn_batch_pair_config.fourth_row_config = group_row_config[3];
            ffn_batch_pair_config.fifth_rows = group_active_rows[4];
            ffn_batch_pair_config.fifth_row_config = group_row_config[4];
            ffn_batch_pair_config.sixth_rows = group_active_rows[5];
            ffn_batch_pair_config.sixth_row_config = group_row_config[5];
            ffn_batch_pair_config.normalized_base = v4_preserved_hidden_base +
                {39'd0,group_output_row_base[0],13'd0};
            ffn_batch_pair_config.normalized_limit = v4_preserved_hidden_base +
                {39'd0,row_total_rows,13'd0};
        end
    end
    always_comb begin
        scheduler_layer_base = configuration_start_layer;
        scheduler_layer_count = separate_l31_metadata && !l31_metadata_active ?
            6'd31 - configuration_start_layer : configuration_layer_count;
        scheduler_step_mask = 11'h7ff;
        scheduler_cache_release_mask = 11'(1 <<
            layer_schedule_pkg::LAYER_OPERATOR_ATTENTION);
        scheduler_qkv_residual_spill_mask = 11'(1 <<
            (row_active_rows > 6'd32 ?
                layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_RMSNORM :
                layer_schedule_pkg::LAYER_OPERATOR_QKV_PREPARATION));
        scheduler_rms_qkv_prefetch_q_mask = 11'd0;
        scheduler_q_head_preprocess_mask = 11'd0;
        if (v4_multiple_rounds) begin
            scheduler_layer_base = v4_layer_index;
            scheduler_layer_count = 6'd1;
            scheduler_cache_release_mask = 11'd0;
            scheduler_qkv_residual_spill_mask = 11'd0;
            case (layer_phase)
                LAYER_PHASE_KV: begin
                    scheduler_step_mask = 11'b00000000011;
                    if (!row_last_round)
                        scheduler_cache_release_mask =
                            11'(1 <<
                                layer_schedule_pkg::
                                    LAYER_OPERATOR_QKV_PREPARATION);
                end
                LAYER_PHASE_ATTENTION, LAYER_PHASE_O_CAPTURE: begin
                    scheduler_step_mask = layer_phase == LAYER_PHASE_O_CAPTURE ?
                        11'b00000000101 : 11'b00000001101;
                    if (v4_output_subset_active && v4_output_token_count == 0)
                        scheduler_step_mask = 11'b00000000101;
                    scheduler_rms_qkv_prefetch_q_mask =
                        11'(1 <<
                            layer_schedule_pkg::
                                LAYER_OPERATOR_ATTENTION_RMSNORM);
                    scheduler_q_head_preprocess_mask =
                        11'(1 <<
                            layer_schedule_pkg::LAYER_OPERATOR_ATTENTION);
                    if (row_last_round)
                        scheduler_cache_release_mask =
                            11'(1 <<
                                layer_schedule_pkg::
                                    LAYER_OPERATOR_ATTENTION);
                end
                LAYER_PHASE_KV_CAPTURE: scheduler_step_mask = 11'b00000000001;
                LAYER_PHASE_KV_PAIR: begin
                    scheduler_step_mask = 11'b00000000010;
                    if (!group_last_round[group_batch_count - 3'd1])
                        scheduler_cache_release_mask = 11'b00000000010;
                end
                LAYER_PHASE_ATTENTION_CAPTURE: begin
                    scheduler_step_mask = 11'b00000000001;
                    scheduler_rms_qkv_prefetch_q_mask = 11'b00000000001;
                end
                LAYER_PHASE_ATTENTION_PAIR: begin
                    scheduler_step_mask = 11'b00000000100;
                    scheduler_q_head_preprocess_mask = 11'b00000000100;
                    if (group_last_round[1]) scheduler_cache_release_mask = 11'b00000000100;
                end
                LAYER_PHASE_ATTENTION_GROUP: begin
                    scheduler_step_mask = 11'b00000000100;
                    // Every metadata batch has its own positions and must
                    // reload RoPE constants. Later batches still reuse the
                    // first batch's grouped Q Matmul output.
                    scheduler_q_head_preprocess_mask = 11'b00000000100;
                    if (group_down_index +
                        {1'b0, attention_group_command_batches} >=
                        group_batch_count &&
                        group_last_round[group_batch_count - 3'd1])
                        scheduler_cache_release_mask = 11'b00000000100;
                end
                LAYER_PHASE_O_MATMUL, LAYER_PHASE_O_SINGLE:
                    scheduler_step_mask = 11'b00000001000;
                LAYER_PHASE_O_REFILL:
                    scheduler_step_mask = 11'b10000000000;
                LAYER_PHASE_FFN_GROUP_CAPTURE:
                    scheduler_step_mask = 11'b00000110000;
                LAYER_PHASE_FFN_GROUP_MATMUL:
                    scheduler_step_mask = ffn_fused_product ?
                        11'b00001000000 : 11'b00011000000;
                LAYER_PHASE_FFN_GROUP_DOWN:
                    scheduler_step_mask = group_down_pair_active ?
                        11'b01000000000 : 11'b11100000000;
                LAYER_PHASE_FFN_GROUP_R4:
                    scheduler_step_mask = 11'b00100000000;
                LAYER_PHASE_FFN_GROUP_REFILL:
                    scheduler_step_mask = 11'b10000000000;
                default: scheduler_step_mask = 11'b11111110000;
            endcase
        end
    end
    assign matmul_row_config = v4_use_group_rows ?
        group_row_config[group_selected_index] : live_matmul_row_config;
    assign live_matmul_row_config.segment_mode = row_segment_mode;
    assign live_matmul_row_config.segment_row_base = row_segment_base;
    assign live_matmul_row_config.segment_row_count = row_segment_count;
    assign live_matmul_row_config.row_enable = v4_row_matmul_valid &
        (v4_subset_compute ? v4_output_token_mask : 48'hffff_ffff_ffff);
    assign live_matmul_row_config.operator_has_w8 = row_operator_has_w8[0];
    assign live_matmul_row_config.mixed_group_enable = 1'b1;
    assign live_matmul_row_config.compute_group_count = v4_subset_compute ?
        v4_output_compute_groups : v4_compute_group_count;
    for (genvar matmul_row = 0; matmul_row < 48; matmul_row++) begin
        assign live_matmul_row_config.row_precision_a8[matmul_row] =
            live_matmul_row_config.row_enable[matmul_row] &&
            v4_row_precision_bits[matmul_row*4 +: 4] == 4'd8;
    end
    assign live_matmul_row_config.row_compute_group = v4_row_compute_group;
    assign live_matmul_row_config.row_pe_slot = v4_row_pe_slot;
    assign live_matmul_row_config.row_phase_mask = v4_row_phase_mask;
    assign hidden_row_config.physical_to_local = row_physical_to_local;
    assign attention_row_config.token_position = row_token_position;
    assign attention_row_config.row_enable = row_matmul_valid;
    assign attention_row_config.token_batch_index = row_token_batch_index;
    assign attention_row_config.kv_write_disable =
        (attention_pair_enable || kv_pair_enable ||
         layer_phase == LAYER_PHASE_ATTENTION_GROUP) ?
        group_kv_write_disable[group_selected_index] : v4_row_kv_write_disable;

    always_comb begin
        v4_operator_has_a8 = '0;
    end

    always_comb begin
        metadata_start_ready = v4_metadata_start_ready;
        metadata_done_valid = v4_metadata_done_valid;
        metadata_done_error = v4_metadata_done_error;
        metadata_done_error_id = v4_metadata_done_error_id;
        dma_read_request_valid[DMA_READ_METADATA] = v4_metadata_request_valid;
        dma_read_request[DMA_READ_METADATA].byte_address = v4_metadata_request_address;
        dma_read_request[DMA_READ_METADATA].byte_count = v4_metadata_request_bytes;
        dma_read_data_ready[DMA_READ_METADATA] = v4_metadata_response_ready;

        row_configuration_valid = v4_row_configuration_valid;
        row_total_rows = v4_row_total_rows;
        row_token_batch_index = v4_row_token_batch_index;
        row_last_round = v4_row_last_round;
        row_active_rows = v4_subset_compute ? v4_output_token_count : v4_row_active_rows;
        row_sequence_length = v4_row_sequence_length;
        row_segment_mode = 6'd0;
        row_segment_base = 18'd0;
        row_segment_count = 18'd0;
        row_physical_to_local = v4_row_physical_to_local;
        row_token_position = v4_row_token_position;
        row_matmul_valid = live_matmul_row_config.row_enable;
        row_operator_has_w8 = v4_operator_has_a8;
        if (v4_use_group_rows) begin
            row_token_batch_index = group_token_batch_index[group_selected_index];
            row_last_round = group_last_round[group_selected_index];
            row_active_rows = group_active_rows[group_selected_index];
            row_physical_to_local =
                group_physical_to_local[group_selected_index];
            row_token_position =
                group_token_position[group_selected_index];
            row_matmul_valid = group_row_config[group_selected_index].row_enable;
        end
    end

    execution_config_loader execution_config_loader (
        .clk(clk), .rst(rst), .start_valid(loader_start_valid),
        .start_ready(loader_start_ready), .start_address(saved_config_address),
        .dma_request_valid(dma_read_request_valid[DMA_READ_CONFIGURATION]),
        .dma_request_ready(dma_read_request_ready[DMA_READ_CONFIGURATION]),
        .dma_request_address(
            dma_read_request[DMA_READ_CONFIGURATION].byte_address),
        .dma_request_bytes(
            dma_read_request[DMA_READ_CONFIGURATION].byte_count),
        .dma_response_valid(dma_read_data_valid[DMA_READ_CONFIGURATION]),
        .dma_response_ready(dma_read_data_ready[DMA_READ_CONFIGURATION]),
        .dma_response_data(dma_read_data[DMA_READ_CONFIGURATION].data),
        .dma_response_byte_enable(
            dma_read_data[DMA_READ_CONFIGURATION].byte_enable),
        .dma_response_last(dma_read_data[DMA_READ_CONFIGURATION].last),
        .dma_request_done(dma_read_completion[DMA_READ_CONFIGURATION].done_pulse),
        .dma_request_error(dma_read_completion[DMA_READ_CONFIGURATION].error),
        .done_valid(loader_done_valid), .done_ready(loader_done_ready),
        .done_error(loader_done_error), .done_error_id(loader_done_error_id),
        .configuration_bits(configuration_bits)
    );

    token_metadata_loader v4_row_configuration (
        .loaded_token_valid(metadata_loaded_token_valid), .loaded_token_position(refresh_loaded_token_position),
        .loaded_token_kv_write(refresh_loaded_token_kv_write),
        .clk(clk), .rst(rst),
        .start_valid(metadata_start_valid),
        .start_ready(v4_metadata_start_ready),
        .start_configuration_bits(configuration_bits),
        .start_l31_layout(l31_metadata_active),
        .dma_request_valid(v4_metadata_request_valid),
        .dma_request_ready(dma_read_request_ready[DMA_READ_METADATA]),
        .dma_request_address(v4_metadata_request_address),
        .dma_request_bytes(v4_metadata_request_bytes),
        .dma_response_valid(dma_read_data_valid[DMA_READ_METADATA]),
        .dma_response_ready(v4_metadata_response_ready),
        .dma_response_data(dma_read_data[DMA_READ_METADATA].data),
        .dma_response_byte_enable(
            dma_read_data[DMA_READ_METADATA].byte_enable),
        .dma_response_last(dma_read_data[DMA_READ_METADATA].last),
        .dma_request_done(dma_read_completion[DMA_READ_METADATA].done_pulse),
        .dma_request_error(dma_read_completion[DMA_READ_METADATA].error),
        .done_valid(v4_metadata_done_valid),
        .done_ready(metadata_done_ready),
        .done_error(v4_metadata_done_error),
        .done_error_id(v4_metadata_done_error_id),
        .config_valid(v4_row_configuration_valid),
        .config_release(row_configuration_release),
        .total_token_count(v4_row_total_rows),
        .token_batch_index(v4_row_token_batch_index),
        .last_round(v4_row_last_round),
        .active_token_count(v4_row_active_rows),
        .compute_group_count(v4_compute_group_count),
        .output_subset_enable(v4_output_subset_enable),
        .qkvo_group_end(v4_qkvo_group_end),
        .output_token_count(v4_output_token_count),
        .output_compute_group_count(v4_output_compute_groups),
        .sequence_length(v4_row_sequence_length),
        .physical_to_local(v4_row_physical_to_local),
        .local_to_physical(),
        .token_position(v4_row_token_position),
        .row_kv_index(),
        .row_kv_write_disable(v4_row_kv_write_disable),
        .matmul_row_valid(v4_row_matmul_valid),
        .row_source_index(v4_row_source_index),
        .row_embedding_source(v4_row_embedding_source),
        .row_precision_bits(v4_row_precision_bits),
        .row_query_group(v4_row_query_group),
        .row_cache_group(v4_row_cache_group),
        .row_compute_group(v4_row_compute_group),
        .row_pe_slot(v4_row_pe_slot),
        .row_phase_mask(v4_row_phase_mask));

    layer_address_loader layer_loader (
        .clk(clk), .rst(rst),
        .entry_invalidate(launch_valid && launch_ready),
        .abort_request(1'b0), .abort_ack(),
        .load_valid(layer_load_pending), .load_ready(layer_load_ready),
        .load_layer_index(command_layer[4:0]),
        .execution_configuration_bits(configuration_bits),
        .dma_request_valid(dma_read_request_valid[DMA_READ_LAYER]),
        .dma_request_ready(dma_read_request_ready[DMA_READ_LAYER]),
        .dma_request_address(dma_read_request[DMA_READ_LAYER].byte_address),
        .dma_request_bytes(dma_read_request[DMA_READ_LAYER].byte_count),
        .dma_response_valid(dma_read_data_valid[DMA_READ_LAYER]),
        .dma_response_ready(dma_read_data_ready[DMA_READ_LAYER]),
        .dma_response_data(dma_read_data[DMA_READ_LAYER].data),
        .dma_response_byte_enable(dma_read_data[DMA_READ_LAYER].byte_enable),
        .dma_response_last(dma_read_data[DMA_READ_LAYER].last),
        .dma_request_done(dma_read_completion[DMA_READ_LAYER].done_pulse),
        .dma_request_error(dma_read_completion[DMA_READ_LAYER].error),
        .done_valid(layer_done_valid), .done_ready(1'b1),
        .done_error(layer_done_error),
        .done_error_id(layer_done_error_id),
        .entry_valid(layer_entry_valid), .entry_layer_index(layer_entry_index),
        .matmul_weight_config(matmul_weight_config),
        .rmsnorm_config(rmsnorm_layer_config),
        .qkv_config(qkv_layer_config),
        .attention_cache_config(attention_cache_config),
        .attention_context_config(loaded_attention_context_config),
        .ffn_workspace_config(loaded_ffn_workspace_config)
    );

    layer_scheduler scheduler (
        .clk(clk), .rst(rst || layer_load_failed || refresh_layer_failed), .start_valid(scheduler_start_valid),
        .start_ready(scheduler_start_ready),
        .start_layer_base(scheduler_layer_base),
        .start_layer_count(scheduler_layer_count),
        .start_step_mask(scheduler_step_mask),
        .start_cache_release_mask(scheduler_cache_release_mask),
        .start_qkv_residual_spill_mask(
            scheduler_qkv_residual_spill_mask),
        .start_rms_qkv_prefetch_q_mask(
            scheduler_rms_qkv_prefetch_q_mask),
        .start_q_head_preprocess_mask(
            scheduler_q_head_preprocess_mask),
        .command_valid(scheduler_command_valid), .command_ready(command_ready && state == SCHEDULE_WAIT),
        .command_layer(command_layer), .layer_operator_index(layer_operator_index),
        .command_is_mixed_matmul(),
        .command_cache_release(command_cache_release),
        .command_qkv_residual_spill(command_qkv_residual_spill),
        .command_rms_qkv_prefetch_q(command_rms_qkv_prefetch_q),
        .command_q_head_preprocess(command_q_head_preprocess),
        .command_done_valid(command_done_valid),
        .command_done_ready(command_done_ready),
        .command_done_error(command_done_error),
        .command_done_error_id(command_done_error_id),
        .done_valid(scheduler_done_valid), .done_ready(scheduler_done_ready),
        .done_error(scheduler_done_error),
        .done_error_id(scheduler_done_error_id)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            refresh_relation_l31 <= 1'b0;
            saved_config_address <= 64'd0;
            refresh_closeout <= 1'b0;
            completion_id <= 32'd0;
            completion_error <= 1'b0;
            completion_error_id <= 16'd0;
            layer_load_pending <= 1'b0;
            v4_output_row_base <= 12'd0;
            layer_phase <= LAYER_PHASE_KV;
            v4_layer_index <= 6'd0;
            group_capture_index <= 3'd0;
            group_down_index <= 3'd0;
            group_down_pair_active <= 1'b0;
            group_prepare_second <= 1'b0;
            group_batch_count <= 3'd1;
            group_compute_group_count <= 6'd0;
            group_workspace_offset <= 64'd0;
        end else begin
            if (command_valid &&
                (!layer_entry_valid || layer_entry_index != command_layer[4:0]))
                layer_load_pending <= 1'b1;
            if (layer_load_pending && layer_load_ready)
                layer_load_pending <= 1'b0;

            case (state)
                IDLE: if (launch_valid && launch_ready) begin
                    saved_config_address <= launch_config_addr;
                    refresh_closeout <= 1'b0;
                    completion_id <= launch_id;
                    completion_error <= 1'b0;
                    completion_error_id <= 16'd0;
                    v4_output_row_base <= 12'd0;
                    layer_phase <= LAYER_PHASE_KV;
                    v4_layer_index <= 6'd0;
                    group_capture_index <= 3'd0;
                    group_down_index <= 3'd0;
                    group_down_pair_active <= 1'b0;
                    group_prepare_second <= 1'b0;
                    group_batch_count <= 3'd1;
                    group_compute_group_count <= 6'd0;
                    group_workspace_offset <= 64'd0;
                    state <= CONFIG_START;
                end

                CONFIG_START: if (loader_start_valid && loader_start_ready)
                    state <= CONFIG_WAIT;

                CONFIG_WAIT: if (loader_done_valid && loader_done_ready) begin
                    if (loader_done_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= loader_done_error_id;
                        state <= COMPLETE;
                    end else begin
                        v4_layer_index <= configuration_start_layer;
                        state <= configuration_refresh_enable ? REFRESH_PREPARE_START : METADATA_START;
                    end
                end

                METADATA_START: if (metadata_start_valid && metadata_start_ready)
                    state <= METADATA_WAIT;

                METADATA_WAIT: if (metadata_done_valid && metadata_done_ready) begin
                    if (metadata_done_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= metadata_done_error_id;
                        // The integrated memory subsystem inserts its common
                        // accepted-transaction drain gate before COMPLETE.
                        state <= COMPLETE;
                    end else if (layer_phase == LAYER_PHASE_KV &&
                                 row_last_round && ffn_pair_first_batch_plus1 >
                                    {2'd0,row_token_batch_index}) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_FFN_PAIR;
                        state <= COMPLETE;
                    end else if (configuration_lm_head_only) begin
                        state <= FORWARD_POSTPROCESS_START;
                    end else if (layer_phase ==
                                 LAYER_PHASE_ATTENTION_GROUP_SCAN) begin
                        group_output_row_base[group_capture_index] <=
                            v4_output_row_base;
                        group_active_rows[group_capture_index] <=
                            row_active_rows;
                        group_token_batch_index[group_capture_index] <=
                            row_token_batch_index;
                        group_last_round[group_capture_index] <=
                            row_last_round;
                        group_physical_to_local[group_capture_index] <=
                            row_physical_to_local;
                        group_token_position[group_capture_index] <=
                            row_token_position;
                        group_kv_write_disable[group_capture_index] <=
                            v4_row_kv_write_disable;
                        group_row_config[group_capture_index] <=
                            live_matmul_row_config;
                        if (group_compute_group_count +
                                live_matmul_row_config.compute_group_count >
                                6'd18) begin
                            completion_error <= 1'b1;
                            completion_error_id <= ERROR_QKVO_GROUP;
                            state <= COMPLETE;
                        end else if (row_last_round || v4_qkvo_group_end ||
                                     group_capture_index == 3'd5) begin
                            if (!row_last_round && !v4_qkvo_group_end) begin
                                completion_error <= 1'b1;
                                completion_error_id <= ERROR_QKVO_GROUP;
                                state <= COMPLETE;
                            end else if (group_capture_index == 0) begin
                                group_batch_count <= 3'd1;
                                group_compute_group_count <=
                                    live_matmul_row_config.compute_group_count;
                                group_down_index <= 3'd0;
                                layer_phase <= LAYER_PHASE_ATTENTION;
                                state <= HIDDEN_INPUT_START;
                            end else begin
                                group_batch_count <= group_capture_index + 3'd1;
                                group_compute_group_count <=
                                    group_compute_group_count +
                                    live_matmul_row_config.compute_group_count;
                                group_down_index <= 3'd0;
                                v4_output_row_base <= group_capture_index == 0 ?
                                    v4_output_row_base : group_output_row_base[0];
                                layer_phase <= LAYER_PHASE_ATTENTION_GROUP;
                                state <= SCHEDULE_START;
                            end
                        end else begin
                            group_compute_group_count <=
                                group_compute_group_count +
                                live_matmul_row_config.compute_group_count;
                            group_capture_index <= group_capture_index + 3'd1;
                            v4_output_row_base <= v4_output_row_base +
                                {6'd0, row_active_rows};
                            state <= METADATA_RELEASE;
                        end
                    end else if (v4_output_subset_active && layer_phase == LAYER_PHASE_FFN &&
                                 v4_output_token_count == 0) begin
                        state <= SKIP_HIDDEN_OUTPUT;
                    end else if (!v4_output_subset_active &&
                        (!qkvo_group && (((attention_output_pair || attention_batch_pair) &&
                        (layer_phase == LAYER_PHASE_ATTENTION ||
                         layer_phase == LAYER_PHASE_O_CAPTURE ||
                         layer_phase == LAYER_PHASE_ATTENTION_CAPTURE)) ||
                         (kv_batch_pair && (layer_phase == LAYER_PHASE_KV ||
                                            layer_phase == LAYER_PHASE_KV_CAPTURE)))) &&
                        (live_matmul_row_config.row_precision_a8 != 0 ||
                         live_matmul_row_config.operator_has_w8 ||
                         !live_matmul_row_config.mixed_group_enable ||
                         live_matmul_row_config.compute_group_count > 6'd3)) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_FFN_PAIR;
                        state <= COMPLETE;
                    end else begin
                        if (!v4_output_subset_active && (kv_batch_pair || qkvo_group) &&
                            layer_phase == LAYER_PHASE_KV && !row_last_round &&
                            (!qkvo_group || !v4_qkvo_group_end)) begin
                            layer_phase <= LAYER_PHASE_KV_CAPTURE;
                            group_capture_index <= 3'd0;
                            group_compute_group_count <= 6'd0;
                            group_down_index <= 3'd0;
                            group_prepare_second <= 1'b0;
                        end
                        if (!v4_output_subset_active && layer_phase == LAYER_PHASE_ATTENTION && !row_last_round &&
                            (attention_output_pair || (attention_batch_pair && row_sequence_length <= 12'd1296))) begin
                            layer_phase <= attention_batch_pair && row_sequence_length <= 12'd1296 ?
                                LAYER_PHASE_ATTENTION_CAPTURE : LAYER_PHASE_O_CAPTURE;
                            group_capture_index <= 3'd0;
                            group_down_index <= 3'd0;
                            group_prepare_second <= 1'b0;
                        end
                        if (!v4_output_subset_active && layer_phase == LAYER_PHASE_FFN &&
                            ffn_pair_start) begin
                            layer_phase <= LAYER_PHASE_FFN_GROUP_CAPTURE;
                            group_capture_index <= 3'd0;
                            group_batch_count <= 3'd1;
                            group_workspace_offset <= 64'd0;
                        end
                        state <= HIDDEN_INPUT_START;
                    end
                end

                METADATA_RELEASE: if (row_configuration_valid)
                    state <= METADATA_START;

                HIDDEN_INPUT_START: if (hidden_start_valid && hidden_start_ready)
                    state <= HIDDEN_INPUT_WAIT;

                HIDDEN_INPUT_WAIT: if (hidden_done_valid && hidden_done_ready) begin
                    if (hidden_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_HIDDEN_INPUT;
                        state <= COMPLETE;
                    end else if (v4_multiple_rounds &&
                                 (layer_phase == LAYER_PHASE_KV || layer_phase == LAYER_PHASE_KV_CAPTURE)) begin
                        if ({1'b0, v4_residual_spill_base} +
                                {40'd0, row_active_rows, 13'd0} >
                                {1'b0, v4_residual_spill_limit} ||
                            (v4_preserve_input &&
                             {1'b0, v4_preserved_hidden_base} +
                                {39'd0, row_total_rows, 14'd0} >
                                {1'b0, v4_preserved_hidden_limit})) begin
                            completion_error <= 1'b1;
                            completion_error_id <= ERROR_HIDDEN_OUTPUT;
                            state <= COMPLETE;
                        end else if (v4_preserve_input) begin
                            state <= HIDDEN_PRESERVE_START;
                        end else begin
                            state <= HIDDEN_RESIDUAL_START;
                        end
                    end else begin
                        state <= SCHEDULE_START;
                    end
                end

                HIDDEN_PRESERVE_START: if (hidden_start_valid && hidden_start_ready)
                    state <= HIDDEN_PRESERVE_WAIT;

                HIDDEN_PRESERVE_WAIT: if (hidden_done_valid && hidden_done_ready) begin
                    if (hidden_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_HIDDEN_OUTPUT;
                        state <= COMPLETE;
                    end else begin
                        state <= HIDDEN_RESIDUAL_START;
                    end
                end

                HIDDEN_RESIDUAL_START:
                    if (hidden_start_valid && hidden_start_ready)
                        state <= HIDDEN_RESIDUAL_WAIT;

                HIDDEN_RESIDUAL_WAIT:
                    if (hidden_done_valid && hidden_done_ready) begin
                        if (hidden_error) begin
                            completion_error <= 1'b1;
                            completion_error_id <= ERROR_HIDDEN_OUTPUT;
                            state <= COMPLETE;
                        end else begin
                            state <= SCHEDULE_START;
                        end
                    end

                SCHEDULE_START: begin
                    if ((layer_phase == LAYER_PHASE_O_CAPTURE ||
                         layer_phase == LAYER_PHASE_ATTENTION_PAIR) && !o_context_address_valid) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_FFN_PAIR;
                        state <= COMPLETE;
                    end else if (scheduler_start_valid && scheduler_start_ready)
                        state <= SCHEDULE_WAIT;
                end

                SCHEDULE_WAIT: if (command_done_valid && command_done_ready &&
                    !command_done_error && (row_last_round ||
                     (layer_phase == LAYER_PHASE_O_MATMUL &&
                      group_last_round[group_batch_count - 3'd1])) &&
                    command_layer < 32 &&
                    (layer_operator_index == layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL ||
                     (v4_output_subset_active && v4_output_token_count == 0 &&
                      layer_operator_index == layer_schedule_pkg::LAYER_OPERATOR_ATTENTION)) &&
                    refresh_relation_layer_mask[command_layer[4:0]]) begin
                    // The scheduler holds its next command while DDR dependency
                    // reduction consumes this layer's completed probability image.
                    refresh_relation_l31 <= command_layer == 6'd31;
                    state <= REFRESH_LAYER_START;
                end else if (scheduler_done_valid && scheduler_done_ready) begin
                    completion_error <= scheduler_done_error;
                    completion_error_id <= scheduler_done_error_id;
                    if (scheduler_done_error) begin
                        state <= COMPLETE;
                    end else if (kv_pair_enable) begin
                        layer_phase <= group_last_round[group_batch_count - 3'd1] ?
                            (qkvo_group ? LAYER_PHASE_ATTENTION_GROUP_SCAN :
                                          LAYER_PHASE_ATTENTION) : LAYER_PHASE_KV;
                        v4_output_row_base <= group_last_round[group_batch_count - 3'd1] ?
                            12'd0 : group_output_row_base[group_batch_count - 3'd1] +
                            {6'd0, group_active_rows[group_batch_count - 3'd1]};
                        if (group_last_round[group_batch_count - 3'd1] &&
                            qkvo_group) begin
                            group_capture_index <= 3'd0;
                            group_compute_group_count <= 6'd0;
                            group_down_index <= 3'd0;
                        end
                        state <= METADATA_START;
                    end else if (v4_multiple_rounds &&
                                 layer_phase == LAYER_PHASE_KV) begin
                        if (row_last_round) begin
                            layer_phase <= qkvo_group ?
                                LAYER_PHASE_ATTENTION_GROUP_SCAN :
                                LAYER_PHASE_ATTENTION;
                            v4_output_row_base <= 12'd0;
                            if (qkvo_group) begin
                                group_capture_index <= 3'd0;
                                group_compute_group_count <= 6'd0;
                                group_down_index <= 3'd0;
                            end
                        end else begin
                            v4_output_row_base <= v4_output_row_base +
                                {6'd0, row_active_rows};
                        end
                        state <= METADATA_START;
                    end else if (layer_phase == LAYER_PHASE_O_CAPTURE) begin
                        group_output_row_base[group_capture_index] <= v4_output_row_base;
                        group_active_rows[group_capture_index] <= row_active_rows;
                        group_token_batch_index[group_capture_index] <= row_token_batch_index;
                        group_last_round[group_capture_index] <= row_last_round;
                        group_physical_to_local[group_capture_index] <= row_physical_to_local;
                        group_token_position[group_capture_index] <= row_token_position;
                        group_row_config[group_capture_index] <= live_matmul_row_config;
                        if (group_capture_index == 0) begin
                            group_capture_index <= 3'd1;
                            v4_output_row_base <= v4_output_row_base + {6'd0, row_active_rows};
                            state <= METADATA_START;
                        end else begin
                            v4_output_row_base <= group_output_row_base[0];
                            layer_phase <= LAYER_PHASE_O_MATMUL;
                            state <= SCHEDULE_START;
                        end
                    end else if (layer_phase == LAYER_PHASE_ATTENTION_PAIR) begin
                        group_prepare_second <= 1'b0;
                        v4_output_row_base <= group_output_row_base[0];
                        layer_phase <= attention_output_pair ? LAYER_PHASE_O_MATMUL : LAYER_PHASE_O_SINGLE;
                        state <= SCHEDULE_START;
                    end else if (layer_phase ==
                                 LAYER_PHASE_ATTENTION_GROUP) begin
                        if (group_down_index +
                            {1'b0, attention_group_command_batches} <
                            group_batch_count) begin
                            group_down_index <= group_down_index +
                                {1'b0, attention_group_command_batches};
                            v4_output_row_base <=
                                group_output_row_base[group_down_index +
                                    {1'b0, attention_group_command_batches}];
                            state <= SCHEDULE_START;
                        end else begin
                            group_down_index <= 3'd0;
                            group_prepare_second <= 1'b0;
                            v4_output_row_base <= group_output_row_base[0];
                            layer_phase <= LAYER_PHASE_O_MATMUL;
                            state <= SCHEDULE_START;
                        end
                    end else if (layer_phase == LAYER_PHASE_O_MATMUL) begin
                        layer_phase <= LAYER_PHASE_O_REFILL;
                        group_down_index <= 3'd0;
                        group_prepare_second <= 1'b0;
                        v4_output_row_base <= group_output_row_base[0];
                        state <= SCHEDULE_START;
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_MATMUL) begin
                        layer_phase <= ffn_down_pair && group_batch_count >= 3'd2 ?
                            LAYER_PHASE_FFN_GROUP_R4 : LAYER_PHASE_FFN_GROUP_DOWN;
                        group_down_pair_active <= ffn_down_pair && group_batch_count >= 3'd2;
                        group_prepare_second <= 1'b0;
                        group_down_index <= 3'd0;
                        group_workspace_offset <= 64'd0;
                        v4_output_row_base <= group_output_row_base[0];
                        state <= SCHEDULE_START;
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_R4) begin
                        if (!group_prepare_second) begin
                            group_prepare_second <= 1'b1;
                            group_workspace_offset <= group_workspace_offset +
                                (64'(group_active_rows[group_down_index]) +
                                 64'(group_active_rows[group_down_index][0])) * 64'(FFN_FEATURES*2);
                            v4_output_row_base <= group_output_row_base[group_down_index + 3'd1];
                        end else begin
                            group_prepare_second <= 1'b0;
                            layer_phase <= LAYER_PHASE_FFN_GROUP_DOWN;
                            v4_output_row_base <= group_output_row_base[group_down_index];
                        end
                        state <= SCHEDULE_START;
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_DOWN && group_down_pair_active) begin
                        layer_phase <= LAYER_PHASE_FFN_GROUP_REFILL;
                        state <= SCHEDULE_START;
                    end else if (v4_output_subset_active && v4_output_token_count == 0 &&
                                 layer_phase == LAYER_PHASE_ATTENTION) begin
                        state <= SKIP_HIDDEN_OUTPUT;
                    end else begin
                        state <= HIDDEN_OUTPUT_START;
                    end
                end

                HIDDEN_OUTPUT_START: if (hidden_start_valid && hidden_start_ready)
                    state <= HIDDEN_OUTPUT_WAIT;

                HIDDEN_OUTPUT_WAIT, SKIP_HIDDEN_OUTPUT:
                if (state == SKIP_HIDDEN_OUTPUT || (hidden_done_valid && hidden_done_ready)) begin
                    if (state != SKIP_HIDDEN_OUTPUT && hidden_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= ERROR_HIDDEN_OUTPUT;
                        state <= COMPLETE;
                    end else if (layer_phase == LAYER_PHASE_O_REFILL ||
                                 layer_phase == LAYER_PHASE_O_SINGLE) begin
                        if (qkvo_group &&
                            layer_phase == LAYER_PHASE_O_REFILL) begin
                            if (group_down_index + 3'd1 <
                                group_batch_count) begin
                                group_down_index <= group_down_index + 3'd1;
                                v4_output_row_base <= group_output_row_base[
                                    group_down_index + 3'd1];
                                state <= SCHEDULE_START;
                            end else if (group_last_round[
                                         group_batch_count - 3'd1]) begin
                                group_down_index <= 3'd0;
                                layer_phase <= LAYER_PHASE_FFN;
                                v4_output_row_base <= 12'd0;
                                state <= METADATA_START;
                            end else begin
                                group_capture_index <= 3'd0;
                                group_compute_group_count <= 6'd0;
                                group_down_index <= 3'd0;
                                layer_phase <=
                                    LAYER_PHASE_ATTENTION_GROUP_SCAN;
                                v4_output_row_base <=
                                    group_output_row_base[
                                        group_batch_count - 3'd1] +
                                    {6'd0, group_active_rows[
                                        group_batch_count - 3'd1]};
                                state <= METADATA_START;
                            end
                        end else if (!group_prepare_second) begin
                            group_prepare_second <= 1'b1;
                            v4_output_row_base <= group_output_row_base[1];
                            state <= SCHEDULE_START;
                        end else begin
                            group_prepare_second <= 1'b0;
                            if (!row_last_round) begin
                                layer_phase <= LAYER_PHASE_ATTENTION;
                                v4_output_row_base <= v4_output_row_base + {6'd0, row_active_rows};
                            end else begin
                                layer_phase <= LAYER_PHASE_FFN;
                                v4_output_row_base <= 12'd0;
                            end
                            state <= METADATA_START;
                        end
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE ||
                                 layer_phase == LAYER_PHASE_ATTENTION_CAPTURE ||
                                 layer_phase == LAYER_PHASE_KV_CAPTURE) begin
                        group_output_row_base[group_capture_index] <= v4_output_row_base;
                        group_active_rows[group_capture_index] <= row_active_rows;
                        group_token_batch_index[group_capture_index] <= row_token_batch_index;
                        group_last_round[group_capture_index] <= row_last_round;
                        group_physical_to_local[group_capture_index] <=
                            row_physical_to_local;
                        group_token_position[group_capture_index] <=
                            row_token_position;
                        group_row_config[group_capture_index] <=
                            live_matmul_row_config;
                        if (layer_phase == LAYER_PHASE_ATTENTION_CAPTURE || layer_phase == LAYER_PHASE_KV_CAPTURE)
                            group_kv_write_disable[group_capture_index] <= v4_row_kv_write_disable;
                        if (qkvo_group &&
                            layer_phase == LAYER_PHASE_KV_CAPTURE &&
                            group_compute_group_count +
                                live_matmul_row_config.compute_group_count > 6'd18) begin
                            completion_error <= 1'b1;
                            completion_error_id <= ERROR_QKVO_GROUP;
                            state <= COMPLETE;
                        end else if (row_last_round ||
                            (qkvo_group &&
                             layer_phase == LAYER_PHASE_KV_CAPTURE &&
                             v4_qkvo_group_end) ||
                            group_capture_index + 3'd1 >=
                                ((layer_phase == LAYER_PHASE_ATTENTION_CAPTURE) ? 3'd2 :
                                 (layer_phase == LAYER_PHASE_KV_CAPTURE ?
                                    (qkvo_group ? 3'd6 : 3'd2) : group_max_batch_count))) begin
                            if (qkvo_group &&
                                layer_phase == LAYER_PHASE_KV_CAPTURE &&
                                !row_last_round && !v4_qkvo_group_end) begin
                                completion_error <= 1'b1;
                                completion_error_id <= ERROR_QKVO_GROUP;
                                state <= COMPLETE;
                            end else begin
                            group_batch_count <=
                                group_capture_index + 3'd1;
                            group_compute_group_count <= group_compute_group_count +
                                live_matmul_row_config.compute_group_count;
                            v4_output_row_base <=
                                layer_phase == LAYER_PHASE_FFN_GROUP_CAPTURE &&
                                group_capture_index == 0 ?
                                    v4_output_row_base : group_output_row_base[0];
                            layer_phase <= layer_phase == LAYER_PHASE_KV_CAPTURE ? LAYER_PHASE_KV_PAIR :
                                layer_phase == LAYER_PHASE_ATTENTION_CAPTURE ?
                                LAYER_PHASE_ATTENTION_PAIR : LAYER_PHASE_FFN_GROUP_MATMUL;
                            state <= SCHEDULE_START;
                            end
                        end else begin
                            v4_output_row_base <= v4_output_row_base +
                                {6'd0,row_active_rows};
                            group_capture_index <= group_capture_index + 3'd1;
                            group_compute_group_count <= group_compute_group_count +
                                live_matmul_row_config.compute_group_count;
                            state <= METADATA_START;
                        end
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_REFILL && !group_prepare_second) begin
                        group_prepare_second <= 1'b1;
                        v4_output_row_base <= group_output_row_base[group_down_index + 3'd1];
                        state <= SCHEDULE_START;
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_REFILL &&
                                 group_down_index + 3'd2 < group_batch_count) begin
                        group_workspace_offset <= group_workspace_offset +
                            (64'(group_active_rows[group_down_index + 3'd1]) +
                             64'(group_active_rows[group_down_index + 3'd1][0])) * 64'(FFN_FEATURES*2);
                        group_down_index <= group_down_index + 3'd2;
                        group_prepare_second <= 1'b0;
                        group_down_pair_active <= group_down_index + 3'd3 < group_batch_count;
                        layer_phase <= group_down_index + 3'd3 < group_batch_count ?
                            LAYER_PHASE_FFN_GROUP_R4 : LAYER_PHASE_FFN_GROUP_DOWN;
                        v4_output_row_base <= group_output_row_base[group_down_index + 3'd2];
                        state <= SCHEDULE_START;
                    end else if (layer_phase == LAYER_PHASE_FFN_GROUP_DOWN &&
                                 group_down_index + 3'd1 <
                                     group_batch_count) begin
                        group_workspace_offset <= group_workspace_offset +
                            (64'(group_active_rows[group_down_index]) +
                             64'(group_active_rows[group_down_index][0])) *
                            (ffn_fused_product ? 64'(FFN_FEATURES*2) :
                                                 64'(FFN_FEATURES*4));
                        group_down_index <= group_down_index + 3'd1;
                        v4_output_row_base <=
                            group_output_row_base[group_down_index + 3'd1];
                        state <= SCHEDULE_START;
                    end else if (v4_multiple_rounds) begin
                        if (layer_phase == LAYER_PHASE_FFN_GROUP_DOWN ||
                            layer_phase == LAYER_PHASE_FFN_GROUP_REFILL)
                            layer_phase <= LAYER_PHASE_FFN;
                        completion_error <= 1'b0;
                        completion_error_id <= 16'd0;
                        if (!row_last_round) begin
                            v4_output_row_base <= v4_output_row_base +
                                {6'd0, v4_output_subset_active ? v4_row_active_rows : row_active_rows};
                            state <= METADATA_START;
                        end else if (layer_phase == LAYER_PHASE_ATTENTION) begin
                            layer_phase <= LAYER_PHASE_FFN;
                            v4_output_row_base <= 12'd0;
                            state <= METADATA_START;
                        end else if (!v4_last_layer) begin
                            v4_layer_index <= v4_layer_index + 1'b1;
                            layer_phase <= LAYER_PHASE_KV;
                            v4_output_row_base <= 12'd0;
                            state <= METADATA_START;
                        end else begin
                            state <= configuration_forward_postprocess_enable ?
                                FORWARD_POSTPROCESS_START : configuration_refresh_enable ? REFRESH_START : COMPLETE;
                        end
                    end else if (row_last_round && separate_l31_metadata && !l31_metadata_active) begin
                        // The single-batch scheduler has completed L0-L30 in place.
                        // Publish its hidden output before gathering the L31 layout.
                        v4_layer_index <= 6'd31;
                        layer_phase <= LAYER_PHASE_KV;
                        v4_output_row_base <= 12'd0;
                        state <= METADATA_START;
                    end else if (row_last_round) begin
                        completion_error <= 1'b0;
                        completion_error_id <= 16'd0;
                        state <= configuration_forward_postprocess_enable ?
                            FORWARD_POSTPROCESS_START : configuration_refresh_enable ? REFRESH_START : COMPLETE;
                    end else begin
                        state <= METADATA_START;
                    end
                end

                FORWARD_POSTPROCESS_START:
                    if (forward_postprocess_start_valid && forward_postprocess_start_ready)
                        state <= FORWARD_POSTPROCESS_WAIT;

                REFRESH_LAYER_START:
                    if (forward_postprocess_start_valid && forward_postprocess_start_ready)
                        state <= REFRESH_LAYER_WAIT;

                REFRESH_LAYER_WAIT: if (forward_postprocess_done_valid && forward_postprocess_done_ready) begin
                    if (forward_postprocess_done_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= {8'h30, forward_postprocess_done_error_id};
                        state <= COMPLETE;
                    end else state <= SCHEDULE_WAIT;
                end

                REFRESH_START:
                    if (refresh_address_sum[64] || joint_address_sum[64]) begin
                        completion_error <= 1'b1;
                        completion_error_id <= 16'h3001;
                        state <= COMPLETE;
                    end else if (forward_postprocess_start_valid && forward_postprocess_start_ready)
                        state <= REFRESH_WAIT;

                REFRESH_PREPARE_START:
                    if (refresh_address_sum[64] || joint_address_sum[64]) begin
                        completion_error <= 1'b1;
                        completion_error_id <= 16'h3001;
                        state <= COMPLETE;
                    end else if (forward_postprocess_start_valid && forward_postprocess_start_ready)
                        state <= REFRESH_PREPARE_WAIT;

                REFRESH_PREPARE_WAIT: if (forward_postprocess_done_valid && forward_postprocess_done_ready) begin
                    if (forward_postprocess_done_error) begin
                        completion_error <= 1'b1;
                        completion_error_id <= {8'h30, forward_postprocess_done_error_id};
                        state <= COMPLETE;
                    end else state <= METADATA_START;
                end

                FORWARD_POSTPROCESS_WAIT:
                    if (forward_postprocess_done_valid && forward_postprocess_done_ready) begin
                        refresh_closeout <= forward_postprocess_block_complete;
                        completion_error <= forward_postprocess_done_error;
                        completion_error_id <= forward_postprocess_done_error ?
                            {8'h20, forward_postprocess_done_error_id} : 16'd0;
                        state <= !forward_postprocess_done_error && configuration_refresh_enable ? REFRESH_START : COMPLETE;
                    end

                REFRESH_WAIT: if (forward_postprocess_done_valid && forward_postprocess_done_ready) begin
                    completion_error <= forward_postprocess_done_error;
                    completion_error_id <= forward_postprocess_done_error ? {8'h30, forward_postprocess_done_error_id} : 16'd0;
                    state <= COMPLETE;
                end

                COMPLETE: if (completion_valid && completion_ready)
                    state <= IDLE;

                default: state <= IDLE;
            endcase
            // A rejected layer entry has no accepted operator command to drain.
            if (layer_load_failed) begin
                layer_load_pending <= 1'b0;
                completion_error <= 1'b1;
                completion_error_id <= 16'h0400 | layer_done_error_id;
                state <= COMPLETE;
            end
        end
    end

`ifndef SYNTHESIS
    logic [48:0] stalled_completion;
    always_ff @(posedge clk) begin
        if (!rst) begin
            if ($past(completion_valid && !completion_ready))
                assert (completion_valid &&
                    {completion_id, completion_error, completion_error_id} == stalled_completion)
                    else $error("execution completion payload changed while stalled");
            if (completion_valid && !completion_ready)
                stalled_completion <= {completion_id, completion_error, completion_error_id};
            if (state != IDLE)
                assert (!launch_ready)
                    else $error("execution controller accepted a second command");
            if (hidden_start_valid && hidden_start_ready && hidden_start_write)
                assert (v4_output_row_base + {6'd0, row_active_rows} <=
                        row_total_rows)
                    else $error(
                        "execution controller V4 hidden output rows exceeded total base=%0d rows=%0d total=%0d",
                        v4_output_row_base, row_active_rows, row_total_rows);
            if (layer_load_failed)
                assert (state == SCHEDULE_WAIT && command_valid && !command_ready)
                    else $error("layer load failed after an operator command was accepted");
        end
    end
`endif
endmodule

`default_nettype wire
