`default_nettype none

package hardware_types_pkg;
    localparam int unsigned PE_TAG_WIDTH = 16;
    localparam int unsigned BF16_LANES = 64;
    localparam int unsigned SRAM_DATA_WIDTH = 128;
    localparam int unsigned SRAM_ADDRESS_WIDTH = 10;
    localparam int unsigned LOGICAL_DMA_TAG_WIDTH = 8;

    typedef enum logic [1:0] {
        PRECISION_W4A4 = 2'd0,
        PRECISION_W4A8 = 2'd1,
        PRECISION_W8A8 = 2'd2
    } precision_mode_t;

    typedef enum logic [1:0] {
        ATTENTION_READ_NONE = 2'd0,
        ATTENTION_READ_Q_MATMUL = 2'd1,
        ATTENTION_READ_CACHE = 2'd2
    } attention_read_source_t;

    typedef struct packed {
        precision_mode_t mode;
        logic [8*32*4-1:0] activation_payload;
        logic [8*32*4-1:0] weight_payload;
        logic [7:0] row_mask;
        logic [31:0] k_mask;
        logic [7:0] column_mask;
        logic first_k_step;
        logic last_k_step;
        logic mixed_phase;
        logic mixed_phase_first;
        logic [7:0] mixed_a8_rows;
        logic [8*16-1:0] activation_scales;
        logic [8*16-1:0] weight_scales;
        logic [PE_TAG_WIDTH-1:0] tag;
    } pe_request_t;

    typedef struct packed {
        logic [64*32-1:0] accumulators;
        logic [63:0] accumulator_mask;
        precision_mode_t mode;
        logic last_k_step;
        logic [8*16-1:0] activation_scales;
        logic [8*16-1:0] weight_scales;
        logic [PE_TAG_WIDTH-1:0] tag;
    } pe_response_t;

    typedef enum logic [2:0] {
        BF16_VECTOR_ADD = 3'd0,
        BF16_VECTOR_MULTIPLY = 3'd1,
        BF16_VECTOR_ROPE = 3'd2,
        BF16_VECTOR_SILU = 3'd3,
        BF16_VECTOR_MULTIPLY_ADD = 3'd4,
        BF16_VECTOR_RMSNORM = 3'd5
    } bf16_vector_operation_t;

    typedef struct packed {
        bf16_vector_operation_t operation;
        logic [BF16_LANES*16-1:0] values;
        logic [BF16_LANES*16-1:0] paired_values;
        logic [BF16_LANES*16-1:0] factor0_values;
        logic [BF16_LANES*16-1:0] factor1_values;
        logic [BF16_LANES-1:0] lane_mask;
        logic [PE_TAG_WIDTH-1:0] tag;
    } bf16_request_t;

    typedef struct packed {
        logic [BF16_LANES*16-1:0] values;
        logic [BF16_LANES-1:0] lane_mask;
        logic [PE_TAG_WIDTH-1:0] tag;
    } bf16_response_t;

    typedef struct packed {
        logic [1023:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } reduction_request_t;

    typedef struct packed {
        logic [127:0] values;
        logic [7:0] row_mask;
        logic [15:0] tag;
    } reduction_response_t;

    typedef struct packed {
        logic magnitude;
        logic [1023:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } maximum_request_t;

    typedef reduction_response_t maximum_response_t;

    typedef struct packed {
        logic [1023:0] delta_values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } softmax_exp_request_t;

    typedef struct packed {
        logic [1023:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } softmax_exp_response_t;

    typedef struct packed {
        logic [127:0] sum_values;
        logic [7:0] row_mask;
        logic [15:0] tag;
    } softmax_reciprocal_request_t;

    typedef struct packed {
        logic [127:0] values;
        logic [7:0] row_mask;
        logic [15:0] tag;
    } softmax_reciprocal_response_t;

    typedef struct packed {
        logic [2047:0] accumulators;
        logic [127:0] activation_scales;
        logic [127:0] weight_scales;
        logic [15:0] qk_scale_bf16;
        logic [1:0] rescale_mode;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } rescale_request_t;

    typedef struct packed {
        logic [1023:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } rescale_response_t;

    typedef struct packed {
        logic [15:0] clip_ratio_bf16;
        logic [7:0] a4_row_mask;
        logic use_static_scale;
        logic [127:0] static_scales_bf16;
        logic [127:0] row_max_abs;
        logic [7:0] row_mask;
    } quant_scale_request_t;

    typedef struct packed {
        logic [127:0] values_bf16;
    } quant_scale_response_t;

    typedef struct packed {
        logic [1023:0] values_bf16;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } quant_values_request_t;

    typedef struct packed {
        logic [511:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } quant_values_response_t;

    typedef struct packed {
        logic [1:0] mode;
        logic [5:0] physical_row_base;
        logic [5:0] row_count;
        logic [15:0] elements_per_row;
        logic [31:0] activation_base_byte_offset;
        logic [31:0] activation_limit_byte_offset;
        logic mixed_rows;
        logic row_partial;
        logic [1:0] segment_count;
        logic [5:0] segment_mode;
        logic [17:0] segment_row_base;
        logic [17:0] segment_row_count;
        logic mixed_group_enable;
        logic [5:0] compute_group_count;
        logic [47:0] row_precision_a8;
        logic [287:0] row_compute_group;
        logic [143:0] row_pe_slot;
        logic [95:0] row_phase_mask;
    } activation_writer_config_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [7:0] row_mask;
        logic [127:0] values_bf16;
    } activation_writer_scale_t;

    typedef struct packed {
        logic [5:0] physical_row_base;
        logic [15:0] element_base;
        logic [511:0] values;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } activation_writer_values_t;

    typedef struct packed {
        logic [5:0] row;
        logic [3:0] row_count;
        logic [15:0] element;
        logic [4:0] bank_base;
    } matmul_source_request_t;

    typedef struct packed {
        logic [1023:0] values;
        logic [63:0] lane_mask;
    } matmul_source_response_t;

    typedef struct packed {
        logic [7:0] slot_valid;
        logic [255:0] byte_address;
        logic [1023:0] data;
        logic [127:0] byte_enable;
    } matmul_activation_write_t;

    typedef struct packed {
        logic [15:0] activation_word_index;
        logic [3:0] activation_word_count;
        logic qkv_padded_layout;
        logic [5:0] physical_row_base;
        logic explicit_activation_rows;
        logic [8*6-1:0] activation_physical_row;
        logic [8*16-1:0] activation_lane_word_index;
        logic panel_select;
        logic [11:0] panel_port_mask;
        logic [119:0] panel_address;
    } matmul_read_request_t;

    typedef struct packed {
        logic [1023:0] activation_data;
        logic [1535:0] panel_data;
    } matmul_read_response_t;

    typedef struct packed {
        logic panel;
        logic [1:0] plane;
        logic [2:0] start_bank;
        logic [9:0] word_row;
        logic [255:0] data;
        logic [31:0] byte_enable;
    } matmul_panel_write_t;

    typedef struct packed {
        logic [9:0] address;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } local_128b_write_t;

    typedef struct packed {
        logic [9:0] address;
    } local_128b_read_request_t;

    typedef struct packed {
        logic [127:0] data;
    } local_128b_read_response_t;

    typedef struct packed {
        logic [31:0] byte_address;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } matmul_output_write_t;

    typedef struct packed {
        logic write;
        logic [14:0] word_index;
        logic [511:0] data;
        logic half;
        logic [15:0] tag;
    } elementwise_product_request_t;

    typedef struct packed {
        logic [511:0] data;
        logic half;
        logic [15:0] tag;
    } elementwise_product_response_t;

    typedef struct packed {
        logic [5:0] row;
        logic [8:0] channel_chunk;
    } matmul_residual_read_request_t;

    typedef struct packed {
        logic [9:0] word;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } rms_gamma_write_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [15:0] element;
        logic [15:0] tag;
    } rms_tile_read_request_t;

    typedef struct packed {
        logic [1023:0] values;
        logic [127:0] gamma;
        logic [63:0] lane_mask;
        logic [15:0] tag;
    } rms_tile_read_response_t;

    typedef struct packed {
        logic bank;
        logic [9:0] left_address;
        logic [9:0] right_address;
        logic [15:0] tag;
    } local_pair_read_request_t;

    typedef struct packed {
        logic [127:0] left_data;
        logic [127:0] right_data;
        logic [15:0] tag;
    } local_pair_read_response_t;

    typedef struct packed {
        logic bank;
        logic [9:0] address;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } local_banked_write_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [15:0] element;
        logic [1023:0] data;
        logic [63:0] lane_mask;
    } rms_norm_write_t;

    typedef struct packed {
        logic [5:0] physical_row;
        logic [8:0] channel_word;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } hidden_local_write_t;

    typedef struct packed {
        logic [5:0] physical_row;
        logic [8:0] channel_word;
    } hidden_local_read_request_t;

    typedef struct packed {
        logic [9:0] address;
        logic [15:0] tag;
    } tagged_local_read_request_t;

    typedef struct packed {
        logic [127:0] data;
        logic [15:0] tag;
    } tagged_local_read_response_t;

    typedef struct packed {
        logic rope_destination;
        logic [5:0] physical_row;
        logic [3:0] word;
    } qkv_head_read_request_t;

    typedef struct packed {
        logic [127:0] data;
        logic [7:0] lane_mask;
    } qkv_head_read_response_t;

    typedef struct packed {
        logic [287:0] lane_address;
    } rope_source_read_request_t;

    typedef struct packed {
        logic [15:0] lane_mask;
        logic [255:0] data;
    } rope_source_read_response_t;

    typedef struct packed {
        logic [5:0] physical_row;
        logic [6:0] pair_base;
    } rope_constant_read_request_t;

    typedef struct packed {
        logic [7:0] lane_mask;
        logic [127:0] data;
    } rope_constant_read_response_t;

    typedef struct packed {
        logic [287:0] lane_address;
        logic [255:0] lane_data;
    } rope_destination_write_t;

    typedef struct packed {
        logic source;
        logic [1:0] bank;
        logic port;
        logic [9:0] word;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } qkv_head_stage_write_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [3:0] word;
        logic [7:0] row_mask;
    } qkv_head_tile_read_request_t;

    typedef struct packed {
        logic [1023:0] data;
        logic [63:0] lane_mask;
    } qkv_head_tile_read_response_t;

    typedef struct packed {
        logic [9:0] word;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } qkv_constant_write_t;

    typedef struct packed {
        logic scale;
        logic slot;
        logic [5:0] physical_row;
        logic [9:0] word;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } qkv_q_write_t;

    typedef struct packed {
        logic target_panel;
        logic target_quarter;
        logic bank;
        logic [9:0] address;
        logic [255:0] data;
        logic [31:0] byte_enable;
    } attention_panel_write_t;

    typedef struct packed {
        logic target_panel;
        logic [10:0] token_base;
        logic [127:0] data;
        logic [15:0] byte_enable;
    } attention_panel_scale_write_t;

    typedef struct packed {
        logic is_pv;
        logic [5:0] query_base;
        logic [11:0] output_base;
        logic [11:0] k_base;
        logic [5:0] active_head;
        logic q_slot;
        logic active_panel;
        logic active_quarter;
        logic [11:0] sequence_length;
    } attention_operand_request_t;

    typedef struct packed {
        logic [1023:0] activation_rows;
        logic [511:0] panel_rows;
        logic [127:0] metadata;
        logic [127:0] panel_metadata;
    } attention_operand_response_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [11:0] key_base;
        logic [11:0] sequence_length;
        logic [1023:0] values;
        logic [63:0] lane_mask;
    } attention_score_write_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [11:0] key;
        logic [5:0] active_row_count;
        logic [11:0] sequence_length;
    } attention_score_read_request_t;

    typedef struct packed {
        logic [1023:0] values;
    } attention_score_read_response_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [11:0] key;
        logic [63:0] lane_mask;
        logic [1023:0] values;
    } attention_probability_write_t;

    typedef struct packed {
        logic [5:0] row_base;
        logic [11:0] key;
        logic [63:0] lane_mask;
        logic [511:0] values;
    } attention_probability_quantized_write_t;

    typedef struct packed {
        logic [5:0] head;
        logic [5:0] row_base;
        logic [7:0] row_mask;
        logic [127:0] values;
    } attention_probability_scale_write_t;

    typedef struct packed {
        logic [63:0] byte_address;
        logic [31:0] byte_count;
        logic [LOGICAL_DMA_TAG_WIDTH-1:0] tag;
        logic wide_data;
    } dma_read_request_t;

    typedef struct packed {
        logic [127:0] data;
        logic [15:0] byte_enable;
        logic last;
        logic [LOGICAL_DMA_TAG_WIDTH-1:0] tag;
    } dma_read_beat_t;

    typedef struct packed {
        logic [255:0] data;
        logic [31:0] byte_enable;
        logic last;
        logic [LOGICAL_DMA_TAG_WIDTH-1:0] tag;
    } dma_wide_read_beat_t;

    typedef struct packed {
        logic [63:0] byte_address;
        logic [31:0] byte_count;
        logic [LOGICAL_DMA_TAG_WIDTH-1:0] tag;
    } dma_write_request_t;

    typedef struct packed {
        logic [127:0] data;
        logic [15:0] byte_enable;
        logic last;
    } dma_write_beat_t;

    typedef struct packed {
        logic done_pulse;
        logic error;
    } dma_completion_t;

    typedef struct packed {
        logic [15:0] query_clip_ratio_bf16;
        logic [15:0] key_clip_ratio_bf16;
        logic [15:0] value_clip_ratio_bf16;
        logic [15:0] attention_output_clip_ratio_bf16;
        logic [15:0] ffn_gate_clip_ratio_bf16;
        logic [15:0] ffn_up_clip_ratio_bf16;
        logic [15:0] ffn_down_clip_ratio_bf16;
        logic [63:0] query_base;
        logic [63:0] query_enhancement;
        logic [63:0] query_scale;
        logic [63:0] key_base;
        logic [63:0] key_enhancement;
        logic [63:0] key_scale;
        logic [63:0] value_base;
        logic [63:0] value_enhancement;
        logic [63:0] value_scale;
        logic [63:0] attention_output_base;
        logic [63:0] attention_output_enhancement;
        logic [63:0] attention_output_scale;
        logic [63:0] ffn_gate_base;
        logic [63:0] ffn_gate_enhancement;
        logic [63:0] ffn_gate_scale;
        logic [63:0] ffn_up_base;
        logic [63:0] ffn_up_enhancement;
        logic [63:0] ffn_up_scale;
        logic [63:0] ffn_down_base;
        logic [63:0] ffn_down_enhancement;
        logic [63:0] ffn_down_scale;
    } matmul_weight_config_t;

    typedef struct packed {
        logic [15:0] attention_epsilon_bf16;
        logic [15:0] ffn_epsilon_bf16;
        logic [63:0] attention_gamma_base;
        logic [63:0] ffn_gamma_base;
    } rmsnorm_layer_config_t;

    typedef struct packed {
        logic v_scale_per_head;
        logic [63:0] rope_cos_base;
        logic [63:0] rope_sin_base;
        logic [63:0] v_scale_base;
    } qkv_layer_config_t;

    typedef struct packed {
        logic [5:0] row_count;
        logic [11:0] sequence_length;
        logic [15:0] token_batch_index;
        logic [15:0] first_token_ordinal;
        logic [31:0] metadata_version;
        logic [31:0] capture_index;
    } token_metadata_config_t;

    typedef struct packed {
        logic [5:0] index;
        logic [16:0] source_index;
        logic [10:0] token_position;
        logic [10:0] kv_index;
        logic embedding_source;
        logic kv_write_disable;
        logic [3:0] bits;
        logic [7:0] query_group;
        logic [7:0] cache_group;
    } token_metadata_row_t;

    typedef struct packed {
        logic [63:0] retained_k_scale_base;
        logic [63:0] current_k_base;
        logic [63:0] current_v_base;
        logic [63:0] retained_k_base;
        logic [63:0] retained_v_base;
        logic [63:0] k_scale_base;
        logic [31:0] head_stride_bytes;
        logic [31:0] token_stride_bytes;
        logic [31:0] k_scale_head_stride_bytes;
    } attention_cache_config_t;

    typedef struct packed {
        logic [63:0] context_base;
        logic [31:0] context_row_stride_bytes;
    } attention_context_config_t;

    typedef struct packed {
        logic enable;
        logic capture_p8;
        logic [11:0] query_begin;
        logic [11:0] query_end;
        logic [255:0] key_groups;
        logic [63:0] output_base;
        logic [63:0] output_limit;
        logic [31:0] head_stride_bytes;
        logic [31:0] batch_stride_bytes;
        logic [31:0] round_stride_bytes;
        logic [31:0] layer_mask;
    } attention_probability_config_t;

    typedef struct packed {
        logic [63:0] gate_up_base;
        logic [63:0] gate_up_limit;
        logic [63:0] residual_base;
        logic [63:0] residual_limit;
    } ffn_workspace_config_t;

    typedef enum logic [2:0] {
        MATMUL_COMMAND_QKV = 3'd0,
        MATMUL_COMMAND_ATTENTION_OUTPUT = 3'd1,
        MATMUL_COMMAND_FFN_GATE = 3'd2,
        MATMUL_COMMAND_FFN_UP = 3'd3,
        MATMUL_COMMAND_FFN_DOWN = 3'd4
    } matmul_command_t;

    typedef struct packed {
        logic [5:0] segment_mode;
        logic [17:0] segment_row_base;
        logic [17:0] segment_row_count;
        logic [47:0] row_enable;
        logic operator_has_w8;
        logic mixed_group_enable;
        logic [5:0] compute_group_count;
        logic [47:0] row_precision_a8;
        logic [287:0] row_compute_group;
        logic [143:0] row_pe_slot;
        logic [95:0] row_phase_mask;
    } matmul_row_config_t;

    typedef struct packed {
        logic prepare_second;
        logic enable;
        logic qkvo_group;
        logic [2:0] active_batch_index;
        logic [2:0] batch_count;
        logic compact_a4_layout;
        logic fused_product;
        logic [31:0] second_activation_base_byte_offset;
        logic [5:0] second_rows;
        matmul_row_config_t second_row_config;
        logic [5:0] third_rows;
        matmul_row_config_t third_row_config;
        logic [5:0] fourth_rows;
        matmul_row_config_t fourth_row_config;
        logic [5:0] fifth_rows;
        matmul_row_config_t fifth_row_config;
        logic [5:0] sixth_rows;
        matmul_row_config_t sixth_row_config;
        logic [63:0] normalized_base;
        logic [63:0] normalized_limit;
        logic [63:0] qkv_staging_base;
        logic [63:0] qkv_staging_limit;
        logic [31:0] qkv_staging_head_stride;
    } ffn_batch_pair_config_t;

    function automatic logic same_matmul_activation_group(
        input ffn_batch_pair_config_t left,
        input ffn_batch_pair_config_t right
    );
        ffn_batch_pair_config_t left_activation;
        ffn_batch_pair_config_t right_activation;
        begin
            left_activation = left;
            right_activation = right;
            left_activation.qkv_staging_base = '0;
            left_activation.qkv_staging_limit = '0;
            left_activation.qkv_staging_head_stride = '0;
            right_activation.qkv_staging_base = '0;
            right_activation.qkv_staging_limit = '0;
            right_activation.qkv_staging_head_stride = '0;
            same_matmul_activation_group =
                left_activation == right_activation;
        end
    endfunction

    typedef struct packed {
        logic [5:0] layer;
        logic [3:0] layer_operator_index;
        logic cache_release;
        logic qkv_residual_spill;
        logic rms_qkv_prefetch_q;
        logic q_head_preprocess;
    } operator_command_t;

    typedef struct packed {
        logic [15:0] operator_tag;
        logic error;
        logic [15:0] error_id;
    } operator_completion_t;

    typedef struct packed {
        logic [1:0] segment_count;
        logic [5:0] segment_mode;
        logic [17:0] segment_row_base;
        logic [17:0] segment_row_count;
        logic [95:0] segment_activation_base_byte_offset;
        logic [47:0] row_enable;
    } activation_layout_t;

    typedef struct packed {
        logic [287:0] physical_to_local;
    } hidden_row_config_t;

    typedef struct packed {
        logic [47:0] kv_write_disable;
        logic [527:0] token_position;
        logic [47:0] row_enable;
        logic [5:0] token_batch_index;
    } attention_row_config_t;

    typedef struct packed {
        logic reuse_group_activation;
        logic enable;
        logic [2:0] active_batch_index;
        logic [2:0] batch_count;
        logic [35:0] row_count;
        logic [287:0] row_enable;
        logic [287:0] kv_write_disable;
        logic [3167:0] token_position;
        logic [63:0] staging_base;
        logic [63:0] staging_limit;
        logic [31:0] staging_head_stride;
    } attention_batch_group_config_t;
endpackage

`default_nettype wire
