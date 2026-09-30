`default_nettype none

// Loads and validates one layer-address entry. A validated entry remains
// available until another layer is requested or a new command invalidates it.
module layer_address_loader (
    input  logic          clk,
    input  logic          rst,
    input  logic          entry_invalidate,
    input  logic          abort_request,
    output logic          abort_ack,

    input  logic          load_valid,
    output logic          load_ready,
    input  logic [4:0]    load_layer_index,
    input  logic [2559:0] execution_configuration_bits,

    output logic          dma_request_valid,
    input  logic          dma_request_ready,
    output logic [63:0]   dma_request_address,
    output logic [31:0]   dma_request_bytes,
    input  logic          dma_response_valid,
    output logic          dma_response_ready,
    input  logic [127:0]  dma_response_data,
    input  logic [15:0]   dma_response_byte_enable,
    input  logic          dma_response_last,
    input  logic          dma_request_done,
    input  logic          dma_request_error,

    output logic          done_valid,
    input  logic          done_ready,
    output logic          done_error,
    output logic [15:0]   done_error_id,
    output logic          entry_valid,
    output logic [4:0]    entry_layer_index,
    output hardware_types_pkg::matmul_weight_config_t matmul_weight_config,
    output hardware_types_pkg::rmsnorm_layer_config_t rmsnorm_config,
    output hardware_types_pkg::qkv_layer_config_t qkv_config,
    output hardware_types_pkg::attention_cache_config_t attention_cache_config,
    output hardware_types_pkg::attention_context_config_t attention_context_config,
    output hardware_types_pkg::ffn_workspace_config_t ffn_workspace_config
);
    import execution_config_pkg::*;
    import layer_address_table_pkg::*;

    localparam logic [15:0] ERROR_NONE = 16'h0000;
    localparam logic [15:0] ERROR_ADDRESS = 16'h0001;
    localparam logic [15:0] ERROR_DMA = 16'h0002;
    localparam logic [15:0] ERROR_STREAM = 16'h0003;
    localparam logic [15:0] ERROR_HEADER = 16'h0004;
    localparam logic [15:0] ERROR_REGION = 16'h0005;
    localparam logic [15:0] ERROR_CONTAINMENT = 16'h0006;

    typedef enum logic [3:0] {
        IDLE,
        CALCULATE_ADDRESS,
        REQUEST,
        STREAM,
        CHECK_HEADER,
        CHECK_SHAPE,
        CHECK_RETAINED_SCALE,
        CHECK_GATE_UP_WORKSPACE,
        CHECK_RESIDUAL_WORKSPACE,
        CHECK_WORKSPACE_RELATION,
        COMPLETE
    } state_t;

    state_t state;
    logic [64:0] address_accumulator;
    logic [4:0] address_additions_remaining;
    logic [63:0] saved_table_base;
    logic [63:0] saved_table_limit;
    logic [31:0] saved_entry_stride;
    logic [6:0] response_count;
    logic [63:0] pending_region_base;
    logic response_last_seen;
    logic stream_error_seen;
    logic region_error_seen;
    logic [15:0] region_error_id;
    logic abort_pending;
    logic abort_ack_sent;

    logic [63:0] current_k_base, current_k_limit;
    logic [63:0] current_v_base, current_v_limit;
    logic [63:0] retained_k_base, retained_k_limit;
    logic [63:0] retained_v_base, retained_v_limit;
    logic [63:0] k_scale_base, k_scale_limit;
    logic [63:0] retained_k_scale_base, retained_k_scale_limit;
    logic [64:0] retained_k_scale_required_end;
    logic [63:0] v_scale_base, v_scale_limit;
    logic [63:0] attention_workspace_base, attention_workspace_limit;
    logic [63:0] bf16_temporary_base, bf16_temporary_limit;
    logic [5:0] saved_active_rows;
    logic saved_allow_bf16_temporary_spill;
    logic [4991:0] entry_image;

    logic response_fire;
    logic response_format_error;
    logic region_check_active;
    logic [5:0] checked_region_index;
    logic [63:0] checked_region_base;
    logic [63:0] checked_region_limit;
    logic containment_check_active;
    logic [63:0] containing_region_base;
    logic [63:0] containing_region_limit;
    logic checked_region_format_error;
    logic checked_region_containment_error;
    logic [64:0] entry_end_address;
    logic [64:0] gate_up_required_bytes;
    logic [6:0] gate_up_layout_rows;
    logic [64:0] residual_required_bytes;
    logic [64:0] gate_up_required_end;
    logic [64:0] residual_required_end;
    logic configuration_is_v4;
    logic [15:0] configuration_start_layer;
    logic [16:0] configuration_layer_end;

    function automatic logic [15:0] execution_u16(
        input logic [2559:0] configuration_bits,
        input integer byte_offset
    );
        execution_u16 = configuration_bits[byte_offset*8 +: 16];
    endfunction

    function automatic logic partial_cache_overlap(
        input logic [63:0] first_base, first_limit, second_base, second_limit
    );
        partial_cache_overlap = first_base < first_limit && second_base < second_limit &&
            first_base < second_limit && second_base < first_limit &&
            !(first_base == second_base && first_limit == second_limit);
    endfunction

    function automatic logic [31:0] execution_u32(
        input logic [2559:0] configuration_bits,
        input integer byte_offset
    );
        execution_u32 = configuration_bits[byte_offset*8 +: 32];
    endfunction

    function automatic logic [63:0] execution_u64(
        input logic [2559:0] configuration_bits,
        input integer byte_offset
    );
        execution_u64 = configuration_bits[byte_offset*8 +: 64];
    endfunction

    assign load_ready = state == IDLE && !abort_request && !entry_invalidate;
    assign dma_request_valid = state == REQUEST && !abort_request;
    assign dma_request_bytes = LAYER_ADDRESS_TABLE_BYTES;
    assign dma_response_ready = state == STREAM;
    assign done_valid = state == COMPLETE && !abort_request;
    assign response_fire = dma_response_valid && dma_response_ready;
    assign response_format_error = response_fire &&
        (dma_response_byte_enable != 16'hffff ||
         dma_response_last != (response_count == 7'd38));
    assign entry_end_address = address_accumulator + 65'(LAYER_ADDRESS_TABLE_BYTES);
    assign configuration_start_layer = execution_u16(execution_configuration_bits, EXECUTION_CONFIG_START_LAYER_OFFSET);
    assign configuration_layer_end = {1'b0, configuration_start_layer} +
        {1'b0, execution_u16(execution_configuration_bits, EXECUTION_CONFIG_LAYER_COUNT_OFFSET)};
    assign gate_up_layout_rows =
        {1'b0, saved_active_rows} + {6'd0, saved_active_rows[0]};
    assign gate_up_required_bytes =
        ({58'd0, gate_up_layout_rows} << 15) +
        ({58'd0, gate_up_layout_rows} << 14);
    assign residual_required_bytes = {59'd0, saved_active_rows} << 13;
    assign gate_up_required_end =
        {1'b0, entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET*8 +: 64]} +
        gate_up_required_bytes;
    assign residual_required_end =
        {1'b0, entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET*8 +: 64]} +
        residual_required_bytes;
    assign configuration_is_v4 =
        execution_u32(execution_configuration_bits,
            EXECUTION_CONFIG_MAGIC_OFFSET) == 32'h344e4c44 &&
        execution_u16(execution_configuration_bits,
            EXECUTION_CONFIG_VERSION_OFFSET) == 16'd4;

    assign matmul_weight_config.query_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_QUERY_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.key_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_KEY_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.value_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_VALUE_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.attention_output_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.ffn_gate_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.ffn_up_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_FFN_UP_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    assign matmul_weight_config.ffn_down_clip_ratio_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_FFN_DOWN_CLIP_RATIO_BF16_OFFSET*8 +: 16];
    logic clipping_header_error;
    always_comb begin
        clipping_header_error =
            matmul_weight_config.query_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.key_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.value_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.attention_output_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.ffn_gate_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.ffn_up_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.ffn_down_clip_ratio_bf16 > 16'h3f80 ||
            matmul_weight_config.query_clip_ratio_bf16 != matmul_weight_config.key_clip_ratio_bf16 ||
            matmul_weight_config.query_clip_ratio_bf16 != matmul_weight_config.value_clip_ratio_bf16 ||
            matmul_weight_config.ffn_gate_clip_ratio_bf16 != matmul_weight_config.ffn_up_clip_ratio_bf16;
        if (!entry_image[LAYER_ADDRESS_TABLE_FLAGS_OFFSET*8])
            clipping_header_error = clipping_header_error ||
                |{matmul_weight_config.query_clip_ratio_bf16,
                    matmul_weight_config.attention_output_clip_ratio_bf16,
                    matmul_weight_config.ffn_gate_clip_ratio_bf16,
                    matmul_weight_config.ffn_down_clip_ratio_bf16};
    end

    assign matmul_weight_config.query_base =
        entry_image[LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.query_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.query_scale =
        entry_image[LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.key_base =
        entry_image[LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.key_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.key_scale =
        entry_image[LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.value_base =
        entry_image[LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.value_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.value_scale =
        entry_image[LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.attention_output_base =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.attention_output_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.attention_output_scale =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_gate_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_gate_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_gate_scale =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_up_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_up_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_up_scale =
        entry_image[LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_down_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_down_enhancement =
        entry_image[LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_BASE_OFFSET*8 +: 64];
    assign matmul_weight_config.ffn_down_scale =
        entry_image[LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_BASE_OFFSET*8 +: 64];

    assign rmsnorm_config.attention_epsilon_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_RMS_EPSILON_BF16_OFFSET*8 +: 16];
    assign rmsnorm_config.ffn_epsilon_bf16 =
        entry_image[LAYER_ADDRESS_TABLE_FFN_RMS_EPSILON_BF16_OFFSET*8 +: 16];
    assign rmsnorm_config.attention_gamma_base =
        entry_image[LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_BASE_OFFSET*8 +: 64];
    assign rmsnorm_config.ffn_gamma_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_BASE_OFFSET*8 +: 64];

    assign qkv_config.rope_cos_base =
        entry_image[LAYER_ADDRESS_TABLE_ROPE_COS_LUT_BASE_OFFSET*8 +: 64];
    assign qkv_config.rope_sin_base =
        entry_image[LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_BASE_OFFSET*8 +: 64];
    assign qkv_config.v_scale_base =
        entry_image[LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET*8 +: 64];
    assign qkv_config.v_scale_per_head = 1'b1;

    assign attention_cache_config.current_k_base =
        entry_image[LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.current_v_base =
        entry_image[LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.retained_k_base =
        entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.retained_v_base =
        entry_image[LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.k_scale_base =
        entry_image[LAYER_ADDRESS_TABLE_K_SCALE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.retained_k_scale_base =
        entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET*8 +: 64];
    assign attention_cache_config.head_stride_bytes =
        entry_image[LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET*8 +: 32];
    assign attention_cache_config.token_stride_bytes =
        entry_image[LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET*8 +: 32];
    assign attention_cache_config.k_scale_head_stride_bytes =
        entry_image[LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET*8 +: 32];

    assign attention_context_config.context_base =
        entry_image[LAYER_ADDRESS_TABLE_CONTEXT_BASE_OFFSET*8 +: 64];
    assign attention_context_config.context_row_stride_bytes =
        entry_image[LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET*8 +: 32];

    assign ffn_workspace_config.gate_up_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET*8 +: 64];
    assign ffn_workspace_config.gate_up_limit =
        entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_LIMIT_OFFSET*8 +: 64];
    assign ffn_workspace_config.residual_base =
        entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET*8 +: 64];
    assign ffn_workspace_config.residual_limit =
        entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_LIMIT_OFFSET*8 +: 64];

    // Regions 0..22 are naturally aligned to a 16-byte response beat. The
    // epsilon fields shift regions 23..33 by eight bytes, so their base is
    // retained from the preceding beat and paired with the next low 64 bits.
    always_comb begin
        region_check_active = 1'b0;
        checked_region_index = 6'd0;
        checked_region_base = 64'd0;
        checked_region_limit = 64'd0;
        if (response_fire && response_count >= 7'd4 && response_count <= 7'd26) begin
            region_check_active = 1'b1;
            checked_region_index = response_count[5:0] - 6'd4;
            checked_region_base = dma_response_data[63:0];
            checked_region_limit = dma_response_data[127:64];
        end else if (response_fire && response_count >= 7'd28 &&
                     response_count <= 7'd38) begin
            region_check_active = 1'b1;
            checked_region_index = response_count[5:0] - 6'd5;
            checked_region_base = pending_region_base;
            checked_region_limit = dma_response_data[63:0];
        end
    end

    always_comb begin
        containment_check_active = 1'b0;
        containing_region_base = 64'd0;
        containing_region_limit = 64'd0;
        case (checked_region_index)
            6'd25: begin
                containment_check_active = 1'b1;
                containing_region_base = current_k_base;
                containing_region_limit = current_k_limit;
            end
            6'd26: begin
                containment_check_active = 1'b1;
                containing_region_base = current_v_base;
                containing_region_limit = current_v_limit;
            end
            6'd27: begin
                containment_check_active = 1'b1;
                containing_region_base = retained_k_base;
                containing_region_limit = retained_k_limit;
            end
            6'd28: begin
                containment_check_active = 1'b1;
                containing_region_base = retained_v_base;
                containing_region_limit = retained_v_limit;
            end
            6'd29: begin
                containment_check_active = 1'b1;
                containing_region_base = k_scale_base;
                containing_region_limit = k_scale_limit;
            end
            6'd30: begin
                containment_check_active = 1'b1;
                containing_region_base = v_scale_base;
                containing_region_limit = v_scale_limit;
            end
            6'd31: begin
                containment_check_active = 1'b1;
                containing_region_base = attention_workspace_base;
                containing_region_limit = attention_workspace_limit;
            end
            6'd32, 6'd33: begin
                containment_check_active = 1'b1;
                containing_region_base = bf16_temporary_base;
                containing_region_limit = bf16_temporary_limit;
            end
            default: begin
                containment_check_active = 1'b0;
            end
        endcase
    end

    assign checked_region_format_error = region_check_active &&
        (checked_region_base[3:0] != 0 || checked_region_limit[3:0] != 0 ||
         checked_region_base > checked_region_limit);
    assign checked_region_containment_error = region_check_active &&
        containment_check_active &&
        (checked_region_base < containing_region_base ||
         checked_region_limit > containing_region_limit);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            address_accumulator <= 65'd0;
            address_additions_remaining <= 5'd0;
            saved_table_base <= 64'd0;
            saved_table_limit <= 64'd0;
            saved_entry_stride <= 32'd0;
            response_count <= 7'd0;
            pending_region_base <= 64'd0;
            response_last_seen <= 1'b0;
            stream_error_seen <= 1'b0;
            region_error_seen <= 1'b0;
            region_error_id <= ERROR_NONE;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            abort_ack_sent <= 1'b0;
            done_error <= 1'b0;
            done_error_id <= ERROR_NONE;
            entry_valid <= 1'b0;
            entry_layer_index <= 5'd0;
            saved_active_rows <= 6'd0;
            saved_allow_bf16_temporary_spill <= 1'b0;
            retained_k_scale_base <= 64'd0;
            retained_k_scale_limit <= 64'd0;
            retained_k_scale_required_end <= 65'd0;
        end else begin
            abort_ack <= 1'b0;
            if (!abort_request)
                abort_ack_sent <= 1'b0;
            if (entry_invalidate)
                entry_valid <= 1'b0;

            if (abort_request && state == IDLE && !abort_ack_sent) begin
                abort_ack <= 1'b1;
                abort_ack_sent <= 1'b1;
            end

            case (state)
                IDLE: if (load_valid && load_ready) begin
                    done_error <= 1'b0;
                    done_error_id <= ERROR_NONE;
                    abort_pending <= 1'b0;
                    if (!configuration_is_v4) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_HEADER;
                        state <= COMPLETE;
                    end else if (entry_valid && entry_layer_index == load_layer_index) begin
                        state <= COMPLETE;
                    end else begin
                        entry_valid <= 1'b0;
                        saved_table_limit <= execution_u64(
                            execution_configuration_bits,
                            EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET);
                        saved_table_base <= execution_u64(
                            execution_configuration_bits,
                            EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET);
                        saved_entry_stride <= execution_u32(
                            execution_configuration_bits,
                            EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET);
                        saved_active_rows <= 6'd48;
                        saved_allow_bf16_temporary_spill <=
                            (execution_u32(
                            execution_configuration_bits,
                            EXECUTION_CONFIG_FLAGS_OFFSET) & 32'd1) != 0;
                        address_accumulator <= {1'b0, execution_u64(
                            execution_configuration_bits,
                            EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET)};
                        address_additions_remaining <= load_layer_index;
                        entry_layer_index <= load_layer_index;
                        current_k_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET);
                        current_k_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_CURRENT_K_CACHE_LIMIT_OFFSET);
                        current_v_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET);
                        current_v_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_CURRENT_V_CACHE_LIMIT_OFFSET);
                        retained_k_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET);
                        retained_k_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_K_CACHE_LIMIT_OFFSET);
                        retained_v_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET);
                        retained_v_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_V_CACHE_LIMIT_OFFSET);
                        k_scale_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_K_SCALE_BASE_OFFSET);
                        k_scale_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_K_SCALE_LIMIT_OFFSET);
                        retained_k_scale_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET);
                        retained_k_scale_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET);
                        v_scale_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_V_SCALE_BASE_OFFSET);
                        v_scale_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_V_SCALE_LIMIT_OFFSET);
                        attention_workspace_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET);
                        attention_workspace_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET);
                        bf16_temporary_base <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET);
                        bf16_temporary_limit <= execution_u64(execution_configuration_bits,
                            EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET);

                        if (execution_u16(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_COUNT_OFFSET) < 1 ||
                            configuration_layer_end > 17'd32 ||
                            {11'd0, load_layer_index} < configuration_start_layer ||
                            {12'd0, load_layer_index} >= configuration_layer_end ||
                            (execution_u64(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET) & 64'hf) != 0 ||
                            (execution_u64(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET) & 64'hf) != 0 ||
                            execution_u64(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET) >
                            execution_u64(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET) ||
                            execution_u32(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET) <
                                32'd880 ||
                            (execution_u32(execution_configuration_bits,
                                EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET) & 32'hf) != 0) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_ADDRESS;
                            state <= COMPLETE;
                        end else begin
                            state <= CALCULATE_ADDRESS;
                        end
                    end
                end

                CALCULATE_ADDRESS: begin
                    if (abort_request) begin
                        if (!abort_ack_sent) begin
                            abort_ack <= 1'b1;
                            abort_ack_sent <= 1'b1;
                        end
                        state <= IDLE;
                    end else if (address_additions_remaining != 0) begin
                        address_accumulator <= address_accumulator +
                            {33'd0, saved_entry_stride};
                        address_additions_remaining <= address_additions_remaining - 1'b1;
                    end else if (address_accumulator[64] || entry_end_address[64] ||
                                 address_accumulator[63:0] < saved_table_base ||
                                 entry_end_address[63:0] > saved_table_limit) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_ADDRESS;
                        state <= COMPLETE;
                    end else begin
                        response_count <= 7'd0;
                        response_last_seen <= 1'b0;
                        stream_error_seen <= 1'b0;
                        region_error_seen <= 1'b0;
                        region_error_id <= ERROR_NONE;
                        pending_region_base <= 64'd0;
                        state <= REQUEST;
                    end
                end

                REQUEST: begin
                    if (abort_request) begin
                        if (!abort_ack_sent) begin
                            abort_ack <= 1'b1;
                            abort_ack_sent <= 1'b1;
                        end
                        state <= IDLE;
                    end else if (dma_request_valid && dma_request_ready) begin
                        state <= STREAM;
                    end
                end

                STREAM: begin
                    if (abort_request)
                        abort_pending <= 1'b1;
                    if (response_fire) begin
                        for (integer entry_beat = 0; entry_beat < 39;
                             entry_beat = entry_beat + 1)
                            if (response_count == 7'(entry_beat))
                                entry_image[entry_beat*128 +: 128] <=
                                    dma_response_data;
                        if (response_format_error)
                            stream_error_seen <= 1'b1;
                        if (dma_response_last)
                            response_last_seen <= 1'b1;
                        if (response_count != 7'h7f)
                            response_count <= response_count + 1'b1;
                        if (response_count == 7'd27 ||
                            (response_count >= 7'd28 && response_count < 7'd38))
                            pending_region_base <= dma_response_data[127:64];
                        if (!region_error_seen && checked_region_format_error) begin
                            region_error_seen <= 1'b1;
                            region_error_id <= ERROR_REGION;
                        end else if (!region_error_seen &&
                                     checked_region_containment_error) begin
                            region_error_seen <= 1'b1;
                            region_error_id <= ERROR_CONTAINMENT;
                        end
                    end

                    if (dma_request_done) begin
                        if (abort_pending || abort_request) begin
                            if (!abort_ack_sent) begin
                                abort_ack <= 1'b1;
                                abort_ack_sent <= 1'b1;
                            end
                            abort_pending <= 1'b0;
                            state <= IDLE;
                        end else if (dma_request_error) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_DMA;
                            state <= COMPLETE;
                        end else if (stream_error_seen || response_format_error ||
                                     !(response_last_seen ||
                                       (response_fire && dma_response_last)) ||
                                     !((response_count == 7'd39) ||
                                       (response_fire && response_count == 7'd38))) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_STREAM;
                            state <= COMPLETE;
                        end else if (region_error_seen || checked_region_format_error ||
                                     checked_region_containment_error) begin
                            done_error <= 1'b1;
                            if (region_error_seen)
                                done_error_id <= region_error_id;
                            else if (checked_region_format_error)
                                done_error_id <= ERROR_REGION;
                            else
                                done_error_id <= ERROR_CONTAINMENT;
                            state <= COMPLETE;
                        end else begin
                            state <= CHECK_HEADER;
                        end
                    end
                end

                CHECK_HEADER: begin
                    if (entry_image[LAYER_ADDRESS_TABLE_MAGIC_OFFSET*8 +: 32] != 32'h3141_544c ||
                        entry_image[LAYER_ADDRESS_TABLE_VERSION_OFFSET*8 +: 16] != 16'd2 ||
                        entry_image[LAYER_ADDRESS_TABLE_HEADER_BYTES_OFFSET*8 +: 16] != 16'd64 ||
                        entry_image[LAYER_ADDRESS_TABLE_ENTRY_BYTES_OFFSET*8 +: 32] !=
                            LAYER_ADDRESS_TABLE_BYTES ||
                        entry_image[LAYER_ADDRESS_TABLE_FLAGS_OFFSET*8+1 +: 31] != 0 ||
                        clipping_header_error) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_HEADER;
                        state <= COMPLETE;
                    end else begin
                        state <= CHECK_SHAPE;
                    end
                end

                CHECK_SHAPE: begin
                    if (entry_image[LAYER_ADDRESS_TABLE_HIDDEN_SIZE_OFFSET*8 +: 32] != 32'd4096 ||
                        entry_image[LAYER_ADDRESS_TABLE_FFN_SIZE_OFFSET*8 +: 32] != 32'd12288 ||
                        entry_image[LAYER_ADDRESS_TABLE_HEAD_COUNT_OFFSET*8 +: 16] != 16'd32 ||
                        entry_image[LAYER_ADDRESS_TABLE_HEAD_DIMENSION_OFFSET*8 +: 16] != 16'd128 ||
                        entry_image[LAYER_ADDRESS_TABLE_MAXIMUM_SEQUENCE_LENGTH_OFFSET*8 +: 16] != 16'd2048 ||
                        entry_image[LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET*8 +: 32] < 32'd262144 ||
                        (entry_image[LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET*8 +: 32] & 32'hf) != 0 ||
                        entry_image[LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET*8 +: 32] != 32'd128 ||
                        entry_image[LAYER_ADDRESS_TABLE_KV_CHUNK_STRIDE_BYTES_OFFSET*8 +: 32] != 32'd8 ||
                        entry_image[LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET*8 +: 32] < 32'd4096 ||
                        (entry_image[LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET*8 +: 32] & 32'hf) != 0 ||
                        entry_image[LAYER_ADDRESS_TABLE_K_SCALE_TOKEN_STRIDE_BYTES_OFFSET*8 +: 32] != 32'd2 ||
                        entry_image[LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET*8 +: 32] < 32'd8192 ||
                        (entry_image[LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET*8 +: 32] & 32'hf) != 0 ||
                        saved_active_rows < 1 ||
                        saved_active_rows > 6'd48 ||
                        !saved_allow_bf16_temporary_spill) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_HEADER;
                        state <= COMPLETE;
                    end else if (entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET*8 +: 64] != 0) begin
                        retained_k_scale_required_end <=
                            {1'b0, entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET*8 +: 64]} +
                            {28'd0, entry_image[LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET*8 +: 32], 5'd0};
                        state <= CHECK_RETAINED_SCALE;
                    end else begin
                        state <= CHECK_GATE_UP_WORKSPACE;
                    end
                end

                CHECK_RETAINED_SCALE: begin
                    if (entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET*8 +: 4] != 0 ||
                        retained_k_scale_base[3:0] != 0 || retained_k_scale_limit[3:0] != 0 ||
                        entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET*8 +: 64] < retained_k_scale_base ||
                        retained_k_scale_required_end > {1'b0, retained_k_scale_limit}) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_CONTAINMENT;
                        state <= COMPLETE;
                    end else begin
                        state <= CHECK_GATE_UP_WORKSPACE;
                    end
                end

                CHECK_GATE_UP_WORKSPACE: begin
                    if (gate_up_required_end[64] ||
                        gate_up_required_end[63:0] >
                            entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_LIMIT_OFFSET*8 +: 64]) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_CONTAINMENT;
                        state <= COMPLETE;
                    end else begin
                        state <= CHECK_RESIDUAL_WORKSPACE;
                    end
                end

                CHECK_RESIDUAL_WORKSPACE: begin
                    if (residual_required_end[64] ||
                        residual_required_end[63:0] >
                            entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_LIMIT_OFFSET*8 +: 64]) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_CONTAINMENT;
                        state <= COMPLETE;
                    end else begin
                        state <= CHECK_WORKSPACE_RELATION;
                    end
                end

                CHECK_WORKSPACE_RELATION: begin
                    if (partial_cache_overlap(
                            entry_image[LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_LIMIT_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_LIMIT_OFFSET*8 +: 64]) ||
                        partial_cache_overlap(
                            entry_image[LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_LIMIT_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET*8 +: 64],
                            entry_image[LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_LIMIT_OFFSET*8 +: 64]) ||
                        !(entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_LIMIT_OFFSET*8 +: 64] <=
                              entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET*8 +: 64] ||
                          entry_image[LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_LIMIT_OFFSET*8 +: 64] <=
                              entry_image[LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET*8 +: 64])) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_CONTAINMENT;
                    end else begin
                        entry_valid <= 1'b1;
                    end
                    state <= COMPLETE;
                end

                COMPLETE: begin
                    if (abort_request) begin
                        if (!abort_ack_sent) begin
                            abort_ack <= 1'b1;
                            abort_ack_sent <= 1'b1;
                        end
                        state <= IDLE;
                    end else if (done_valid && done_ready) begin
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign dma_request_address = address_accumulator[63:0];

`ifndef SYNTHESIS
    logic [95:0] stalled_request;
    logic [16:0] stalled_completion;
    always_ff @(posedge clk) begin
        if (!rst) begin
            if ($past(dma_request_valid && !dma_request_ready))
                assert (dma_request_valid &&
                        {dma_request_address, dma_request_bytes} == stalled_request)
                    else $error("layer address DMA request changed while stalled");
            if (dma_request_valid && !dma_request_ready)
                stalled_request <= {dma_request_address, dma_request_bytes};
            if ($past(done_valid && !done_ready))
                assert (done_valid && {done_error, done_error_id} == stalled_completion)
                    else $error("layer address completion changed while stalled");
            if (done_valid && !done_ready)
                stalled_completion <= {done_error, done_error_id};
            if (response_fire)
                assert (response_count < 7'd39)
                    else $error("layer address loader accepted more than 39 response beats");
            if (entry_invalidate)
                assert (state == IDLE)
                    else $error("layer address entry invalidated while loader was active");
        end
    end
`endif
endmodule

`default_nettype wire
