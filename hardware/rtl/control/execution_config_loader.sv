`default_nettype none

module execution_config_loader (
    input  logic          clk,
    input  logic          rst,

    input  logic          start_valid,
    output logic          start_ready,
    input  logic [63:0]   start_address,

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
    output logic [2559:0] configuration_bits
);
    import execution_config_pkg::*;
    import layer_address_table_pkg::*;

    localparam logic [15:0] ERROR_NONE = 16'h0000;
    localparam logic [15:0] ERROR_ADDRESS = 16'h0001;
    localparam logic [15:0] ERROR_DMA = 16'h0002;
    localparam logic [15:0] ERROR_STREAM = 16'h0003;
    localparam logic [15:0] ERROR_HEADER = 16'h0004;
    localparam logic [15:0] ERROR_GEOMETRY = 16'h0005;
    localparam logic [15:0] ERROR_REGION = 16'h0006;

    typedef enum logic [3:0] {
        IDLE,
        REQUEST,
        STREAM,
        CHECK_HEADER,
        CHECK_GEOMETRY,
        CHECK_REGION,
        OVERLAP_LOAD_LEFT,
        OVERLAP_LOAD_RIGHT,
        OVERLAP_CHECK,
        CHECK_LAYER_TABLE,
        COMPLETE
    } state_t;

    state_t state;
    logic [63:0] saved_address;
    logic [4:0] response_beat;
    logic [3:0] region_index;
    logic validation_error;
    logic response_last_seen;
    logic response_fire;
    logic response_format_error;

    logic [63:0] current_region_base;
    logic [63:0] current_region_limit;
    logic current_region_required;
    logic [64:0] layer_table_limit_accumulator;
    logic [15:0] layer_table_entries_remaining;
    logic configuration_v4;
    logic v4_header_error;
    logic [3:0] final_region_index;
    logic [3:0] overlap_left_index;
    logic [3:0] overlap_right_index;
    logic [63:0] overlap_left_base;
    logic [63:0] overlap_left_limit;
    logic [63:0] overlap_right_base;
    logic [63:0] overlap_right_limit;

    function automatic logic [7:0] cfg_u8(input integer byte_offset);
        cfg_u8 = configuration_bits[byte_offset*8 +: 8];
    endfunction

    function automatic logic [15:0] cfg_u16(input integer byte_offset);
        cfg_u16 = configuration_bits[byte_offset*8 +: 16];
    endfunction

    function automatic logic [31:0] cfg_u32(input integer byte_offset);
        cfg_u32 = configuration_bits[byte_offset*8 +: 32];
    endfunction

    function automatic logic [63:0] cfg_u64(input integer byte_offset);
        cfg_u64 = configuration_bits[byte_offset*8 +: 64];
    endfunction

    function automatic logic [63:0] v4_region_base(input logic [3:0] index);
        case (index)
            4'd0: v4_region_base = cfg_u64(
                EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET);
            4'd1: v4_region_base = cfg_u64(EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET);
            4'd2: v4_region_base = cfg_u64(EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET);
            4'd3: v4_region_base = cfg_u64(EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET);
            4'd4: v4_region_base = cfg_u64(EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET);
            4'd5: v4_region_base = cfg_u64(EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET);
            4'd6: v4_region_base = cfg_u64(EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET);
            4'd7: v4_region_base = cfg_u64(EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET);
            4'd8: v4_region_base = cfg_u64(EXECUTION_CONFIG_K_SCALE_BASE_OFFSET);
            4'd9: v4_region_base = cfg_u64(EXECUTION_CONFIG_V_SCALE_BASE_OFFSET);
            4'd10: v4_region_base = cfg_u64(EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET);
            4'd11: v4_region_base = cfg_u64(EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET);
            4'd12: v4_region_base = cfg_u64(EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET);
            4'd13: v4_region_base = cfg_u64(
                EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET);
            4'd14: v4_region_base = cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET);
            default: v4_region_base = cfg_u64(EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET);
        endcase
    endfunction

    function automatic logic [63:0] v4_region_limit(input logic [3:0] index);
        case (index)
            4'd0: v4_region_limit = cfg_u64(
                EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET);
            4'd1: v4_region_limit = cfg_u64(EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET);
            4'd2: v4_region_limit = cfg_u64(EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET);
            4'd3: v4_region_limit = cfg_u64(EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET);
            4'd4: v4_region_limit = cfg_u64(EXECUTION_CONFIG_CURRENT_K_CACHE_LIMIT_OFFSET);
            4'd5: v4_region_limit = cfg_u64(EXECUTION_CONFIG_CURRENT_V_CACHE_LIMIT_OFFSET);
            4'd6: v4_region_limit = cfg_u64(EXECUTION_CONFIG_RETAINED_K_CACHE_LIMIT_OFFSET);
            4'd7: v4_region_limit = cfg_u64(EXECUTION_CONFIG_RETAINED_V_CACHE_LIMIT_OFFSET);
            4'd8: v4_region_limit = cfg_u64(EXECUTION_CONFIG_K_SCALE_LIMIT_OFFSET);
            4'd9: v4_region_limit = cfg_u64(EXECUTION_CONFIG_V_SCALE_LIMIT_OFFSET);
            4'd10: v4_region_limit = cfg_u64(EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET);
            4'd11: v4_region_limit = cfg_u64(EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET);
            4'd12: v4_region_limit = cfg_u64(EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET);
            4'd13: v4_region_limit = cfg_u64(
                EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET);
            4'd14: v4_region_limit = cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET);
            default: v4_region_limit = cfg_u64(EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET);
        endcase
    endfunction

    always_comb begin
        configuration_v4 =
            cfg_u32(EXECUTION_CONFIG_MAGIC_OFFSET) == 32'h344e4c44 &&
            cfg_u16(EXECUTION_CONFIG_VERSION_OFFSET) == 16'd4;
        final_region_index = 4'd14;
        if (configuration_v4 &&
            (cfg_u64(EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET) != 0 ||
             cfg_u64(EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET) != 0))
            final_region_index = 4'd15;
    end


    always_comb begin
        v4_header_error =
            cfg_u16(EXECUTION_CONFIG_HEADER_BYTES_OFFSET) !=
                16'(EXECUTION_CONFIG_BYTES) ||
            cfg_u32(EXECUTION_CONFIG_TOTAL_BYTES_OFFSET) !=
                EXECUTION_CONFIG_BYTES ||
            (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'hffff_c000) != 0 ||
            (configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+13] &&
             (cfg_u16(EXECUTION_CONFIG_START_LAYER_OFFSET) +
              cfg_u16(EXECUTION_CONFIG_LAYER_COUNT_OFFSET) != 16'd32 ||
              configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+3])) ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h1000) != 0 &&
             (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h0e00) != 0) ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h100) != 0 &&
             (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h80) == 0) ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h10) != 0 &&
             cfg_u8(EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET) == 0) ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h20) != 0 &&
             ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h10) == 0 ||
              cfg_u8(EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET) == 0)) ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h40) != 0 &&
             ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h10) == 0 ||
              cfg_u8(EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET) == 0)) ||
            (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h60) == 32'h60 ||
            ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h80) != 0 &&
             ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h10) == 0 ||
              (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h60) == 0)) ||
            cfg_u16(EXECUTION_CONFIG_TOKEN_METADATA_FORMAT_OFFSET) != 16'd3 ||
            (cfg_u16(EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET) != 0 &&
             (cfg_u16(EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET) < 16'd320 ||
              (cfg_u16(EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET) & 16'hf) != 0)) ||
            cfg_u8(EXECUTION_CONFIG_ROTATION_ENABLE_MASK_OFFSET) != 0 ||
            cfg_u8(EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET) > 8'd63 ||
            (cfg_u16(EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET) != 0 &&
             (cfg_u16(EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET) == 0 ||
              cfg_u16(EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET) < 16'd320 ||
              (cfg_u16(EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET) & 16'hf) != 0)) ||
            cfg_u16(EXECUTION_CONFIG_TOKEN_METADATA_ENTRY_BYTES_OFFSET) != 16'd16 ||
            cfg_u16(EXECUTION_CONFIG_TOKEN_METADATA_BATCH_HEADER_BYTES_OFFSET) !=
                16'd32 ||
            cfg_u16(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BYTES_OFFSET) !=
                16'd192 ||
            cfg_u16(EXECUTION_CONFIG_PREDICTION_ENTRY_BYTES_OFFSET) != 16'd32 ||
            cfg_u16(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_FORMAT_OFFSET) !=
                16'd1 ||
            cfg_u16(EXECUTION_CONFIG_PREDICTION_TABLE_FORMAT_OFFSET) != 16'd2 ||
            cfg_u32(EXECUTION_CONFIG_EMBEDDING_ROW_BYTES_OFFSET) != 32'd8192 ||
            cfg_u16(EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET) >
                cfg_u16(EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET) ||
            cfg_u64(EXECUTION_CONFIG_DEPLOYMENT_ARTIFACT_ID_OFFSET) == 0 ||
            (cfg_u64(EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET) & 64'h1fff) != 0 ||
            {1'b0, cfg_u64(EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET)} +
                65'd1035993088 >
                {1'b0, cfg_u64(EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET)} ||
            cfg_u64(EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET) -
                cfg_u64(EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET) < 64'd64;
        if ((cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h4) == 0) begin
            v4_header_error = v4_header_error ||
                (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h8) != 0 ||
                cfg_u16(EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET) != 0 ||
                cfg_u8(EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET) != 0 ||
                cfg_u8(EXECUTION_CONFIG_SUPPRESSED_TOKEN_COUNT_OFFSET) != 0 ||
                cfg_u64(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET) !=
                    cfg_u64(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET) ||
                cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET) !=
                    cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET);
        end else begin
            v4_header_error = v4_header_error ||
                cfg_u16(EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET) < 1 ||
                cfg_u16(EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET) > 96 ||
                cfg_u8(EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET) < 1 ||
                cfg_u8(EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET) > 4 ||
                cfg_u64(EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET) -
                    cfg_u64(EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET) < 64'd1696 ||
                {1'b0, cfg_u64(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET)} + 65'd192 !=
                    {1'b0, cfg_u64(EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET)} ||
                {1'b0, cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET)} +
                    {44'd0, cfg_u16(EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET), 5'd0} !=
                    {1'b0, cfg_u64(EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET)};
        end
    end

    always_comb begin
        current_region_base = v4_region_base(region_index);
        current_region_limit = v4_region_limit(region_index);
        current_region_required = 1'b1;
        case (region_index)
            4'd0, 4'd13, 4'd14:
                current_region_required =
                    (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h4) != 0;
            4'd2: current_region_required =
                    (cfg_u32(EXECUTION_CONFIG_FLAGS_OFFSET) & 32'h1) != 0;
            4'd6, 4'd7, 4'd15: current_region_required = 1'b0;
            default: ;
        endcase
    end

    assign start_ready = state == IDLE;
    assign dma_request_valid = state == REQUEST;
    assign dma_request_address = saved_address;
    assign dma_request_bytes = EXECUTION_CONFIG_BYTES;
    assign dma_response_ready = state == STREAM;
    assign done_valid = state == COMPLETE;
    assign response_fire = dma_response_valid && dma_response_ready;
    assign response_format_error = response_fire &&
        (dma_response_byte_enable != 16'hffff ||
         dma_response_last != (response_beat == 19));
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_address <= 64'd0;
            response_beat <= 5'd0;
            region_index <= 4'd0;
            overlap_left_index <= 4'd0;
            overlap_right_index <= 4'd0;
            overlap_left_base <= 64'd0;
            overlap_left_limit <= 64'd0;
            overlap_right_base <= 64'd0;
            overlap_right_limit <= 64'd0;
            validation_error <= 1'b0;
            response_last_seen <= 1'b0;
            layer_table_limit_accumulator <= 65'd0;
            layer_table_entries_remaining <= 16'd0;
            done_error <= 1'b0;
            done_error_id <= ERROR_NONE;
        end else begin
            case (state)
                IDLE: if (start_valid && start_ready) begin
                    saved_address <= start_address;
                    response_beat <= 5'd0;
                    validation_error <= 1'b0;
                    response_last_seen <= 1'b0;
                    done_error <= 1'b0;
                    done_error_id <= ERROR_NONE;
                    if (start_address[3:0] != 0) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_ADDRESS;
                        state <= COMPLETE;
                    end else begin
                        state <= REQUEST;
                    end
                end

                REQUEST: if (dma_request_valid && dma_request_ready) begin
                    response_beat <= 5'd0;
                    response_last_seen <= 1'b0;
                    state <= STREAM;
                end

                STREAM: begin
                    if (response_fire) begin
                        if (response_beat < 20)
                            configuration_bits[response_beat*128 +: 128] <= dma_response_data;
                        if (response_format_error && !validation_error) begin
                            validation_error <= 1'b1;
                        end
                        if (dma_response_last)
                            response_last_seen <= 1'b1;
                        if (response_beat != 5'h1f)
                            response_beat <= response_beat + 1'b1;
                    end

                    if (dma_request_done) begin
                        if (dma_request_error) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_DMA;
                            state <= COMPLETE;
                        end else if (validation_error || response_format_error ||
                                     !(response_last_seen ||
                                       (response_fire && dma_response_last)) ||
                                     !((response_beat == 20) ||
                                       (response_fire && response_beat == 19))) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_STREAM;
                            state <= COMPLETE;
                        end else begin
                            state <= CHECK_HEADER;
                        end
                    end
                end

                CHECK_HEADER: begin
                    if (!configuration_v4 || v4_header_error) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_HEADER;
                        state <= COMPLETE;
                    end else begin
                        state <= CHECK_GEOMETRY;
                    end
                end

                CHECK_GEOMETRY: begin
                    if (cfg_u16(EXECUTION_CONFIG_START_LAYER_OFFSET) > 16'd31 ||
                        {1'b0, cfg_u16(EXECUTION_CONFIG_START_LAYER_OFFSET)} +
                        {1'b0, cfg_u16(EXECUTION_CONFIG_LAYER_COUNT_OFFSET)} > 17'd32 ||
                        cfg_u16(EXECUTION_CONFIG_LAYER_COUNT_OFFSET) < 1 ||
                        cfg_u16(EXECUTION_CONFIG_LAYER_COUNT_OFFSET) > 32 ||
                        cfg_u16(EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET) < 1 ||
                        cfg_u16(EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET) > 2048 ||
                        cfg_u16(EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET) < 1 ||
                        cfg_u16(EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET) > 2048 ||
                        cfg_u16(EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET) >
                            cfg_u16(EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET)) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_GEOMETRY;
                        state <= COMPLETE;
                    end else begin
                        region_index <= 4'd0;
                        state <= CHECK_REGION;
                    end
                end

                CHECK_REGION: begin
                    if (current_region_base[3:0] != 0 || current_region_limit[3:0] != 0 ||
                        current_region_base > current_region_limit ||
                        (current_region_required && current_region_base == current_region_limit)) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_REGION;
                        state <= COMPLETE;
                    end else if (region_index == final_region_index) begin
                        if (cfg_u32(EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET) <
                                32'd880 ||
                            (cfg_u32(EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET) & 32'hf) != 0) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_REGION;
                            state <= COMPLETE;
                        end else begin
                            overlap_left_index <= 4'd0;
                            overlap_right_index <= 4'd1;
                            state <= OVERLAP_LOAD_LEFT;
                        end
                    end else begin
                        region_index <= region_index + 1'b1;
                    end
                end

                OVERLAP_LOAD_LEFT: begin
                    overlap_left_base <= v4_region_base(overlap_left_index);
                    overlap_left_limit <= v4_region_limit(overlap_left_index);
                    state <= OVERLAP_LOAD_RIGHT;
                end

                OVERLAP_LOAD_RIGHT: begin
                    overlap_right_base <= v4_region_base(overlap_right_index);
                    overlap_right_limit <= v4_region_limit(overlap_right_index);
                    state <= OVERLAP_CHECK;
                end

                OVERLAP_CHECK: begin
                    if (overlap_left_base != overlap_left_limit &&
                        overlap_right_base != overlap_right_limit &&
                        overlap_left_base < overlap_right_limit &&
                        overlap_right_base < overlap_left_limit &&
                        !(((overlap_left_index == 4'd4 && overlap_right_index == 4'd6) ||
                           (overlap_left_index == 4'd5 && overlap_right_index == 4'd7)) &&
                          overlap_left_base == overlap_right_base &&
                          overlap_left_limit == overlap_right_limit)) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_REGION;
                        state <= COMPLETE;
                    end else if (overlap_right_index == final_region_index) begin
                        if (overlap_left_index + 4'd1 == final_region_index) begin
                            layer_table_limit_accumulator <= {1'b0,
                                cfg_u64(EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET)};
                            layer_table_entries_remaining <=
                                cfg_u16(EXECUTION_CONFIG_LAYER_COUNT_OFFSET) +
                                cfg_u16(EXECUTION_CONFIG_START_LAYER_OFFSET);
                            state <= CHECK_LAYER_TABLE;
                        end else begin
                            overlap_left_index <= overlap_left_index + 1'b1;
                            overlap_right_index <= overlap_left_index + 4'd2;
                            state <= OVERLAP_LOAD_LEFT;
                        end
                    end else begin
                        overlap_right_index <= overlap_right_index + 1'b1;
                        state <= OVERLAP_LOAD_RIGHT;
                    end
                end

                CHECK_LAYER_TABLE: begin
                    if (layer_table_entries_remaining != 0) begin
                        layer_table_limit_accumulator <=
                            layer_table_limit_accumulator + {33'd0,
                                cfg_u32(
                                    EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET)};
                        layer_table_entries_remaining <=
                            layer_table_entries_remaining - 1'b1;
                    end else if (layer_table_limit_accumulator > {1'b0,
                        cfg_u64(EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET)}) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_REGION;
                        state <= COMPLETE;
                    end else begin
                        state <= COMPLETE;
                    end
                end

                COMPLETE: if (done_valid && done_ready)
                    state <= IDLE;

                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic [95:0] stalled_request;
    logic [16:0] stalled_completion;
    always_ff @(posedge clk) begin
        if (!rst) begin
            if ($past(dma_request_valid && !dma_request_ready))
                assert (dma_request_valid && {dma_request_address, dma_request_bytes} == stalled_request)
                    else $error("configuration DMA request changed while stalled");
            if (dma_request_valid && !dma_request_ready)
                stalled_request <= {dma_request_address, dma_request_bytes};
            if ($past(done_valid && !done_ready))
                assert (done_valid && {done_error, done_error_id} == stalled_completion)
                    else $error("configuration completion changed while stalled");
            if (done_valid && !done_ready)
                stalled_completion <= {done_error, done_error_id};
        end
    end
`endif
endmodule

`default_nettype wire
