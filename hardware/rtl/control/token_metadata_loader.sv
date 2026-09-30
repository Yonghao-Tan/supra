`default_nettype none

module token_metadata_loader (
    input  logic clk,
    input  logic rst,
    input  logic start_valid,
    output logic start_ready,
    input  logic [2559:0] start_configuration_bits,
    input  logic start_l31_layout,

    output logic dma_request_valid,
    input  logic dma_request_ready,
    output logic [63:0] dma_request_address,
    output logic [31:0] dma_request_bytes,
    input  logic dma_response_valid,
    output logic dma_response_ready,
    input  logic [127:0] dma_response_data,
    input  logic [15:0] dma_response_byte_enable,
    input  logic dma_response_last,
    input  logic dma_request_done,
    input  logic dma_request_error,

    output logic loaded_token_valid,
    output logic [10:0] loaded_token_position,
    output logic loaded_token_kv_write,
    output logic done_valid,
    input  logic done_ready,
    output logic done_error,
    output logic [15:0] done_error_id,
    output logic config_valid,
    input  logic config_release,
    output logic [11:0] total_token_count,
    output logic [5:0] token_batch_index,
    output logic last_round,
    output logic [5:0] active_token_count,
    output logic [5:0] compute_group_count,
    output logic output_subset_enable,
    output logic qkvo_group_end,
    output logic [5:0] output_token_count,
    output logic [5:0] output_compute_group_count,
    output logic [11:0] sequence_length,
    output logic [287:0] physical_to_local,
    output logic [287:0] local_to_physical,
    output logic [527:0] token_position,
    output logic [527:0] row_kv_index,
    output logic [47:0] row_kv_write_disable,
    output logic [47:0] matmul_row_valid,
    output logic [815:0] row_source_index,
    output logic [47:0] row_embedding_source,
    output logic [191:0] row_precision_bits,
    output logic [383:0] row_query_group,
    output logic [383:0] row_cache_group,
    output logic [287:0] row_compute_group,
    output logic [143:0] row_pe_slot,
    output logic [95:0] row_phase_mask
);
    import execution_config_pkg::*;

    localparam logic [15:0] ERROR_NONE = 16'h0000;
    localparam logic [15:0] ERROR_REGION = 16'h0301;
    localparam logic [15:0] ERROR_DMA = 16'h0302;
    localparam logic [15:0] ERROR_STREAM = 16'h0303;
    localparam logic [15:0] ERROR_HEADER = 16'h0304;
    localparam logic [15:0] ERROR_ENTRY = 16'h0305;
    localparam logic [15:0] ERROR_MAPPING = 16'h0306;
    localparam logic [15:0] ERROR_GROUP = 16'h0307;

    typedef enum logic [4:0] {
        IDLE,
        SCAN_HEADER_REQUEST,
        SCAN_HEADER_STREAM,
        SCAN_HEADER_CHECK,
        SCAN_PAYLOAD_REQUEST,
        SCAN_PAYLOAD_STREAM,
        SCAN_PAYLOAD_CHECK,
        LOAD_HEADER_REQUEST,
        LOAD_HEADER_STREAM,
        LOAD_HEADER_CHECK,
        LOAD_PAYLOAD_REQUEST,
        LOAD_PAYLOAD_STREAM,
        LOAD_PAYLOAD_CHECK,
        COMMIT,
        COMPLETE,
        ACTIVE
    } state_t;

    state_t state;
    logic [63:0] metadata_base;
    logic [63:0] metadata_limit;
    logic [63:0] scan_address;
    logic [63:0] load_address;
    logic [11:0] load_first_row;
    logic scan_complete;
    logic [11:0] scan_first_row;
    logic [5:0] scan_token_batch_index;
    logic [31:0] saved_metadata_version;
    logic [31:0] saved_capture_index;
    logic saved_output_subset_enable;
    logic layout_header_pending;
    logic selected_l31_layout;

    logic [127:0] header_low;
    logic [127:0] header_high;
    logic [255:0] header_bits;
    logic [5:0] header_rows;
    logic [5:0] header_compute_groups;
    logic [5:0] header_semantic_groups;
    logic header_embedding_present;
    logic output_subset_allowed;
    logic [15:0] header_token_batch_index;
    logic [15:0] header_first_row;
    logic [15:0] header_token_batch_bytes;
    logic [15:0] header_inverse_offset;
    logic [15:0] header_inverse_bytes;
    logic [31:0] header_metadata_version;
    logic [31:0] header_capture_index;
    logic [31:0] payload_bytes;
    logic [7:0] payload_beats;

    logic [7:0] transaction_beat_count;
    logic [7:0] transaction_expected_beats;
    logic transaction_data_done;
    logic transaction_done_seen;
    logic transaction_error_seen;
    logic transaction_stream_error;
    logic response_fire;
    logic response_is_last;
    logic transaction_complete;

    logic [47:0] scan_logical_seen;
    logic [47:0] scan_group_seen;
    logic [47:0] scan_embedding_seen;
    logic scan_validation_error;
    logic [15:0] scan_validation_error_id;

    logic [16:0] loaded_source [0:47];
    logic [10:0] loaded_position [0:47];
    logic [10:0] loaded_kv_index [0:47];
    logic [5:0] loaded_token_ordinal [0:47];
    logic loaded_embedding [0:47];
    logic [3:0] loaded_bits [0:47];
    logic [7:0] loaded_query [0:47];
    logic [7:0] loaded_cache [0:47];
    logic [5:0] loaded_compute_group [0:47];
    logic [2:0] loaded_pe_slot [0:47];
    logic [1:0] loaded_phase_mask [0:47];
    logic [127:0] loaded_inverse_word [0:2];
    logic [47:0] loaded_rows;

    logic entry_position_duplicate;
    logic [5:0] entry_physical_row;
    logic [16:0] entry_source_index;
    logic [10:0] entry_token_position;
    logic [10:0] entry_kv_index;
    logic [5:0] entry_token_ordinal;
    logic entry_embedding_source;
    logic [3:0] entry_bits;
    logic [7:0] entry_query_group;
    logic [7:0] entry_cache_group;
    logic [5:0] entry_compute_group;
    logic [2:0] entry_pe_slot;
    logic [1:0] entry_phase_mask;
    logic entry_format_error;
    logic entry_group_error;
    logic entry_slot_error;
    logic entry_logical_seen;
    logic entry_group_seen;

    function automatic logic [63:0] cfg_u64(
        input logic [2559:0] bits, input integer byte_offset);
        cfg_u64 = bits[byte_offset*8 +: 64];
    endfunction

    function automatic logic [15:0] cfg_u16(
        input logic [2559:0] bits, input integer byte_offset);
        cfg_u16 = bits[byte_offset*8 +: 16];
    endfunction

    function automatic logic [47:0] low_mask(input logic [5:0] count);
        logic [47:0] value;
        begin
            if (count == 6'd48)
                low_mask = 48'hffff_ffff_ffff;
            else begin
                value = (48'd1 << count) - 48'd1;
                low_mask = value;
            end
        end
    endfunction

    assign header_bits = {header_high, header_low};
    assign header_rows = header_bits[5:0];
    assign header_compute_groups = header_bits[13:8];
    assign header_semantic_groups = header_bits[21:16];
    assign header_embedding_present = header_bits[24];
    assign output_subset_enable = header_bits[25];
    assign qkvo_group_end = header_bits[26];
    assign output_token_count = output_subset_enable ? header_bits[245:240] : header_rows;
    assign output_compute_group_count = output_subset_enable ? header_bits[253:248] : header_compute_groups;
    assign header_token_batch_index = header_bits[47:32];
    assign header_first_row = header_bits[63:48];
    assign header_token_batch_bytes = header_bits[79:64];
    assign header_inverse_offset = header_bits[111:96];
    assign header_inverse_bytes = header_bits[127:112];
    assign header_metadata_version = header_bits[159:128];
    assign header_capture_index = header_bits[191:160];
    // Speculative membership becomes consumable only after successful forward
    // completion. Failed metadata/compute transactions never update pending.
    assign loaded_token_valid = state == LOAD_PAYLOAD_STREAM && response_fire &&
        transaction_beat_count < 8'(header_rows) && entry_physical_row < header_rows;
    assign loaded_token_position = entry_token_position;
    assign loaded_token_kv_write = entry_physical_row < 48 && !header_bits[192+entry_physical_row];
    assign row_kv_write_disable = header_bits[239:192];
    assign payload_bytes = {16'd0, header_token_batch_bytes} - 32'd32;
    assign payload_beats = 8'(payload_bytes >> 4);

    assign start_ready = state == IDLE;
    assign done_valid = state == COMPLETE;
    assign config_valid = state == ACTIVE;
    assign compute_group_count = header_compute_groups;
    assign response_fire = dma_response_valid && dma_response_ready;
    assign response_is_last = response_fire && dma_response_last;
    assign transaction_complete =
        (transaction_data_done || response_is_last) &&
        (transaction_done_seen || dma_request_done);
    assign dma_request_valid = state == SCAN_HEADER_REQUEST ||
        state == SCAN_PAYLOAD_REQUEST || state == LOAD_HEADER_REQUEST ||
        state == LOAD_PAYLOAD_REQUEST;
    assign dma_request_address =
        state == SCAN_HEADER_REQUEST ? scan_address :
        state == SCAN_PAYLOAD_REQUEST ? scan_address + 64'd32 :
        state == LOAD_HEADER_REQUEST ? load_address : load_address + 64'd32;
    assign dma_request_bytes =
        (state == SCAN_HEADER_REQUEST || state == LOAD_HEADER_REQUEST) ?
            32'd32 : payload_bytes;
    assign dma_response_ready = state == SCAN_HEADER_STREAM ||
        state == SCAN_PAYLOAD_STREAM || state == LOAD_HEADER_STREAM ||
        state == LOAD_PAYLOAD_STREAM;

    assign entry_physical_row = transaction_beat_count[5:0];
    assign entry_source_index = dma_response_data[16:0];
    assign entry_token_position = dma_response_data[42:32];
    assign entry_kv_index = dma_response_data[58:48];
    assign entry_token_ordinal = dma_response_data[69:64];
    assign entry_embedding_source = dma_response_data[72];
    assign entry_bits = dma_response_data[83:80];
    assign entry_query_group = dma_response_data[95:88];
    assign entry_cache_group = dma_response_data[103:96];
    assign entry_compute_group = dma_response_data[109:104];
    assign entry_pe_slot = dma_response_data[114:112];
    assign entry_phase_mask = dma_response_data[121:120];

    always_comb begin
        entry_position_duplicate = 1'b0;
        for (integer prior = 0; prior < 48; prior++) begin
            if (prior < entry_physical_row &&
                loaded_position[prior] == entry_token_position)
                entry_position_duplicate = 1'b1;
        end
    end
    always_comb begin
        entry_logical_seen = 1'b1;
        entry_group_seen = 1'b0;
        entry_group_error = 1'b0;
        entry_slot_error = 1'b0;
        if (entry_token_ordinal < 6'd48)
            entry_logical_seen = scan_logical_seen[entry_token_ordinal];
        if (entry_compute_group < 6'd48) begin
            entry_group_seen = scan_group_seen[entry_compute_group];
            entry_group_error = entry_group_seen &&
                (loaded_query[entry_compute_group] != entry_query_group ||
                 loaded_cache[entry_compute_group] != entry_cache_group ||
                 loaded_compute_group[entry_compute_group] +
                     (entry_bits == 4'd8 ? 6'd2 : 6'd1) > 6'd16);
            entry_slot_error = entry_group_seen &&
                ((entry_phase_mask[0] &&
                  loaded_source[entry_compute_group]
                      [{2'b00, entry_pe_slot}]) ||
                 (entry_phase_mask[1] &&
                  loaded_source[entry_compute_group]
                      [{2'b01, entry_pe_slot}]));
        end
    end
    assign entry_format_error =
        entry_physical_row >= header_rows ||
        entry_token_ordinal >= header_rows ||
        (output_subset_enable &&
         ((entry_physical_row < output_token_count) !=
              (entry_token_ordinal < output_token_count) ||
          (entry_physical_row < output_token_count) !=
              (entry_compute_group < output_compute_group_count))) ||
        entry_logical_seen ||
        entry_position_duplicate ||
        {1'b0, entry_token_position} >= sequence_length ||
        {1'b0, entry_kv_index} >= sequence_length ||
        entry_kv_index != entry_token_position ||
        (entry_embedding_source && entry_source_index >= 17'd126464) ||
        (!entry_embedding_source && entry_source_index >= 17'd2048) ||
        (entry_bits != 4'd4 && entry_bits != 4'd8) ||
        entry_compute_group >= header_compute_groups ||
        (entry_bits == 4'd8 && entry_phase_mask != 2'b11) ||
        (entry_bits == 4'd4 &&
         entry_phase_mask != 2'b01 && entry_phase_mask != 2'b10) ||
        dma_response_data[127:122] != 0 ||
        dma_response_data[119:115] != 0 ||
        dma_response_data[111:110] != 0 ||
        dma_response_data[87:84] != 0 ||
        dma_response_data[79:73] != 0 ||
        dma_response_data[71:70] != 0 ||
        dma_response_data[63:59] != 0 ||
        dma_response_data[47:43] != 0 ||
        dma_response_data[31:17] != 0;
    always_comb begin
        physical_to_local = '0;
        local_to_physical = '0;
        token_position = '0;
        row_kv_index = '0;
        matmul_row_valid = loaded_rows;
        row_source_index = '0;
        row_embedding_source = '0;
        row_precision_bits = '0;
        row_query_group = '0;
        row_cache_group = '0;
        row_compute_group = '0;
        row_pe_slot = '0;
        row_phase_mask = '0;
        for (integer row = 0; row < 48; row++) begin
            if (row < header_rows)
                local_to_physical[row*6 +: 6] =
                    loaded_inverse_word[row/16][(row%16)*8 +: 6];
            if (loaded_rows[row]) begin
                physical_to_local[row*6 +: 6] = loaded_token_ordinal[row];
                token_position[row*11 +: 11] = loaded_position[row];
                row_kv_index[row*11 +: 11] = loaded_kv_index[row];
                row_source_index[row*17 +: 17] = loaded_source[row];
                row_embedding_source[row] = loaded_embedding[row];
                row_precision_bits[row*4 +: 4] = loaded_bits[row];
                row_query_group[row*8 +: 8] = loaded_query[row];
                row_cache_group[row*8 +: 8] = loaded_cache[row];
                row_compute_group[row*6 +: 6] = loaded_compute_group[row];
                row_pe_slot[row*3 +: 3] = loaded_pe_slot[row];
                row_phase_mask[row*2 +: 2] = loaded_phase_mask[row];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            metadata_base <= '0;
            metadata_limit <= '0;
            scan_address <= '0;
            load_address <= '0;
            load_first_row <= '0;
            scan_complete <= 1'b0;
            scan_first_row <= '0;
            scan_token_batch_index <= '0;
            saved_metadata_version <= '0;
            saved_capture_index <= '0;
            saved_output_subset_enable <= 1'b0;
            layout_header_pending <= 1'b0;
            selected_l31_layout <= 1'b0;
            output_subset_allowed <= 1'b0;
            header_low <= '0;
            header_high <= '0;
            transaction_beat_count <= '0;
            transaction_expected_beats <= '0;
            transaction_data_done <= 1'b0;
            transaction_done_seen <= 1'b0;
            transaction_error_seen <= 1'b0;
            transaction_stream_error <= 1'b0;
            scan_logical_seen <= '0;
            scan_group_seen <= '0;
            scan_embedding_seen <= '0;
            scan_validation_error <= 1'b0;
            scan_validation_error_id <= ERROR_NONE;
            loaded_rows <= '0;
            total_token_count <= '0;
            token_batch_index <= '0;
            last_round <= 1'b0;
            active_token_count <= '0;
            sequence_length <= '0;
            done_error <= 1'b0;
            done_error_id <= ERROR_NONE;
        end else begin
            if (dma_request_done) begin
                transaction_done_seen <= 1'b1;
                transaction_error_seen <= dma_request_error;
            end
            if (response_is_last)
                transaction_data_done <= 1'b1;
            if (response_fire) begin
                if (dma_response_byte_enable != 16'hffff ||
                    dma_response_last !=
                        (transaction_beat_count + 8'd1 ==
                         transaction_expected_beats))
                    transaction_stream_error <= 1'b1;
                transaction_beat_count <= transaction_beat_count + 8'd1;
            end

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    done_error <= 1'b0;
                    done_error_id <= ERROR_NONE;
                    if (!scan_complete) begin
                        layout_header_pending <= start_configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+13];
                        selected_l31_layout <= start_l31_layout;
                        output_subset_allowed <=
                            17'(cfg_u16(start_configuration_bits, EXECUTION_CONFIG_START_LAYER_OFFSET)) +
                            17'(cfg_u16(start_configuration_bits, EXECUTION_CONFIG_LAYER_COUNT_OFFSET)) == 17'd32 &&
                            !start_configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+3] &&
                            (!start_configuration_bits[EXECUTION_CONFIG_FLAGS_OFFSET*8+13] || start_l31_layout);
                        metadata_base <= cfg_u64(start_configuration_bits,
                            EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET);
                        metadata_limit <= cfg_u64(start_configuration_bits,
                            EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET);
                        total_token_count <= 12'(cfg_u16(start_configuration_bits,
                            EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET));
                        sequence_length <= 12'(cfg_u16(start_configuration_bits,
                            EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET));
                        scan_address <= cfg_u64(start_configuration_bits,
                            EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET);
                        scan_first_row <= '0;
                        scan_token_batch_index <= '0;
                        token_batch_index <= '0;
                        load_first_row <= '0;
                        if ((cfg_u64(start_configuration_bits,
                                EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET) &
                                64'hf) != 0 ||
                            {1'b0, cfg_u64(start_configuration_bits,
                                EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET)} + 65'd32 >
                            {1'b0, cfg_u64(start_configuration_bits,
                                EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET)}) begin
                            done_error <= 1'b1;
                            done_error_id <= ERROR_REGION;
                            state <= COMPLETE;
                        end else begin
                            state <= SCAN_HEADER_REQUEST;
                        end
                    end else begin
                        state <= LOAD_HEADER_REQUEST;
                    end
                end

                SCAN_HEADER_REQUEST, LOAD_HEADER_REQUEST:
                    if (dma_request_valid && dma_request_ready) begin
                        transaction_beat_count <= '0;
                        transaction_expected_beats <= 8'd2;
                        transaction_data_done <= 1'b0;
                        transaction_done_seen <= 1'b0;
                        transaction_error_seen <= 1'b0;
                        transaction_stream_error <= 1'b0;
                        state <= state == SCAN_HEADER_REQUEST ?
                            SCAN_HEADER_STREAM : LOAD_HEADER_STREAM;
                    end

                SCAN_HEADER_STREAM, LOAD_HEADER_STREAM: begin
                    if (response_fire) begin
                        if (transaction_beat_count == 0)
                            header_low <= dma_response_data;
                        else
                            header_high <= dma_response_data;
                    end
                    if (transaction_complete)
                        state <= state == SCAN_HEADER_STREAM ?
                            SCAN_HEADER_CHECK : LOAD_HEADER_CHECK;
                end

                SCAN_HEADER_CHECK, LOAD_HEADER_CHECK: begin
                    logic header_error;
                    header_error = transaction_error_seen ||
                        transaction_stream_error ||
                        transaction_beat_count != 8'd2 ||
                        header_rows == 0 || header_rows > 48 ||
                        header_compute_groups == 0 ||
                        header_compute_groups > header_rows ||
                        header_semantic_groups == 0 ||
                        header_semantic_groups > header_compute_groups ||
                        header_bits[31:27] != 0 || header_bits[23:22] != 0 ||
                        header_bits[15:14] != 0 || header_bits[7:6] != 0 ||
                        header_token_batch_index != (state == SCAN_HEADER_CHECK ?
                            {10'd0, scan_token_batch_index} : {10'd0, token_batch_index}) ||
                        header_first_row != (state == SCAN_HEADER_CHECK ?
                            {4'd0, scan_first_row} : {4'd0, load_first_row}) ||
                        header_bits[95:80] != 16'd16 ||
                        header_inverse_offset !=
                            16'd32 + {6'd0, header_rows, 4'd0} ||
                        header_inverse_bytes !=
                            {10'd0, (header_rows + 6'd15) & 6'h30} ||
                        header_token_batch_bytes != 16'd32 +
                            {6'd0, header_rows, 4'd0} + header_inverse_bytes ||
                        (output_subset_enable ?
                            (!output_subset_allowed || header_bits[247:246] != 0 ||
                             header_bits[255:254] != 0 || output_token_count > header_rows ||
                             output_compute_group_count > header_compute_groups ||
                             (output_token_count == 0) != (output_compute_group_count == 0) ||
                             output_compute_group_count > output_token_count) :
                            header_bits[255:240] != 0) ||
                        (header_bits[239:192] >> header_rows) != 0 ||
                        {1'b0, (state == SCAN_HEADER_CHECK ? scan_address :
                            load_address)} + {49'd0, header_token_batch_bytes} >
                            {1'b0, metadata_limit};
                    if (state == LOAD_HEADER_CHECK) begin
                        header_error = header_error ||
                            header_metadata_version != saved_metadata_version ||
                            header_capture_index != saved_capture_index ||
                            output_subset_enable != saved_output_subset_enable;
                    end
                    if (state == SCAN_HEADER_CHECK && scan_token_batch_index != 0) begin
                        header_error = header_error ||
                            header_metadata_version != saved_metadata_version ||
                            header_capture_index != saved_capture_index ||
                            output_subset_enable != saved_output_subset_enable;
                    end
                    if (layout_header_pending) begin
                        // The directory and both streams share the checked metadata region.
                        if (transaction_error_seen || transaction_stream_error ||
                            transaction_beat_count != 8'd2 ||
                            {1'b0, header_low[63:0]} != {1'b0, metadata_base} + 65'd32 ||
                            header_low[127:64] <= header_low[63:0] ||
                            header_high[63:0] != header_low[127:64] ||
                            header_high[127:64] <= header_high[63:0] ||
                            header_high[127:64] > metadata_limit ||
                            (header_low[67:64] | header_high[67:64]) != 4'd0) begin
                            done_error <= 1'b1;
                            done_error_id <= transaction_error_seen ? ERROR_DMA :
                                transaction_stream_error ? ERROR_STREAM : ERROR_REGION;
                            state <= COMPLETE;
                        end else begin
                            metadata_base <= selected_l31_layout ? header_high[63:0] : header_low[63:0];
                            metadata_limit <= selected_l31_layout ? header_high[127:64] : header_low[127:64];
                            scan_address <= selected_l31_layout ? header_high[63:0] : header_low[63:0];
                            layout_header_pending <= 1'b0;
                            state <= SCAN_HEADER_REQUEST;
                        end
                    end else if (header_error) begin
                        done_error <= 1'b1;
                        done_error_id <= transaction_error_seen ? ERROR_DMA :
                            transaction_stream_error ? ERROR_STREAM :
                            ERROR_HEADER;
                        state <= COMPLETE;
                    end else begin
                        scan_logical_seen <= '0;
                        scan_group_seen <= '0;
                        scan_embedding_seen <= '0;
                        scan_validation_error <= 1'b0;
                        scan_validation_error_id <= ERROR_NONE;
                        if (state == SCAN_HEADER_CHECK && scan_token_batch_index == 0) begin
                            saved_metadata_version <= header_metadata_version;
                            saved_capture_index <= header_capture_index;
                            saved_output_subset_enable <= output_subset_enable;
                        end
                        if (state == LOAD_HEADER_CHECK) begin
                            active_token_count <= header_rows;
                            last_round <= 12'(header_first_row) +
                                12'(header_rows) == total_token_count;
                            loaded_rows <= '0;
                        end
                        state <= state == SCAN_HEADER_CHECK ?
                            SCAN_PAYLOAD_REQUEST : LOAD_PAYLOAD_REQUEST;
                    end
                end

                SCAN_PAYLOAD_REQUEST, LOAD_PAYLOAD_REQUEST:
                    if (dma_request_valid && dma_request_ready) begin
                        transaction_beat_count <= '0;
                        transaction_expected_beats <= payload_beats;
                        transaction_data_done <= 1'b0;
                        transaction_done_seen <= 1'b0;
                        transaction_error_seen <= 1'b0;
                        transaction_stream_error <= 1'b0;
                        state <= state == SCAN_PAYLOAD_REQUEST ?
                            SCAN_PAYLOAD_STREAM : LOAD_PAYLOAD_STREAM;
                    end

                SCAN_PAYLOAD_STREAM: begin
                    if (response_fire) begin
                        if (transaction_beat_count < 8'(header_rows)) begin
                            if (entry_format_error) begin
                                if (!scan_validation_error) begin
                                    scan_validation_error <= 1'b1;
                                    scan_validation_error_id <= ERROR_ENTRY;
                                end
                            end else if (entry_group_error || entry_slot_error) begin
                                if (!scan_validation_error) begin
                                    scan_validation_error <= 1'b1;
                                    scan_validation_error_id <= ERROR_GROUP;
                                end
                            end else begin
                                loaded_token_ordinal[entry_physical_row] <=
                                    entry_token_ordinal;
                                loaded_position[entry_physical_row] <=
                                    entry_token_position;
                                scan_logical_seen[entry_token_ordinal] <= 1'b1;
                                scan_embedding_seen[entry_physical_row] <=
                                    entry_embedding_source;
                                if (!scan_group_seen[entry_compute_group]) begin
                                    scan_group_seen[entry_compute_group] <= 1'b1;
                                    loaded_query[entry_compute_group] <=
                                        entry_query_group;
                                    loaded_cache[entry_compute_group] <=
                                        entry_cache_group;
                                    loaded_compute_group[entry_compute_group] <=
                                        entry_bits == 4'd8 ? 6'd2 : 6'd1;
                                    loaded_source[entry_compute_group] <= {
                                        1'b0,
                                        entry_phase_mask[1] ?
                                            (8'b1 << entry_pe_slot) : 8'd0,
                                        entry_phase_mask[0] ?
                                            (8'b1 << entry_pe_slot) : 8'd0};
                                end else begin
                                    loaded_compute_group[entry_compute_group] <=
                                        loaded_compute_group[entry_compute_group] +
                                        (entry_bits == 4'd8 ? 6'd2 : 6'd1);
                                    if (entry_phase_mask[0])
                                        loaded_source[entry_compute_group]
                                            [{2'b00, entry_pe_slot}] <= 1'b1;
                                    if (entry_phase_mask[1])
                                        loaded_source[entry_compute_group]
                                            [{2'b01, entry_pe_slot}] <= 1'b1;
                                end
                            end
                        end else begin
                            for (integer lane = 0; lane < 16; lane++) begin
                                integer token_ordinal_number;
                                logic [5:0] physical_row_number;
                                token_ordinal_number =
                                    (integer'(transaction_beat_count) -
                                     integer'(header_rows)) * 16 + lane;
                                physical_row_number =
                                    dma_response_data[lane*8 +: 6];
                                if (token_ordinal_number < integer'(header_rows)) begin
                                    if (dma_response_data[lane*8 + 7 -: 2] != 0 ||
                                        physical_row_number >= header_rows ||
                                        loaded_token_ordinal[
                                            physical_row_number] !=
                                            6'(token_ordinal_number)) begin
                                        if (!scan_validation_error) begin
                                            scan_validation_error <= 1'b1;
                                            scan_validation_error_id <= ERROR_MAPPING;
                                        end
                                    end
                                end else if (dma_response_data[lane*8 +: 8] != 0) begin
                                    if (!scan_validation_error) begin
                                        scan_validation_error <= 1'b1;
                                        scan_validation_error_id <= ERROR_MAPPING;
                                    end
                                end
                            end
                        end
                    end
                    if (transaction_complete)
                        state <= SCAN_PAYLOAD_CHECK;
                end

                SCAN_PAYLOAD_CHECK: begin
                    integer semantic_count;
                    semantic_count = 0;
                    for (integer group = 0; group < 48; group++) begin
                        logic first_semantic;
                        first_semantic = group < header_compute_groups;
                        for (integer prior = 0; prior < group; prior++) begin
                            if (prior < header_compute_groups &&
                                loaded_query[prior] == loaded_query[group] &&
                                loaded_cache[prior] == loaded_cache[group])
                                first_semantic = 1'b0;
                        end
                        if (first_semantic)
                            semantic_count = semantic_count + 1;
                    end
                    if (transaction_error_seen || transaction_stream_error ||
                        transaction_beat_count != payload_beats ||
                        scan_validation_error ||
                        scan_logical_seen != low_mask(header_rows) ||
                        scan_group_seen != low_mask(header_compute_groups) ||
                        semantic_count != integer'(header_semantic_groups) ||
                        header_embedding_present !=
                            (scan_embedding_seen != 48'd0)) begin
                        done_error <= 1'b1;
                        done_error_id <= transaction_error_seen ? ERROR_DMA :
                            transaction_stream_error ? ERROR_STREAM :
                            scan_validation_error ? scan_validation_error_id :
                            ERROR_GROUP;
                        state <= COMPLETE;
                    end else if (scan_first_row + {6'd0, header_rows} ==
                                 total_token_count) begin
                        scan_complete <= 1'b1;
                        load_address <= metadata_base;
                        load_first_row <= '0;
                        token_batch_index <= '0;
                        state <= LOAD_HEADER_REQUEST;
                    end else if (scan_first_row + {6'd0, header_rows} >
                                 total_token_count || scan_token_batch_index == 6'd63) begin
                        done_error <= 1'b1;
                        done_error_id <= ERROR_HEADER;
                        state <= COMPLETE;
                    end else begin
                        scan_first_row <= scan_first_row +
                            {6'd0, header_rows};
                        scan_token_batch_index <= scan_token_batch_index + 6'd1;
                        scan_address <= scan_address +
                            {48'd0, header_token_batch_bytes};
                        state <= SCAN_HEADER_REQUEST;
                    end
                end

                LOAD_PAYLOAD_STREAM: begin
                    if (response_fire &&
                        transaction_beat_count < 8'(header_rows)) begin
                        if (entry_physical_row >= header_rows ||
                            entry_token_ordinal >= header_rows ||
                            entry_compute_group >= header_compute_groups ||
                            (output_subset_enable &&
                             ((entry_physical_row < output_token_count) !=
                                  (entry_token_ordinal < output_token_count) ||
                              (entry_physical_row < output_token_count) !=
                                  (entry_compute_group < output_compute_group_count)))) begin
                            if (!scan_validation_error) begin
                                scan_validation_error <= 1'b1;
                                scan_validation_error_id <= ERROR_ENTRY;
                            end
                        end else begin
                            loaded_source[entry_physical_row] <= entry_source_index;
                            loaded_position[entry_physical_row] <=
                                entry_token_position;
                            loaded_kv_index[entry_physical_row] <= entry_kv_index;
                            loaded_token_ordinal[entry_physical_row] <=
                                entry_token_ordinal;
                            loaded_embedding[entry_physical_row] <=
                                entry_embedding_source;
                            loaded_bits[entry_physical_row] <= entry_bits;
                            loaded_query[entry_physical_row] <= entry_query_group;
                            loaded_cache[entry_physical_row] <= entry_cache_group;
                            loaded_compute_group[entry_physical_row] <=
                                entry_compute_group;
                            loaded_pe_slot[entry_physical_row] <= entry_pe_slot;
                            loaded_phase_mask[entry_physical_row] <=
                                entry_phase_mask;
                            loaded_rows[entry_physical_row] <= 1'b1;
                        end
                    end else if (response_fire) begin
                        case (transaction_beat_count - 8'(header_rows))
                            8'd0: loaded_inverse_word[0] <= dma_response_data;
                            8'd1: loaded_inverse_word[1] <= dma_response_data;
                            8'd2: loaded_inverse_word[2] <= dma_response_data;
                            default: begin end
                        endcase
                    end
                    if (transaction_complete)
                        state <= LOAD_PAYLOAD_CHECK;
                end

                LOAD_PAYLOAD_CHECK: begin
                    if (transaction_error_seen || transaction_stream_error ||
                        transaction_beat_count != payload_beats ||
                        scan_validation_error ||
                        loaded_rows != low_mask(header_rows)) begin
                        done_error <= 1'b1;
                        done_error_id <= transaction_error_seen ? ERROR_DMA :
                            transaction_stream_error ? ERROR_STREAM :
                            scan_validation_error ? scan_validation_error_id :
                            ERROR_MAPPING;
                        state <= COMPLETE;
                    end else begin
                        state <= COMMIT;
                    end
                end

                COMMIT: begin
                    load_address <= load_address + {48'd0, header_token_batch_bytes};
                    load_first_row <= load_first_row + {6'd0, header_rows};
                    state <= COMPLETE;
                end

                COMPLETE: if (done_valid && done_ready) begin
                    state <= done_error ? IDLE : ACTIVE;
                    if (done_error)
                        scan_complete <= 1'b0;
                end

                ACTIVE: if (config_release) begin
                    loaded_rows <= '0;
                    active_token_count <= '0;
                    if (last_round) begin
                        scan_complete <= 1'b0;
                        total_token_count <= '0;
                        token_batch_index <= '0;
                        last_round <= 1'b0;
                    end else begin
                        token_batch_index <= token_batch_index + 6'd1;
                    end
                    state <= IDLE;
                end

                default: begin
                    done_error <= 1'b1;
                    done_error_id <= ERROR_STREAM;
                    state <= COMPLETE;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (rst)
        dma_request_valid && !dma_request_ready |=> dma_request_valid &&
            $stable({dma_request_address, dma_request_bytes}))
        else $error("token metadata v3 DMA request changed while stalled");
    assert property (@(posedge clk) disable iff (rst)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_error, done_error_id}))
        else $error("token metadata v3 completion changed while stalled");
`endif
endmodule

`default_nettype wire
