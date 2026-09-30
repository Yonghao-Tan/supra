`default_nettype none

module token_issue_packer #(
    parameter integer MAX_ROWS = 48
) (
    input  logic clk,
    input  logic rst,
    input  logic abort_request,
    output logic abort_ack,

    input  logic start_valid,
    output logic start_ready,
    input  logic [5:0] start_row_count,
    input  logic [11:0] start_sequence_length,
    input  logic [15:0] start_token_batch_index,
    input  logic [15:0] start_first_token_ordinal,
    input  logic [31:0] start_metadata_version,
    input  logic [31:0] start_capture_index,

    input  logic row_valid,
    output logic row_ready,
    input  logic [5:0] row_index,
    input  logic [16:0] row_source_index,
    input  logic [10:0] row_token_position,
    input  logic [10:0] row_kv_index,
    input  logic row_embedding_source,
    input  logic row_kv_write_disable,
    input  logic [3:0] activation_bits,
    input  logic [7:0] row_query_group,
    input  logic [7:0] row_cache_group,

    output logic beat_valid,
    input  logic beat_ready,
    output logic [127:0] beat_data,
    output logic beat_last,

    output logic done_valid,
    input  logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic [5:0] compute_group_count,
    output logic [5:0] semantic_group_count
);
    localparam logic [7:0] ERROR_START = 8'h01;
    localparam logic [7:0] ERROR_ROW = 8'h02;
    localparam logic [7:0] ERROR_DUPLICATE = 8'h03;
    localparam logic [7:0] ERROR_INTERNAL = 8'h04;

    typedef enum logic [3:0] {
        IDLE,
        LOAD_ROWS,
        CHECK_ROWS,
        FIND_GROUP,
        PACK_A8,
        PACK_A4,
        FINISH_GROUP,
        PREPARE_BEAT,
        SEND_BEAT,
        COMPLETE,
        ABORT_WAIT_LOW
    } state_t;

    state_t state;
    logic [5:0] saved_row_count;
    logic [11:0] saved_sequence_length;
    logic [15:0] saved_token_batch_index;
    logic [15:0] saved_first_token_ordinal;
    logic [31:0] saved_metadata_version;
    logic [31:0] saved_capture_index;
    logic embedding_source_present;

    logic [16:0] source_index_memory [0:MAX_ROWS-1];
    logic [10:0] token_position_memory [0:MAX_ROWS-1];
    logic [10:0] kv_index_memory [0:MAX_ROWS-1];
    logic embedding_source_memory [0:MAX_ROWS-1];
    logic kv_write_disable_memory [0:MAX_ROWS-1];
    logic [47:0] physical_kv_write_disable;
    logic [3:0] activation_bits_memory [0:MAX_ROWS-1];
    logic [7:0] query_group_memory [0:MAX_ROWS-1];
    logic [7:0] cache_group_memory [0:MAX_ROWS-1];
    logic [5:0] physical_logical_index [0:MAX_ROWS-1];
    logic [1:0] physical_stage [0:MAX_ROWS-1];
    logic [3:0] physical_pe_slot [0:MAX_ROWS-1];
    logic [5:0] physical_compute_group [0:MAX_ROWS-1];
    logic [5:0] logical_to_physical [0:MAX_ROWS-1];
    logic [MAX_ROWS-1:0] assigned_rows;

    logic [5:0] load_count;
    logic [5:0] duplicate_outer;
    logic [5:0] duplicate_inner;
    logic duplicate_group_seen;
    logic [5:0] assigned_count;
    logic [5:0] physical_count;
    logic [5:0] find_index;
    logic [5:0] pack_scan_index;
    logic [7:0] active_query_group;
    logic [7:0] active_cache_group;
    logic [3:0] active_a8_count;
    logic [3:0] active_phase0_a4_count;
    logic [3:0] active_phase1_a4_count;
    logic [6:0] stream_index;
    logic [6:0] stream_beat_count;
    logic [127:0] stream_data;
    logic terminal_error;
    logic [7:0] terminal_error_id;

    logic row_fire;
    logic beat_fire;
    logic duplicate_position;
    logic duplicate_group;
    logic selected_assigned;
    logic selected_group_match;
    logic selected_a8;
    logic selected_a4;
    logic [3:0] a4_phase_capacity;
    logic [15:0] inverse_offset;
    logic [15:0] inverse_bytes;
    logic [15:0] token_batch_bytes;
    logic [5:0] stream_physical_index;
    logic [5:0] stream_logical_index;
    logic [127:0] stream_physical_row;

    assign start_ready = state == IDLE && !abort_request;
    assign row_ready = state == LOAD_ROWS && !abort_request;
    assign row_fire = row_valid && row_ready;
    assign beat_valid = state == SEND_BEAT && !abort_request;
    assign beat_data = stream_data;
    assign beat_last = stream_index + 7'd1 == stream_beat_count;
    assign beat_fire = beat_valid && beat_ready;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;

    assign duplicate_position =
        token_position_memory[duplicate_outer] ==
        token_position_memory[duplicate_inner];
    assign duplicate_group =
        query_group_memory[duplicate_outer] ==
            query_group_memory[duplicate_inner] &&
        cache_group_memory[duplicate_outer] ==
            cache_group_memory[duplicate_inner];
    assign selected_assigned = assigned_rows[pack_scan_index];
    assign selected_group_match =
        query_group_memory[pack_scan_index] == active_query_group &&
        cache_group_memory[pack_scan_index] == active_cache_group;
    assign selected_a8 = !selected_assigned && selected_group_match &&
        activation_bits_memory[pack_scan_index] == 4'd8;
    assign selected_a4 = !selected_assigned && selected_group_match &&
        activation_bits_memory[pack_scan_index] == 4'd4;
    assign a4_phase_capacity = 4'd8 - active_a8_count;
    assign inverse_offset = 16'd32 + {6'd0, saved_row_count, 4'd0};
    assign inverse_bytes = {10'd0, (saved_row_count + 6'd15) & 6'h30};
    assign stream_beat_count = 7'd2 + 7'(saved_row_count) +
        7'((saved_row_count + 6'd15) >> 4);
    assign token_batch_bytes = 16'(stream_beat_count) << 4;

    always_comb begin
        stream_physical_index = '0;
        stream_logical_index = '0;
        stream_physical_row = '0;
        if (stream_index >= 7'd2 &&
            stream_index < 7'd2 + 7'(saved_row_count)) begin
            stream_physical_index = 6'(stream_index - 7'd2);
            stream_logical_index =
                physical_logical_index[stream_physical_index];
            stream_physical_row = {
                {6'd0, physical_stage[stream_physical_index]},
                {4'd0, physical_pe_slot[stream_physical_index]},
                {2'd0, physical_compute_group[stream_physical_index]},
                cache_group_memory[stream_logical_index],
                query_group_memory[stream_logical_index],
                {4'd0, activation_bits_memory[stream_logical_index]},
                {7'd0, embedding_source_memory[stream_logical_index]},
                {2'd0, stream_logical_index},
                {5'd0, kv_index_memory[stream_logical_index]},
                {5'd0, token_position_memory[stream_logical_index]},
                {15'd0, source_index_memory[stream_logical_index]}};
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_row_count <= '0;
            saved_sequence_length <= '0;
            saved_token_batch_index <= '0;
            saved_first_token_ordinal <= '0;
            saved_metadata_version <= '0;
            saved_capture_index <= '0;
            embedding_source_present <= 1'b0;
            assigned_rows <= '0;
            physical_kv_write_disable <= '0;
            load_count <= '0;
            duplicate_outer <= '0;
            duplicate_inner <= '0;
            duplicate_group_seen <= 1'b0;
            assigned_count <= '0;
            physical_count <= '0;
            find_index <= '0;
            pack_scan_index <= '0;
            active_query_group <= '0;
            active_cache_group <= '0;
            active_a8_count <= '0;
            active_phase0_a4_count <= '0;
            active_phase1_a4_count <= '0;
            stream_index <= '0;
            stream_data <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            compute_group_count <= '0;
            semantic_group_count <= '0;
            abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (abort_request && state != IDLE && state != COMPLETE &&
                state != ABORT_WAIT_LOW) begin
                abort_ack <= 1'b1;
                state <= ABORT_WAIT_LOW;
            end else begin
                case (state)
                    IDLE: if (start_valid && start_ready) begin
                        saved_row_count <= start_row_count;
                        saved_sequence_length <= start_sequence_length;
                        saved_token_batch_index <= start_token_batch_index;
                        saved_first_token_ordinal <= start_first_token_ordinal;
                        saved_metadata_version <= start_metadata_version;
                        saved_capture_index <= start_capture_index;
                        embedding_source_present <= 1'b0;
                        assigned_rows <= '0;
            physical_kv_write_disable <= '0;
                        load_count <= '0;
                        assigned_count <= '0;
                        physical_count <= '0;
                        compute_group_count <= '0;
                        semantic_group_count <= '0;
                        terminal_error <= 1'b0;
                        terminal_error_id <= '0;
                        if (start_row_count == 0 ||
                            start_row_count > 6'(MAX_ROWS) ||
                            start_sequence_length == 0 || start_sequence_length > 12'd2048) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_START;
                            state <= COMPLETE;
                        end else begin
                            state <= LOAD_ROWS;
                        end
                    end

                    LOAD_ROWS: if (row_fire) begin
                        if (row_index != load_count ||
                            (row_embedding_source &&
                             row_source_index >= 17'd126464) ||
                            (!row_embedding_source &&
                             row_source_index >= 17'd2048) ||
                            (activation_bits != 4'd4 && activation_bits != 4'd8) ||
                            {1'b0, row_token_position} >= saved_sequence_length ||
                            {1'b0, row_kv_index} >= saved_sequence_length) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_ROW;
                            state <= COMPLETE;
                        end else begin
                            source_index_memory[load_count] <= row_source_index;
                            token_position_memory[load_count] <=
                                row_token_position;
                            kv_index_memory[load_count] <= row_kv_index;
                            kv_write_disable_memory[load_count] <= row_kv_write_disable;
                            embedding_source_memory[load_count] <=
                                row_embedding_source;
                            activation_bits_memory[load_count] <= activation_bits;
                            query_group_memory[load_count] <= row_query_group;
                            cache_group_memory[load_count] <= row_cache_group;
                            embedding_source_present <=
                                embedding_source_present || row_embedding_source;
                            if (load_count + 6'd1 == saved_row_count) begin
                                if (saved_row_count == 6'd1) begin
                                    semantic_group_count <= 6'd1;
                                    find_index <= '0;
                                    state <= FIND_GROUP;
                                end else begin
                                    duplicate_outer <= 6'd1;
                                    duplicate_inner <= '0;
                                    duplicate_group_seen <= 1'b0;
                                    semantic_group_count <= 6'd1;
                                    state <= CHECK_ROWS;
                                end
                            end else begin
                                load_count <= load_count + 6'd1;
                            end
                        end
                    end

                    CHECK_ROWS: begin
                        if (duplicate_position) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_DUPLICATE;
                            state <= COMPLETE;
                        end else if (duplicate_inner + 6'd1 == duplicate_outer) begin
                            if (!(duplicate_group_seen || duplicate_group))
                                semantic_group_count <=
                                    semantic_group_count + 6'd1;
                            duplicate_group_seen <= 1'b0;
                            duplicate_inner <= '0;
                            if (duplicate_outer + 6'd1 == saved_row_count) begin
                                find_index <= '0;
                                state <= FIND_GROUP;
                            end else begin
                                duplicate_outer <= duplicate_outer + 6'd1;
                            end
                        end else begin
                            duplicate_group_seen <=
                                duplicate_group_seen || duplicate_group;
                            duplicate_inner <= duplicate_inner + 6'd1;
                        end
                    end

                    FIND_GROUP: begin
                        if (assigned_count == saved_row_count) begin
                            if (physical_count != saved_row_count) begin
                                terminal_error <= 1'b1;
                                terminal_error_id <= ERROR_INTERNAL;
                                state <= COMPLETE;
                            end else begin
                                stream_index <= '0;
                                state <= PREPARE_BEAT;
                            end
                        end else if (!assigned_rows[find_index]) begin
                            active_query_group <= query_group_memory[find_index];
                            active_cache_group <= cache_group_memory[find_index];
                            active_a8_count <= '0;
                            active_phase0_a4_count <= '0;
                            active_phase1_a4_count <= '0;
                            pack_scan_index <= '0;
                            state <= PACK_A8;
                        end else if (find_index + 6'd1 == saved_row_count) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_INTERNAL;
                            state <= COMPLETE;
                        end else begin
                            find_index <= find_index + 6'd1;
                        end
                    end

                    PACK_A8: begin
                        if (selected_a8 && active_a8_count < 4'd8) begin
                            physical_kv_write_disable[physical_count] <= kv_write_disable_memory[pack_scan_index];
                            physical_logical_index[physical_count] <=
                                pack_scan_index;
                            physical_stage[physical_count] <= 2'd3;
                            physical_pe_slot[physical_count] <=
                                active_a8_count;
                            physical_compute_group[physical_count] <=
                                compute_group_count;
                            logical_to_physical[pack_scan_index] <= physical_count;
                            assigned_rows[pack_scan_index] <= 1'b1;
                            assigned_count <= assigned_count + 6'd1;
                            physical_count <= physical_count + 6'd1;
                            active_a8_count <= active_a8_count + 4'd1;
                            if (active_a8_count == 4'd7) begin
                                state <= FINISH_GROUP;
                            end else if (pack_scan_index + 6'd1 ==
                                         saved_row_count) begin
                                pack_scan_index <= '0;
                                state <= PACK_A4;
                            end else begin
                                pack_scan_index <= pack_scan_index + 6'd1;
                            end
                        end else if (pack_scan_index + 6'd1 ==
                                     saved_row_count) begin
                            pack_scan_index <= '0;
                            state <= PACK_A4;
                        end else begin
                            pack_scan_index <= pack_scan_index + 6'd1;
                        end
                    end

                    PACK_A4: begin
                        if (selected_a4 &&
                            (active_phase0_a4_count < a4_phase_capacity ||
                             active_phase1_a4_count < a4_phase_capacity)) begin
                            logic use_phase1;
                            logic [3:0] phase_row;
                            use_phase1 =
                                active_phase0_a4_count >= a4_phase_capacity;
                            phase_row = use_phase1 ? active_phase1_a4_count :
                                active_phase0_a4_count;
                            physical_kv_write_disable[physical_count] <= kv_write_disable_memory[pack_scan_index];
                            physical_logical_index[physical_count] <=
                                pack_scan_index;
                            physical_stage[physical_count] <=
                                use_phase1 ? 2'd2 : 2'd1;
                            physical_pe_slot[physical_count] <=
                                active_a8_count + phase_row;
                            physical_compute_group[physical_count] <=
                                compute_group_count;
                            logical_to_physical[pack_scan_index] <= physical_count;
                            assigned_rows[pack_scan_index] <= 1'b1;
                            assigned_count <= assigned_count + 6'd1;
                            physical_count <= physical_count + 6'd1;
                            if (use_phase1) begin
                                active_phase1_a4_count <=
                                    active_phase1_a4_count + 4'd1;
                                if (active_phase1_a4_count + 4'd1 ==
                                    a4_phase_capacity)
                                    state <= FINISH_GROUP;
                                else if (pack_scan_index + 6'd1 ==
                                         saved_row_count)
                                    state <= FINISH_GROUP;
                                else
                                    pack_scan_index <= pack_scan_index + 6'd1;
                            end else begin
                                active_phase0_a4_count <=
                                    active_phase0_a4_count + 4'd1;
                                if (pack_scan_index + 6'd1 == saved_row_count)
                                    state <= FINISH_GROUP;
                                else
                                    pack_scan_index <= pack_scan_index + 6'd1;
                            end
                        end else if (pack_scan_index + 6'd1 ==
                                     saved_row_count) begin
                            state <= FINISH_GROUP;
                        end else begin
                            pack_scan_index <= pack_scan_index + 6'd1;
                        end
                    end

                    FINISH_GROUP: begin
                        if (active_a8_count == 0 &&
                            active_phase0_a4_count == 0 &&
                            active_phase1_a4_count == 0) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_INTERNAL;
                            state <= COMPLETE;
                        end else begin
                            compute_group_count <= compute_group_count + 6'd1;
                            find_index <= '0;
                            state <= FIND_GROUP;
                        end
                    end

                    PREPARE_BEAT: begin
                        stream_data <= '0;
                        if (stream_index == 0) begin
                            stream_data[7:0] <= {2'd0, saved_row_count};
                            stream_data[15:8] <= {2'd0, compute_group_count};
                            stream_data[23:16] <= {2'd0, semantic_group_count};
                            stream_data[31:24] <=
                                {7'd0, embedding_source_present};
                            stream_data[47:32] <= saved_token_batch_index;
                            stream_data[63:48] <= saved_first_token_ordinal;
                            stream_data[79:64] <= token_batch_bytes;
                            stream_data[95:80] <= 16'd16;
                            stream_data[111:96] <= inverse_offset;
                            stream_data[127:112] <= inverse_bytes;
                        end else if (stream_index == 1) begin
                            stream_data[31:0] <= saved_metadata_version;
                            stream_data[63:32] <= saved_capture_index;
                            stream_data[111:64] <= physical_kv_write_disable;
                        end else if (stream_index <
                                     7'd2 + 7'(saved_row_count)) begin
                            stream_data <= stream_physical_row;
                        end else begin
                            for (integer lane = 0; lane < 16; lane++) begin
                                integer inverse_index;
                                inverse_index =
                                    (integer'(stream_index) - 2 -
                                     integer'(saved_row_count)) * 16 + lane;
                                if (inverse_index < integer'(saved_row_count))
                                    stream_data[lane*8 +: 8] <= {2'd0,
                                        logical_to_physical[6'(inverse_index)]};
                            end
                        end
                        state <= SEND_BEAT;
                    end

                    SEND_BEAT: if (beat_fire) begin
                        if (beat_last)
                            state <= COMPLETE;
                        else begin
                            stream_index <= stream_index + 7'd1;
                            state <= PREPARE_BEAT;
                        end
                    end

                    COMPLETE: if (done_valid && done_ready)
                        state <= IDLE;
                    ABORT_WAIT_LOW: if (!abort_request)
                        state <= IDLE;
                    default: begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_INTERNAL;
                        state <= COMPLETE;
                    end
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (rst)
        beat_valid && !beat_ready |=> beat_valid &&
            $stable({beat_data, beat_last}))
        else $error("next token metadata beat changed while stalled");
    assert property (@(posedge clk) disable iff (rst)
        beat_valid |-> !terminal_error)
        else $error("token issue packer emitted metadata after an error");
`endif
endmodule

`default_nettype wire
