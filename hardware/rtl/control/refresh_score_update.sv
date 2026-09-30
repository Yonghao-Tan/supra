`default_nettype none

// Read persistent dependency rows, apply this forward's changes, and publish
// pending vectors in place. Changes occupy 18 bits x 96 = 216 B because every
// query reuses the same keys. One 16-byte pending entry and one DMA beat are
// retained; the dependency matrix remains in DDR. All scalar math uses the
// existing shared BF16 pipe, independently of the activation SRAM ports.
module refresh_score_update (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [11:0] start_rows, start_block_end,
    input logic [6:0] start_keys,
    input logic [3:0] start_relation_row_shift,
    input logic start_all_changes,
    input logic start_live_consumed, input logic live_consumed,
    input logic start_clear_relation, start_advance_block,
    input logic [1:0] start_confidence_mode, // Cross-block: 0 all_changes, 1 stable_unmask, 2 remask_only.
    input logic start_state_changes,
    input logic start_update_row_table, start_precision_enabled,
    input logic [63:0] start_row_table_base, start_row_table_limit,
    input logic [6:0] state_read_index,
    output logic state_read_valid,
    output logic [2:0] state_read_value,
    output logic [15:0] state_read_confidence,
    output logic state_read_confidence_valid,
    output logic state_read_source_a_pending,
    input logic [11:0] start_change_position_begin,
    input logic [31:0] start_change_capture_index,
    input logic [63:0] start_relation_base, start_relation_limit,
    input logic [63:0] start_change_base, start_change_limit,
    input logic [63:0] start_pending_base, start_pending_limit,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic score_valid,
    output logic [11:0] score_row,
    output logic [15:0] score_bf16,
    output logic score_mandatory,
    output logic read_request_valid,
    input logic read_request_ready,
    output logic [63:0] read_request_address,
    output logic [31:0] read_request_bytes,
    output logic [7:0] read_request_tag,
    input logic read_valid,
    output logic read_ready,
    input logic [127:0] read_data,
    input logic [15:0] read_byte_enable,
    input logic read_last,
    input logic [7:0] read_tag,
    input logic read_error, read_abort_ack,
    output logic write_request_valid,
    input logic write_request_ready,
    output logic [63:0] write_request_address,
    output logic [31:0] write_request_bytes,
    output logic [7:0] write_request_tag,
    output logic write_valid,
    input logic write_ready,
    output logic [127:0] write_data,
    output logic [15:0] write_byte_enable,
    output logic write_last,
    input logic write_done, write_error,
    output logic arithmetic_valid,
    input logic arithmetic_ready,
    output logic arithmetic_multiply,
    output logic [15:0] arithmetic_lhs, arithmetic_rhs,
    input logic arithmetic_result_valid,
    output logic arithmetic_result_ready,
    input logic [15:0] arithmetic_result,
    output logic arithmetic_abort_request,
    input logic arithmetic_abort_ack
);
    typedef enum logic [4:0] {
        IDLE, RANGE, CHECK, CHANGE_START, CHANGE_WAIT, PENDING_START,
        PENDING_WAIT, REDUCE_START, RELATION_START, KEY_READ, KEY_SEND,
        RELATION_DRAIN, REDUCE_WAIT, WRITE_START, WRITE_STREAM, WRITE_WAIT,
        NEXT_TOKEN, COMPLETE, ABORT_DRAIN, ABORT_LOW,
        TABLE_READ_START, TABLE_WRITE_START, TABLE_LOAD, TABLE_LOW, TABLE_HIGH, TABLE_STREAM, TABLE_DRAIN,
        CLEAR_RELATION_START, CLEAR_RELATION_STREAM, CLEAR_RELATION_WAIT
    } state_t;
    state_t state;
    logic [11:0] rows, block_end, row_index;
    logic [6:0] keys, changed_count, key_index, change_index;
    logic [8:0] row_stride;
    logic all_changes, terminal_error, aborting;
    logic clear_relation, advance_block;
    logic [19:0] clear_byte_offset;
    logic clear_writer_start, clear_writer_stream;
    logic [1:0] confidence_mode;
    logic [7:0] terminal_error_id;
    logic [63:0] relation_address, change_address, pending_address;
    logic [63:0] relation_base, relation_limit, change_limit, pending_base, pending_limit;
    logic [64:0] relation_end, change_end, pending_end;
    logic [19:0] relation_span;
    logic [31:0] relation_bytes, change_bytes;
    logic [17:0] changes [0:95];
    // 39 bits x 96 state entries reuse the change table's key index and valid
    // mask. The existing pending_entry holds each table beat during rewriting.
    logic [38:0] state_rows [0:95];
    logic [19:0] state_low_payload;
    logic update_row_table, precision_enabled;
    logic [63:0] row_table_address, row_table_limit;
    logic [64:0] row_table_start, row_table_end;
    logic [31:0] row_table_bytes;
    logic table_reader_finished, table_writer_finished;
    logic [11:0] table_position;
    logic [20:0] table_state;
    logic table_writer_start, table_writer_stream;
    logic [127:0] table_beat;
    logic [95:0] change_seen;
    logic state_changes, state_change_high;
    logic [11:0] change_position_begin, state_change_position;
    logic [31:0] change_capture_index;
    logic [12:0] change_position_end;
    logic [11:0] state_change_index;
    logic state_change_in_range;
    logic [17:0] key_change;
    logic [127:0] pending_entry;
    logic [1:0] change_lane;
    logic [31:0] change_record;
    logic reader_start_valid, reader_start_ready, reader_valid, reader_ready;
    logic reader_done, reader_error, reader_abort_ack;
    logic [127:0] reader_data;
    logic [63:0] reader_address;
    logic [31:0] reader_bytes;
    logic writer_start_ready, writer_ready, writer_done, writer_error, writer_abort_ack;
    logic kernel_start_ready, kernel_value_ready, kernel_result_valid, kernel_error, kernel_abort_ack;
    logic [15:0] invalidation, remask, pending_result;
    logic use_live_consumed;
    logic stop_children, future_row, consumed, key_accepted, kernel_abort_seen;

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign score_valid = state == REDUCE_WAIT && kernel_result_valid && !kernel_error && !stop_children;
    assign score_row = row_index;
    assign score_bf16 = pending_result;
    assign score_mandatory = pending_entry[81] ||
        (!all_changes && row_index < {5'd0, keys} && change_seen[row_index[6:0]] && changes[row_index[6:0]][16]);
    assign change_position_end = {1'b0, change_position_begin} + {6'd0, keys};
    assign state_change_index = state_change_position-change_position_begin;
    assign state_change_in_range = state_change_position >= change_position_begin && {1'b0, state_change_position} < change_position_end;
    assign table_position = change_position_begin + {5'd0, change_index};
    assign table_state = state_rows[change_index][20:0];
    assign state_read_valid = update_row_table && state_read_index < keys && change_seen[state_read_index];
    assign state_read_value = state_rows[state_read_index][19:17];
    assign state_read_confidence = state_rows[state_read_index][36:21];
    assign state_read_confidence_valid = state_rows[state_read_index][37];
    assign state_read_source_a_pending = state_read_valid && state_rows[state_read_index][38];
    assign row_table_start = {1'b0, start_row_table_base} + {50'd0, start_change_position_begin, 3'd0};
    assign row_table_end = {1'b0, row_table_address} + {33'd0, row_table_bytes};
    assign clear_writer_start = state == CLEAR_RELATION_START;
    assign clear_writer_stream = state == CLEAR_RELATION_STREAM;
    assign table_writer_start = state == TABLE_WRITE_START;
    assign table_writer_stream = state == TABLE_STREAM;
    function automatic logic [63:0] state_record(input logic [63:0] old_record,
            input logic [20:0] value, input logic [11:0] position_value, input logic source_a_pending);
        logic [63:0] result;
        logic current_row;
        begin
            current_row = position_value >= block_end-12'd32 && position_value < block_end;
            result = old_record;
            result[8] = current_row;
            result[9] = !current_row;
            result[10] = value[20] && !(source_a_pending && !current_row);
            result[54] = source_a_pending && !current_row;
            result[34] = value[19];
            result[51:35] = value[16:0];
            result[52] = (current_row && value[18:17] == 0) || (!precision_enabled && value[19]);
            result[53] = current_row && value[18:17] != 2;
            state_record = result;
        end
    endfunction
    always_comb begin
        table_beat = pending_entry;
        if (change_index < keys && change_seen[change_index]) begin
            if (state == TABLE_LOW)
                table_beat[63:0] = state_record(pending_entry[63:0], table_state, table_position, state_rows[change_index][38]);
            else
                table_beat[127:64] = state_record(pending_entry[127:64], table_state, table_position, state_rows[change_index][38]);
        end
    end
    assign stop_children = abort_request || state == ABORT_DRAIN;
    assign future_row = all_changes && row_index >= block_end;
    assign consumed = use_live_consumed ? live_consumed : pending_entry[80];
    assign change_record = reader_data[change_lane*32 +: 32];
    assign key_accepted = reader_valid && (key_index >= keys || !key_change[16] || kernel_value_ready);
    assign reader_start_valid = !stop_children &&
        (state == CHANGE_START || state == PENDING_START || state == RELATION_START || state == TABLE_READ_START);
    assign reader_address = state == TABLE_READ_START ? row_table_address : state == CHANGE_START ? change_address : state == PENDING_START ? pending_address : relation_address;
    assign reader_bytes = state == TABLE_READ_START ? row_table_bytes : state == CHANGE_START ? change_bytes : state == PENDING_START ? 32'd16 : relation_bytes;
    assign reader_ready = state == CHANGE_WAIT ? state_changes || change_lane == 2'd3 :
        state == KEY_SEND ? key_accepted && key_index[2:0] == 3'd7 :
        state == PENDING_WAIT || state == TABLE_LOAD || state == ABORT_DRAIN;

    operator_dma_reader #(.REQUEST_TAG(8'hba)) reader (
        .clk, .rst, .abort_request(stop_children), .start_valid(reader_start_valid), .start_ready(reader_start_ready),
        .start_address(reader_address), .start_bytes(reader_bytes), .start_total_bytes(reader_bytes),
        .request_valid(read_request_valid), .request_ready(read_request_ready), .request_address(read_request_address),
        .request_bytes(read_request_bytes), .request_tag(read_request_tag), .read_valid, .read_ready, .read_data,
        .read_byte_enable, .read_last, .read_tag, .response_error(read_error), .upstream_abort_ack(read_abort_ack),
        .data_valid(reader_valid), .data_ready(reader_ready), .data(reader_data), .data_byte_enable(),
        .data_last(), .data_byte_offset(), .done_pulse(reader_done), .error(reader_error), .abort_ack(reader_abort_ack)
    );
    operator_dma_writer #(.REQUEST_TAG(8'hbb)) writer (
        .clk, .rst, .abort_request(stop_children), .start_valid((state == WRITE_START || table_writer_start || clear_writer_start) && !stop_children),
        .start_ready(writer_start_ready), .start_address(clear_writer_start ? relation_base : table_writer_start ? row_table_address : pending_address),
        .start_bytes(clear_writer_start ? {12'd0, relation_span} : table_writer_start ? row_table_bytes : 32'd16),
        .request_valid(write_request_valid), .request_ready(write_request_ready), .request_address(write_request_address),
        .request_bytes(write_request_bytes), .request_tag(write_request_tag),
        .data_valid(state == WRITE_STREAM || table_writer_stream || clear_writer_stream), .data_ready(writer_ready),
        .data(clear_writer_stream ? 128'd0 : pending_entry),
        .data_byte_enable(table_writer_stream && change_index > keys ? 16'h00ff : 16'hffff),
        .data_last(clear_writer_stream ? clear_byte_offset + 20'd16 == relation_span : !table_writer_stream || change_index >= keys), .write_valid, .write_ready, .write_data,
        .write_byte_enable, .write_last, .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_ack)
    );
    attention_invalidation kernel (
        .clk, .rst, .abort_request(stop_children), .abort_ack(kernel_abort_ack),
        .start_valid(state == REDUCE_START && !stop_children), .start_ready(kernel_start_ready),
        .start_all_changes(all_changes && confidence_mode == 0),
        .start_stable_unmask(all_changes && confidence_mode == 1), .start_changed_count({5'd0, changed_count}),
        .start_pending(future_row ? pending_entry[63:48] : pending_entry[15:0]), .start_consumed(consumed),
        .value_valid(state == KEY_SEND && reader_valid && key_index < keys && key_change[16]),
        .value_ready(kernel_value_ready), .value_dependency(reader_data[key_index[2:0]*16 +: 16]),
        .value_confidence(key_change[15:0]), .value_remasked(key_change[17]),
        .arithmetic_valid, .arithmetic_ready, .arithmetic_multiply, .arithmetic_lhs, .arithmetic_rhs,
        .arithmetic_result_valid, .arithmetic_result_ready, .arithmetic_result,
        .arithmetic_abort_request, .arithmetic_abort_ack,
        .result_valid(kernel_result_valid), .result_ready(state == REDUCE_WAIT),
        .result_invalidation(invalidation), .result_remask(remask), .result_pending(pending_result), .error(kernel_error)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            use_live_consumed <= 1'b0;
            state <= IDLE; rows <= '0; block_end <= '0; row_index <= '0;
            keys <= '0; changed_count <= '0; key_index <= '0; change_index <= '0;
            clear_relation <= 1'b0; advance_block <= 1'b0; clear_byte_offset <= '0;
            row_stride <= '0; all_changes <= 1'b0; confidence_mode <= '0; terminal_error <= 1'b0; aborting <= 1'b0;
            terminal_error_id <= '0; relation_address <= '0; change_address <= '0; pending_address <= '0;
            relation_base <= '0; relation_limit <= '0; change_limit <= '0; pending_base <= '0; pending_limit <= '0;
            relation_end <= '0; change_end <= '0; pending_end <= '0; relation_span <= '0;
            relation_bytes <= '0; change_bytes <= '0; key_change <= '0; pending_entry <= '0;
            change_lane <= '0; abort_ack <= 1'b0;
            change_seen <= '0; state_changes <= 1'b0; state_change_high <= 1'b0;
            change_position_begin <= '0; state_change_position <= '0;
            change_capture_index <= '0;
            kernel_abort_seen <= 1'b0;
            state_low_payload <= '0; update_row_table <= 1'b0; precision_enabled <= 1'b0;
            row_table_address <= '0; row_table_limit <= '0; row_table_bytes <= '0;
            table_reader_finished <= 1'b0; table_writer_finished <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (reader_done) table_reader_finished <= 1'b1;
            if (writer_done) table_writer_finished <= 1'b1;
            if (kernel_abort_ack) kernel_abort_seen <= 1'b1;
            if ((abort_request || reader_error || writer_error || kernel_error) &&
                state != IDLE && state != ABORT_DRAIN && state != ABORT_LOW && state != COMPLETE) begin
                aborting <= abort_request; terminal_error <= !abort_request;
                terminal_error_id <= reader_error ? 8'h03 : writer_error ? 8'h04 : 8'h05;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    rows <= start_rows; keys <= start_keys; block_end <= start_block_end;
                    clear_relation <= start_clear_relation; advance_block <= start_advance_block; clear_byte_offset <= '0;
                    all_changes <= start_all_changes; confidence_mode <= start_confidence_mode; row_index <= '0; changed_count <= '0;
                    change_index <= '0; change_lane <= '0; terminal_error <= 1'b0; terminal_error_id <= '0;
                    change_seen <= '0; state_changes <= start_state_changes; state_change_high <= 1'b0;
                    change_position_begin <= start_change_position_begin;
                    change_capture_index <= start_change_capture_index;
                    update_row_table <= start_update_row_table; precision_enabled <= start_precision_enabled;
                    row_table_address <= row_table_start[63:0]; row_table_limit <= start_row_table_limit;
                    row_table_bytes <= {22'd0, start_keys, 3'd0};
                    kernel_abort_seen <= 1'b0;
                    use_live_consumed <= start_live_consumed;
                    relation_address <= start_relation_base; relation_base <= start_relation_base; relation_limit <= start_relation_limit;
                    change_address <= start_change_base; change_limit <= start_change_limit;
                    pending_address <= start_pending_base; pending_base <= start_pending_base; pending_limit <= start_pending_limit;
                    case (start_relation_row_shift)
                        4'd6: begin row_stride <= 9'd64; relation_span <= {2'd0, start_rows, 6'd0}; end
                        4'd7: begin row_stride <= 9'd128; relation_span <= {1'd0, start_rows, 7'd0}; end
                        default: begin row_stride <= 9'd256; relation_span <= {start_rows, 8'd0}; end
                    endcase
                    relation_bytes <= {24'd0, (({1'b0, start_keys}+8'd7)>>3)} << 4;
                    change_bytes <= start_state_changes ? 32'(start_change_limit-start_change_base) :
                        {24'd0, (({1'b0, start_keys}+8'd3)>>2)} << 4;
                    if (start_rows == 0 || start_rows > 2048 || start_keys == 0 || start_keys > 96 ||
                        start_relation_row_shift < 6 || start_relation_row_shift > 8 ||
                        (!start_update_row_table && start_block_end > start_rows) ||
                        (start_update_row_table && (!start_state_changes || start_all_changes ||
                            row_table_start[64] || start_row_table_base[3:0] != 0 ||
                            {1'b0, start_block_end} < {1'b0, start_change_position_begin}+13'd32 ||
                            {1'b0, start_block_end} > {1'b0, start_change_position_begin}+{6'd0, start_keys})) ||
                        (start_clear_relation && !start_advance_block) ||
                        (start_advance_block && (!start_all_changes || start_update_row_table)) ||
                        (start_confidence_mode == 3 || (!start_all_changes && start_confidence_mode != 0)) ||
                        (start_live_consumed && !start_all_changes) ||
                        (start_all_changes && start_keys != 32) || start_relation_base[3:0] != 0 ||
                        start_change_base[3:0] != 0 || start_pending_base[3:0] != 0 ||
                        (start_state_changes && (start_change_limit <= start_change_base ||
                         start_change_limit-start_change_base > 4096 || (start_change_limit-start_change_base) % 32 != 0 ||
                         {1'b0, start_change_position_begin}+{6'd0, start_keys} > 2048))) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                    end else state <= RANGE;
                end
                RANGE: begin
                    relation_end <= {1'b0, relation_base} + {45'd0, relation_span};
                    change_end <= {1'b0, change_address} + {33'd0, change_bytes};
                    pending_end <= {1'b0, pending_base} + {49'd0, rows, 4'd0};
                    state <= CHECK;
                end
                CHECK: begin
                    if (relation_bytes > {23'd0, row_stride} || relation_end[64] || relation_end[63:0] > relation_limit ||
                        change_end[64] || change_end[63:0] > change_limit || pending_end[64] || pending_end[63:0] > pending_limit ||
                        (update_row_table && (row_table_end[64] || row_table_end[63:0] > row_table_limit ||
                            (row_table_address < pending_end[63:0] && pending_base < row_table_end[63:0]) ||
                            (row_table_address < relation_end[63:0] && relation_base < row_table_end[63:0]) ||
                            (row_table_address < change_end[63:0] && change_address < row_table_end[63:0]))) ||
                        (pending_base < relation_end[63:0] && relation_base < pending_end[63:0]) ||
                        (pending_base < change_end[63:0] && change_address < pending_end[63:0])) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h02; state <= COMPLETE;
                    end else state <= clear_relation ? CLEAR_RELATION_START : CHANGE_START;
                end
                CLEAR_RELATION_START: if (writer_start_ready) state <= CLEAR_RELATION_STREAM;
                CLEAR_RELATION_STREAM: if (writer_ready) begin
                    clear_byte_offset <= clear_byte_offset + 20'd16;
                    if (clear_byte_offset + 20'd16 == relation_span) state <= CLEAR_RELATION_WAIT;
                end
                CLEAR_RELATION_WAIT: if (writer_done) state <= COMPLETE;
                CHANGE_START: if (reader_start_ready) state <= CHANGE_WAIT;
                CHANGE_WAIT: begin
                    if (reader_valid && state_changes) begin
                        state_change_high <= !state_change_high;
                        if (!state_change_high) begin
                            state_change_position <= reader_data[11:0];
                            state_low_payload <= {reader_data[59:56] == 8, reader_data[41:40], reader_data[80:64]};
                            if (reader_data[15:11] != 0 || (update_row_table && (reader_data[47:40] > 2 ||
                                (reader_data[63:56] != 4 && reader_data[63:56] != 8) ||
                                reader_data[95:64] >= 126464))) begin
                                terminal_error <= 1'b1; terminal_error_id <= 8'h06; state <= ABORT_DRAIN; aborting <= 1'b0;
                            end
                        end else if (reader_data[63:32] != change_capture_index) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h06; state <= ABORT_DRAIN; aborting <= 1'b0;
                        end else if (state_change_in_range) begin
                            if (update_row_table) state_rows[state_change_index[6:0]] <=
                                {reader_data[120], reader_data[112], reader_data[111:96], reader_data[24], state_low_payload};
                            // Remask consumes the old-token probability. Ordinary cross-block
                            // changes consume the actual proposal confidence, never the
                            // change record's sentinel BF16 one.
                            changes[state_change_index[6:0]] <= {reader_data[73:72],
                                all_changes && confidence_mode != 2 && !reader_data[73] ?
                                    reader_data[111:96] : reader_data[95:80]};
                            change_seen[state_change_index[6:0]] <= 1'b1;
                            changed_count <= changed_count + {6'd0, reader_data[72]};
                            if (change_seen[state_change_index[6:0]] || reader_data[79:74] != 0 || reader_data[127:121] != 0 || reader_data[119:113] != 0 ||
                                (reader_data[120] && state_low_payload[18:17] != 1) ||
                                (reader_data[112] && reader_data[111:96] > 16'h3f80) ||
                                (all_changes && confidence_mode != 2 && reader_data[72] && !reader_data[73] && !reader_data[112]) ||
                                (reader_data[73] && !reader_data[72]) || (reader_data[72] && reader_data[95:80] > 16'h3f80)) begin
                                terminal_error <= 1'b1; terminal_error_id <= 8'h06; state <= ABORT_DRAIN; aborting <= 1'b0;
                            end
                        end
                    end else if (reader_valid) begin
                        if (change_index < keys) begin
                            changes[change_index] <= change_record[17:0];
                            change_seen[change_index] <= 1'b1;
                            changed_count <= changed_count + {6'd0, change_record[16]};
                            if (change_record[31:18] != 0 || (change_record[17] && !change_record[16])) begin
                                terminal_error <= 1'b1; terminal_error_id <= 8'h06; state <= ABORT_DRAIN; aborting <= 1'b0;
                            end
                        end
                        change_index <= change_index + 7'd1; change_lane <= change_lane + 2'd1;
                    end
                    if (reader_done) begin change_index <= '0; state <= update_row_table ? TABLE_READ_START : PENDING_START; end
                end
                TABLE_READ_START: if (reader_start_ready) begin table_reader_finished <= 1'b0; state <= TABLE_WRITE_START; end
                TABLE_WRITE_START: if (writer_start_ready) begin table_writer_finished <= 1'b0; state <= TABLE_LOAD; end
                TABLE_LOAD: if (reader_valid) begin pending_entry <= reader_data; state <= TABLE_LOW; end
                TABLE_LOW: begin pending_entry <= table_beat; change_index <= change_index + 7'd1; state <= TABLE_HIGH; end
                TABLE_HIGH: begin pending_entry <= table_beat; change_index <= change_index + 7'd1; state <= TABLE_STREAM; end
                TABLE_STREAM: if (writer_ready) state <= change_index >= keys ? TABLE_DRAIN : TABLE_LOAD;
                TABLE_DRAIN: if ((table_reader_finished || reader_done) && (table_writer_finished || writer_done)) state <= PENDING_START;
                PENDING_START: if (reader_start_ready) state <= PENDING_WAIT;
                PENDING_WAIT: begin
                    if (reader_valid && reader_ready) begin
                        pending_entry <= reader_data;
                        // The boundary selection used the previous pending state.
                        // Merge only now, before observe consumes refreshed tokens.
                        if (advance_block && row_index < block_end) begin
                            pending_entry[15:0] <= reader_data[15:0] > reader_data[63:48] ? reader_data[15:0] : reader_data[63:48];
                            pending_entry[31:16] <= reader_data[31:16] > reader_data[79:64] ? reader_data[31:16] : reader_data[79:64];
                            pending_entry[79:48] <= '0;
                        end
                        if (update_row_table && row_index < {5'd0, keys} && change_seen[row_index[6:0]])
                            pending_entry[81] <= change_position_begin+row_index >= block_end-12'd32 &&
                                change_position_begin+row_index < block_end && state_rows[row_index[6:0]][18:17] != 2;
                    end
                    if (reader_done) state <= REDUCE_START;
                end
                REDUCE_START: if (kernel_start_ready) begin
                    key_index <= '0;
                    state <= changed_count == 0 ? REDUCE_WAIT : RELATION_START;
                end
                RELATION_START: if (reader_start_ready) state <= KEY_READ;
                KEY_READ: if (reader_valid) begin
                    key_change <= key_index < keys && change_seen[key_index] ? changes[key_index] : 18'd0;
                    state <= KEY_SEND;
                end
                KEY_SEND: if (key_accepted) begin
                    key_index <= key_index + 7'd1;
                    state <= {25'd0, key_index}+32'd1 == (relation_bytes >> 1) ? RELATION_DRAIN : KEY_READ;
                end
                RELATION_DRAIN: if (reader_done) state <= REDUCE_WAIT;
                REDUCE_WAIT: if (kernel_result_valid) begin
                    if (consumed) pending_entry[79:0] <= '0;
                    pending_entry[80] <= 1'b0;
                    pending_entry[111:96] <= invalidation;
                    if (future_row) begin
                        pending_entry[63:48] <= pending_result;
                        pending_entry[79:64] <= consumed || remask > pending_entry[79:64] ? remask : pending_entry[79:64];
                    end else begin
                        pending_entry[15:0] <= pending_result;
                        if (all_changes) begin
                            pending_entry[31:16] <= consumed || remask > pending_entry[31:16] ? remask : pending_entry[31:16];
                            pending_entry[47:32] <= consumed || remask > pending_entry[47:32] ? remask : pending_entry[47:32];
                        end
                    end
                    state <= WRITE_START;
                end
                WRITE_START: if (writer_start_ready) state <= WRITE_STREAM;
                WRITE_STREAM: if (writer_ready) state <= WRITE_WAIT;
                WRITE_WAIT: if (writer_done) state <= NEXT_TOKEN;
                NEXT_TOKEN: begin
                    row_index <= row_index + 12'd1;
                    pending_address <= pending_address + 64'd16;
                    relation_address <= relation_address + {55'd0, row_stride};
                    state <= row_index + 12'd1 == rows ? COMPLETE : PENDING_START;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: if ((reader_start_ready || reader_abort_ack) && writer_abort_ack &&
                    (kernel_start_ready || kernel_abort_ack || kernel_abort_seen)) begin
                    abort_ack <= aborting; state <= aborting ? ABORT_LOW : COMPLETE;
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
