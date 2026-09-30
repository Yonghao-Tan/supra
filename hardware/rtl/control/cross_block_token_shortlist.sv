`default_nettype none

// Stable top-K in the existing refresh scratch SRAM, before metadata owns it.
// 27 bits x 800 entries, packed two per 59-bit scratch word (400 words).
// One ordered read may be outstanding; used_token_count makes old contents
// invalid without clearing SRAM. Completion/abort drains its response first.
module cross_block_token_shortlist (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [11:0] start_sequence_length,
    input logic [9:0] start_shortlist_token_count,
    input logic [3:0] start_flags,
    input logic [9:0] start_target_token_count,
    input logic [15:0] start_relative_score_floor,
    input logic [11:0] start_protected_begin, start_protected_end,
    output logic [11:0] selected_target_token_count,
    output logic multiply_valid,
    input logic multiply_ready,
    output logic [15:0] multiply_lhs, multiply_rhs,
    input logic multiply_result_valid,
    output logic multiply_result_ready,
    input logic [15:0] multiply_result,
    input logic multiply_error,
    output logic scratch_req_valid,
    input logic scratch_req_ready,
    output logic scratch_write,
    output logic [8:0] scratch_address,
    output logic [58:0] scratch_write_data, scratch_write_enable,
    input logic scratch_rsp_valid,
    input logic [58:0] scratch_rsp_data,
    input logic [63:0] start_pending_base, start_pending_limit,
    input logic [63:0] start_table_base, start_table_limit,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id,
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
    input logic write_done, write_error
);
    typedef enum logic [4:0] {
        IDLE, RANGE, CHECK, TABLE_START, TABLE_WAIT, PENDING_START, PENDING_WAIT,
        RANK_READ, RANK_SELECT, RANK_WRITE, NEXT_TOKEN,
        WRITE_START, WRITE_DATA, WRITE_WAIT, TOP_READ, TOP_SELECT,
        COMPLETE, ABORT_DRAIN, ABORT_LOW, TARGET_CHECK, FLOOR_START, FLOOR_WAIT, FINISH_TOP
    } state_t;
    state_t state;
    logic scratch_read_pending;
    logic [26:0] scratch_entry;
    logic [26:0] incoming, entry;
    logic [9:0] rank_index, used_token_count, shortlist_token_count;
    logic [11:0] sequence_length, position;
    logic [3:0] flags;
    logic tie_tail;
    logic [11:0] protected_begin, protected_end, current_count, mandatory_count;
    logic [11:0] target_token_count, effective_target, retained_top_count;
    logic [15:0] relative_score_floor, optional_max, cutoff;
    logic multiply_pending, row_mandatory;
    logic [15:0] incoming_score;
    logic ranking, mark_only, shifting, aborting, terminal_error;
    logic [63:0] pending_base, pending_limit, table_base, table_limit, table_record, updated_record;
    logic [64:0] pending_end, table_end;
    logic reader_start_ready, reader_valid, reader_done, reader_error, reader_abort_done;
    logic [127:0] reader_data;
    logic writer_start_ready, writer_ready, writer_done, writer_error, writer_abort_done;
    logic [63:0] table_address;
    assign table_address = table_base + {49'd0, position, 3'd0};
    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign row_mandatory = !table_record[8] && (table_record[10] ||
        (flags[2] && position >= protected_begin && position < protected_end));
    assign incoming_score = flags[0] && reader_data[63:48] > reader_data[15:0] ?
        reader_data[63:48] : reader_data[15:0];
    assign effective_target = current_count + mandatory_count > target_token_count ?
        current_count + mandatory_count : target_token_count;
    assign multiply_valid = state == FLOOR_START && !abort_request;
    assign multiply_lhs = optional_max;
    assign multiply_rhs = relative_score_floor;
    assign multiply_result_ready = state == FLOOR_WAIT || state == ABORT_DRAIN;
    always_comb begin
        updated_record = table_record;
        updated_record[10] = row_mandatory;
        updated_record[9] = !table_record[8] && ((mark_only && !tie_tail &&
            {2'd0, rank_index} < retained_top_count) || row_mandatory || (!flags[1] && table_record[11]));
        if (flags[1]) updated_record[11] = 1'b0;
        updated_record[22:12] = flags[3] && !flags[1] ?
            (mark_only ? (tie_tail ? 11'd800 : 11'd0) + {1'b0, rank_index} : 11'd2047) : position[10:0];
    end
    assign scratch_req_valid = !abort_request && (state == RANK_WRITE || state == TOP_READ ||
        (state == RANK_READ && rank_index != used_token_count));
    assign scratch_write = state == RANK_WRITE;
    assign scratch_address = rank_index[9:1];
    assign scratch_write_data = rank_index[0] ? {5'd0, incoming, 27'd0} : {32'd0, incoming};
    assign scratch_write_enable = rank_index[0] ? {5'd0, {27{1'b1}}, 27'd0} : {32'd0, {27{1'b1}}};
    assign scratch_entry = rank_index[0] ? scratch_rsp_data[53:27] : scratch_rsp_data[26:0];
    operator_dma_reader #(.REQUEST_TAG(8'hbc)) reader (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid((state == TABLE_START || state == PENDING_START) && !abort_request),
        .start_ready(reader_start_ready),
        .start_address(state == TABLE_START ? table_address : pending_base + {48'd0, position, 4'd0}),
        .start_bytes(state == TABLE_START ? 32'd8 : 32'd16),
        .start_total_bytes(state == TABLE_START ? 32'd8 : 32'd16),
        .request_valid(read_request_valid), .request_ready(read_request_ready),
        .request_address(read_request_address), .request_bytes(read_request_bytes), .request_tag(read_request_tag),
        .read_valid, .read_ready, .read_data, .read_byte_enable, .read_last, .read_tag,
        .response_error(read_error), .upstream_abort_ack(read_abort_ack),
        .data_valid(reader_valid), .data_ready(state == TABLE_WAIT || state == PENDING_WAIT || state == ABORT_DRAIN),
        .data(reader_data), .data_byte_enable(), .data_last(), .data_byte_offset(),
        .done_pulse(reader_done), .error(reader_error), .abort_ack(reader_abort_done)
    );
    operator_dma_writer #(.REQUEST_TAG(8'hbd)) writer (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid(state == WRITE_START && !abort_request), .start_ready(writer_start_ready),
        .start_address(table_address), .start_bytes(32'd8),
        .request_valid(write_request_valid), .request_ready(write_request_ready),
        .request_address(write_request_address), .request_bytes(write_request_bytes), .request_tag(write_request_tag),
        .data_valid(state == WRITE_DATA && !abort_request), .data_ready(writer_ready),
        .data({64'd0, updated_record}), .data_byte_enable(16'h00ff), .data_last(1'b1),
        .write_valid, .write_ready, .write_data, .write_byte_enable, .write_last,
        .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_done)
    );
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; scratch_read_pending <= 1'b0; abort_ack <= 1'b0; terminal_error <= 1'b0; error_id <= '0; aborting <= 1'b0;
            rank_index <= '0; used_token_count <= '0; shortlist_token_count <= '0; sequence_length <= '0; position <= '0;
            ranking <= 1'b0; mark_only <= 1'b0; shifting <= 1'b0; incoming <= '0; entry <= '0;
            pending_base <= '0; pending_limit <= '0; table_base <= '0; table_limit <= '0;
            pending_end <= '0; table_end <= '0; table_record <= '0;
            flags <= '0; tie_tail <= 1'b0; protected_begin <= '0; protected_end <= '0;
            current_count <= '0; mandatory_count <= '0; target_token_count <= '0;
            retained_top_count <= '0; selected_target_token_count <= '0;
            relative_score_floor <= '0; optional_max <= '0; cutoff <= '0; multiply_pending <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (multiply_valid && multiply_ready) multiply_pending <= 1'b1;
            if (multiply_result_valid && multiply_result_ready) multiply_pending <= 1'b0;
            if (scratch_rsp_valid) scratch_read_pending <= 1'b0;
            if (scratch_req_valid && scratch_req_ready && !scratch_write) scratch_read_pending <= 1'b1;
            if (state == TABLE_WAIT && reader_valid) table_record <= reader_data[63:0];
            if (state == PENDING_WAIT && reader_valid) incoming <= {incoming_score, position[10:0]};
            if (abort_request && state != IDLE && state != ABORT_DRAIN && state != ABORT_LOW) begin
                aborting <= 1'b1; state <= ABORT_DRAIN;
            end else if ((reader_error || writer_error) && state != IDLE && state != COMPLETE && state != ABORT_DRAIN && state != ABORT_LOW) begin
                terminal_error <= 1'b1; error_id <= reader_error ? 8'h03 : 8'h04;
                aborting <= 1'b0; state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    sequence_length <= start_sequence_length; shortlist_token_count <= start_shortlist_token_count;
                    pending_base <= start_pending_base; pending_limit <= start_pending_limit;
                    table_base <= start_table_base; table_limit <= start_table_limit;
                    rank_index <= '0; used_token_count <= '0; position <= '0;
                    ranking <= 1'b1; mark_only <= 1'b0; shifting <= 1'b0;
                    flags <= start_flags; tie_tail <= 1'b0; protected_begin <= start_protected_begin; protected_end <= start_protected_end;
                    target_token_count <= {2'd0, start_target_token_count};
                    selected_target_token_count <= {2'd0, start_target_token_count};
                    relative_score_floor <= start_relative_score_floor; optional_max <= '0; cutoff <= '0;
                    current_count <= '0; mandatory_count <= '0; retained_top_count <= '0;
                    if (start_flags[2]) shortlist_token_count <= 10'd800;
                    terminal_error <= 1'b0; error_id <= '0; aborting <= 1'b0; state <= RANGE;
                end
                RANGE: begin
                    pending_end <= {1'b0, pending_base} + {49'd0, sequence_length, 4'd0};
                    table_end <= {1'b0, table_base} + {50'd0, sequence_length, 3'd0};
                    state <= CHECK;
                end
                CHECK: begin
                    if (sequence_length == 0 || sequence_length > 2048 || shortlist_token_count == 0 || shortlist_token_count > 800 ||
                        (flags[2] && (target_token_count == 0 || target_token_count > 432 || target_token_count > sequence_length ||
                            protected_begin > protected_end || protected_end > sequence_length)) ||
                        (!flags[2] && (flags[3] || flags[1:0] != 0 || relative_score_floor != 0 || protected_begin != 0 || protected_end != 0)) ||
                        (flags[0] && !flags[1]) ||
                        (relative_score_floor != 0 && (!flags[1] || relative_score_floor > 16'h3e80)) ||
                        pending_base[3:0] != 0 || table_base[2:0] != 0 || pending_end[64] || table_end[64] ||
                        pending_end[63:0] > pending_limit || table_end[63:0] > table_limit ||
                        (pending_base < table_end[63:0] && table_base < pending_end[63:0])) begin
                        terminal_error <= 1'b1; error_id <= 8'h01; state <= COMPLETE;
                    end else state <= TABLE_START;
                end
                TABLE_START: if (reader_start_ready) state <= TABLE_WAIT;
                TABLE_WAIT: if (reader_done && !reader_error) begin
                    if (table_record[63:55] != 0) begin terminal_error <= 1'b1; error_id <= 8'h02; state <= COMPLETE; end
                    else begin
                        if (ranking && !tie_tail) begin
                            current_count <= current_count + {11'd0, table_record[8]};
                            mandatory_count <= mandatory_count + {11'd0, row_mandatory};
                        end
                        state <= ranking ? (table_record[8] || (flags[1] && row_mandatory) ||
                            (tie_tail && (!table_record[11] || row_mandatory || table_record[22:12] != 11'd2047)) ?
                            NEXT_TOKEN : PENDING_START) : WRITE_START;
                    end
                end
                PENDING_START: if (reader_start_ready) state <= PENDING_WAIT;
                PENDING_WAIT: if (reader_done && !reader_error) begin
                    rank_index <= '0; shifting <= 1'b0;
                    if (incoming[26:11] > 16'h3f80) begin terminal_error <= 1'b1; error_id <= 8'h02; state <= COMPLETE; end
                    else begin
                        if (incoming[26:11] > optional_max) optional_max <= incoming[26:11];
                        state <= RANK_READ;
                    end
                end
                RANK_READ: begin
                    if (rank_index == used_token_count) state <= RANK_WRITE;
                    else if (scratch_req_valid && scratch_req_ready) state <= RANK_SELECT;
                end
                RANK_SELECT: if (scratch_rsp_valid) begin
                    entry <= scratch_entry;
                    if (shifting || incoming[26:11] > scratch_entry[26:11]) state <= RANK_WRITE;
                    else if (rank_index + 10'd1 == shortlist_token_count) state <= NEXT_TOKEN;
                    else begin rank_index <= rank_index + 10'd1; state <= RANK_READ; end
                end
                RANK_WRITE: if (scratch_req_valid && scratch_req_ready) begin
                    if (rank_index == used_token_count) begin used_token_count <= used_token_count + 10'd1; state <= NEXT_TOKEN; end
                    else if (rank_index + 10'd1 == shortlist_token_count) state <= NEXT_TOKEN;
                    else begin incoming <= entry; rank_index <= rank_index + 10'd1; shifting <= 1'b1; state <= RANK_READ; end
                end
                NEXT_TOKEN: begin
                    if (mark_only) begin
                        if (rank_index + 10'd1 == used_token_count ||
                            ((!flags[3] || flags[1]) && {2'd0, rank_index} + 12'd1 == retained_top_count)) state <= FINISH_TOP;
                        else begin rank_index <= rank_index + 10'd1; state <= TOP_READ; end
                    end else if (position + 12'd1 < sequence_length) begin position <= position + 12'd1; state <= TABLE_START; end
                    else if (ranking) begin ranking <= 1'b0; position <= '0; state <= TARGET_CHECK; end
                    else if (used_token_count == 0) state <= FINISH_TOP;
                    else begin mark_only <= 1'b1; rank_index <= '0; state <= TOP_READ; end
                end
                TARGET_CHECK: begin
                    if (tie_tail) begin
                        mark_only <= 1'b1; rank_index <= '0;
                        state <= used_token_count == 0 ? COMPLETE : TOP_READ;
                    end else if (flags[2] && (current_count != 32 || effective_target > 432 ||
                            (relative_score_floor != 0 && current_count + mandatory_count > target_token_count))) begin
                        terminal_error <= 1'b1; error_id <= 8'h05; state <= COMPLETE;
                    end else begin
                        if (flags[2]) begin
                            retained_top_count <= flags[1] ? effective_target - current_count - mandatory_count :
                                (effective_target - current_count) << 1;
                            selected_target_token_count <= flags[1] ? current_count + mandatory_count : effective_target;
                        end else retained_top_count <= {2'd0, used_token_count};
                        state <= relative_score_floor != 0 ? FLOOR_START : TABLE_START;
                    end
                end
                FLOOR_START: if (multiply_valid && multiply_ready) state <= FLOOR_WAIT;
                FLOOR_WAIT: if (multiply_result_valid) begin
                    if (multiply_error || multiply_result > 16'h3f80) begin
                        terminal_error <= 1'b1; error_id <= 8'h06; state <= COMPLETE;
                    end else begin cutoff <= multiply_result; state <= TABLE_START; end
                end
                TOP_READ: if (scratch_req_valid && scratch_req_ready) state <= TOP_SELECT;
                TOP_SELECT: if (scratch_rsp_valid) begin
                    if (((!flags[3] || flags[1]) && {2'd0, rank_index} >= retained_top_count) ||
                            (flags[1] && optional_max != 0 && scratch_entry[26:11] < cutoff)) state <= FINISH_TOP;
                    else begin position <= {1'b0, scratch_entry[10:0]}; state <= TABLE_START; end
                end
                WRITE_START: if (writer_start_ready) state <= WRITE_DATA;
                WRITE_DATA: if (writer_ready) state <= WRITE_WAIT;
                WRITE_WAIT: if (writer_done && !writer_error) begin
                    if (mark_only && flags[1]) selected_target_token_count <= selected_target_token_count + 12'd1;
                    state <= NEXT_TOKEN;
                end
                FINISH_TOP: begin
                    if (flags[3] && !flags[1] && !tie_tail) begin
                        // All first800 ranks have been written to DDR. Reuse
                        // that same scratch for required-context candidates
                        // below the top800, without increasing sort capacity.
                        tie_tail <= 1'b1; ranking <= 1'b1; mark_only <= 1'b0;
                        used_token_count <= '0; rank_index <= '0; position <= '0;
                        state <= TABLE_START;
                    end else state <= COMPLETE;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: if ((!multiply_pending || multiply_result_valid) && (!scratch_read_pending || scratch_rsp_valid) && (reader_abort_done || reader_start_ready) && (writer_abort_done || writer_start_ready)) begin
                    if (aborting) begin abort_ack <= 1'b1; state <= ABORT_LOW; end
                    else state <= COMPLETE;
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) if (!rst) begin
        if (scratch_req_valid && scratch_req_ready) begin
            assert (scratch_address < 9'd400) else $error("shortlist scratch address overflow");
            assert (!scratch_read_pending) else $error("shortlist reused scratch before read drained");
        end
        if (scratch_rsp_valid)
            assert (scratch_read_pending) else $error("shortlist unexpected SRAM response");
        if (done_valid || abort_ack)
            assert (!scratch_read_pending) else $error("shortlist released live SRAM read");
    end
`endif
endmodule

`default_nettype wire
