`default_nettype none

// Builds resident rounds from selected positions and their source/precision
// records. The existing token_issue_packer remains the only physical row encoder.
module refresh_token_metadata_writer (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
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
    input logic [8:0] start_selected_count,
    input logic [11:0] start_sequence_length,
    input logic [63:0] start_table_base, start_table_limit, start_output_base, start_output_limit,
    input logic [31:0] start_metadata_version, start_capture_index,
    input logic start_embedding_source,
    input logic start_qkvo_group,
    input logic start_deep_precision_enabled,
    input logic [6:0] start_deep_a8_limit,
    input logic [63:0] start_joint_result_base, start_joint_result_limit,
    input logic [127:0] start_joint_result,
    input logic [63:0] start_attempt_base,
    input logic [1151:0] start_attempt_payload,
    input logic [63:0] start_budget_base, start_budget_limit,
    input logic [127:0] start_budget_record,
    input logic [63:0] start_pending_base, start_pending_limit,
    input logic [6:0] start_pending_rows,
    input logic [95:0] start_pending_selected_mask,
    input logic position_valid,
    output logic position_ready,
    input logic [10:0] position,
    input logic position_last,
    input logic position_precision_valid, position_a8,
    input logic start_write_output, start_publish_result,
    output logic position_table_ready,
    output logic scratch_req_valid,
    input logic scratch_req_ready,
    output logic scratch_write,
    output logic [8:0] scratch_address,
    output logic [58:0] scratch_write_data,
    output logic [58:0] scratch_write_enable,
    input logic scratch_rsp_valid,
    input logic [58:0] scratch_rsp_data,
    output logic scratch_aux_write_valid,
    input logic scratch_aux_write_ready,
    output logic [8:0] scratch_aux_write_address,
    output logic [19:0] scratch_aux_write_data,
    output logic [19:0] scratch_aux_write_enable,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic [31:0] output_bytes,
    output logic [3:0] output_rounds,
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
    typedef enum logic [5:0] {
        IDLE, LOAD_POSITIONS, READ_RECORD, RECORD_FETCH, RECORD_ADDRESS,
        RECORD_START, RECORD_WAIT, COUNT_ROUNDS, CHOOSE_ROUND, ROUND_START,
        SCAN_RECORD,
        FINISH_ROUND, COMPLETE, ABORT_DRAIN, ABORT_LOW, JOINT_START, JOINT_STREAM, JOINT_WAIT,
        BUDGET_START, BUDGET_STREAM, BUDGET_WAIT,
        PENDING_READ_START, PENDING_WRITE_START, PENDING_STREAM, PENDING_WAIT,
        DEEP_CHECK, DEEP_READ, DEEP_PICK,
        ATTEMPT_START, ATTEMPT_STREAM, ATTEMPT_WAIT, RESULT_START, RESULT_STREAM, RESULT_WAIT
    } state_t;
    state_t state;
    logic publish_result;
    logic [127:0] result_record;
    logic [38:0] record;
    logic [38:0] scan_record;
    logic deep_precision_enabled, deep_best_valid;
    logic [6:0] deep_a8_limit;
    logic [8:0] deep_protected_count, deep_best_index;
    logic [7:0] deep_best_score;
    logic record_a8;
    logic [8:0] total_token_count, received_rows, record_index, remaining_a4, remaining_a8;
    logic [6:0] round_a4, round_a8, needed_a4, needed_a8;
    logic qkvo_group, metadata_header_pending, group_end;
    logic [2:0] group_batches;
    logic [5:0] group_compute_groups, round_compute_groups, group_compute_total;
    logic [127:0] metadata_beat_data;
    logic [5:0] round_rows, feed_index;
    logic [8:0] first_ordinal;
    logic [3:0] remaining_rounds;
    logic [11:0] sequence_length;
    logic [10:0] previous_position;
    logic [8:0] run_start_index, run_length;
    logic write_output;
    // 32 bits x 14 words = 56 B, one assignment bit per selected token.
    // Needed across rounds; reset only on a new selection. Payload stays in SRAM.
    logic [31:0] assigned_words [0:13];
    logic [31:0] assigned_word;
    logic [63:0] table_base, output_limit, record_address, current_output;
    logic [31:0] metadata_version, capture_index, token_batch_bytes;
    logic [64:0] output_end;
    logic [64:0] table_end;
    logic [9:0] remaining_slots;
    logic [6:0] candidate_a4;
    logic [9:0] future_rows_capacity, future_slots_capacity;
    logic reader_start_ready, reader_done, reader_error, reader_abort_done;
    logic reader_valid, reader_last;
    logic [127:0] reader_data;
    logic [15:0] reader_byte_enable;
    logic [31:0] reader_offset;
    wire packer_done = packer_done_valid;
    logic writer_start_ready, writer_done, writer_error, writer_abort_done;
    logic packer_started, writer_started, packer_finished, writer_finished;
    logic aborting, terminal_error, packer_drained, writer_drained;
    logic [7:0] terminal_error_id;
    logic selected_for_round;
    logic scratch_stream_active, scratch_stream_finished;
    logic [8:0] scratch_issue_index;
    logic [2:0] scratch_read_outstanding;
    logic [2:0] scratch_fifo_occupancy;
    logic scratch_fifo_input_ready, scratch_fifo_output_valid;
    logic scratch_fifo_output_ready;
    logic [38:0] scratch_fifo_output_data;
    logic [3:0] scratch_reserved_count;
    logic scratch_stream_request;
    logic scratch_stream_pop;
    logic scratch_stream_empty_after_cycle;
    logic scan_selected_for_round, scan_completes_round;
    logic [63:0] joint_result_base;
    logic [127:0] joint_result;
    logic [63:0] attempt_base;
    logic [3:0] attempt_beat;
    logic [64:0] joint_result_end;
    logic writer_data_ready;
    logic embedding_source;
    logic [63:0] budget_base;
    logic [127:0] budget_record;
    logic [64:0] budget_end;
    logic [63:0] pending_base;
    logic [10:0] pending_bytes;
    logic [95:0] pending_selected_mask;
    logic pending_read_finished;
    logic [64:0] pending_end;
    logic [127:0] pending_data;
    logic pending_read_start, pending_write_start, pending_stream;
    logic position_write_accepted, run_write_accepted, position_extends_run;

    assign result_record = {metadata_version, capture_index, output_bytes, 12'd0, output_rounds, 7'd0, total_token_count};

    assign start_ready = state == IDLE && !abort_request;
    assign position_extends_run = received_rows != 0 && position == previous_position + 11'd1;
    assign position_ready = state == LOAD_POSITIONS && !abort_request &&
        (position_write_accepted || scratch_req_ready) &&
        (!position_extends_run || run_write_accepted || scratch_aux_write_ready);
    assign position_table_ready = received_rows == total_token_count && total_token_count != 0;
    assign done_valid = state == COMPLETE && (!publish_result || terminal_error);
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign scratch_stream_active = state == SCAN_RECORD || state == DEEP_READ;
    assign scratch_reserved_count = {1'b0, scratch_fifo_occupancy} +
        {1'b0, scratch_read_outstanding};
    assign scratch_stream_request = scratch_stream_active &&
        !scratch_stream_finished && scratch_issue_index < total_token_count &&
        scratch_reserved_count < 4'd3;
    assign scratch_req_valid = !abort_request &&
        ((state == LOAD_POSITIONS && position_valid && !position_write_accepted) ||
         state == READ_RECORD || scratch_stream_request ||
         state == DEEP_PICK || (state == RECORD_WAIT && reader_valid));
    assign scratch_write = state == LOAD_POSITIONS || state == DEEP_PICK ||
        state == RECORD_WAIT;
    assign scratch_address = state == LOAD_POSITIONS ? received_rows :
        state == DEEP_PICK ? deep_best_index :
        scratch_stream_active ? scratch_issue_index : record_index;
    assign scratch_write_data = state == LOAD_POSITIONS ?
        {{position, 9'd1},
         {9'd0, position_precision_valid, position_a8, 17'd0, position}} :
        state == RECORD_WAIT ?
        {20'd0, reader_data[8] | reader_data[10], reader_data[7:0],
         reader_data[54], record_a8, reader_data[51:35], record[10:0]} : 59'd0;
    assign scratch_write_enable = state == DEEP_PICK ?
        (59'd1 << 28) : state == RECORD_WAIT ?
        {{20{1'b0}}, {39{1'b1}}} : {59{1'b1}};
    assign scratch_aux_write_valid = state == LOAD_POSITIONS &&
        position_valid && position_extends_run && !run_write_accepted && !abort_request;
    assign scratch_aux_write_address = run_start_index;
    assign scratch_aux_write_data = {11'd0, run_length + 9'd1};
    assign scratch_aux_write_enable = {11'd0, 9'h1ff};
    assign record_a8 = record[29] ? record[28] : reader_data[34];
    assign remaining_slots = {1'b0, remaining_a4} + {remaining_a8, 1'b0};
    always_comb begin
        candidate_a4 = remaining_a4 > 9'd48 ? 7'd48 : remaining_a4[6:0];
        if (candidate_a4 > 7'd48 - round_a8) candidate_a4 = 7'd48 - round_a8;
        if (candidate_a4 > 7'd64 - (round_a8 << 1)) candidate_a4 = 7'd64 - (round_a8 << 1);
    end
    assign future_rows_capacity = (10'(remaining_rounds) - 10'd1) * 10'd48;
    assign future_slots_capacity = (10'(remaining_rounds) - 10'd1) << 6;
    assign scan_record = scratch_fifo_output_data[38:0];
    always_comb begin
        case (record_index[8:5])
            4'd0: assigned_word = assigned_words[0];
            4'd1: assigned_word = assigned_words[1];
            4'd2: assigned_word = assigned_words[2];
            4'd3: assigned_word = assigned_words[3];
            4'd4: assigned_word = assigned_words[4];
            4'd5: assigned_word = assigned_words[5];
            4'd6: assigned_word = assigned_words[6];
            4'd7: assigned_word = assigned_words[7];
            4'd8: assigned_word = assigned_words[8];
            4'd9: assigned_word = assigned_words[9];
            4'd10: assigned_word = assigned_words[10];
            4'd11: assigned_word = assigned_words[11];
            4'd12: assigned_word = assigned_words[12];
            4'd13: assigned_word = assigned_words[13];
            default: assigned_word = '0;
        endcase
    end
    assign scan_selected_for_round = !assigned_word[record_index[4:0]] &&
        (scan_record[28] ? needed_a8 != 0 : needed_a4 != 0);
    assign scan_completes_round = scan_selected_for_round &&
        feed_index + 6'd1 == round_rows;
    assign selected_for_round = scan_selected_for_round;
    assign scratch_fifo_output_ready = state == DEEP_READ ||
        (state == SCAN_RECORD &&
         (scratch_stream_finished || !scan_selected_for_round || packer_row_ready));
    assign scratch_stream_pop = scratch_fifo_output_valid &&
        scratch_fifo_output_ready;
    assign scratch_stream_empty_after_cycle =
        ({1'b0, scratch_fifo_occupancy} +
         {3'd0, scratch_rsp_valid}) == {3'd0, scratch_stream_pop} &&
        scratch_read_outstanding == {2'd0, scratch_rsp_valid};
    assign token_batch_bytes = 32'd32 + {22'd0, round_rows, 4'd0} +
        (({26'd0, round_rows} + 32'd15) & 32'hfffffff0);
    assign output_end = {1'b0, current_output} + {33'd0, token_batch_bytes};
    assign table_end = {1'b0, start_table_base} + {50'd0, start_sequence_length, 3'd0};
    assign joint_result_end = {1'b0, start_joint_result_base} + 65'd16;
    assign budget_end = {1'b0, start_budget_base} + 65'd16;
    assign pending_end = {1'b0, start_pending_base} + {54'd0, start_pending_rows, 4'd0};
    assign pending_read_start = state == PENDING_READ_START;
    assign pending_write_start = state == PENDING_WRITE_START;
    assign pending_stream = state == PENDING_STREAM;
    always_comb begin
        pending_data = reader_data;
        pending_data[80] = pending_selected_mask[reader_offset[10:4]];
    end

    operator_dma_reader #(.REQUEST_TAG(8'hb4)) reader (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid((state == RECORD_START || pending_read_start) && !abort_request), .start_ready(reader_start_ready),
        .start_address(pending_read_start ? pending_base : record_address),
        .start_bytes(pending_read_start ? {21'd0, pending_bytes} : 32'd8),
        .start_total_bytes(pending_read_start ? {21'd0, pending_bytes} : 32'd8),
        .request_valid(read_request_valid), .request_ready(read_request_ready),
        .request_address(read_request_address), .request_bytes(read_request_bytes), .request_tag(read_request_tag),
        .read_valid, .read_ready, .read_data, .read_byte_enable, .read_last, .read_tag,
        .response_error(read_error), .upstream_abort_ack(read_abort_ack), .data_valid(reader_valid),
        .data_ready((state == RECORD_WAIT && scratch_req_ready) ||
            state == ABORT_DRAIN || (pending_stream && writer_data_ready)), .data(reader_data),
        .data_byte_enable(reader_byte_enable), .data_last(reader_last), .data_byte_offset(reader_offset),
        .done_pulse(reader_done), .error(reader_error), .abort_ack(reader_abort_done)
    );
    assign packer_abort_request = state == ABORT_DRAIN;
    assign round_compute_groups = 6'((round_a4 + 7'd15) >> 4) +
                                  6'((round_a8 + 7'd7) >> 3);
    assign group_compute_total = group_compute_groups + round_compute_groups;
    // A subsequent mixed round needs at most five compute groups; an A4-only
    // round needs at most three. Close before accepting a round that may not fit.
    assign group_end = group_batches == 3'd5 ||
        first_ordinal + {3'd0, round_rows} == total_token_count ||
        group_compute_total + (remaining_a8 > {2'd0, round_a8} ? 6'd5 : 6'd3) > 6'd18;
    always_comb begin
        metadata_beat_data = packer_beat_data;
        if (qkvo_group && metadata_header_pending)
            metadata_beat_data[26] = group_end;
    end
    assign packer_start_valid = state == ROUND_START && !packer_started && !abort_request &&
        output_end <= {1'b0, output_limit};
    assign packer_config = '{row_count: round_rows, sequence_length: sequence_length,
        token_batch_index: {12'd0, output_rounds}, first_token_ordinal: {7'd0, first_ordinal},
        metadata_version: metadata_version, capture_index: capture_index};
    assign packer_row_valid = state == SCAN_RECORD &&
        scratch_fifo_output_valid && !scratch_stream_finished &&
        scan_selected_for_round && !abort_request;
    assign packer_row = '{index: feed_index, source_index: scan_record[27:11],
        token_position: scan_record[10:0], kv_index: scan_record[10:0],
        embedding_source: embedding_source, kv_write_disable: scan_record[29],
        bits: scan_record[28] ? 4'd8 : 4'd4,
        query_group: 8'd0, cache_group: 8'd0};
    assign packer_done_ready = 1'b1;
    assign packer_beat_ready = writer_data_ready && !pending_stream && state != JOINT_STREAM && state != BUDGET_STREAM && state != ATTEMPT_STREAM && state != RESULT_STREAM;
    operator_dma_writer #(.REQUEST_TAG(8'hb5)) writer (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid(!abort_request && (state == RESULT_START || state == JOINT_START || state == BUDGET_START || state == ATTEMPT_START || pending_write_start ||
            (state == ROUND_START && !writer_started && output_end <= {1'b0, output_limit}))),
        .start_ready(writer_start_ready), .start_address(state == RESULT_START ? output_limit : state == ATTEMPT_START ? attempt_base : pending_write_start ? pending_base : state == BUDGET_START ? budget_base : state == JOINT_START ? joint_result_base : current_output),
        .start_bytes(state == ATTEMPT_START ? 32'd144 : pending_write_start ? {21'd0, pending_bytes} : state == RESULT_START || state == JOINT_START || state == BUDGET_START ? 32'd16 : token_batch_bytes),
        .request_valid(write_request_valid), .request_ready(write_request_ready),
        .request_address(write_request_address), .request_bytes(write_request_bytes), .request_tag(write_request_tag),
        .data_valid(pending_stream ? reader_valid : state == RESULT_STREAM || state == ATTEMPT_STREAM || state == JOINT_STREAM || state == BUDGET_STREAM || packer_beat_valid), .data_ready(writer_data_ready),
        .data(state == RESULT_STREAM ? result_record : state == ATTEMPT_STREAM ? start_attempt_payload[attempt_beat*128 +: 128] : pending_stream ? pending_data : state == BUDGET_STREAM ? budget_record : state == JOINT_STREAM ? joint_result : metadata_beat_data),
        .data_byte_enable(16'hffff), .data_last(state == ATTEMPT_STREAM ? attempt_beat == 8 : pending_stream ? reader_last : state == RESULT_STREAM || state == JOINT_STREAM || state == BUDGET_STREAM || packer_beat_last), .write_valid, .write_ready,
        .write_data, .write_byte_enable, .write_last, .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_done)
    );

    ready_valid_fifo #(.DATA_WIDTH(39), .DEPTH(3)) scratch_response_fifo (
        .clk, .rst,
        .input_valid(scratch_rsp_valid && scratch_stream_active),
        .input_ready(scratch_fifo_input_ready),
        .input_data(scratch_rsp_data[38:0]),
        .output_valid(scratch_fifo_output_valid),
        .output_ready(scratch_fifo_output_ready),
        .output_data(scratch_fifo_output_data),
        .occupancy(scratch_fifo_occupancy)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            publish_result <= 1'b0;
            position_write_accepted <= 1'b0; run_write_accepted <= 1'b0;
            total_token_count <= '0; received_rows <= '0; record_index <= '0;
            remaining_a4 <= '0; remaining_a8 <= '0;
            qkvo_group <= 1'b0; metadata_header_pending <= 1'b0;
            group_batches <= '0; group_compute_groups <= '0;
            round_a4 <= '0; round_a8 <= '0; needed_a4 <= '0; needed_a8 <= '0;
            round_rows <= '0; feed_index <= '0; first_ordinal <= '0; remaining_rounds <= '0;
            sequence_length <= '0; previous_position <= '0;
            for (integer word = 0; word < 14; word++)
                assigned_words[word] <= '0;
            run_start_index <= '0; run_length <= '0; write_output <= 1'b0;
            table_base <= '0; output_limit <= '0; record_address <= '0;
            current_output <= '0; metadata_version <= '0; capture_index <= '0; record <= '0;
            output_bytes <= '0; output_rounds <= '0;
            packer_started <= 1'b0; writer_started <= 1'b0;
            packer_finished <= 1'b0; writer_finished <= 1'b0;
            aborting <= 1'b0; terminal_error <= 1'b0; terminal_error_id <= '0;
            packer_drained <= 1'b0; writer_drained <= 1'b0; abort_ack <= 1'b0;
            joint_result_base <= '0; joint_result <= '0;
            attempt_base <= '0; attempt_beat <= '0;
            embedding_source <= 1'b0;
            deep_precision_enabled <= 1'b0; deep_a8_limit <= '0; deep_protected_count <= '0;
            deep_best_valid <= 1'b0; deep_best_index <= '0; deep_best_score <= '0;
            scratch_stream_finished <= 1'b0; scratch_issue_index <= '0;
            scratch_read_outstanding <= '0;
            budget_base <= '0; budget_record <= '0;
            pending_base <= '0; pending_bytes <= '0; pending_selected_mask <= '0; pending_read_finished <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (packer_beat_valid && packer_beat_ready) metadata_header_pending <= 1'b0;
            // Both writes refer to the held position payload. Either SRAM port
            // may accept first; do not reissue it while waiting for the other.
            if (state == LOAD_POSITIONS) begin
                if (scratch_req_valid && scratch_req_ready) position_write_accepted <= 1'b1;
                if (scratch_aux_write_valid && scratch_aux_write_ready) run_write_accepted <= 1'b1;
                if (position_valid && position_ready) begin
                    position_write_accepted <= 1'b0; run_write_accepted <= 1'b0;
                end
            end
            if (scratch_stream_active) begin
                if (scratch_req_valid && scratch_req_ready)
                    scratch_issue_index <= scratch_issue_index + 9'd1;
                case ({scratch_req_valid && scratch_req_ready,
                       scratch_rsp_valid})
                    2'b10: scratch_read_outstanding <=
                        scratch_read_outstanding + 3'd1;
                    2'b01: scratch_read_outstanding <=
                        scratch_read_outstanding - 3'd1;
                    default: scratch_read_outstanding <=
                        scratch_read_outstanding;
                endcase
            end
            if ((abort_request || reader_error || writer_error || packer_error) &&
                    state != IDLE && state != COMPLETE && state != ABORT_DRAIN && state != ABORT_LOW) begin
                aborting <= abort_request;
                if (!abort_request) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= reader_error ? 8'h04 : writer_error ? 8'h05 : 8'h10 | packer_error_id;
                end
                packer_drained <= !packer_started || packer_finished || packer_done;
                writer_drained <= !writer_started || writer_finished || writer_done;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    position_write_accepted <= 1'b0; run_write_accepted <= 1'b0;
                    joint_result_base <= start_joint_result_base;
                    joint_result <= start_joint_result;
                    attempt_base <= start_attempt_base; attempt_beat <= '0;
                    budget_base <= start_budget_base;
                    budget_record <= start_budget_record;
                    pending_base <= start_pending_base;
                    pending_bytes <= {start_pending_rows, 4'd0};
                    pending_selected_mask <= start_pending_selected_mask;
                    embedding_source <= start_embedding_source;
                    qkvo_group <= start_qkvo_group;
                    group_batches <= '0; group_compute_groups <= '0;
                    metadata_header_pending <= 1'b0;
                    deep_precision_enabled <= start_deep_precision_enabled;
                    deep_a8_limit <= start_deep_a8_limit; deep_protected_count <= '0;
                    total_token_count <= start_selected_count;
                    sequence_length <= start_sequence_length;
                    write_output <= start_write_output;
                    table_base <= start_table_base;
                    current_output <= start_output_base;
                    output_limit <= start_output_limit - (start_publish_result ? 64'd16 : 64'd0);
                    publish_result <= start_publish_result;
                    metadata_version <= start_metadata_version;
                    capture_index <= start_capture_index;
                    received_rows <= '0; record_index <= '0; first_ordinal <= '0;
                    run_start_index <= '0; run_length <= '0;
                    remaining_a4 <= '0; remaining_a8 <= '0;
                    for (integer word = 0; word < 14; word++)
                        assigned_words[word] <= '0;
                    remaining_rounds <= 4'd1;
                    output_bytes <= '0; output_rounds <= '0;
                    packer_started <= 1'b0; writer_started <= 1'b0;
                    terminal_error <= 1'b0; terminal_error_id <= '0;
                    if ((start_publish_result && (!start_write_output || start_output_limit[3:0] != 0 ||
                            start_output_limit <= start_output_base || start_output_limit-start_output_base <= 64'd16)) ||
                            (start_deep_precision_enabled && (start_deep_a8_limit > 96 || start_embedding_source)) ||
                            (start_selected_count == 0 && start_budget_base == 0) || start_selected_count > 9'd432 ||
                            (start_pending_base != 0 && (start_pending_base[3:0] != 0 || start_pending_rows == 0 ||
                                start_pending_rows > 96 || pending_end[64] || pending_end[63:0] > start_pending_limit ||
                                (start_pending_base < start_output_limit && start_output_base < pending_end[63:0]) ||
                                (start_pending_base < start_table_limit && start_table_base < pending_end[63:0]) ||
                                (start_budget_base != 0 && start_pending_base < budget_end[63:0] && start_budget_base < pending_end[63:0]) ||
                                (start_joint_result_base != 0 && start_pending_base < joint_result_end[63:0] && start_joint_result_base < pending_end[63:0]))) ||
                            start_sequence_length == 0 || start_sequence_length > 12'd2048 ||
                            (start_write_output && (start_table_base[3:0] != 0 || start_output_base[3:0] != 0 ||
                                start_output_base >= start_output_limit || table_end[64] ||
                                table_end[63:0] > start_table_limit ||
                                (start_table_base < start_output_limit && start_output_base < start_table_limit))) ||
                            (start_budget_base != 0 && (start_budget_base[3:0] != 0 || budget_end[64] ||
                                budget_end[63:0] > start_budget_limit ||
                                (start_budget_base < start_output_limit && start_output_base < budget_end[63:0]) ||
                                (start_budget_base < start_table_limit && start_table_base < budget_end[63:0]) ||
                                (start_joint_result_base != 0 && start_budget_base < joint_result_end[63:0] && start_joint_result_base < budget_end[63:0]))) ||
                            (start_joint_result_base != 0 && (start_joint_result_base[3:0] != 0 ||
                                joint_result_end[64] || joint_result_end[63:0] > start_joint_result_limit ||
                                (start_joint_result_base < start_output_limit && start_output_base < joint_result_end[63:0]) ||
                                (start_joint_result_base < start_table_limit && start_table_base < joint_result_end[63:0])))) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                    end else state <= start_selected_count == 0 ?
                        (start_pending_base != 0 ? PENDING_READ_START : BUDGET_START) : LOAD_POSITIONS;
                end
                LOAD_POSITIONS: if (position_valid && position_ready) begin
                    if ({1'b0, position} >= sequence_length ||
                            (received_rows != 0 && position <= previous_position) ||
                            position_last != (received_rows + 9'd1 == total_token_count)) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h02; state <= COMPLETE;
                    end else begin
                        previous_position <= position;
                        received_rows <= received_rows + 9'd1;
                        if (received_rows == 0 || position != previous_position + 11'd1) begin
                            run_start_index <= received_rows;
                            run_length <= 9'd1;
                        end else run_length <= run_length + 9'd1;
                        if (position_last) state <= write_output ? READ_RECORD : COMPLETE;
                    end
                end
                READ_RECORD: if (scratch_req_ready) state <= RECORD_FETCH;
                RECORD_FETCH: if (scratch_rsp_valid) begin
                    record <= scratch_rsp_data[38:0];
                    state <= RECORD_ADDRESS;
                end
                RECORD_ADDRESS: begin
                    record_address <= table_base + {50'd0, record[10:0], 3'd0};
                    state <= RECORD_START;
                end
                RECORD_START: if (reader_start_ready) state <= RECORD_WAIT;
                RECORD_WAIT: begin
                    if (reader_valid && scratch_req_ready) begin
                        remaining_a8 <= remaining_a8 + {8'd0, record_a8};
                        remaining_a4 <= remaining_a4 + {8'd0, !record_a8};
                        if (record_a8 && (reader_data[8] || reader_data[10]))
                            deep_protected_count <= deep_protected_count + 9'd1;
                        if (reader_data[63:55] != 0 ||
                            reader_data[51:35] >= (embedding_source ? 17'd126464 : 17'd2048)) begin
                            terminal_error <= 1'b1; terminal_error_id <= 8'h06;
                        end
                    end
                    if (reader_done) begin
                        if (terminal_error) state <= COMPLETE;
                        else if (record_index + 9'd1 == total_token_count) state <= deep_precision_enabled ? DEEP_CHECK : COUNT_ROUNDS;
                        else begin record_index <= record_index + 9'd1; state <= READ_RECORD; end
                    end
                end
                DEEP_CHECK: begin
                    record_index <= '0; deep_best_valid <= 1'b0;
                    scratch_issue_index <= '0;
                    scratch_read_outstanding <= '0;
                    scratch_stream_finished <= 1'b0;
                    state <= remaining_a8 > {2'd0, deep_a8_limit} && remaining_a8 > deep_protected_count ? DEEP_READ : COUNT_ROUNDS;
                end
                DEEP_READ: if (scratch_stream_pop) begin
                    // Drop the lowest score; later ordinals lose equal-score ties.
                    if (scan_record[28] && !scan_record[38] &&
                            (!deep_best_valid || scan_record[37:30] <= deep_best_score)) begin
                        deep_best_valid <= 1'b1; deep_best_index <= record_index;
                        deep_best_score <= scan_record[37:30];
                    end
                    if (record_index + 9'd1 == total_token_count) state <= DEEP_PICK;
                    else record_index <= record_index + 9'd1;
                end
                DEEP_PICK: if (scratch_req_ready) begin
                    if (!deep_best_valid) begin terminal_error <= 1'b1; terminal_error_id <= 8'h07; state <= COMPLETE; end
                    else begin remaining_a8 <= remaining_a8 - 9'd1; remaining_a4 <= remaining_a4 + 9'd1; state <= DEEP_CHECK; end
                end
                COUNT_ROUNDS: begin
                    round_a8 <= remaining_a8 > 9'd32 ? 7'd32 : remaining_a8[6:0];
                    if (remaining_a4 + remaining_a8 <= 9'd96) begin
                        remaining_rounds <= (remaining_slots > 10'd128) ? 4'd3 :
                            (remaining_slots > 10'd64 || remaining_a4+remaining_a8 > 9'd48) ? 4'd2 : 4'd1;
                        state <= CHOOSE_ROUND;
                    end else if (10'(remaining_a4) + 10'(remaining_a8) > 10'(remaining_rounds)*10'd48 ||
                                 remaining_slots > (10'(remaining_rounds) << 6)) begin
                        remaining_rounds <= remaining_rounds + 4'd1;
                    end else state <= CHOOSE_ROUND;
                end
                CHOOSE_ROUND: begin
                    if (10'(remaining_a4) + 10'(remaining_a8) - 10'(round_a8) - 10'(candidate_a4) <= future_rows_capacity &&
                            remaining_slots - 10'(round_a8)*10'd2 - 10'(candidate_a4) <= future_slots_capacity) begin
                        round_a4 <= candidate_a4; needed_a4 <= candidate_a4; needed_a8 <= round_a8;
                        round_rows <= 6'(candidate_a4 + round_a8);
                        metadata_header_pending <= 1'b1;
                        record_index <= '0; feed_index <= '0;
                        scratch_issue_index <= '0;
                        scratch_read_outstanding <= '0;
                        scratch_stream_finished <= 1'b0;
                        packer_started <= 1'b0; writer_started <= 1'b0;
                        packer_finished <= 1'b0; writer_finished <= 1'b0;
                        state <= ROUND_START;
                    end else round_a8 <= round_a8 - 7'd1;
                end
                ROUND_START: begin
                    if (output_end > {1'b0, output_limit}) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h03; state <= COMPLETE;
                    end else begin
                        if (packer_start_ready) packer_started <= 1'b1;
                        if (writer_start_ready) writer_started <= 1'b1;
                        if ((packer_started || packer_start_ready) &&
                                (writer_started || writer_start_ready)) begin
                            scratch_issue_index <= '0;
                            scratch_read_outstanding <= '0;
                            scratch_stream_finished <= 1'b0;
                            state <= SCAN_RECORD;
                        end
                    end
                end
                SCAN_RECORD: begin
                    if (scratch_stream_pop && !scratch_stream_finished) begin
                        if (scan_selected_for_round) begin
                            case (record_index[8:5])
                                4'd0: assigned_words[0] <= assigned_words[0] |
                                    (32'd1 << record_index[4:0]);
                                4'd1: assigned_words[1] <= assigned_words[1] |
                                    (32'd1 << record_index[4:0]);
                                4'd2: assigned_words[2] <= assigned_words[2] |
                                    (32'd1 << record_index[4:0]);
                                4'd3: assigned_words[3] <= assigned_words[3] |
                                    (32'd1 << record_index[4:0]);
                                4'd4: assigned_words[4] <= assigned_words[4] |
                                    (32'd1 << record_index[4:0]);
                                4'd5: assigned_words[5] <= assigned_words[5] |
                                    (32'd1 << record_index[4:0]);
                                4'd6: assigned_words[6] <= assigned_words[6] |
                                    (32'd1 << record_index[4:0]);
                                4'd7: assigned_words[7] <= assigned_words[7] |
                                    (32'd1 << record_index[4:0]);
                                4'd8: assigned_words[8] <= assigned_words[8] |
                                    (32'd1 << record_index[4:0]);
                                4'd9: assigned_words[9] <= assigned_words[9] |
                                    (32'd1 << record_index[4:0]);
                                4'd10: assigned_words[10] <= assigned_words[10] |
                                    (32'd1 << record_index[4:0]);
                                4'd11: assigned_words[11] <= assigned_words[11] |
                                    (32'd1 << record_index[4:0]);
                                4'd12: assigned_words[12] <= assigned_words[12] |
                                    (32'd1 << record_index[4:0]);
                                4'd13: assigned_words[13] <= assigned_words[13] |
                                    (32'd1 << record_index[4:0]);
                                default: begin end
                            endcase
                            feed_index <= feed_index + 6'd1;
                            if (scan_record[28]) needed_a8 <= needed_a8 - 7'd1;
                            else needed_a4 <= needed_a4 - 7'd1;
                        end
                        if (scan_completes_round)
                            scratch_stream_finished <= 1'b1;
                        else if (record_index + 9'd1 == total_token_count) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= 8'h07;
                            scratch_stream_finished <= 1'b1;
                        end else
                            record_index <= record_index + 9'd1;
                    end
                    if ((scratch_stream_finished ||
                            (scratch_stream_pop && scan_completes_round)) &&
                            scratch_stream_empty_after_cycle)
                        state <= terminal_error ||
                            (scratch_stream_pop && !scan_completes_round &&
                             record_index + 9'd1 == total_token_count) ?
                            COMPLETE : FINISH_ROUND;
                end
                /*
                 * Responses already issued when a round fills are drained in
                 * SCAN_RECORD. This keeps the fixed SRAM port response stream
                 * empty before another owner or round starts.
                 */
                FINISH_ROUND: begin
                    if (packer_done) packer_finished <= 1'b1;
                    if (writer_done) writer_finished <= 1'b1;
                    if ((packer_finished || packer_done) && (writer_finished || writer_done)) begin
                        remaining_a4 <= remaining_a4 - {2'd0, round_a4};
                        remaining_a8 <= remaining_a8 - {2'd0, round_a8};
                        remaining_rounds <= 4'd1;
                        first_ordinal <= first_ordinal + {3'd0, round_rows};
                        output_bytes <= output_bytes + token_batch_bytes;
                        current_output <= output_end[63:0];
                        output_rounds <= output_rounds + 4'd1;
                        group_batches <= group_end ? 3'd0 : group_batches + 3'd1;
                        group_compute_groups <= group_end ? 6'd0 : group_compute_total;
                        state <= first_ordinal + {3'd0, round_rows} == total_token_count ?
                            (pending_base != 0 ? PENDING_READ_START : joint_result_base != 0 ? JOINT_START : budget_base != 0 ? BUDGET_START : COMPLETE) : COUNT_ROUNDS;
                    end
                end
                JOINT_START: if (writer_start_ready && !abort_request) begin
                    writer_started <= 1'b1; writer_finished <= 1'b0; state <= JOINT_STREAM;
                end
                JOINT_STREAM: if (writer_data_ready) state <= JOINT_WAIT;
                JOINT_WAIT: if (writer_done) begin writer_finished <= 1'b1; state <= budget_base != 0 ? BUDGET_START : attempt_base != 0 ? ATTEMPT_START : COMPLETE; end
                BUDGET_START: if (writer_start_ready && !abort_request) begin
                    writer_started <= 1'b1; writer_finished <= 1'b0; state <= BUDGET_STREAM;
                end
                BUDGET_STREAM: if (writer_data_ready) state <= BUDGET_WAIT;
                BUDGET_WAIT: if (writer_done) begin writer_finished <= 1'b1; state <= attempt_base != 0 ? ATTEMPT_START : COMPLETE; end
                ATTEMPT_START: if (writer_start_ready && !abort_request) begin
                    writer_started <= 1'b1; writer_finished <= 1'b0; attempt_beat <= '0; state <= ATTEMPT_STREAM;
                end
                ATTEMPT_STREAM: if (writer_data_ready) begin
                    if (attempt_beat == 8) state <= ATTEMPT_WAIT;
                    else attempt_beat <= attempt_beat + 4'd1;
                end
                ATTEMPT_WAIT: if (writer_done) begin writer_finished <= 1'b1; state <= COMPLETE; end
                PENDING_READ_START: if (reader_start_ready) begin
                    pending_read_finished <= 1'b0; state <= PENDING_WRITE_START;
                end
                PENDING_WRITE_START: if (writer_start_ready) begin
                    writer_started <= 1'b1; writer_finished <= 1'b0; state <= PENDING_STREAM;
                end
                PENDING_STREAM: begin
                    if (reader_done) pending_read_finished <= 1'b1;
                    if (writer_done) writer_finished <= 1'b1;
                    if (reader_valid && writer_data_ready && reader_last) state <= PENDING_WAIT;
                end
                PENDING_WAIT: begin
                    if (reader_done) pending_read_finished <= 1'b1;
                    if (writer_done) writer_finished <= 1'b1;
                    if ((pending_read_finished || reader_done) && (writer_finished || writer_done))
                        state <= joint_result_base != 0 ? JOINT_START : budget_base != 0 ? BUDGET_START : COMPLETE;
                end
                // Publish only after every metadata/state write has completed.
                // The descriptor reserves the final aligned beat of its output region.
                RESULT_START: if (writer_start_ready && !abort_request) begin
                    writer_started <= 1'b1; writer_finished <= 1'b0; state <= RESULT_STREAM;
                end
                RESULT_STREAM: if (writer_data_ready) state <= RESULT_WAIT;
                RESULT_WAIT: if (writer_done) begin
                    writer_finished <= 1'b1; publish_result <= 1'b0; state <= COMPLETE;
                end
                COMPLETE: begin
                    if (publish_result && !terminal_error) state <= RESULT_START;
                    else if (done_ready) state <= IDLE;
                end
                ABORT_DRAIN: begin
                    if (packer_abort_ack || packer_done) packer_drained <= 1'b1;
                    if (writer_abort_done || writer_done) writer_drained <= 1'b1;
                    if ((reader_start_ready || reader_abort_done || reader_done) &&
                            (packer_drained || packer_abort_ack || packer_done) &&
                            (writer_drained || writer_abort_done || writer_done)) begin
                        abort_ack <= aborting; state <= aborting ? ABORT_LOW : COMPLETE;
                    end
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && reader_valid && state == RECORD_WAIT)
            assert (reader_last && reader_byte_enable == 16'hff && reader_offset == 0)
                else $error("refresh_token_metadata_writer invalid row source record");
        if (!rst && reader_valid && pending_stream)
            assert (reader_byte_enable == 16'hffff && reader_offset[3:0] == 0 &&
                reader_offset < {21'd0, pending_bytes} && reader_last == (reader_offset+32'd16 == {21'd0, pending_bytes}))
                else $error("refresh_token_metadata_writer invalid pending record stream");
        if (!rst && state == CHOOSE_ROUND)
            assert ({2'd0, round_a8} <= remaining_a8 && remaining_rounds != 0)
                else $error("refresh_token_metadata_writer round search exceeded remaining rows");
        if (!rst && state == SCAN_RECORD && scratch_fifo_output_valid &&
                !scratch_stream_finished)
            assert (record_index < total_token_count)
                else $error("refresh_token_metadata_writer did not fill its selected round");
        if (!rst && scratch_rsp_valid && scratch_stream_active)
            assert (scratch_fifo_input_ready && scratch_read_outstanding != 0)
                else $error("refresh_token_metadata_writer scratch response overflow or underflow");
        if (!rst && packer_done && !packer_error)
            assert (packer_compute_groups <= 6'd4 && packer_semantic_groups == 6'd1)
                else $error("refresh_token_metadata_writer exceeded activation slots");
    end
`endif
endmodule

`default_nettype wire
