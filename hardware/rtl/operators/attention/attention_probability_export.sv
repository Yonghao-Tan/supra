`default_nettype none

// Exports a BF16 batch before P8 writes, or the actual streamed PV P8 operands.
// Two 128-byte tiles cover the SRAM return and current DMA drain; a registered
// 16-byte output removes the wide tile-word mux from the DMA ready path.
module attention_probability_export (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [11:0] start_sequence_length,
    input logic [7:0] start_row_mask,
    input logic start_p8_stream,
    input logic [255:0] start_key_groups,
    input logic [63:0] start_output_address, start_output_limit,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic [31:0] output_bytes,

    input logic stream_valid,
    output logic stream_ready,
    input logic [11:0] stream_key,
    input logic [1023:0] stream_values,

    output logic source_request_valid,
    input logic source_request_ready,
    output logic [11:0] source_request_key,
    output logic [15:0] source_request_tag,
    input logic source_response_valid,
    input logic [1023:0] source_response_values,
    input logic [63:0] source_response_lane_mask,
    input logic [15:0] source_response_tag,

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
    typedef enum logic [3:0] {
        IDLE, COUNT_GROUPS, CHECK_OUTPUT, WRITER_START, LOAD_GROUP_WORD,
        FIND_GROUP, WAIT_DONE, COMPLETE, ABORT_DRAIN, ABORT_LOW
    } state_t;
    state_t state;
    logic [11:0] sequence_length;
    logic [7:0] row_mask;
    logic [255:0] key_groups;
    logic [63:0] output_address, output_limit;
    logic [64:0] output_end;
    logic [8:0] group_count, issued_groups, received_groups;
    logic [4:0] group_word;
    logic [7:0] active_groups, selected_group_byte;
    logic [2:0] first_group;
    logic [8:0] allowed_group_count;
    logic [7:0] allowed_group_byte;
    logic [31:0] transfer_bytes;
    logic source_pending, fill_slot, drain_slot;
    logic [63:0] requested_mask;
    logic [15:0] requested_tag;
    logic [1:0] tile_valid;
    logic [127:0] tile_words [0:1][0:7];
    logic [2:0] drain_word;
    logic word_valid, word_last;
    logic [127:0] word_data;
    logic [31:0] staged_bytes;
    logic writer_start_ready, writer_data_ready, writer_done, writer_error, writer_abort_ack;
    logic writer_started, writer_finished, writer_drained;
    logic terminal_error, aborting;
    logic [7:0] terminal_error_id;
    logic stage_word, request_fire;
    logic response_bad;
    logic p8_stream, stream_finished, stream_selected, stream_fire, stream_bad;
    logic [11:0] next_stream_key;

    function automatic logic [2:0] lowest_group(input logic [7:0] bits);
        casez (bits)
            8'b???????1: lowest_group = 3'd0;
            8'b??????10: lowest_group = 3'd1;
            8'b?????100: lowest_group = 3'd2;
            8'b????1000: lowest_group = 3'd3;
            8'b???10000: lowest_group = 3'd4;
            8'b??100000: lowest_group = 3'd5;
            8'b?1000000: lowest_group = 3'd6;
            default: lowest_group = 3'd7;
        endcase
    endfunction

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign selected_group_byte = key_groups[group_word*8 +: 8];
    assign allowed_group_count = 9'(({1'b0, sequence_length} + 13'd7) >> 3);
    always_comb begin
        allowed_group_byte = '0;
        for (integer group_bit = 0; group_bit < 8; group_bit++)
            allowed_group_byte[group_bit] = {1'b0, group_word, 3'd0} + 9'(group_bit) < allowed_group_count;
    end
    assign first_group = lowest_group(active_groups);
    assign transfer_bytes = {16'd0, group_count, 7'd0};
    assign output_end = {1'b0, output_address} + {33'd0, transfer_bytes};
    assign source_request_valid = state == FIND_GROUP && active_groups != 0 &&
        !p8_stream && !source_pending && !tile_valid[fill_slot] && !abort_request;
    assign source_request_key = {1'b0, group_word, first_group, 3'd0};
    assign source_request_tag = {8'h60, group_word, first_group};
    assign request_fire = source_request_valid && source_request_ready;
    assign stream_selected = key_groups[stream_key[10:3]];
    assign stream_ready = p8_stream && state == FIND_GROUP && !stream_finished &&
        (!stream_selected || !tile_valid[fill_slot]) && !abort_request;
    assign stream_fire = stream_valid && stream_ready;
    assign stream_bad = stream_fire && (stream_key != next_stream_key || stream_key >= sequence_length);
    assign response_bad = stream_bad || (!p8_stream && source_response_valid && (!source_pending || source_response_tag != requested_tag ||
        (source_response_lane_mask & requested_mask) != requested_mask));
    assign stage_word = (state == LOAD_GROUP_WORD || state == FIND_GROUP || state == WAIT_DONE) &&
        tile_valid[drain_slot] && (!word_valid || writer_data_ready) && !abort_request;

    operator_dma_writer #(.REQUEST_TAG(8'hb6)) writer (
        .clk, .rst, .abort_request(state == ABORT_DRAIN || abort_request || response_bad || writer_error),
        .start_valid(state == WRITER_START && !abort_request), .start_ready(writer_start_ready),
        .start_address(output_address), .start_bytes(transfer_bytes),
        .request_valid(write_request_valid), .request_ready(write_request_ready),
        .request_address(write_request_address), .request_bytes(write_request_bytes), .request_tag(write_request_tag),
        .data_valid(word_valid), .data_ready(writer_data_ready), .data(word_data),
        .data_byte_enable(16'hffff), .data_last(word_last), .write_valid, .write_ready,
        .write_data, .write_byte_enable, .write_last, .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_ack)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            sequence_length <= '0; row_mask <= '0; key_groups <= '0;
            p8_stream <= 1'b0; stream_finished <= 1'b0; next_stream_key <= '0;
            output_address <= '0; output_limit <= '0; output_bytes <= '0;
            group_count <= '0; issued_groups <= '0; received_groups <= '0;
            group_word <= '0; active_groups <= '0;
            source_pending <= 1'b0; fill_slot <= 1'b0; drain_slot <= 1'b0;
            requested_mask <= '0; requested_tag <= '0; tile_valid <= '0; drain_word <= '0;
            word_valid <= 1'b0; word_last <= 1'b0; word_data <= '0; staged_bytes <= '0;
            writer_started <= 1'b0; writer_finished <= 1'b0; writer_drained <= 1'b0;
            terminal_error <= 1'b0; terminal_error_id <= '0; aborting <= 1'b0; abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (request_fire) begin
                source_pending <= 1'b1;
                requested_tag <= source_request_tag;
                issued_groups <= issued_groups + 9'd1;
                active_groups[first_group] <= 1'b0;
                for (integer row = 0; row < 8; row++)
                    for (integer key = 0; key < 8; key++)
                        requested_mask[row*8+key] <= row_mask[row] && source_request_key + 12'(key) < sequence_length;
            end
            if (source_response_valid && source_pending) begin
                source_pending <= 1'b0;
                received_groups <= received_groups + 9'd1;
                if (state != ABORT_DRAIN && !abort_request && !response_bad) begin
                    for (integer row = 0; row < 8; row++)
                        for (integer key = 0; key < 8; key++)
                            tile_words[fill_slot][row][key*16 +: 16] <= requested_mask[row*8+key] ?
                                source_response_values[(row*8+key)*16 +: 16] : 16'd0;
                    tile_valid[fill_slot] <= 1'b1;
                    fill_slot <= !fill_slot;
                end
            end
            if (stream_fire && !stream_bad) begin
                next_stream_key <= next_stream_key + 12'd8;
                stream_finished <= {1'b0, stream_key} + 13'd8 >= {1'b0, sequence_length};
                if (stream_selected) begin
                    for (integer row = 0; row < 8; row++) begin
                        tile_words[fill_slot][row] <= row_mask[row] ? stream_values[row*128 +: 128] : 128'd0;
                        for (integer key = 0; key < 8; key++)
                            if (stream_key + 12'(key) >= sequence_length)
                                tile_words[fill_slot][row][key*8 +: 8] <= '0;
                    end
                    tile_valid[fill_slot] <= 1'b1;
                    fill_slot <= !fill_slot;
                    issued_groups <= issued_groups + 9'd1;
                    received_groups <= received_groups + 9'd1;
                end
            end
            if (word_valid && writer_data_ready) word_valid <= 1'b0;
            if (stage_word) begin
                word_valid <= 1'b1;
                word_data <= tile_words[drain_slot][drain_word];
                word_last <= staged_bytes + 32'd16 == transfer_bytes;
                staged_bytes <= staged_bytes + 32'd16;
                drain_word <= drain_word + 3'd1;
                if (drain_word == 3'd7) begin
                    tile_valid[drain_slot] <= 1'b0;
                    drain_slot <= !drain_slot;
                end
            end
            if (writer_done) writer_finished <= 1'b1;
            if ((abort_request || writer_error || response_bad) &&
                    state != ABORT_DRAIN && state != ABORT_LOW && state != COMPLETE) begin
                aborting <= abort_request;
                if (!abort_request) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= writer_error ? 8'h04 : 8'h03;
                end
                word_valid <= 1'b0;
                tile_valid <= '0;
                writer_drained <= !writer_started || writer_done || writer_finished;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    sequence_length <= start_sequence_length;
                    p8_stream <= start_p8_stream;
                    stream_finished <= 1'b0; next_stream_key <= '0;
                    row_mask <= start_row_mask; key_groups <= start_key_groups;
                    output_address <= start_output_address; output_limit <= start_output_limit;
                    output_bytes <= '0; group_word <= '0; group_count <= '0;
                    issued_groups <= '0; received_groups <= '0;
                    fill_slot <= 1'b0; drain_slot <= 1'b0; drain_word <= '0;
                    tile_valid <= '0; word_valid <= 1'b0; source_pending <= 1'b0; staged_bytes <= '0;
                    writer_started <= 1'b0; writer_finished <= 1'b0;
                    terminal_error <= 1'b0; terminal_error_id <= '0;
                    if (start_sequence_length == 0 || start_sequence_length > 12'd2048 || start_output_address[3:0] != 0) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                    end else state <= start_row_mask == 0 ? COMPLETE : COUNT_GROUPS;
                end
                COUNT_GROUPS: begin
                    group_count <= group_count + 9'($countones(selected_group_byte));
                    group_word <= group_word + 5'd1;
                    if ((selected_group_byte & ~allowed_group_byte) != 0) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                    end else if (group_word == 5'd31) state <= CHECK_OUTPUT;
                end
                CHECK_OUTPUT: begin
                    // Empty selections do not access the output buffer. The
                    // parent's per-batch address may already be past its limit.
                    if (group_count == 0) state <= COMPLETE;
                    else if (output_end[64] || output_end[63:0] > output_limit) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h02; state <= COMPLETE;
                    end else begin output_bytes <= transfer_bytes; state <= WRITER_START; end
                end
                WRITER_START: if (writer_start_ready) begin
                    writer_started <= 1'b1;
                    state <= p8_stream ? FIND_GROUP : LOAD_GROUP_WORD;
                end
                LOAD_GROUP_WORD: begin active_groups <= selected_group_byte; state <= FIND_GROUP; end
                FIND_GROUP: begin
                    if (p8_stream) begin
                        if (stream_finished) state <= WAIT_DONE;
                    end else if (active_groups == 0) begin
                        group_word <= group_word + 5'd1;
                        state <= group_word == 5'd31 ? WAIT_DONE : LOAD_GROUP_WORD;
                    end
                end
                WAIT_DONE: if ((writer_done || writer_finished) && !source_pending && tile_valid == 0 && !word_valid) begin
                    if (received_groups != group_count || issued_groups != group_count) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h03;
                    end
                    state <= COMPLETE;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: begin
                    if (writer_abort_ack || writer_done) writer_drained <= 1'b1;
                    if (!source_pending && (writer_drained || writer_abort_ack || writer_done)) begin
                        abort_ack <= aborting;
                        state <= aborting ? ABORT_LOW : COMPLETE;
                    end
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (source_response_valid && state != ABORT_DRAIN)
                assert (source_pending && !tile_valid[fill_slot])
                    else $error("attention_probability_export has no reserved SRAM return slot");
            if (stage_word && drain_word == 3'd7 && source_response_valid && !response_bad)
                assert (fill_slot != drain_slot)
                    else $error("attention_probability_export filled and released the same tile");
        end
    end
`endif
endmodule

`default_nettype wire
