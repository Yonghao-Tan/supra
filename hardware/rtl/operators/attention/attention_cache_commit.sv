`default_nettype none

// Selected Layer0 positions arrive sorted. Merge adjacent positions into runs,
// then copy K, K-scale and all sixteen V stripes through the existing DMA path.
// Bases/limits are packed as source K,V,scale followed by destination K,V,scale.
module attention_cache_commit (
    input logic clk,
    input logic rst,
    input logic abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [11:0] start_sequence_length,
    input logic [8:0] start_selected_count,
    input logic [383:0] start_bases,
    input logic [383:0] start_limits,
    input logic [31:0] start_head_stride,
    input logic [31:0] start_scale_head_stride,
    input logic position_table_ready,
    output logic position_read_req_valid,
    input logic position_read_req_ready,
    output logic [8:0] position_read_req_index,
    input logic position_read_rsp_valid,
    input logic [19:0] position_read_rsp_data,
    input logic copy_enable,
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
    input logic read_error,
    input logic read_abort_ack,

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
    input logic write_done,
    input logic write_error
);
    localparam integer MAX_SELECTED = 432;
    typedef enum logic [3:0] {
        IDLE, CHECK_EXTENT, CHECK_LIMIT, CHECK_OVERLAP,
        LOAD_PLANE, LOAD_RUN, PREPARE_COPY, COPY_START, COPY_WAIT, NEXT_RUN,
        COMPLETE, ABORT_DRAIN, ABORT_LOW, WAIT_COPY
    } state_t;
    state_t state;
    logic [63:0] bases [0:5], limits [0:5];
    logic [31:0] head_stride, scale_head_stride;
    logic [11:0] sequence_length;
    logic [8:0] selected_count, run_index;
    logic [10:0] loaded_run_start;
    logic [8:0] loaded_run_length;
    logic [11:0] previous_run_end;
    logic [2:0] check_index, overlap_left, overlap_right;
    logic [64:0] required_end;
    logic [4:0] head;
    logic [1:0] kind;
    logic [3:0] v_stripe;
    logic [63:0] source_plane, destination_plane;
    logic [63:0] head_offset, scale_head_offset;
    logic [63:0] copy_source, copy_destination;
    logic [31:0] copy_bytes;
    logic reader_started, writer_started, reader_finished, writer_finished;
    logic reader_start_ready, writer_start_ready;
    logic reader_done, reader_error, reader_abort_done;
    logic writer_done, writer_error, writer_abort_done;
    logic reader_valid, reader_ready, reader_last;
    logic [127:0] reader_data;
    logic [15:0] reader_byte_enable;
    logic [31:0] reader_offset;
    logic writer_data_ready;
    logic aborting, reader_drained, writer_drained;
    logic terminal_error;
    logic [7:0] terminal_error_id;
    logic run_fetch_pending, prefetched_run_valid;
    logic [19:0] prefetched_run;
    logic [19:0] selected_run;
    logic current_run_finishes_plane, any_copy_active, future_copy_exists;

    assign start_ready = state == IDLE && !abort_request;
    assign current_run_finishes_plane =
        {1'b0, run_index} + {1'b0, loaded_run_length} >= {1'b0, selected_count};
    assign any_copy_active = bases[0] != bases[3] || bases[1] != bases[4] ||
        bases[2] != bases[5];
    assign future_copy_exists = !current_run_finishes_plane ||
        (kind == 2'd0 && (bases[1] != bases[4] || bases[2] != bases[5] ||
                         (head != 5'd31 && any_copy_active))) ||
        (kind == 2'd1 && (v_stripe != 4'd15 || bases[2] != bases[5] ||
                         (head != 5'd31 && any_copy_active))) ||
        (kind == 2'd2 && head != 5'd31 && any_copy_active);
    assign position_read_req_valid = !abort_request && !run_fetch_pending &&
        !prefetched_run_valid &&
        ((state == WAIT_COPY && copy_enable && position_table_ready && any_copy_active) ||
         (state == COPY_WAIT &&
          (reader_finished || reader_done) && (writer_finished || writer_done) &&
          future_copy_exists));
    assign position_read_req_index = state == COPY_WAIT && !current_run_finishes_plane ?
        run_index + loaded_run_length : 9'd0;
    assign selected_run = prefetched_run_valid ? prefetched_run : position_read_rsp_data;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign reader_ready = state == ABORT_DRAIN || writer_data_ready;

    operator_dma_reader #(.REQUEST_TAG(8'hb2)) reader (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid(state == COPY_START && !reader_started && !abort_request),
        .start_ready(reader_start_ready), .start_address(copy_source),
        .start_bytes(copy_bytes), .start_total_bytes(copy_bytes),
        .request_valid(read_request_valid), .request_ready(read_request_ready),
        .request_address(read_request_address), .request_bytes(read_request_bytes),
        .request_tag(read_request_tag), .read_valid, .read_ready, .read_data,
        .read_byte_enable, .read_last, .read_tag, .response_error(read_error),
        .upstream_abort_ack(read_abort_ack), .data_valid(reader_valid), .data_ready(reader_ready),
        .data(reader_data), .data_byte_enable(reader_byte_enable), .data_last(reader_last),
        .data_byte_offset(reader_offset), .done_pulse(reader_done), .error(reader_error),
        .abort_ack(reader_abort_done)
    );
    operator_dma_writer #(.REQUEST_TAG(8'hb3)) writer (
        .clk, .rst, .abort_request(state == ABORT_DRAIN),
        .start_valid(state == COPY_START && !writer_started && !abort_request),
        .start_ready(writer_start_ready), .start_address(copy_destination), .start_bytes(copy_bytes),
        .request_valid(write_request_valid), .request_ready(write_request_ready),
        .request_address(write_request_address), .request_bytes(write_request_bytes),
        .request_tag(write_request_tag), .data_valid(reader_valid), .data_ready(writer_data_ready),
        .data(reader_data), .data_byte_enable(reader_byte_enable), .data_last(reader_last),
        .write_valid, .write_ready, .write_data, .write_byte_enable, .write_last,
        .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_done)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            for (integer item = 0; item < 6; item++) begin
                bases[item] <= '0;
                limits[item] <= '0;
            end
            head_stride <= '0;
            scale_head_stride <= '0;
            sequence_length <= '0;
            selected_count <= '0;
            run_index <= '0;
            loaded_run_start <= '0;
            loaded_run_length <= '0;
            previous_run_end <= '0;
            check_index <= '0;
            overlap_left <= '0;
            overlap_right <= 3'd1;
            required_end <= '0;
            head <= '0;
            kind <= '0;
            v_stripe <= '0;
            source_plane <= '0;
            destination_plane <= '0;
            head_offset <= '0;
            scale_head_offset <= '0;
            copy_source <= '0;
            copy_destination <= '0;
            copy_bytes <= '0;
            reader_started <= 1'b0;
            writer_started <= 1'b0;
            reader_finished <= 1'b0;
            writer_finished <= 1'b0;
            reader_drained <= 1'b0;
            writer_drained <= 1'b0;
            aborting <= 1'b0;
            abort_ack <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            run_fetch_pending <= 1'b0;
            prefetched_run_valid <= 1'b0;
            prefetched_run <= '0;
        end else begin
            abort_ack <= 1'b0;
            if (position_read_req_valid && position_read_req_ready)
                run_fetch_pending <= 1'b1;
            if (position_read_rsp_valid) begin
                run_fetch_pending <= 1'b0;
                if (state != LOAD_RUN) begin
                    prefetched_run <= position_read_rsp_data;
                    prefetched_run_valid <= 1'b1;
                end
            end
            if ((abort_request || reader_error || writer_error) &&
                    state != IDLE && state != ABORT_DRAIN && state != ABORT_LOW) begin
                aborting <= abort_request;
                if (!abort_request) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= reader_error ? 8'h04 : 8'h05;
                end
                reader_drained <= !reader_started || reader_finished || reader_done || reader_start_ready;
                writer_drained <= !writer_started || writer_finished || writer_done || writer_start_ready;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    for (integer item = 0; item < 6; item++) begin
                        bases[item] <= start_bases[item*64 +: 64];
                        limits[item] <= start_limits[item*64 +: 64];
                    end
                    sequence_length <= start_sequence_length;
                    selected_count <= start_selected_count;
                    head_stride <= start_head_stride;
                    scale_head_stride <= start_scale_head_stride;
                    run_index <= '0;
                    previous_run_end <= '0;
                    head <= '0;
                    kind <= '0;
                    v_stripe <= '0;
                    head_offset <= '0;
                    scale_head_offset <= '0;
                    check_index <= '0;
                    overlap_left <= '0;
                    overlap_right <= 3'd1;
                    reader_started <= 1'b0;
                    writer_started <= 1'b0;
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    run_fetch_pending <= 1'b0;
                    prefetched_run_valid <= 1'b0;
                    if (start_sequence_length == 0 || start_sequence_length > 12'd2048 ||
                            start_selected_count > 9'(MAX_SELECTED) || {3'd0, start_selected_count} > start_sequence_length ||
                            start_head_stride < 32'd262144 || start_head_stride[3:0] != 0 ||
                            start_scale_head_stride < 32'd4096 || start_scale_head_stride[3:0] != 0) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= 8'h01;
                        state <= COMPLETE;
                    end else state <= CHECK_EXTENT;
                end
                CHECK_EXTENT: begin
                    required_end <= {1'b0, bases[check_index]} +
                        (check_index == 3'd2 || check_index == 3'd5 ?
                            (65'(scale_head_stride) << 5) - 65'(scale_head_stride) + 65'd4096 :
                            (65'(head_stride) << 5) - 65'(head_stride) + 65'd262144);
                    state <= CHECK_LIMIT;
                end
                CHECK_LIMIT: begin
                    if (bases[check_index][3:0] != 0 || required_end[64] ||
                            required_end[63:0] > limits[check_index]) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= 8'h02;
                        state <= COMPLETE;
                    end else if (check_index == 3'd5) state <= CHECK_OVERLAP;
                    else begin check_index <= check_index + 3'd1; state <= CHECK_EXTENT; end
                end
                CHECK_OVERLAP: begin
                    if (bases[overlap_left] < limits[overlap_right] && bases[overlap_right] < limits[overlap_left] &&
                            !(overlap_right == overlap_left + 3'd3 && bases[overlap_left] == bases[overlap_right] &&
                              limits[overlap_left] == limits[overlap_right])) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= 8'h02;
                        state <= COMPLETE;
                    end else if (overlap_left == 3'd4)
                        state <= selected_count == 0 ? COMPLETE : WAIT_COPY;
                    else if (overlap_right == 3'd5) begin
                        overlap_left <= overlap_left + 3'd1;
                        overlap_right <= overlap_left + 3'd2;
                    end else overlap_right <= overlap_right + 3'd1;
                end
                WAIT_COPY: if (copy_enable && position_table_ready &&
                        (!any_copy_active ||
                         (position_read_req_valid && position_read_req_ready)))
                    state <= LOAD_PLANE;
                LOAD_PLANE: begin
                    source_plane <= bases[{1'b0, kind}] + (kind == 2 ? scale_head_offset : head_offset) +
                        (kind == 1 ? {46'd0, v_stripe, 14'd0} : 64'd0);
                    destination_plane <= bases[{1'b0, kind}+3'd3] + (kind == 2 ? scale_head_offset : head_offset) +
                        (kind == 1 ? {46'd0, v_stripe, 14'd0} : 64'd0);
                    run_index <= bases[{1'b0, kind}] == bases[{1'b0, kind}+3'd3] ? selected_count : 9'd0;
                    state <= bases[{1'b0, kind}] == bases[{1'b0, kind}+3'd3] ? NEXT_RUN : LOAD_RUN;
                end
                LOAD_RUN: if (prefetched_run_valid || position_read_rsp_valid) begin
                    prefetched_run_valid <= 1'b0;
                    loaded_run_start <= selected_run[19:9];
                    loaded_run_length <= selected_run[8:0];
                    previous_run_end <= {1'b0, selected_run[19:9]} +
                        {3'd0, selected_run[8:0]} - 12'd1;
                    if (selected_run[8:0] == 0 ||
                            {1'b0, run_index} + {1'b0, selected_run[8:0]} > {1'b0, selected_count} ||
                            {1'b0, selected_run[19:9]} + {3'd0, selected_run[8:0]} > sequence_length ||
                            (run_index != 0 && {1'b0, selected_run[19:9]} <= previous_run_end)) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= 8'h03;
                        state <= COMPLETE;
                    end else state <= PREPARE_COPY;
                end
                PREPARE_COPY: begin
                    copy_source <= source_plane + (kind == 0 ? {46'd0, loaded_run_start, 7'd0} :
                        kind == 1 ? {50'd0, loaded_run_start, 3'd0} : {52'd0, loaded_run_start, 1'b0});
                    copy_destination <= destination_plane + (kind == 0 ? {46'd0, loaded_run_start, 7'd0} :
                        kind == 1 ? {50'd0, loaded_run_start, 3'd0} : {52'd0, loaded_run_start, 1'b0});
                    copy_bytes <= kind == 0 ? {16'd0, loaded_run_length, 7'd0} :
                        kind == 1 ? {20'd0, loaded_run_length, 3'd0} :
                        {22'd0, loaded_run_length, 1'b0};
                    reader_started <= 1'b0;
                    writer_started <= 1'b0;
                    reader_finished <= 1'b0;
                    writer_finished <= 1'b0;
                    state <= COPY_START;
                end
                COPY_START: begin
                    if (reader_start_ready) reader_started <= 1'b1;
                    if (writer_start_ready) writer_started <= 1'b1;
                    if ((reader_started || reader_start_ready) && (writer_started || writer_start_ready)) state <= COPY_WAIT;
                end
                COPY_WAIT: begin
                    if (reader_done) reader_finished <= 1'b1;
                    if (writer_done) writer_finished <= 1'b1;
                    if ((reader_finished || reader_done) && (writer_finished || writer_done) &&
                            (!future_copy_exists ||
                             (position_read_req_valid && position_read_req_ready)))
                        state <= NEXT_RUN;
                end
                NEXT_RUN: begin
                    if ({1'b0, run_index} + {1'b0, loaded_run_length} < {1'b0, selected_count}) begin
                        run_index <= run_index + loaded_run_length;
                        state <= LOAD_RUN;
                    end
                    else if (kind == 1 && v_stripe != 4'd15) begin v_stripe <= v_stripe + 4'd1; state <= LOAD_PLANE; end
                    else if (kind != 2) begin kind <= kind + 2'd1; state <= LOAD_PLANE; end
                    else if (head != 5'd31) begin
                        head <= head + 5'd1;
                        kind <= '0;
                        v_stripe <= '0;
                        head_offset <= head_offset + {32'd0, head_stride};
                        scale_head_offset <= scale_head_offset + {32'd0, scale_head_stride};
                        state <= LOAD_PLANE;
                    end else state <= COMPLETE;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: begin
                    if (reader_abort_done || reader_done) reader_drained <= 1'b1;
                    if (writer_abort_done || writer_done) writer_drained <= 1'b1;
                    if ((reader_drained || reader_abort_done || reader_done) &&
                            (writer_drained || writer_abort_done || writer_done) &&
                            !run_fetch_pending && !position_read_rsp_valid) begin
                        state <= aborting ? ABORT_LOW : COMPLETE;
                        abort_ack <= aborting;
                        prefetched_run_valid <= 1'b0;
                    end
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && reader_valid && reader_ready && state != ABORT_DRAIN)
            assert (reader_offset < copy_bytes)
                else $error("attention_cache_commit received data outside its selected run");
        if (!rst && position_read_rsp_valid)
            assert ((run_fetch_pending && !prefetched_run_valid) ||
                    prefetched_run_valid || state == PREPARE_COPY ||
                    (state == COMPLETE && terminal_error_id == 8'h03))
                else $error("attention_cache_commit received an unexpected run descriptor state=%0d pending=%0b prefetched=%0b",
                    state, run_fetch_pending, prefetched_run_valid);
    end
`endif
endmodule

`default_nettype wire
