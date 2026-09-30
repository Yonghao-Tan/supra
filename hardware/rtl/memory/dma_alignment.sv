`default_nettype none

module dma_alignment #(
    parameter integer DATA_WIDTH = 128,
    parameter integer TAG_WIDTH = 8
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic                         request_write,
    input  logic [63:0]                  request_address,
    input  logic [31:0]                  request_bytes,
    input  logic [TAG_WIDTH-1:0]         request_tag,
    output logic                         request_done,
    output logic                         request_error,

    output logic                         read_data_valid,
    input  logic                         read_data_ready,
    output logic [DATA_WIDTH-1:0]        read_data,
    output logic [DATA_WIDTH/8-1:0]      read_byte_enable,
    output logic                         read_data_last,
    output logic [TAG_WIDTH-1:0]         read_data_tag,

    input  logic                         write_data_valid,
    output logic                         write_data_ready,
    input  logic [DATA_WIDTH-1:0]        write_data,
    input  logic [DATA_WIDTH/8-1:0]      write_byte_enable,
    input  logic                         write_data_last,

    output logic                         aligned_request_valid,
    input  logic                         aligned_request_ready,
    output logic                         aligned_request_write,
    output logic [63:0]                  aligned_request_address,
    output logic [31:0]                  aligned_request_bytes,
    output logic [TAG_WIDTH-1:0]         aligned_request_tag,
    input  logic                         aligned_request_done,
    input  logic                         aligned_request_error,

    input  logic                         aligned_read_valid,
    output logic                         aligned_read_ready,
    input  logic [DATA_WIDTH-1:0]        aligned_read_data,
    input  logic [DATA_WIDTH/8-1:0]      aligned_read_byte_enable,
    input  logic                         aligned_read_last,
    input  logic [TAG_WIDTH-1:0]         aligned_read_tag,

    output logic                         aligned_write_valid,
    input  logic                         aligned_write_ready,
    output logic [DATA_WIDTH-1:0]        aligned_write_data,
    output logic [DATA_WIDTH/8-1:0]      aligned_write_byte_enable,
    output logic                         aligned_write_last
);
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    localparam integer BEAT_SHIFT = $clog2(BEAT_BYTES);
    localparam integer FIFO_BYTES = 2 * BEAT_BYTES;
    localparam integer FIFO_BITS = FIFO_BYTES * 8;
    localparam integer FIFO_COUNT_WIDTH = $clog2(FIFO_BYTES + 1);
    localparam integer WRITE_COMPLETION_DEPTH = 8;
    localparam integer WRITE_COMPLETION_PTR_WIDTH = 3;
    localparam integer WRITE_COMPLETION_COUNT_WIDTH = 4;
    localparam logic [31:0] BEAT_BYTES_32 = BEAT_BYTES;
    localparam logic [31:0] FIFO_BYTES_32 = FIFO_BYTES;
    localparam logic [31:0] BEAT_BYTES_MINUS_ONE_32 = BEAT_BYTES - 1;

    typedef enum logic [2:0] {
        IDLE,
        ISSUE_ALIGNED_REQUEST,
        TRANSFER_READ,
        TRANSFER_WRITE,
        WAIT_ALIGNED_DONE,
        REPORT_ERROR
    } state_t;

    state_t state;
    logic saved_write;
    logic [63:0] saved_aligned_address;
    logic [31:0] saved_bytes;
    logic [TAG_WIDTH-1:0] saved_tag;
    logic [BEAT_SHIFT-1:0] saved_byte_offset;
    logic [31:0] saved_aligned_bytes;
    logic [FIFO_BITS-1:0] byte_fifo;
    logic [FIFO_BITS-1:0] byte_fifo_next;
    logic [FIFO_COUNT_WIDTH-1:0] fifo_count;
    logic [31:0] logical_received;
    logic [31:0] logical_delivered;
    logic [31:0] physical_transferred;
    logic aligned_done_seen;
    logic aligned_error_seen;
    logic protocol_error;
    logic write_completion_payload_done [0:WRITE_COMPLETION_DEPTH-1];
    logic write_completion_aligned_done [0:WRITE_COMPLETION_DEPTH-1];
    logic write_completion_error [0:WRITE_COMPLETION_DEPTH-1];
    logic [WRITE_COMPLETION_PTR_WIDTH-1:0] write_completion_write;
    logic [WRITE_COMPLETION_PTR_WIDTH-1:0] write_completion_read;
    logic [WRITE_COMPLETION_PTR_WIDTH-1:0] write_active_entry;
    logic [WRITE_COMPLETION_COUNT_WIDTH-1:0] write_completion_count;

    logic [31:0] read_output_bytes;
    logic [31:0] write_input_bytes;
    logic [31:0] write_beat_prefix;
    logic [31:0] write_beat_bytes;
    logic [31:0] read_append_bytes;
    logic read_output_handshake;
    logic aligned_read_handshake;
    logic write_input_handshake;
    logic aligned_write_handshake;
    logic [31:0] request_byte_offset;
    logic [31:0] saved_byte_offset_32;
    logic [31:0] fifo_count_32;
    logic [31:0] read_retained_bytes;
    logic [31:0] write_retained_bytes;
    logic [31:0] read_beat_start;
    logic [31:0] read_beat_end;
    logic [31:0] read_logical_start;
    logic [31:0] read_logical_end;
    logic [31:0] read_copy_start;
    logic [31:0] read_copy_end;
    logic [31:0] read_copy_start_lane;
    logic request_invalid;
    logic request_fire, write_completion_pop;
    logic [WRITE_COMPLETION_PTR_WIDTH-1:0] aligned_done_target;
    logic aligned_done_has_target;
    logic read_terminal_error_event, write_terminal_error_event;

    function automatic logic [BEAT_BYTES-1:0] low_byte_mask(
        input logic [31:0] byte_count
    );
        integer byte_index;
        begin
            low_byte_mask = '0;
            for (byte_index = 0; byte_index < BEAT_BYTES; byte_index = byte_index + 1)
                if (byte_index < byte_count)
                    low_byte_mask[byte_index] = 1'b1;
        end
    endfunction

    function automatic logic [FIFO_BITS-1:0] low_fifo_mask(
        input logic [31:0] byte_count
    );
        begin
            low_fifo_mask = {FIFO_BITS{1'b1}} >>
                ((FIFO_BYTES_32 - byte_count) * 8);
        end
    endfunction

    assign request_byte_offset =
        {{(32-BEAT_SHIFT){1'b0}}, request_address[BEAT_SHIFT-1:0]};
    assign saved_byte_offset_32 = {{(32-BEAT_SHIFT){1'b0}}, saved_byte_offset};
    assign fifo_count_32 = {{(32-FIFO_COUNT_WIDTH){1'b0}}, fifo_count};
    assign request_invalid = request_bytes == 0 ||
        request_address > 64'hffff_ffff_ffff_ffff - {{32{1'b0}}, request_bytes} ||
        request_bytes > 32'hffff_ffff - request_byte_offset - BEAT_BYTES_MINUS_ONE_32;

    assign write_completion_pop = write_completion_count != 0 &&
        write_completion_payload_done[write_completion_read] &&
        write_completion_aligned_done[write_completion_read];
    assign request_ready = !abort_request && state == IDLE &&
        (request_write ?
            (write_completion_count < WRITE_COMPLETION_COUNT_WIDTH'(
                WRITE_COMPLETION_DEPTH) && !write_completion_pop) :
            (write_completion_count == 0));
    assign request_fire = request_valid && request_ready;
    assign aligned_done_target = write_completion_read +
        WRITE_COMPLETION_PTR_WIDTH'(write_completion_pop);
    assign aligned_done_has_target = write_completion_count >
        WRITE_COMPLETION_COUNT_WIDTH'(write_completion_pop);
    assign read_terminal_error_event = (aligned_done_seen && aligned_error_seen) ||
        (aligned_request_done && aligned_request_error);
    assign write_terminal_error_event = aligned_request_done && aligned_request_error &&
        aligned_done_has_target && aligned_done_target == write_active_entry;
    assign aligned_request_valid = state == ISSUE_ALIGNED_REQUEST;
    assign aligned_request_write = saved_write;
    assign aligned_request_address = saved_aligned_address;
    assign aligned_request_bytes = saved_aligned_bytes;
    assign aligned_request_tag = saved_tag;

    assign read_output_bytes = saved_bytes - logical_delivered < BEAT_BYTES ?
        saved_bytes - logical_delivered : BEAT_BYTES;
    assign read_data_valid = state == TRANSFER_READ &&
        (fifo_count_32 >= BEAT_BYTES_32 ||
         (physical_transferred == saved_aligned_bytes && fifo_count != 0));
    assign read_byte_enable = low_byte_mask(read_output_bytes);
    assign read_data_last = logical_delivered + read_output_bytes == saved_bytes;
    assign read_data_tag = saved_tag;
    assign read_output_handshake = read_data_valid && read_data_ready;

    assign read_retained_bytes = fifo_count_32 -
        (read_output_handshake ? read_output_bytes : 32'd0);
    assign read_beat_start = physical_transferred;
    assign read_beat_end = physical_transferred + BEAT_BYTES_32;
    assign read_logical_start = saved_byte_offset_32;
    assign read_logical_end = saved_byte_offset_32 + saved_bytes;
    assign read_copy_start = read_beat_start < read_logical_start ?
        read_logical_start : read_beat_start;
    assign read_copy_end = read_beat_end < read_logical_end ?
        read_beat_end : read_logical_end;
    assign read_copy_start_lane = read_copy_start - read_beat_start;
    assign read_append_bytes = read_copy_end > read_copy_start ?
        read_copy_end - read_copy_start : 32'd0;
    assign aligned_read_ready = state == TRANSFER_READ &&
        read_retained_bytes + read_append_bytes <= FIFO_BYTES_32;
    assign aligned_read_handshake = aligned_read_valid && aligned_read_ready;

    assign write_input_bytes = saved_bytes - logical_received < BEAT_BYTES ?
        saved_bytes - logical_received : BEAT_BYTES;
    assign write_retained_bytes = fifo_count_32 -
        (aligned_write_handshake ? write_beat_bytes : 32'd0);
    assign write_data_ready = state == TRANSFER_WRITE &&
        logical_received < saved_bytes &&
        write_retained_bytes + write_input_bytes <= FIFO_BYTES_32;
    assign write_input_handshake = write_data_valid && write_data_ready;
    assign write_beat_prefix = physical_transferred == 0 ? saved_byte_offset_32 : 32'd0;
    assign write_beat_bytes = saved_bytes - logical_delivered < BEAT_BYTES - write_beat_prefix ?
        saved_bytes - logical_delivered : BEAT_BYTES - write_beat_prefix;
    assign aligned_write_valid = state == TRANSFER_WRITE &&
        physical_transferred < saved_aligned_bytes && fifo_count_32 >= write_beat_bytes;
    assign aligned_write_last = physical_transferred + BEAT_BYTES == saved_aligned_bytes;
    assign aligned_write_handshake = aligned_write_valid && aligned_write_ready;

    assign read_data = byte_fifo[DATA_WIDTH-1:0];
    assign aligned_write_data =
        byte_fifo[DATA_WIDTH-1:0] << (write_beat_prefix * 8);
    assign aligned_write_byte_enable =
        low_byte_mask(write_beat_bytes) << write_beat_prefix;

    always_comb begin : build_byte_fifo_next
        integer retained_count;
        logic [FIFO_BITS-1:0] retained_payload;
        logic [FIFO_BITS-1:0] append_payload;

        byte_fifo_next = byte_fifo;
        retained_count = 0;
        retained_payload = '0;
        append_payload = '0;

        if (state == TRANSFER_READ &&
            (aligned_read_handshake || read_output_handshake)) begin
            retained_count = fifo_count_32 -
                (read_output_handshake ? read_output_bytes : 32'd0);
            retained_payload = (read_output_handshake ?
                byte_fifo >> (read_output_bytes * 8) : byte_fifo) &
                low_fifo_mask(retained_count);
            if (aligned_read_handshake)
                append_payload =
                    ({{DATA_WIDTH{1'b0}}, aligned_read_data} >>
                        (read_copy_start_lane * 8)) <<
                    (retained_count * 8);
            byte_fifo_next = retained_payload | append_payload;
        end else if (state == TRANSFER_WRITE &&
                     (write_input_handshake || aligned_write_handshake)) begin
            retained_count = fifo_count_32 -
                (aligned_write_handshake ? write_beat_bytes : 32'd0);
            retained_payload = (aligned_write_handshake ?
                byte_fifo >> (write_beat_bytes * 8) : byte_fifo) &
                low_fifo_mask(retained_count);
            if (write_input_handshake)
                append_payload = {{DATA_WIDTH{1'b0}}, write_data} <<
                    (retained_count * 8);
            byte_fifo_next = retained_payload | append_payload;
        end
    end

    always_ff @(posedge clk) begin : update_alignment_state
        if (rst) begin
            state <= IDLE;
            saved_write <= 1'b0;
            saved_aligned_address <= '0;
            saved_bytes <= '0;
            saved_tag <= '0;
            saved_byte_offset <= '0;
            saved_aligned_bytes <= '0;
            byte_fifo <= '0;
            fifo_count <= '0;
            logical_received <= '0;
            logical_delivered <= '0;
            physical_transferred <= '0;
            aligned_done_seen <= 1'b0;
            aligned_error_seen <= 1'b0;
            protocol_error <= 1'b0;
            write_completion_write <= '0;
            write_completion_read <= '0;
            write_active_entry <= '0;
            write_completion_count <= '0;
            for (integer completion = 0;
                 completion < WRITE_COMPLETION_DEPTH; completion = completion + 1) begin
                write_completion_payload_done[completion] <= 1'b0;
                write_completion_aligned_done[completion] <= 1'b0;
                write_completion_error[completion] <= 1'b0;
            end
            request_done <= 1'b0;
            request_error <= 1'b0;
        end else begin
            request_done <= 1'b0;
            request_error <= 1'b0;

            if (abort_request && state != IDLE)
                protocol_error <= 1'b1;
            if (abort_request)
                for (integer completion = 0;
                     completion < WRITE_COMPLETION_DEPTH; completion = completion + 1)
                    write_completion_error[completion] <= 1'b1;

            if (aligned_request_done) begin
                if (aligned_done_has_target) begin
                    write_completion_aligned_done[aligned_done_target] <= 1'b1;
                    if (aligned_request_error)
                        write_completion_error[aligned_done_target] <= 1'b1;
                end else begin
                    aligned_done_seen <= 1'b1;
                    if (aligned_request_error)
                        aligned_error_seen <= 1'b1;
                end
            end

            if (write_completion_pop) begin
                request_done <= 1'b1;
                request_error <= abort_request ||
                    write_completion_error[write_completion_read];
                write_completion_payload_done[write_completion_read] <= 1'b0;
                write_completion_aligned_done[write_completion_read] <= 1'b0;
                write_completion_error[write_completion_read] <= 1'b0;
                write_completion_read <= write_completion_read + 1'b1;
            end

            case ({request_fire && request_write, write_completion_pop})
                2'b10: write_completion_count <= write_completion_count + 1'b1;
                2'b01: write_completion_count <= write_completion_count - 1'b1;
                default: write_completion_count <= write_completion_count;
            endcase

            case (state)
                IDLE: if (request_fire) begin
                    saved_write <= request_write;
                    saved_aligned_address <=
                        {request_address[63:BEAT_SHIFT], {BEAT_SHIFT{1'b0}}};
                    saved_bytes <= request_bytes;
                    saved_tag <= request_tag;
                    saved_byte_offset <= request_address[BEAT_SHIFT-1:0];
                    saved_aligned_bytes <=
                        (request_bytes + request_byte_offset + BEAT_BYTES_MINUS_ONE_32) &
                        ~BEAT_BYTES_MINUS_ONE_32;
                    byte_fifo <= '0;
                    fifo_count <= '0;
                    logical_received <= '0;
                    logical_delivered <= '0;
                    physical_transferred <= '0;
                    aligned_done_seen <= 1'b0;
                    aligned_error_seen <= 1'b0;
                    protocol_error <= 1'b0;
                    if (request_write) begin
                        write_active_entry <= write_completion_write;
                        write_completion_payload_done[write_completion_write] <=
                            request_invalid;
                        write_completion_aligned_done[write_completion_write] <=
                            request_invalid;
                        write_completion_error[write_completion_write] <=
                            request_invalid;
                        write_completion_write <= write_completion_write + 1'b1;
                        if (request_invalid)
                            state <= IDLE;
                        else
                            state <= ISSUE_ALIGNED_REQUEST;
                    end else if (request_invalid) begin
                        protocol_error <= 1'b1;
                        state <= REPORT_ERROR;
                    end else begin
                        state <= ISSUE_ALIGNED_REQUEST;
                    end
                end

                ISSUE_ALIGNED_REQUEST: begin
                    if (abort_request) begin
                        if (saved_write) begin
                            write_completion_payload_done[write_active_entry] <= 1'b1;
                            write_completion_aligned_done[write_active_entry] <= 1'b1;
                            write_completion_error[write_active_entry] <= 1'b1;
                            state <= IDLE;
                        end else begin
                            request_done <= 1'b1;
                            request_error <= 1'b1;
                            state <= IDLE;
                        end
                    end else if (aligned_request_valid && aligned_request_ready) begin
                        state <= saved_write ? TRANSFER_WRITE : TRANSFER_READ;
                    end
                end

                TRANSFER_READ: begin
                    if (read_terminal_error_event) begin
                        request_done <= 1'b1;
                        request_error <= 1'b1;
                        state <= IDLE;
                    end else if (aligned_read_handshake ||
                                 read_output_handshake) begin
                        if (read_output_handshake) begin
                            logical_delivered <=
                                logical_delivered + read_output_bytes;
                            if (read_data_last)
                                state <= WAIT_ALIGNED_DONE;
                        end
                        if (aligned_read_handshake) begin
                            logical_received <=
                                logical_received + read_append_bytes;
                            physical_transferred <=
                                physical_transferred + BEAT_BYTES;
                            if (aligned_read_byte_enable !=
                                    {BEAT_BYTES{1'b1}} ||
                                aligned_read_tag != saved_tag ||
                                aligned_read_last !=
                                    (physical_transferred + BEAT_BYTES ==
                                     saved_aligned_bytes))
                                protocol_error <= 1'b1;
                        end
                        byte_fifo <= byte_fifo_next;
                        fifo_count <= FIFO_COUNT_WIDTH'(
                            fifo_count_32 +
                            (aligned_read_handshake ? read_append_bytes : 32'd0) -
                            (read_output_handshake ? read_output_bytes : 32'd0));
                    end
                end

                TRANSFER_WRITE: begin
                    if (write_terminal_error_event) begin
                        write_completion_payload_done[write_active_entry] <= 1'b1;
                        write_completion_error[write_active_entry] <= 1'b1;
                        fifo_count <= '0;
                        state <= IDLE;
                    end else if (write_input_handshake ||
                                 aligned_write_handshake) begin
                        if (aligned_write_handshake) begin
                            logical_delivered <=
                                logical_delivered + write_beat_bytes;
                            physical_transferred <=
                                physical_transferred + BEAT_BYTES;
                            if (aligned_write_last) begin
                                write_completion_payload_done[write_active_entry]
                                    <= 1'b1;
                                if (abort_request || protocol_error ||
                                    logical_delivered + write_beat_bytes !=
                                        saved_bytes ||
                                    logical_received != saved_bytes ||
                                    fifo_count_32 != write_beat_bytes ||
                                    physical_transferred + BEAT_BYTES !=
                                        saved_aligned_bytes)
                                    write_completion_error[write_active_entry]
                                        <= 1'b1;
                                state <= IDLE;
                            end
                        end
                        if (write_input_handshake) begin
                            logical_received <=
                                logical_received + write_input_bytes;
                            if (write_byte_enable !=
                                    low_byte_mask(write_input_bytes) ||
                                write_data_last !=
                                    (logical_received + write_input_bytes ==
                                     saved_bytes))
                                protocol_error <= 1'b1;
                        end
                        byte_fifo <= byte_fifo_next;
                        fifo_count <= FIFO_COUNT_WIDTH'(
                            fifo_count_32 +
                            (write_input_handshake ? write_input_bytes : 32'd0) -
                            (aligned_write_handshake ? write_beat_bytes : 32'd0));
                    end
                end

                WAIT_ALIGNED_DONE: begin
                    if (aligned_done_seen || aligned_request_done) begin
                        request_done <= 1'b1;
                        request_error <= abort_request || protocol_error || aligned_error_seen ||
                            aligned_request_error || logical_delivered != saved_bytes ||
                            logical_received != saved_bytes || fifo_count != 0 ||
                            physical_transferred != saved_aligned_bytes;
                        state <= IDLE;
                    end
                end

                REPORT_ERROR: begin
                    request_done <= 1'b1;
                    request_error <= 1'b1;
                    state <= IDLE;
                end

                default: state <= REPORT_ERROR;
            endcase
        end
    end

    initial begin
        if ((DATA_WIDTH != 128 && DATA_WIDTH != 256) || TAG_WIDTH < 1 ||
            FIFO_BYTES != 2 * BEAT_BYTES)
            $error("dma_alignment requires a 128-bit or 256-bit data path");
    end

`ifndef SYNTHESIS
    logic request_was_stalled;
    logic [104:0] stalled_request;
    logic read_was_stalled;
    logic [DATA_WIDTH+DATA_WIDTH/8+TAG_WIDTH:0] stalled_read;
    logic write_was_stalled;
    logic [DATA_WIDTH+DATA_WIDTH/8:0] stalled_write;
    always_ff @(posedge clk) begin
        if (rst) begin
            request_was_stalled <= 1'b0;
            stalled_request <= '0;
            read_was_stalled <= 1'b0;
            stalled_read <= '0;
            write_was_stalled <= 1'b0;
            stalled_write <= '0;
        end else begin
            if (request_was_stalled && !abort_request)
                assert (aligned_request_valid &&
                    {aligned_request_write, aligned_request_address,
                     aligned_request_bytes, aligned_request_tag} == stalled_request)
                    else $error("dma alignment adapter changed an aligned request while stalled");
            if (read_was_stalled && !abort_request && !read_terminal_error_event)
                assert (read_data_valid &&
                    {read_data, read_byte_enable, read_data_last, read_data_tag} == stalled_read)
                    else $error("dma alignment adapter changed read data while stalled");
            if (write_was_stalled && !abort_request && !write_terminal_error_event)
                assert (aligned_write_valid &&
                    {aligned_write_data, aligned_write_byte_enable,
                     aligned_write_last} == stalled_write)
                    else $error("dma alignment adapter changed write data while stalled");
            request_was_stalled <= aligned_request_valid && !aligned_request_ready &&
                !abort_request;
            read_was_stalled <= read_data_valid && !read_data_ready && !abort_request &&
                !read_terminal_error_event;
            write_was_stalled <= aligned_write_valid && !aligned_write_ready &&
                !abort_request && !write_terminal_error_event;
            if (aligned_request_valid && !aligned_request_ready)
                stalled_request <= {aligned_request_write, aligned_request_address,
                    aligned_request_bytes, aligned_request_tag};
            if (read_data_valid && !read_data_ready)
                stalled_read <= {read_data, read_byte_enable, read_data_last, read_data_tag};
            if (aligned_write_valid && !aligned_write_ready)
                stalled_write <= {aligned_write_data, aligned_write_byte_enable,
                    aligned_write_last};
        end
    end
`endif
endmodule

`default_nettype wire
