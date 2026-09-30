`default_nettype none

// Writes the complete next-state image and per-block events before publishing
// one completion record. Only the completion response makes the new version
// visible to the next forward.
module token_state_writer (
    input  logic clk,
    input  logic rst,
    input  logic abort_request,
    output logic abort_ack,

    input  logic start_valid,
    output logic start_ready,
    input  logic [63:0] start_state_base,
    input  logic [63:0] start_state_limit,
    input  logic [7:0] start_state_entry_count,
    input  logic [63:0] start_metadata_base,
    input  logic [63:0] start_metadata_limit,
    input  logic [15:0] start_metadata_bytes,
    input  logic [63:0] start_event_base,
    input  logic [63:0] start_event_limit,
    input  logic [2:0] start_block_count,
    input  logic [63:0] start_completion_base,
    input  logic [63:0] start_completion_limit,
    input  logic [127:0] start_completion_data,

    input  logic state_data_valid,
    output logic state_data_ready,
    input  logic [127:0] state_data,
    input  logic metadata_data_valid,
    output logic metadata_data_ready,
    input  logic [127:0] metadata_data,
    input  logic event_data_valid,
    output logic event_data_ready,
    input  logic [127:0] event_data,

    output logic memory_request_valid,
    input  logic memory_request_ready,
    output logic [63:0] memory_request_address,
    output logic [31:0] memory_request_bytes,
    output logic [7:0] memory_request_tag,
    output logic memory_data_valid,
    input  logic memory_data_ready,
    output logic [127:0] memory_data,
    output logic [15:0] memory_data_byte_enable,
    output logic memory_data_last,
    input  logic memory_done_pulse,
    input  logic memory_error,

    output logic done_valid,
    input  logic done_ready,
    output logic error,
    output logic completion_published,
    output logic [63:0] accepted_state_beat_count,
    output logic [63:0] accepted_metadata_beat_count,
    output logic [63:0] accepted_event_beat_count
);
    localparam logic [7:0] DMA_TAG = 8'hb7;
    localparam logic [31:0] STATE_ENTRY_BYTES = 32'd32;
    localparam logic [31:0] EVENT_ENTRY_BYTES = 32'd64;
    localparam logic [31:0] COMPLETION_BYTES = 32'd16;

    typedef enum logic [3:0] {
        IDLE,
        STATE_START,
        STATE_STREAM,
        STATE_WAIT,
        METADATA_START,
        METADATA_STREAM,
        METADATA_WAIT,
        EVENT_START,
        EVENT_STREAM,
        EVENT_WAIT,
        COMPLETION_START,
        COMPLETION_STREAM,
        COMPLETION_WAIT,
        FINISH,
        ABORT_WAIT_LOW
    } state_t;

    state_t state;
    logic [63:0] saved_state_base;
    logic [31:0] saved_state_bytes;
    logic [63:0] saved_event_base;
    logic [31:0] saved_event_bytes;
    logic [63:0] saved_metadata_base;
    logic [15:0] saved_metadata_bytes;
    logic [63:0] saved_completion_base;
    logic [127:0] saved_completion_data;
    logic [15:0] state_beat_count;
    logic [7:0] event_beat_count;
    logic [7:0] metadata_beat_count;
    logic abort_pending;
    logic completion_request_accepted;
    logic completion_written;
    logic terminal_error;

    logic writer_abort_request;
    logic writer_abort_ack;
    logic writer_start_valid;
    logic writer_start_ready;
    logic [63:0] writer_start_address;
    logic [31:0] writer_start_bytes;
    logic writer_input_valid;
    logic writer_input_ready;
    logic [127:0] writer_input_data;
    logic [15:0] writer_input_byte_enable;
    logic writer_input_last;
    logic writer_done_pulse;
    logic writer_error;
    logic writer_start_fire;
    logic writer_input_fire;
    logic completion_request_fire;

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == FINISH;
    assign error = done_valid && terminal_error;
    assign completion_published = completion_written;

    assign writer_start_valid = state == STATE_START ||
        state == METADATA_START || state == EVENT_START ||
        state == COMPLETION_START;
    assign writer_start_address = state == STATE_START ? saved_state_base :
        state == METADATA_START ? saved_metadata_base :
        state == EVENT_START ? saved_event_base : saved_completion_base;
    assign writer_start_bytes = state == STATE_START ? saved_state_bytes :
        state == METADATA_START ? {16'd0, saved_metadata_bytes} :
        state == EVENT_START ? saved_event_bytes : COMPLETION_BYTES;
    assign writer_start_fire = writer_start_valid && writer_start_ready;

    always_comb begin
        writer_input_valid = 1'b0;
        writer_input_data = '0;
        writer_input_byte_enable = 16'hffff;
        writer_input_last = 1'b0;
        state_data_ready = 1'b0;
        metadata_data_ready = 1'b0;
        event_data_ready = 1'b0;
        case (state)
            STATE_STREAM: begin
                writer_input_valid = state_data_valid && !abort_request &&
                    !abort_pending;
                writer_input_data = state_data;
                writer_input_last = state_beat_count + 16'd1 ==
                    16'(saved_state_bytes >> 4);
                state_data_ready = writer_input_ready && !abort_request &&
                    !abort_pending;
            end
            METADATA_STREAM: begin
                writer_input_valid = metadata_data_valid && !abort_request &&
                    !abort_pending;
                writer_input_data = metadata_data;
                writer_input_last = metadata_beat_count + 8'd1 ==
                    8'(saved_metadata_bytes >> 4);
                metadata_data_ready = writer_input_ready && !abort_request &&
                    !abort_pending;
            end
            EVENT_STREAM: begin
                writer_input_valid = event_data_valid && !abort_request &&
                    !abort_pending;
                writer_input_data = event_data;
                writer_input_last = event_beat_count + 8'd1 ==
                    8'(saved_event_bytes >> 4);
                event_data_ready = writer_input_ready && !abort_request &&
                    !abort_pending;
            end
            COMPLETION_STREAM: begin
                writer_input_valid = 1'b1;
                writer_input_data = saved_completion_data;
                writer_input_last = 1'b1;
            end
            default: begin end
        endcase
    end
    assign writer_input_fire = writer_input_valid && writer_input_ready;

    assign writer_abort_request = (abort_pending || abort_request) &&
        !completion_request_accepted;
    assign completion_request_fire = memory_request_valid &&
        memory_request_ready &&
        (state == COMPLETION_STREAM || state == COMPLETION_WAIT);

    operator_dma_writer #(
        .DATA_WIDTH(128), .TAG_WIDTH(8), .REQUEST_TAG(DMA_TAG),
        .MAX_PENDING_WRITES(1)
    ) writer (
        .clk(clk), .rst(rst), .abort_request(writer_abort_request),
        .start_valid(writer_start_valid), .start_ready(writer_start_ready),
        .start_address(writer_start_address),
        .start_bytes(writer_start_bytes),
        .request_valid(memory_request_valid),
        .request_ready(memory_request_ready),
        .request_address(memory_request_address),
        .request_bytes(memory_request_bytes), .request_tag(memory_request_tag),
        .data_valid(writer_input_valid), .data_ready(writer_input_ready),
        .data(writer_input_data),
        .data_byte_enable(writer_input_byte_enable),
        .data_last(writer_input_last), .write_valid(memory_data_valid),
        .write_ready(memory_data_ready), .write_data(memory_data),
        .write_byte_enable(memory_data_byte_enable),
        .write_last(memory_data_last), .transaction_done(memory_done_pulse),
        .transaction_error(memory_error), .done_pulse(writer_done_pulse),
        .error(writer_error), .abort_ack(writer_abort_ack));

`ifdef SYNTHESIS
    always_comb begin
        accepted_state_beat_count = '0;
        accepted_metadata_beat_count = '0;
        accepted_event_beat_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        logic [64:0] state_end;
        logic [64:0] event_end;
        logic [64:0] metadata_end;
        logic [64:0] completion_end;
        if (rst) begin
            state <= IDLE;
            saved_state_base <= '0;
            saved_state_bytes <= '0;
            saved_event_base <= '0;
            saved_event_bytes <= '0;
            saved_metadata_base <= '0;
            saved_metadata_bytes <= '0;
            saved_completion_base <= '0;
            saved_completion_data <= '0;
            state_beat_count <= '0;
            event_beat_count <= '0;
            metadata_beat_count <= '0;
            abort_pending <= 1'b0;
            completion_request_accepted <= 1'b0;
            completion_written <= 1'b0;
            terminal_error <= 1'b0;
            abort_ack <= 1'b0;
`ifndef SYNTHESIS
            accepted_state_beat_count <= '0;
            accepted_metadata_beat_count <= '0;
            accepted_event_beat_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            if (completion_request_fire)
                completion_request_accepted <= 1'b1;

            if (abort_request && state != IDLE && state != FINISH &&
                state != ABORT_WAIT_LOW)
                abort_pending <= 1'b1;

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    state_end = {1'b0, start_state_base} +
                        65'(start_state_entry_count) * STATE_ENTRY_BYTES;
                    event_end = {1'b0, start_event_base} +
                        65'(start_block_count) * EVENT_ENTRY_BYTES;
                    metadata_end = {1'b0, start_metadata_base} +
                        {49'd0, start_metadata_bytes};
                    completion_end = {1'b0, start_completion_base} +
                        65'(COMPLETION_BYTES);
                    terminal_error <= 1'b0;
                    abort_pending <= 1'b0;
                    completion_request_accepted <= 1'b0;
                    completion_written <= 1'b0;
`ifndef SYNTHESIS
                    accepted_state_beat_count <= '0;
                    accepted_metadata_beat_count <= '0;
                    accepted_event_beat_count <= '0;
`endif
                    if (start_state_entry_count == 0 ||
                        start_state_entry_count > 128 ||
                        start_block_count == 0 || start_block_count > 4 ||
                        start_state_base[3:0] != 0 ||
                        start_event_base[3:0] != 0 ||
                        start_metadata_base[3:0] != 0 ||
                        start_metadata_limit[3:0] != 0 ||
                        start_metadata_bytes[3:0] != 0 ||
                        start_completion_base[3:0] != 0 ||
                        state_end[64] || event_end[64] || metadata_end[64] ||
                        completion_end[64] ||
                        state_end > {1'b0, start_state_limit} ||
                        event_end > {1'b0, start_event_limit} ||
                        (start_metadata_bytes != 0 &&
                         (start_metadata_base == start_metadata_limit ||
                          metadata_end > {1'b0, start_metadata_limit})) ||
                        completion_end > {1'b0, start_completion_limit}) begin
                        terminal_error <= 1'b1;
                        state <= FINISH;
                    end else begin
                        saved_state_base <= start_state_base;
                        saved_state_bytes <=
                            32'(start_state_entry_count) * STATE_ENTRY_BYTES;
                        saved_event_base <= start_event_base;
                        saved_event_bytes <=
                            32'(start_block_count) * EVENT_ENTRY_BYTES;
                        saved_metadata_base <= start_metadata_base;
                        saved_metadata_bytes <= start_metadata_bytes;
                        saved_completion_base <= start_completion_base;
                        saved_completion_data <= start_completion_data;
                        state <= STATE_START;
                    end
                end

                STATE_START: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_start_fire) begin
                        state_beat_count <= '0;
                        state <= STATE_STREAM;
                    end
                end
                STATE_STREAM: begin
                    if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_done_pulse) begin
                        terminal_error <= 1'b1;
                        state <= FINISH;
                    end else if (writer_input_fire) begin
                        state_beat_count <= state_beat_count + 1'b1;
`ifndef SYNTHESIS
                        accepted_state_beat_count <=
                            accepted_state_beat_count + 1'b1;
`endif
                        if (writer_input_last)
                            state <= STATE_WAIT;
                    end
                end
                STATE_WAIT: begin
                    if (writer_done_pulse && !abort_pending && !abort_request) begin
                        if (writer_error) begin
                            terminal_error <= 1'b1;
                            state <= FINISH;
                        end else
                            state <= saved_metadata_bytes != 0 ?
                                METADATA_START : EVENT_START;
                    end else if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end
                end

                METADATA_START: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_start_fire) begin
                        metadata_beat_count <= '0;
                        state <= METADATA_STREAM;
                    end
                end
                METADATA_STREAM: begin
                    if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_done_pulse) begin
                        terminal_error <= 1'b1;
                        state <= FINISH;
                    end else if (writer_input_fire) begin
                        metadata_beat_count <= metadata_beat_count + 1'b1;
`ifndef SYNTHESIS
                        accepted_metadata_beat_count <=
                            accepted_metadata_beat_count + 1'b1;
`endif
                        if (writer_input_last)
                            state <= METADATA_WAIT;
                    end
                end
                METADATA_WAIT: begin
                    if (writer_done_pulse && !abort_pending && !abort_request) begin
                        if (writer_error) begin
                            terminal_error <= 1'b1;
                            state <= FINISH;
                        end else
                            state <= EVENT_START;
                    end else if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end
                end

                EVENT_START: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_start_fire) begin
                        event_beat_count <= '0;
                        state <= EVENT_STREAM;
                    end
                end
                EVENT_STREAM: begin
                    if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_done_pulse) begin
                        terminal_error <= 1'b1;
                        state <= FINISH;
                    end else if (writer_input_fire) begin
                        event_beat_count <= event_beat_count + 1'b1;
`ifndef SYNTHESIS
                        accepted_event_beat_count <=
                            accepted_event_beat_count + 1'b1;
`endif
                        if (writer_input_last)
                            state <= EVENT_WAIT;
                    end
                end
                EVENT_WAIT: begin
                    if (writer_done_pulse && !abort_pending && !abort_request) begin
                        if (writer_error) begin
                            terminal_error <= 1'b1;
                            state <= FINISH;
                        end else
                            state <= COMPLETION_START;
                    end else if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end
                end

                COMPLETION_START: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_start_fire) begin
                        completion_request_accepted <= 1'b0;
                        state <= COMPLETION_STREAM;
                    end
                end
                COMPLETION_STREAM: begin
                    if (writer_abort_ack) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (writer_input_fire)
                        state <= COMPLETION_WAIT;
                end
                COMPLETION_WAIT: if (writer_done_pulse) begin
                    terminal_error <= writer_error;
                    completion_written <= !writer_error;
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else
                        state <= FINISH;
                end

                FINISH: if (done_valid && done_ready)
                    state <= IDLE;
                ABORT_WAIT_LOW: if (!abort_request)
                    state <= IDLE;
                default: begin
                    terminal_error <= 1'b1;
                    state <= FINISH;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (rst)
        done_valid && !done_ready |=>
            done_valid && $stable({error, completion_published}))
        else $error("forward-postprocess state completion changed while stalled");
    assert property (@(posedge clk) disable iff (rst)
        completion_request_accepted && abort_pending |->
            !writer_abort_request)
        else $error("forward-postprocess writer revoked an accepted completion request");
`endif
endmodule

`default_nettype wire
