`default_nettype none
module operator_dma_writer #(
    parameter integer DATA_WIDTH = 128,
    parameter integer TAG_WIDTH = 8,
    parameter logic [TAG_WIDTH-1:0] REQUEST_TAG = 8'h92,
    parameter integer MAX_PENDING_WRITES = 1
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     abort_request,
    input  logic                     start_valid,
    output logic                     start_ready,
    input  logic [63:0]              start_address,
    input  logic [31:0]              start_bytes,
    output logic                     request_valid,
    input  logic                     request_ready,
    output logic [63:0]              request_address,
    output logic [31:0]              request_bytes,
    output logic [TAG_WIDTH-1:0]     request_tag,
    input  logic                     data_valid,
    output logic                     data_ready,
    input  logic [DATA_WIDTH-1:0]    data,
    input  logic [DATA_WIDTH/8-1:0]  data_byte_enable,
    input  logic                     data_last,
    output logic                     write_valid,
    input  logic                     write_ready,
    output logic [DATA_WIDTH-1:0]    write_data,
    output logic [DATA_WIDTH/8-1:0]  write_byte_enable,
    output logic                     write_last,
    input  logic                     transaction_done,
    input  logic                     transaction_error,
    output logic                     done_pulse,
    output logic                     error,
    output logic                     abort_ack
);
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    localparam integer PENDING_COUNT_WIDTH = MAX_PENDING_WRITES > 1 ?
        $clog2(MAX_PENDING_WRITES + 1) : 1;
    typedef enum logic [2:0] {IDLE, REQUEST, STREAM, WAIT_DONE, ABORT_DRAIN} state_t;
    state_t state;
    logic [31:0] accepted_bytes;
    logic [31:0] enabled_bytes;
    logic [31:0] expected_beat_bytes;
    logic abort_pending;
    logic abort_hold_valid;
    logic [DATA_WIDTH-1:0] abort_hold_data;
    logic [DATA_WIDTH/8-1:0] abort_hold_byte_enable;
    logic abort_hold_last;
    logic [DATA_WIDTH/8-1:0] abort_fill_byte_enable;
    logic [PENDING_COUNT_WIDTH-1:0] pending_write_count;
    logic request_fire;
    logic completion_event;
    logic completion_error_event;
    logic abort_last_completion;

    assign start_ready = state == IDLE && !abort_request && !abort_pending &&
        pending_write_count < PENDING_COUNT_WIDTH'(MAX_PENDING_WRITES);
    // A request that has not handshaken is still revocable. Suppress valid in
    // the abort cycle so request_ready cannot accept work that will not drain.
    assign request_valid = state == REQUEST && !abort_request;
    assign request_fire = request_valid && request_ready;
    assign completion_event = transaction_done || transaction_error;
    assign completion_error_event = completion_event && transaction_error;
    assign abort_last_completion = completion_event &&
        pending_write_count == PENDING_COUNT_WIDTH'(1) &&
        (abort_pending || abort_request);
    assign request_tag = REQUEST_TAG;
    assign expected_beat_bytes = request_bytes - accepted_bytes > BEAT_BYTES ?
        BEAT_BYTES : request_bytes - accepted_bytes;
    assign abort_fill_byte_enable =
        {DATA_WIDTH/8{1'b1}} >> (BEAT_BYTES - expected_beat_bytes);
    assign data_ready = state == STREAM && write_ready;
    assign write_valid = state == STREAM ? data_valid : state == ABORT_DRAIN;
    assign write_data = state == STREAM ? data :
        abort_hold_valid ? abort_hold_data : '0;
    assign write_byte_enable = state == STREAM ? data_byte_enable :
        abort_hold_valid ? abort_hold_byte_enable : abort_fill_byte_enable;
    assign write_last = state == STREAM ? data_last :
        abort_hold_valid ? abort_hold_last :
        accepted_bytes + expected_beat_bytes == request_bytes;
    assign enabled_bytes = $countones(write_byte_enable);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            request_address <= '0;
            request_bytes <= '0;
            accepted_bytes <= '0;
            abort_pending <= 1'b0;
            abort_hold_valid <= 1'b0;
            abort_hold_data <= '0;
            abort_hold_byte_enable <= '0;
            abort_hold_last <= 1'b0;
            pending_write_count <= '0;
            done_pulse <= 1'b0;
            error <= 1'b0;
            abort_ack <= 1'b0;
        end else begin
            done_pulse <= 1'b0;
            error <= 1'b0;
            abort_ack <= 1'b0;
            case ({request_fire, completion_event})
                2'b10: pending_write_count <= pending_write_count + 1'b1;
                2'b01: pending_write_count <= pending_write_count - 1'b1;
                default: pending_write_count <= pending_write_count;
            endcase
            if (completion_event) begin
                done_pulse <= 1'b1;
                error <= transaction_error;
                if (completion_error_event) begin
                    // With multiple outstanding writes, this response can
                    // belong to an older request while a later request is
                    // still being issued or streamed. Preserve that state;
                    // the parent stops new work after observing the error.
                    if (abort_last_completion) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= IDLE;
                    end else if (state == WAIT_DONE ||
                                 (state == STREAM &&
                                  pending_write_count ==
                                      PENDING_COUNT_WIDTH'(1))) begin
                        state <= IDLE;
                    end
                end else if (abort_last_completion) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= IDLE;
                end
            end
            if (!abort_last_completion &&
                !(completion_error_event &&
                  (state == WAIT_DONE ||
                   (state == STREAM && pending_write_count ==
                    PENDING_COUNT_WIDTH'(1))))) case (state)
                IDLE: begin
                    if (abort_request) begin
                        if (pending_write_count == 0)
                            abort_ack <= 1'b1;
                        else
                            abort_pending <= 1'b1;
                    end else if (start_valid && start_ready) begin
                        request_address <= start_address;
                        request_bytes <= start_bytes;
                        accepted_bytes <= '0;
                        abort_hold_valid <= 1'b0;
                        if (start_bytes == 0 ||
                            start_address + start_bytes < start_address) begin
                            done_pulse <= 1'b1;
                            error <= 1'b1;
                        end else begin
                            state <= REQUEST;
                        end
                    end
                end
                REQUEST: begin
                    if (abort_request) begin
                        if (pending_write_count == 0)
                            abort_ack <= 1'b1;
                        else
                            abort_pending <= 1'b1;
                        state <= IDLE;
                    end else if (request_fire) begin
                        state <= STREAM;
                    end
                end
                STREAM, ABORT_DRAIN: begin
                    if (abort_request) begin
                        abort_pending <= 1'b1;
                        state <= ABORT_DRAIN;
                        if (state == STREAM && data_valid && !write_ready) begin
                            abort_hold_valid <= 1'b1;
                            abort_hold_data <= data;
                            abort_hold_byte_enable <= data_byte_enable;
                            abort_hold_last <= data_last;
                        end
                    end
                    if (state == STREAM && data_valid && data_ready) begin
                        if (enabled_bytes != expected_beat_bytes ||
                            data_last != (accepted_bytes + enabled_bytes == request_bytes)) begin
                            done_pulse <= 1'b1;
                            error <= 1'b1;
                            state <= IDLE;
                        end else if (data_last) begin
                            accepted_bytes <= accepted_bytes + enabled_bytes;
                            state <= MAX_PENDING_WRITES == 1 ?
                                WAIT_DONE : IDLE;
                        end else begin
                            accepted_bytes <= accepted_bytes + enabled_bytes;
                        end
                    end else if (state == ABORT_DRAIN && write_valid &&
                                 write_ready) begin
                        abort_hold_valid <= 1'b0;
                        if (accepted_bytes + expected_beat_bytes ==
                            request_bytes) begin
                            accepted_bytes <= request_bytes;
                            state <= WAIT_DONE;
                        end else begin
                            accepted_bytes <=
                                accepted_bytes + expected_beat_bytes;
                        end
                    end
                end
                WAIT_DONE: begin
                    if (abort_request)
                        abort_pending <= 1'b1;
                    if (completion_event &&
                        !(abort_pending || abort_request)) begin
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (DATA_WIDTH != 128 || TAG_WIDTH < 8 ||
            MAX_PENDING_WRITES < 1 || MAX_PENDING_WRITES > 8)
            $error("operator_dma_writer requires a 128-bit stream, at least 8 tag bits and 1..8 pending writes");
    end

`ifndef SYNTHESIS
    logic request_was_stalled;
    logic [95:0] stalled_request;
    logic data_was_stalled;
    logic [DATA_WIDTH+DATA_WIDTH/8:0] stalled_data;
    always_ff @(posedge clk) begin
        if (rst) begin
            request_was_stalled <= 1'b0;
            stalled_request <= '0;
            data_was_stalled <= 1'b0;
            stalled_data <= '0;
        end else begin
            if (request_was_stalled && !abort_request && state != IDLE)
                assert (request_valid && {request_address, request_bytes} == stalled_request)
                    else $error("operator_dma_writer changed request while stalled");
            if (data_was_stalled && !completion_error_event)
                assert (write_valid &&
                    {write_data, write_byte_enable, write_last} == stalled_data)
                    else $error("operator_dma_writer changed write data while stalled");
            request_was_stalled <= request_valid && !request_ready && !abort_request;
            data_was_stalled <= write_valid && !write_ready &&
                !completion_error_event;
            if (request_valid && !request_ready)
                stalled_request <= {request_address, request_bytes};
            if (write_valid && !write_ready)
                stalled_data <= {write_data, write_byte_enable, write_last};
            assert (!(request_valid && abort_request))
                else $error("operator_dma_writer exposed a revocable request during abort");
            assert (!(completion_event && pending_write_count == 0))
                else $error("operator_dma_writer received completion without a pending write");
            assert (pending_write_count <=
                    PENDING_COUNT_WIDTH'(MAX_PENDING_WRITES))
                else $error("operator_dma_writer pending write count exceeded its limit");
        end
    end
`endif
endmodule

`default_nettype wire
