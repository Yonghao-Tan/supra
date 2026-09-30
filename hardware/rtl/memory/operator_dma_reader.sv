`default_nettype none
module operator_dma_reader #(
    parameter integer DATA_WIDTH = 128,
    parameter integer TAG_WIDTH = 8,
    parameter logic [TAG_WIDTH-1:0] REQUEST_TAG = TAG_WIDTH'(8'h80)
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    abort_request,
    input  logic                    start_valid,
    output logic                    start_ready,
    input  logic [63:0]             start_address,
    input  logic [31:0]             start_bytes,
    input  logic [31:0]             start_total_bytes,
    output logic                    request_valid,
    input  logic                    request_ready,
    output logic [63:0]             request_address,
    output logic [31:0]             request_bytes,
    output logic [TAG_WIDTH-1:0]    request_tag,
    input  logic                    read_valid,
    output logic                    read_ready,
    input  logic [DATA_WIDTH-1:0]   read_data,
    input  logic [DATA_WIDTH/8-1:0] read_byte_enable,
    input  logic                    read_last,
    input  logic [TAG_WIDTH-1:0]    read_tag,
    input  logic                    response_error,
    input  logic                    upstream_abort_ack,
    output logic                    data_valid,
    input  logic                    data_ready,
    output logic [DATA_WIDTH-1:0]   data,
    output logic [DATA_WIDTH/8-1:0] data_byte_enable,
    output logic                    data_last,
    output logic [31:0]             data_byte_offset,
    output logic                    done_pulse,
    output logic                    error,
    output logic                    abort_ack
);
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    typedef enum logic [1:0] {IDLE, REQUEST, STREAM, ABORT_DRAIN} state_t;
    state_t state;
    logic [31:0] accepted_bytes;
    logic [31:0] total_bytes;
    logic [31:0] expected_beat_bytes;
    logic [31:0] enabled_bytes;

    assign start_ready = state == IDLE;
    assign request_valid = state == REQUEST && !abort_request;
    assign request_tag = REQUEST_TAG;
    assign read_ready = (state == STREAM || state == ABORT_DRAIN) && data_ready;
    assign data_valid = state == STREAM && !abort_request && !response_error &&
        !upstream_abort_ack && read_valid && read_tag == REQUEST_TAG;
    assign data = read_data;
    assign data_byte_enable = read_byte_enable;
    assign data_last = read_last;
    assign data_byte_offset = accepted_bytes;
    assign enabled_bytes = $countones(read_byte_enable);
    assign expected_beat_bytes = total_bytes - accepted_bytes > BEAT_BYTES ?
        BEAT_BYTES : total_bytes - accepted_bytes;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            request_address <= '0;
            request_bytes <= '0;
            accepted_bytes <= '0;
            total_bytes <= '0;
            done_pulse <= 1'b0;
            error <= 1'b0;
            abort_ack <= 1'b0;
        end else begin
            done_pulse <= 1'b0;
            error <= 1'b0;
            abort_ack <= 1'b0;
            case (state)
                IDLE: if (start_valid) begin
                    request_address <= start_address;
                    request_bytes <= start_bytes;
                    total_bytes <= start_total_bytes;
                    accepted_bytes <= '0;
                    if (start_bytes == 0 || start_total_bytes < start_bytes ||
                        start_address + 64'(start_bytes) < start_address) begin
                        done_pulse <= 1'b1;
                        error <= 1'b1;
                    end else begin
                        state <= REQUEST;
                    end
                end
                REQUEST: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end else if (request_valid && request_ready) begin
                        state <= STREAM;
                    end
                end
                STREAM: begin
                    if (upstream_abort_ack) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end else if (response_error) begin
                        done_pulse <= 1'b1;
                        error <= 1'b1;
                        state <= IDLE;
                    end else if (abort_request) begin
                        state <= ABORT_DRAIN;
                        if (read_valid && read_ready && read_tag == REQUEST_TAG && read_last) begin
                            abort_ack <= 1'b1;
                            state <= IDLE;
                        end
                    end else if (read_valid && read_ready && read_tag == REQUEST_TAG) begin
                        if (enabled_bytes != expected_beat_bytes ||
                            read_last != (accepted_bytes + enabled_bytes == total_bytes)) begin
                            done_pulse <= 1'b1;
                            error <= 1'b1;
                            state <= IDLE;
                        end else if (read_last) begin
                            accepted_bytes <= accepted_bytes + enabled_bytes;
                            done_pulse <= 1'b1;
                            state <= IDLE;
                        end else begin
                            accepted_bytes <= accepted_bytes + enabled_bytes;
                        end
                    end
                end
                ABORT_DRAIN: begin
                    if (upstream_abort_ack) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end else if (read_valid && read_ready && read_tag == REQUEST_TAG && read_last) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if ((DATA_WIDTH != 128 && DATA_WIDTH != 256) || TAG_WIDTH < 8)
            $error("operator_dma_reader requires a 128- or 256-bit stream and at least 8 tag bits");
    end

`ifndef SYNTHESIS
    logic request_was_stalled;
    logic [95:0] stalled_request;
    always_ff @(posedge clk) begin
        if (rst) begin
            request_was_stalled <= 1'b0;
            stalled_request <= '0;
        end else begin
            if (request_was_stalled && !abort_request && state != IDLE)
                assert (request_valid && {request_address, request_bytes} == stalled_request)
                    else $error("operator_dma_reader changed request while stalled");
            request_was_stalled <= request_valid && !request_ready && !abort_request;
            if (request_valid && !request_ready)
                stalled_request <= {request_address, request_bytes};
            if (state == STREAM && read_valid && !response_error && !upstream_abort_ack)
                assert (read_tag == REQUEST_TAG)
                    else $error("operator_dma_reader received an unexpected response tag");
        end
    end
`endif
endmodule

`default_nettype wire
