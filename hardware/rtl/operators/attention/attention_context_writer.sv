`default_nettype none

// Drains one BF16 head-context buffer into at most 48 token-major 256-byte
// logical writes. The source bank is released after the final payload beat;
// operator completion still waits for the final write response.
module attention_context_writer #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     abort_request,
    output logic                     abort_ack,

    input  logic                     start_valid,
    output logic                     start_ready,
    input  logic [5:0]               start_query_count,
    input  logic [5:0]               start_head,
    input  logic [ADDR_WIDTH-1:0]    start_context_base,
    input  logic [31:0]              start_context_query_stride,
    input  logic [TAG_WIDTH-1:0]     start_tag,
    output logic                     busy,
    output logic                     done_pulse,
    output logic                     error,
    output logic [3:0]               error_id,
    output logic                     source_buffer_released,

    output logic                     context_read_req_valid,
    input  logic                     context_read_req_ready,
    output logic [9:0]               context_read_req_address,
    output logic [TAG_WIDTH-1:0]     context_read_req_tag,
    input  logic                     context_read_rsp_valid,
    output logic                     context_read_rsp_ready,
    input  logic [127:0]             context_read_rsp_data,
    input  logic [TAG_WIDTH-1:0]     context_read_rsp_tag,

    output logic                     write_request_valid,
    input  logic                     write_request_ready,
    output logic [ADDR_WIDTH-1:0]    write_request_address,
    output logic [31:0]              write_request_bytes,
    output logic [TAG_WIDTH-1:0]     write_request_tag,
    input  logic                     write_request_done,
    input  logic                     write_request_error,
    output logic                     write_data_valid,
    input  logic                     write_data_ready,
    output logic [127:0]             write_data,
    output logic [15:0]              write_byte_enable,
    output logic                     write_data_last,

    output logic [63:0]              accepted_read_request_count,
    output logic [63:0]              completed_read_response_count,
    output logic [63:0]              accepted_fragment_request_count,
    output logic [63:0]              accepted_write_data_count,
    output logic [63:0]              completed_fragment_response_count,
    output logic [63:0]              accepted_write_byte_count,
    output logic [63:0]              completed_command_count
);
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_READ_RESPONSE = 4'h2;
    localparam logic [3:0] ERROR_WRITE_RESPONSE = 4'h3;

    typedef enum logic [3:0] {
        IDLE,
        ISSUE_FRAGMENT,
        ISSUE_READ,
        WAIT_READ,
        SEND_DATA,
        WAIT_RESPONSES,
        COMPLETE,
        ERROR_DRAIN
    } state_t;

    state_t state;
    logic [5:0] query_count;
    logic [5:0] head;
    logic [5:0] query_index;
    logic [4:0] beat_index;
    logic [31:0] context_query_stride;
    logic [ADDR_WIDTH-1:0] current_query_address;
    logic [TAG_WIDTH-1:0] operation_tag;
    logic [127:0] held_read_data;
    logic read_outstanding;
    logic fragment_active;
    logic [4:0] outstanding_fragments;
    logic source_release_seen;
    logic abort_pending;
    logic terminal_error;
    logic [3:0] terminal_error_id;

    logic start_configuration_valid;
    logic read_request_fire;
    logic read_response_fire;
    logic fragment_request_fire;
    logic fragment_response_fire;
    logic write_data_fire;
    logic final_beat;
    logic final_query;

    function automatic [TAG_WIDTH-1:0] make_fragment_tag(
        input logic [4:0] tag_head,
        input logic [5:0] tag_query,
        input logic [TAG_WIDTH-1:0] base_tag
    );
        logic [15:0] packed_tag;
        begin
            packed_tag = {tag_head, tag_query, 5'd0};
            make_fragment_tag = base_tag ^ TAG_WIDTH'(packed_tag);
        end
    endfunction

    assign start_configuration_valid = start_query_count != 0 &&
        start_query_count <= 6'd48 && start_head < 6'd32 &&
        start_context_query_stride >= 32'd8192 &&
        start_context_query_stride[7:0] == 8'd0 &&
        start_context_base <= {ADDR_WIDTH{1'b1}} -
            ADDR_WIDTH'(start_context_query_stride) *
            ADDR_WIDTH'(start_query_count);

    assign start_ready = state == IDLE && !abort_request;
    assign busy = state != IDLE;
    assign done_pulse = state == COMPLETE;
    assign error = done_pulse && terminal_error;
    assign error_id = terminal_error_id;

    assign write_request_valid = state == ISSUE_FRAGMENT && !abort_request &&
        !terminal_error && !(write_request_done && write_request_error) &&
        outstanding_fragments < 5'd8;
    assign write_request_address = current_query_address;
    assign write_request_bytes = 32'd256;
    assign write_request_tag = make_fragment_tag(head[4:0], query_index[5:0],
        operation_tag);
    assign fragment_request_fire = write_request_valid && write_request_ready;
    assign fragment_response_fire = write_request_done &&
        outstanding_fragments != 0;

    assign context_read_req_valid = state == ISSUE_READ;
    assign context_read_req_address = {query_index[5:0], beat_index[3:0]};
    assign context_read_req_tag = make_fragment_tag(head[4:0], query_index[5:0],
        operation_tag);
    assign read_request_fire = context_read_req_valid && context_read_req_ready;
    assign context_read_rsp_ready = read_outstanding &&
        (state == WAIT_READ || state == ERROR_DRAIN);
    assign read_response_fire = context_read_rsp_valid && context_read_rsp_ready;

    assign write_data_valid = state == SEND_DATA;
    assign write_data = held_read_data;
    assign write_byte_enable = 16'hffff;
    assign write_data_last = final_beat;
    assign write_data_fire = write_data_valid && write_data_ready;
    assign final_beat = beat_index == 5'd15;
    assign final_query = query_index + 6'd1 >= query_count;

`ifdef SYNTHESIS
    always_comb begin
        accepted_read_request_count = '0;
        completed_read_response_count = '0;
        accepted_fragment_request_count = '0;
        accepted_write_data_count = '0;
        completed_fragment_response_count = '0;
        accepted_write_byte_count = '0;
        completed_command_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            source_buffer_released <= 1'b0;
            query_count <= '0;
            head <= '0;
            query_index <= '0;
            beat_index <= '0;
            context_query_stride <= '0;
            current_query_address <= '0;
            operation_tag <= '0;
            held_read_data <= '0;
            read_outstanding <= 1'b0;
            fragment_active <= 1'b0;
            outstanding_fragments <= '0;
            source_release_seen <= 1'b0;
            abort_pending <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
`ifndef SYNTHESIS
            accepted_read_request_count <= '0;
            completed_read_response_count <= '0;
            accepted_fragment_request_count <= '0;
            accepted_write_data_count <= '0;
            completed_fragment_response_count <= '0;
            accepted_write_byte_count <= '0;
            completed_command_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            source_buffer_released <= 1'b0;

            if (start_valid && start_ready) begin
                query_count <= start_query_count;
                head <= start_head;
                query_index <= 6'd0;
                beat_index <= 5'd0;
                context_query_stride <= start_context_query_stride;
                current_query_address <= start_context_base +
                    (ADDR_WIDTH'(start_head) << 8);
                operation_tag <= start_tag;
                read_outstanding <= 1'b0;
                fragment_active <= 1'b0;
                outstanding_fragments <= '0;
                source_release_seen <= 1'b0;
                abort_pending <= 1'b0;
                terminal_error <= !start_configuration_valid;
                terminal_error_id <= start_configuration_valid ?
                    4'd0 : ERROR_CONFIGURATION;
                state <= start_configuration_valid ? ISSUE_FRAGMENT : COMPLETE;
            end

            if (fragment_request_fire) begin
                fragment_active <= 1'b1;
                beat_index <= 5'd0;
`ifndef SYNTHESIS
                accepted_fragment_request_count <=
                    accepted_fragment_request_count + 64'd1;
`endif
                state <= ISSUE_READ;
            end

            case ({fragment_request_fire, fragment_response_fire})
                2'b10: outstanding_fragments <= outstanding_fragments + 5'd1;
                2'b01: outstanding_fragments <= outstanding_fragments - 5'd1;
                default: outstanding_fragments <= outstanding_fragments;
            endcase

            if (read_request_fire) begin
                read_outstanding <= 1'b1;
`ifndef SYNTHESIS
                accepted_read_request_count <=
                    accepted_read_request_count + 64'd1;
`endif
                state <= WAIT_READ;
            end

            if (read_response_fire) begin
                read_outstanding <= 1'b0;
`ifndef SYNTHESIS
                completed_read_response_count <=
                    completed_read_response_count + 64'd1;
`endif
                if (context_read_rsp_tag !=
                    make_fragment_tag(head[4:0], query_index[5:0],
                        operation_tag)) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_READ_RESPONSE;
                end
                if (state == WAIT_READ) begin
                    held_read_data <= context_read_rsp_data;
                    state <= SEND_DATA;
                end
            end

            if (write_data_fire) begin
`ifndef SYNTHESIS
                accepted_write_data_count <= accepted_write_data_count + 64'd1;
                accepted_write_byte_count <= accepted_write_byte_count + 64'd16;
`endif
                if (final_beat) begin
                    fragment_active <= 1'b0;
                    if (final_query && !source_release_seen) begin
                        source_buffer_released <= 1'b1;
                        source_release_seen <= 1'b1;
                    end
                    if (final_query || abort_pending || abort_request ||
                        terminal_error ||
                        (write_request_done && write_request_error)) begin
                        state <= WAIT_RESPONSES;
                    end else begin
                        query_index <= query_index + 6'd1;
                        current_query_address <= current_query_address +
                            ADDR_WIDTH'(context_query_stride);
                        state <= ISSUE_FRAGMENT;
                    end
                end else begin
                    beat_index <= beat_index + 5'd1;
                    state <= ISSUE_READ;
                end
            end

            if (fragment_response_fire) begin
`ifndef SYNTHESIS
                completed_fragment_response_count <=
                    completed_fragment_response_count + 64'd1;
`endif
                if (write_request_error) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_WRITE_RESPONSE;
                end
            end else if (write_request_done) begin
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_WRITE_RESPONSE;
                if (!fragment_active)
                    state <= ERROR_DRAIN;
            end

            if (state == WAIT_RESPONSES && outstanding_fragments == 0 &&
                !fragment_active && !read_outstanding) begin
                if (abort_pending || abort_request) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else begin
                    state <= COMPLETE;
`ifndef SYNTHESIS
                    if (!terminal_error)
                        completed_command_count <=
                            completed_command_count + 64'd1;
`endif
                end
            end

            if (state == ERROR_DRAIN && !read_outstanding &&
                !fragment_active && outstanding_fragments == 0)
                state <= COMPLETE;

            if (abort_request && state == ISSUE_FRAGMENT &&
                !fragment_active) begin
                if (outstanding_fragments == 0) begin
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else begin
                    abort_pending <= 1'b1;
                    state <= WAIT_RESPONSES;
                end
            end else if (abort_request && state != IDLE &&
                         state != COMPLETE)
                abort_pending <= 1'b1;

            if ((terminal_error || (write_request_done && write_request_error)) &&
                state == ISSUE_FRAGMENT && !fragment_active)
                state <= outstanding_fragments == 0 ? ERROR_DRAIN : WAIT_RESPONSES;

            if (state == COMPLETE)
                state <= IDLE;
        end
    end

    initial begin
        if (ADDR_WIDTH != 64 || TAG_WIDTH < 16)
            $error("attention_context_writer requires 64-bit addresses and TAG_WIDTH >= 16");
    end

`ifndef SYNTHESIS
    logic held_write_data;
    logic [128+16:0] held_write_payload;
    logic [63:0] command_start_byte_count;
    always_ff @(posedge clk) begin
        if (rst) begin
            held_write_data <= 1'b0;
            held_write_payload <= '0;
            command_start_byte_count <= '0;
        end else begin
            if (start_valid && start_ready)
                command_start_byte_count <= accepted_write_byte_count;
            if (held_write_data)
                assert (write_data_valid &&
                    {write_data, write_byte_enable, write_data_last} ==
                    held_write_payload)
                    else $error("attention context writer changed stalled write data");
            held_write_data <= write_data_valid && !write_data_ready;
            held_write_payload <= {write_data, write_byte_enable, write_data_last};
            assert (completed_read_response_count <= accepted_read_request_count)
                else $error("attention context writer completed an unaccepted SRAM read");
            assert (completed_fragment_response_count <=
                    accepted_fragment_request_count)
                else $error("attention context writer completed an unaccepted write fragment");
            if (source_buffer_released)
                assert (accepted_write_byte_count - command_start_byte_count ==
                    64'(query_count) * 64'd256)
                    else $error("attention context writer released its source before all payload bytes");
        end
    end
`endif
endmodule

`default_nettype wire
