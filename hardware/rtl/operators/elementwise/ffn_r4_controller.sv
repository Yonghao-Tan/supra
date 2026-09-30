`default_nettype none

module ffn_r4_controller #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    output logic                         abort_ack,
    input  logic                         arithmetic_abort_ack,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [2:0]                   start_row_count,
    input  logic [14:0]                  start_word_base,
    input  logic                         prefetch_valid,
    input  logic [2:0]                   prefetch_row_count,
    input  logic [14:0]                  prefetch_word_base,
    output logic                         done_valid,
    input  logic                         done_ready,
    output logic                         error,
    output logic [3:0]                   error_id,
    output logic [127:0]                 row_max_abs,

    output logic                         product_req_valid,
    input  logic                         product_req_ready,
    output hardware_types_pkg::elementwise_product_request_t product_req,
    input  logic                         product_rsp_valid,
    output logic                         product_rsp_ready,
    input  hardware_types_pkg::elementwise_product_response_t product_rsp,

    output logic                         arithmetic_req_valid,
    input  logic                         arithmetic_req_ready,
    output logic [2:0]                   arithmetic_req_operation,
    output logic [1023:0]                arithmetic_req_values,
    output logic [1023:0]                arithmetic_req_paired_values,
    output logic [1023:0]                arithmetic_req_factor0_values,
    output logic [1023:0]                arithmetic_req_factor1_values,
    output logic [63:0]                  arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         arithmetic_req_tag,
    input  logic                         arithmetic_rsp_valid,
    output logic                         arithmetic_rsp_ready,
    input  logic [1023:0]                arithmetic_rsp_values,
    input  logic [63:0]                  arithmetic_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         arithmetic_rsp_tag,

    output logic                         max_req_valid,
    input  logic                         max_req_ready,
    output logic [1023:0]                max_req_values,
    output logic [63:0]                  max_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         max_req_tag,
    input  logic                         max_rsp_valid,
    output logic                         max_rsp_ready,
    input  logic [127:0]                 max_rsp_values,
    input  logic [7:0]                   max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0]         max_rsp_tag,

    output logic [31:0]                  accepted_product_read_count,
    output logic [31:0]                  accepted_product_write_count,
    output logic [31:0]                  accepted_add_count,
    output logic [31:0]                  accepted_multiply_count,
    output logic [31:0]                  accepted_max_count
);
`ifdef SYNTHESIS
    always_comb begin
        accepted_product_read_count = '0;
        accepted_product_write_count = '0;
        accepted_add_count = '0;
        accepted_multiply_count = '0;
        accepted_max_count = '0;
    end
`endif
    localparam logic [2:0] BF16_ADD = 3'd0;
    localparam logic [3:0] ERROR_CONFIG = 4'h1;
    localparam logic [3:0] ERROR_ARITHMETIC_RESPONSE = 4'h3;
    localparam logic [3:0] ERROR_MAX_RESPONSE = 4'h4;

    typedef enum logic [3:0] {
        IDLE,
        H1024_START,
        H1024_WAIT,
        H12_START,
        H12_WAIT,
        MAX_SCAN,
        COMPLETE,
        ERROR_HOLD,
        ABORT_DRAIN
    } state_t;
    state_t state;

    logic [2:0] row_count;
    logic [14:0] word_base;
    logic [11:0] max_issue_count;
    logic [11:0] max_response_count;
    logic [127:0] max_values;
    logic [2:0] product_read_outstanding;
    logic [4:0] arithmetic_outstanding;
    logic [7:0] max_outstanding;
    logic terminal_error;
    logic [3:0] terminal_error_id;
    logic [63:0] active_lane_mask;
    logic [7:0] active_max_row_mask;

    logic max_fifo_input_valid;
    logic max_fifo_input_ready;
    logic [527:0] max_fifo_input_data;
    logic max_fifo_output_valid;
    logic max_fifo_output_ready;
    logic [527:0] max_fifo_output_data;
    logic [2:0] max_fifo_occupancy;
    logic [TAG_WIDTH-1:0] max_fifo_tag;
    logic [511:0] max_fifo_data;

    logic h1024_active;
    logic h1024_started;
    logic h1024_start_ready;
    logic h1024_done_valid;
    logic h1024_done_error;
    logic h1024_abort_ack;
    logic h1024_product_req_valid;
    logic h1024_product_req_ready;
    hardware_types_pkg::elementwise_product_request_t h1024_product_req;
    logic h1024_product_rsp_ready;
    logic h1024_arithmetic_req_valid;
    logic h1024_arithmetic_req_ready;
    logic [1023:0] h1024_arithmetic_req_values;
    logic [1023:0] h1024_arithmetic_req_paired_values;
    logic [63:0] h1024_arithmetic_req_lane_mask;
    logic [TAG_WIDTH-1:0] h1024_arithmetic_req_tag;
    logic h1024_arithmetic_rsp_ready;
    logic h1024_prefetch_pending;
    logic h1024_is_prefetch;
    logic h1024_prefetched_valid;
    logic [2:0] h1024_prefetch_row_count;
    logic [14:0] h1024_prefetch_word_base;
    logic [2:0] h1024_pending_row_count;
    logic [14:0] h1024_pending_word_base;
    logic h1024_prefetch_launch_valid;
    logic h1024_child_start_valid;
    logic [2:0] h1024_child_start_row_count;
    logic [14:0] h1024_child_start_word_base;
    logic h1024_prefetch_matches_start;

    logic h12_active;
    logic h12_started;
    logic h12_start_ready;
    logic h12_done_valid;
    logic h12_done_error;
    logic h12_abort_ack;
    logic h12_product_req_valid;
    logic h12_product_req_ready;
    hardware_types_pkg::elementwise_product_request_t h12_product_req;
    logic h12_product_rsp_ready;
    logic h12_arithmetic_req_valid;
    logic h12_arithmetic_req_ready;
    logic [2:0] h12_arithmetic_req_operation;
    logic [1023:0] h12_arithmetic_req_values;
    logic [1023:0] h12_arithmetic_req_paired_values;
    logic [1023:0] h12_arithmetic_req_factor0_values;
    logic [63:0] h12_arithmetic_req_lane_mask;
    logic [TAG_WIDTH-1:0] h12_arithmetic_req_tag;
    logic h12_arithmetic_rsp_ready;
    logic h12_max_req_valid;
    logic h12_max_req_ready;
    logic [1023:0] h12_max_req_values;
    logic [63:0] h12_max_req_lane_mask;
    logic [TAG_WIDTH-1:0] h12_max_req_tag;
    logic h12_done_ready;

    always_comb begin
        active_lane_mask = '0;
        active_max_row_mask = '0;
        for (integer row = 0; row < 4; row++) begin
            active_lane_mask[row*16 +: 16] = {16{row < row_count}};
            active_max_row_mask[row] = row < row_count;
        end
    end

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE || state == ERROR_HOLD;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign row_max_abs = max_values;
    assign h1024_prefetch_matches_start =
        h1024_prefetch_row_count == start_row_count &&
        h1024_prefetch_word_base == start_word_base;
    assign h1024_prefetch_launch_valid = h1024_prefetch_pending &&
        !h1024_active &&
        (state == H12_START || state == H12_WAIT || state == MAX_SCAN);
    assign h1024_child_start_valid = state == H1024_START ||
        h1024_prefetch_launch_valid;
    assign h1024_child_start_row_count = h1024_prefetch_launch_valid ?
        h1024_pending_row_count : row_count;
    assign h1024_child_start_word_base = h1024_prefetch_launch_valid ?
        h1024_pending_word_base : word_base;

    always_comb begin
        product_req_valid = 1'b0;
        product_req = '0;
        product_rsp_ready = state == ABORT_DRAIN;
        h1024_product_req_ready = 1'b0;
        h12_product_req_ready = 1'b0;
        if (h12_active && h12_product_req_valid) begin
            product_req_valid = h12_product_req_valid;
            product_req = h12_product_req;
            product_req.tag[15:14] = 2'b10;
            h12_product_req_ready = product_req_ready;
        end else if (state == MAX_SCAN) begin
            product_req_valid = !abort_request && max_issue_count < 12'd1536 &&
                ({1'b0, max_fifo_occupancy} +
                 {1'b0, product_read_outstanding} < 4'd3);
            product_req.write = 1'b0;
            product_req.word_index = 15'(integer'(word_base) +
                integer'(max_issue_count) * 8);
            product_req.data = '0;
            product_req.half = 1'b0;
            product_req.tag = TAG_WIDTH'(max_issue_count);
        end else if (h1024_active && h1024_product_req_valid) begin
            product_req_valid = h1024_product_req_valid;
            product_req = h1024_product_req;
            product_req.tag[15:14] = 2'b01;
            h1024_product_req_ready = product_req_ready;
        end

        if (state != ABORT_DRAIN && product_rsp_valid) begin
            case (product_rsp.tag[15:14])
                2'b01: product_rsp_ready = h1024_product_rsp_ready;
                2'b10: product_rsp_ready = h12_product_rsp_ready;
                default: product_rsp_ready = max_fifo_input_ready;
            endcase
        end

        arithmetic_req_valid = 1'b0;
        arithmetic_req_operation = BF16_ADD;
        arithmetic_req_values = '0;
        arithmetic_req_paired_values = '0;
        arithmetic_req_factor0_values = '0;
        arithmetic_req_factor1_values = '0;
        arithmetic_req_lane_mask = active_lane_mask;
        arithmetic_req_tag = '0;
        arithmetic_rsp_ready = state == ABORT_DRAIN;
        h1024_arithmetic_req_ready = 1'b0;
        h12_arithmetic_req_ready = 1'b0;
        if (h12_active && h12_arithmetic_req_valid) begin
            arithmetic_req_valid = h12_arithmetic_req_valid;
            arithmetic_req_operation = h12_arithmetic_req_operation;
            arithmetic_req_values = h12_arithmetic_req_values;
            arithmetic_req_paired_values = h12_arithmetic_req_paired_values;
            arithmetic_req_factor0_values = h12_arithmetic_req_factor0_values;
            arithmetic_req_lane_mask = h12_arithmetic_req_lane_mask;
            arithmetic_req_tag = h12_arithmetic_req_tag;
            arithmetic_req_tag[13] = 1'b1;
            h12_arithmetic_req_ready = arithmetic_req_ready;
        end else if (h1024_active && h1024_arithmetic_req_valid) begin
            arithmetic_req_valid = h1024_arithmetic_req_valid;
            arithmetic_req_values = h1024_arithmetic_req_values;
            arithmetic_req_paired_values =
                h1024_arithmetic_req_paired_values;
            arithmetic_req_lane_mask = h1024_arithmetic_req_lane_mask;
            arithmetic_req_tag = h1024_arithmetic_req_tag;
            arithmetic_req_tag[13] = 1'b0;
            h1024_arithmetic_req_ready = arithmetic_req_ready;
        end

        if (state != ABORT_DRAIN && arithmetic_rsp_valid) begin
            if (arithmetic_rsp_tag[13])
                arithmetic_rsp_ready = h12_arithmetic_rsp_ready;
            else
                arithmetic_rsp_ready = h1024_arithmetic_rsp_ready;
        end
    end

    assign max_fifo_input_valid = state == MAX_SCAN && product_rsp_valid &&
        product_rsp.tag[15:14] == 2'b00;
    assign max_fifo_input_data = {product_rsp.tag, product_rsp.data};
    assign {max_fifo_tag, max_fifo_data} = max_fifo_output_data;
    assign max_fifo_output_ready = state == MAX_SCAN && max_req_ready &&
        !abort_request;
    assign max_req_valid = !abort_request &&
        ((h12_active && h12_max_req_valid) ||
         (state == MAX_SCAN && max_fifo_output_valid));
    assign h12_max_req_ready = h12_active && max_req_ready;
    assign max_req_values = h12_active ? h12_max_req_values :
        {512'd0, max_fifo_data};
    assign max_req_lane_mask = h12_active ? h12_max_req_lane_mask :
        {32'd0, {8{active_max_row_mask[3]}},
         {8{active_max_row_mask[2]}}, {8{active_max_row_mask[1]}},
         {8{active_max_row_mask[0]}}};
    assign max_req_tag = h12_active ? h12_max_req_tag : max_fifo_tag;
    assign max_rsp_ready = state == H12_WAIT || state == MAX_SCAN ||
        state == ABORT_DRAIN;
    assign h12_done_ready = max_outstanding == 0 &&
        (h12_done_error ||
         (max_issue_count == 12'd1536 && max_response_count == 12'd1536));

    ready_valid_fifo #(.DATA_WIDTH(528), .DEPTH(3)) max_input_fifo (
        .clk, .rst(rst || state != MAX_SCAN),
        .input_valid(max_fifo_input_valid), .input_ready(max_fifo_input_ready),
        .input_data(max_fifo_input_data), .output_valid(max_fifo_output_valid),
        .output_ready(max_fifo_output_ready), .output_data(max_fifo_output_data),
        .occupancy(max_fifo_occupancy)
    );

    ffn_r4_h1024_pipeline #(.TAG_WIDTH(TAG_WIDTH)) h1024_pipeline (
        .clk, .rst, .abort_request(abort_request || state == ABORT_DRAIN),
        .abort_ack(h1024_abort_ack),
        .arithmetic_abort_ack,
        .start_valid(h1024_child_start_valid),
        .start_ready(h1024_start_ready),
        .start_row_count(h1024_child_start_row_count),
        .start_word_base(h1024_child_start_word_base),
        .done_valid(h1024_done_valid), .done_ready(1'b1),
        .error(h1024_done_error),
        .product_req_valid(h1024_product_req_valid),
        .product_req_ready(h1024_product_req_ready),
        .product_req(h1024_product_req),
        .product_rsp_valid(product_rsp_valid && h1024_active &&
            product_rsp.tag[15:14] == 2'b01),
        .product_rsp_ready(h1024_product_rsp_ready), .product_rsp,
        .arithmetic_req_valid(h1024_arithmetic_req_valid),
        .arithmetic_req_ready(h1024_arithmetic_req_ready),
        .arithmetic_req_values(h1024_arithmetic_req_values),
        .arithmetic_req_paired_values(h1024_arithmetic_req_paired_values),
        .arithmetic_req_lane_mask(h1024_arithmetic_req_lane_mask),
        .arithmetic_req_tag(h1024_arithmetic_req_tag),
        .arithmetic_rsp_valid(arithmetic_rsp_valid && h1024_active &&
            !arithmetic_rsp_tag[13]),
        .arithmetic_rsp_ready(h1024_arithmetic_rsp_ready),
        .arithmetic_rsp_values, .arithmetic_rsp_lane_mask, .arithmetic_rsp_tag
    );

    ffn_r4_h12_pipeline #(.TAG_WIDTH(TAG_WIDTH)) h12_pipeline (
        .clk, .rst, .abort_request(abort_request || state == ABORT_DRAIN),
        .abort_ack(h12_abort_ack),
        .arithmetic_abort_ack,
        .start_valid(state == H12_START), .start_ready(h12_start_ready),
        .start_row_count(row_count), .start_word_base(word_base),
        .done_valid(h12_done_valid), .done_ready(h12_done_ready),
        .error(h12_done_error),
        .product_req_valid(h12_product_req_valid),
        .product_req_ready(h12_product_req_ready), .product_req(h12_product_req),
        .product_rsp_valid(product_rsp_valid && h12_active &&
            product_rsp.tag[15:14] == 2'b10),
        .product_rsp_ready(h12_product_rsp_ready), .product_rsp,
        .arithmetic_req_valid(h12_arithmetic_req_valid),
        .arithmetic_req_ready(h12_arithmetic_req_ready),
        .arithmetic_req_operation(h12_arithmetic_req_operation),
        .arithmetic_req_values(h12_arithmetic_req_values),
        .arithmetic_req_paired_values(h12_arithmetic_req_paired_values),
        .arithmetic_req_factor0_values(h12_arithmetic_req_factor0_values),
        .arithmetic_req_lane_mask(h12_arithmetic_req_lane_mask),
        .arithmetic_req_tag(h12_arithmetic_req_tag),
        .arithmetic_rsp_valid(arithmetic_rsp_valid && h12_active &&
            arithmetic_rsp_tag[13]),
        .arithmetic_rsp_ready(h12_arithmetic_rsp_ready),
        .arithmetic_rsp_values, .arithmetic_rsp_lane_mask, .arithmetic_rsp_tag,
        .max_req_valid(h12_max_req_valid),
        .max_req_ready(h12_max_req_ready),
        .max_req_values(h12_max_req_values),
        .max_req_lane_mask(h12_max_req_lane_mask),
        .max_req_tag(h12_max_req_tag)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            row_count <= '0;
            word_base <= '0;
            max_issue_count <= '0;
            max_response_count <= '0;
            max_values <= '0;
            product_read_outstanding <= '0;
            arithmetic_outstanding <= '0;
            max_outstanding <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            h1024_active <= 1'b0;
            h1024_started <= 1'b0;
            h1024_prefetch_pending <= 1'b0;
            h1024_is_prefetch <= 1'b0;
            h1024_prefetched_valid <= 1'b0;
            h1024_prefetch_row_count <= '0;
            h1024_prefetch_word_base <= '0;
            h1024_pending_row_count <= '0;
            h1024_pending_word_base <= '0;
            h12_active <= 1'b0;
            h12_started <= 1'b0;
`ifndef SYNTHESIS
            accepted_product_read_count <= '0;
            accepted_product_write_count <= '0;
            accepted_add_count <= '0;
            accepted_multiply_count <= '0;
            accepted_max_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;

            case ({product_req_valid && product_req_ready && !product_req.write,
                   product_rsp_valid && product_rsp_ready})
                2'b10: product_read_outstanding <=
                    product_read_outstanding + 3'd1;
                2'b01: product_read_outstanding <=
                    product_read_outstanding - 3'd1;
                default: begin end
            endcase
            case ({arithmetic_req_valid && arithmetic_req_ready,
                   arithmetic_rsp_valid && arithmetic_rsp_ready})
                2'b10: arithmetic_outstanding <=
                    arithmetic_outstanding + 5'd1;
                2'b01: arithmetic_outstanding <=
                    arithmetic_outstanding - 5'd1;
                default: begin end
            endcase
            if (arithmetic_abort_ack)
                arithmetic_outstanding <= '0;
            case ({max_req_valid && max_req_ready,
                   max_rsp_valid && max_rsp_ready})
                2'b10: max_outstanding <= max_outstanding + 8'd1;
                2'b01: max_outstanding <= max_outstanding - 8'd1;
                default: begin end
            endcase

`ifndef SYNTHESIS
            if (product_req_valid && product_req_ready) begin
                if (product_req.write)
                    accepted_product_write_count <=
                        accepted_product_write_count + 32'd1;
                else
                    accepted_product_read_count <=
                        accepted_product_read_count + 32'd1;
            end
            if (arithmetic_req_valid && arithmetic_req_ready) begin
                if (arithmetic_req_operation == BF16_ADD)
                    accepted_add_count <= accepted_add_count + 32'd1;
                else
                    accepted_multiply_count <=
                        accepted_multiply_count + 32'd1;
            end
            if (max_req_valid && max_req_ready)
                accepted_max_count <= accepted_max_count + 32'd1;
`endif

            if (state == H12_WAIT && h12_max_req_valid &&
                h12_max_req_ready)
                max_issue_count <= max_issue_count + 12'd1;
            if (state == H12_WAIT && max_rsp_valid && max_rsp_ready) begin
                max_response_count <= max_response_count + 12'd1;
                if ((max_rsp_row_mask & 8'h0f) !=
                    {4'd0, active_max_row_mask[3:0]}) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_MAX_RESPONSE;
                end else begin
                    for (integer row = 0; row < 4; row++)
                        if (row < row_count &&
                            max_rsp_values[row*16 +: 15] >
                                max_values[row*16 +: 15])
                            max_values[row*16 +: 16] <=
                                {1'b0, max_rsp_values[row*16 +: 15]};
                end
            end

            if (h1024_prefetch_launch_valid && h1024_start_ready) begin
                h1024_active <= 1'b1;
                h1024_started <= 1'b1;
                h1024_is_prefetch <= 1'b1;
                h1024_prefetch_pending <= 1'b0;
                h1024_prefetch_row_count <= h1024_pending_row_count;
                h1024_prefetch_word_base <= h1024_pending_word_base;
            end

            if (h1024_done_valid && h1024_is_prefetch &&
                !(state == IDLE && start_valid && start_ready &&
                  h1024_prefetch_matches_start)) begin
                h1024_active <= 1'b0;
                h1024_started <= 1'b0;
                h1024_is_prefetch <= 1'b0;
                h1024_prefetched_valid <= !h1024_done_error;
            end

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    row_count <= start_row_count;
                    word_base <= start_word_base;
                    max_issue_count <= '0;
                    max_response_count <= '0;
                    max_values <= '0;
                    terminal_error <= start_row_count == 0 ||
                        start_row_count > 3'd4 ||
                        integer'(start_word_base) + 1535 * 8 + 3 >= 30720;
                    terminal_error_id <= start_row_count == 0 ||
                        start_row_count > 3'd4 ||
                        integer'(start_word_base) + 1535 * 8 + 3 >= 30720 ?
                            ERROR_CONFIG : 4'd0;
                    h1024_prefetch_pending <= prefetch_valid;
                    h1024_pending_row_count <= prefetch_row_count;
                    h1024_pending_word_base <= prefetch_word_base;
                    h12_active <= 1'b0;
                    h12_started <= 1'b0;
                    if (start_row_count == 0 || start_row_count > 3'd4 ||
                        integer'(start_word_base) + 1535 * 8 + 3 >= 30720 ||
                        (prefetch_valid &&
                         (prefetch_row_count == 0 ||
                          prefetch_row_count > 3'd4 ||
                          integer'(prefetch_word_base) + 1535 * 8 + 3 >=
                              30720))) begin
                        h1024_active <= 1'b0;
                        h1024_started <= 1'b0;
                        h1024_is_prefetch <= 1'b0;
                        h1024_prefetched_valid <= 1'b0;
                        h1024_prefetch_pending <= 1'b0;
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_CONFIG;
                        state <= ERROR_HOLD;
                    end else if (h1024_prefetched_valid &&
                                 h1024_prefetch_matches_start) begin
                        h1024_prefetched_valid <= 1'b0;
                        h12_active <= 1'b1;
                        state <= H12_START;
                    end else if (h1024_active && h1024_is_prefetch &&
                                 h1024_prefetch_matches_start) begin
                        h1024_is_prefetch <= 1'b0;
                        if (h1024_done_valid) begin
                            h1024_active <= 1'b0;
                            h1024_started <= 1'b0;
                            if (h1024_done_error) begin
                                terminal_error <= 1'b1;
                                terminal_error_id <= ERROR_ARITHMETIC_RESPONSE;
                                state <= ERROR_HOLD;
                            end else begin
                                h12_active <= 1'b1;
                                state <= H12_START;
                            end
                        end else begin
                            state <= H1024_WAIT;
                        end
                    end else if (h1024_active || h1024_prefetched_valid) begin
                        h1024_prefetch_pending <= 1'b0;
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_CONFIG;
                        state <= ABORT_DRAIN;
                    end else begin
                        h1024_active <= 1'b1;
                        h1024_started <= 1'b0;
                        h1024_is_prefetch <= 1'b0;
                        state <= H1024_START;
                    end
                end
                H1024_START: if (h1024_start_ready) begin
                    h1024_started <= 1'b1;
                    state <= H1024_WAIT;
                end
                H1024_WAIT: if (h1024_done_valid) begin
                    h1024_active <= 1'b0;
                    h1024_started <= 1'b0;
                    h1024_is_prefetch <= 1'b0;
                    if (h1024_done_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_ARITHMETIC_RESPONSE;
                        state <= ERROR_HOLD;
                    end else begin
                        h12_active <= 1'b1;
                        h12_started <= 1'b0;
                        state <= H12_START;
                    end
                end
                H12_START: if (h12_start_ready) begin
                    h12_started <= 1'b1;
                    state <= H12_WAIT;
                end
                H12_WAIT: if (h12_done_valid && h12_done_ready) begin
                    h12_active <= 1'b0;
                    h12_started <= 1'b0;
                    if (h12_done_error || terminal_error) begin
                        terminal_error <= 1'b1;
                        if (h12_done_error)
                            terminal_error_id <= ERROR_ARITHMETIC_RESPONSE;
                        state <= ERROR_HOLD;
                    end else begin
                        state <= COMPLETE;
                    end
                end
                MAX_SCAN: begin
                    if (product_req_valid && product_req_ready &&
                        product_req.tag[15:14] == 2'b00)
                        max_issue_count <= max_issue_count + 12'd1;
                    if (max_rsp_valid && max_rsp_ready) begin
                        max_response_count <= max_response_count + 12'd1;
                        if ((max_rsp_row_mask & 8'h0f) !=
                            {4'd0, active_max_row_mask[3:0]}) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_MAX_RESPONSE;
                            state <= ERROR_HOLD;
                        end else begin
                            for (integer row = 0; row < 4; row++)
                                if (row < row_count &&
                                    max_rsp_values[row*16 +: 15] >
                                        max_values[row*16 +: 15])
                                    max_values[row*16 +: 16] <= {
                                        1'b0, max_rsp_values[row*16 +: 15]};
                        end
                    end
                    if (max_issue_count == 12'd1536 &&
                        product_read_outstanding == 0 &&
                        max_fifo_occupancy == 0 && max_outstanding == 0 &&
                        max_response_count == 12'd1536)
                        state <= COMPLETE;
                end
                COMPLETE, ERROR_HOLD: if (done_valid && done_ready)
                    state <= IDLE;
                ABORT_DRAIN: begin
                    if (h1024_active && !h1024_started)
                        h1024_active <= 1'b0;
                    if (h12_active && !h12_started)
                        h12_active <= 1'b0;
                    if (h1024_abort_ack) begin
                        h1024_active <= 1'b0;
                        h1024_started <= 1'b0;
                    end
                    if (h12_abort_ack) begin
                        h12_active <= 1'b0;
                        h12_started <= 1'b0;
                    end
                    if (product_read_outstanding == 0 &&
                        arithmetic_outstanding == 0 && max_outstanding == 0 &&
                        !h1024_active && !h12_active) begin
                        h1024_prefetch_pending <= 1'b0;
                        h1024_prefetched_valid <= 1'b0;
                        h1024_is_prefetch <= 1'b0;
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase

            if (abort_request && state != IDLE && state != ABORT_DRAIN &&
                state != COMPLETE && state != ERROR_HOLD) begin
                h1024_prefetch_pending <= 1'b0;
                h1024_prefetched_valid <= 1'b0;
                state <= ABORT_DRAIN;
            end
            if (h1024_done_valid && h1024_is_prefetch &&
                h1024_done_error) begin
                terminal_error <= 1'b1;
                terminal_error_id <= ERROR_ARITHMETIC_RESPONSE;
                h1024_prefetch_pending <= 1'b0;
                h1024_prefetched_valid <= 1'b0;
                state <= ABORT_DRAIN;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (TAG_WIDTH < 16)
            $error("FFN R4 controller requires TAG_WIDTH >= 16");
    end

    always_ff @(posedge clk) begin
        if (!rst && product_rsp_valid)
            assert (product_rsp_ready)
                else $error("FFN R4 product SRAM response met a stalled consumer");
        if (!rst && arithmetic_rsp_valid)
            assert (arithmetic_rsp_ready)
                else $error("FFN R4 BF16 response met a stalled consumer: controller_state=%0d tag=0x%0h h1024_active=%0b h1024_ready=%0b h12_active=%0b h12_ready=%0b controller_outstanding=%0d h1024_state=%0d h1024_outstanding=%0d h12_state=%0d h12_outstanding=%0d",
                    state, arithmetic_rsp_tag, h1024_active,
                    h1024_arithmetic_rsp_ready, h12_active,
                    h12_arithmetic_rsp_ready, arithmetic_outstanding,
                    h1024_pipeline.state,
                    h1024_pipeline.arithmetic_outstanding,
                    h12_pipeline.state, h12_pipeline.arithmetic_outstanding);
        if (!rst && max_rsp_valid)
            assert (max_rsp_ready)
                else $error("FFN R4 max response met a stalled consumer");
    end
`endif
endmodule

`default_nettype wire
