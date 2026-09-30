`default_nettype none

module lm_head_raw_w8_panel_loader #(
    parameter integer PANEL_COUNT = 15808,
    parameter integer PANEL_BYTES = 32768,
    parameter integer SCALE_BLOCK_PANELS = 128
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [13:0]   start_panel_count,
    input  logic [13:0]   start_panel_index,
    input  logic          start_panel_select,
    input  logic [63:0]   start_weight_base,
    input  logic [63:0]   start_weight_limit,
    input  logic [63:0]   start_scale_base,
    input  logic [63:0]   start_scale_limit,

    output logic          dma_request_valid,
    input  logic          dma_request_ready,
    output logic [63:0]   dma_request_address,
    output logic [31:0]   dma_request_bytes,
    output logic [7:0]    dma_request_tag,
    input  logic          dma_response_valid,
    output logic          dma_response_ready,
    input  logic [127:0]  dma_response_data,
    input  logic [15:0]   dma_response_byte_enable,
    input  logic          dma_response_last,
    input  logic          dma_done_pulse,
    input  logic          dma_error,

    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [6:0]    scale_write_address,
    output logic [127:0]  scale_write_data,
    output logic          scale_read_valid,
    input  logic          scale_read_ready,
    output logic [6:0]    scale_read_address,
    input  logic          scale_read_rsp_valid,
    input  logic [127:0]  scale_read_rsp_data,

    output logic          panel_write_valid,
    input  logic          panel_write_ready,
    output logic          panel_write_panel,
    output logic [1:0]    panel_write_stripe,
    output logic [8:0]    panel_write_row,
    output logic [127:0]  panel_write_data,
    output logic [15:0]   panel_write_byte_enable,
    output logic [127:0]  panel_scales_bf16,

    output logic          done_valid,
    input  logic          done_ready,
    output logic          error,
    output logic [3:0]    error_id,
    output logic [63:0]   accepted_weight_bytes,
    output logic [63:0]   accepted_scale_bytes
);
    localparam logic [3:0] ERROR_START = 4'h1;
    localparam logic [3:0] ERROR_SCALE_DMA = 4'h2;
    localparam logic [3:0] ERROR_SCALE_VALUE = 4'h3;
    localparam logic [3:0] ERROR_WEIGHT_DMA = 4'h4;
    localparam logic [3:0] ERROR_WEIGHT_VALUE = 4'h5;
    localparam integer WEIGHT_BEATS = PANEL_BYTES / 16;

    typedef enum logic [3:0] {
        IDLE,
        SCALE_REQUEST,
        SCALE_STREAM,
        SCALE_READ_REQUEST,
        SCALE_READ_WAIT,
        WEIGHT_REQUEST,
        WEIGHT_STREAM,
        COMPLETE
    } state_t;

    state_t state;
    logic [13:0] panel_count;
    logic [13:0] panel_index;
    logic panel_select;
    logic [63:0] weight_base;
    logic [63:0] scale_base;
    logic scale_cache_valid;
    logic [6:0] scale_cache_block;
    logic [127:0] panel_scales;
    logic [7:0] scale_beat;
    logic [11:0] weight_beat;
    logic [7:0] current_scale_panel_count;
    logic dma_data_done;
    logic dma_completion_seen;
    logic terminal_error;
    logic [3:0] terminal_error_id;
    logic scale_value_error;
    logic scale_response_invalid;
    logic weight_value_error;
    logic weight_response_invalid;
    logic response_fire;
    logic response_last_fire;
    logic transaction_complete;
    logic scale_block_hit;
    logic [6:0] requested_scale_block;
    logic [13:0] scale_block_panel_base;
    logic [14:0] remaining_scale_panels;
    logic abort_pending;

    function automatic logic bf16_positive_finite(input logic [15:0] value);
        bf16_positive_finite = !value[15] && value[14:7] != 8'hff &&
            value[14:0] != 15'd0;
    endfunction

    assign start_ready = state == IDLE && !abort_request && !abort_pending;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign requested_scale_block = start_panel_index[13:7];
    assign scale_block_hit = scale_cache_valid &&
        scale_cache_block == requested_scale_block && start_panel_index != 0;

    assign scale_block_panel_base = {panel_index[13:7], 7'b0};
    assign remaining_scale_panels = {1'b0, panel_count} -
        {1'b0, scale_block_panel_base};
    assign current_scale_panel_count = remaining_scale_panels >=
        15'(SCALE_BLOCK_PANELS) ? 8'(SCALE_BLOCK_PANELS) :
        8'(remaining_scale_panels);

    assign dma_request_valid = (state == SCALE_REQUEST ||
        state == WEIGHT_REQUEST) && !abort_request && !abort_pending;
    assign dma_request_address = state == SCALE_REQUEST ?
        scale_base + 64'(scale_block_panel_base) * 64'd16 :
        weight_base + 64'(panel_index) * PANEL_BYTES;
    assign dma_request_bytes = state == SCALE_REQUEST ?
        32'(current_scale_panel_count) * 32'd16 : PANEL_BYTES;
    assign dma_request_tag = state == SCALE_REQUEST ? 8'hc0 : 8'hc1;
    assign dma_response_ready = state == SCALE_STREAM ?
        ((abort_request || abort_pending || terminal_error ||
          scale_response_invalid) ?
            1'b1 : scale_write_ready) :
        state == WEIGHT_STREAM ?
            ((abort_request || abort_pending) ? 1'b1 : panel_write_ready) :
        1'b0;
    assign response_fire = dma_response_valid && dma_response_ready;
    assign response_last_fire = response_fire && dma_response_last;
    assign transaction_complete = dma_completion_seen || dma_done_pulse ?
        (terminal_error || dma_error || dma_data_done || response_last_fire) :
        1'b0;
    assign scale_write_valid = state == SCALE_STREAM && dma_response_valid &&
        !abort_request && !abort_pending && !terminal_error &&
        !scale_response_invalid;
    assign scale_write_address = scale_beat[6:0];
    assign scale_write_data = dma_response_data;
    assign scale_read_valid = state == SCALE_READ_REQUEST &&
        !abort_request && !abort_pending;
    assign scale_read_address = panel_index[6:0];

    assign panel_write_valid = state == WEIGHT_STREAM &&
        dma_response_valid && !terminal_error && !weight_response_invalid &&
        !abort_request && !abort_pending;
    assign panel_write_panel = panel_select;
    assign panel_write_stripe = weight_beat[1:0];
    assign panel_write_row = weight_beat[10:2];
    assign panel_write_data = dma_response_data;
    assign panel_write_byte_enable = dma_response_byte_enable;
    assign panel_scales_bf16 = panel_scales;

    always_comb begin
        scale_response_invalid = 1'b0;
        for (integer lane = 0; lane < 8; lane++) begin
            if (!bf16_positive_finite(dma_response_data[lane*16 +: 16]))
                scale_response_invalid = 1'b1;
        end
        weight_response_invalid = 1'b0;
        for (integer byte_lane = 0; byte_lane < 16; byte_lane++) begin
            if (dma_response_byte_enable[byte_lane] &&
                dma_response_data[byte_lane*8 +: 8] == 8'h80)
                weight_response_invalid = 1'b1;
        end
    end

`ifdef SYNTHESIS
    always_comb begin
        accepted_weight_bytes = '0;
        accepted_scale_bytes = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            panel_count <= '0;
            panel_index <= '0;
            panel_select <= 1'b0;
            weight_base <= '0;
            scale_base <= '0;
            scale_cache_valid <= 1'b0;
            scale_cache_block <= '0;
            panel_scales <= '0;
            scale_beat <= '0;
            weight_beat <= '0;
            dma_data_done <= 1'b0;
            dma_completion_seen <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            scale_value_error <= 1'b0;
            weight_value_error <= 1'b0;
`ifndef SYNTHESIS
            accepted_weight_bytes <= '0;
            accepted_scale_bytes <= '0;
`endif
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (abort_request)
                abort_pending <= 1'b1;
            case (state)
                IDLE: if (abort_request || abort_pending) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                end else if (start_valid && start_ready) begin
                    panel_count <= start_panel_count;
                    panel_index <= start_panel_index;
                    panel_select <= start_panel_select;
                    weight_base <= start_weight_base;
                    scale_base <= start_scale_base;
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    if (start_panel_count == 0 ||
                        start_panel_count > 14'(PANEL_COUNT) ||
                        start_panel_index >= start_panel_count ||
                        start_weight_base[3:0] != 0 ||
                        start_scale_base[3:0] != 0 ||
                        start_weight_limit < start_weight_base ||
                        start_weight_limit - start_weight_base <
                            64'(start_panel_count) * PANEL_BYTES ||
                        start_scale_limit < start_scale_base ||
                        start_scale_limit - start_scale_base <
                            64'(start_panel_count) * 64'd16) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_START;
                        state <= COMPLETE;
                    end else begin
                        state <= scale_block_hit ?
                            SCALE_READ_REQUEST : SCALE_REQUEST;
                    end
                end

                SCALE_REQUEST: if (abort_request || abort_pending) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else if (dma_request_valid && dma_request_ready) begin
                    scale_beat <= '0;
                    dma_data_done <= 1'b0;
                    dma_completion_seen <= 1'b0;
                    scale_value_error <= 1'b0;
                    state <= SCALE_STREAM;
                end

                SCALE_STREAM: begin
                    if (dma_done_pulse) begin
                        dma_completion_seen <= 1'b1;
                        if (dma_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SCALE_DMA;
                        end
                    end
                    if (response_fire) begin
`ifndef SYNTHESIS
                        accepted_scale_bytes <= accepted_scale_bytes + 64'd16;
`endif
                        if (scale_response_invalid)
                            scale_value_error <= 1'b1;
                        if (dma_response_byte_enable != 16'hffff ||
                            dma_response_last !=
                                (scale_beat + 8'd1 ==
                                 current_scale_panel_count)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SCALE_DMA;
                        end
                        scale_beat <= scale_beat + 8'd1;
                        if (dma_response_last)
                            dma_data_done <= 1'b1;
                    end
                    if (transaction_complete) begin
                        if (abort_request || abort_pending) begin
                            abort_pending <= 1'b0;
                            abort_ack <= 1'b1;
                            state <= IDLE;
                        end else if (terminal_error || dma_error || scale_value_error ||
                            (response_fire && scale_response_invalid)) begin
                            terminal_error <= 1'b1;
                            if (scale_value_error ||
                                (response_fire && scale_response_invalid))
                                terminal_error_id <= ERROR_SCALE_VALUE;
                            else if (!terminal_error)
                                terminal_error_id <= ERROR_SCALE_DMA;
                            state <= COMPLETE;
                        end else begin
                            scale_cache_valid <= 1'b1;
                            scale_cache_block <= panel_index[13:7];
                            state <= SCALE_READ_REQUEST;
                        end
                    end
                end

                SCALE_READ_REQUEST: if (abort_request || abort_pending) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else if (scale_read_valid && scale_read_ready) begin
                    state <= SCALE_READ_WAIT;
                end

                SCALE_READ_WAIT: if (scale_read_rsp_valid) begin
                    if (abort_request || abort_pending) begin
                        abort_pending <= 1'b0;
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end else begin
                        panel_scales <= scale_read_rsp_data;
                        state <= WEIGHT_REQUEST;
                    end
                end

                WEIGHT_REQUEST: if (abort_request || abort_pending) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else if (dma_request_valid && dma_request_ready) begin
                    weight_beat <= '0;
                    dma_data_done <= 1'b0;
                    dma_completion_seen <= 1'b0;
                    weight_value_error <= 1'b0;
                    state <= WEIGHT_STREAM;
                end

                WEIGHT_STREAM: begin
                    if (dma_done_pulse) begin
                        dma_completion_seen <= 1'b1;
                        if (dma_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_WEIGHT_DMA;
                        end
                    end
                    if (response_fire) begin
`ifndef SYNTHESIS
                        accepted_weight_bytes <= accepted_weight_bytes + 64'd16;
`endif
                        if (weight_response_invalid)
                            weight_value_error <= 1'b1;
                        if (dma_response_byte_enable != 16'hffff ||
                            dma_response_last !=
                                (weight_beat == 12'(WEIGHT_BEATS-1))) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_WEIGHT_DMA;
                        end
                        weight_beat <= weight_beat + 12'd1;
                        if (dma_response_last)
                            dma_data_done <= 1'b1;
                    end
                    if (transaction_complete) begin
                        if (abort_request || abort_pending) begin
                            abort_pending <= 1'b0;
                            abort_ack <= 1'b1;
                            state <= IDLE;
                        end else if (terminal_error || dma_error || weight_value_error ||
                            (response_fire && weight_response_invalid)) begin
                            terminal_error <= 1'b1;
                            if (weight_value_error ||
                                (response_fire && weight_response_invalid))
                                terminal_error_id <= ERROR_WEIGHT_VALUE;
                            else if (!terminal_error)
                                terminal_error_id <= ERROR_WEIGHT_DMA;
                        end
                        state <= COMPLETE;
                    end
                end

                COMPLETE: if (abort_request || abort_pending) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end else if (done_valid && done_ready) begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (PANEL_COUNT != 15808 || PANEL_BYTES != 32768 ||
            SCALE_BLOCK_PANELS != 128)
            $error("LM-head panel loader parameter configuration is invalid");
    end
endmodule

`default_nettype wire
