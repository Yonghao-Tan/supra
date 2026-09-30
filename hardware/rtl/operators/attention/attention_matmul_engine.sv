`default_nettype none

// One tile scheduler for both QK and PV. Two elastic response entries reserve
// one slot for every accepted SRAM request until the PE consumes its response.
module attention_matmul_engine #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    output logic                         abort_ack,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic                         start_is_pv,
    input  logic [5:0]                   start_query_count,
    input  logic [11:0]                  start_output_base,
    input  logic [11:0]                  start_output_count,
    input  logic [11:0]                  start_reduction_count,
    output logic                         busy,
    output logic                         done_pulse,
    output logic                         error,
    output logic [3:0]                   error_id,

    output logic                         operand_req_valid,
    input  logic                         operand_req_ready,
    output logic                         operand_req_is_pv,
    output logic [5:0]                   operand_req_query_base,
    output logic [11:0]                  operand_req_output_base,
    output logic [11:0]                  operand_req_k_base,
    output logic [7:0]                   operand_req_row_mask,
    output logic [7:0]                   operand_req_k_mask,
    output logic [7:0]                   operand_req_col_mask,
    output logic                         operand_req_first_k_step,
    output logic                         operand_req_last_k_step,
    output logic [TAG_WIDTH-1:0]         operand_req_tag,
    input  logic [15:0]                  operand_v_static_scale,
    input  logic                         operand_rsp_valid,
    input  logic [1023:0]                operand_rsp_activation_rows,
    input  logic [511:0]                 operand_rsp_panel_rows,
    input  logic [127:0]                 operand_rsp_metadata,
    input  logic [127:0]                 operand_rsp_panel_metadata,
    output logic                         operand_abort_request,
    input  logic                         operand_abort_ack,

    output logic                         pe_req_valid,
    input  logic                         pe_req_ready,
    output logic [1:0]                   pe_req_mode,
    output logic [1023:0]                pe_req_activation_payload,
    output logic [1023:0]                pe_req_weight_payload,
    output logic [7:0]                   pe_req_row_mask,
    output logic [31:0]                  pe_req_k_mask,
    output logic [7:0]                   pe_req_col_mask,
    output logic                         pe_req_first_k_step,
    output logic                         pe_req_last_k_step,
    output logic [127:0]                 pe_req_activation_scales,
    output logic [127:0]                 pe_req_weight_scales,
    output logic [TAG_WIDTH-1:0]         pe_req_tag,
    input  logic                         pe_accum_result_valid,
    output logic                         pe_accum_result_ready,
    input  logic [2047:0]                pe_accum_result_accumulators,
    input  logic [63:0]                  pe_accum_result_mask,
    input  logic [1:0]                   pe_accum_result_mode,
    input  logic                         pe_accum_result_last_k_step,
    input  logic [127:0]                 pe_accum_result_activation_scales,
    input  logic [127:0]                 pe_accum_result_weight_scales,
    input  logic [TAG_WIDTH-1:0]         pe_accum_result_tag,
    output logic                         pe_abort_request,
    input  logic                         pe_abort_ack,

    output logic                         tile_result_valid,
    input  logic                         tile_result_ready,
    output logic                         tile_result_is_pv,
    output logic [5:0]                   tile_result_query_base,
    output logic [11:0]                  tile_result_output_base,
    output logic [2047:0]                tile_result_accumulators,
    output logic [63:0]                  tile_result_mask,
    output logic [127:0]                 tile_result_activation_scales,
    output logic [127:0]                 tile_result_weight_scales,
    output logic [TAG_WIDTH-1:0]         tile_result_tag,

    output logic [63:0]                  accepted_operand_request_count,
    output logic [63:0]                  completed_operand_response_count,
    output logic [63:0]                  accepted_pe_request_count,
    output logic [63:0]                  completed_pe_accum_result_count,
    output logic [63:0]                  completed_output_tile_count
);
    localparam logic [1:0] MODE_W8A8 = 2'd2;
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_PE_ACCUM_RESULT = 4'h2;
    localparam integer FIFO_WIDTH = 512 + 512 + 128 + 128 + 8 + 8 + 8 + 1 + 1 + TAG_WIDTH;

    typedef enum logic [2:0] {
        IDLE, ISSUE_TILE, WAIT_TILE, COMPLETE, ERROR_DRAIN, ABORT_DRAIN
    } state_t;
    state_t state;

    logic is_pv;
    logic [15:0] v_static_scale;
    logic [5:0] query_count;
    logic [11:0] output_start, output_count, reduction_count;
    logic [5:0] tile_query_base;
    logic [11:0] tile_output_base, issue_k_base;
    logic tile_all_requests_issued;
    logic [2:0] operand_requests_pending;

    logic fifo_input_ready, fifo_output_valid, fifo_output_ready;
    logic [FIFO_WIDTH-1:0] fifo_input_data, fifo_output_data;
    logic [2:0] fifo_occupancy;
    logic [FIFO_WIDTH-1:0] fifo_storage [0:1];
    logic fifo_read_pointer, fifo_write_pointer;
    logic [1:0] fifo_stored_count;
    logic fifo_input_valid, fifo_input_fire, fifo_output_fire;
    logic [511:0] fifo_activation_values, fifo_weight_values;
    logic [127:0] fifo_activation_scales, fifo_weight_scales;
    logic [7:0] fifo_row_mask, fifo_k_mask, fifo_col_mask;
    logic fifo_first_k_step, fifo_last_k_step;
    logic [TAG_WIDTH-1:0] fifo_tag;
    logic [511:0] formatted_activation_values;
    logic [511:0] formatted_weight_values;
    logic [127:0] formatted_activation_scales;
    logic [127:0] formatted_weight_scales;
    logic [1:0] operand_response_metadata_pending;
    logic saved_operand_is_pv [0:1];
    logic [11:0] saved_operand_k_base [0:1];
    logic [7:0] saved_operand_row_mask [0:1];
    logic [7:0] saved_operand_k_mask [0:1];
    logic [7:0] saved_operand_col_mask [0:1];
    logic saved_operand_first [0:1];
    logic saved_operand_last [0:1];
    logic [TAG_WIDTH-1:0] saved_operand_tag [0:1];

    logic output_pending;
    logic [2047:0] output_accumulators;
    logic [63:0] output_mask;
    logic [127:0] output_activation_scales, output_weight_scales;
    logic [TAG_WIDTH-1:0] output_tag;
    logic terminal_error;
    logic [3:0] terminal_error_id;
    logic tile_result_pending;
    logic [TAG_WIDTH-1:0] pending_tile_tag;
    logic pending_tile_advanced;
    logic pending_tile_final;
    logic output_tile_advanced;
    logic output_final_tile;

    logic operand_req_fire, operand_rsp_fire, pe_req_fire;
    logic operand_rsp_ready;
    logic pe_accum_result_fire, tile_result_fire;
    logic pe_accum_result_error_fire;

    function automatic [7:0] mask8(input [12:0] remaining);
        begin
            for (integer lane = 0; lane < 8; lane = lane + 1)
                mask8[lane] = remaining > 13'(lane);
        end
    endfunction

    function automatic [TAG_WIDTH-1:0] make_tag(
        input tag_is_pv,
        input [5:0] query_base,
        input [11:0] output_base
    );
        logic [15:0] packed_tag;
        begin
            packed_tag = {tag_is_pv, query_base[5:3], output_base[10:3], 4'b0000};
            make_tag = TAG_WIDTH'(packed_tag);
        end
    endfunction

    wire start_configuration_valid = start_query_count != 0 &&
        start_query_count <= 6'd48 && start_output_count != 0 &&
        start_reduction_count != 0 && start_reduction_count <= 12'd2048 &&
        {1'b0, start_output_base} + {1'b0, start_output_count} <= 13'd2048;
    wire [12:0] query_remaining = {7'd0, query_count} - {7'd0, tile_query_base};
    wire [12:0] output_consumed = {1'b0, tile_output_base} - {1'b0, output_start};
    wire [12:0] output_remaining = {1'b0, output_count} - output_consumed;
    wire [12:0] reduction_remaining = {1'b0, reduction_count} - {1'b0, issue_k_base};
    wire current_tile_final =
        tile_output_base + 12'd8 >= output_start + output_count &&
        tile_query_base + 6'd8 >= query_count;
    wire pipeline_next_tile = reduction_count >= 12'd128 &&
        !tile_result_pending;

    assign start_ready = state == IDLE && !abort_request;
    assign busy = state != IDLE;
    assign done_pulse = state == COMPLETE;
    assign error = done_pulse && terminal_error;
    assign error_id = terminal_error_id;

    assign operand_req_valid = state == ISSUE_TILE && !tile_all_requests_issued &&
        (operand_requests_pending < 3'd2 || pe_req_fire) && !abort_request &&
        !(operand_req_last_k_step && tile_result_pending);
    assign operand_req_is_pv = is_pv;
    assign operand_req_query_base = tile_query_base;
    assign operand_req_output_base = tile_output_base;
    assign operand_req_k_base = issue_k_base;
    assign operand_req_row_mask = mask8(query_remaining);
    assign operand_req_k_mask = mask8(reduction_remaining);
    assign operand_req_col_mask = mask8(output_remaining);
    assign operand_req_first_k_step = issue_k_base == 0;
    assign operand_req_last_k_step = issue_k_base + 12'd8 >= reduction_count;
    assign operand_req_tag = make_tag(is_pv, tile_query_base, tile_output_base);
    assign operand_req_fire = operand_req_valid && operand_req_ready;

    always_comb begin : format_operand_response
        formatted_activation_values = '0;
        formatted_weight_values = '0;
        formatted_activation_scales = operand_rsp_metadata;
        formatted_weight_scales = saved_operand_is_pv[1] ?
            {8{v_static_scale}} : operand_rsp_panel_metadata;
        for (integer operand_row = 0; operand_row < 8; operand_row++) begin
            formatted_activation_values[operand_row*64 +: 64] =
                saved_operand_k_base[1][3] ?
                    operand_rsp_activation_rows[operand_row*128+64 +: 64] :
                    operand_rsp_activation_rows[operand_row*128 +: 64];
        end
        for (integer operand_k = 0; operand_k < 8; operand_k++) begin
            integer operand_pair;
            operand_pair = operand_k >> 1;
            for (integer operand_column = 0; operand_column < 8;
                 operand_column++) begin
                if (saved_operand_is_pv[1]) begin
                    formatted_weight_values[
                        (operand_k*8+operand_column)*8 +: 8] =
                        operand_rsp_panel_rows[operand_pair*128 +
                            (operand_k[0] ? 64 : 0) + operand_column*8 +: 8];
                end else begin
                    formatted_weight_values[
                        (operand_k*8+operand_column)*8 +: 8] =
                        operand_rsp_panel_rows[operand_pair*128 +
                            operand_column*16 +
                            (operand_k[0] ? 8 : 0) +: 8];
                end
            end
        end
    end

    assign fifo_input_data = {
        formatted_activation_values, formatted_weight_values,
        formatted_activation_scales, formatted_weight_scales,
        saved_operand_row_mask[1], saved_operand_k_mask[1],
        saved_operand_col_mask[1], saved_operand_first[1],
        saved_operand_last[1], saved_operand_tag[1]
    };
    assign fifo_input_valid = operand_rsp_valid && !abort_request &&
        (state == ISSUE_TILE || state == WAIT_TILE);
    assign fifo_input_ready = fifo_stored_count < 2 ||
        (fifo_stored_count != 0 && fifo_output_ready);
    assign operand_rsp_ready = fifo_input_ready &&
        (state == ISSUE_TILE || state == WAIT_TILE);
    assign operand_rsp_fire = fifo_input_valid && fifo_input_ready;

    assign fifo_output_valid = fifo_stored_count != 0 || fifo_input_valid;
    assign fifo_output_data = fifo_stored_count == 0 ?
        fifo_input_data : fifo_storage[fifo_read_pointer];
    assign fifo_occupancy = {1'b0, fifo_stored_count};
    assign fifo_input_fire = operand_rsp_fire;
    assign fifo_output_fire = fifo_output_valid && fifo_output_ready;

    always_ff @(posedge clk) begin
        if (rst || state == ABORT_DRAIN || state == ERROR_DRAIN) begin
            fifo_read_pointer <= 1'b0;
            fifo_write_pointer <= 1'b0;
            fifo_stored_count <= 2'd0;
        end else begin
            if (fifo_input_fire &&
                !(fifo_stored_count == 0 && fifo_output_fire)) begin
                fifo_storage[fifo_write_pointer] <= fifo_input_data;
                fifo_write_pointer <= ~fifo_write_pointer;
            end
            if (fifo_output_fire && fifo_stored_count != 0)
                fifo_read_pointer <= ~fifo_read_pointer;
            case ({fifo_input_fire &&
                       !(fifo_stored_count == 0 && fifo_output_fire),
                   fifo_output_fire && fifo_stored_count != 0})
                2'b10: fifo_stored_count <= fifo_stored_count + 1'b1;
                2'b01: fifo_stored_count <= fifo_stored_count - 1'b1;
                default: fifo_stored_count <= fifo_stored_count;
            endcase
        end
    end

    assign {
        fifo_activation_values, fifo_weight_values,
        fifo_activation_scales, fifo_weight_scales,
        fifo_row_mask, fifo_k_mask, fifo_col_mask,
        fifo_first_k_step, fifo_last_k_step, fifo_tag
    } = fifo_output_data;

    assign pe_req_valid = fifo_output_valid &&
        (state == ISSUE_TILE || state == WAIT_TILE) && !abort_request;
    assign fifo_output_ready = pe_req_ready &&
        (state == ISSUE_TILE || state == WAIT_TILE) && !abort_request;
    assign pe_req_mode = MODE_W8A8;
    assign pe_req_row_mask = fifo_row_mask;
    assign pe_req_k_mask = {24'd0, fifo_k_mask};
    assign pe_req_col_mask = fifo_col_mask;
    assign pe_req_first_k_step = fifo_first_k_step;
    assign pe_req_last_k_step = fifo_last_k_step;
    assign pe_req_tag = fifo_tag;
    assign pe_req_fire = pe_req_valid && pe_req_ready;

    assign pe_req_activation_payload = {512'd0, fifo_activation_values};
    assign pe_req_weight_payload = {512'd0, fifo_weight_values};
    assign pe_req_activation_scales = fifo_activation_scales;
    assign pe_req_weight_scales = fifo_weight_scales;

    assign pe_accum_result_ready = !pe_accum_result_last_k_step || !output_pending;
    assign pe_accum_result_fire = pe_accum_result_valid && pe_accum_result_ready;
    assign pe_accum_result_error_fire = pe_accum_result_fire &&
        (pe_accum_result_mode != MODE_W8A8 ||
         (pe_accum_result_last_k_step &&
          (!tile_result_pending || pe_accum_result_tag != pending_tile_tag)));
    assign tile_result_valid = output_pending && !abort_request;
    assign tile_result_is_pv = is_pv;
    assign tile_result_query_base = {output_tag[14:12], 3'b000};
    assign tile_result_output_base = {1'b0, output_tag[11:4], 3'b000};
    assign tile_result_accumulators = output_accumulators;
    assign tile_result_mask = output_mask;
    assign tile_result_activation_scales = output_activation_scales;
    assign tile_result_weight_scales = output_weight_scales;
    assign tile_result_tag = output_tag;
    assign tile_result_fire = tile_result_valid && tile_result_ready;

    assign operand_abort_request = state == ABORT_DRAIN || state == ERROR_DRAIN;
    assign pe_abort_request = state == ABORT_DRAIN || state == ERROR_DRAIN;

`ifdef SYNTHESIS
    always_comb begin
        accepted_operand_request_count = '0;
        completed_operand_response_count = '0;
        accepted_pe_request_count = '0;
        completed_pe_accum_result_count = '0;
        completed_output_tile_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            operand_response_metadata_pending <= '0;
            saved_operand_is_pv[0] <= 1'b0;
            saved_operand_is_pv[1] <= 1'b0;
            saved_operand_k_base[0] <= '0;
            saved_operand_k_base[1] <= '0;
            saved_operand_row_mask[0] <= '0;
            saved_operand_row_mask[1] <= '0;
            saved_operand_k_mask[0] <= '0;
            saved_operand_k_mask[1] <= '0;
            saved_operand_col_mask[0] <= '0;
            saved_operand_col_mask[1] <= '0;
            saved_operand_first[0] <= 1'b0;
            saved_operand_first[1] <= 1'b0;
            saved_operand_last[0] <= 1'b0;
            saved_operand_last[1] <= 1'b0;
            saved_operand_tag[0] <= '0;
            saved_operand_tag[1] <= '0;
        end else begin
            operand_response_metadata_pending <= {
                operand_response_metadata_pending[0], operand_req_fire};
            saved_operand_is_pv[1] <= saved_operand_is_pv[0];
            saved_operand_k_base[1] <= saved_operand_k_base[0];
            saved_operand_row_mask[1] <= saved_operand_row_mask[0];
            saved_operand_k_mask[1] <= saved_operand_k_mask[0];
            saved_operand_col_mask[1] <= saved_operand_col_mask[0];
            saved_operand_first[1] <= saved_operand_first[0];
            saved_operand_last[1] <= saved_operand_last[0];
            saved_operand_tag[1] <= saved_operand_tag[0];
            if (operand_req_fire) begin
                saved_operand_is_pv[0] <= operand_req_is_pv;
                saved_operand_k_base[0] <= operand_req_k_base;
                saved_operand_row_mask[0] <= operand_req_row_mask;
                saved_operand_k_mask[0] <= operand_req_k_mask;
                saved_operand_col_mask[0] <= operand_req_col_mask;
                saved_operand_first[0] <= operand_req_first_k_step;
                saved_operand_last[0] <= operand_req_last_k_step;
                saved_operand_tag[0] <= operand_req_tag;
            end
            if (operand_abort_request)
                operand_response_metadata_pending <= '0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            is_pv <= 1'b0;
            v_static_scale <= '0;
            query_count <= '0;
            output_start <= '0;
            output_count <= '0;
            reduction_count <= '0;
            tile_query_base <= '0;
            tile_output_base <= '0;
            issue_k_base <= '0;
            tile_all_requests_issued <= 1'b0;
            operand_requests_pending <= '0;
            output_pending <= 1'b0;
            output_accumulators <= '0;
            output_mask <= '0;
            output_activation_scales <= '0;
            output_weight_scales <= '0;
            output_tag <= '0;
            tile_result_pending <= 1'b0;
            pending_tile_tag <= '0;
            pending_tile_advanced <= 1'b0;
            pending_tile_final <= 1'b0;
            output_tile_advanced <= 1'b0;
            output_final_tile <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
`ifndef SYNTHESIS
            accepted_operand_request_count <= '0;
            completed_operand_response_count <= '0;
            accepted_pe_request_count <= '0;
            completed_pe_accum_result_count <= '0;
            completed_output_tile_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            if (operand_req_fire) begin
`ifndef SYNTHESIS
                accepted_operand_request_count <= accepted_operand_request_count + 64'd1;
`endif
                if (operand_req_last_k_step) begin
                    tile_result_pending <= 1'b1;
                    pending_tile_tag <= operand_req_tag;
                    pending_tile_advanced <= pipeline_next_tile;
                    pending_tile_final <= current_tile_final;
                    if (pipeline_next_tile) begin
                        issue_k_base <= 12'd0;
                        tile_all_requests_issued <= 1'b0;
                        if (tile_output_base + 12'd8 <
                            output_start + output_count) begin
                            tile_output_base <= tile_output_base + 12'd8;
                        end else if (tile_query_base + 6'd8 < query_count) begin
                            tile_query_base <= tile_query_base + 6'd8;
                            tile_output_base <= output_start;
                        end else begin
                            state <= WAIT_TILE;
                        end
                    end else begin
                        tile_all_requests_issued <= 1'b1;
                    end
                end else begin
                    issue_k_base <= issue_k_base + 12'd8;
                end
            end
            case ({operand_req_fire, pe_req_fire})
                2'b10: operand_requests_pending <= operand_requests_pending + 3'd1;
                2'b01: operand_requests_pending <= operand_requests_pending - 3'd1;
                default: operand_requests_pending <= operand_requests_pending;
            endcase
`ifndef SYNTHESIS
            if (operand_rsp_fire)
                completed_operand_response_count <= completed_operand_response_count + 64'd1;
            if (pe_req_fire)
                accepted_pe_request_count <= accepted_pe_request_count + 64'd1;
`endif
            if (pe_accum_result_fire) begin
`ifndef SYNTHESIS
                completed_pe_accum_result_count <= completed_pe_accum_result_count + 64'd1;
`endif
                if (pe_accum_result_error_fire) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_PE_ACCUM_RESULT;
                    state <= ERROR_DRAIN;
                end else if (pe_accum_result_last_k_step) begin
                    tile_result_pending <= 1'b0;
                    output_pending <= 1'b1;
                    output_accumulators <= pe_accum_result_accumulators;
                    output_mask <= pe_accum_result_mask;
                    output_activation_scales <= pe_accum_result_activation_scales;
                    output_weight_scales <= pe_accum_result_weight_scales;
                    output_tag <= pe_accum_result_tag;
                    output_tile_advanced <= pending_tile_advanced;
                    output_final_tile <= pending_tile_final;
                    if (!pending_tile_advanced)
                        state <= WAIT_TILE;
                end
            end
            if (tile_result_fire) begin
                output_pending <= 1'b0;
`ifndef SYNTHESIS
                completed_output_tile_count <= completed_output_tile_count + 64'd1;
`endif
                if (output_tile_advanced) begin
                    if (output_final_tile)
                        state <= COMPLETE;
                end else begin
                    issue_k_base <= 12'd0;
                    tile_all_requests_issued <= 1'b0;
                    if (tile_output_base + 12'd8 < output_start + output_count) begin
                        tile_output_base <= tile_output_base + 12'd8;
                        state <= ISSUE_TILE;
                    end else if (tile_query_base + 6'd8 < query_count) begin
                        tile_query_base <= tile_query_base + 6'd8;
                        tile_output_base <= output_start;
                        state <= ISSUE_TILE;
                    end else begin
                        state <= COMPLETE;
                    end
                end
            end

            if (!pe_accum_result_error_fire) case (state)
                IDLE: if (start_valid && start_ready) begin
                    terminal_error <= !start_configuration_valid;
                    terminal_error_id <= start_configuration_valid ?
                        4'd0 : ERROR_CONFIGURATION;
                    is_pv <= start_is_pv;
                    if (start_is_pv)
                        v_static_scale <= operand_v_static_scale;
                    query_count <= start_query_count;
                    output_start <= start_output_base;
                    output_count <= start_output_count;
                    reduction_count <= start_reduction_count;
                    tile_query_base <= 6'd0;
                    tile_output_base <= start_output_base;
                    issue_k_base <= 12'd0;
                    tile_all_requests_issued <= 1'b0;
                    operand_requests_pending <= 3'd0;
                    output_pending <= 1'b0;
                    tile_result_pending <= 1'b0;
                    state <= start_configuration_valid ? ISSUE_TILE : COMPLETE;
                end
                ISSUE_TILE: if (tile_all_requests_issued)
                    state <= WAIT_TILE;
                WAIT_TILE: begin end
                COMPLETE: state <= IDLE;
                ERROR_DRAIN: begin
                    output_pending <= 1'b0;
                    tile_result_pending <= 1'b0;
                    if (operand_abort_ack && pe_abort_ack) begin
                        operand_requests_pending <= 3'd0;
                        state <= COMPLETE;
                    end
                end
                ABORT_DRAIN: begin
                    output_pending <= 1'b0;
                    tile_result_pending <= 1'b0;
                    if (operand_abort_ack && pe_abort_ack) begin
                        abort_ack <= 1'b1;
                        operand_requests_pending <= 3'd0;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase

            if (abort_request && state != IDLE && state != ABORT_DRAIN)
                state <= ABORT_DRAIN;
        end
    end

`ifndef SYNTHESIS
    logic held_tile_result;
    logic [2047:0] held_tile_result_accumulators;
    logic [63:0] held_tile_result_mask;
    logic [127:0] held_tile_result_activation_scales;
    logic [127:0] held_tile_result_weight_scales;
    logic [TAG_WIDTH-1:0] held_tile_result_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            held_tile_result <= 1'b0;
            held_tile_result_accumulators <= '0;
            held_tile_result_mask <= '0;
            held_tile_result_activation_scales <= '0;
            held_tile_result_weight_scales <= '0;
            held_tile_result_tag <= '0;
        end else begin
            assert (operand_requests_pending <= 3'd2)
                else $error("attention matmul exceeded two reserved operand response entries");
            assert (fifo_stored_count <= 2)
                else $error("attention matmul operand response storage overflow");
            if (operand_rsp_valid &&
                (state == ISSUE_TILE || state == WAIT_TILE) && !abort_request)
                assert (fifo_input_ready)
                    else $error("attention matmul had no reserved FIFO entry for an SRAM response");
            if (operand_rsp_valid)
                assert (operand_response_metadata_pending[1])
                    else $error("attention matmul received an operand response without matching metadata");
            if (operand_response_metadata_pending[1] && !operand_abort_request)
                assert (operand_rsp_valid)
                    else $error("attention matmul operand response missed the fixed SRAM latency");
            if (pe_req_fire)
                assert (operand_requests_pending != 0)
                    else $error("attention matmul consumed an operand without a reserved FIFO entry");
            if (pe_req_valid)
                assert (pe_req_mode == MODE_W8A8)
                    else $error("attention matmul issued a non-A8W8 PE request");
            if (tile_result_valid && !tile_result_ready) begin
                if (held_tile_result)
                    assert (tile_result_accumulators == held_tile_result_accumulators &&
                            tile_result_mask == held_tile_result_mask &&
                            tile_result_activation_scales ==
                                held_tile_result_activation_scales &&
                            tile_result_weight_scales == held_tile_result_weight_scales &&
                            tile_result_tag == held_tile_result_tag)
                        else $error("attention matmul changed a stalled tile result");
                held_tile_result <= 1'b1;
                held_tile_result_accumulators <= tile_result_accumulators;
                held_tile_result_mask <= tile_result_mask;
                held_tile_result_activation_scales <= tile_result_activation_scales;
                held_tile_result_weight_scales <= tile_result_weight_scales;
                held_tile_result_tag <= tile_result_tag;
            end else begin
                held_tile_result <= 1'b0;
            end
        end
    end
`endif
endmodule

`default_nettype wire
