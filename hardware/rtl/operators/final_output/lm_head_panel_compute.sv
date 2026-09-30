`default_nettype none

module lm_head_panel_compute #(
    parameter integer MAX_ROWS = 96,
    parameter integer MAX_K_TILES = 512
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    output logic          operand_abort_request,
    input  logic          operand_abort_ack,
    output logic          pe_abort_request,
    input  logic          pe_abort_ack,
    output logic          rescale_abort_request,
    input  logic          rescale_abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [6:0]    start_rows,
    input  logic [13:0]   start_panel_index,
    input  logic          start_panel_select,
    input  logic [9:0]    start_k_tiles,
    input  logic [127:0]  start_weight_scales_bf16,

    output logic          operand_req_valid,
    input  logic          operand_req_ready,
    output logic          operand_req_panel,
    output logic [3:0]    operand_req_group,
    output logic [8:0]    operand_req_k_tile,
    output logic [7:0]    operand_req_row_mask,
    output logic [15:0]   operand_req_tag,
    input  logic          operand_rsp_valid,
    output logic          operand_rsp_ready,
    input  logic [511:0]  operand_rsp_activation,
    input  logic [511:0]  operand_rsp_weight,
    input  logic [127:0]  operand_rsp_activation_scales_bf16,
    input  logic [7:0]    operand_rsp_row_mask,
    input  logic [15:0]   operand_rsp_tag,

    output logic          pe_req_valid,
    input  logic          pe_req_ready,
    output logic [1:0]    pe_req_mode,
    output logic [1023:0] pe_req_activation,
    output logic [1023:0] pe_req_weight,
    output logic [7:0]    pe_req_row_mask,
    output logic [31:0]   pe_req_k_mask,
    output logic [7:0]    pe_req_column_mask,
    output logic          pe_req_first_k_step,
    output logic          pe_req_last_k_step,
    output logic [127:0]  pe_req_activation_scales_bf16,
    output logic [127:0]  pe_req_weight_scales_bf16,
    output logic [15:0]   pe_req_tag,
    input  logic          pe_accum_valid,
    output logic          pe_accum_ready,
    input  logic [2047:0] pe_accumulators,
    input  logic [63:0]   pe_accum_mask,
    input  logic [1:0]    pe_accum_mode,
    input  logic          pe_accum_last_k_step,
    input  logic [127:0]  pe_accum_activation_scales_bf16,
    input  logic [127:0]  pe_accum_weight_scales_bf16,
    input  logic [15:0]   pe_accum_tag,
    input  logic          pe_idle,

    output logic          rescale_req_valid,
    input  logic          rescale_req_ready,
    output logic [2047:0] rescale_req_accumulators,
    output logic [127:0] rescale_req_activation_scales_bf16,
    output logic [127:0] rescale_req_weight_scales_bf16,
    output logic [63:0]   rescale_req_lane_mask,
    output logic [15:0]   rescale_req_tag,
    input  logic          rescale_rsp_valid,
    output logic          rescale_rsp_ready,
    input  logic [1023:0] rescale_rsp_values,
    input  logic [63:0]   rescale_rsp_lane_mask,
    input  logic [15:0]   rescale_rsp_tag,
    input  logic          rescale_idle,

    output logic          logit_valid,
    input  logic          logit_ready,
    output logic [6:0]    logit_row_base,
    output logic [16:0]   logit_vocab_base,
    output logic [1023:0] logit_values_bf16,
    output logic [63:0]   logit_lane_mask,

    output logic          done_valid,
    input  logic          done_ready,
    output logic          error,
    output logic [3:0]    error_id,
    output logic [63:0]   accepted_operand_count,
    output logic [63:0]   accepted_pe_count,
    output logic [63:0]   completed_logit_count,
    output logic [63:0]   operand_stall_cycles
);
    localparam logic [1:0] MODE_W8A8 = 2'd2;
    localparam logic [3:0] ERROR_START = 4'h1;
    localparam logic [3:0] ERROR_OPERAND = 4'h2;
    localparam logic [3:0] ERROR_PE = 4'h3;
    localparam logic [3:0] ERROR_RESCALE = 4'h4;
    localparam integer FIFO_WIDTH = 512 + 512 + 128 + 8 + 1 + 1 + 5;

    typedef enum logic [2:0] {
        IDLE,
        ISSUE,
        DRAIN,
        COMPLETE
    } state_t;

    state_t state;
    logic [6:0] saved_rows;
    logic [13:0] saved_panel_index;
    logic saved_panel_select;
    logic [9:0] saved_k_tiles;
    logic [127:0] saved_weight_scales;
    logic [3:0] issue_group;
    logic [8:0] issue_k_tile;
    logic [3:0] group_count;
    logic [3:0] completed_groups;
    logic [2:0] operand_outstanding;
    logic terminal_error;
    logic [3:0] terminal_error_id;
    logic abort_pending;
    logic operand_abort_seen, pe_abort_seen, rescale_abort_seen;
    logic abort_active;
    logic all_abort_ack;

    logic fifo_input_ready, fifo_output_valid, fifo_output_ready;
    logic [FIFO_WIDTH-1:0] fifo_input_data, fifo_output_data;
    logic [2:0] fifo_occupancy;
    logic [511:0] fifo_activation, fifo_weight;
    logic [127:0] fifo_activation_scales;
    logic [7:0] fifo_row_mask;
    logic fifo_first, fifo_last;
    logic [4:0] fifo_operand_identity;
    logic [15:0] fifo_pe_tag;
    logic [7:0] expected_response_mask;

    logic rescale_metadata_valid;
    logic [6:0] rescale_row_base;
    logic [16:0] rescale_vocab_base;
    logic [15:0] rescale_expected_tag;

    logic operand_req_fire, operand_rsp_fire, pe_req_fire;
    logic pe_accum_fire, rescale_req_fire, rescale_rsp_fire;

    function automatic logic [7:0] row_mask_for_group(
        input logic [6:0] rows,
        input logic [3:0] group
    );
        logic [7:0] mask;
        integer remaining;
        begin
            remaining = integer'(rows) - integer'(group) * 8;
            for (integer row = 0; row < 8; row++)
                mask[row] = remaining > row;
            row_mask_for_group = mask;
        end
    endfunction

    function automatic logic [15:0] make_operand_tag(
        input logic panel,
        input logic [3:0] group,
        input logic [8:0] k_tile
    );
        make_operand_tag = {panel, group, k_tile, 2'b00};
    endfunction

    function automatic logic [15:0] make_pe_tag(
        input logic panel,
        input logic [3:0] group
    );
        make_pe_tag = {panel, group, 11'd0};
    endfunction

    assign abort_active = abort_request || abort_pending || terminal_error;
    assign operand_abort_request = abort_active;
    assign pe_abort_request = abort_active;
    assign rescale_abort_request = abort_active;
    assign all_abort_ack =
        (operand_abort_seen || operand_abort_ack) &&
        (pe_abort_seen || pe_abort_ack) &&
        (rescale_abort_seen || rescale_abort_ack);
    assign start_ready = state == IDLE && !abort_active;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;

    assign operand_req_valid = state == ISSUE && !abort_active &&
        ({1'b0, operand_outstanding} + {1'b0, fifo_occupancy} < 4'd3 ||
         pe_req_fire);
    assign operand_req_panel = saved_panel_select;
    assign operand_req_group = issue_group;
    assign operand_req_k_tile = issue_k_tile;
    assign operand_req_row_mask = row_mask_for_group(saved_rows, issue_group);
    assign operand_req_tag = make_operand_tag(
        saved_panel_select, issue_group, issue_k_tile);
    assign operand_req_fire = operand_req_valid && operand_req_ready;
    assign expected_response_mask = row_mask_for_group(
        saved_rows, operand_rsp_tag[14:11]);
    assign operand_rsp_ready = abort_active || fifo_input_ready;
    assign operand_rsp_fire = operand_rsp_valid && operand_rsp_ready;
    assign fifo_input_data = {
        operand_rsp_activation, operand_rsp_weight,
        operand_rsp_activation_scales_bf16, operand_rsp_row_mask,
        operand_rsp_tag[10:2] == 0,
        {1'b0, operand_rsp_tag[10:2]} + 10'd1 == saved_k_tiles,
        operand_rsp_tag[15:11]
    };

    ready_valid_fifo #(.DATA_WIDTH(FIFO_WIDTH), .DEPTH(3)) operand_fifo (
        .clk, .rst(rst || abort_active),
        .input_valid(operand_rsp_fire && !abort_active &&
            operand_rsp_row_mask == expected_response_mask &&
            operand_rsp_tag[15] == saved_panel_select &&
            operand_rsp_tag[1:0] == 2'b00 &&
            {1'b0, operand_rsp_tag[10:2]} < saved_k_tiles),
        .input_ready(fifo_input_ready), .input_data(fifo_input_data),
        .output_valid(fifo_output_valid), .output_ready(fifo_output_ready),
        .output_data(fifo_output_data), .occupancy(fifo_occupancy)
    );

    assign {
        fifo_activation, fifo_weight, fifo_activation_scales,
        fifo_row_mask, fifo_first, fifo_last, fifo_operand_identity
    } = fifo_output_data;
    assign fifo_pe_tag = {fifo_operand_identity, 11'd0};

    assign pe_req_valid = fifo_output_valid && !abort_active;
    assign fifo_output_ready = abort_active || pe_req_ready;
    assign pe_req_mode = MODE_W8A8;
    assign pe_req_activation = {512'd0, fifo_activation};
    assign pe_req_weight = {512'd0, fifo_weight};
    assign pe_req_row_mask = fifo_row_mask;
    assign pe_req_k_mask = 32'h0000_00ff;
    assign pe_req_column_mask = 8'hff;
    assign pe_req_first_k_step = fifo_first;
    assign pe_req_last_k_step = fifo_last;
    assign pe_req_activation_scales_bf16 = fifo_activation_scales;
    assign pe_req_weight_scales_bf16 = saved_weight_scales;
    assign pe_req_tag = fifo_pe_tag;
    assign pe_req_fire = pe_req_valid && pe_req_ready;

    assign rescale_req_valid = pe_accum_valid && pe_accum_last_k_step &&
        pe_accum_mode == MODE_W8A8 && !rescale_metadata_valid &&
        !abort_active;
    assign pe_accum_ready = abort_active || !pe_accum_last_k_step ||
        (rescale_req_ready && !rescale_metadata_valid);
    assign pe_accum_fire = pe_accum_valid && pe_accum_ready;
    assign rescale_req_fire = rescale_req_valid && rescale_req_ready;
    assign rescale_req_accumulators = pe_accumulators;
    assign rescale_req_lane_mask = pe_accum_mask;
    assign rescale_req_tag = pe_accum_tag;
    assign rescale_req_activation_scales_bf16 =
        pe_accum_activation_scales_bf16;
    assign rescale_req_weight_scales_bf16 = pe_accum_weight_scales_bf16;

    assign rescale_rsp_ready = abort_active ||
        (logit_ready && rescale_metadata_valid);
    assign rescale_rsp_fire = rescale_rsp_valid && rescale_rsp_ready;
    assign logit_valid = rescale_rsp_valid && rescale_metadata_valid &&
        rescale_rsp_tag == rescale_expected_tag && !abort_active;
    assign logit_row_base = rescale_row_base;
    assign logit_vocab_base = rescale_vocab_base;
    assign logit_values_bf16 = rescale_rsp_values;
    assign logit_lane_mask = rescale_rsp_lane_mask;

`ifdef SYNTHESIS
    always_comb begin
        accepted_operand_count = '0;
        accepted_pe_count = '0;
        completed_logit_count = '0;
        operand_stall_cycles = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_rows <= '0;
            saved_panel_index <= '0;
            saved_panel_select <= 1'b0;
            saved_k_tiles <= '0;
            saved_weight_scales <= '0;
            issue_group <= '0;
            issue_k_tile <= '0;
            group_count <= '0;
            completed_groups <= '0;
            operand_outstanding <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            rescale_metadata_valid <= 1'b0;
            rescale_row_base <= '0;
            rescale_vocab_base <= '0;
            rescale_expected_tag <= '0;
`ifndef SYNTHESIS
            accepted_operand_count <= '0;
            accepted_pe_count <= '0;
            completed_logit_count <= '0;
            operand_stall_cycles <= '0;
`endif
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            operand_abort_seen <= 1'b0;
            pe_abort_seen <= 1'b0;
            rescale_abort_seen <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (abort_request && !abort_pending) begin
                abort_pending <= 1'b1;
                operand_abort_seen <= 1'b0;
                pe_abort_seen <= 1'b0;
                rescale_abort_seen <= 1'b0;
                terminal_error <= 1'b0;
                terminal_error_id <= '0;
                operand_outstanding <= '0;
                rescale_metadata_valid <= 1'b0;
                state <= DRAIN;
            end else if (abort_pending) begin
                operand_outstanding <= '0;
                rescale_metadata_valid <= 1'b0;
                if (operand_abort_ack)
                    operand_abort_seen <= 1'b1;
                if (pe_abort_ack)
                    pe_abort_seen <= 1'b1;
                if (rescale_abort_ack)
                    rescale_abort_seen <= 1'b1;
                if (all_abort_ack) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                end
            end else if (terminal_error && state != COMPLETE) begin
                operand_outstanding <= '0;
                rescale_metadata_valid <= 1'b0;
                if (operand_abort_ack)
                    operand_abort_seen <= 1'b1;
                if (pe_abort_ack)
                    pe_abort_seen <= 1'b1;
                if (rescale_abort_ack)
                    rescale_abort_seen <= 1'b1;
                if (all_abort_ack)
                    state <= COMPLETE;
            end else begin
`ifndef SYNTHESIS
                if (operand_req_valid && !operand_req_ready)
                    operand_stall_cycles <= operand_stall_cycles + 64'd1;
                if (operand_req_fire)
                    accepted_operand_count <= accepted_operand_count + 64'd1;
                if (pe_req_fire)
                    accepted_pe_count <= accepted_pe_count + 64'd1;
`endif
                if (rescale_rsp_fire) begin
                    rescale_metadata_valid <= 1'b0;
                    completed_groups <= completed_groups + 4'd1;
`ifndef SYNTHESIS
                    completed_logit_count <= completed_logit_count + 64'd1;
`endif
                end
                case ({operand_req_fire, operand_rsp_fire})
                    2'b10: operand_outstanding <= operand_outstanding + 3'd1;
                    2'b01: operand_outstanding <= operand_outstanding - 3'd1;
                    default: begin end
                endcase

                case (state)
                    IDLE: if (start_valid && start_ready) begin
                        saved_rows <= start_rows;
                        saved_panel_index <= start_panel_index;
                        saved_panel_select <= start_panel_select;
                        saved_k_tiles <= start_k_tiles;
                        saved_weight_scales <= start_weight_scales_bf16;
                        issue_group <= '0;
                        issue_k_tile <= '0;
                        group_count <= 4'((integer'(start_rows) + 7) / 8);
                        completed_groups <= '0;
                        terminal_error <= 1'b0;
                        terminal_error_id <= '0;
                        operand_abort_seen <= 1'b0;
                        pe_abort_seen <= 1'b0;
                        rescale_abort_seen <= 1'b0;
                        if (start_rows == 0 || start_rows > 7'(MAX_ROWS) ||
                            start_panel_index >= 15808 || start_k_tiles == 0 ||
                            start_k_tiles > 10'(MAX_K_TILES)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_START;
                            state <= COMPLETE;
                        end else begin
                            state <= ISSUE;
                        end
                    end

                    ISSUE: if (operand_req_fire) begin
                        if (issue_k_tile + 10'd1 == saved_k_tiles) begin
                            issue_k_tile <= '0;
                            if (issue_group + 4'd1 == group_count) begin
                                state <= DRAIN;
                            end else begin
                                issue_group <= issue_group + 4'd1;
                            end
                        end else begin
                            issue_k_tile <= issue_k_tile + 9'd1;
                        end
                    end

                    DRAIN: begin
                        if (completed_groups == group_count &&
                            operand_outstanding == 0 && fifo_occupancy == 0 &&
                            pe_idle && rescale_idle && !rescale_metadata_valid)
                            state <= COMPLETE;
                    end

                    COMPLETE: if (done_valid && done_ready) begin
                        terminal_error <= 1'b0;
                        state <= IDLE;
                    end

                    default: state <= IDLE;
                endcase

                if (operand_rsp_fire &&
                    (operand_rsp_row_mask != expected_response_mask ||
                     operand_rsp_tag[15] != saved_panel_select ||
                     operand_rsp_tag[1:0] != 0 ||
                     {1'b0, operand_rsp_tag[10:2]} >= saved_k_tiles)) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_OPERAND;
                    operand_abort_seen <= 1'b0;
                    pe_abort_seen <= 1'b0;
                    rescale_abort_seen <= 1'b0;
                    state <= DRAIN;
                end
                if (pe_accum_fire &&
                    (pe_accum_mode != MODE_W8A8 ||
                     pe_accum_tag[15] != saved_panel_select)) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_PE;
                    operand_abort_seen <= 1'b0;
                    pe_abort_seen <= 1'b0;
                    rescale_abort_seen <= 1'b0;
                    state <= DRAIN;
                end
                if (rescale_req_fire) begin
                    rescale_metadata_valid <= 1'b1;
                    rescale_row_base <= {pe_accum_tag[14:11], 3'b000};
                    rescale_vocab_base <= 17'(saved_panel_index) * 17'd8;
                    rescale_expected_tag <= pe_accum_tag;
                end
                if (rescale_rsp_valid && rescale_metadata_valid &&
                    rescale_rsp_tag != rescale_expected_tag) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_RESCALE;
                    operand_abort_seen <= 1'b0;
                    pe_abort_seen <= 1'b0;
                    rescale_abort_seen <= 1'b0;
                    state <= DRAIN;
                end
            end
        end
    end

    initial begin
        if (MAX_ROWS != 96 || MAX_K_TILES != 512)
            $error("LM-head panel compute parameter configuration is invalid");
    end
endmodule

`default_nettype wire
