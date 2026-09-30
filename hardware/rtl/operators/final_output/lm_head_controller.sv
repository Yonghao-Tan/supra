`default_nettype none

module lm_head_controller #(
    parameter integer PANEL_COUNT = 15808
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    output logic          pe_abort_request,
    input  logic          pe_abort_ack,
    output logic          rescale_abort_request,
    input  logic          rescale_abort_ack,

    input  logic          start_valid,
    output logic          start_ready,
    input  logic [6:0]    start_rows,
    input  logic [13:0]   start_panel_count,
    input  logic [63:0]   start_weight_base,
    input  logic [63:0]   start_weight_limit,
    input  logic [63:0]   start_scale_base,
    input  logic [63:0]   start_scale_limit,
    input  logic          activation_scale_load_valid,
    output logic          activation_scale_load_ready,
    input  logic [3:0]    activation_scale_load_group,
    input  logic [127:0]  activation_scale_load_data,

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

    output logic          local_scale_write_valid,
    input  logic          local_scale_write_ready,
    output logic [6:0]    local_scale_write_address,
    output logic [127:0]  local_scale_write_data,
    output logic          local_scale_read_valid,
    input  logic          local_scale_read_ready,
    output logic [6:0]    local_scale_read_address,
    input  logic          local_scale_read_rsp_valid,
    input  logic [127:0]  local_scale_read_rsp_data,
    output logic          local_panel_write_valid,
    input  logic          local_panel_write_ready,
    output hardware_types_pkg::matmul_panel_write_t local_panel_write,
    output logic          local_read_valid,
    input  logic          local_read_ready,
    output hardware_types_pkg::matmul_read_request_t local_read,
    input  logic          local_rsp_valid,
    input  hardware_types_pkg::matmul_read_response_t local_rsp,

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
    output logic [7:0]    error_id,
    output logic [63:0]   accepted_weight_bytes,
    output logic [63:0]   accepted_scale_bytes,
    output logic [63:0]   accepted_operand_count,
    output logic [63:0]   accepted_pe_count,
    output logic [63:0]   completed_logit_count,
    output logic [63:0]   operand_stall_cycles,
    output logic [63:0]   local_request_count,
    output logic [63:0]   local_response_count
);
    localparam logic [7:0] ERROR_START = 8'h01;
    localparam logic [7:0] ERROR_PANEL_LOAD = 8'h20;
    localparam logic [7:0] ERROR_PANEL_COMPUTE = 8'h30;

    typedef enum logic [2:0] {
        IDLE,
        FIRST_LOAD_START,
        FIRST_LOAD_WAIT,
        PANEL_START,
        PANEL_RUN,
        COMPLETE
    } state_t;

    state_t state;
    logic [6:0] saved_rows;
    logic [13:0] saved_panel_count;
    logic [63:0] saved_weight_base, saved_weight_limit;
    logic [63:0] saved_scale_base, saved_scale_limit;
    logic [1535:0] saved_activation_scales;
    logic [11:0] activation_scale_valid_mask;
    logic [13:0] active_panel_index;
    logic active_panel_select;
    logic compute_started, load_started;
    logic compute_finished, load_finished;
    logic compute_error_saved, load_error_saved;
    logic [3:0] compute_error_id_saved, load_error_id_saved;
    logic terminal_error;
    logic [7:0] terminal_error_id;
    logic abort_pending;
    logic loader_abort_seen, compute_abort_seen;

    logic loader_start_valid, loader_start_ready;
    logic [13:0] loader_start_index;
    logic loader_start_select;
    logic loader_panel_write_valid, loader_panel_write_ready;
    logic loader_panel_write_panel;
    logic [1:0] loader_panel_write_stripe;
    logic [8:0] loader_panel_write_row;
    logic [127:0] loader_panel_write_data;
    logic [15:0] loader_panel_write_byte_enable;
    logic [127:0] loader_scales;
    logic loader_done_valid, loader_done_ready;
    logic loader_error;
    logic [3:0] loader_error_id;
    logic loader_abort_ack;

    logic compute_start_valid, compute_start_ready;
    logic compute_done_valid, compute_done_ready;
    logic compute_error;
    logic [3:0] compute_error_id;
    logic operand_req_valid, operand_req_ready, operand_req_panel;
    logic [3:0] operand_req_group;
    logic [8:0] operand_req_k_tile;
    logic [7:0] operand_req_row_mask;
    logic [15:0] operand_req_tag;
    logic operand_rsp_valid, operand_rsp_ready;
    logic [511:0] operand_rsp_activation, operand_rsp_weight;
    logic [127:0] operand_rsp_activation_scales;
    logic [7:0] operand_rsp_row_mask;
    logic [15:0] operand_rsp_tag;
    logic operand_abort_ack;
    logic compute_abort_ack;
    logic compute_operand_abort_request;

    logic loader_start_fire, loader_done_fire;
    logic compute_start_fire, compute_done_fire;
    logic next_panel_exists;
    logic starts_complete, panel_work_complete;
    logic next_compute_error, next_load_error;

    assign start_ready = state == IDLE && !abort_pending && !abort_request;
    assign activation_scale_load_ready = state == IDLE &&
        !abort_pending && !abort_request && activation_scale_load_group < 12;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign next_panel_exists = active_panel_index + 14'd1 < saved_panel_count;

    assign loader_start_valid =
        !abort_pending && !abort_request &&
        ((state == FIRST_LOAD_START) ||
         (state == PANEL_START && next_panel_exists && !load_started));
    assign loader_start_index = state == FIRST_LOAD_START ? 14'd0 :
        active_panel_index + 14'd1;
    assign loader_start_select = state == FIRST_LOAD_START ? 1'b0 :
        !active_panel_select;
    assign loader_start_fire = loader_start_valid && loader_start_ready;
    assign loader_done_ready = state == FIRST_LOAD_WAIT || state == PANEL_RUN;
    assign loader_done_fire = loader_done_valid && loader_done_ready;

    assign compute_start_valid = state == PANEL_START && !compute_started &&
        !abort_pending && !abort_request;
    assign compute_start_fire = compute_start_valid && compute_start_ready;
    assign compute_done_ready = state == PANEL_RUN;
    assign compute_done_fire = compute_done_valid && compute_done_ready;
    assign starts_complete = (compute_started || compute_start_fire) &&
        (!next_panel_exists || load_started || loader_start_fire);
    assign panel_work_complete =
        (compute_finished || compute_done_fire) &&
        (!next_panel_exists || load_finished || loader_done_fire);
    assign next_compute_error = compute_error_saved ||
        (compute_done_fire && compute_error);
    assign next_load_error = load_error_saved ||
        (loader_done_fire && loader_error);

    always_comb begin
        local_panel_write = '0;
        local_panel_write.panel = loader_panel_write_panel;
        local_panel_write.plane = 2'd0;
        local_panel_write.start_bank = {1'b0, loader_panel_write_stripe};
        local_panel_write.word_row = {1'b0, loader_panel_write_row};
        local_panel_write.data[127:0] = loader_panel_write_data;
        local_panel_write.byte_enable[15:0] =
            loader_panel_write_byte_enable;
    end
    assign local_panel_write_valid = loader_panel_write_valid;
    assign loader_panel_write_ready = local_panel_write_ready;

    lm_head_raw_w8_panel_loader loader (
        .clk, .rst, .abort_request(abort_pending || abort_request),
        .abort_ack(loader_abort_ack), .start_valid(loader_start_valid),
        .start_ready(loader_start_ready),
        .start_panel_count(saved_panel_count),
        .start_panel_index(loader_start_index),
        .start_panel_select(loader_start_select),
        .start_weight_base(saved_weight_base),
        .start_weight_limit(saved_weight_limit),
        .start_scale_base(saved_scale_base),
        .start_scale_limit(saved_scale_limit),
        .dma_request_valid, .dma_request_ready, .dma_request_address,
        .dma_request_bytes, .dma_request_tag, .dma_response_valid,
        .dma_response_ready, .dma_response_data, .dma_response_byte_enable,
        .dma_response_last, .dma_done_pulse, .dma_error,
        .scale_write_valid(local_scale_write_valid),
        .scale_write_ready(local_scale_write_ready),
        .scale_write_address(local_scale_write_address),
        .scale_write_data(local_scale_write_data),
        .scale_read_valid(local_scale_read_valid),
        .scale_read_ready(local_scale_read_ready),
        .scale_read_address(local_scale_read_address),
        .scale_read_rsp_valid(local_scale_read_rsp_valid),
        .scale_read_rsp_data(local_scale_read_rsp_data),
        .panel_write_valid(loader_panel_write_valid),
        .panel_write_ready(loader_panel_write_ready),
        .panel_write_panel(loader_panel_write_panel),
        .panel_write_stripe(loader_panel_write_stripe),
        .panel_write_row(loader_panel_write_row),
        .panel_write_data(loader_panel_write_data),
        .panel_write_byte_enable(loader_panel_write_byte_enable),
        .panel_scales_bf16(loader_scales), .done_valid(loader_done_valid),
        .done_ready(loader_done_ready), .error(loader_error),
        .error_id(loader_error_id), .accepted_weight_bytes,
        .accepted_scale_bytes
    );

    lm_head_operand_path operand_path (
        .clk, .rst, .abort_request(abort_pending || abort_request ||
            compute_operand_abort_request),
        .abort_ack(operand_abort_ack), .req_valid(operand_req_valid),
        .req_ready(operand_req_ready), .req_panel(operand_req_panel),
        .req_group(operand_req_group), .req_k_tile(operand_req_k_tile),
        .req_row_mask(operand_req_row_mask), .req_tag(operand_req_tag),
        .activation_scales_bf16(saved_activation_scales),
        .local_read_valid, .local_read_ready, .local_read,
        .local_rsp_valid, .local_rsp, .rsp_valid(operand_rsp_valid),
        .rsp_ready(operand_rsp_ready), .rsp_activation(operand_rsp_activation),
        .rsp_weight(operand_rsp_weight),
        .rsp_activation_scales_bf16(operand_rsp_activation_scales),
        .rsp_row_mask(operand_rsp_row_mask), .rsp_tag(operand_rsp_tag),
        .accepted_request_count(local_request_count),
        .returned_response_count(local_response_count)
    );

    lm_head_panel_compute compute (
        .clk, .rst, .abort_request(abort_pending || abort_request),
        .abort_ack(compute_abort_ack),
        .operand_abort_request(compute_operand_abort_request),
        .operand_abort_ack(operand_abort_ack),
        .pe_abort_request, .pe_abort_ack,
        .rescale_abort_request, .rescale_abort_ack,
        .start_valid(compute_start_valid),
        .start_ready(compute_start_ready), .start_rows(saved_rows),
        .start_panel_index(active_panel_index),
        .start_panel_select(active_panel_select), .start_k_tiles(10'd512),
        .start_weight_scales_bf16(loader_scales), .operand_req_valid,
        .operand_req_ready, .operand_req_panel, .operand_req_group,
        .operand_req_k_tile, .operand_req_row_mask, .operand_req_tag,
        .operand_rsp_valid, .operand_rsp_ready, .operand_rsp_activation,
        .operand_rsp_weight,
        .operand_rsp_activation_scales_bf16(operand_rsp_activation_scales),
        .operand_rsp_row_mask, .operand_rsp_tag, .pe_req_valid,
        .pe_req_ready, .pe_req_mode, .pe_req_activation, .pe_req_weight,
        .pe_req_row_mask, .pe_req_k_mask, .pe_req_column_mask,
        .pe_req_first_k_step, .pe_req_last_k_step,
        .pe_req_activation_scales_bf16,
        .pe_req_weight_scales_bf16, .pe_req_tag, .pe_accum_valid,
        .pe_accum_ready, .pe_accumulators, .pe_accum_mask, .pe_accum_mode,
        .pe_accum_last_k_step, .pe_accum_activation_scales_bf16,
        .pe_accum_weight_scales_bf16, .pe_accum_tag, .pe_idle,
        .rescale_req_valid, .rescale_req_ready, .rescale_req_accumulators,
        .rescale_req_activation_scales_bf16,
        .rescale_req_weight_scales_bf16, .rescale_req_lane_mask,
        .rescale_req_tag, .rescale_rsp_valid, .rescale_rsp_ready,
        .rescale_rsp_values, .rescale_rsp_lane_mask, .rescale_rsp_tag,
        .rescale_idle, .logit_valid, .logit_ready, .logit_row_base,
        .logit_vocab_base, .logit_values_bf16, .logit_lane_mask,
        .done_valid(compute_done_valid), .done_ready(compute_done_ready),
        .error(compute_error), .error_id(compute_error_id),
        .accepted_operand_count, .accepted_pe_count,
        .completed_logit_count, .operand_stall_cycles
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_rows <= '0;
            saved_panel_count <= '0;
            saved_weight_base <= '0;
            saved_weight_limit <= '0;
            saved_scale_base <= '0;
            saved_scale_limit <= '0;
            saved_activation_scales <= '0;
            activation_scale_valid_mask <= '0;
            active_panel_index <= '0;
            active_panel_select <= 1'b0;
            compute_started <= 1'b0;
            load_started <= 1'b0;
            compute_finished <= 1'b0;
            load_finished <= 1'b0;
            compute_error_saved <= 1'b0;
            load_error_saved <= 1'b0;
            compute_error_id_saved <= '0;
            load_error_id_saved <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            loader_abort_seen <= 1'b0;
            compute_abort_seen <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (activation_scale_load_valid && activation_scale_load_ready) begin
                if (activation_scale_load_group == 0)
                    activation_scale_valid_mask <= 12'b1;
                else
                    activation_scale_valid_mask[
                        activation_scale_load_group] <= 1'b1;
                case (activation_scale_load_group)
                    4'd0: saved_activation_scales[0*128 +: 128] <= activation_scale_load_data;
                    4'd1: saved_activation_scales[1*128 +: 128] <= activation_scale_load_data;
                    4'd2: saved_activation_scales[2*128 +: 128] <= activation_scale_load_data;
                    4'd3: saved_activation_scales[3*128 +: 128] <= activation_scale_load_data;
                    4'd4: saved_activation_scales[4*128 +: 128] <= activation_scale_load_data;
                    4'd5: saved_activation_scales[5*128 +: 128] <= activation_scale_load_data;
                    4'd6: saved_activation_scales[6*128 +: 128] <= activation_scale_load_data;
                    4'd7: saved_activation_scales[7*128 +: 128] <= activation_scale_load_data;
                    4'd8: saved_activation_scales[8*128 +: 128] <= activation_scale_load_data;
                    4'd9: saved_activation_scales[9*128 +: 128] <= activation_scale_load_data;
                    4'd10: saved_activation_scales[10*128 +: 128] <= activation_scale_load_data;
                    default: saved_activation_scales[11*128 +: 128] <= activation_scale_load_data;
                endcase
            end
            if (abort_request && !abort_pending) begin
                abort_pending <= 1'b1;
                loader_abort_seen <= 1'b0;
                compute_abort_seen <= 1'b0;
            end
            if (abort_pending || abort_request) begin
                if (loader_abort_ack)
                    loader_abort_seen <= 1'b1;
                if (compute_abort_ack)
                    compute_abort_seen <= 1'b1;
                if ((loader_abort_seen || loader_abort_ack) &&
                    (compute_abort_seen || compute_abort_ack)) begin
                    abort_pending <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= IDLE;
                    compute_started <= 1'b0;
                    load_started <= 1'b0;
                    compute_finished <= 1'b0;
                    load_finished <= 1'b0;
                end
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    saved_rows <= start_rows;
                    saved_panel_count <= start_panel_count;
                    saved_weight_base <= start_weight_base;
                    saved_weight_limit <= start_weight_limit;
                    saved_scale_base <= start_scale_base;
                    saved_scale_limit <= start_scale_limit;
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    if (start_rows == 0 || start_rows > 7'd96 ||
                        (activation_scale_valid_mask &
                         ((12'b1 << ((start_rows + 7) >> 3)) - 1'b1)) !=
                        ((12'b1 << ((start_rows + 7) >> 3)) - 1'b1) ||
                        start_panel_count == 0 ||
                        start_panel_count > 14'(PANEL_COUNT)) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_START;
                        state <= COMPLETE;
                    end else begin
                        state <= FIRST_LOAD_START;
                    end
                end

                FIRST_LOAD_START: if (loader_start_fire)
                    state <= FIRST_LOAD_WAIT;

                FIRST_LOAD_WAIT: if (loader_done_fire) begin
                    if (loader_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_PANEL_LOAD |
                            {4'd0, loader_error_id};
                        state <= COMPLETE;
                    end else begin
                        active_panel_index <= 14'd0;
                        active_panel_select <= 1'b0;
                        compute_started <= 1'b0;
                        load_started <= 1'b0;
                        compute_finished <= 1'b0;
                        load_finished <= 1'b0;
                        compute_error_saved <= 1'b0;
                        load_error_saved <= 1'b0;
                        state <= PANEL_START;
                    end
                end

                PANEL_START: begin
                    if (compute_start_fire)
                        compute_started <= 1'b1;
                    if (loader_start_fire)
                        load_started <= 1'b1;
                    if (starts_complete) begin
                        compute_finished <= 1'b0;
                        load_finished <= !next_panel_exists;
                        compute_error_saved <= 1'b0;
                        load_error_saved <= 1'b0;
                        compute_error_id_saved <= '0;
                        load_error_id_saved <= '0;
                        state <= PANEL_RUN;
                    end
                end

                PANEL_RUN: begin
                    if (compute_done_fire) begin
                        compute_finished <= 1'b1;
                        compute_error_saved <= compute_error;
                        compute_error_id_saved <= compute_error_id;
                    end
                    if (loader_done_fire) begin
                        load_finished <= 1'b1;
                        load_error_saved <= loader_error;
                        load_error_id_saved <= loader_error_id;
                    end
                    if (panel_work_complete) begin
                        if (next_compute_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_PANEL_COMPUTE |
                                {4'd0, compute_done_fire ?
                                    compute_error_id : compute_error_id_saved};
                            state <= COMPLETE;
                        end else if (next_load_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_PANEL_LOAD |
                                {4'd0, loader_done_fire ?
                                    loader_error_id : load_error_id_saved};
                            state <= COMPLETE;
                        end else if (!next_panel_exists) begin
                            state <= COMPLETE;
                        end else begin
                            active_panel_index <= active_panel_index + 14'd1;
                            active_panel_select <= !active_panel_select;
                            compute_started <= 1'b0;
                            load_started <= 1'b0;
                            compute_finished <= 1'b0;
                            load_finished <= 1'b0;
                            state <= PANEL_START;
                        end
                    end
                end

                COMPLETE: if (done_valid && done_ready)
                    state <= IDLE;

                default: state <= IDLE;
            endcase

        end
    end

    initial begin
        if (PANEL_COUNT != 15808)
            $error("LM-head panel count must be 15808");
    end
endmodule

`default_nettype wire
