`default_nettype none

module ffn_r4_h1024_pipeline #(
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
    output logic                         done_valid,
    input  logic                         done_ready,
    output logic                         error,

    output logic                         product_req_valid,
    input  logic                         product_req_ready,
    output hardware_types_pkg::elementwise_product_request_t product_req,
    input  logic                         product_rsp_valid,
    output logic                         product_rsp_ready,
    input  hardware_types_pkg::elementwise_product_response_t product_rsp,

    output logic                         arithmetic_req_valid,
    input  logic                         arithmetic_req_ready,
    output logic [1023:0]                arithmetic_req_values,
    output logic [1023:0]                arithmetic_req_paired_values,
    output logic [63:0]                  arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         arithmetic_req_tag,
    input  logic                         arithmetic_rsp_valid,
    output logic                         arithmetic_rsp_ready,
    input  logic [1023:0]                arithmetic_rsp_values,
    input  logic [63:0]                  arithmetic_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         arithmetic_rsp_tag
);
    localparam integer SLOT_COUNT = 6;

    typedef enum logic [2:0] {
        IDLE, ACTIVE, COMPLETE, ERROR_HOLD, ABORT_DRAIN
    } state_t;
    state_t state;

    logic [2:0] row_count;
    logic [14:0] word_base;
    logic [3:0] stage_index;
    logic [9:0] operation_issue_count;
    logic [9:0] operation_complete_count;

    logic [SLOT_COUNT-1:0] slot_valid;
    logic [SLOT_COUNT-1:0] slot_read_a_issued;
    logic [SLOT_COUNT-1:0] slot_read_b_issued;
    logic [SLOT_COUNT-1:0] slot_read_a_valid;
    logic [SLOT_COUNT-1:0] slot_read_b_valid;
    logic [SLOT_COUNT-1:0] slot_add_issued;
    logic [SLOT_COUNT-1:0] slot_result_valid;
    logic [SLOT_COUNT-1:0] slot_low_written;
    logic [1:0] slot_local_stage [0:SLOT_COUNT-1];
    logic [14:0] slot_word_a [0:SLOT_COUNT-1];
    logic [14:0] slot_word_b [0:SLOT_COUNT-1];
    logic [511:0] slot_read_a [0:SLOT_COUNT-1];
    logic [511:0] slot_read_b [0:SLOT_COUNT-1];

    logic allocate_valid;
    logic [2:0] allocate_slot;
    logic [14:0] allocate_word_a;
    logic [14:0] allocate_word_b;
    logic product_select_valid;
    logic [2:0] product_select_slot;
    logic [2:0] product_select_action;
    logic arithmetic_select_valid;
    logic [2:0] arithmetic_select_slot;
    logic [1023:0] selected_add_lhs;
    logic [1023:0] selected_add_rhs;
    logic [63:0] active_lane_mask;
    logic [2:0] product_read_outstanding;
    logic [2:0] arithmetic_outstanding;

    always_comb begin : allocate_address
        integer block;
        integer pair;
        integer first_word;
        integer second_word;

        allocate_valid = 1'b0;
        allocate_slot = '0;
        for (integer slot = 0; slot < SLOT_COUNT; slot++) begin
            if (!allocate_valid && !slot_valid[slot]) begin
                allocate_valid = 1'b1;
                allocate_slot = 3'(slot);
            end
        end

        block = integer'(operation_issue_count) >> 6;
        pair = integer'(operation_issue_count) & 63;
        first_word = 0;
        second_word = 0;
        case (stage_index)
            4'd0, 4'd1, 4'd2, 4'd3: begin
                first_word = block * 128 + pair * 2;
                second_word = first_word + 1;
            end
            4'd4: begin
                first_word = block * 128 + (pair >> 1) * 4 + (pair & 1);
                second_word = first_word + 2;
            end
            4'd5: begin
                first_word = block * 128 + (pair >> 2) * 8 + (pair & 3);
                second_word = first_word + 4;
            end
            4'd6: begin
                first_word = block * 128 + (pair >> 3) * 16 + (pair & 7);
                second_word = first_word + 8;
            end
            4'd7: begin
                first_word = block * 128 + (pair >> 4) * 32 + (pair & 15);
                second_word = first_word + 16;
            end
            4'd8: begin
                first_word = block * 128 + (pair >> 5) * 64 + (pair & 31);
                second_word = first_word + 32;
            end
            default: begin
                first_word = block * 128 + pair;
                second_word = first_word + 64;
            end
        endcase
        allocate_word_a = 15'(integer'(word_base) + first_word * 8);
        allocate_word_b = 15'(integer'(word_base) + second_word * 8);
    end

    always_comb begin : request_select
        product_select_valid = 1'b0;
        product_select_slot = '0;
        product_select_action = '0;
        for (integer slot = 0; slot < SLOT_COUNT; slot++) begin
            if (!product_select_valid && slot_valid[slot] &&
                slot_result_valid[slot]) begin
                product_select_valid = 1'b1;
                product_select_slot = 3'(slot);
                product_select_action = slot_low_written[slot] ? 3'd4 : 3'd3;
            end
        end
        for (integer slot = 0; slot < SLOT_COUNT; slot++) begin
            if (!product_select_valid && slot_valid[slot] &&
                !slot_read_a_issued[slot]) begin
                product_select_valid = 1'b1;
                product_select_slot = 3'(slot);
                product_select_action = 3'd1;
            end else if (!product_select_valid && slot_valid[slot] &&
                slot_read_a_issued[slot] && !slot_read_b_issued[slot]) begin
                product_select_valid = 1'b1;
                product_select_slot = 3'(slot);
                product_select_action = 3'd2;
            end
        end

        arithmetic_select_valid = 1'b0;
        arithmetic_select_slot = '0;
        for (integer slot = 0; slot < SLOT_COUNT; slot++) begin
            if (!arithmetic_select_valid && slot_valid[slot] &&
                slot_read_a_valid[slot] && slot_read_b_valid[slot] &&
                !slot_add_issued[slot]) begin
                arithmetic_select_valid = 1'b1;
                arithmetic_select_slot = 3'(slot);
            end
        end
    end

    always_comb begin : butterfly_format
        integer stride;
        integer local_position;
        integer butterfly_base;
        integer paired_position;
        integer source_lane;
        logic [3:0] butterfly_stage;
        logic [15:0] lhs_value;
        logic [15:0] rhs_value;

        butterfly_stage = stage_index == 0 ?
            {2'd0, slot_local_stage[arithmetic_select_slot]} : stage_index;
        stride = 1 << butterfly_stage;
        local_position = 0;
        butterfly_base = 0;
        paired_position = 0;
        source_lane = 0;
        lhs_value = '0;
        rhs_value = '0;
        selected_add_lhs = '0;
        selected_add_rhs = '0;
        for (integer lane = 0; lane < 64; lane++) begin
            if (stride < 8) begin
                local_position = lane & 7;
                case (butterfly_stage)
                    4'd0: butterfly_base = local_position & 6;
                    4'd1: butterfly_base = local_position & 4;
                    default: butterfly_base = 0;
                endcase
                paired_position = local_position < butterfly_base + stride ?
                    local_position + stride : local_position - stride;
                if (local_position < butterfly_base + stride) begin
                    lhs_value = lane < 32 ?
                        slot_read_a[arithmetic_select_slot]
                            [(lane/8)*128 + local_position*16 +: 16] :
                        slot_read_b[arithmetic_select_slot]
                            [((lane-32)/8)*128 + local_position*16 +: 16];
                    rhs_value = lane < 32 ?
                        slot_read_a[arithmetic_select_slot]
                            [(lane/8)*128 + paired_position*16 +: 16] :
                        slot_read_b[arithmetic_select_slot]
                            [((lane-32)/8)*128 + paired_position*16 +: 16];
                end else begin
                    lhs_value = lane < 32 ?
                        slot_read_a[arithmetic_select_slot]
                            [(lane/8)*128 + paired_position*16 +: 16] :
                        slot_read_b[arithmetic_select_slot]
                            [((lane-32)/8)*128 + paired_position*16 +: 16];
                    rhs_value = lane < 32 ?
                        slot_read_a[arithmetic_select_slot]
                            [(lane/8)*128 + local_position*16 +: 16] :
                        slot_read_b[arithmetic_select_slot]
                            [((lane-32)/8)*128 + local_position*16 +: 16];
                end
                selected_add_lhs[lane*16 +: 16] = lhs_value;
                selected_add_rhs[lane*16 +: 16] = {
                    rhs_value[15] ^ (local_position >= butterfly_base + stride),
                    rhs_value[14:0]};
            end else begin
                source_lane = lane & 31;
                lhs_value = slot_read_a[arithmetic_select_slot]
                    [source_lane*16 +: 16];
                rhs_value = slot_read_b[arithmetic_select_slot]
                    [source_lane*16 +: 16];
                selected_add_lhs[lane*16 +: 16] = lhs_value;
                selected_add_rhs[lane*16 +: 16] = {
                    rhs_value[15] ^ lane[5], rhs_value[14:0]};
            end
        end
    end

    always_comb begin
        active_lane_mask = '0;
        for (integer row = 0; row < 4; row++)
            active_lane_mask[row*16 +: 16] = {16{row < row_count}};
    end

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE || state == ERROR_HOLD;
    assign error = state == ERROR_HOLD;

    assign product_req_valid = state == ACTIVE && product_select_valid &&
        !abort_request;
    assign product_req.write = product_select_action >= 3'd3;
    assign product_req.word_index = product_select_action == 3'd1 ||
        product_select_action == 3'd3 ?
            slot_word_a[product_select_slot] :
            slot_word_b[product_select_slot];
    assign product_req.data = product_select_action == 3'd3 ?
        slot_read_a[product_select_slot] : slot_read_b[product_select_slot];
    assign product_req.half = product_select_action == 3'd2 ||
        product_select_action == 3'd4;
    assign product_req.tag = TAG_WIDTH'({12'd0, product_select_slot,
        product_select_action == 3'd2});
    assign product_rsp_ready = state == ACTIVE || state == ABORT_DRAIN;

    assign arithmetic_req_valid = state == ACTIVE &&
        arithmetic_select_valid && !abort_request;
    assign arithmetic_req_values = selected_add_lhs;
    assign arithmetic_req_paired_values = selected_add_rhs;
    assign arithmetic_req_lane_mask = active_lane_mask;
    assign arithmetic_req_tag = TAG_WIDTH'({13'd0, arithmetic_select_slot});
    assign arithmetic_rsp_ready = state == ACTIVE || state == ABORT_DRAIN;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            row_count <= '0;
            word_base <= '0;
            stage_index <= '0;
            operation_issue_count <= '0;
            operation_complete_count <= '0;
            slot_valid <= '0;
            slot_read_a_issued <= '0;
            slot_read_b_issued <= '0;
            slot_read_a_valid <= '0;
            slot_read_b_valid <= '0;
            slot_add_issued <= '0;
            slot_result_valid <= '0;
            slot_low_written <= '0;
            product_read_outstanding <= '0;
            arithmetic_outstanding <= '0;
            for (integer slot = 0; slot < SLOT_COUNT; slot++)
                slot_local_stage[slot] <= '0;
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
                    arithmetic_outstanding + 3'd1;
                2'b01: arithmetic_outstanding <=
                    arithmetic_outstanding - 3'd1;
                default: begin end
            endcase
            if (arithmetic_abort_ack)
                arithmetic_outstanding <= '0;

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    row_count <= start_row_count;
                    word_base <= start_word_base;
                    stage_index <= '0;
                    operation_issue_count <= '0;
                    operation_complete_count <= '0;
                    slot_valid <= '0;
                    slot_read_a_issued <= '0;
                    slot_read_b_issued <= '0;
                    slot_read_a_valid <= '0;
                    slot_read_b_valid <= '0;
                    slot_add_issued <= '0;
                    slot_result_valid <= '0;
                    slot_low_written <= '0;
                    for (integer slot = 0; slot < SLOT_COUNT; slot++)
                        slot_local_stage[slot] <= '0;
                    state <= start_row_count == 0 || start_row_count > 4 ?
                        ERROR_HOLD : ACTIVE;
                end
                ACTIVE: begin
                    if (allocate_valid && operation_issue_count < 10'd768) begin
                        slot_valid[allocate_slot] <= 1'b1;
                        slot_read_a_issued[allocate_slot] <= 1'b0;
                        slot_read_b_issued[allocate_slot] <= 1'b0;
                        slot_read_a_valid[allocate_slot] <= 1'b0;
                        slot_read_b_valid[allocate_slot] <= 1'b0;
                        slot_add_issued[allocate_slot] <= 1'b0;
                        slot_result_valid[allocate_slot] <= 1'b0;
                        slot_low_written[allocate_slot] <= 1'b0;
                        slot_local_stage[allocate_slot] <= '0;
                        slot_word_a[allocate_slot] <= allocate_word_a;
                        slot_word_b[allocate_slot] <= allocate_word_b;
                        operation_issue_count <= operation_issue_count + 10'd1;
                    end
                    if (product_req_valid && product_req_ready) begin
                        case (product_select_action)
                            3'd1: slot_read_a_issued[product_select_slot] <= 1'b1;
                            3'd2: slot_read_b_issued[product_select_slot] <= 1'b1;
                            3'd3: slot_low_written[product_select_slot] <= 1'b1;
                            3'd4: begin
                                slot_valid[product_select_slot] <= 1'b0;
                                slot_result_valid[product_select_slot] <= 1'b0;
                                operation_complete_count <=
                                    operation_complete_count + 10'd1;
                                if (operation_complete_count == 10'd767) begin
                                    if (stage_index == 4'd9) begin
                                        state <= COMPLETE;
                                    end else begin
                                        stage_index <= stage_index == 0 ?
                                            4'd4 : stage_index + 4'd1;
                                        operation_issue_count <= '0;
                                        operation_complete_count <= '0;
                                        slot_valid <= '0;
                                    end
                                end
                            end
                            default: begin end
                        endcase
                    end
                    if (product_rsp_valid && product_rsp_ready) begin
                        if (!slot_valid[product_rsp.tag[3:1]]) begin
                            state <= ERROR_HOLD;
                        end else if (product_rsp.tag[0]) begin
                            slot_read_b[product_rsp.tag[3:1]] <= product_rsp.data;
                            slot_read_b_valid[product_rsp.tag[3:1]] <= 1'b1;
                        end else begin
                            slot_read_a[product_rsp.tag[3:1]] <= product_rsp.data;
                            slot_read_a_valid[product_rsp.tag[3:1]] <= 1'b1;
                        end
                    end
                    if (arithmetic_req_valid && arithmetic_req_ready)
                        slot_add_issued[arithmetic_select_slot] <= 1'b1;
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                        if (!slot_valid[arithmetic_rsp_tag[2:0]] ||
                            arithmetic_rsp_lane_mask != active_lane_mask) begin
                            state <= ERROR_HOLD;
                        end else if (stage_index == 0 &&
                            slot_local_stage[arithmetic_rsp_tag[2:0]] < 2'd3) begin
                            slot_read_a[arithmetic_rsp_tag[2:0]] <=
                                arithmetic_rsp_values[511:0];
                            slot_read_b[arithmetic_rsp_tag[2:0]] <=
                                arithmetic_rsp_values[1023:512];
                            slot_add_issued[arithmetic_rsp_tag[2:0]] <= 1'b0;
                            slot_local_stage[arithmetic_rsp_tag[2:0]] <=
                                slot_local_stage[arithmetic_rsp_tag[2:0]] + 2'd1;
                        end else begin
                            slot_read_a[arithmetic_rsp_tag[2:0]] <=
                                arithmetic_rsp_values[511:0];
                            slot_read_b[arithmetic_rsp_tag[2:0]] <=
                                arithmetic_rsp_values[1023:512];
                            slot_result_valid[arithmetic_rsp_tag[2:0]] <= 1'b1;
                        end
                    end
                end
                COMPLETE, ERROR_HOLD: if (done_valid && done_ready)
                    state <= IDLE;
                ABORT_DRAIN: begin
                    if (product_read_outstanding == 0 &&
                        arithmetic_outstanding == 0) begin
                        slot_valid <= '0;
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
            if (abort_request && state == ACTIVE)
                state <= ABORT_DRAIN;
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && product_rsp_valid)
            assert (product_rsp_ready)
                else $error("R4 H1024 SRAM response met a stalled consumer");
        if (!rst && arithmetic_rsp_valid)
            assert (arithmetic_rsp_ready)
                else $error("R4 H1024 ADD response met a stalled consumer");
    end
`endif
endmodule

`default_nettype wire
