`default_nettype none

module ffn_r4_h12_pipeline #(
    parameter integer TAG_WIDTH = 16
) (
    input logic clk, input logic rst, input logic abort_request,
    output logic abort_ack, input logic arithmetic_abort_ack,
    input logic start_valid, output logic start_ready,
    input logic [2:0] start_row_count, input logic [14:0] start_word_base,
    output logic done_valid, input logic done_ready, output logic error,
    output logic product_req_valid, input logic product_req_ready,
    output hardware_types_pkg::elementwise_product_request_t product_req,
    input logic product_rsp_valid, output logic product_rsp_ready,
    input hardware_types_pkg::elementwise_product_response_t product_rsp,
    output logic arithmetic_req_valid, input logic arithmetic_req_ready,
    output logic [2:0] arithmetic_req_operation,
    output logic [1023:0] arithmetic_req_values,
    output logic [1023:0] arithmetic_req_paired_values,
    output logic [1023:0] arithmetic_req_factor0_values,
    output logic [63:0] arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0] arithmetic_req_tag,
    input logic arithmetic_rsp_valid, output logic arithmetic_rsp_ready,
    input logic [1023:0] arithmetic_rsp_values,
    input logic [63:0] arithmetic_rsp_lane_mask,
    input logic [TAG_WIDTH-1:0] arithmetic_rsp_tag,
    output logic max_req_valid, input logic max_req_ready,
    output logic [1023:0] max_req_values,
    output logic [63:0] max_req_lane_mask,
    output logic [TAG_WIDTH-1:0] max_req_tag
);
    localparam logic [2:0] BF16_ADD = 3'd0;
    localparam logic [2:0] BF16_MULTIPLY = 3'd1;
    localparam logic [15:0] NORMALIZATION = 16'h3c14;
    typedef enum logic [3:0] {
        IDLE, LOAD, REDUCE, NORMALIZE, WRITE, ADVANCE,
        COMPLETE, ERROR_HOLD, ABORT_DRAIN
    } state_t;
    state_t state;

    logic [2:0] row_count;
    logic [14:0] word_base;
    logic [5:0] chunk_index;
    logic [2:0] output_pair;
    logic [4:0] load_issue_count, load_response_count;
    logic [1023:0] source_vector [0:11];
    logic [1:0] context_level [0:1];
    logic [3:0] context_issue_count [0:1];
    logic [3:0] context_response_count [0:1];
    logic context_reduced [0:1];
    logic [1023:0] context0_term [0:5];
    logic [1023:0] context1_term [0:5];
    logic context_normalize_issued [0:1];
    logic context_normalize_valid [0:1];
    logic [2:0] write_index;
    logic [2:0] product_read_outstanding;
    logic [5:0] arithmetic_outstanding;
    logic product_write_accepted;
    logic max_write_accepted;
    logic [63:0] active_lane_mask;
    logic reduce_select_valid, reduce_select_context;
    logic [4:0] reduce_lhs_term, reduce_rhs_term;
    logic [3:0] reduce_output_block;
    logic [1023:0] reduce_lhs, reduce_rhs;
    logic normalize_select_valid, normalize_select_context;
    logic [1023:0] write_result;

    function automatic logic paley_negative(
        input logic [3:0] output_block,
        input logic [3:0] input_block
    );
        integer row;
        integer column;
        integer difference;
        begin
            paley_negative = 1'b0;
            if (output_block != 0 && input_block != 0) begin
                row = integer'(output_block) - 1;
                column = integer'(input_block) - 1;
                if (row == column) begin
                    paley_negative = 1'b1;
                end else begin
                    difference = row + 11 - column;
                    if (difference >= 11)
                        difference = difference - 11;
                    case (difference)
                        1, 3, 4, 5, 9: paley_negative = 1'b0;
                        default: paley_negative = 1'b1;
                    endcase
                end
            end
        end
    endfunction

    function automatic logic [3:0] reduction_input_count(
        input logic [1:0] level
    );
        case (level)
            2'd0: reduction_input_count = 4'd12;
            2'd1: reduction_input_count = 4'd6;
            2'd2: reduction_input_count = 4'd3;
            default: reduction_input_count = 4'd2;
        endcase
    endfunction

    function automatic logic [3:0] reduction_pair_count(
        input logic [1:0] level
    );
        case (level)
            2'd0: reduction_pair_count = 4'd6;
            2'd1: reduction_pair_count = 4'd3;
            2'd2: reduction_pair_count = 4'd2;
            default: reduction_pair_count = 4'd1;
        endcase
    endfunction

    function automatic logic [1023:0] reduction_value(
        input logic context_index,
        input logic [3:0] term_index
    );
        if (term_index >= 6)
            reduction_value = '0;
        else if (context_index)
            reduction_value = context1_term[term_index[2:0]];
        else
            reduction_value = context0_term[term_index[2:0]];
    endfunction

    always_comb begin
        active_lane_mask = '0;
        for (integer row = 0; row < 4; row++)
            active_lane_mask[row*16 +: 16] = {16{row < row_count}};

        reduce_select_valid = 1'b0;
        reduce_select_context = 1'b0;
        for (integer context_index = 0; context_index < 2;
             context_index++) begin
            if (!reduce_select_valid && !context_reduced[context_index] &&
                context_issue_count[context_index] <
                    reduction_pair_count(context_level[context_index])) begin
                reduce_select_valid = 1'b1;
                reduce_select_context = 1'(context_index);
            end
        end
        reduce_lhs_term = {context_issue_count[reduce_select_context], 1'b0};
        reduce_rhs_term = reduce_lhs_term + 5'd1;
        reduce_output_block = {output_pair, 1'b0} +
            4'(reduce_select_context);

        reduce_lhs = '0;
        reduce_rhs = '0;
        if (context_level[reduce_select_context] == 0) begin
            for (integer lane = 0; lane < 64; lane++) begin
                if (reduce_lhs_term < 12)
                    reduce_lhs[lane*16 +: 16] = {
                        source_vector[4'(reduce_lhs_term)][lane*16+15] ^
                            paley_negative(reduce_output_block,
                                4'(reduce_lhs_term)),
                        source_vector[4'(reduce_lhs_term)][lane*16 +: 15]};
                if (reduce_rhs_term < 12)
                    reduce_rhs[lane*16 +: 16] = {
                        source_vector[4'(reduce_rhs_term)][lane*16+15] ^
                            paley_negative(reduce_output_block,
                                4'(reduce_rhs_term)),
                        source_vector[4'(reduce_rhs_term)][lane*16 +: 15]};
            end
        end else begin
            if (reduce_lhs_term < 5'(
                    reduction_input_count(context_level[reduce_select_context])))
                reduce_lhs = reduction_value(
                    reduce_select_context, {1'b0, reduce_lhs_term[2:0]});
            if (reduce_rhs_term < 5'(
                    reduction_input_count(context_level[reduce_select_context])))
                reduce_rhs = reduction_value(
                    reduce_select_context, {1'b0, reduce_rhs_term[2:0]});
        end

        normalize_select_valid = 1'b0;
        normalize_select_context = 1'b0;
        for (integer context_index = 0; context_index < 2;
             context_index++) begin
            if (!normalize_select_valid && context_reduced[context_index] &&
                !context_normalize_issued[context_index]) begin
                normalize_select_valid = 1'b1;
                normalize_select_context = 1'(context_index);
            end
        end
    end

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE || state == ERROR_HOLD;
    assign error = state == ERROR_HOLD;

    assign product_req_valid = !abort_request &&
        ((state == LOAD && load_issue_count < 5'd24) ||
         (state == WRITE && !product_write_accepted));
    assign product_req.write = state == WRITE;
    assign product_req.word_index = state == LOAD ?
        15'(integer'(word_base) +
            ((integer'(load_issue_count[4:1]) * 128) +
             integer'(chunk_index) * 2 + integer'(load_issue_count[0])) * 8) :
        15'(integer'(word_base) +
            ((integer'({output_pair, 1'b0}) + integer'(write_index[1])) * 128 +
             integer'(chunk_index) * 2 + integer'(write_index[0])) * 8);
    assign write_result = reduction_value(write_index[1], 4'd0);
    assign product_req.data = write_result[write_index[0]*512 +: 512];
    assign product_req.half = state == LOAD ? load_issue_count[0] :
        write_index[0];
    assign product_req.tag = state == LOAD ?
        TAG_WIDTH'({11'd0, load_issue_count[4:1], load_issue_count[0]}) :
        TAG_WIDTH'({12'd0, write_index});
    assign product_rsp_ready = state == LOAD || state == ABORT_DRAIN;

    assign max_req_valid = state == WRITE && !max_write_accepted &&
        !abort_request;
    assign max_req_values = {512'd0, product_req.data[511:0]};
    assign max_req_lane_mask = {32'd0,
        {8{row_count > 3'd3}}, {8{row_count > 3'd2}},
        {8{row_count > 3'd1}}, {8{row_count > 3'd0}}};
    assign max_req_tag = TAG_WIDTH'(
        {5'd0, chunk_index, output_pair, write_index[1:0]});

    assign arithmetic_req_valid = !abort_request &&
        ((state == REDUCE && reduce_select_valid) ||
         (state == NORMALIZE && normalize_select_valid));
    assign arithmetic_req_operation = state == NORMALIZE ?
        BF16_MULTIPLY : BF16_ADD;
    assign arithmetic_req_values = state == NORMALIZE ?
        reduction_value(normalize_select_context, 4'd0) : reduce_lhs;
    assign arithmetic_req_paired_values = state == NORMALIZE ? '0 : reduce_rhs;
    assign arithmetic_req_factor0_values = state == NORMALIZE ?
        {64{NORMALIZATION}} : '0;
    assign arithmetic_req_lane_mask = active_lane_mask;
    assign arithmetic_req_tag = state == NORMALIZE ?
        TAG_WIDTH'({1'b1, 14'd0, normalize_select_context}) :
        TAG_WIDTH'({1'b0, 10'd0, reduce_select_context,
                    context_issue_count[reduce_select_context]});
    assign arithmetic_rsp_ready = state == REDUCE || state == NORMALIZE ||
        state == ABORT_DRAIN;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            row_count <= '0;
            word_base <= '0;
            chunk_index <= '0;
            output_pair <= '0;
            load_issue_count <= '0;
            load_response_count <= '0;
            context_level[0] <= '0;
            context_level[1] <= '0;
            context_issue_count[0] <= '0;
            context_issue_count[1] <= '0;
            context_response_count[0] <= '0;
            context_response_count[1] <= '0;
            context_reduced[0] <= 1'b0;
            context_reduced[1] <= 1'b0;
            context_normalize_issued[0] <= 1'b0;
            context_normalize_issued[1] <= 1'b0;
            context_normalize_valid[0] <= 1'b0;
            context_normalize_valid[1] <= 1'b0;
            write_index <= '0;
            product_read_outstanding <= '0;
            arithmetic_outstanding <= '0;
            product_write_accepted <= 1'b0;
            max_write_accepted <= 1'b0;
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
                    arithmetic_outstanding + 6'd1;
                2'b01: arithmetic_outstanding <=
                    arithmetic_outstanding - 6'd1;
                default: begin end
            endcase
            if (arithmetic_abort_ack)
                arithmetic_outstanding <= '0;

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    row_count <= start_row_count;
                    word_base <= start_word_base;
                    chunk_index <= '0;
                    output_pair <= '0;
                    load_issue_count <= '0;
                    load_response_count <= '0;
                    state <= start_row_count == 0 || start_row_count > 4 ?
                        ERROR_HOLD : LOAD;
                end
                LOAD: begin
                    if (product_req_valid && product_req_ready)
                        load_issue_count <= load_issue_count + 5'd1;
                    if (product_rsp_valid && product_rsp_ready) begin
                        if (product_rsp.tag[4:1] >= 12 ||
                            product_rsp.half != product_rsp.tag[0]) begin
                            state <= ERROR_HOLD;
                        end else begin
                            source_vector[product_rsp.tag[4:1]]
                                [product_rsp.tag[0]*512 +: 512] <=
                                product_rsp.data;
                            load_response_count <= load_response_count + 5'd1;
                            if (load_response_count == 5'd23) begin
                                context_level[0] <= '0;
                                context_level[1] <= '0;
                                context_issue_count[0] <= '0;
                                context_issue_count[1] <= '0;
                                context_response_count[0] <= '0;
                                context_response_count[1] <= '0;
                                context_reduced[0] <= 1'b0;
                                context_reduced[1] <= 1'b0;
                                context_normalize_issued[0] <= 1'b0;
                                context_normalize_issued[1] <= 1'b0;
                                context_normalize_valid[0] <= 1'b0;
                                context_normalize_valid[1] <= 1'b0;
                                state <= REDUCE;
                            end
                        end
                    end
                end
                REDUCE: begin
                    if (arithmetic_req_valid && arithmetic_req_ready)
                        context_issue_count[reduce_select_context] <=
                            context_issue_count[reduce_select_context] + 4'd1;
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                        logic response_context;
                        logic [3:0] response_pair;
                        response_context = arithmetic_rsp_tag[4];
                        response_pair = arithmetic_rsp_tag[3:0];
                        if (arithmetic_rsp_tag[15] ||
                            arithmetic_rsp_lane_mask != active_lane_mask ||
                            response_pair >= reduction_pair_count(
                                context_level[response_context])) begin
                            state <= ERROR_HOLD;
                        end else begin
                            if (response_pair < 6) begin
                                if (response_context)
                                    context1_term[3'(response_pair)] <=
                                        arithmetic_rsp_values;
                                else
                                    context0_term[3'(response_pair)] <=
                                        arithmetic_rsp_values;
                            end
                            context_response_count[response_context] <=
                                context_response_count[response_context] + 4'd1;
                            if (context_response_count[response_context] + 4'd1 ==
                                reduction_pair_count(
                                    context_level[response_context])) begin
                                if (context_level[response_context] == 2'd3) begin
                                    context_reduced[response_context] <= 1'b1;
                                end else begin
                                    context_level[response_context] <=
                                        context_level[response_context] + 2'd1;
                                    context_issue_count[response_context] <= '0;
                                    context_response_count[response_context] <= '0;
                                end
                            end
                        end
                    end
                    if (context_reduced[0] && context_reduced[1] &&
                        arithmetic_outstanding == 0)
                        state <= NORMALIZE;
                end
                NORMALIZE: begin
                    if (arithmetic_req_valid && arithmetic_req_ready)
                        context_normalize_issued[normalize_select_context] <= 1'b1;
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                        if (!arithmetic_rsp_tag[15] ||
                            arithmetic_rsp_lane_mask != active_lane_mask) begin
                            state <= ERROR_HOLD;
                        end else begin
                            if (arithmetic_rsp_tag[0])
                                context1_term[0] <= arithmetic_rsp_values;
                            else
                                context0_term[0] <= arithmetic_rsp_values;
                            context_normalize_valid[arithmetic_rsp_tag[0]] <= 1'b1;
                        end
                    end
                    if (context_normalize_valid[0] &&
                        context_normalize_valid[1] &&
                        arithmetic_outstanding == 0) begin
                        write_index <= '0;
                        product_write_accepted <= 1'b0;
                        max_write_accepted <= 1'b0;
                        state <= WRITE;
                    end
                end
                WRITE: begin
                    if (product_req_valid && product_req_ready)
                        product_write_accepted <= 1'b1;
                    if (max_req_valid && max_req_ready)
                        max_write_accepted <= 1'b1;
                    if ((product_write_accepted ||
                         (product_req_valid && product_req_ready)) &&
                        (max_write_accepted ||
                         (max_req_valid && max_req_ready))) begin
                        product_write_accepted <= 1'b0;
                        max_write_accepted <= 1'b0;
                        if (write_index == 3'd3)
                            state <= ADVANCE;
                        else
                            write_index <= write_index + 3'd1;
                    end
                end
                ADVANCE: begin
                    if (output_pair != 3'd5) begin
                        output_pair <= output_pair + 3'd1;
                        context_level[0] <= '0;
                        context_level[1] <= '0;
                        context_issue_count[0] <= '0;
                        context_issue_count[1] <= '0;
                        context_response_count[0] <= '0;
                        context_response_count[1] <= '0;
                        context_reduced[0] <= 1'b0;
                        context_reduced[1] <= 1'b0;
                        context_normalize_issued[0] <= 1'b0;
                        context_normalize_issued[1] <= 1'b0;
                        context_normalize_valid[0] <= 1'b0;
                        context_normalize_valid[1] <= 1'b0;
                        state <= REDUCE;
                    end else if (chunk_index != 6'd63) begin
                        chunk_index <= chunk_index + 6'd1;
                        output_pair <= '0;
                        load_issue_count <= '0;
                        load_response_count <= '0;
                        state <= LOAD;
                    end else begin
                        state <= COMPLETE;
                    end
                end
                COMPLETE, ERROR_HOLD: if (done_valid && done_ready)
                    state <= IDLE;
                ABORT_DRAIN: begin
                    if (product_read_outstanding == 0 &&
                        arithmetic_outstanding == 0) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
            if (abort_request && state != IDLE && state != COMPLETE &&
                state != ERROR_HOLD && state != ABORT_DRAIN) begin
                product_write_accepted <= 1'b0;
                max_write_accepted <= 1'b0;
                state <= ABORT_DRAIN;
            end
        end
    end
endmodule

`default_nettype wire
