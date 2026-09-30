`default_nettype none

// The single integer compute resource shared by mixed Matmul and fixed A8xW8
// QKV/QK/PV. It contains the three-stage arithmetic path, two sets of 64 signed
// 27-bit accumulators for mixed phases, and one ordered accum_result entry.
module mixed_precision_pe #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       abort_request,
    output logic                       abort_ack,

    input  logic                       req_valid,
    output logic                       req_ready,
    input  logic [1:0]                 req_mode,
    input  logic [8*32*4-1:0]          req_activation_payload,
    input  logic [32*8*4-1:0]          req_weight_payload,
    input  logic [7:0]                 req_row_mask,
    input  logic [31:0]                req_k_mask,
    input  logic [7:0]                 req_col_mask,
    input  logic                       req_first_k_step,
    input  logic                       req_last_k_step,
    input  logic                       req_mixed_phase,
    input  logic                       req_mixed_phase_first,
    input  logic [7:0]                 req_mixed_a8_rows,
    input  logic [8*16-1:0]            req_activation_scales,
    input  logic [8*16-1:0]            req_weight_scales,
    input  logic [TAG_WIDTH-1:0]       req_tag,

    output logic                       accum_result_valid,
    input  logic                       accum_result_ready,
    output logic [64*32-1:0]           accum_result_accumulators,
    output logic [63:0]                accum_result_mask,
    output logic [1:0]                 accum_result_mode,
    output logic                       accum_result_last_k_step,
    output logic [8*16-1:0]            accum_result_activation_scales,
    output logic [8*16-1:0]            accum_result_weight_scales,
    output logic [TAG_WIDTH-1:0]       accum_result_tag,

    output logic                       idle,
    output logic [63:0]                accepted_operand_count,
    output logic [63:0]                accepted_accum_result_count
);
    localparam logic [1:0] MODE_MIXED_W4 = 2'd3;
    localparam integer ACCUM_RESULT_WIDTH =
        64*27 + 64 + 2 + 1 + 8*16 + 8*16 + TAG_WIDTH;

    logic product_valid;
    logic product_ready;
    logic [64*18-1:0] product_step_sums;
    logic [63:0] product_mask;
    logic [1:0] product_mode;
    logic product_first;
    logic product_last;
    logic product_mixed_phase;
    logic product_mixed_phase_first;
    logic [7:0] product_mixed_a8_rows;
    logic [8*16-1:0] product_activation_scales;
    logic [8*16-1:0] product_weight_scales;
    logic [TAG_WIDTH-1:0] product_tag;
    logic product_idle;

    logic [128*27-1:0] accumulators;
    logic [64*27-1:0] selected_accumulators;
    logic [64*27-1:0] next_selected_accumulators;
    logic [64*27-1:0] low_result_accumulators;
    wire signed [27:0] extended_sums [0:63];
    logic [63:0] accumulation_overflow;
    logic [63:0] high_accumulator_select;
    logic [63:0] phase0_result_mask;
    logic [63:0] phase1_a4_result_mask;

    logic accum_result_input_ready;
    logic [ACCUM_RESULT_WIDTH-1:0] accum_result_input_data;
    logic accum_result_fifo_valid;
    logic [ACCUM_RESULT_WIDTH-1:0] accum_result_fifo_data;
    logic accum_result_output_ready;
    logic [64*27-1:0] compact_accum_result_accumulators;
    logic accum_result_input_valid;
    logic second_result_pending;
    logic [63:0] second_result_mask;
    logic [8*16-1:0] second_result_activation_scales;
    logic [8*16-1:0] second_result_weight_scales;
    logic [TAG_WIDTH-1:0] second_result_tag;
    logic [63:0] saved_phase0_mask;
    logic [8*16-1:0] saved_phase0_activation_scales;
    logic [8*16-1:0] saved_phase0_weight_scales;
    logic [TAG_WIDTH-1:0] saved_phase0_tag;
    logic product_req_ready;
    logic abort_pending;
    logic abort_request_seen;
    logic input_sequence_active;
    logic accumulator_sequence_active;
    logic [1:0] input_sequence_mode;
    logic [1:0] accumulator_sequence_mode;
    logic [TAG_WIDTH-1:0] input_sequence_tag;
    logic [TAG_WIDTH-1:0] accumulator_sequence_tag;
    logic input_order_ok;

    assign input_order_ok = req_first_k_step == !input_sequence_active &&
        (!input_sequence_active ||
         (req_mode == input_sequence_mode && req_tag == input_sequence_tag));
    assign req_ready = !abort_pending && !abort_request && product_req_ready &&
        input_order_ok;

    pe_product_pipeline #(.TAG_WIDTH(TAG_WIDTH)) product_pipeline (
        .clk(clk), .rst(rst),
        .req_valid(req_valid && !abort_pending && !abort_request && input_order_ok),
        .req_ready(product_req_ready), .req_mode(req_mode),
        .req_activation_payload(req_activation_payload),
        .req_weight_payload(req_weight_payload),
        .req_row_mask(req_row_mask), .req_k_mask(req_k_mask),
        .req_col_mask(req_col_mask), .req_first_k_step(req_first_k_step),
        .req_last_k_step(req_last_k_step),
        .req_mixed_phase(req_mixed_phase),
        .req_mixed_phase_first(req_mixed_phase_first),
        .req_mixed_a8_rows(req_mixed_a8_rows),
        .req_activation_scales(req_activation_scales),
        .req_weight_scales(req_weight_scales), .req_tag(req_tag),
        .rsp_valid(product_valid), .rsp_ready(product_ready),
        .rsp_step_sums(product_step_sums), .rsp_sum_mask(product_mask),
        .rsp_mode(product_mode), .rsp_first_k_step(product_first),
        .rsp_last_k_step(product_last),
        .rsp_mixed_phase(product_mixed_phase),
        .rsp_mixed_phase_first(product_mixed_phase_first),
        .rsp_mixed_a8_rows(product_mixed_a8_rows),
        .rsp_activation_scales(product_activation_scales),
        .rsp_weight_scales(product_weight_scales), .rsp_tag(product_tag),
        .idle(product_idle)
    );

    generate
        for (genvar lane = 0; lane < 64; lane = lane + 1) begin : accumulation_lanes
            localparam integer ROW = lane / 8;
            assign high_accumulator_select[lane] =
                product_mode == MODE_MIXED_W4 && product_mixed_phase &&
                !product_mixed_a8_rows[ROW];
            assign selected_accumulators[lane*27 +: 27] =
                high_accumulator_select[lane] ?
                accumulators[(64+lane)*27 +: 27] :
                accumulators[lane*27 +: 27];
            assign extended_sums[lane] =
                $signed({selected_accumulators[lane*27+26],
                         selected_accumulators[lane*27 +: 27]}) +
                $signed({{10{product_step_sums[lane*18+17]}},
                         product_step_sums[lane*18 +: 18]});
            assign next_selected_accumulators[lane*27 +: 27] =
                !product_mask[lane] ? 27'd0 :
                (product_first ||
                 (product_mode == MODE_MIXED_W4 &&
                  product_mixed_phase_first && high_accumulator_select[lane])) ?
                    {{9{product_step_sums[lane*18+17]}},
                     product_step_sums[lane*18 +: 18]} :
                    extended_sums[lane][26:0];
            assign accumulation_overflow[lane] = product_mask[lane] &&
                !(product_first ||
                  (product_mode == MODE_MIXED_W4 &&
                   product_mixed_phase_first && high_accumulator_select[lane])) &&
                extended_sums[lane][27] != extended_sums[lane][26];

            assign low_result_accumulators[lane*27 +: 27] =
                high_accumulator_select[lane] ?
                accumulators[lane*27 +: 27] :
                next_selected_accumulators[lane*27 +: 27];
            assign accum_result_accumulators[lane*32 +: 32] =
                {{5{compact_accum_result_accumulators[lane*27+26]}},
                 compact_accum_result_accumulators[lane*27 +: 27]};
        end
    endgenerate

    assign phase0_result_mask = saved_phase0_mask;
    assign phase1_a4_result_mask = product_mask & high_accumulator_select;
    assign product_ready = abort_pending ||
        (!second_result_pending && accum_result_input_ready);
    assign accum_result_input_valid = second_result_pending ||
        (product_valid && !abort_pending && !abort_request);
    assign accum_result_input_data = second_result_pending ?
        {accumulators[64*27 +: 64*27], second_result_mask,
         MODE_MIXED_W4, 1'b1, second_result_activation_scales,
         second_result_weight_scales, second_result_tag} :
        product_mode == MODE_MIXED_W4 && product_last ?
            {low_result_accumulators, phase0_result_mask,
             product_mode, product_last,
             saved_phase0_activation_scales, saved_phase0_weight_scales,
             saved_phase0_tag} :
            {next_selected_accumulators, product_mask, product_mode,
             product_last, product_activation_scales,
             product_weight_scales, product_tag};

    assign {
        compact_accum_result_accumulators, accum_result_mask, accum_result_mode,
        accum_result_last_k_step, accum_result_activation_scales,
        accum_result_weight_scales, accum_result_tag
    } = accum_result_fifo_data;
    assign accum_result_valid = accum_result_fifo_valid && !abort_pending;
    assign idle = product_idle && !accum_result_fifo_valid &&
        !second_result_pending &&
        !input_sequence_active && !accumulator_sequence_active && !abort_pending;

    assign accum_result_output_ready = accum_result_ready &&
        !abort_pending && !abort_request;
    assign accum_result_input_ready = !accum_result_fifo_valid ||
        accum_result_output_ready;

    // A single elastic entry is sufficient because the product pipeline
    // stops before overwriting a stalled result. Mixed mode's second result
    // remains in the high accumulators under second_result_pending.
    always_ff @(posedge clk) begin
        if (rst || abort_pending || abort_request) begin
            accum_result_fifo_valid <= 1'b0;
        end else begin
            case ({accum_result_input_valid && accum_result_input_ready,
                   accum_result_fifo_valid && accum_result_output_ready})
                2'b10: accum_result_fifo_valid <= 1'b1;
                2'b01: accum_result_fifo_valid <= 1'b0;
                default: begin end
            endcase
            if (accum_result_input_valid && accum_result_input_ready)
                accum_result_fifo_data <= accum_result_input_data;
        end
    end

`ifdef SYNTHESIS
    always_comb begin
        accepted_operand_count = '0;
        accepted_accum_result_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            accumulators <= '0;
            abort_pending <= 1'b0;
            abort_request_seen <= 1'b0;
            abort_ack <= 1'b0;
            input_sequence_active <= 1'b0;
            accumulator_sequence_active <= 1'b0;
            second_result_pending <= 1'b0;
`ifndef SYNTHESIS
            accepted_operand_count <= 64'd0;
            accepted_accum_result_count <= 64'd0;
`endif
        end else begin
            abort_ack <= 1'b0;
            if (!abort_request)
                abort_request_seen <= 1'b0;
            if (abort_request && !abort_request_seen) begin
                abort_pending <= 1'b1;
                abort_request_seen <= 1'b1;
                second_result_pending <= 1'b0;
            end
            if (req_valid && req_ready) begin
                input_sequence_active <= !req_last_k_step;
                input_sequence_mode <= req_mode;
                input_sequence_tag <= req_tag;
`ifndef SYNTHESIS
                accepted_operand_count <= accepted_operand_count + 64'd1;
`endif
            end
            if (second_result_pending && accum_result_input_ready)
                second_result_pending <= 1'b0;
            if (product_valid && product_ready && !abort_pending) begin
                for (integer lane = 0; lane < 64; lane = lane + 1) begin
                    if (high_accumulator_select[lane])
                        accumulators[(64+lane)*27 +: 27] <=
                            next_selected_accumulators[lane*27 +: 27];
                    else
                        accumulators[lane*27 +: 27] <=
                            next_selected_accumulators[lane*27 +: 27];
                end
                accumulator_sequence_active <= !product_last;
                accumulator_sequence_mode <= product_mode;
                accumulator_sequence_tag <= product_tag;
                if (product_mode == MODE_MIXED_W4 && !product_mixed_phase) begin
                    saved_phase0_mask <= product_mask;
                    saved_phase0_activation_scales <= product_activation_scales;
                    saved_phase0_weight_scales <= product_weight_scales;
                    saved_phase0_tag <= product_tag;
                end
                if (product_mode == MODE_MIXED_W4 && product_last &&
                    |phase1_a4_result_mask) begin
                    second_result_pending <= 1'b1;
                    second_result_mask <= phase1_a4_result_mask;
                    second_result_activation_scales <=
                        product_activation_scales;
                    second_result_weight_scales <= product_weight_scales;
                    second_result_tag <= product_tag;
                end
            end
`ifndef SYNTHESIS
            if (accum_result_valid && accum_result_ready)
                accepted_accum_result_count <= accepted_accum_result_count + 64'd1;
`endif
            if (abort_pending && product_idle && !accum_result_fifo_valid) begin
                abort_pending <= 1'b0;
                abort_ack <= 1'b1;
                input_sequence_active <= 1'b0;
                accumulator_sequence_active <= 1'b0;
                accumulators <= '0;
                second_result_pending <= 1'b0;
            end
        end
    end

    initial begin
        if (TAG_WIDTH < 1)
            $error("mixed_precision_pe requires TAG_WIDTH >= 1");
    end

`ifndef SYNTHESIS
    logic stalled_accum_result;
    logic [ACCUM_RESULT_WIDTH-1:0] stalled_accum_result_data;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_accum_result <= 1'b0;
            stalled_accum_result_data <= '0;
        end else begin
            if (req_valid && req_ready) begin
                assert (req_mode == MODE_MIXED_W4 ||
                        (!req_mixed_phase && !req_mixed_phase_first &&
                         req_mixed_a8_rows == 8'd0))
                    else $error("mixed_precision_pe received mixed row fields for format %0d",
                                req_mode);
                assert (req_first_k_step == !input_sequence_active)
                    else $error("mixed_precision_pe input first-K order is inconsistent");
                if (input_sequence_active)
                    assert (req_mode == input_sequence_mode &&
                            req_tag == input_sequence_tag)
                        else $error("mixed_precision_pe input numeric mode/tag changed within tile");
            end
            if (product_valid && product_ready && !abort_pending) begin
                assert (product_first == !accumulator_sequence_active)
                    else $error("mixed_precision_pe accumulator first-K order is inconsistent");
                if (accumulator_sequence_active)
                    assert (product_mode == accumulator_sequence_mode &&
                            product_tag == accumulator_sequence_tag)
                        else $error("mixed_precision_pe accumulator numeric mode/tag changed within tile");
                assert (!(|accumulation_overflow))
                    else $error("mixed_precision_pe accumulation overflowed signed 27 bit");
                if (product_mode == MODE_MIXED_W4) begin
                    assert (!product_last || product_mixed_phase)
                        else $error("mixed_precision_pe mixed result ended before phase 1");
                    assert (!product_first || !product_mixed_phase)
                        else $error("mixed_precision_pe mixed sequence started at phase 1");
                end
            end
            if (stalled_accum_result && !abort_pending)
                assert (accum_result_valid && accum_result_fifo_data == stalled_accum_result_data)
                    else $error("mixed_precision_pe changed a stalled accum_result");
            stalled_accum_result <= accum_result_valid && !accum_result_ready && !abort_pending;
            stalled_accum_result_data <= accum_result_fifo_data;
        end
    end
`endif
endmodule

`default_nettype wire
