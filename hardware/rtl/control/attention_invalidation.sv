`default_nettype none

// Reduce one saved dependency row in increasing key order. The caller streams
// only changed keys and retains the matrix and pending vector in DDR. Scalar
// operations use the shared BF16 pipe; no activation SRAM access is required.
module attention_invalidation (
    input logic clk, rst,
    input logic abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic start_all_changes,
    input logic start_stable_unmask,
    input logic [11:0] start_changed_count,
    input logic [15:0] start_pending,
    input logic start_consumed,
    input logic value_valid,
    output logic value_ready,
    input logic [15:0] value_dependency,
    input logic [15:0] value_confidence,
    input logic value_remasked,
    output logic arithmetic_valid,
    input logic arithmetic_ready,
    output logic arithmetic_multiply,
    output logic [15:0] arithmetic_lhs, arithmetic_rhs,
    input logic arithmetic_result_valid,
    output logic arithmetic_result_ready,
    input logic [15:0] arithmetic_result,
    output logic arithmetic_abort_request,
    input logic arithmetic_abort_ack,
    output logic result_valid,
    input logic result_ready,
    output logic [15:0] result_invalidation,
    output logic [15:0] result_remask,
    output logic [15:0] result_pending,
    output logic error
);
    typedef enum logic [3:0] {
        IDLE, LOAD, SUBTRACT, FACTOR, SCALE, UNION_MULTIPLY,
        UNION_SUBTRACT, UNION_ADD, UPDATE, FINISH, DRAIN, ABORT_LOW
    } state_t;
    state_t state;
    logic all_changes, stable_unmask, remasked, have_remask, have_ordinary, outstanding;
    logic [11:0] remaining;
    logic [15:0] pending, dependency, confidence, risk, intermediate;
    logic [15:0] ordinary_max, remask_union;
    logic terminal_error;

    function automatic logic [15:0] clean_dependency(input logic [15:0] value);
        if (value[14:0] == 0) clean_dependency = value;
        else if (value[15] || value[14:7] == 8'hff) clean_dependency = 16'd0;
        else clean_dependency = value > 16'h3f80 ? 16'h3f80 : value;
    endfunction
    function automatic logic [15:0] clean_confidence(input logic [15:0] value);
        if (value[14:0] == 0) clean_confidence = value;
        else if (value[15] || value[14:0] > 15'h7f80) clean_confidence = 16'd0;
        else clean_confidence = value > 16'h3f80 ? 16'h3f80 : value;
    endfunction
    function automatic logic greater(input logic [15:0] lhs, rhs);
        // Both arguments are finite nonnegative values, including signed zero.
        greater = lhs[14:0] > rhs[14:0];
    endfunction

    assign start_ready = state == IDLE && !abort_request;
    assign value_ready = state == LOAD && !abort_request;
    assign result_valid = state == FINISH && !abort_request;
    assign result_remask = remask_union;
    assign result_invalidation = have_remask && greater(remask_union, ordinary_max) ?
        remask_union : ordinary_max;
    assign result_pending = greater(result_invalidation, pending) ? result_invalidation : pending;
    assign error = result_valid && terminal_error;
    assign arithmetic_abort_request = state == DRAIN;
    assign arithmetic_result_ready = outstanding;
    always_comb begin
        arithmetic_valid = !outstanding && !abort_request;
        arithmetic_multiply = 1'b0;
        arithmetic_lhs = '0;
        arithmetic_rhs = '0;
        case (state)
            SUBTRACT: begin
                arithmetic_lhs = remasked ? 16'h3f80 : 16'h4000;
                arithmetic_rhs = confidence ^ 16'h8000;
            end
            FACTOR: begin
                arithmetic_lhs = 16'h3f80;
                arithmetic_rhs = intermediate;
            end
            SCALE: begin
                arithmetic_multiply = 1'b1;
                arithmetic_lhs = dependency;
                arithmetic_rhs = intermediate;
            end
            UNION_MULTIPLY: begin
                arithmetic_multiply = 1'b1;
                arithmetic_lhs = remask_union;
                arithmetic_rhs = risk;
            end
            UNION_SUBTRACT: begin
                arithmetic_lhs = risk;
                arithmetic_rhs = intermediate ^ 16'h8000;
            end
            UNION_ADD: begin
                arithmetic_lhs = remask_union;
                arithmetic_rhs = intermediate;
            end
            default: arithmetic_valid = 1'b0;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            all_changes <= 1'b0; stable_unmask <= 1'b0; remasked <= 1'b0;
            have_remask <= 1'b0; have_ordinary <= 1'b0; outstanding <= 1'b0;
            remaining <= '0; pending <= '0; dependency <= '0; confidence <= '0;
            risk <= '0; intermediate <= '0; ordinary_max <= '0; remask_union <= '0;
            terminal_error <= 1'b0; abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (arithmetic_valid && arithmetic_ready) outstanding <= 1'b1;
            if (arithmetic_result_valid && arithmetic_result_ready) outstanding <= 1'b0;
            if (abort_request && state != DRAIN && state != ABORT_LOW) begin
                state <= DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    all_changes <= start_all_changes; stable_unmask <= start_stable_unmask;
                    remaining <= start_changed_count;
                    pending <= start_consumed ? 16'd0 : clean_dependency(start_pending);
                    ordinary_max <= '0; remask_union <= '0;
                    have_ordinary <= 1'b0; have_remask <= 1'b0;
                    terminal_error <= start_changed_count > 12'd2048 || (start_all_changes && start_stable_unmask);
                    state <= start_changed_count == 0 || start_changed_count > 12'd2048 ||
                        (start_all_changes && start_stable_unmask) ? FINISH : LOAD;
                end
                LOAD: if (value_valid && value_ready) begin
                    dependency <= clean_dependency(value_dependency);
                    confidence <= clean_confidence(value_confidence);
                    remasked <= value_remasked;
                    risk <= clean_dependency(value_dependency);
                    intermediate <= clean_confidence(value_confidence);
                    state <= value_remasked || all_changes ? SUBTRACT : stable_unmask ? SCALE : UPDATE;
                end
                SUBTRACT, FACTOR, SCALE, UNION_MULTIPLY, UNION_SUBTRACT, UNION_ADD:
                    if (arithmetic_result_valid && arithmetic_result_ready) begin
                        intermediate <= arithmetic_result;
                        if (arithmetic_result[14:7] == 8'hff) begin
                            terminal_error <= 1'b1;
                            state <= FINISH;
                        end else case (state)
                            SUBTRACT: state <= remasked ? FACTOR : SCALE;
                            FACTOR: state <= SCALE;
                            SCALE: begin
                                risk <= greater(arithmetic_result, 16'h3f80) ? 16'h3f80 : arithmetic_result;
                                state <= remasked && have_remask ? UNION_MULTIPLY : UPDATE;
                            end
                            UNION_MULTIPLY: state <= UNION_SUBTRACT;
                            UNION_SUBTRACT: state <= UNION_ADD;
                            default: begin
                                risk <= arithmetic_result;
                                state <= UPDATE;
                            end
                        endcase
                    end
                UPDATE: begin
                    if (remasked) begin
                        remask_union <= risk;
                        have_remask <= 1'b1;
                    end else begin
                        if (!have_ordinary || greater(risk, ordinary_max)) ordinary_max <= risk;
                        have_ordinary <= 1'b1;
                    end
                    remaining <= remaining - 12'd1;
                    state <= remaining == 1 ? FINISH : LOAD;
                end
                FINISH: if (result_valid && result_ready) state <= IDLE;
                DRAIN: if (!outstanding || arithmetic_abort_ack) begin
                    outstanding <= 1'b0;
                    abort_ack <= 1'b1;
                    state <= abort_request ? ABORT_LOW : IDLE;
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
