`default_nettype none

// Canonical joint selection retains the base rows and ranks at most 32 future
// positions. Candidate storage is 68 bits x 8 entries x 4 banks (272 B); each
// bank read and the bank selection are registered before tuple comparison.
// Base positions/actual/forecast bits use 13 bits x 48 (78 B); A4/A8 ordered indices
// use 5 bits x 32 x 2 (40 B). No activation or weight data is stored here.
module utilization_aware_prefetch_scheduler (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [5:0] start_base_token_count, start_future_token_count,
    input logic [5:0] start_target_token_count, start_max_next_tokens,
    input logic [6:0] start_prediction_target,
    input logic [15:0] start_min_reuse_score,
    input logic start_state_eligibility, start_allow_new_admission,
    input logic [10:0] start_next_block_start,
    input logic start_rank_priorities, start_low_dependency,
    input logic start_source_b_dependency_tie_rank, start_source_b_a4_only,
    input logic [15:0] start_block_step_index,
    input logic [5:0] start_available_rows,
    input logic base_valid,
    output logic base_ready,
    input logic [5:0] base_index,
    input logic [10:0] base_position,
    input logic base_a8, base_next_pass_a8,
    input logic base_prediction,
    input logic future_valid,
    output logic future_ready,
    input logic [4:0] future_index,
    input logic future_a8, future_tentative, future_eligible,
    input logic future_admission_allowed,
    input logic [15:0] future_priority_bf16, future_dependency_bf16,
    input logic [31:0] future_service_count,
    output logic add_valid,
    input logic add_ready,
    output logic [15:0] add_lhs_bf16, add_rhs_bf16,
    input logic add_result_valid,
    input logic [15:0] add_result_bf16,
    output logic add_abort_request,
    input logic add_abort_ack,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [31:0] future_prediction_mask, added_future_token_mask,
    output logic [5:0] future_prediction_count, added_future_token_count,
    output logic [7:0] base_activation_slots, joint_activation_slots, next_pass_activation_slots
);
    typedef enum logic [5:0] {
        IDLE, BASE_LOAD, FUTURE_LOAD, BASE_FIND, ORDER_START, ORDER_READ,
        ORDER_SELECT, ORDER_COMPARE, ORDER_FINISH, OVERLAP_APPLY,
        PREFIX_BEGIN, PREFIX_READ, PREFIX_SELECT, PREFIX_ADD, PREFIX_WAIT,
        CANDIDATE_CHECK, CANDIDATE_SCORE, NEXT_PREFIX, RESULT_BEGIN,
        RESULT_SCAN, COMPLETE, ABORT_DRAIN, ABORT_LOW,
        RANK_TARGET_READ, RANK_TARGET_SELECT, RANK_READ, RANK_SELECT,
        RANK_COMPARE, PRIORITY_READ, PRIORITY_SELECT, PRIORITY_NORMALIZE,
        PRIORITY_DIVIDE, PRIORITY_ROUND, PRIORITY_ADD, PRIORITY_WAIT, PRIORITY_STORE,
        PREDICTION_LIMIT, EXISTING_LIMIT, RESULT_FILTER
    } state_t;
    typedef struct packed {
        logic [31:0] service;
        logic [15:0] priority_bf16, dependency_bf16;
        logic eligible, tentative, a8, next_pass_a8_upgrade;
    } candidate_t;
    state_t state;
    candidate_t candidates [0:3][0:7];
    candidate_t bank_read [0:3];
    candidate_t selected_candidate, best_candidate, incoming_candidate;
    logic [12:0] base_positions [0:47];
    logic [4:0] a4_order [0:31], a8_order [0:31];
    logic [31:0] overlap_mask, visited_mask;
    logic [5:0] base_token_count, future_token_count, target_token_count, max_next_tokens;
    logic [6:0] prediction_target, prediction_remaining;
    logic [5:0] current_predictions, reused_confirmation_count, added_confirmation_count, progress_limit, existing_limit;
    logic [10:0] next_block_start;
    logic [5:0] load_index, base_scan, order_scan;
    logic [4:0] incoming_index, best_index;
    logic best_valid, terminal_error, add_outstanding;
    logic incoming_future_admission_allowed;
    logic [5:0] a4_candidate_count, a8_candidate_count, trial_a4_count, trial_a8_count, selected_a4_count, selected_a8_count;
    logic [5:0] add_limit;
    logic [5:0] tentative4, tentative8;
    logic [15:0] a4_priority_sum_bf16, candidate_priority_sum_bf16;
    logic [7:0] original_forecast_units, upgraded_forecast_units;
    logic [7:0] activation_slot_capacity, next_pass_activation_slot_capacity;
    logic [7:0] candidate_units, candidate_forecast_units;
    logic [5:0] candidate_rows, candidate_tentatives;
    logic [35:0] candidate_selection_key, best_selection_key;
    logic source_b_dependency_tie_rank, source_b_a4_only, source_b_late_step;
    logic [5:0] selected_tentative4, selected_tentative8;
    logic [1:0] selected_bucket, best_bucket;
    logic selected_before_best, selected_overlap, best_overlap;
    logic prefix_is_a4;
    logic [4:0] prefix_index;
    logic [5:0] result_index;
    logic rank_priorities, low_dependency;
    logic state_eligibility, allow_new_admission;
    logic [5:0] available_rows, rank_target, rank_scan;
    logic [4:0] ranks [0:31];
    logic [4:0] rank_count;
    logic [15:0] target_dependency, ranked_priority;
    logic [15:0] min_reuse_score;
    logic [5:0] numerator, denominator, remainder;
    logic [7:0] ratio_exponent;
    logic [6:0] ratio_fraction;
    logic [2:0] division_bit;
    logic [6:0] twice_remainder;
    assign twice_remainder = {remainder, 1'b0};

    assign start_ready = state == IDLE && !abort_request;
    assign base_ready = state == BASE_LOAD && !abort_request;
    assign future_ready = state == FUTURE_LOAD && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign add_valid = (state == PREFIX_ADD || state == PRIORITY_ADD) && !abort_request;
    assign add_abort_request = state == ABORT_DRAIN;
    assign add_lhs_bf16 = state == PRIORITY_ADD ? 16'h3f80 : prefix_is_a4 ? a4_priority_sum_bf16 : candidate_priority_sum_bf16;
    assign add_rhs_bf16 = state == PRIORITY_ADD ? ranked_priority ^ 16'h8000 : selected_candidate.priority_bf16;
    assign candidate_selection_key = {candidate_tentatives, candidate_priority_sum_bf16, candidate_rows,
        candidate_units};

    // Bucket order [0,3,1,2] for priorities in [0,.25,.5,.75,1].
    // Raw dependency remains separate from normalized priority.
    assign selected_bucket = selected_candidate.priority_bf16 < 16'h3e80 ? 2'd0 :
        selected_candidate.priority_bf16 < 16'h3f00 ? 2'd3 :
        selected_candidate.priority_bf16 < 16'h3f40 ? 2'd1 : 2'd2;
    assign best_bucket = best_candidate.priority_bf16 < 16'h3e80 ? 2'd0 :
        best_candidate.priority_bf16 < 16'h3f00 ? 2'd3 :
        best_candidate.priority_bf16 < 16'h3f40 ? 2'd1 : 2'd2;
    assign selected_overlap = overlap_mask[order_scan[4:0]];
    assign best_overlap = overlap_mask[best_index];
    always_comb begin
        // Source A is processed first, with its unchanged ordering. Source B
        // has independent prefixes; interleaving A/B has no selection benefit.
        selected_before_best = selected_candidate.priority_bf16 > best_candidate.priority_bf16;
        if (source_b_dependency_tie_rank && !selected_overlap) begin
            if (source_b_late_step && selected_bucket != best_bucket)
                selected_before_best = selected_bucket > best_bucket;
            if (!selected_candidate.tentative &&
                selected_candidate.dependency_bf16 != best_candidate.dependency_bf16)
                selected_before_best = selected_candidate.dependency_bf16 < best_candidate.dependency_bf16;
        end
        if (selected_candidate.service != best_candidate.service)
            selected_before_best = selected_candidate.service < best_candidate.service;
        if (selected_candidate.tentative != best_candidate.tentative)
            selected_before_best = selected_candidate.tentative;
        if (selected_overlap != best_overlap)
            selected_before_best = selected_overlap;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; base_token_count <= '0; future_token_count <= '0; target_token_count <= '0;
            max_next_tokens <= '0; next_block_start <= '0; load_index <= '0; base_scan <= '0;
            prediction_target <= '0; prediction_remaining <= '0; current_predictions <= '0;
            reused_confirmation_count <= '0; added_confirmation_count <= '0; progress_limit <= '0; existing_limit <= '0;
            order_scan <= '0; incoming_index <= '0; best_index <= '0;
            best_valid <= 1'b0; terminal_error <= 1'b0; add_outstanding <= 1'b0;
            incoming_future_admission_allowed <= 1'b0;
            abort_ack <= 1'b0;
            overlap_mask <= '0; visited_mask <= '0; future_prediction_mask <= '0; added_future_token_mask <= '0;
            future_prediction_count <= '0; added_future_token_count <= '0; base_activation_slots <= '0;
            joint_activation_slots <= '0; next_pass_activation_slots <= '0;
            a4_candidate_count <= '0; a8_candidate_count <= '0; trial_a4_count <= '0; trial_a8_count <= '0; selected_a4_count <= '0; selected_a8_count <= '0;
            tentative4 <= '0; tentative8 <= '0; a4_priority_sum_bf16 <= '0; candidate_priority_sum_bf16 <= '0;
            original_forecast_units <= '0; upgraded_forecast_units <= '0;
            activation_slot_capacity <= '0; next_pass_activation_slot_capacity <= '0;
            candidate_units <= '0; candidate_forecast_units <= '0; candidate_rows <= '0;
            candidate_tentatives <= '0; best_selection_key <= '0;
            add_limit <= '0; prefix_is_a4 <= 1'b0; prefix_index <= '0; result_index <= '0;
            selected_candidate <= '0; best_candidate <= '0; incoming_candidate <= '0;
            source_b_dependency_tie_rank <= 1'b0; source_b_a4_only <= 1'b0; source_b_late_step <= 1'b0;
            selected_tentative4 <= '0; selected_tentative8 <= '0;
            rank_priorities <= 1'b0; low_dependency <= 1'b0; available_rows <= '0;
            rank_target <= '0; rank_scan <= '0; rank_count <= '0; target_dependency <= '0;
            ranked_priority <= '0; numerator <= '0; denominator <= '0; remainder <= '0;
            min_reuse_score <= '0; state_eligibility <= 1'b0; allow_new_admission <= 1'b0;
            ratio_exponent <= '0; ratio_fraction <= '0; division_bit <= '0;
        end else begin
            abort_ack <= 1'b0;
            if (add_valid && add_ready) add_outstanding <= 1'b1;
            if (add_result_valid || add_abort_ack) add_outstanding <= 1'b0;
            if (abort_request && state != IDLE && state != ABORT_DRAIN && state != ABORT_LOW) begin
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    base_token_count <= start_base_token_count; future_token_count <= start_future_token_count;
                    target_token_count <= start_target_token_count; max_next_tokens <= start_max_next_tokens;
                    prediction_target <= start_prediction_target; current_predictions <= '0;
                    min_reuse_score <= start_min_reuse_score;
                    state_eligibility <= start_state_eligibility; allow_new_admission <= start_allow_new_admission;
                    reused_confirmation_count <= '0; added_confirmation_count <= '0;
                    next_block_start <= start_next_block_start;
                    source_b_dependency_tie_rank <= start_source_b_dependency_tie_rank;
                    source_b_a4_only <= start_source_b_a4_only;
                    source_b_late_step <= start_block_step_index >= 16'd2;
                    rank_priorities <= start_rank_priorities;
                    low_dependency <= start_low_dependency;
                    available_rows <= start_available_rows;
                    rank_target <= '0; rank_scan <= '0; rank_count <= '0;
                    load_index <= '0; base_activation_slots <= '0; original_forecast_units <= '0;
                    upgraded_forecast_units <= '0; future_prediction_mask <= '0; added_future_token_mask <= '0;
                    future_prediction_count <= '0; added_future_token_count <= '0; overlap_mask <= '0; visited_mask <= '0;
                    a4_candidate_count <= '0; a8_candidate_count <= '0; terminal_error <= 1'b0;
                    if (start_base_token_count > 48 || start_future_token_count > 32 || start_target_token_count == 0 ||
                        start_target_token_count > 48 || start_max_next_tokens == 0 || start_max_next_tokens > 32 || start_prediction_target > 64 || start_min_reuse_score > 16'h3f80 ||
                        {1'b0, start_next_block_start} + {6'd0, start_future_token_count} > 2048 ||
                        (start_rank_priorities && start_available_rows > start_future_token_count)) begin
                        terminal_error <= 1'b1; state <= COMPLETE;
                    end else state <= start_base_token_count == 0 ? FUTURE_LOAD : BASE_LOAD;
                end
                BASE_LOAD: if (base_valid && base_ready) begin
                    if (base_index != load_index) begin terminal_error <= 1'b1; state <= COMPLETE; end
                    else begin
                        base_positions[load_index] <= {base_a8, base_next_pass_a8, base_position};
                        current_predictions <= current_predictions + {5'd0, base_prediction};
                        base_activation_slots <= base_activation_slots + (base_a8 ? 8'd2 : 8'd1);
                        original_forecast_units <= original_forecast_units + (base_next_pass_a8 ? 8'd2 : 8'd1);
                        load_index <= load_index + 6'd1;
                        if (load_index + 6'd1 == base_token_count) begin load_index <= '0; state <= FUTURE_LOAD; end
                    end
                end
                FUTURE_LOAD: begin
                    if (future_token_count == 0) state <= ORDER_START;
                    else if (future_valid && future_ready) begin
                        if ({1'b0, future_index} != load_index || future_priority_bf16 > 16'h3f80 || future_service_count[31] ||
                            (source_b_dependency_tie_rank && future_dependency_bf16 > 16'h3f80)) begin
                            terminal_error <= 1'b1; state <= COMPLETE;
                        end else begin
                            incoming_candidate <= '{service: future_service_count, priority_bf16: future_priority_bf16,
                                dependency_bf16: future_dependency_bf16,
                                eligible: future_eligible, tentative: future_tentative, a8: future_a8, next_pass_a8_upgrade: 1'b0};
                            incoming_future_admission_allowed <= future_admission_allowed;
                            incoming_index <= future_index; base_scan <= '0; state <= BASE_FIND;
                        end
                    end
                end
                BASE_FIND: begin
                    if (base_scan < base_token_count && base_positions[base_scan][10:0] == next_block_start + {6'd0, incoming_index}) begin
                        overlap_mask[incoming_index] <= 1'b1;
                        incoming_candidate.next_pass_a8_upgrade <= !base_positions[base_scan][11];
                        if (state_eligibility && incoming_candidate.a8 && !base_positions[base_scan][12])
                            incoming_candidate.eligible <= 1'b0;
                    end
                    if (base_scan == base_token_count) begin
                        candidates[incoming_index[4:3]][incoming_index[2:0]] <= incoming_candidate;
                        candidates[incoming_index[4:3]][incoming_index[2:0]].eligible <= incoming_candidate.eligible &&
                            (overlap_mask[incoming_index] || incoming_candidate.tentative ||
                                (incoming_future_admission_allowed && (!state_eligibility || allow_new_admission))) &&
                            (rank_priorities || incoming_candidate.priority_bf16 >= min_reuse_score);
                        if (incoming_candidate.eligible && incoming_candidate.tentative &&
                                (rank_priorities || incoming_candidate.priority_bf16 >= min_reuse_score)) begin
                            if (overlap_mask[incoming_index]) reused_confirmation_count <= reused_confirmation_count + 6'd1;
                            else added_confirmation_count <= added_confirmation_count + 6'd1;
                        end
                        load_index <= load_index + 6'd1;
                        state <= load_index + 6'd1 == future_token_count ?
                            (rank_priorities ? RANK_TARGET_READ : ORDER_START) : FUTURE_LOAD;
                    end else base_scan <= base_scan + 6'd1;
                end
                RANK_TARGET_READ: begin
                    for (integer bank = 0; bank < 4; bank++) bank_read[bank] <= candidates[bank][rank_target[2:0]];
                    rank_scan <= '0; rank_count <= '0;
                    state <= RANK_TARGET_SELECT;
                end
                RANK_TARGET_SELECT: begin
                    target_dependency <= bank_read[rank_target[4:3]].priority_bf16;
                    state <= RANK_READ;
                end
                RANK_READ: begin
                    for (integer bank = 0; bank < 4; bank++) bank_read[bank] <= candidates[bank][rank_scan[2:0]];
                    state <= RANK_SELECT;
                end
                RANK_SELECT: begin selected_candidate <= bank_read[rank_scan[4:3]]; state <= RANK_COMPARE; end
                RANK_COMPARE: begin
                    if (rank_scan < available_rows) begin
                        if (selected_candidate.priority_bf16 < target_dependency ||
                            (selected_candidate.priority_bf16 == target_dependency && rank_scan < rank_target))
                            rank_count <= rank_count + 5'd1;
                        rank_scan <= rank_scan + 6'd1;
                        state <= RANK_READ;
                    end else begin
                        ranks[rank_target[4:0]] <= rank_count;
                        if (rank_target + 6'd1 == future_token_count) begin rank_target <= '0; state <= PRIORITY_READ; end
                        else begin rank_target <= rank_target + 6'd1; state <= RANK_TARGET_READ; end
                    end
                end
                PRIORITY_READ: begin
                    for (integer bank = 0; bank < 4; bank++) bank_read[bank] <= candidates[bank][rank_target[2:0]];
                    numerator <= {1'b0, ranks[rank_target[4:0]]};
                    denominator <= available_rows - 6'd1;
                    ratio_exponent <= 8'd127; ratio_fraction <= '0; division_bit <= '0;
                    state <= PRIORITY_SELECT;
                end
                PRIORITY_SELECT: begin
                    selected_candidate <= bank_read[rank_target[4:3]];
                    if (bank_read[rank_target[4:3]].tentative || overlap_mask[rank_target[4:0]]) begin
                        ranked_priority <= 16'h3f80; state <= PRIORITY_STORE;
                    end else if (available_rows <= 1 || rank_target >= available_rows) begin
                        ranked_priority <= '0; state <= PRIORITY_STORE;
                    end else if (numerator == 0) begin
                        ranked_priority <= '0; state <= low_dependency ? PRIORITY_ADD : PRIORITY_STORE;
                    end else state <= PRIORITY_NORMALIZE;
                end
                // rank/(available-1), with both integers <=31: exact rational
                // RNE equals FP32 divide followed by BF16 RNE for this domain.
                PRIORITY_NORMALIZE: begin
                    if (numerator < denominator) begin
                        numerator <= numerator << 1; ratio_exponent <= ratio_exponent - 8'd1;
                    end else begin remainder <= numerator - denominator; state <= PRIORITY_DIVIDE; end
                end
                PRIORITY_DIVIDE: begin
                    ratio_fraction <= {ratio_fraction[5:0], twice_remainder >= {1'b0, denominator}};
                    remainder <= twice_remainder >= {1'b0, denominator} ?
                        6'(twice_remainder - {1'b0, denominator}) : twice_remainder[5:0];
                    division_bit <= division_bit + 3'd1;
                    if (division_bit == 3'd6) state <= PRIORITY_ROUND;
                end
                PRIORITY_ROUND: begin
                    ranked_priority <= {1'b0, ratio_exponent, ratio_fraction} +
                        16'(twice_remainder > {1'b0, denominator} ||
                            (twice_remainder == {1'b0, denominator} && ratio_fraction[0]));
                    state <= low_dependency ? PRIORITY_ADD : PRIORITY_STORE;
                end
                PRIORITY_ADD: if (add_valid && add_ready) state <= PRIORITY_WAIT;
                PRIORITY_WAIT: if (add_result_valid) begin ranked_priority <= add_result_bf16; state <= PRIORITY_STORE; end
                PRIORITY_STORE: begin
                    for (integer bank = 0; bank < 4; bank++)
                        if (rank_target[4:3] == 2'(bank)) begin
                            candidates[bank][rank_target[2:0]].priority_bf16 <= ranked_priority;
                            candidates[bank][rank_target[2:0]].eligible <= selected_candidate.eligible && ranked_priority >= min_reuse_score;
                        end
                    if (rank_target + 6'd1 == future_token_count) state <= ORDER_START;
                    else begin rank_target <= rank_target + 6'd1; state <= PRIORITY_READ; end
                end
                ORDER_START: begin
                    activation_slot_capacity <= base_activation_slots == 0 ? 8'd0 : base_activation_slots <= 64 ? 8'd64 : 8'd128;
                    next_pass_activation_slot_capacity <= original_forecast_units == 0 ? 8'd0 : original_forecast_units <= 64 ? 8'd64 : 8'd128;
                    upgraded_forecast_units <= original_forecast_units;
                    order_scan <= '0; best_valid <= 1'b0;
                    prediction_remaining <= prediction_target > {1'b0, current_predictions} ?
                        prediction_target - {1'b0, current_predictions} : 7'd0;
                    state <= PREDICTION_LIMIT;
                end
                PREDICTION_LIMIT: begin
                    if (prediction_target == 0) progress_limit <= max_next_tokens;
                    else if (prediction_remaining >= {1'b0, max_next_tokens} || reused_confirmation_count + added_confirmation_count >= max_next_tokens)
                        progress_limit <= max_next_tokens;
                    else progress_limit <= prediction_remaining > {1'b0, reused_confirmation_count + added_confirmation_count} ?
                        prediction_remaining[5:0] : reused_confirmation_count + added_confirmation_count;
                    state <= EXISTING_LIMIT;
                end
                EXISTING_LIMIT: begin
                    existing_limit <= prediction_target == 0 ? progress_limit :
                        progress_limit > added_confirmation_count ? progress_limit - added_confirmation_count : 6'd0;
                    state <= future_token_count == 0 ? PREFIX_BEGIN : ORDER_READ;
                end
                ORDER_READ: begin
                    for (integer bank = 0; bank < 4; bank++) bank_read[bank] <= candidates[bank][order_scan[2:0]];
                    state <= ORDER_SELECT;
                end
                ORDER_SELECT: begin selected_candidate <= bank_read[order_scan[4:3]]; state <= ORDER_COMPARE; end
                ORDER_COMPARE: begin
                    if (selected_candidate.eligible && !visited_mask[order_scan[4:0]] &&
                        (!best_valid || selected_before_best)) begin
                        best_valid <= 1'b1; best_candidate <= selected_candidate; best_index <= order_scan[4:0];
                    end
                    order_scan <= order_scan + 6'd1;
                    state <= order_scan + 6'd1 == future_token_count ? ORDER_FINISH : ORDER_READ;
                end
                ORDER_FINISH: begin
                    if (!best_valid) state <= PREFIX_BEGIN;
                    else begin
                        visited_mask[best_index] <= 1'b1;
                        if (overlap_mask[best_index]) state <= OVERLAP_APPLY;
                        else begin
                            if (best_candidate.a8) begin a8_order[a8_candidate_count[4:0]] <= best_index; a8_candidate_count <= a8_candidate_count + 6'd1; end
                            else begin a4_order[a4_candidate_count[4:0]] <= best_index; a4_candidate_count <= a4_candidate_count + 6'd1; end
                            order_scan <= '0; best_valid <= 1'b0; state <= ORDER_READ;
                        end
                    end
                end
                OVERLAP_APPLY: begin
                    if (future_prediction_count < existing_limit && upgraded_forecast_units + {7'd0, best_candidate.next_pass_a8_upgrade} <= next_pass_activation_slot_capacity) begin
                        future_prediction_mask[best_index] <= 1'b1; future_prediction_count <= future_prediction_count + 6'd1;
                        upgraded_forecast_units <= upgraded_forecast_units + {7'd0, best_candidate.next_pass_a8_upgrade};
                    end
                    order_scan <= '0; best_valid <= 1'b0; state <= ORDER_READ;
                end
                PREFIX_BEGIN: begin
                    add_limit <= target_token_count <= base_token_count ? 6'd0 :
                        target_token_count - base_token_count < progress_limit - future_prediction_count ?
                            target_token_count - base_token_count : progress_limit - future_prediction_count;
                    trial_a4_count <= '0; trial_a8_count <= '0; selected_a4_count <= '0; selected_a8_count <= '0;
                    selected_tentative4 <= '0; selected_tentative8 <= '0;
                    tentative4 <= '0; tentative8 <= '0; a4_priority_sum_bf16 <= '0; candidate_priority_sum_bf16 <= '0; best_selection_key <= '0;
                    joint_activation_slots <= base_activation_slots; next_pass_activation_slots <= upgraded_forecast_units;
                    state <= CANDIDATE_CHECK;
                end
                CANDIDATE_CHECK: begin
                    candidate_rows <= trial_a4_count + trial_a8_count;
                    candidate_tentatives <= tentative4 + tentative8;
                    candidate_units <= base_activation_slots + {2'd0, trial_a4_count} + {1'b0, trial_a8_count, 1'b0};
                    candidate_forecast_units <= upgraded_forecast_units + {1'b0, trial_a4_count, 1'b0} + {1'b0, trial_a8_count, 1'b0};
                    state <= CANDIDATE_SCORE;
                end
                CANDIDATE_SCORE: begin
                    if ({1'b0, base_token_count} + {1'b0, candidate_rows} <= {1'b0, target_token_count} &&
                        {1'b0, future_prediction_count} + {1'b0, candidate_rows} <= {1'b0, progress_limit} &&
                        candidate_units <= activation_slot_capacity && candidate_forecast_units <= next_pass_activation_slot_capacity &&
                        candidate_selection_key > best_selection_key) begin
                        best_selection_key <= candidate_selection_key; selected_a4_count <= trial_a4_count; selected_a8_count <= trial_a8_count;
                        selected_tentative4 <= tentative4; selected_tentative8 <= tentative8;
                        joint_activation_slots <= candidate_units; next_pass_activation_slots <= candidate_forecast_units;
                    end
                    state <= NEXT_PREFIX;
                end
                NEXT_PREFIX: begin
                    if (trial_a8_count < a8_candidate_count && trial_a4_count + trial_a8_count < add_limit) begin
                        prefix_is_a4 <= 1'b0; prefix_index <= a8_order[trial_a8_count[4:0]]; state <= PREFIX_READ;
                    end else if (trial_a4_count < a4_candidate_count && trial_a4_count < add_limit) begin
                        prefix_is_a4 <= 1'b1; prefix_index <= a4_order[trial_a4_count[4:0]]; state <= PREFIX_READ;
                    end else state <= RESULT_FILTER;
                end
                PREFIX_READ: begin
                    for (integer bank = 0; bank < 4; bank++) bank_read[bank] <= candidates[bank][prefix_index[2:0]];
                    state <= PREFIX_SELECT;
                end
                PREFIX_SELECT: begin selected_candidate <= bank_read[prefix_index[4:3]]; state <= PREFIX_ADD; end
                PREFIX_ADD: if (add_valid && add_ready) state <= PREFIX_WAIT;
                PREFIX_WAIT: if (add_result_valid) begin
                    if (prefix_is_a4) begin
                        a4_priority_sum_bf16 <= add_result_bf16; candidate_priority_sum_bf16 <= add_result_bf16;
                        trial_a4_count <= trial_a4_count + 6'd1; trial_a8_count <= '0;
                        tentative4 <= tentative4 + {5'd0, selected_candidate.tentative}; tentative8 <= '0;
                    end else begin
                        candidate_priority_sum_bf16 <= add_result_bf16; trial_a8_count <= trial_a8_count + 6'd1;
                        tentative8 <= tentative8 + {5'd0, selected_candidate.tentative};
                    end
                    state <= CANDIDATE_CHECK;
                end
                RESULT_FILTER: begin
                    // Reject the winning optional group as a whole. Tentative
                    // rows are prefix-first and remain, as does Source A.
                    if (source_b_a4_only && selected_a8_count > selected_tentative8) begin
                        selected_a4_count <= selected_tentative4;
                        selected_a8_count <= selected_tentative8;
                        joint_activation_slots <= base_activation_slots +
                            {2'd0, selected_tentative4} + {1'b0, selected_tentative8, 1'b0};
                        next_pass_activation_slots <= upgraded_forecast_units +
                            {1'b0, selected_tentative4, 1'b0} + {1'b0, selected_tentative8, 1'b0};
                    end
                    state <= RESULT_BEGIN;
                end
                RESULT_BEGIN: begin
                    result_index <= '0; added_future_token_count <= selected_a4_count + selected_a8_count;
                    future_prediction_count <= future_prediction_count + selected_a4_count + selected_a8_count;
                    state <= RESULT_SCAN;
                end
                RESULT_SCAN: begin
                    if (result_index < selected_a4_count) begin
                        added_future_token_mask[a4_order[result_index[4:0]]] <= 1'b1; future_prediction_mask[a4_order[result_index[4:0]]] <= 1'b1;
                    end
                    if (result_index < selected_a8_count) begin
                        added_future_token_mask[a8_order[result_index[4:0]]] <= 1'b1; future_prediction_mask[a8_order[result_index[4:0]]] <= 1'b1;
                    end
                    result_index <= result_index + 6'd1;
                    if (result_index >= selected_a4_count && result_index >= selected_a8_count) state <= COMPLETE;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: if (!add_outstanding || add_abort_ack) begin abort_ack <= 1'b1; state <= ABORT_LOW; end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (add_result_valid)
                assert (add_outstanding) else $error("joint priority response has no pending addition");
            if (base_valid && base_ready && base_index == load_index)
                for (integer previous = 0; previous < 48; previous++)
                    if (previous < load_index)
                        assert (base_positions[previous][10:0] != base_position)
                            else $error("joint base contains duplicate logical position");
        end
    end
`endif
endmodule

`default_nettype wire
