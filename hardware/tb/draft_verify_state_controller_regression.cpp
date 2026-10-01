#include "Vdraft_verify_state_controller_top.h"
#include "verilated.h"
#include "json.hpp"

extern "C" {
#include "../cmodel/forward_postprocess_model.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint8_t MASKED = 0;
constexpr std::uint8_t TENTATIVE = 1;
constexpr std::uint8_t LOCKED = 2;
constexpr std::uint8_t ORIGIN_NONE = 0;
constexpr std::uint8_t ORIGIN_HIGH = 1;
constexpr std::uint8_t ORIGIN_STABLE = 2;
constexpr std::uint8_t ORIGIN_FALLBACK = 3;
constexpr std::uint32_t INVALID_TOKEN = 0xffffffffu;

struct Config {
    std::uint32_t masked = 0;
    std::uint32_t tentative = 0;
    std::uint32_t locked = 0;
    std::uint32_t mask_token = 99;
    std::uint32_t vocabulary = 128;
    std::uint16_t high_confidence_threshold = rtl_f32_to_bf16(0.90f);
    std::uint16_t tau_tail = rtl_f32_to_bf16(0.75f);
    std::uint16_t low_confidence_threshold = rtl_f32_to_bf16(0.75f);
    std::uint16_t verify_threshold = rtl_f32_to_bf16(0.75f);
    std::uint16_t bonus = rtl_f32_to_bf16(0.05f);
    std::uint16_t budget = rtl_f32_to_bf16(1.0f);
    std::uint16_t scheduled = 1;
    std::uint16_t remaining = 1;
    std::uint16_t step = 0;
    std::uint16_t tail_after = 8;
    std::uint16_t maturity = 3;
    bool tail_enable = true;
    bool tail_all = false;
    bool tail_bypass_all = false;
    bool tail_bypass_stable_only = false;
    unsigned closeout_kind = 0;
    bool canonical_future = false;
    unsigned max_handoff = 0;
    bool source_a_handoff = false;
    bool transfer_only = false;
    std::uint32_t observed = 0;
};

struct Row {
    std::uint32_t token = 0;
    std::uint32_t last_top1 = INVALID_TOKEN;
    std::int16_t age = -1;
    std::uint16_t logical = 0;
    std::uint8_t state = MASKED;
    std::uint8_t origin = ORIGIN_NONE;
    std::uint8_t bits = 4;
    std::uint32_t top1 = 0;
    std::uint16_t selected_probability = 0;
    std::uint16_t action_confidence = 0;
    bool suppressed = false;
    bool cache_valid = true;
    bool refresh_required = false;
    bool prediction_flag = true;
    bool action_confidence_valid = false;
    bool source_a = false, source_a_pending = false;
};

struct Result {
    std::vector<Row> rows;
    bool error = false;
    std::uint8_t error_id = 0;
    std::uint32_t confirmed = 0;
    std::uint32_t remasked = 0;
    std::uint32_t selected = 0;
    std::uint32_t direct = 0;
    std::uint32_t stable = 0;
    std::uint32_t fallback = 0;
    std::uint32_t mandatory = 0;
    std::uint32_t cache_commit = 0;
    std::uint32_t cache_invalidate = 0;
    std::uint32_t cache_keep = 0;
    std::uint32_t token_changed = 0;
    std::uint32_t tail_closed = 0;
};

Result c11_result(const Config& cfg, const std::vector<Row>& input) {
    Result result;
    const std::size_t count = input.size();
    std::vector<forward_postprocess_row_state> c_rows(count);
    std::vector<forward_postprocess_candidate_result> candidates(count);
    for (std::size_t index = 0; index < count; ++index) {
        const Row& row = input[index];
        c_rows[index] = forward_postprocess_row_state{
            row.token, row.last_top1, row.age, row.logical, row.state,
            row.origin, row.bits, static_cast<std::uint8_t>(row.cache_valid),
            static_cast<std::uint8_t>(row.refresh_required),
            static_cast<std::uint8_t>(row.prediction_flag),
            static_cast<std::uint8_t>(row.source_a), static_cast<std::uint8_t>(row.source_a_pending)};
        candidates[index] = forward_postprocess_candidate_result{
            row.top1, 0, 0, 0, row.selected_probability,
            row.action_confidence, static_cast<std::uint8_t>(row.suppressed), 0};
    }
    if (cfg.tail_all) {
        std::uint32_t closed = 0;
        const int status = forward_postprocess_draft_verify_tail_all(
            c_rows.data(), count, &closed);
        if (status != 0) {
            result.error = true;
            result.error_id = status == -2 ? 1 : 2;
            return result;
        }
        result.tail_closed = closed;
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t bit = std::uint32_t{1} << index;
            if (c_rows[index].cache_valid &&
                !c_rows[index].refresh_required)
                result.cache_keep |= bit;
            if (c_rows[index].refresh_required) result.mandatory |= bit;
        }
    } else {
        forward_postprocess_draft_verify_config config{};
        config.mask_token_id = cfg.mask_token;
        config.high_confidence_threshold_bf16 = cfg.high_confidence_threshold;
        config.tail_high_confidence_threshold_bf16 = cfg.tau_tail;
        config.low_confidence_threshold_bf16 = cfg.low_confidence_threshold;
        config.verify_threshold_bf16 = cfg.verify_threshold;
        config.stability_bonus_bf16 = cfg.bonus;
        config.budget_scale_bf16 = cfg.budget;
        config.scheduled_quota = cfg.scheduled;
        config.remaining_forwards = cfg.remaining;
        config.step_index = cfg.step;
        config.tail_after_step = cfg.tail_after;
        config.maturity_age = cfg.maturity;
        config.tail_threshold_enable = cfg.tail_enable;
        config.tail_bypass_all = cfg.tail_bypass_all;
        config.tail_bypass_stable_only = cfg.tail_bypass_stable_only;
        config.source_a_handoff = cfg.source_a_handoff;
        forward_postprocess_draft_verify_result c_result{};
        const int status = cfg.transfer_only ? forward_postprocess_transfer_update(
            c_rows.data(), count, cfg.masked, cfg.tentative, cfg.locked,
            candidates.data(), &config, &c_result) : cfg.closeout_kind ? forward_postprocess_draft_verify_closeout(
            c_rows.data(), count, cfg.closeout_kind, candidates.data(), &config, &c_result) :
            cfg.canonical_future ? forward_postprocess_canonical_future_update(
            c_rows.data(), count, cfg.masked, cfg.tentative, cfg.locked, cfg.observed,
            candidates.data(), &config, &c_result) : forward_postprocess_draft_verify_update(
            c_rows.data(), count, cfg.masked, cfg.tentative, cfg.locked,
            candidates.data(), &config, &c_result);
        if (status != 0)
            throw std::runtime_error("C11 PSME update rejected a valid focused case");
        result.confirmed = c_result.confirmed_mask;
        result.remasked = c_result.remasked_mask;
        result.selected = c_result.selected_mask;
        result.direct = c_result.direct_locked_mask;
        result.stable = c_result.stable_tentative_mask;
        result.fallback = c_result.fallback_tentative_mask;
        result.mandatory = c_result.mandatory_refresh_mask;
        result.cache_commit = c_result.cache_commit_mask;
        result.cache_invalidate = c_result.cache_invalidate_mask;
        result.cache_keep = c_result.cache_keep_mask;
        result.token_changed = c_result.token_changed_mask;
        result.tail_closed = c_result.tail_closed_mask;
    }
    result.rows.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& row = c_rows[index];
        result.rows[index] = Row{
            row.token_id, row.last_top1, row.precision_age,
            row.token_position, row.state, row.commit_origin, row.activation_bits,
            input[index].top1, input[index].selected_probability,
            input[index].action_confidence, input[index].suppressed,
            row.cache_valid != 0, row.refresh_required != 0,
            row.prediction_flag != 0, input[index].action_confidence_valid,
            input[index].source_a, row.source_a_pending != 0};
    }
    return result;
}

template <std::size_t Words>
std::uint32_t get_bits(const VlWide<Words>& value, unsigned lsb, unsigned width) {
    std::uint64_t combined = value[lsb / 32u] >> (lsb % 32u);
    if ((lsb % 32u) + width > 32u)
        combined |= std::uint64_t{value[lsb / 32u + 1u]} << (32u - lsb % 32u);
    const std::uint64_t mask = width == 32 ? 0xffffffffull :
        (std::uint64_t{1} << width) - 1u;
    return static_cast<std::uint32_t>(combined & mask);
}

template <std::size_t Words>
void set_bits(VlWide<Words>& value, unsigned lsb, unsigned width,
              std::uint32_t payload) {
    const unsigned word = lsb / 32u;
    const unsigned offset = lsb % 32u;
    const std::uint64_t mask = width == 32 ? 0xffffffffull :
        (std::uint64_t{1} << width) - 1u;
    const std::uint64_t shifted_mask = mask << offset;
    std::uint64_t combined = value[word];
    if (offset + width > 32u) combined |= std::uint64_t{value[word + 1u]} << 32u;
    combined = (combined & ~shifted_mask) |
        ((std::uint64_t{payload} & mask) << offset);
    value[word] = static_cast<std::uint32_t>(combined);
    if (offset + width > 32u) value[word + 1u] = static_cast<std::uint32_t>(combined >> 32u);
}

class Simulation {
  public:
    struct Events {
        bool start = false;
        bool row = false;
        bool next = false;
        bool done = false;
        bool abort_ack = false;
        bool bf16_request = false;
        Row next_value{};
        std::uint8_t next_index = 0;
        Result summary{};
    };

    explicit Simulation(std::uint32_t seed) : random_(seed) {
        context_.threads(1);
        dut_ = new Vdraft_verify_state_controller_top(&context_);
        clear_inputs();
        dut_->rst = 1;
        for (int cycle = 0; cycle < 4; ++cycle) step();
        dut_->rst = 0;
        step();
        if (dut_->threads() != 1) fail("focused generated model did not report one runtime thread");
    }

    ~Simulation() {
        dut_->final();
        delete dut_;
    }

    void set_minimum_response_delay(unsigned value) { minimum_response_delay_ = value; }

    Events step() {
        drive_bf16();
        dut_->clk = 0;
        dut_->eval();

        Events events;
        events.start = dut_->start_valid && dut_->start_ready;
        events.row = dut_->row_valid && dut_->row_ready;
        events.next = dut_->next_token_valid && dut_->next_token_ready;
        events.done = dut_->done_valid && dut_->done_ready;
        events.abort_ack = dut_->abort_ack;
        events.bf16_request = dut_->bf16_req_valid && dut_->bf16_req_ready;
        if (events.next) {
            events.next_index = dut_->next_token_index;
            events.next_value.token = dut_->next_token_token_id;
            events.next_value.last_top1 = dut_->next_token_last_top1;
            events.next_value.age = static_cast<std::int16_t>(dut_->next_token_precision_age);
            events.next_value.logical = dut_->next_token_token_position;
            events.next_value.state = dut_->next_token_state;
            events.next_value.origin = dut_->next_token_origin;
            events.next_value.bits = dut_->next_activation_bits;
            events.next_value.cache_valid = dut_->next_token_cache_valid;
            events.next_value.refresh_required =
                dut_->next_token_refresh_required;
            events.next_value.prediction_flag = dut_->next_token_prediction_flag;
            events.next_value.source_a_pending = dut_->next_token_source_a_pending;
            events.next_value.action_confidence = dut_->next_token_action_confidence_bf16;
            events.next_value.action_confidence_valid = dut_->next_token_action_confidence_valid;
        }
        if (events.done) capture_summary(events.summary);

        const bool request_fire = dut_->bf16_req_valid && dut_->bf16_req_ready;
        const bool response_fire = dut_->bf16_rsp_valid && dut_->bf16_rsp_ready;
        const bool shared_abort = dut_->bf16_abort_request && dut_->bf16_abort_ack;
        Pending accepted{};
        if (request_fire) accepted = make_response();

        dut_->clk = 1;
        dut_->eval();
        context_.timeInc(1);
        dut_->clk = 0;
        dut_->eval();
        context_.timeInc(1);

        if (response_fire) pending_.valid = false;
        if (request_fire) {
            if (pending_.valid) fail("BF16 request overlapped a pending response");
            pending_ = accepted;
        }
        if (shared_abort) pending_.valid = false;
        if (pending_.valid && pending_.delay != 0) --pending_.delay;
        ++cycles_;
        if (cycles_ > 200000) fail("simulation made no bounded progress");
        return events;
    }

    Vdraft_verify_state_controller_top& dut() { return *dut_; }

  private:
    struct Pending {
        bool valid = false;
        unsigned delay = 0;
        std::uint16_t tag = 0;
        std::uint16_t value = 0;
    };

    void clear_inputs() {
        dut_->clk = 0;
        dut_->rst = 0;
        dut_->abort_request = 0;
        dut_->start_valid = 0;
        dut_->row_valid = 0;
        dut_->row_cache_valid = 0;
        dut_->row_refresh_required = 0;
        dut_->row_prediction_flag = 0;
        dut_->bf16_req_ready = 0;
        dut_->bf16_rsp_valid = 0;
        dut_->bf16_abort_ack = 0;
        dut_->next_token_ready = 0;
        dut_->done_ready = 1;
        for (unsigned word = 0; word < 35; ++word) dut_->bf16_rsp[word] = 0;
    }

    void drive_bf16() {
        dut_->bf16_req_ready = (random_() & 3u) != 0;
        dut_->bf16_abort_ack = dut_->bf16_abort_request;
        dut_->bf16_rsp_valid = pending_.valid && pending_.delay == 0;
        for (unsigned word = 0; word < 35; ++word) dut_->bf16_rsp[word] = 0;
        if (dut_->bf16_rsp_valid) {
            set_bits(dut_->bf16_rsp, 0, 16, pending_.tag);
            set_bits(dut_->bf16_rsp, 16, 1, 1);
            set_bits(dut_->bf16_rsp, 80, 16, pending_.value);
        }
    }

    Pending make_response() {
        const auto& request = dut_->bf16_req;
        const std::uint16_t tag = get_bits(request, 0, 16);
        const std::uint16_t value = get_bits(request, 3152, 16);
        const std::uint16_t paired = get_bits(request, 2128, 16);
        const std::uint16_t factor0 = get_bits(request, 1104, 16);
        const std::uint8_t operation = get_bits(request, 4176, 3);
        if (get_bits(request, 16, 1) != 1) fail("BF16 lane zero was not enabled");
        std::uint16_t response = 0;
        if (operation == 0)
            response = rtl_bf16_add(value, paired);
        else if (operation == 1)
            response = rtl_bf16_mul(value, factor0);
        else
            fail("unexpected BF16 operation");
        return Pending{true, minimum_response_delay_ +
            static_cast<unsigned>(random_() & 3u), tag, response};
    }

    void capture_summary(Result& result) {
        result.error = dut_->error;
        result.error_id = dut_->error_id;
        result.confirmed = dut_->confirmed_mask;
        result.remasked = dut_->remasked_mask;
        result.selected = dut_->selected_mask;
        result.direct = dut_->direct_locked_mask;
        result.stable = dut_->stable_tentative_mask;
        result.fallback = dut_->fallback_tentative_mask;
        result.mandatory = dut_->mandatory_refresh_mask;
        result.cache_commit = dut_->cache_commit_mask;
        result.cache_invalidate = dut_->cache_invalidate_mask;
        result.cache_keep = dut_->cache_keep_mask;
        result.token_changed = dut_->token_changed_mask;
        result.tail_closed = dut_->tail_closed_mask;
    }

    [[noreturn]] static void fail(const std::string& message) {
        throw std::runtime_error(message);
    }

    VerilatedContext context_;
    Vdraft_verify_state_controller_top* dut_ = nullptr;
    std::mt19937 random_;
    Pending pending_{};
    unsigned minimum_response_delay_ = 0;
    std::uint64_t cycles_ = 0;
};

void drive_config(Vdraft_verify_state_controller_top& dut, const Config& cfg,
                  std::size_t rows) {
    dut.start_row_count = rows;
    dut.start_masked_mask = cfg.masked;
    dut.start_tentative_mask = cfg.tentative;
    dut.start_locked_mask = cfg.locked;
    dut.start_mask_token_id = cfg.mask_token;
    dut.start_vocabulary_size = cfg.vocabulary;
    dut.start_high_confidence_threshold_bf16 = cfg.high_confidence_threshold;
    dut.start_tail_high_confidence_threshold_bf16 = cfg.tau_tail;
    dut.start_low_confidence_threshold_bf16 = cfg.low_confidence_threshold;
    dut.start_verify_threshold_bf16 = cfg.verify_threshold;
    dut.start_stability_bonus_bf16 = cfg.bonus;
    dut.start_budget_scale_bf16 = cfg.budget;
    dut.start_scheduled_quota = cfg.scheduled;
    dut.start_max_handoff_tokens = cfg.max_handoff;
    dut.start_remaining_forwards = cfg.remaining;
    dut.start_step_index = cfg.step;
    dut.start_tail_after_step = cfg.tail_after;
    dut.start_maturity_age = cfg.maturity;
    dut.start_tail_threshold_enable = cfg.tail_enable;
    dut.start_tail_all = cfg.tail_all;
    dut.start_tail_bypass_all = cfg.tail_bypass_all;
    dut.start_tail_bypass_stable_only = cfg.tail_bypass_stable_only;
    dut.start_closeout_kind = cfg.closeout_kind;
    dut.start_canonical_future = cfg.canonical_future;
    dut.start_source_a_handoff = cfg.source_a_handoff;
    dut.start_transfer_only = cfg.transfer_only;
    dut.start_observed_mask = cfg.observed;
}

void drive_row(Vdraft_verify_state_controller_top& dut, const Row& row,
               std::size_t index) {
    dut.row_index = index;
    dut.row_token_id = row.token;
    dut.row_last_top1 = row.last_top1;
    dut.row_precision_age = static_cast<std::uint16_t>(row.age);
    dut.row_token_position = row.logical;
    dut.row_origin = row.origin;
    dut.activation_bits = row.bits;
    dut.row_cache_valid = row.cache_valid;
    dut.row_refresh_required = row.refresh_required;
    dut.row_prediction_flag = row.prediction_flag;
    dut.row_candidate_top1 = row.top1;
    dut.row_selected_probability_bf16 = row.selected_probability;
    dut.row_action_confidence_bf16 = row.action_confidence;
    dut.row_action_confidence_valid = 1;
    dut.row_source_a = row.source_a;
    dut.row_source_a_pending = row.source_a_pending;
    dut.row_suppressed_winner = row.suppressed;
}

Result run_case(Simulation& simulation, const Config& cfg,
                const std::vector<Row>& rows, std::uint32_t ready_seed = 1) {
    auto& dut = simulation.dut();
    drive_config(dut, cfg, rows.size());
    dut.start_valid = 1;
    while (true) {
        const auto event = simulation.step();
        if (event.start) break;
    }
    dut.start_valid = 0;

    for (std::size_t index = 0; index < rows.size(); ++index) {
        drive_row(dut, rows[index], index);
        dut.row_valid = 1;
        while (true) {
            const auto event = simulation.step();
            if (event.row) break;
            if (event.done) {
                dut.row_valid = 0;
                return event.summary;
            }
        }
        dut.row_valid = 0;
    }

    std::mt19937 ready_random(ready_seed);
    Result result;
    result.rows.resize(rows.size());
    std::size_t output_count = 0;
    while (true) {
        dut.next_token_ready = (ready_random() & 3u) != 0;
        const auto event = simulation.step();
        if (event.next) {
            if (event.next_index != output_count || output_count >= rows.size())
                throw std::runtime_error("next-token stream order mismatch");
            if (!event.next_value.action_confidence_valid || event.next_value.action_confidence != rows[output_count].action_confidence)
                throw std::runtime_error("latest action confidence changed during state update");
            result.rows[output_count++] = event.next_value;
        }
        if (event.done) {
            const auto saved_rows = result.rows;
            result = event.summary;
            result.rows = saved_rows;
            break;
        }
    }
    dut.next_token_ready = 0;
    if (!result.error && output_count != rows.size())
        throw std::runtime_error("next-token stream ended early");
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void compare_result(const std::string& name, const Result& actual,
                    const Result& expected) {
    require(actual.error == expected.error, name + ": error mismatch");
    if (expected.error) return;
    require(actual.rows.size() == expected.rows.size(), name + ": row count mismatch");
    for (std::size_t index = 0; index < actual.rows.size(); ++index) {
        const Row& lhs = actual.rows[index];
        const Row& rhs = expected.rows[index];
        require(lhs.token == rhs.token, name + ": token mismatch row " + std::to_string(index));
        require(lhs.last_top1 == rhs.last_top1,
                name + ": last_top1 mismatch row " + std::to_string(index));
        require(lhs.age == rhs.age, name + ": age mismatch row " + std::to_string(index));
        require(lhs.logical == rhs.logical,
                name + ": logical position mismatch row " + std::to_string(index));
        require(lhs.state == rhs.state, name + ": state mismatch row " + std::to_string(index));
        require(lhs.origin == rhs.origin, name + ": origin mismatch row " + std::to_string(index));
        require(lhs.bits == rhs.bits, name + ": row bits mismatch row " + std::to_string(index));
        require(lhs.cache_valid == rhs.cache_valid,
                name + ": cache valid mismatch row " + std::to_string(index));
        require(lhs.refresh_required == rhs.refresh_required,
                name + ": refresh-required mismatch row " + std::to_string(index));
        require(lhs.source_a_pending == rhs.source_a_pending, name + ": Source A pending mismatch");
        require(lhs.prediction_flag == rhs.prediction_flag,
                name + ": prediction flag mismatch row " + std::to_string(index));
    }
    require(actual.confirmed == expected.confirmed, name + ": confirmed mask mismatch");
    require(actual.remasked == expected.remasked, name + ": remasked mask mismatch");
    require(actual.selected == expected.selected, name + ": selected mask mismatch");
    require(actual.direct == expected.direct, name + ": direct mask mismatch");
    require(actual.stable == expected.stable, name + ": stable mask mismatch");
    require(actual.fallback == expected.fallback, name + ": fallback mask mismatch");
    require(actual.mandatory == expected.mandatory, name + ": mandatory mask mismatch");
    require(actual.cache_commit == expected.cache_commit, name + ": cache commit mask mismatch");
    require(actual.cache_invalidate == expected.cache_invalidate,
            name + ": cache invalidate mask mismatch");
    require(actual.cache_keep == expected.cache_keep, name + ": cache keep mask mismatch");
    require(actual.token_changed == expected.token_changed, name + ": token changed mask mismatch");
    require(actual.tail_closed == expected.tail_closed, name + ": tail-closed mask mismatch");
}

Row masked_row(std::uint16_t logical, std::uint32_t top1, float confidence,
               std::uint8_t bits = 8) {
    Row row;
    row.token = 99;
    row.logical = logical;
    row.state = MASKED;
    row.bits = bits;
    row.top1 = top1;
    row.selected_probability = rtl_f32_to_bf16(0.01f);
    row.action_confidence = rtl_f32_to_bf16(confidence);
    return row;
}

void test_reference_case(Simulation& simulation) {
    Config cfg;
    cfg.masked = 0x6;
    cfg.tentative = 0x1;
    cfg.locked = 0x8;
    cfg.scheduled = 2;
    cfg.remaining = 2;
    std::vector<Row> rows(4);
    rows[0] = Row{10, INVALID_TOKEN, -1, 0, TENTATIVE, ORIGIN_STABLE, 8,
                  10, rtl_f32_to_bf16(0.80f), rtl_f32_to_bf16(0.80f), false};
    rows[1] = masked_row(1, 21, 0.95f, 8);
    rows[1].last_top1 = 20;
    rows[2] = masked_row(2, 31, 0.95f, 4);
    rows[2].last_top1 = 31;
    rows[3] = Row{40, INVALID_TOKEN, 1, 3, LOCKED, ORIGIN_HIGH, 8,
                  41, 0, 0, false};
    rows[3].cache_valid = false;
    rows[3].refresh_required = true;
    rows[3].prediction_flag = false;
    compare_result("reference", run_case(simulation, cfg, rows, 11),
                   c11_result(cfg, rows));
}

void test_quota_tie_tail(Simulation& simulation) {
    Config cfg;
    cfg.masked = 0x1f;
    cfg.scheduled = 3;
    cfg.remaining = 2;
    cfg.budget = rtl_f32_to_bf16(1.3359375f);
    std::vector<Row> rows;
    for (std::size_t index = 0; index < 5; ++index) {
        rows.push_back(masked_row(static_cast<std::uint16_t>(20-index),
                                  30+index, 0.90f, 8));
    }
    const Result expected = c11_result(cfg, rows);
    require(__builtin_popcount(expected.selected) == 4,
            "quota expected did not select four rows");
    compare_result("quota-tie", run_case(simulation, cfg, rows, 17), expected);

    Config tail;
    tail.masked = 0xf;
    tail.scheduled = 3;
    tail.remaining = 4;
    tail.step = 8;
    tail.tail_after = 8;
    std::vector<Row> tail_rows;
    tail_rows.push_back(masked_row(0, 20, 0.80f, 8));
    tail_rows.push_back(masked_row(1, 21, 0.80f, 8));
    tail_rows.back().last_top1 = 21;
    tail_rows.push_back(masked_row(2, 22, 0.95f, 4));
    tail_rows.push_back(masked_row(3, 23, 0.10f, 8));
    compare_result("tail-a4-a8", run_case(simulation, tail, tail_rows, 23),
                   c11_result(tail, tail_rows));
}

void test_a4_direct_boundary(Simulation& simulation) {
    Config cfg;
    cfg.masked = 0x7;
    cfg.scheduled = 3;
    cfg.remaining = 1;
    std::vector<Row> rows = {
        masked_row(0, 20, 0.0f, 4),
        masked_row(1, 21, 0.0f, 4),
        masked_row(2, 22, 0.0f, 4),
    };
    rows[0].action_confidence = 0x3f65;
    rows[1].action_confidence = 0x3f66;
    rows[2].action_confidence = 0x3f67;
    const Result expected = c11_result(cfg, rows);
    require(expected.selected == 0x7, "A4 boundary did not select all three rows");
    // Expected state transitions come from the algorithm generator.
    require(expected.direct == 0x6,
            "A4 direct threshold differs from GPU BF16 raw 0x3f66");
    require(expected.rows[0].state == TENTATIVE &&
            expected.rows[1].state == LOCKED && expected.rows[2].state == LOCKED,
            "A4 boundary state transition mismatch");
    require(expected.rows[1].bits == 8 &&
            expected.rows[1].refresh_required &&
            !expected.rows[1].cache_valid,
            "A4 direct row did not require the next A8 cache refresh");
    compare_result("a4-direct-boundary",
                   run_case(simulation, cfg, rows, 29), expected);
}

void test_full_rows_and_three_forwards(Simulation& simulation) {
    Config full;
    full.masked = 0xffffffffu;
    full.scheduled = 4;
    full.remaining = 4;
    full.budget = rtl_f32_to_bf16(1.0f);
    std::vector<Row> rows;
    for (std::size_t index = 0; index < 32; ++index) {
        rows.push_back(masked_row(static_cast<std::uint16_t>(100 + (31-index)),
                                  40+index, index < 1 ? 0.95f : 0.20f,
                                  index & 1 ? 8 : 4));
    }
    rows[1].cache_valid = false;
    rows[1].refresh_required = true;
    compare_result("full-32", run_case(simulation, full, rows, 31),
                   c11_result(full, rows));

    Config cfg;
    cfg.masked = 0xff;
    cfg.scheduled = 2;
    cfg.remaining = 3;
    std::vector<Row> chain;
    for (std::size_t index = 0; index < 8; ++index)
        chain.push_back(masked_row(index, 60+index, 0.95f, index < 4 ? 8 : 4));
    for (unsigned forward = 0; forward < 3; ++forward) {
        const Result expected = c11_result(cfg, chain);
        const Result actual = run_case(simulation, cfg, chain, 41+forward);
        compare_result("three-forward-" + std::to_string(forward), actual, expected);
        chain = expected.rows;
        cfg.masked = cfg.tentative = cfg.locked = 0;
        for (std::size_t index = 0; index < chain.size(); ++index) {
            const std::uint32_t bit = std::uint32_t{1} << index;
            if (chain[index].state == MASKED) cfg.masked |= bit;
            else if (chain[index].state == TENTATIVE) cfg.tentative |= bit;
            else cfg.locked |= bit;
            if (chain[index].state == TENTATIVE) {
                chain[index].top1 = chain[index].token;
                chain[index].selected_probability = rtl_f32_to_bf16(0.90f);
            } else if (chain[index].state == MASKED) {
                chain[index].top1 = 70 + forward*8 + index;
                chain[index].action_confidence = rtl_f32_to_bf16(0.95f);
            }
        }
        cfg.remaining = static_cast<std::uint16_t>(std::max(1u, 2u-forward));
    }
}

void test_tail_all_and_errors(Simulation& simulation) {
    Config tail;
    tail.masked = 0;
    tail.tentative = 0x3;
    tail.locked = 0x4;
    tail.tail_all = true;
    std::vector<Row> rows(3);
    rows[0] = Row{10, INVALID_TOKEN, -1, 0, TENTATIVE, ORIGIN_STABLE, 8,
                  10, 0, 0, false};
    rows[1] = Row{11, INVALID_TOKEN, -1, 1, TENTATIVE, ORIGIN_FALLBACK, 8,
                  11, 0, 0, false};
    rows[2] = Row{12, INVALID_TOKEN, 3, 2, LOCKED, ORIGIN_HIGH, 4,
                  12, 0, 0, false};
    rows[2].prediction_flag = false;
    rows[0].cache_valid = false;
    rows[0].refresh_required = true;
    compare_result("tail-all", run_case(simulation, tail, rows, 53),
                   c11_result(tail, rows));

    Config invalid_tail = tail;
    invalid_tail.masked = 1;
    invalid_tail.tentative = 2;
    const Result bad_tail = run_case(simulation, invalid_tail, rows, 59);
    require(bad_tail.error && bad_tail.error_id == 1,
            "tail-all with MASKED position was not rejected");

    Config duplicate;
    duplicate.masked = 0x3;
    duplicate.remaining = 2;
    std::vector<Row> duplicate_rows = {
        masked_row(7, 20, 0.8f, 8), masked_row(7, 21, 0.8f, 8)};
    const Result duplicate_result = run_case(simulation, duplicate, duplicate_rows, 61);
    require(duplicate_result.error && duplicate_result.error_id == 3,
            "duplicate logical position was not rejected");

    Config invalid_row;
    invalid_row.masked = 1;
    Row bad_row = masked_row(0, 20, 0.8f, 8);
    bad_row.token = 7;
    const Result bad_row_result = run_case(simulation, invalid_row, {bad_row}, 67);
    require(bad_row_result.error && bad_row_result.error_id == 2,
            "invalid MASKED token was not rejected");

    Row bad_prediction = masked_row(0, 20, 0.8f, 8);
    bad_prediction.prediction_flag = false;
    const Result bad_prediction_result = run_case(
        simulation, invalid_row, {bad_prediction}, 68);
    require(bad_prediction_result.error && bad_prediction_result.error_id == 2,
            "prediction flag inconsistent with start state was not rejected");

    Config suppressed;
    suppressed.tentative = 1;
    Row suppressed_row{10, INVALID_TOKEN, -1, 0, TENTATIVE, ORIGIN_STABLE, 8,
                       10, rtl_f32_to_bf16(0.99f), rtl_f32_to_bf16(0.99f), true};
    compare_result("suppressed-confirmation",
                   run_case(simulation, suppressed, {suppressed_row}, 69),
                   c11_result(suppressed, {suppressed_row}));

    Config saturated;
    saturated.locked = 1;
    Row saturated_row{12, INVALID_TOKEN, std::numeric_limits<std::int16_t>::max(),
                      0, LOCKED, ORIGIN_HIGH, 4, 12, 0, 0, false};
    saturated_row.cache_valid = false;
    saturated_row.prediction_flag = false;
    compare_result("age-saturation",
                   run_case(simulation, saturated, {saturated_row}, 70),
                   c11_result(saturated, {saturated_row}));

    Config maturity;
    maturity.locked = 1;
    Row maturity_row{13, INVALID_TOKEN, 2, 0, LOCKED, ORIGIN_HIGH, 8,
                     13, 0, 0, false};
    maturity_row.prediction_flag = false;
    compare_result("maturity-row-bits",
                   run_case(simulation, maturity, {maturity_row}, 72),
                   c11_result(maturity, {maturity_row}));
}

void test_packed_context_precision(Simulation& simulation) {
    Config cfg;
    cfg.canonical_future = true;
    cfg.tail_enable = false;
    cfg.tentative = 1;
    cfg.locked = 6;
    cfg.observed = 7;
    std::vector<Row> rows = {
        Row{10, INVALID_TOKEN, -1, 32, TENTATIVE, ORIGIN_STABLE, 4,
            10, rtl_f32_to_bf16(0.90f), rtl_f32_to_bf16(0.90f), false},
        Row{11, INVALID_TOKEN, 0, 33, LOCKED, ORIGIN_HIGH, 4,
            11, 0, 0, false},
        Row{12, INVALID_TOKEN, 3, 34, LOCKED, ORIGIN_HIGH, 8,
            12, 0, 0, false}};
    rows[1].prediction_flag = rows[2].prediction_flag = false;
    const auto expected = c11_result(cfg, rows);
    require(expected.confirmed == 1 && expected.rows[0].bits == 8 &&
        expected.rows[1].bits == 8 && expected.rows[2].bits == 4,
        "packed precision case did not cross stored/executed precision");
    require(expected.cache_invalidate == 0 && expected.mandatory == 0,
        "precision allocation alone invalidated unchanged-token cache");
    compare_result("packed-context-executed-precision",
        run_case(simulation, cfg, rows, 73), expected);
}

void test_normal_tail_bypass(Simulation& simulation) {
    Config config; config.masked = 3; config.remaining = 1; config.tail_bypass_all = true;
    std::vector<Row> rows{masked_row(0, 10, 0.1f, 4), masked_row(1, 11, 0.1f, 8)};
    auto expected = c11_result(config, rows);
    require(expected.tail_closed == 3 && expected.fallback == 3 && expected.cache_commit == 0 &&
        expected.cache_invalidate == 3 && expected.mandatory == 3,
        "tail bypass must preserve invalid cache after token admission");
    compare_result("normal-forward-tail-bypass", run_case(simulation, config, rows, 81), expected);
    config.remaining = 2;
    expected = c11_result(config, rows);
    require(expected.tail_closed == 0, "tail bypass closed an incomplete block");
    compare_result("normal-forward-masked-remains", run_case(simulation, config, rows, 82), expected);
}

void test_stable_tail_and_budget(Simulation& simulation) {
    Config cfg;
    cfg.masked = 3; cfg.remaining = 1; cfg.tail_enable = false;
    cfg.tail_bypass_stable_only = true;
    std::vector<Row> stable{masked_row(0, 10, .8f, 4), masked_row(1, 11, .8f, 4)};
    stable[0].last_top1 = 10; stable[1].last_top1 = 11;
    auto expected = c11_result(cfg, stable);
    require(expected.tail_closed == 3 && expected.stable == 3, "stable-only tail failed to close stable block");
    compare_result("stable-only-complete", run_case(simulation, cfg, stable, 91), expected);
    for (unsigned index = 0; index < 2; ++index) {
        auto mixed = stable;
        mixed[index] = masked_row(index, 10 + index, .1f, 4);
        expected = c11_result(cfg, mixed);
        require(expected.tail_closed == 0 && expected.fallback == (1u << index), "one fallback must veto the entire stable tail");
        compare_result("stable-only-fallback-veto", run_case(simulation, cfg, mixed, 92 + index), expected);
    }
    cfg.remaining = 2;
    auto incomplete = stable;
    incomplete[1] = masked_row(1, 11, .1f, 4);
    expected = c11_result(cfg, incomplete);
    require(expected.tail_closed == 0 && expected.rows[1].state == MASKED, "masked token must veto stable tail");
    compare_result("stable-only-masked-veto", run_case(simulation, cfg, incomplete, 94), expected);
    cfg.tail_bypass_stable_only = false;
    cfg.remaining = 32; cfg.masked = 0xffffffffu;
    cfg.budget = rtl_f32_to_bf16(21.f); // converter's exact encoding of algorithm scale 20.01
    std::vector<Row> rows;
    for (unsigned i = 0; i < 32; ++i) rows.push_back(masked_row(i, 20+i, .95f, 4));
    expected = c11_result(cfg, rows);
    require(expected.selected == 0x1fffffu, "20.01 must admit 21, tie by logical position");
    compare_result("fractional-budget-block32", run_case(simulation, cfg, rows, 95), expected);
}

void test_abort() {
    Simulation simulation(71);
    simulation.set_minimum_response_delay(40);
    Config cfg;
    cfg.masked = 1;
    Row row = masked_row(0, 20, 0.90f, 8);
    row.last_top1 = 20;
    auto& dut = simulation.dut();
    drive_config(dut, cfg, 1);
    dut.start_valid = 1;
    while (!simulation.step().start) {}
    dut.start_valid = 0;
    drive_row(dut, row, 0);
    dut.row_valid = 1;
    while (!simulation.step().row) {}
    dut.row_valid = 0;
    bool request_accepted = false;
    for (unsigned cycle = 0; cycle < 200 && !request_accepted; ++cycle)
        request_accepted = simulation.step().bf16_request;
    require(request_accepted, "abort test did not accept BF16 work");
    dut.abort_request = 1;
    bool acknowledged = false;
    for (unsigned cycle = 0; cycle < 100 && !acknowledged; ++cycle)
        acknowledged = simulation.step().abort_ack;
    require(acknowledged, "abort was not acknowledged after BF16 drain");
    dut.abort_request = 0;
    for (unsigned cycle = 0; cycle < 20 && !dut.start_ready; ++cycle) simulation.step();
    require(dut.start_ready, "controller did not return to idle after abort");
}

}  // namespace

void test_actual_future_test_case(Simulation& simulation, const char* path) {
    std::ifstream input(path);
    require(input.good(), "missing canonical future test_case");
    nlohmann::json test_case;
    input >> test_case;
    std::size_t checked = 0;
    std::vector<Row> persisted_rows;
    std::size_t handoff_locked = 0;
    const auto& records = test_case.at("records");
    for (std::size_t index = 1; index+1 < records.size(); ++index) {
        const auto& record = records[index];
        if (record.at("event") != "admit" || record.at("token_storage_offset") != 33)
            continue;
        const auto& capture = test_case.at("captures").at(record.at("forward_capture").get<unsigned>());
        const bool future = capture.at("block_index") == 0;
        if (!future && !(capture.at("block_index") == 1 && capture.at("step_index") == 0)) continue;
        const auto& decision = records[index-1];
        const auto& end = records[index+1];
        require(decision.at("event") == "select" && end.at("event") == "age" &&
            end.at("state_id") == record.at("state_id"), "future selection/state event association failed");
        Config config;
        config.canonical_future = future; config.tail_enable = !future;
        config.mask_token = 3; config.vocabulary = 4;
        config.scheduled = decision.at("quota").at("raw").at(0);
        config.remaining = decision.at("parameters").at("remaining_forwards");
        for (auto position : capture.at("input_positions").at("raw")) {
            const auto absolute = position.get<unsigned>();
            if (absolute >= 33 && absolute < 65) config.observed |= 1u << (absolute-33);
        }
        std::vector<Row> rows = persisted_rows.empty() ? std::vector<Row>(32) : persisted_rows;
        std::uint32_t expected_selected = 0, expected_direct = 0, expected_stable = 0, expected_fallback = 0;
        for (unsigned row = 0; row < 32; ++row) {
            auto& value = rows[row];
            const auto& before = record.at("before");
            if (persisted_rows.empty()) {
                value.state = before.at("state").at("raw").at(0).at(row);
                value.origin = before.at("commit_origin").at("raw").at(0).at(row);
                value.age = before.at("precision_age").at("raw").at(0).at(row);
                value.last_top1 = before.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>();
                value.token = record.at("tokens_before").at("raw").at(0).at(row);
            } else {
                require(value.state == before.at("state").at("raw").at(0).at(row) &&
                    value.origin == before.at("commit_origin").at("raw").at(0).at(row) &&
                    value.age == before.at("precision_age").at("raw").at(0).at(row) &&
                    value.last_top1 == std::uint32_t(before.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>()) &&
                    value.token == record.at("tokens_before").at("raw").at(0).at(row),
                    "persisted RTL state differs from next generator input at update " + std::to_string(checked) + " row " + std::to_string(row));
            }
            if (!future && value.state == LOCKED) ++handoff_locked;
            value.logical = 33+row;
            value.bits = value.state == MASKED || (value.state == LOCKED && value.age >= 3) ? 4 : 8;
            value.prediction_flag = value.state != LOCKED && ((config.observed >> row)&1);
            if ((config.observed >> row)&1) { value.cache_valid = true; value.refresh_required = false; }
            value.top1 = record.at("proposal").at("raw").at(0).at(row).get<std::int64_t>();
            value.action_confidence = decision.at("confidence").at("raw").at(0).at(row);
            value.selected_probability = value.action_confidence;
            if (value.state == MASKED) config.masked |= 1u << row;
            else if (value.state == TENTATIVE) config.tentative |= 1u << row;
            else config.locked |= 1u << row;
            if (record.at("selected").at("raw").at(0).at(row).get<bool>()) expected_selected |= 1u << row;
            if (record.at("direct").at("raw").at(0).at(row).get<bool>()) expected_direct |= 1u << row;
            if (record.at("tentative").at("raw").at(0).at(row).get<bool>()) expected_stable |= 1u << row;
            if (record.at("fallback").at("raw").at(0).at(row).get<bool>()) expected_fallback |= 1u << row;
        }
        const auto actual = run_case(simulation, config, rows);
        compare_result("canonical_future_c11", actual, c11_result(config, rows));
        require(!actual.error, "canonical future RTL error " + std::to_string(actual.error_id));
        require(actual.selected == expected_selected && actual.direct == expected_direct &&
            actual.stable == expected_stable && actual.fallback == expected_fallback,
            "canonical future selection/direct masks differ from generator");
        for (unsigned row = 0; row < 32; ++row) {
            const auto& expected = end.at("after");
            const auto& value = actual.rows[row];
            const auto label = "future checkpoint " + std::to_string(checked) + " row " + std::to_string(row);
            require(value.token == record.at("tokens_after").at("raw").at(0).at(row), label+" token");
            require(value.state == expected.at("state").at("raw").at(0).at(row), label+" state");
            require(value.origin == expected.at("commit_origin").at("raw").at(0).at(row), label+" origin");
            require(value.age == expected.at("precision_age").at("raw").at(0).at(row), label+" age");
            require(value.last_top1 == std::uint32_t(expected.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>()), label+" history");
        }
        persisted_rows = actual.rows;
        ++checked;
    }
    require(checked == 4 && handoff_locked == 17, "future test_case did not exercise three future updates and LOCKED handoff");
    std::cout << "PASS canonical_future actual_generator_updates=" << checked
              << " positions=32 future_updates=3 current_updates=1 handoff_locked=" << handoff_locked
              << " persisted_rtl_state=1 candidates=synthetic cache_forward=synthetic\n";
}

void check_source_a_reference(Simulation& simulation, const nlohmann::json& test_case) {
    const bool handoff = test_case.at("config").at("dynamic_block_source_a_confirm_at_handoff");
    std::size_t checked = 0;
    std::vector<Row> persisted_rows, before_handoff;
    std::size_t handoff_locked = 0;
    const auto& records = test_case.at("records");
    for (std::size_t index = 1; index+1 < records.size(); ++index) {
        const auto& record = records[index];
        if (record.at("event") != "admit" || record.at("token_storage_offset") != 33)
            continue;
        const auto& capture = test_case.at("captures").at(record.at("forward_capture").get<unsigned>());
        const bool future = capture.at("block_index") == 0;
        if (!future && capture.at("step_index") != 0) continue;
        const auto& decision = records[index-1];
        const auto& end = records[index+1];
        require(decision.at("event") == "select" && end.at("event") == "age" &&
            end.at("state_id") == record.at("state_id"), "future selection/state event association failed");
        Config config;
        config.canonical_future = future; config.tail_enable = !future;
        config.source_a_handoff = future && handoff;
        config.mask_token = 3; config.vocabulary = 4;
        config.scheduled = decision.at("quota").at("raw").at(0);
        config.remaining = decision.at("parameters").at("remaining_forwards");
        config.budget = rtl_f32_to_bf16(decision.at("parameters").at("budget_scale").get<float>());
        config.step = future ? 0 : capture.at("step_index").get<unsigned>();
        for (auto position : capture.at(future ? "future_prediction_positions" : "prediction_positions").at("raw")) {
            const auto absolute = position.get<unsigned>();
            if (absolute >= 33 && absolute < 65) config.observed |= 1u << (absolute-33);
        }
        std::vector<Row> rows = persisted_rows.empty() ? std::vector<Row>(32) : persisted_rows;
        std::uint32_t expected_selected = 0, expected_direct = 0, expected_stable = 0, expected_fallback = 0;
        std::uint32_t expected_confirmed = 0, expected_remasked = 0;
        for (unsigned row = 0; row < 32; ++row) {
            auto& value = rows[row];
            const auto& before = record.at("before");
            if (persisted_rows.empty()) {
                value.state = before.at("state").at("raw").at(0).at(row);
                value.origin = before.at("commit_origin").at("raw").at(0).at(row);
                value.age = before.at("precision_age").at("raw").at(0).at(row);
                value.last_top1 = before.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>();
                value.token = record.at("tokens_before").at("raw").at(0).at(row);
            } else if (future) {
                require(value.state == before.at("state").at("raw").at(0).at(row) &&
                    value.origin == before.at("commit_origin").at("raw").at(0).at(row) &&
                    value.age == before.at("precision_age").at("raw").at(0).at(row) &&
                    value.last_top1 == std::uint32_t(before.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>()) &&
                    value.token == record.at("tokens_before").at("raw").at(0).at(row),
                    "persisted RTL state differs from next generator input at update " + std::to_string(checked) + " row " + std::to_string(row));
            }
            if (!future) {
                // The admit snapshot is after confirmation. Its input must not
                // replace the state retained from the preceding RTL forward.
                require(value.state == capture.at("state").at("raw").at(0).at(row) &&
                    value.token == capture.at("tokens").at("raw").at(0).at(33+row),
                    "Source A handoff input differs from the generator pre-forward state");
                if (value.state == LOCKED) ++handoff_locked;
                if (value.state == TENTATIVE) {
                    const unsigned after_verify = before.at("state").at("raw").at(0).at(row);
                    if (after_verify == LOCKED) expected_confirmed |= 1u << row;
                    if (after_verify == MASKED) expected_remasked |= 1u << row;
                }
            }
            value.logical = 33+row;
            value.source_a = false;
            for (const auto& position : capture.at("source_a_positions").at("raw"))
                if (position.get<unsigned>() == value.logical) value.source_a = true;
            const bool pending_before = capture.at("source_a_pending").at("raw").at(0).at(value.logical);
            require(future ? value.source_a_pending == pending_before : !pending_before,
                "Source A pending differs from generator boundary clearing");
            value.bits = value.state == MASKED || (value.state == LOCKED && value.age >= 3) ? 4 : 8;
            const auto& positions = capture.at("input_positions").at("raw");
            const auto& bit_record = capture.at("activation_bits").at("raw");
            const auto& bits = bit_record.at(0).is_array() ? bit_record.at(0) : bit_record;
            for (unsigned physical = 0; physical < positions.size(); ++physical)
                if (positions.at(physical) == value.logical) value.bits = bits.at(physical);
            value.prediction_flag = value.state != LOCKED && ((config.observed >> row)&1);
            if ((config.observed >> row)&1) { value.cache_valid = true; value.refresh_required = false; }
            value.top1 = record.at("proposal").at("raw").at(0).at(row).get<std::int64_t>();
            value.action_confidence = decision.at("confidence").at("raw").at(0).at(row);
            value.selected_probability = value.action_confidence;
            if (value.state == MASKED) config.masked |= 1u << row;
            else if (value.state == TENTATIVE) config.tentative |= 1u << row;
            else config.locked |= 1u << row;
            if (record.at("selected").at("raw").at(0).at(row).get<bool>()) expected_selected |= 1u << row;
            if (record.at("direct").at("raw").at(0).at(row).get<bool>()) expected_direct |= 1u << row;
            if (record.at("tentative").at("raw").at(0).at(row).get<bool>()) expected_stable |= 1u << row;
            if (record.at("fallback").at("raw").at(0).at(row).get<bool>()) expected_fallback |= 1u << row;
        }
        const auto actual = run_case(simulation, config, rows);
        compare_result("canonical_future_c11", actual, c11_result(config, rows));
        require(!actual.error, "canonical future RTL error " + std::to_string(actual.error_id));
        require(actual.selected == expected_selected && actual.direct == expected_direct &&
            actual.stable == expected_stable && actual.fallback == expected_fallback,
            "canonical future selection/direct masks differ from generator");
        if (!future) require(actual.confirmed == expected_confirmed && actual.remasked == expected_remasked,
            "Source A handoff confirm/remask masks differ from generator");
        for (unsigned row = 0; row < 32; ++row) {
            const auto& expected = end.at("after");
            const auto& value = actual.rows[row];
            const auto label = "future checkpoint " + std::to_string(checked) + " row " + std::to_string(row);
            require(value.token == record.at("tokens_after").at("raw").at(0).at(row), label+" token");
            require(value.state == expected.at("state").at("raw").at(0).at(row), label+" state");
            require(value.origin == expected.at("commit_origin").at("raw").at(0).at(row), label+" origin");
            require(value.age == expected.at("precision_age").at("raw").at(0).at(row), label+" age");
            require(value.last_top1 == std::uint32_t(expected.at("last_top1").at("raw").at(0).at(row).get<std::int64_t>()), label+" history");
        }
        if (future) before_handoff = actual.rows;
        persisted_rows = actual.rows;
        ++checked;
    }
    require(checked == 3 && handoff_locked > 0, "Source A reference needs two future updates and actual current handoff");
    if (handoff) {
        unsigned pending = 0;
        Config current; current.tail_enable = false;
        current.mask_token = 3; current.vocabulary = 4;
        for (auto& row : before_handoff) {
            pending += row.source_a_pending;
            row.source_a = false; row.bits = 4;
            row.prediction_flag = row.state != LOCKED;
            row.top1 = row.token == 3 ? 0 : row.token;
            row.action_confidence = row.selected_probability = 0x3f80;
            const auto bit = 1u << (row.logical-33);
            if (row.state == MASKED) current.masked |= bit;
            else if (row.state == TENTATIVE) current.tentative |= bit;
            else current.locked |= bit;
        }
        require(pending > 0, "Source A reference did not retain a pending draft");
        const auto actual = run_case(simulation, current, before_handoff);
        compare_result("source_a_current_a4", actual, c11_result(current, before_handoff));
        require(!actual.confirmed && !actual.remasked, "A4 handoff confirmed a draft");
        for (const auto& row : actual.rows) require(!row.source_a_pending, "handoff did not clear pending");
    }
    std::cout << "PASS source_a_handoff=" << handoff << " actual_generator_updates=" << checked
              << " positions=32"
              << " persisted_rtl_state=1 candidates=synthetic cache_forward=synthetic\n";
}

void test_actual_closeouts(Simulation& simulation) {
    std::ifstream stream("cases/control/psme_block_closeouts.json");
    require(stream.good(), "missing actual generator closeout test_case");
    nlohmann::json document; stream >> document;
    const auto values = [](const nlohmann::json& tensor) {
        auto data = tensor.at("raw");
        while (data.size() == 1 && data.at(0).is_array()) data = data.at(0);
        return data;
    };
    unsigned checked = 0;
    for (const auto& test_case : document.at("draft_verify")) {
        const auto& captures = test_case.at("captures");
        for (unsigned index = 1; index < captures.size(); ++index) {
            const auto& capture = captures.at(index);
            const auto& before = capture.at("state_fields_at_forward_start");
            const bool force = capture.at("forward_kind") == "local_forced_finish";
            const auto& after = index+1 < captures.size() ? captures.at(index+1).at("state_fields_at_forward_start") : test_case.at("final_state");
            const auto after_tokens = index+1 < captures.size() ? values(captures.at(index+1).at("tokens_at_forward_start")) : values(test_case.at("final_tokens"));
            Config config; config.closeout_kind = force ? 2 : 1; config.mask_token = 3; config.vocabulary = 4;
            std::vector<Row> rows(32);
            const auto positions = values(capture.at("token_positions"));
            const auto bits = values(capture.at("activation_bits"));
            const auto proposals = values(capture.at("proposal"));
            const auto bf16_code = [&](const nlohmann::json& tensor, unsigned local) {
                const auto raw = values(tensor).at(local).get<std::uint32_t>();
                if (tensor.at("dtype") == "torch.bfloat16") return std::uint16_t(raw);
                require(tensor.at("dtype") == "torch.float32" && (raw & 0xffffu) == 0,
                    "closeout FP32 container has additional precision");
                return std::uint16_t(raw>>16);
            };
            for (unsigned local = 0; local < 32; ++local) {
                auto& row = rows[local]; row.logical = local+1;
                row.state = values(before.at("state")).at(local); row.token = values(capture.at("tokens_at_forward_start")).at(local+1);
                row.origin = values(before.at("commit_origin")).at(local); row.age = values(before.at("precision_age")).at(local);
                row.last_top1 = std::uint32_t(values(before.at("last_top1")).at(local).get<std::int64_t>());
                const auto position = std::find(positions.begin(), positions.end(), row.logical);
                if (position == positions.end()) {
                    require(row.state == LOCKED, "unresolved closeout row omitted from actual query");
                    row.bits = values(before.at("row_bits")).at(local);
                } else {
                    row.bits = bits.at(position-positions.begin());
                }
                row.top1 = proposals.at(local);
                row.prediction_flag = force ? row.state == MASKED : row.state == TENTATIVE;
                row.selected_probability = force ? 0 : bf16_code(capture.at("confirmation").at("selected_token_probability"), local);
                row.action_confidence = bf16_code(capture.at("action_confidence"), local);
                if (row.state == MASKED) config.masked |= 1u<<local;
                else if (row.state == TENTATIVE) config.tentative |= 1u<<local;
                else config.locked |= 1u<<local;
            }
            const auto actual = run_case(simulation, config, rows);
            compare_result("actual_closeout_c11", actual, c11_result(config, rows));
            for (unsigned local = 0; local < 32; ++local) {
                const auto& row = actual.rows.at(local);
                require(row.state == values(after.at("state")).at(local) &&
                    row.origin == values(after.at("commit_origin")).at(local) &&
                    row.age == values(after.at("precision_age")).at(local) &&
                    row.last_top1 == std::uint32_t(values(after.at("last_top1")).at(local).get<std::int64_t>()) &&
                    row.token == after_tokens.at(local+(index+1 < captures.size() ? 1 : 0)),
                    "closeout RTL differs from actual generator case="+test_case.at("kind").get<std::string>()+
                    " capture="+std::to_string(index)+" row="+std::to_string(local)+
                    " actual(state/origin/age/history/token)="+std::to_string(row.state)+"/"+
                    std::to_string(row.origin)+"/"+std::to_string(row.age)+"/"+std::to_string(row.last_top1)+"/"+std::to_string(row.token)+
                    " expected="+values(after.at("state")).at(local).dump()+"/"+values(after.at("commit_origin")).at(local).dump()+"/"+
                    values(after.at("precision_age")).at(local).dump()+"/"+values(after.at("last_top1")).at(local).dump()+"/"+
                    after_tokens.at(local+(index+1 < captures.size() ? 1 : 0)).dump());
            }
            ++checked;
        }
    }
    Config force; force.masked = 1; force.closeout_kind = 2;
    Row row; row.token = force.mask_token; row.logical = 9; row.last_top1 = 7; row.origin = ORIGIN_FALLBACK; row.top1 = 2;
    const auto preserved = run_case(simulation, force, {row});
    compare_result("forced_history", preserved, c11_result(force, {row}));
    require(preserved.rows.at(0).last_top1 == 7 && preserved.rows.at(0).origin == ORIGIN_FALLBACK,
        "forced finish discarded history/origin");
    Config confirm; confirm.closeout_kind = 1; confirm.masked = 1; confirm.tentative = 2; confirm.locked = 4;
    std::vector<Row> mixed(3);
    mixed[0] = row; mixed[0].prediction_flag = false; mixed[0].refresh_required = true; mixed[0].cache_valid = false;
    mixed[1].logical = 10; mixed[1].state = TENTATIVE; mixed[1].bits = 8; mixed[1].age = -1;
    mixed[1].token = mixed[1].top1 = 5; mixed[1].last_top1 = 7; mixed[1].origin = ORIGIN_FALLBACK;
    mixed[1].selected_probability = rtl_f32_to_bf16(0.9f);
    mixed[2].logical = 11; mixed[2].state = LOCKED; mixed[2].age = 2; mixed[2].prediction_flag = false;
    mixed[2].token = 6; mixed[2].last_top1 = 9; mixed[2].refresh_required = true;
    const auto mixed_result = run_case(simulation, confirm, mixed);
    compare_result("confirmation_subset", mixed_result, c11_result(confirm, mixed));
    require(mixed_result.rows[0].state == MASKED && mixed_result.rows[0].last_top1 == 7 &&
        mixed_result.rows[0].refresh_required && mixed_result.rows[1].last_top1 == 7 &&
        mixed_result.rows[2].last_top1 == 9 && mixed_result.rows[2].age == 3 && mixed_result.selected == 0,
        "confirmation-only changed an unrelated MASKED history or performed admission");
    for (const unsigned kind : {2u, 3u}) {
        auto invalid = confirm; invalid.closeout_kind = kind;
        const auto rejected = run_case(simulation, invalid, mixed);
        require(rejected.error && rejected.error_id == 1, "invalid closeout operation was not rejected");
    }
    compare_result("closeout_restart", run_case(simulation, confirm, mixed), mixed_result);
    std::cout << "PASS closeout actual_generator_updates=" << checked << " forced_history_preserved=1\n";
}

void test_actual_precision_cache(Simulation& simulation) {
    std::ifstream stream("cases/control/psme_state_boundaries.json");
    require(stream.good(), "missing actual generator cache test_case");
    nlohmann::json document; stream >> document;
    const auto values = [](const nlohmann::json& tensor) {
        auto data = tensor.at("raw");
        while (data.size() == 1 && data.at(0).is_array()) data = data.at(0);
        return data;
    };
    unsigned checked = 0;
    const auto& test_case = document.at("draft_verify").at(0);
    const auto& capture = test_case.at("captures").at(1);
    const auto& end = test_case.at("decisions").at(1).at("forward_end");
    const auto& previous = test_case.at("decisions").at(0).at("forward_end");
    const auto positions = values(end.at("decision_positions"));
    const auto queries = values(capture.at("token_positions"));
    const auto executed_bits = values(capture.at("activation_bits"));
    for (unsigned local = 0; local < positions.size(); ++local) {
        const unsigned position = positions.at(local);
        auto found = std::find(queries.begin(), queries.end(), position);
        if (found == queries.end()) continue;
        const unsigned bits = executed_bits.at(found-queries.begin());
        if (values(capture.at("state_at_forward_start")).at(local) != LOCKED ||
                values(end.at("state")).at(local) != LOCKED ||
                values(capture.at("tokens_at_forward_start")).at(position) != values(end.at("tokens")).at(local) ||
                bits == values(end.at("next_current_activation_bits")).at(local)) continue;
        require(!values(end.at("cache_refresh_due")).at(position).get<bool>(),
            "actual generator unexpectedly refreshed unchanged-token precision change");
        Config config; config.locked = 1;
        Row row{};
        row.logical = position; row.token = values(end.at("tokens")).at(local);
        row.age = values(previous.at("precision_age")).at(local);
        row.state = LOCKED; row.origin = values(previous.at("commit_origin")).at(local);
        row.bits = bits; row.prediction_flag = false;
        const auto actual = run_case(simulation, config, {row}, 83+checked);
        compare_result("actual-generator-precision-cache", actual, c11_result(config, {row}));
        require(!actual.error && actual.cache_invalidate == 0 && actual.mandatory == 0 &&
            actual.rows.at(0).bits == values(end.at("next_current_activation_bits")).at(local) &&
            actual.rows.at(0).age == values(end.at("precision_age")).at(local),
            "precision/cache fields differ from actual normal-forward-end");
        ++checked;
        break;
    }
    require(checked == 1, "actual generator test_case lacks precision-only cache counterexample");
    std::cout << "PASS actual_generator_precision_cache cases=" << checked
              << " executed=A4 next=A8 unchanged_token=1 cache_refresh_due=0\n";
}

void test_fixed_k_state(Simulation& simulation) {
    std::ifstream stream("cases/control/psme_fixed_k_decoding.json");
    require(stream.good(), "missing fixed-k state fixture");
    nlohmann::json test_case; stream >> test_case;
    unsigned checked = 0;
    for (const auto& record : test_case.at("records")) {
        const auto& input = record.at("inputs");
        const auto& expected = record.at("expected");
        const auto tokens = input.at("tokens").at("raw").at(0).get<std::vector<unsigned>>();
        Config config;
        config.transfer_only = true; config.tail_enable = false;
        config.mask_token = input.at("mask_token_id"); config.vocabulary = 8;
        config.scheduled = input.at("scheduled_quota"); config.remaining = 1;
        std::vector<Row> rows(tokens.size());
        unsigned selected = 0;
        for (unsigned index = 0; index < rows.size(); ++index) {
            auto& row = rows[index];
            row.token = tokens[index]; row.logical = index; row.bits = 8;
            row.top1 = expected.at("proposal").at("raw").at(0).at(index);
            row.action_confidence = input.at("action_confidence").at("raw").at(0).at(index);
            row.selected_probability = input.at("selected_probability").at("raw").at(0).at(index);
            row.last_top1 = input.at("last_top1").at("raw").at(0).at(index).get<int>();
            row.age = input.at("precision_age").at("raw").at(0).at(index).get<int>();
            row.state = input.at("state").at("raw").at(0).at(index);
            row.origin = input.at("commit_origin").at("raw").at(0).at(index);
            row.suppressed = !input.at("suppressed_candidate_token_ids").empty();
            if (row.state == MASKED) config.masked |= 1u << index;
            else {
                config.locked |= 1u << index;
                row.prediction_flag = false; row.cache_valid = false; row.refresh_required = true;
            }
            if (expected.at("selected").at("raw").at(0).at(index).get<bool>()) selected |= 1u << index;
        }
        const auto actual = run_case(simulation, config, rows, 91 + checked);
        compare_result(record.at("name"), actual, c11_result(config, rows));
        require(actual.selected == selected, "fixed-k selection differs from algorithm");
        for (unsigned index = 0; index < rows.size(); ++index) {
            const auto& row = actual.rows[index];
            const auto check = [&](const char* field, std::int64_t value) {
                require(value == expected.at(field).at("raw").at(0).at(index).get<std::int64_t>(),
                        record.at("name").get<std::string>() + ": fixed-k " + field +
                        " row=" + std::to_string(index));
            };
            check("tokens", row.token); check("state", row.state);
            check("last_top1", row.last_top1 == INVALID_TOKEN ? -1 : std::int64_t(row.last_top1));
            check("precision_age", row.age); check("commit_origin", row.origin);
            require(row.bits == 8, "fixed-k next precision differs from A8");
        }
        for (unsigned precision : {4u, 0u}) {
            auto precision_rows = rows;
            for (unsigned i = 0; i < precision_rows.size(); ++i)
                precision_rows[i].bits = precision ? precision : (i % 2 ? 4 : 8);
            const auto precision_result = run_case(simulation, config, precision_rows, 191 + checked + precision);
            compare_result("transfer_activation_precision", precision_result, c11_result(config, precision_rows));
            compare_result("transfer_precision_independent", precision_result, actual);
        }
        ++checked;
    }
    require(checked == 12, "fixed-k test_case coverage changed");
    Config config;
    config.transfer_only = true; config.tail_enable = false; config.masked = 1; config.locked = 2;
    config.mask_token = 7; config.vocabulary = 8; config.scheduled = 0;
    std::vector<Row> rows(2);
    rows[0].token = 7; rows[0].top1 = 0; rows[0].bits = 8;
    rows[1].token = 6; rows[1].top1 = 1; rows[1].bits = 8; rows[1].logical = 1;
    rows[1].state = LOCKED; rows[1].origin = ORIGIN_HIGH; rows[1].age = 5;
    rows[1].prediction_flag = false; rows[1].cache_valid = false;
    const auto transport = run_case(simulation, config, rows, 101);
    compare_result("transfer_locked_invalid_cache", transport, c11_result(config, rows));
    require(!transport.rows[1].cache_valid && !transport.rows[1].refresh_required &&
        transport.rows[1].age == 6 && transport.cache_keep == 1,
        "transfer-only fabricated a prior LOCKED cache refresh");
    std::cout << "PASS fixed_k_state algorithm_cases=" << checked
              << " quota0/1/2/3/32/ties/history/age/mask_proposal/stalls=checked cache_state=checked\n";
}

void test_a4_inherited_draft(Simulation& simulation) {
    Config config;
    config.tentative = 3; config.tail_enable = false; config.tail_bypass_all = true;
    std::vector<Row> rows(2);
    for (unsigned index = 0; index < 2; ++index) {
        auto& row = rows[index];
        row.logical = index; row.state = TENTATIVE; row.origin = ORIGIN_STABLE;
        row.token = 7; row.last_top1 = INVALID_TOKEN; row.bits = 4;
        row.top1 = index == 0 ? 7 : 8;
        row.selected_probability = row.action_confidence = 0x3f80;
        row.cache_valid = false; row.refresh_required = true;
    }
    const auto a4 = run_case(simulation, config, rows, 203);
    compare_result("inherited_a4_deferred", a4, c11_result(config, rows));
    require(!a4.confirmed && !a4.remasked && !a4.cache_commit && !a4.token_changed,
            "A4 execution confirmed or remasked an inherited draft");
    for (const auto& row : a4.rows)
        require(row.state == TENTATIVE && row.token == 7 && row.bits == 8 && row.refresh_required,
                "deferred draft lost its token or next A8 precision");
    config.tail_bypass_all = false;
    config.tail_bypass_stable_only = true;
    const auto stable_tail = run_case(simulation, config, rows, 211);
    compare_result("inherited_a4_stable_tail_deferred", stable_tail, a4);
    config.tail_bypass_stable_only = false;
    config.tail_bypass_all = true;
    rows = a4.rows;
    // Published state is persistent; candidate logits are new forward inputs.
    for (unsigned index = 0; index < rows.size(); ++index) {
        rows[index].top1 = index == 0 ? 7 : 8;
        rows[index].selected_probability = rows[index].action_confidence = 0x3f80;
    }
    const auto a8 = run_case(simulation, config, rows, 204);
    compare_result("inherited_next_a8", a8, c11_result(config, rows));
    require(a8.confirmed == 1 && a8.remasked == 2 && a8.cache_commit == 1,
            "subsequent A8 did not confirm/remask inherited drafts");
    std::cout << "PASS inherited_draft A4_defer_then_A8_confirm_remask=checked\n";
}

void test_live_handoff_quota(Simulation& simulation) {
    std::ifstream stream("cases/control/uaps_live_state.json");
    nlohmann::json document; stream >> document;
    for (const auto& record : document.at("records")) {
        if (record.at("expected").at("quota").is_null()) continue;
        Config config;
        config.canonical_future=true; config.source_a_handoff=true; config.tail_enable=false;
        config.max_handoff=record.at("max_handoff"); config.scheduled=record.at("admission_budget");
        std::vector<Row> rows(32);
        for (unsigned i=0;i<32;++i) {
            auto& row=rows[i]; row.logical=64+i; row.state=record.at("future_states").at(i);
            row.token=row.state==MASKED ? config.mask_token:7;
            row.top1=row.state==MASKED ? 8:7;
            row.bits=row.state==TENTATIVE ? 8:4;
            row.origin=row.state==TENTATIVE ? ORIGIN_STABLE:ORIGIN_NONE;
            row.age=row.state==LOCKED ? 3:-1;
            row.source_a_pending=i==2;
            row.prediction_flag=row.state!=LOCKED && !row.source_a_pending;
            row.selected_probability=row.action_confidence=rtl_f32_to_bf16(.99f);
            if (row.prediction_flag) config.observed|=1u<<i;
            if (row.state==MASKED) config.masked|=1u<<i;
            else if (row.state==TENTATIVE) config.tentative|=1u<<i;
            else config.locked|=1u<<i;
        }
        auto effective=config;
        // The generator computes this quota by counting
        // incoming state before either confirmation or new admission.
        effective.max_handoff=0; effective.scheduled=record.at("expected").at("quota");
        const auto expected=c11_result(effective,rows);
        const auto actual=run_case(simulation,config,rows);
        compare_result("live_handoff_"+record.at("name").get<std::string>(),actual,expected);
        require(actual.confirmed==9 && actual.rows[2].source_a_pending,
            "quota calculation lost required confirmations or A-pending");
    }
    std::cout << "PASS live_handoff_quota incoming_tentative_before_confirmation includes_A_pending\n";
}

void test_confidence_history(Simulation& simulation) {
    Config config;
    config.canonical_future = true; config.masked = 1; config.observed = 1;
    config.tail_enable = false;
    std::vector<Row> rows{masked_row(32, 1, .25f)};
    const auto first = run_case(simulation, config, rows, 201);
    compare_result("history_first", first, c11_result(config, rows));
    require(first.rows[0].state == MASKED, "history test unexpectedly admitted first candidate");
    rows = first.rows;
    rows[0].prediction_flag = false;
    config.observed = 0;
    const auto idle = run_case(simulation, config, rows, 202);
    compare_result("history_unobserved", idle, c11_result(config, rows));
    require(idle.rows[0].action_confidence == 0x3e80, "unobserved row lost actual previous confidence");
    rows = idle.rows;
    rows[0].prediction_flag = true; rows[0].action_confidence = 0; rows[0].suppressed = true;
    config.observed = 1;
    const auto suppressed = run_case(simulation, config, rows, 203);
    compare_result("history_suppressed", suppressed, c11_result(config, rows));
    require(suppressed.rows[0].action_confidence == 0, "suppressed confidence did not replace history");
    std::cout << "PASS action_confidence_history actual_three_forward_chain observed/unobserved/suppressed\n";
}

void check_feature2_step(Simulation& simulation, const nlohmann::json& input) {
    require(input.at("schema") == "supra-observed-feature2-step/v1" ||
            input.at("schema") == "supra-feature2-step/v1", "wrong Feature2 input schema");
    const auto& config = input.at("config");
    Config cfg;
    cfg.mask_token = config.at("mask_token"); cfg.vocabulary = config.at("vocabulary");
    cfg.high_confidence_threshold = config.at("high"); cfg.low_confidence_threshold = config.at("low");
    cfg.verify_threshold = config.at("verify"); cfg.bonus = config.at("bonus"); cfg.budget = config.at("budget");
    cfg.scheduled = config.at("scheduled"); cfg.remaining = config.at("remaining"); cfg.step = config.at("step");
    cfg.tail_after = config.at("tail_after"); cfg.tail_enable = config.at("tail_enable"); cfg.tau_tail = config.at("tail");
    cfg.tail_bypass_all = config.at("tail_bypass_all"); cfg.tail_bypass_stable_only = config.at("tail_bypass_stable_only");
    cfg.tail_all = config.value("tail_all", false);
    cfg.transfer_only = config.value("transfer_only", false);
    cfg.closeout_kind = config.value("closeout_kind", 0u);
    std::vector<Row> rows;
    for (const auto& item : input.at("rows")) {
        Row row;
        row.token = item.at("token"); row.last_top1 = item.at("last_top1").get<int>();
        row.age = item.at("age"); row.logical = item.at("logical"); row.state = item.at("state");
        row.origin = item.at("origin"); row.bits = item.at("bits"); row.top1 = item.at("top1");
        row.selected_probability = item.at("selected_probability"); row.action_confidence = item.at("action_confidence");
        row.suppressed = item.at("suppressed"); row.refresh_required = item.at("refresh_required");
        row.prediction_flag = item.value("prediction_flag",
            cfg.closeout_kind == 1 ? row.state == TENTATIVE :
            cfg.closeout_kind == 2 ? row.state == MASKED : row.state != LOCKED);
        row.cache_valid = item.value("cache_valid", true);
        require(rows.size() < 32, "observed Feature2 block exceeds 32 positions");
        const std::uint32_t bit = std::uint32_t{1} << rows.size();
        if (row.state == MASKED) cfg.masked |= bit;
        else if (row.state == TENTATIVE) cfg.tentative |= bit;
        else if (row.state == LOCKED) cfg.locked |= bit;
        else throw std::runtime_error("invalid observed token state");
        rows.push_back(row);
    }
    const auto actual = run_case(simulation, cfg, rows, 179);
    require(!actual.error, "RTL rejected observed Feature2 input");
    compare_result("observed_feature2_cmodel", actual, c11_result(cfg, rows));
    const auto& expected = input.at("expected");
    const auto& masks = expected.at("masks");
    const auto check = [&](const char* field, std::uint32_t value) {
        require(value == masks.at(field).get<std::uint32_t>(),
                std::string("observed Feature2 mask differs: ") + field + " actual=" + std::to_string(value) +
                " expected=" + std::to_string(masks.at(field).get<std::uint32_t>()));
    };
    check("selected", actual.selected); check("direct", actual.direct); check("stable", actual.stable);
    check("fallback", actual.fallback); check("confirmed", actual.confirmed); check("remasked", actual.remasked);
    check("token_changed", actual.token_changed);
    if (masks.contains("tail_closed")) check("tail_closed", actual.tail_closed);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = actual.rows.at(i);
        const auto same = [&](const char* field, std::int64_t value) {
            require(value == expected.at(field).at(i).get<std::int64_t>(),
                    std::string("observed Feature2 row differs: ") + field + " row=" + std::to_string(i));
        };
        same("tokens", row.token); same("state", row.state); same("precision_age", row.age);
        same("commit_origin", row.origin); same("bits", row.bits);
        same("last_top1", row.last_top1 == INVALID_TOKEN ? -1 : static_cast<std::int64_t>(row.last_top1));
        if (expected.contains("refresh_required")) same("refresh_required", row.refresh_required);
        if (expected.contains("cache_valid")) same("cache_valid", row.cache_valid);
    }
    std::cout << "PASS feature2 rows=" << rows.size()
              << " independent_expected=generator state/history/age/bits/masks\n";
}

void test_observed_feature2_step(Simulation& simulation, const char* path) {
    std::ifstream stream(path);
    require(stream.good(), "missing Feature2 input");
    nlohmann::json input; stream >> input;
    if (input.at("schema") == "supra-psme-state-updates/v1") {
        for (const auto& record : input.at("records")) check_feature2_step(simulation, record);
    } else {
        check_feature2_step(simulation, input);
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Simulation simulation(1);
        if (argc > 2 && std::string(argv[1]) == "--observed-feature2") {
            for (int i = 2; i < argc; ++i) test_observed_feature2_step(simulation, argv[i]);
            return 0;
        }
        test_reference_case(simulation);
        test_quota_tie_tail(simulation);
        test_a4_direct_boundary(simulation);
        test_full_rows_and_three_forwards(simulation);
        test_tail_all_and_errors(simulation);
        test_packed_context_precision(simulation);
        test_normal_tail_bypass(simulation);
        test_stable_tail_and_budget(simulation);
        test_actual_closeouts(simulation);
        test_observed_feature2_step(simulation, "cases/control/psme_state_updates.json");
        test_actual_precision_cache(simulation);
        test_abort();
        test_fixed_k_state(simulation);
        test_confidence_history(simulation);
        test_live_handoff_quota(simulation);
        test_a4_inherited_draft(simulation);
        std::ifstream source_a_file("cases/control/uaps_source_a_handoff.json");
        require(source_a_file.good(), "missing current Source A reference");
        nlohmann::json source_a_reference; source_a_file >> source_a_reference;
        for (const auto& variant : source_a_reference.at("variants")) check_source_a_reference(simulation, variant);
        if (argc > 1) test_actual_future_test_case(simulation, argv[1]);
        std::cout << "PASS draft_verify_state_controller rows=1/32 forwards=3 "
                     "a4_direct_threshold=0x3f66 "
                     "bf16_stall=enabled result_stall=enabled abort=drained\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL draft_verify_state_controller: " << error.what() << '\n';
        return 1;
    }
}
