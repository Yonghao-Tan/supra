`default_nettype none

// Reads one complete K step from the quantized-activation banks and one weight
// panel. One accepted bundle reserves all SRAM ports used by that K step.
module matmul_operand_path #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       abort_request,
    output logic                       abort_ack,

    input  logic                       scale_cfg_valid,
    output logic                       scale_cfg_ready,
    input  logic [8*16-1:0]            scale_cfg_activation,
    input  logic [8*16-1:0]            scale_cfg_weight,
    input  logic [15:0]                scale_cfg_row_batch_id,
    input  logic [15:0]                scale_cfg_stripe_id,

    input  logic                       issue_valid,
    output logic                       issue_ready,
    input  logic [1:0]                 issue_mode,
    input  logic [3:0]                 issue_active_rows,
    input  logic                       issue_panel,
    input  logic [7:0]                 issue_row_mask,
    input  logic [31:0]                issue_k_mask,
    input  logic [7:0]                 issue_column_mask,
    input  logic                       issue_first_k_step,
    input  logic                       issue_last_k_step,
    input  logic                       issue_mixed_phase,
    input  logic                       issue_mixed_phase_first,
    input  logic [7:0]                 issue_mixed_a8_rows,
    input  logic                       issue_reuse_panel,
    input  logic                       issue_explicit_activation_rows,
    input  logic [47:0]                issue_activation_physical_rows,
    input  logic [127:0]               issue_activation_lane_word_index,
    input  logic                       issue_expanded_activation_layout,
    input  logic [127:0]               issue_activation_scales,
    input  logic [127:0]               issue_weight_scales,
    input  logic [TAG_WIDTH-1:0]       issue_tag,
    input  logic [15:0]                issue_row_batch_id,
    input  logic [15:0]                issue_stripe_id,
    input  logic [31:0]                issue_activation_byte_address,
    input  logic                       issue_qkv_padded_layout,
    input  logic [5:0]                 issue_physical_row_base,
    input  logic [2:0]                 issue_panel_start_bank,
    input  logic [9:0]                 issue_panel_word_row,
    input  logic [2:0]                 issue_panel_bank_count,
    input  logic [9:0]                 issue_panel_plane_rows,

    output logic                       read_bundle_valid,
    input  logic                       read_bundle_ready,
    output logic [15:0]                activation_read_word_index,
    output logic [3:0]                 activation_read_word_count,
    output logic                       activation_read_qkv_padded_layout,
    output logic [5:0]                 activation_read_physical_row_base,
    output logic                       activation_read_explicit_rows,
    output logic [47:0]                activation_read_physical_rows,
    output logic [127:0]               activation_read_lane_word_index,
    output logic                       panel_read_panel,
    output logic [11:0]                panel_read_port_mask,
    output logic [119:0]               panel_read_address,
    input  logic                       read_response_valid,
    input  logic [8*128-1:0]           activation_read_data,
    input  logic [12*128-1:0]          panel_read_data,

    output logic                       req_valid,
    input  logic                       req_ready,
    output logic [1:0]                 req_mode,
    output logic [1023:0]              req_activation_payload,
    output logic [1023:0]              req_weight_payload,
    output logic [7:0]                 req_row_mask,
    output logic [31:0]                req_k_mask,
    output logic [7:0]                 req_col_mask,
    output logic                       req_first_k_step,
    output logic                       req_last_k_step,
    output logic                       req_mixed_phase,
    output logic                       req_mixed_phase_first,
    output logic [7:0]                 req_mixed_a8_rows,
    output logic [127:0]               req_activation_scales,
    output logic [127:0]               req_weight_scales,
    output logic [TAG_WIDTH-1:0]       req_tag,

    output logic [63:0]                accepted_issue_count,
    output logic [63:0]                accepted_request_count
);
`ifdef SYNTHESIS
    always_comb begin
        accepted_issue_count = '0;
        accepted_request_count = '0;
    end
`endif
    localparam logic [1:0] MODE_W4A4 = 2'd0;
    localparam logic [1:0] MODE_W4A8 = 2'd1;
    localparam logic [1:0] MODE_W8A8 = 2'd2;
    localparam logic [1:0] MODE_MIXED_W4 = 2'd3;
    localparam integer PTR_WIDTH = 1;
    localparam integer META_WIDTH =
        2+8+32+8+1+1+1+1+8+1+TAG_WIDTH+4+128+128;
    localparam integer RESPONSE_WIDTH = META_WIDTH+8*128+8*128+2*128;
    localparam integer RESPONSE_CAPACITY = 2;

    logic scale_cache_valid;
    logic [8*16-1:0] activation_scale_cache;
    logic [8*16-1:0] weight_scale_cache;
    logic [15:0] scale_row_batch_id;
    logic [15:0] scale_stripe_id;
    logic scale_match;

    logic [1:0] response_pending;
    logic [META_WIDTH-1:0] pending_meta [0:1];
    logic [2:0] pending_panel_start_bank [0:1];
    logic [2:0] pending_panel_bank_count [0:1];
    logic response_fifo_input_valid;
    logic response_fifo_input_ready;
    logic [RESPONSE_WIDTH-1:0] response_fifo_input_data;
    logic response_fifo_output_valid;
    logic response_fifo_output_ready;
    logic [RESPONSE_WIDTH-1:0] response_fifo_output_data;
    logic [PTR_WIDTH:0] response_fifo_occupancy;

    logic [META_WIDTH-1:0] response_meta;
    logic [8*128-1:0] response_activation_physical;
    logic [8*128-1:0] response_base_words;
    logic [2*128-1:0] response_enhancement_words;
    logic [1:0] response_mode;
    logic [7:0] response_row_mask;
    logic [31:0] response_k_mask;
    logic [7:0] response_col_mask;
    logic response_first;
    logic response_last;
    logic response_mixed_phase;
    logic response_mixed_phase_first;
    logic [7:0] response_mixed_a8_rows;
    logic response_reuse_panel;
    logic [TAG_WIDTH-1:0] response_tag;
    logic [3:0] response_activation_byte_offset;
    logic [8*16-1:0] response_activation_scales;
    logic [8*16-1:0] response_weight_scales;

    logic [3:0] activation_beat_count;
    logic [10:0] activation_payload_bits;
    logic [3:0] panel_base_beat_count;
    logic [31:0] activation_word_index;
    logic [9:0] panel_last_word_row;
    logic address_valid;
    logic explicit_address_valid;
    logic response_credit_available;
    logic [PTR_WIDTH+1:0] reserved_response_count;
    logic issue_accept;
    logic abort_pending;

    logic [8*128-1:0] selected_response_base;
    logic [2*128-1:0] selected_response_enhancement;
    logic [8*128-1:0] aligned_activation;
    logic [1023:0] formatted_activation_payload;
    logic [32*8*4-1:0] formatted_weight_base;
    logic [32*8*4-1:0] formatted_weight_enhancement;
    logic [1023:0] formatted_weight_payload;
    logic [1023:0] held_mixed_weight_payload;
    logic held_mixed_weight_valid;

    function automatic logic [7:0] combine_signed_digits(
        input logic [3:0] base_nibble,
        input logic [3:0] enhancement_nibble
    );
        logic signed [7:0] base_times_sixteen;
        logic signed [7:0] signed_enhancement;
        begin
            base_times_sixteen = $signed({base_nibble, 4'b0000});
            signed_enhancement =
                $signed({{4{enhancement_nibble[3]}}, enhancement_nibble});
            combine_signed_digits = base_times_sixteen + signed_enhancement;
        end
    endfunction

    assign activation_payload_bits = issue_active_rows *
        (issue_mode == MODE_W8A8 ? 11'd64 : 11'd128);
    assign activation_beat_count = 4'((activation_payload_bits +
        {4'd0, issue_activation_byte_address[3:0], 3'b000} + 11'd127) >> 7);
    assign panel_base_beat_count = issue_mode == MODE_W8A8 ? 4'd2 :
                                   issue_mode == MODE_W4A8 ? 4'd4 : 4'd8;
    assign activation_word_index = issue_activation_byte_address >> 4;
    always_comb begin
        integer last_bank_sum;
        last_bank_sum = integer'(issue_panel_start_bank) +
            integer'(panel_base_beat_count) - 1;
        panel_last_word_row = issue_panel_word_row;
        if (issue_panel_bank_count == 3'd2)
            panel_last_word_row = issue_panel_word_row +
                10'(last_bank_sum >> 1);
        else if (issue_panel_bank_count == 3'd4)
            panel_last_word_row = issue_panel_word_row +
                10'(last_bank_sum >> 2);
        else if (last_bank_sum >= 12)
            panel_last_word_row = issue_panel_word_row + 10'd2;
        else if (last_bank_sum >= 6)
            panel_last_word_row = issue_panel_word_row + 10'd1;
    end

    // Each activation group contains 8192 words. A panel plane uses two banks for
    // QKV W8, four banks for W4, or six banks for the K=12288 layout.
    always_comb begin
        explicit_address_valid = 1'b1;
        for (integer slot = 0; slot < 8; slot++) begin
            if (slot < issue_active_rows &&
                (issue_activation_physical_rows[slot*6 +: 6] >= 6'd48 ||
                 issue_activation_lane_word_index[slot*16 +: 16] >=
                    (issue_expanded_activation_layout ? 16'd36864 : 16'd24576)))
                explicit_address_valid = 1'b0;
        end
    end
    assign address_valid = issue_active_rows >= 1 && issue_active_rows <= 8 &&
        (issue_mode == MODE_W8A8 ? issue_activation_byte_address[2:0] == 3'd0 :
                                   issue_activation_byte_address[3:0] == 4'd0) &&
        (issue_panel_bank_count == 3'd2 ||
         issue_panel_bank_count == 3'd4 ||
         issue_panel_bank_count == 3'd6) &&
        issue_panel_start_bank < issue_panel_bank_count &&
        activation_word_index <
            (issue_expanded_activation_layout ? 32'd36864 : 32'd24576) &&
        activation_word_index + {28'd0, activation_beat_count} <=
            (issue_expanded_activation_layout ? 32'd36864 : 32'd24576) &&
        panel_last_word_row < issue_panel_plane_rows &&
        issue_panel_plane_rows != 0 && issue_panel_plane_rows <= 10'd768 &&
        (!issue_explicit_activation_rows || explicit_address_valid) &&
        (!issue_reuse_panel ||
         (issue_mode == MODE_MIXED_W4 && issue_mixed_phase));
    assign scale_match = issue_mode == MODE_MIXED_W4 ||
        (scale_cache_valid &&
        scale_row_batch_id == issue_row_batch_id &&
        scale_stripe_id == issue_stripe_id);
    assign scale_cfg_ready = !abort_pending && !abort_request &&
        !(|response_pending) && response_fifo_occupancy == 0;
    // Two response credits are sufficient for the non-backpressurable local
    // memory response: one response may be stalled at the PE while the next
    // accepted reads are in the fixed two-cycle SRAM response pipeline.
    // A same-cycle PE handshake releases one credit for a new read issue.
    assign reserved_response_count =
        (PTR_WIDTH+2)'(response_fifo_occupancy) +
        (PTR_WIDTH+2)'($countones(response_pending));
    assign response_credit_available =
        reserved_response_count < (PTR_WIDTH+2)'(RESPONSE_CAPACITY) ||
        (response_fifo_output_valid && response_fifo_output_ready);

    // The returned bundle cannot be backpressured. A new issue is accepted only
    // when its fixed two-cycle SRAM response has reserved FIFO capacity.
    assign issue_ready = !abort_pending && !abort_request && scale_match &&
        address_valid && read_bundle_ready && response_credit_available;
    assign issue_accept = issue_valid && issue_ready;
    assign read_bundle_valid = issue_valid && !abort_pending && !abort_request &&
        scale_match && address_valid && response_credit_available;
    assign activation_read_word_index = activation_word_index[15:0];
    assign activation_read_word_count = activation_beat_count;
    assign activation_read_qkv_padded_layout = issue_qkv_padded_layout;
    assign activation_read_physical_row_base = issue_physical_row_base;
    assign activation_read_explicit_rows = issue_explicit_activation_rows;
    assign activation_read_physical_rows = issue_activation_physical_rows;
    assign activation_read_lane_word_index = issue_activation_lane_word_index;
    assign panel_read_panel = issue_panel;

    always_comb begin
        panel_read_port_mask = '0;
        panel_read_address = '0;
        for (integer beat = 0; beat < 8; beat = beat + 1) begin
            integer panel_bank;
            integer panel_bank_sum;
            integer panel_port;
            integer panel_row_carry;
            panel_bank_sum = integer'(issue_panel_start_bank) + beat;
            if (issue_panel_bank_count == 3'd2) begin
                panel_row_carry = panel_bank_sum >> 1;
                panel_bank = panel_bank_sum & 1;
            end else if (issue_panel_bank_count == 3'd4) begin
                panel_row_carry = panel_bank_sum >> 2;
                panel_bank = panel_bank_sum & 3;
            end else begin
                panel_row_carry = panel_bank_sum >= 12 ? 2 :
                                  panel_bank_sum >= 6 ? 1 : 0;
                panel_bank = panel_bank_sum - panel_row_carry * 6;
            end
            panel_port = beat < issue_panel_bank_count ? panel_bank : panel_bank + 6;
            if (beat < panel_base_beat_count && !issue_reuse_panel) begin
                panel_read_port_mask[panel_port] = 1'b1;
                panel_read_address[panel_port*10 +: 10] =
                    issue_panel_word_row + 10'(panel_row_carry);
            end
            if (issue_mode == MODE_W8A8 && beat < 2) begin
                panel_port = panel_bank + 6;
                panel_read_port_mask[panel_port] = 1'b1;
                panel_read_address[panel_port*10 +: 10] =
                    issue_panel_plane_rows + issue_panel_word_row +
                    10'(panel_row_carry);
            end
        end
    end

    assign response_fifo_input_valid = response_pending[1] && read_response_valid;
    always_comb begin
        selected_response_base = '0;
        selected_response_enhancement = '0;
        for (integer beat = 0; beat < 8; beat = beat + 1) begin
            integer panel_bank;
            integer panel_bank_sum;
            integer panel_port;
            panel_bank_sum = integer'(pending_panel_start_bank[1]) + beat;
            if (pending_panel_bank_count[1] == 3'd2)
                panel_bank = panel_bank_sum & 1;
            else if (pending_panel_bank_count[1] == 3'd4)
                panel_bank = panel_bank_sum & 3;
            else if (panel_bank_sum >= 12)
                panel_bank = panel_bank_sum - 12;
            else if (panel_bank_sum >= 6)
                panel_bank = panel_bank_sum - 6;
            else
                panel_bank = panel_bank_sum;
            panel_port = beat < pending_panel_bank_count[1] ?
                panel_bank : panel_bank + 6;
            case (panel_port)
                0: selected_response_base[beat*128 +: 128] = panel_read_data[0*128 +: 128];
                1: selected_response_base[beat*128 +: 128] = panel_read_data[1*128 +: 128];
                2: selected_response_base[beat*128 +: 128] = panel_read_data[2*128 +: 128];
                3: selected_response_base[beat*128 +: 128] = panel_read_data[3*128 +: 128];
                4: selected_response_base[beat*128 +: 128] = panel_read_data[4*128 +: 128];
                5: selected_response_base[beat*128 +: 128] = panel_read_data[5*128 +: 128];
                6: selected_response_base[beat*128 +: 128] = panel_read_data[6*128 +: 128];
                7: selected_response_base[beat*128 +: 128] = panel_read_data[7*128 +: 128];
                8: selected_response_base[beat*128 +: 128] = panel_read_data[8*128 +: 128];
                9: selected_response_base[beat*128 +: 128] = panel_read_data[9*128 +: 128];
                10: selected_response_base[beat*128 +: 128] = panel_read_data[10*128 +: 128];
                11: selected_response_base[beat*128 +: 128] = panel_read_data[11*128 +: 128];
                default: selected_response_base[beat*128 +: 128] = '0;
            endcase
            if (beat < 2) begin
                case (panel_bank)
                    0: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[6*128 +: 128];
                    1: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[7*128 +: 128];
                    2: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[8*128 +: 128];
                    3: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[9*128 +: 128];
                    4: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[10*128 +: 128];
                    5: selected_response_enhancement[beat*128 +: 128] =
                        panel_read_data[11*128 +: 128];
                    default: selected_response_enhancement[beat*128 +: 128] = '0;
                endcase
            end
        end
    end
    assign response_fifo_input_data = {pending_meta[1], activation_read_data,
        selected_response_base, selected_response_enhancement};
    assign {response_meta, response_activation_physical, response_base_words,
            response_enhancement_words} = response_fifo_output_data;
    assign {response_mode, response_row_mask, response_k_mask, response_col_mask,
            response_first, response_last, response_mixed_phase,
            response_mixed_phase_first, response_mixed_a8_rows,
            response_reuse_panel, response_tag,
            response_activation_byte_offset,
            response_activation_scales,
            response_weight_scales} = response_meta;

    logic [RESPONSE_WIDTH-1:0] response_storage [0:1];
    logic response_read_pointer;
    logic response_write_pointer;
    logic [1:0] response_stored_count;
    logic response_input_fire;
    logic response_output_fire;

    assign response_fifo_input_ready = response_stored_count < 2 ||
        (response_fifo_output_valid && response_fifo_output_ready);
    assign response_fifo_output_valid = response_stored_count != 0 ||
        response_fifo_input_valid;
    assign response_fifo_output_data = response_stored_count == 0 ?
        response_fifo_input_data : response_storage[response_read_pointer];
    assign response_fifo_occupancy = (PTR_WIDTH+1)'(response_stored_count);
    assign response_input_fire = response_fifo_input_valid &&
        response_fifo_input_ready;
    assign response_output_fire = response_fifo_output_valid &&
        response_fifo_output_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            response_read_pointer <= 1'b0;
            response_write_pointer <= 1'b0;
            response_stored_count <= 2'd0;
        end else begin
            if (response_input_fire &&
                !(response_stored_count == 0 && response_output_fire)) begin
                response_storage[response_write_pointer] <=
                    response_fifo_input_data;
                response_write_pointer <= ~response_write_pointer;
            end
            if (response_output_fire && response_stored_count != 0)
                response_read_pointer <= ~response_read_pointer;
            case ({response_input_fire &&
                       !(response_stored_count == 0 && response_output_fire),
                   response_output_fire && response_stored_count != 0})
                2'b10: response_stored_count <=
                    response_stored_count + 1'b1;
                2'b01: response_stored_count <=
                    response_stored_count - 1'b1;
                default: response_stored_count <= response_stored_count;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic stalled_response_output;
    logic [RESPONSE_WIDTH-1:0] stalled_response_output_data;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_response_output <= 1'b0;
            stalled_response_output_data <= '0;
        end else begin
            assert (response_stored_count <= 2)
                else $error("matmul operand response storage overflow");
            if (stalled_response_output)
                assert (response_fifo_output_valid &&
                        response_fifo_output_data ===
                            stalled_response_output_data)
                    else $error("matmul operand stalled response changed");
            stalled_response_output <= response_fifo_output_valid &&
                !response_fifo_output_ready;
            stalled_response_output_data <= response_fifo_output_data;
        end
    end
`endif

    always_comb begin
        // W4 requests are 16-byte aligned. W8 requests can start at either
        // 64-bit half of the first SRAM word, so only a fixed 0/64-bit mux is
        // required; a runtime shift here creates a 1024-bit barrel shifter.
        aligned_activation = response_activation_byte_offset == 4'd8 ?
            {64'd0, response_activation_physical[1023:64]} :
            response_activation_physical;
        formatted_activation_payload = '0;
        formatted_weight_base = '0;
        formatted_weight_enhancement = '0;
        formatted_weight_payload = '0;
        if (response_mode == MODE_W8A8) begin
            for (integer row_pair = 0; row_pair < 4; row_pair = row_pair + 1)
                for (integer row_in_pair = 0; row_in_pair < 2; row_in_pair = row_in_pair + 1)
                    for (integer k_lane = 0; k_lane < 8; k_lane = k_lane + 1)
                        formatted_activation_payload[
                            ((row_pair*2+row_in_pair)*8+k_lane)*8 +: 8] =
                            aligned_activation[row_pair*128+row_in_pair*64+k_lane*8 +: 8];
        end else if (response_mode == MODE_MIXED_W4) begin
            formatted_activation_payload = response_activation_physical;
        end else if (response_mode == MODE_W4A8) begin
            for (integer row = 0; row < 8; row = row + 1)
                for (integer k_lane = 0; k_lane < 16; k_lane = k_lane + 1)
                    formatted_activation_payload[(row*16+k_lane)*8 +: 8] =
                        aligned_activation[row*128+k_lane*8 +: 8];
        end else begin
            for (integer row = 0; row < 8; row = row + 1)
                for (integer pair = 0; pair < 16; pair = pair + 1) begin
                    formatted_activation_payload[(row*32+pair*2)*4 +: 4] =
                        aligned_activation[row*128+pair*8 +: 4];
                    formatted_activation_payload[(row*32+pair*2+1)*4 +: 4] =
                        aligned_activation[row*128+pair*8+4 +: 4];
                end
        end
        for (integer beat = 0; beat < 8; beat = beat + 1)
            for (integer pair = 0; pair < 2; pair = pair + 1)
                for (integer column = 0; column < 8; column = column + 1) begin
                    formatted_weight_base[((beat*4+pair*2)*8+column)*4 +: 4] =
                        response_base_words[beat*128+(pair*8+column)*8 +: 4];
                    formatted_weight_base[((beat*4+pair*2+1)*8+column)*4 +: 4] =
                        response_base_words[beat*128+(pair*8+column)*8+4 +: 4];
                    if (response_mode == MODE_W8A8 && beat < 2) begin
                        formatted_weight_enhancement[
                            ((beat*4+pair*2)*8+column)*4 +: 4] =
                            response_enhancement_words[
                                beat*128+(pair*8+column)*8 +: 4];
                        formatted_weight_enhancement[
                            ((beat*4+pair*2+1)*8+column)*4 +: 4] =
                            response_enhancement_words[
                                beat*128+(pair*8+column)*8+4 +: 4];
                    end
                end
        if (response_mode == MODE_W8A8) begin
            for (integer k = 0; k < 8; k = k + 1)
                for (integer column = 0; column < 8; column = column + 1) begin
                    // QKV stores two signed radix-16 digits; the compact PE
                    // interface carries the resulting signed A8xW8 byte.
                    formatted_weight_payload[(k*8+column)*8 +: 8] =
                        combine_signed_digits(
                            formatted_weight_base[(k*8+column)*4 +: 4],
                            formatted_weight_enhancement[(k*8+column)*4 +: 4]);
                end
        end else if (response_mode == MODE_MIXED_W4 && response_reuse_panel) begin
            formatted_weight_payload = held_mixed_weight_payload;
        end else begin
            formatted_weight_payload = formatted_weight_base;
        end
    end

    assign req_valid = response_fifo_output_valid && !abort_pending && !abort_request;
    assign response_fifo_output_ready = abort_pending || abort_request || req_ready;
    assign req_mode = response_mode;
    assign req_activation_payload = formatted_activation_payload;
    assign req_weight_payload = formatted_weight_payload;
    assign req_row_mask = response_row_mask;
    assign req_k_mask = response_k_mask;
    assign req_col_mask = response_col_mask;
    assign req_first_k_step = response_first;
    assign req_last_k_step = response_last;
    assign req_mixed_phase = response_mixed_phase;
    assign req_mixed_phase_first = response_mixed_phase_first;
    assign req_mixed_a8_rows = response_mixed_a8_rows;
    assign req_activation_scales = response_activation_scales;
    assign req_weight_scales = response_weight_scales;
    assign req_tag = response_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            scale_cache_valid <= 1'b0;
            activation_scale_cache <= '0;
            weight_scale_cache <= '0;
            scale_row_batch_id <= '0;
            scale_stripe_id <= '0;
            response_pending <= '0;
            pending_meta[0] <= '0;
            pending_meta[1] <= '0;
            pending_panel_start_bank[0] <= 3'd0;
            pending_panel_start_bank[1] <= 3'd0;
            pending_panel_bank_count[0] <= 3'd4;
            pending_panel_bank_count[1] <= 3'd4;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            held_mixed_weight_payload <= '0;
            held_mixed_weight_valid <= 1'b0;
`ifndef SYNTHESIS
            accepted_issue_count <= 64'd0;
            accepted_request_count <= 64'd0;
`endif
        end else begin
            abort_ack <= 1'b0;
            response_pending <= {response_pending[0], issue_accept};
            pending_meta[1] <= pending_meta[0];
            pending_panel_start_bank[1] <= pending_panel_start_bank[0];
            pending_panel_bank_count[1] <= pending_panel_bank_count[0];
            if (scale_cfg_valid && scale_cfg_ready) begin
                scale_cache_valid <= 1'b1;
                activation_scale_cache <= scale_cfg_activation;
                weight_scale_cache <= scale_cfg_weight;
                scale_row_batch_id <= scale_cfg_row_batch_id;
                scale_stripe_id <= scale_cfg_stripe_id;
            end
            if (issue_accept) begin
                pending_meta[0] <= {issue_mode, issue_row_mask, issue_k_mask,
                    issue_column_mask, issue_first_k_step, issue_last_k_step,
                    issue_mixed_phase, issue_mixed_phase_first,
                    issue_mixed_a8_rows, issue_reuse_panel,
                    issue_tag, issue_activation_byte_address[3:0],
                    issue_mode == MODE_MIXED_W4 ? issue_activation_scales :
                        activation_scale_cache,
                    issue_mode == MODE_MIXED_W4 ? issue_weight_scales :
                        weight_scale_cache};
                pending_panel_start_bank[0] <= issue_panel_start_bank;
                pending_panel_bank_count[0] <= issue_panel_bank_count;
`ifndef SYNTHESIS
                accepted_issue_count <= accepted_issue_count + 64'd1;
`endif
            end
            if (req_valid && req_ready) begin
`ifndef SYNTHESIS
                accepted_request_count <= accepted_request_count + 64'd1;
`endif
                if (response_mode == MODE_MIXED_W4 &&
                    !response_reuse_panel) begin
                    held_mixed_weight_payload <= formatted_weight_payload;
                    held_mixed_weight_valid <= 1'b1;
                end
            end
            if (abort_request) begin
                abort_pending <= 1'b1;
                held_mixed_weight_valid <= 1'b0;
            end
            if ((abort_pending || abort_request) && !(|response_pending) &&
                response_fifo_occupancy == 0) begin
                abort_pending <= 1'b0;
                scale_cache_valid <= 1'b0;
                held_mixed_weight_valid <= 1'b0;
                abort_ack <= 1'b1;
            end
        end
    end

    initial begin
        if (TAG_WIDTH < 1)
            $error("matmul_operand_path parameter relation is invalid");
    end

`ifndef SYNTHESIS
    logic stalled_request;
    logic [2+2*1024+8+32+8+2+1+1+8+2*128+TAG_WIDTH-1:0]
        stalled_request_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_request <= 1'b0;
            stalled_request_payload <= '0;
        end else begin
            assert (!read_response_valid || response_pending[1])
                else $error("matmul_operand_path received an unowned SRAM response");
            assert (!response_pending[1] || read_response_valid)
                else $error("matmul_operand_path did not receive the fixed-latency SRAM response");
            if (response_fifo_input_valid)
                assert (response_fifo_input_ready)
                    else $error("matmul_operand_path response FIFO credit was exhausted");
            if (issue_accept)
                assert ($countones(panel_read_port_mask) ==
                        (issue_reuse_panel ? 4'd0 :
                         panel_base_beat_count +
                         (issue_mode == MODE_W8A8 ? 4'd2 : 4'd0)))
                    else $error("matmul_operand_path assigned two words to one panel port");
            if (req_valid && req_ready && response_mode == MODE_MIXED_W4 &&
                response_reuse_panel)
                assert (held_mixed_weight_valid)
                    else $error("matmul_operand_path reused a mixed weight before phase 0 acceptance");
            if (stalled_request && !abort_pending && !abort_request)
                assert (req_valid && {req_mode, req_activation_payload,
                    req_weight_payload, req_row_mask, req_k_mask, req_col_mask,
                    req_first_k_step, req_last_k_step, req_mixed_phase,
                    req_mixed_phase_first, req_mixed_a8_rows,
                    req_activation_scales,
                    req_weight_scales, req_tag} == stalled_request_payload)
                    else $error("matmul_operand_path changed a stalled PE request");
            stalled_request <= req_valid && !req_ready &&
                !abort_pending && !abort_request;
            stalled_request_payload <= {req_mode, req_activation_payload,
                req_weight_payload, req_row_mask, req_k_mask, req_col_mask,
                req_first_k_step, req_last_k_step, req_mixed_phase,
                req_mixed_phase_first, req_mixed_a8_rows,
                req_activation_scales,
                req_weight_scales, req_tag};
        end
    end
`endif
endmodule

`default_nettype wire
