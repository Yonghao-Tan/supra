`default_nettype none

// Traverses N8 stripes, precision segments, row batches, and K steps. Final
// tile descriptors remain ordered through the shared PE and rescale resources.
module phase_shared_compute_scheduler #(
    parameter integer TAG_WIDTH = 16,
    parameter integer RESULT_DEPTH = 4,
    parameter bit TRUSTED_START_CONFIGURATION = 1'b0
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       abort_request,
    output logic                       abort_ack,
    output logic                       datapath_abort_request,
    input  logic                       datapath_abort_ack,

    input  logic                       start_valid,
    output logic                       start_ready,
    input  logic [15:0]                start_input_features,
    input  logic [15:0]                start_output_features,
    input  logic [1:0]                 start_segment_count,
    input  logic [17:0]                start_segment_row_base,
    input  logic [17:0]                start_segment_row_count,
    input  logic [5:0]                 start_segment_mode,
    input  logic [95:0]                start_segment_activation_base_byte_offset,
    input  logic [47:0]                start_row_enable,
    input  logic                       start_mixed_group_enable,
    input  logic [5:0]                 start_compute_group_count,
    input  logic [47:0]                start_row_precision_a8,
    input  logic [287:0]               start_row_compute_group,
    input  logic [143:0]               start_row_pe_slot,
    input  logic [95:0]                start_row_phase_mask,
    input  hardware_types_pkg::ffn_batch_pair_config_t start_batch_group,
    input  logic [2:0]                 start_panel_bank_count,
    input  logic [9:0]                 start_panel_plane_rows,
    input  logic                       start_qkv_padded_layout,
    output logic                       configuration_error,

    output logic                       issue_valid,
    output logic [2:0]                 issue_batch_index,
    input  logic                       issue_ready,
    output logic [1:0]                 issue_mode,
    output logic [5:0]                 issue_physical_row_base,
    output logic [3:0]                 issue_active_rows,
    output logic [15:0]                issue_input_feature_base,
    output logic [15:0]                issue_output_feature_base,
    output logic [7:0]                 issue_row_mask,
    output logic [31:0]                issue_k_mask,
    output logic [7:0]                 issue_column_mask,
    output logic                       issue_first_k_step,
    output logic                       issue_last_k_step,
    output logic                       issue_last_for_stripe,
    output logic                       issue_mixed_phase,
    output logic                       issue_mixed_phase_first,
    output logic [7:0]                 issue_mixed_a8_rows,
    output logic                       issue_reuse_panel,
    output logic                       issue_explicit_activation_rows,
    output logic [47:0]                issue_activation_physical_rows,
    output logic [127:0]               issue_activation_lane_word_index,
    output logic [TAG_WIDTH-1:0]       issue_tag,
    output logic [15:0]                issue_row_batch_id,
    output logic [15:0]                issue_stripe_id,
    output logic [31:0]                issue_activation_byte_address,
    output logic                       issue_qkv_padded_layout,
    output logic [2:0]                 issue_panel_start_bank,
    output logic [9:0]                 issue_panel_word_row,
    output logic [2:0]                 issue_panel_bank_count,
    output logic [9:0]                 issue_panel_plane_rows,

    input  logic                       tile_result_valid,
    output logic                       tile_result_ready,
    input  logic [64*16-1:0]           tile_result_bf16,
    input  logic [63:0]                tile_result_mask,
    input  logic [TAG_WIDTH-1:0]       tile_result_tag,
    output logic                       result_valid,
    output logic [2:0]                 result_batch_index,
    input  logic                       result_ready,
    output logic [64*16-1:0]           result_bf16,
    output logic [63:0]                result_mask,
    output logic [5:0]                 result_physical_row_base,
    output logic [15:0]                result_output_feature_base,
    output logic [7:0]                 result_row_mask,
    output logic [7:0]                 result_column_mask,
    output logic [TAG_WIDTH-1:0]       result_tag,

    output logic                       done_valid,
    input  logic                       done_ready,
    output logic [63:0]                accepted_issue_count,
    output logic [63:0]                accepted_result_count,
    output logic [63:0]                completed_tile_count
);
`ifdef SYNTHESIS
    always_comb begin
        accepted_issue_count = '0;
        accepted_result_count = '0;
        completed_tile_count = '0;
    end
`endif
    localparam logic [1:0] MODE_W4A4 = 2'd0;
    localparam logic [1:0] MODE_W4A8 = 2'd1;
    localparam logic [1:0] MODE_W8A8 = 2'd2;
    localparam logic [1:0] MODE_MIXED_W4 = 2'd3;
    localparam integer PTR_WIDTH = $clog2(RESULT_DEPTH);
    localparam integer DESC_WIDTH = TAG_WIDTH+3+6+16+8+8;

    typedef enum logic [2:0] {IDLE, BUILD_GROUPS, LOAD_GROUP, ISSUE,
                              DRAIN_RESULTS, HOLD_DONE,
                              ABORT_DRAIN} state_t;
    state_t state;

    logic [15:0] input_features;
    logic [15:0] output_features;
    logic [1:0] segment_count;
    logic [17:0] segment_row_base;
    logic [17:0] segment_row_count;
    logic [5:0] segment_mode;
    logic [95:0] segment_activation_base_byte_offset;
    logic [47:0] row_enable;
    logic [1:0] segment_index;
    logic [5:0] row_offset;
    logic [15:0] k_base;
    logic [15:0] n_base;
    logic [2:0] panel_start_bank;
    logic [9:0] panel_word_row;
    logic [2:0] panel_bank_count;
    logic [9:0] panel_plane_rows;
    logic qkv_padded_layout;
    logic [TAG_WIDTH-1:0] next_tag;
    logic mixed_group_enable;
    logic [5:0] compute_group_count;
    logic [47:0] row_precision_a8;
    logic [287:0] row_compute_group;
    logic [143:0] row_pe_slot;
    logic [95:0] row_phase_mask;
    logic [5:0] build_row;
    logic [5:0] build_last_group;
    logic [2:0] batch_count;
    logic compact_a4_layout;
    logic [2:0] build_batch;
    logic [5:0] batch_group_offset;
    logic [5:0] current_batch_group_count;
    logic [5:0] group_index;
    logic mixed_phase;
    logic [5:0] group_row_base [0:47];
    logic [3:0] group_a8_count [0:47];
    logic [3:0] group_phase0_a4_count [0:47];
    logic [3:0] group_phase1_a4_count [0:47];
    logic [8:0] group_unit_base [0:47];
    logic [2:0] group_batch [0:47];
    logic [2:0] loaded_group_batch;
    logic [5:0] loaded_group_row_base;
    logic [3:0] loaded_group_a8_count;
    logic [3:0] loaded_group_phase0_a4_count;
    logic [3:0] loaded_group_phase1_a4_count;
    logic [7:0] mixed_phase0_mask;
    logic [7:0] mixed_phase1_mask;
    logic [7:0] mixed_a8_mask;
    logic [7:0] mixed_phase1_a4_mask;
    logic mixed_phase1_active;
    logic mixed_single_phase_a4;
    logic [15:0] mixed_k_block_count;
    logic [24:0] mixed_group_word_base;
    logic [24:0] mixed_k_word_base;

    logic [5:0] current_segment_base;
    logic [5:0] current_segment_count;
    logic [5:0] current_physical_row;
    logic [1:0] current_mode;
    logic [31:0] current_activation_base_byte_offset;
    logic [5:0] rows_remaining;
    logic [15:0] k_step;
    logic [15:0] active_k;
    logic [15:0] active_columns;
    logic [5:0] active_token_count;
    logic [15:0] issue_bytes;
    logic [15:0] row_bytes;
    logic [32:0] row_batch_byte_offset;
    logic [32:0] issue_byte_offset;
    logic [32:0] qkv_batch_byte_offset;
    logic [32:0] qkv_group_byte_offset;
    logic [3:0] panel_word_step;
    logic [3:0] panel_advance_sum;
    logic last_scheduled_tile;
    logic checked_start_configuration_valid;
    logic start_configuration_valid;
    logic [6:0] configured_row_end;
    logic [32:0] configured_activation_end;
    logic [15:0] configured_row_bytes;

    logic [DESC_WIDTH-1:0] descriptor_memory [0:RESULT_DEPTH-1];
    logic [PTR_WIDTH-1:0] descriptor_write_pointer;
    logic [PTR_WIDTH-1:0] descriptor_read_pointer;
    logic [PTR_WIDTH:0] descriptor_count;
    logic [DESC_WIDTH-1:0] descriptor_head;
    logic descriptor_push;
    logic descriptor_push_second;
    logic [1:0] descriptor_push_count;
    logic descriptor_pop;
    logic descriptor_credit;
    logic [TAG_WIDTH-1:0] descriptor_tag;
    logic [5:0] descriptor_row_base;
    logic [15:0] descriptor_n_base;
    logic [7:0] descriptor_row_mask;
    logic [7:0] descriptor_col_mask;

    function automatic logic [PTR_WIDTH-1:0] next_descriptor_pointer(
        input logic [PTR_WIDTH-1:0] pointer
    );
        if (pointer == PTR_WIDTH'(RESULT_DEPTH-1))
            next_descriptor_pointer = '0;
        else
            next_descriptor_pointer = pointer + 1'b1;
    endfunction

    assign current_segment_base =
        segment_row_base[segment_index*6 +: 6];
    assign current_segment_count = segment_row_count[segment_index*6 +: 6];
    assign current_physical_row = current_segment_base + row_offset;
    assign current_mode = mixed_group_enable ?
        (mixed_single_phase_a4 ? MODE_W4A4 : MODE_MIXED_W4) :
        segment_mode[segment_index*2 +: 2];
    assign current_activation_base_byte_offset = segment_activation_base_byte_offset[segment_index*32 +: 32];
    assign rows_remaining = current_segment_count - row_offset;
    assign mixed_a8_mask = 8'hff >> (8 - loaded_group_a8_count);
    assign mixed_phase0_mask = 8'hff >>
        (8 - loaded_group_a8_count - loaded_group_phase0_a4_count);
    assign mixed_phase1_mask = 8'hff >>
        (8 - loaded_group_a8_count - loaded_group_phase1_a4_count);
    assign mixed_phase1_a4_mask = mixed_phase1_mask & ~mixed_a8_mask;
    assign mixed_phase1_active = loaded_group_a8_count != 0 ||
        loaded_group_phase1_a4_count != 0;
    assign mixed_single_phase_a4 = loaded_group_a8_count == 0 &&
        loaded_group_phase1_a4_count == 0;
    assign mixed_k_block_count = input_features >> 5;
    assign active_token_count = mixed_group_enable ?
        (mixed_phase ?
            loaded_group_a8_count + loaded_group_phase1_a4_count :
            loaded_group_a8_count + loaded_group_phase0_a4_count) :
        (rows_remaining >= 6'd8 ? 6'd8 : rows_remaining);
    assign active_columns = output_features - n_base >= 16'd8 ?
                            16'd8 : output_features - n_base;
    assign k_step = current_mode == MODE_W4A4 ||
                    current_mode == MODE_MIXED_W4 ? 16'd32 :
                    current_mode == MODE_W4A8 ? 16'd16 : 16'd8;
    assign active_k = input_features - k_base >= k_step ?
                      k_step : input_features - k_base;
    assign row_bytes = current_mode == MODE_W4A4 ?
                       input_features >> 1 : input_features;
    assign issue_bytes = current_mode == MODE_W8A8 ?
                         {7'd0, active_token_count, 3'b000} :
                         {6'd0, active_token_count, 4'b0000};

    always_comb begin
        checked_start_configuration_valid = start_input_features != 0 &&
            start_output_features != 0 &&
            (start_mixed_group_enable ?
                (start_segment_count == 0 &&
                 start_compute_group_count != 0 &&
                 start_compute_group_count <= 6'd4 &&
                 (start_input_features == 16'd4096 ||
                  start_input_features == 16'd12288)) :
                start_segment_count != 0) &&
            (!start_batch_group.enable ||
             (start_mixed_group_enable &&
              (start_input_features == 16'd4096 ||
               (start_input_features == 16'd12288 &&
                start_batch_group.batch_count == 3'd2 &&
                start_batch_group.compact_a4_layout &&
                start_compute_group_count <= 6'd3 &&
                start_batch_group.second_row_config.compute_group_count <= 6'd3)) &&
              start_batch_group.batch_count >= 3'd2 &&
              start_batch_group.batch_count <= 3'd6 &&
              start_batch_group.second_row_config.mixed_group_enable &&
              !start_batch_group.second_row_config.operator_has_w8 &&
              start_batch_group.second_row_config.row_enable != 0 &&
              start_batch_group.second_row_config.compute_group_count != 0 &&
              start_batch_group.second_row_config.compute_group_count <= 6'd4 &&
              (start_batch_group.batch_count < 3'd3 ||
               (start_batch_group.third_row_config.mixed_group_enable &&
                !start_batch_group.third_row_config.operator_has_w8 &&
                start_batch_group.third_row_config.row_enable != 0 &&
                start_batch_group.third_row_config.compute_group_count != 0 &&
                start_batch_group.third_row_config.compute_group_count <= 6'd4)) &&
              (start_batch_group.batch_count < 3'd4 ||
               (start_batch_group.fourth_row_config.mixed_group_enable &&
                !start_batch_group.fourth_row_config.operator_has_w8 &&
                start_batch_group.fourth_row_config.row_enable != 0 &&
                start_batch_group.fourth_row_config.compute_group_count != 0 &&
                start_batch_group.fourth_row_config.compute_group_count <= 6'd4)) &&
              (start_batch_group.batch_count < 3'd5 ||
               (start_batch_group.fifth_row_config.mixed_group_enable &&
                !start_batch_group.fifth_row_config.operator_has_w8 &&
                start_batch_group.fifth_row_config.row_enable != 0 &&
                start_batch_group.fifth_row_config.compute_group_count != 0 &&
                start_batch_group.fifth_row_config.compute_group_count <= 6'd4)) &&
              (start_batch_group.batch_count < 3'd6 ||
               (start_batch_group.sixth_row_config.mixed_group_enable &&
                !start_batch_group.sixth_row_config.operator_has_w8 &&
                start_batch_group.sixth_row_config.row_enable != 0 &&
                start_batch_group.sixth_row_config.compute_group_count != 0 &&
                start_batch_group.sixth_row_config.compute_group_count <= 6'd4)) &&
              ({1'b0, start_compute_group_count} +
               {1'b0, start_batch_group.second_row_config.compute_group_count} +
               (start_batch_group.batch_count >= 3'd3 ?
                    {1'b0, start_batch_group.third_row_config.compute_group_count} : 7'd0) +
               (start_batch_group.batch_count >= 3'd4 ?
                    {1'b0, start_batch_group.fourth_row_config.compute_group_count} : 7'd0) +
               (start_batch_group.batch_count >= 3'd5 ?
                    {1'b0, start_batch_group.fifth_row_config.compute_group_count} : 7'd0) +
               (start_batch_group.batch_count >= 3'd6 ?
                    {1'b0, start_batch_group.sixth_row_config.compute_group_count} : 7'd0)
               <= (start_input_features == 16'd12288 ? 7'd6 : 7'd18)))) &&
            start_input_features[4:0] == 0 &&
            (start_panel_bank_count == 3'd2 ||
             start_panel_bank_count == 3'd4 ||
             (start_panel_bank_count == 3'd6 &&
              start_input_features == 16'd12288)) &&
            start_panel_plane_rows != 0 && start_panel_plane_rows <= 10'd768 &&
            start_panel_plane_rows == (start_panel_bank_count == 3'd6 ?
                10'd512 :
                start_panel_bank_count == 3'd2 ?
                10'(start_input_features >> 3) :
                10'(start_input_features >> 4));
        configured_row_end = 7'd0;
        configured_activation_end = {1'b0, start_segment_activation_base_byte_offset[31:0]};
        configured_row_bytes = 16'd0;
        for (integer index = 0; index < 3; index = index + 1) begin
            if (index < start_segment_count) begin
                configured_row_bytes =
                    start_segment_mode[index*2 +: 2] == MODE_W4A4 ?
                    start_input_features >> 1 : start_input_features;
                if (start_segment_row_count[index*6 +: 6] == 0 ||
                    start_segment_mode[index*2 +: 2] > MODE_W8A8 ||
                    (start_segment_mode[index*2 +: 2] == MODE_W8A8) !=
                        (start_panel_bank_count == 3'd2) ||
                    {1'b0, start_segment_row_base[index*6 +: 6]} != configured_row_end ||
                    start_segment_activation_base_byte_offset[index*32 +: 4] != 0 ||
                    {1'b0, start_segment_activation_base_byte_offset[index*32 +: 32]} !=
                        configured_activation_end)
                    checked_start_configuration_valid = 1'b0;
                configured_row_end =
                    {1'b0, start_segment_row_base[index*6 +: 6]} +
                    {1'b0, start_segment_row_count[index*6 +: 6]};
                if (configured_row_end > 7'd48)
                    checked_start_configuration_valid = 1'b0;
                configured_activation_end = configured_activation_end +
                    33'(start_segment_row_count[index*6 +: 6]) *
                    33'(configured_row_bytes);
                if (configured_activation_end > 33'd393216)
                    checked_start_configuration_valid = 1'b0;
            end
        end
    end

    assign start_configuration_valid = TRUSTED_START_CONFIGURATION ?
        1'b1 : checked_start_configuration_valid;

    assign descriptor_push_second = mixed_group_enable &&
        issue_last_k_step && |mixed_phase1_a4_mask;
    assign descriptor_push_count = descriptor_push ?
        (descriptor_push_second ? 2'd2 : 2'd1) : 2'd0;
    assign descriptor_credit = descriptor_count +
        (descriptor_push_second ? 2'd2 : 2'd1) <=
        (PTR_WIDTH+1)'(RESULT_DEPTH) +
        (descriptor_pop ? (PTR_WIDTH+1)'(1) : (PTR_WIDTH+1)'(0));
    assign issue_valid = state == ISSUE && !abort_request &&
        (!issue_last_k_step || descriptor_credit);
    assign issue_mode = current_mode;
    assign issue_physical_row_base = mixed_group_enable ?
        loaded_group_row_base : current_segment_base + row_offset;
    assign issue_active_rows = active_token_count[3:0];
    assign issue_input_feature_base = k_base;
    assign issue_output_feature_base = n_base;
    assign issue_row_mask = mixed_group_enable ?
        (mixed_phase ? mixed_phase1_mask : mixed_phase0_mask) :
        (8'hff >> (8 - active_token_count)) &
            8'(row_enable >> (current_segment_base + row_offset));
    assign issue_k_mask = 32'hffff_ffff >> (32 - active_k);
    assign issue_column_mask = 8'hff >> (8 - active_columns);
    assign issue_first_k_step = mixed_group_enable ?
        k_base == 0 && !mixed_phase : k_base == 0;
    assign issue_last_k_step = mixed_group_enable ?
        k_base + active_k == input_features &&
            (mixed_phase || !mixed_phase1_active) :
        k_base + active_k == input_features;
    assign issue_last_for_stripe = mixed_group_enable ?
        issue_last_k_step && group_index + 6'd1 >= compute_group_count :
        issue_last_k_step && row_offset + active_token_count >= current_segment_count &&
            segment_index + 1'b1 >= segment_count;
    assign issue_mixed_phase = mixed_group_enable &&
        !mixed_single_phase_a4 && mixed_phase;
    assign issue_mixed_phase_first = mixed_group_enable &&
        !mixed_single_phase_a4 && k_base == 0;
    assign issue_mixed_a8_rows = mixed_group_enable &&
        !mixed_single_phase_a4 ? mixed_a8_mask : 8'd0;
    assign issue_reuse_panel = mixed_group_enable &&
        !mixed_single_phase_a4 && mixed_phase;
    assign issue_explicit_activation_rows = mixed_group_enable;
    assign issue_tag = next_tag;
    assign issue_batch_index = mixed_group_enable ? loaded_group_batch : 3'd0;
    assign issue_row_batch_id = mixed_group_enable ?
        {9'd0, group_index, mixed_phase} :
        {8'd0, segment_index, row_offset};
    assign issue_stripe_id = n_base >> 3;
    always_comb begin
        case (current_physical_row >> 3)
            6'd1: qkv_batch_byte_offset = 33'(input_features) << 3;
            6'd2: qkv_batch_byte_offset = 33'(input_features) << 4;
            6'd3: qkv_batch_byte_offset = (33'(input_features) << 4) +
                (33'(input_features) << 3);
            6'd4: qkv_batch_byte_offset = 33'(input_features) << 5;
            6'd5: qkv_batch_byte_offset = (33'(input_features) << 5) +
                (33'(input_features) << 3);
            default: qkv_batch_byte_offset = '0;
        endcase
        qkv_group_byte_offset = current_mode == MODE_W4A4 ?
            (33'(k_base >> 5) << 7) : (33'(k_base >> 4) << 7);
    end
    assign issue_activation_byte_address = mixed_group_enable ? 32'd0 :
        qkv_padded_layout ?
        segment_activation_base_byte_offset[31:0] + qkv_batch_byte_offset[31:0] +
            qkv_group_byte_offset[31:0] +
            {25'd0, current_physical_row[2:0], 4'b0000} :
        current_activation_base_byte_offset + issue_byte_offset[31:0];
    assign issue_qkv_padded_layout = qkv_padded_layout && !mixed_group_enable;
    assign issue_panel_start_bank = panel_start_bank;
    assign issue_panel_word_row = panel_word_row;
    assign issue_panel_bank_count = panel_bank_count;
    assign issue_panel_plane_rows = panel_plane_rows;
    always_comb begin
        issue_activation_physical_rows = '0;
        issue_activation_lane_word_index = '0;
        for (integer slot = 0; slot < 8; slot++) begin
            if (mixed_phase && slot >= integer'(loaded_group_a8_count))
                issue_activation_physical_rows[slot*6 +: 6] =
                    loaded_group_row_base +
                    {2'd0, loaded_group_phase0_a4_count} + 6'(slot);
            else
                issue_activation_physical_rows[slot*6 +: 6] =
                    loaded_group_row_base + 6'(slot);
            issue_activation_lane_word_index[slot*16 +: 16] =
                segment_activation_base_byte_offset[19:4] + 16'(mixed_k_word_base) +
                (mixed_phase ?
                    {8'd0, loaded_group_a8_count} +
                        {8'd0, loaded_group_phase0_a4_count} : 16'd0) +
                16'(slot);
        end
    end
    assign panel_word_step = current_mode == MODE_W4A4 ||
                             current_mode == MODE_MIXED_W4 ? 4'd8 :
                             current_mode == MODE_W4A8 ? 4'd4 : 4'd2;
    assign panel_advance_sum = {1'b0, panel_start_bank} + panel_word_step;
    assign last_scheduled_tile = mixed_group_enable ?
        group_index + 6'd1 >= compute_group_count &&
            n_base + active_columns >= output_features :
        row_offset + active_token_count >= current_segment_count &&
            segment_index + 1'b1 >= segment_count &&
            n_base + active_columns >= output_features;
    assign descriptor_push = issue_valid && issue_ready && issue_last_k_step;

    assign descriptor_head = descriptor_memory[descriptor_read_pointer];
    assign {descriptor_tag, result_batch_index, descriptor_row_base, descriptor_n_base,
            descriptor_row_mask, descriptor_col_mask} = descriptor_head;
    assign result_valid = descriptor_count != 0 && tile_result_valid &&
                          tile_result_tag == descriptor_tag && !abort_request;
    assign tile_result_ready = (state != ABORT_DRAIN && !abort_request) ?
        (descriptor_count != 0 && tile_result_tag == descriptor_tag && result_ready) : 1'b0;
    assign result_bf16 = tile_result_bf16;
    assign result_mask = tile_result_mask;
    assign result_physical_row_base = descriptor_row_base;
    assign result_output_feature_base = descriptor_n_base;
    assign result_row_mask = descriptor_row_mask;
    assign result_column_mask = descriptor_col_mask;
    assign result_tag = descriptor_tag;
    assign descriptor_pop = result_valid && result_ready;
    assign start_ready = state == IDLE && descriptor_count == 0 && !abort_request;
    assign datapath_abort_request = state == ABORT_DRAIN;
    assign done_valid = state == HOLD_DONE;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            configuration_error <= 1'b0;
            input_features <= 16'd0;
            output_features <= 16'd0;
            segment_count <= 2'd0;
            segment_row_base <= 18'd0;
            segment_row_count <= 18'd0;
            segment_mode <= 6'd0;
            segment_activation_base_byte_offset <= 96'd0;
            row_enable <= 48'd0;
            segment_index <= 2'd0;
            row_offset <= 6'd0;
            k_base <= 16'd0;
            n_base <= 16'd0;
            panel_start_bank <= 3'd0;
            panel_word_row <= 10'd0;
            row_batch_byte_offset <= 33'd0;
            issue_byte_offset <= 33'd0;
            panel_bank_count <= 3'd4;
            panel_plane_rows <= 10'd0;
            qkv_padded_layout <= 1'b0;
            next_tag <= '0;
            mixed_group_enable <= 1'b0;
            compute_group_count <= '0;
            row_precision_a8 <= '0;
            row_compute_group <= '0;
            row_pe_slot <= '0;
            row_phase_mask <= '0;
            build_row <= '0;
            build_last_group <= 6'h3f;
            batch_count <= 3'd1;
            compact_a4_layout <= 1'b0;
            build_batch <= 3'd0;
            batch_group_offset <= '0;
            current_batch_group_count <= '0;
            loaded_group_batch <= 2'd0;
            group_index <= '0;
            mixed_phase <= 1'b0;
            loaded_group_row_base <= '0;
            loaded_group_a8_count <= '0;
            loaded_group_phase0_a4_count <= '0;
            loaded_group_phase1_a4_count <= '0;
            mixed_group_word_base <= '0;
            mixed_k_word_base <= '0;
            descriptor_write_pointer <= '0;
            descriptor_read_pointer <= '0;
            descriptor_count <= '0;
`ifndef SYNTHESIS
            accepted_issue_count <= 64'd0;
            accepted_result_count <= 64'd0;
            completed_tile_count <= 64'd0;
`endif
        end else begin
            abort_ack <= 1'b0;
            configuration_error <= 1'b0;
            if (descriptor_push) begin
                descriptor_memory[descriptor_write_pointer] <=
                    {next_tag, issue_batch_index,
                     mixed_group_enable ? loaded_group_row_base :
                         current_segment_base + row_offset,
                     n_base,
                     mixed_group_enable ? mixed_phase0_mask : issue_row_mask,
                     issue_column_mask};
                if (descriptor_push_second)
                    descriptor_memory[next_descriptor_pointer(
                        descriptor_write_pointer)] <= {
                        next_tag, issue_batch_index,
                        loaded_group_row_base +
                            {2'd0, loaded_group_a8_count} +
                            {2'd0, loaded_group_phase0_a4_count},
                        n_base, mixed_phase1_a4_mask, issue_column_mask};
                descriptor_write_pointer <= descriptor_push_second ?
                    next_descriptor_pointer(next_descriptor_pointer(
                        descriptor_write_pointer)) :
                    next_descriptor_pointer(descriptor_write_pointer);
            end
            if (descriptor_pop)
                descriptor_read_pointer <=
                    next_descriptor_pointer(descriptor_read_pointer);
            descriptor_count <= descriptor_count +
                (PTR_WIDTH+1)'(descriptor_push_count) -
                (descriptor_pop ? (PTR_WIDTH+1)'(1) : (PTR_WIDTH+1)'(0));
`ifndef SYNTHESIS
            if (issue_valid && issue_ready)
                accepted_issue_count <= accepted_issue_count + 64'd1;
`endif
            if (descriptor_pop) begin
`ifndef SYNTHESIS
                accepted_result_count <= accepted_result_count + 64'd1;
                completed_tile_count <= completed_tile_count + 64'd1;
`endif
            end

            case (state)
                IDLE: begin
                    if (abort_request) begin
                        abort_ack <= 1'b1;
                    end else if (start_valid && start_ready) begin
                        if (!start_configuration_valid) begin
                            configuration_error <= 1'b1;
                        end else begin
                            input_features <= start_input_features;
                            output_features <= start_output_features;
                            segment_count <= start_segment_count;
                            segment_row_base <= start_segment_row_base;
                            segment_row_count <= start_segment_row_count;
                            segment_mode <= start_segment_mode;
                            segment_activation_base_byte_offset <= start_segment_activation_base_byte_offset;
                            row_enable <= start_row_enable;
                            segment_index <= 2'd0;
                            row_offset <= 6'd0;
                            k_base <= 16'd0;
                            n_base <= 16'd0;
                            panel_start_bank <= 3'd0;
                            panel_word_row <= 10'd0;
                            row_batch_byte_offset <= 33'd0;
                            issue_byte_offset <= 33'd0;
                            panel_bank_count <= start_panel_bank_count;
                            panel_plane_rows <= start_panel_plane_rows;
                            qkv_padded_layout <= start_qkv_padded_layout;
                            mixed_group_enable <= start_mixed_group_enable;
                            compute_group_count <= start_compute_group_count +
                                (start_batch_group.enable ?
                                    start_batch_group.second_row_config.compute_group_count : 6'd0) +
                                (start_batch_group.enable && start_batch_group.batch_count >= 3'd3 ?
                                    start_batch_group.third_row_config.compute_group_count : 6'd0) +
                                (start_batch_group.enable && start_batch_group.batch_count >= 3'd4 ?
                                    start_batch_group.fourth_row_config.compute_group_count : 6'd0) +
                                (start_batch_group.enable && start_batch_group.batch_count >= 3'd5 ?
                                    start_batch_group.fifth_row_config.compute_group_count : 6'd0) +
                                (start_batch_group.enable && start_batch_group.batch_count >= 3'd6 ?
                                    start_batch_group.sixth_row_config.compute_group_count : 6'd0);
                            batch_count <= start_batch_group.enable ?
                                start_batch_group.batch_count : 3'd1;
                            compact_a4_layout <= start_batch_group.enable &&
                                start_batch_group.compact_a4_layout;
                            build_batch <= 3'd0;
                            batch_group_offset <= 6'd0;
                            current_batch_group_count <= start_compute_group_count;
                            row_precision_a8 <= start_row_precision_a8;
                            row_compute_group <= start_row_compute_group;
                            row_pe_slot <= start_row_pe_slot;
                            row_phase_mask <= start_row_phase_mask;
                            build_row <= '0;
                            build_last_group <= 6'h3f;
                            group_index <= '0;
                            mixed_phase <= 1'b0;
                            state <= start_mixed_group_enable ?
                                BUILD_GROUPS : ISSUE;
                        end
                    end
                end
                BUILD_GROUPS: begin
                    logic [5:0] row_group;
                    logic [2:0] row_slot;
                    logic [1:0] phase_mask;
                    logic row_a8;
                    row_group = row_compute_group[build_row*6 +: 6] +
                        batch_group_offset;
                    row_slot = row_pe_slot[build_row*3 +: 3];
                    phase_mask = row_phase_mask[build_row*2 +: 2];
                    row_a8 = row_precision_a8[build_row];
                    if (row_group != build_last_group) begin
                        group_row_base[row_group] <= build_row;
                        group_unit_base[row_group] <= compact_a4_layout ?
                            (9'(row_group) << 4) :
                            {2'b00, build_batch[0],
                             row_compute_group[build_row*6 +: 2], 4'b0000};
                        group_batch[row_group] <= build_batch;
                        group_a8_count[row_group] <= row_a8 ? 4'd1 : 4'd0;
                        group_phase0_a4_count[row_group] <=
                            !row_a8 && phase_mask[0] ? 4'd1 : 4'd0;
                        group_phase1_a4_count[row_group] <=
                            !row_a8 && phase_mask[1] ? 4'd1 : 4'd0;
                        build_last_group <= row_group;
                    end else begin
                        if (row_a8)
                            group_a8_count[row_group] <=
                                group_a8_count[row_group] + 4'd1;
                        else if (phase_mask[0])
                            group_phase0_a4_count[row_group] <=
                                group_phase0_a4_count[row_group] + 4'd1;
                        else
                            group_phase1_a4_count[row_group] <=
                                group_phase1_a4_count[row_group] + 4'd1;
                    end
                    if (build_row + 6'd1 ==
                        6'($countones(row_enable))) begin
                        if ({1'b0, build_batch} + 3'd1 < batch_count) begin
                            build_batch <= build_batch + 3'd1;
                            batch_group_offset <= batch_group_offset +
                                current_batch_group_count;
                            build_row <= '0;
                            build_last_group <= 6'h3f;
                            case (build_batch + 3'd1)
                                3'd1: begin
                                    row_enable <= start_batch_group.second_row_config.row_enable;
                                    row_precision_a8 <= start_batch_group.second_row_config.row_precision_a8;
                                    row_compute_group <= start_batch_group.second_row_config.row_compute_group;
                                    row_pe_slot <= start_batch_group.second_row_config.row_pe_slot;
                                    row_phase_mask <= start_batch_group.second_row_config.row_phase_mask;
                                    current_batch_group_count <=
                                        start_batch_group.second_row_config.compute_group_count;
                                end
                                3'd2: begin
                                    row_enable <= start_batch_group.third_row_config.row_enable;
                                    row_precision_a8 <= start_batch_group.third_row_config.row_precision_a8;
                                    row_compute_group <= start_batch_group.third_row_config.row_compute_group;
                                    row_pe_slot <= start_batch_group.third_row_config.row_pe_slot;
                                    row_phase_mask <= start_batch_group.third_row_config.row_phase_mask;
                                    current_batch_group_count <=
                                        start_batch_group.third_row_config.compute_group_count;
                                end
                                3'd3: begin
                                    row_enable <= start_batch_group.fourth_row_config.row_enable;
                                    row_precision_a8 <= start_batch_group.fourth_row_config.row_precision_a8;
                                    row_compute_group <= start_batch_group.fourth_row_config.row_compute_group;
                                    row_pe_slot <= start_batch_group.fourth_row_config.row_pe_slot;
                                    row_phase_mask <= start_batch_group.fourth_row_config.row_phase_mask;
                                    current_batch_group_count <=
                                        start_batch_group.fourth_row_config.compute_group_count;
                                end
                                3'd4: begin
                                    row_enable <= start_batch_group.fifth_row_config.row_enable;
                                    row_precision_a8 <= start_batch_group.fifth_row_config.row_precision_a8;
                                    row_compute_group <= start_batch_group.fifth_row_config.row_compute_group;
                                    row_pe_slot <= start_batch_group.fifth_row_config.row_pe_slot;
                                    row_phase_mask <= start_batch_group.fifth_row_config.row_phase_mask;
                                    current_batch_group_count <=
                                        start_batch_group.fifth_row_config.compute_group_count;
                                end
                                default: begin
                                    row_enable <= start_batch_group.sixth_row_config.row_enable;
                                    row_precision_a8 <= start_batch_group.sixth_row_config.row_precision_a8;
                                    row_compute_group <= start_batch_group.sixth_row_config.row_compute_group;
                                    row_pe_slot <= start_batch_group.sixth_row_config.row_pe_slot;
                                    row_phase_mask <= start_batch_group.sixth_row_config.row_phase_mask;
                                    current_batch_group_count <=
                                        start_batch_group.sixth_row_config.compute_group_count;
                                end
                            endcase
                        end else begin
                            group_index <= '0;
                            state <= LOAD_GROUP;
                        end
                    end else begin
                        build_row <= build_row + 6'd1;
                    end
                end
                LOAD_GROUP: begin
                    loaded_group_row_base <= group_row_base[group_index];
                    loaded_group_batch <= group_batch[group_index];
                    loaded_group_a8_count <= group_a8_count[group_index];
                    loaded_group_phase0_a4_count <=
                        group_phase0_a4_count[group_index];
                    loaded_group_phase1_a4_count <=
                        group_phase1_a4_count[group_index];
                    mixed_group_word_base <= input_features == 16'd4096 ?
                        (25'(group_unit_base[group_index]) << 7) :
                        ((25'(group_unit_base[group_index]) << 8) +
                         (25'(group_unit_base[group_index]) << 7));
                    mixed_k_word_base <= input_features == 16'd4096 ?
                        (25'(group_unit_base[group_index]) << 7) :
                        ((25'(group_unit_base[group_index]) << 8) +
                         (25'(group_unit_base[group_index]) << 7));
                    mixed_phase <= 1'b0;
                    k_base <= 16'd0;
                    panel_start_bank <= 3'd0;
                    panel_word_row <= 10'd0;
                    state <= ISSUE;
                end
                ISSUE: begin
                    if (abort_request) begin
                        state <= ABORT_DRAIN;
                    end else if (issue_valid && issue_ready) begin
                        if (mixed_group_enable) begin
                            if (!mixed_phase && mixed_phase1_active) begin
                                mixed_phase <= 1'b1;
                            end else begin
                                mixed_phase <= 1'b0;
                                if (k_base + active_k < input_features) begin
                                    k_base <= k_base + active_k;
                                    mixed_k_word_base <= mixed_k_word_base +
                                        25'd16;
                                    if (panel_bank_count == 3'd4) begin
                                        panel_start_bank <=
                                            {1'b0, panel_advance_sum[1:0]};
                                        panel_word_row <= panel_word_row +
                                            10'(panel_advance_sum >> 2);
                                    end else if (panel_advance_sum >= 4'd12) begin
                                        panel_start_bank <=
                                            3'(panel_advance_sum - 4'd12);
                                        panel_word_row <= panel_word_row + 10'd2;
                                    end else if (panel_advance_sum >= 4'd6) begin
                                        panel_start_bank <=
                                            3'(panel_advance_sum - 4'd6);
                                        panel_word_row <= panel_word_row + 10'd1;
                                    end else begin
                                        panel_start_bank <= panel_advance_sum[2:0];
                                    end
                                end else begin
                                    next_tag <= next_tag + 1'b1;
                                    if (last_scheduled_tile) begin
                                        state <= DRAIN_RESULTS;
                                    end else if (group_index + 6'd1 <
                                                 compute_group_count) begin
                                        group_index <= group_index + 6'd1;
                                        state <= LOAD_GROUP;
                                    end else begin
                                        n_base <= n_base + active_columns;
                                        group_index <= '0;
                                        state <= LOAD_GROUP;
                                    end
                                end
                            end
                        end else if (!issue_last_k_step) begin
                            k_base <= k_base + active_k;
                            issue_byte_offset <= issue_byte_offset + issue_bytes;
                            if (panel_bank_count == 3'd2) begin
                                panel_start_bank <= {2'b00, panel_advance_sum[0]};
                                panel_word_row <= panel_word_row +
                                    10'(panel_advance_sum >> 1);
                            end else if (panel_bank_count == 3'd4) begin
                                panel_start_bank <= {1'b0, panel_advance_sum[1:0]};
                                panel_word_row <= panel_word_row +
                                    10'(panel_advance_sum >> 2);
                            end else if (panel_advance_sum >= 4'd12) begin
                                panel_start_bank <= 3'(panel_advance_sum - 4'd12);
                                panel_word_row <= panel_word_row + 10'd2;
                            end else if (panel_advance_sum >= 4'd6) begin
                                panel_start_bank <= 3'(panel_advance_sum - 4'd6);
                                panel_word_row <= panel_word_row + 10'd1;
                            end else begin
                                panel_start_bank <= panel_advance_sum[2:0];
                            end
                        end else begin
                            next_tag <= next_tag + 1'b1;
                            k_base <= 16'd0;
                            panel_start_bank <= 3'd0;
                            panel_word_row <= 10'd0;
                            if (last_scheduled_tile) begin
                                state <= DRAIN_RESULTS;
                            end else if (row_offset + active_token_count < current_segment_count) begin
                                row_offset <= row_offset + active_token_count;
                                row_batch_byte_offset <= row_batch_byte_offset +
                                    {14'd0, row_bytes, 3'b000};
                                issue_byte_offset <= row_batch_byte_offset +
                                    {14'd0, row_bytes, 3'b000};
                            end else if (segment_index + 1'b1 < segment_count) begin
                                segment_index <= segment_index + 1'b1;
                                row_offset <= 6'd0;
                                row_batch_byte_offset <= 33'd0;
                                issue_byte_offset <= 33'd0;
                            end else begin
                                n_base <= n_base + active_columns;
                                segment_index <= 2'd0;
                                row_offset <= 6'd0;
                                row_batch_byte_offset <= 33'd0;
                                issue_byte_offset <= 33'd0;
                            end
                        end
                    end
                end
                DRAIN_RESULTS: begin
                    if (abort_request)
                        state <= ABORT_DRAIN;
                    else if (descriptor_pop && descriptor_count == 1)
                        state <= HOLD_DONE;
                end
                HOLD_DONE: begin
                    if (abort_request)
                        state <= ABORT_DRAIN;
                    else if (done_valid && done_ready)
                        state <= IDLE;
                end
                ABORT_DRAIN: begin
                    if (datapath_abort_ack) begin
                        descriptor_write_pointer <= '0;
                        descriptor_read_pointer <= '0;
                        descriptor_count <= '0;
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (TAG_WIDTH < 1 || RESULT_DEPTH < 2 || RESULT_DEPTH > (1 << PTR_WIDTH))
            $error("phase_shared_compute_scheduler parameter relation is invalid");
    end

`ifndef SYNTHESIS
    logic stalled_issue;
    logic [3+2+6+4+16+16+8+32+8+3+TAG_WIDTH+16+16+32+3+10+3+10-1:0]
        stalled_issue_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_issue <= 1'b0;
            stalled_issue_payload <= '0;
        end else begin
            assert (descriptor_count <= (PTR_WIDTH+1)'(RESULT_DEPTH))
                else $error("phase_shared_compute_scheduler descriptor FIFO overflow");
            if (state == BUILD_GROUPS)
                assert ($stable(start_batch_group))
                    else $error("phase_shared_compute_scheduler batch configuration changed while building groups");
            if (tile_result_valid && descriptor_count != 0)
                assert (tile_result_tag == descriptor_tag)
                    else $error("phase_shared_compute_scheduler received an out-of-order result tag");
            if (stalled_issue && !abort_request)
                assert (issue_valid && {issue_batch_index, issue_mode, issue_physical_row_base,
                    issue_active_rows,
                    issue_input_feature_base, issue_output_feature_base,
                    issue_row_mask, issue_k_mask, issue_column_mask,
                    issue_first_k_step, issue_last_k_step, issue_last_for_stripe,
                    issue_tag,
                    issue_row_batch_id, issue_stripe_id, issue_activation_byte_address,
                    issue_panel_start_bank, issue_panel_word_row,
                    issue_panel_bank_count, issue_panel_plane_rows} == stalled_issue_payload)
                    else $error("phase_shared_compute_scheduler changed a stalled issue");
            stalled_issue <= issue_valid && !issue_ready && !abort_request;
            stalled_issue_payload <= {issue_batch_index, issue_mode, issue_physical_row_base,
                issue_active_rows,
                issue_input_feature_base, issue_output_feature_base,
                issue_row_mask, issue_k_mask, issue_column_mask,
                issue_first_k_step, issue_last_k_step, issue_last_for_stripe, issue_tag,
                issue_row_batch_id, issue_stripe_id, issue_activation_byte_address,
                issue_panel_start_bank, issue_panel_word_row,
                issue_panel_bank_count, issue_panel_plane_rows};
        end
    end
`endif
endmodule

`default_nettype wire
