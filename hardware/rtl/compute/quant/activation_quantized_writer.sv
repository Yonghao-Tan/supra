`default_nettype none

// Converts one 8-row x 8-element quantized tile into the fixed compact issue
// layout consumed by the PE. Mixed A4/A8 rows retain one tile while rows that
// target the same SRAM macro are emitted in successive registered bundles.
module activation_quantized_writer #(
    parameter integer ADDR_WIDTH = 32,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    input  logic                         cfg_valid,
    output logic                         cfg_ready,
    input  logic [1:0]                   cfg_mode,
    input  logic [5:0]                   cfg_physical_row_base,
    input  logic [5:0]                   cfg_row_count,
    input  logic [15:0]                  cfg_elements_per_row,
    input  logic [ADDR_WIDTH-1:0]        cfg_activation_base_byte_offset,
    input  logic [ADDR_WIDTH-1:0]        cfg_activation_limit_byte_offset,
    input  logic                         cfg_mixed_rows,
    input  logic                         cfg_row_partial,
    input  logic [1:0]                   cfg_segment_count,
    input  logic [5:0]                   cfg_segment_mode,
    input  logic [17:0]                  cfg_segment_row_base,
    input  logic [17:0]                  cfg_segment_row_count,
    input  logic                         cfg_mixed_group_enable,
    input  logic [5:0]                   cfg_compute_group_count,
    input  logic [47:0]                  cfg_row_precision_a8,
    input  logic [287:0]                 cfg_row_compute_group,
    input  logic [143:0]                 cfg_row_pe_slot,
    input  logic [95:0]                  cfg_row_phase_mask,

    input  logic                         scale_valid,
    output logic                         scale_ready,
    input  logic [5:0]                   scale_row_base,
    input  logic [7:0]                   scale_row_mask,
    input  logic [8*16-1:0]              scale_values_bf16,
    output logic                         scale_write_valid,
    input  logic                         scale_write_ready,
    output logic [5:0]                   scale_write_row_base,
    output logic [7:0]                   scale_write_row_mask,
    output logic [8*16-1:0]              scale_write_values_bf16,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [5:0]                   in_physical_row_base,
    input  logic [15:0]                  in_element_base,
    input  logic [64*8-1:0]              in_values,
    input  logic [63:0]                  in_lane_mask,
    input  logic [TAG_WIDTH-1:0]         in_tag,

    output logic                         write_valid,
    input  logic                         write_ready,
    output logic [7:0]                   write_slot_valid,
    output logic [8*ADDR_WIDTH-1:0]      write_byte_address,
    output logic [8*128-1:0]             write_data,
    output logic [8*16-1:0]              write_byte_enable,
    output logic [TAG_WIDTH-1:0]         write_tag,
    output logic                         done_pulse,
    output logic                         error
);
    localparam logic [1:0] MODE_W4A4 = 2'd0;
    localparam logic [1:0] MODE_W4A8 = 2'd1;
    localparam logic [1:0] MODE_W8A8 = 2'd2;

    logic active;
    logic final_tile_pending;
    logic [1:0] saved_mode;
    logic [5:0] saved_row_base;
    logic [5:0] saved_row_count;
    logic [15:0] saved_elements;
    logic [ADDR_WIDTH-1:0] saved_base;
    logic [ADDR_WIDTH-1:0] saved_limit;
    logic saved_mixed_rows;
    logic saved_row_partial;
    logic [1:0] saved_segment_count;
    logic [5:0] saved_segment_mode;
    logic [17:0] saved_segment_row_base;
    logic [17:0] saved_segment_row_count;
    logic saved_mixed_group_enable;
    logic [5:0] saved_compute_group_count;
    logic [47:0] saved_row_precision_a8;
    logic [287:0] saved_row_compute_group;
    logic [143:0] saved_row_pe_slot;
    logic [95:0] saved_row_phase_mask;
    logic group_layout_ready;
    logic [5:0] group_build_row;
    logic [5:0] group_build_last;
    logic [5:0] group_unit_base [0:47];
    logic [3:0] group_phase0_count [0:47];
    logic [5:0] expected_row_base;
    logic [15:0] expected_element_base;
    logic scale_received;

    logic [5:0] batch_row_count;
    logic [15:0] row_bytes;
    logic [15:0] issue_bytes;
    logic [32:0] batch_offset;
    logic [32:0] issue_offset;
    logic [32:0] partial_block_offset;
    logic [3:0] partial_row_index;
    logic [32:0] tile_end;
    logic input_metadata_ok;
    logic input_mask_ok;
    logic input_is_last_element;
    logic input_is_last_batch;
    logic input_completes_storage_group;
    logic write_fire;
    logic scale_write_fire;
    logic [7:0] expected_scale_mask;
    logic [7:0] next_write_slot_valid;
    logic [8*ADDR_WIDTH-1:0] next_write_byte_address;
    logic [8*128-1:0] next_write_data;
    logic [8*16-1:0] next_write_byte_enable;
    logic partial_input_ok;
    logic [7:0] partial_next_write_slot_valid;
    logic [8*ADDR_WIDTH-1:0] partial_next_write_byte_address;
    logic [8*128-1:0] partial_next_write_data;
    logic [8*16-1:0] partial_next_write_byte_enable;
    logic [32:0] partial_block_bytes;
    logic [ADDR_WIDTH-1:0] issue_byte_address;
    logic [ADDR_WIDTH-1:0] issue_aligned_address;
    logic mixed_tile_valid;
    logic [5:0] mixed_tile_row_base;
    logic [15:0] mixed_tile_element_base;
    logic [511:0] mixed_tile_values;
    logic [15:0] mixed_tile_row_mode;
    logic [8*ADDR_WIDTH-1:0] mixed_tile_byte_address;
    logic [TAG_WIDTH-1:0] mixed_tile_tag;
    logic [7:0] mixed_pending_rows;
    logic [7:0] mixed_selected_rows;
    logic [7:0] mixed_next_write_slot_valid;
    logic [8*ADDR_WIDTH-1:0] mixed_next_write_byte_address;
    logic [8*128-1:0] mixed_next_write_data;
    logic [8*16-1:0] mixed_next_write_byte_enable;
    logic mixed_bundle_launch;
    logic mixed_write_finishes_tile;
    logic mixed_input_mask_ok;
    logic mixed_input_rows_found;
    logic mixed_input_address_ok;
    logic [15:0] mixed_input_row_mode;
    logic [8*ADDR_WIDTH-1:0] mixed_input_byte_address;

    always_comb begin
        integer lane;
        integer row;
        logic [3:0] column;

        batch_row_count = saved_row_count - (expected_row_base - saved_row_base);
        if (batch_row_count > 8) batch_row_count = 8;
        row_bytes = saved_mode == MODE_W4A4 ? saved_elements >> 1 : saved_elements;
        issue_bytes = saved_mode == MODE_W8A8 ?
            {7'd0, batch_row_count, 3'b000} :
            {6'd0, batch_row_count, 4'b0000};
        tile_end = {1'b0, saved_base} + issue_offset + 33'(issue_bytes);
        input_mask_ok = 1'b1;
        for (lane = 0; lane < 64; lane = lane + 1) begin
            row = lane >> 3;
            column = 4'(lane & 7);
            if (row < batch_row_count &&
                expected_element_base + 16'(column) < saved_elements) begin
                if (!in_lane_mask[lane]) input_mask_ok = 1'b0;
                if (saved_mode == MODE_W4A4 &&
                    ($signed(in_values[lane*8 +: 8]) < -8'sd7 ||
                     $signed(in_values[lane*8 +: 8]) > 8'sd7))
                    input_mask_ok = 1'b0;
                if (saved_mode != MODE_W4A4 &&
                    $signed(in_values[lane*8 +: 8]) == -8'sd128)
                    input_mask_ok = 1'b0;
            end else if (in_lane_mask[lane]) begin
                input_mask_ok = 1'b0;
            end
        end
        input_metadata_ok = saved_row_partial ? partial_input_ok :
            active && scale_received &&
                in_physical_row_base == expected_row_base &&
                in_element_base == expected_element_base &&
                (saved_mixed_rows ?
                    (mixed_input_mask_ok && mixed_input_rows_found &&
                     mixed_input_address_ok) : input_mask_ok) &&
                expected_element_base < saved_elements &&
                (saved_mixed_rows || tile_end <= {1'b0, saved_limit});
        input_is_last_element = expected_element_base + 16'd8 >= saved_elements;
        input_is_last_batch = expected_row_base + batch_row_count >=
            saved_row_base + saved_row_count;
        input_completes_storage_group = saved_mode == MODE_W8A8 ||
            (saved_mode == MODE_W4A8 && expected_element_base[3]) ||
            (saved_mode == MODE_W4A4 && expected_element_base[4:3] == 2'b11);
    end

    assign issue_byte_address = saved_base + ADDR_WIDTH'(issue_offset);
    assign issue_aligned_address = issue_byte_address & ~ADDR_WIDTH'(15);

    always_comb begin : partial_row_mapping
        integer column;
        integer segment;
        logic [1:0] row_mode;
        logic row_segment_found;
        logic mask_ok;
        logic values_ok;
        logic [15:0] partial_issue_bytes;
        logic [32:0] group_byte_offset;
        logic [32:0] byte_address_value;
        logic [ADDR_WIDTH-1:0] row_byte_address;
        logic [63:0] row_a8_data;
        logic [31:0] row_a4_data;

        partial_next_write_slot_valid = '0;
        partial_next_write_byte_address = '0;
        partial_next_write_data = '0;
        partial_next_write_byte_enable = '0;
        partial_block_bytes = '0;
        row_mode = saved_mode;
        row_segment_found = !saved_mixed_rows;
        for (segment = 0; segment < 3; segment = segment + 1) begin
            if (saved_mixed_rows && segment < saved_segment_count &&
                in_physical_row_base >= saved_segment_row_base[segment*6 +: 6] &&
                in_physical_row_base < saved_segment_row_base[segment*6 +: 6] +
                    saved_segment_row_count[segment*6 +: 6]) begin
                row_mode = saved_segment_mode[segment*2 +: 2];
                row_segment_found = 1'b1;
            end
        end

        mask_ok = 1'b1;
        values_ok = 1'b1;
        for (column = 0; column < 8; column = column + 1) begin
            if (in_element_base + 16'(column) < saved_elements) begin
                if (!in_lane_mask[column]) mask_ok = 1'b0;
                if (row_mode == MODE_W4A4 &&
                    ($signed(in_values[column*8 +: 8]) < -8'sd7 ||
                     $signed(in_values[column*8 +: 8]) > 8'sd7))
                    values_ok = 1'b0;
                if (row_mode != MODE_W4A4 &&
                    $signed(in_values[column*8 +: 8]) == -8'sd128)
                    values_ok = 1'b0;
            end else if (in_lane_mask[column]) begin
                mask_ok = 1'b0;
            end
        end
        if (|in_lane_mask[63:8]) mask_ok = 1'b0;

        partial_issue_bytes = row_mode == MODE_W8A8 ?
            {7'd0, batch_row_count, 3'b000} :
            {6'd0, batch_row_count, 4'b0000};
        group_byte_offset = partial_block_offset;
        case (row_mode)
            MODE_W4A4: if (in_element_base[5])
                group_byte_offset = partial_block_offset +
                    33'(partial_issue_bytes);
            MODE_W4A8: case (in_element_base[5:4])
                2'd0: group_byte_offset = partial_block_offset;
                2'd1: group_byte_offset = partial_block_offset +
                    33'(partial_issue_bytes);
                2'd2: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 1);
                default: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 1) + 33'(partial_issue_bytes);
            endcase
            MODE_W8A8: case (in_element_base[5:3])
                3'd0: group_byte_offset = partial_block_offset;
                3'd1: group_byte_offset = partial_block_offset +
                    33'(partial_issue_bytes);
                3'd2: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 1);
                3'd3: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 1) + 33'(partial_issue_bytes);
                3'd4: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 2);
                3'd5: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 2) + 33'(partial_issue_bytes);
                3'd6: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 2) +
                    (33'(partial_issue_bytes) << 1);
                default: group_byte_offset = partial_block_offset +
                    (33'(partial_issue_bytes) << 2) +
                    (33'(partial_issue_bytes) << 1) + 33'(partial_issue_bytes);
            endcase
            default: group_byte_offset = partial_block_offset;
        endcase
        if (saved_mixed_rows) begin
            group_byte_offset = row_mode == MODE_W4A4 ?
                (33'(in_element_base) >> 5) << 7 :
                (33'(in_element_base) >> 4) << 7;
        end

        byte_address_value = {1'b0, saved_base} + batch_offset +
            group_byte_offset + (row_mode == MODE_W8A8 ?
                (33'(partial_row_index) << 3) :
                (33'(partial_row_index) << 4));
        row_byte_address = row_mode == MODE_W8A8 ?
            byte_address_value[ADDR_WIDTH-1:0] & ~ADDR_WIDTH'(15) :
            byte_address_value[ADDR_WIDTH-1:0];
        row_a8_data = in_values[63:0];
        row_a4_data = {
            in_values[7*8 +: 4], in_values[6*8 +: 4],
            in_values[5*8 +: 4], in_values[4*8 +: 4],
            in_values[3*8 +: 4], in_values[2*8 +: 4],
            in_values[1*8 +: 4], in_values[0*8 +: 4]};
        partial_next_write_slot_valid[0] = 1'b1;
        partial_next_write_byte_address[0 +: ADDR_WIDTH] = row_byte_address;
        case (row_mode)
            MODE_W4A4: case (in_element_base[4:3])
                2'd0: begin
                    partial_next_write_data[0 +: 32] = row_a4_data;
                    partial_next_write_byte_enable[0 +: 16] = 16'h000f;
                end
                2'd1: begin
                    partial_next_write_data[32 +: 32] = row_a4_data;
                    partial_next_write_byte_enable[0 +: 16] = 16'h00f0;
                end
                2'd2: begin
                    partial_next_write_data[64 +: 32] = row_a4_data;
                    partial_next_write_byte_enable[0 +: 16] = 16'h0f00;
                end
                default: begin
                    partial_next_write_data[96 +: 32] = row_a4_data;
                    partial_next_write_byte_enable[0 +: 16] = 16'hf000;
                end
            endcase
            MODE_W4A8: if (in_element_base[3]) begin
                partial_next_write_data[64 +: 64] = row_a8_data;
                partial_next_write_byte_enable[0 +: 16] = 16'hff00;
            end else begin
                partial_next_write_data[0 +: 64] = row_a8_data;
                partial_next_write_byte_enable[0 +: 16] = 16'h00ff;
            end
            MODE_W8A8: if (byte_address_value[3]) begin
                partial_next_write_data[64 +: 64] = row_a8_data;
                partial_next_write_byte_enable[0 +: 16] = 16'hff00;
            end else begin
                partial_next_write_data[0 +: 64] = row_a8_data;
                partial_next_write_byte_enable[0 +: 16] = 16'h00ff;
            end
            default: begin
                partial_next_write_slot_valid = '0;
                values_ok = 1'b0;
            end
        endcase

        case (row_mode)
            MODE_W4A4: partial_block_bytes = 33'(partial_issue_bytes) << 1;
            MODE_W4A8: partial_block_bytes = 33'(partial_issue_bytes) << 2;
            MODE_W8A8: partial_block_bytes = 33'(partial_issue_bytes) << 3;
            default: partial_block_bytes = '0;
        endcase
        partial_input_ok = active && scale_received && saved_row_partial &&
            row_segment_found && row_mode <= MODE_W8A8 && mask_ok && values_ok &&
            in_physical_row_base == expected_row_base + 6'(partial_row_index) &&
            in_element_base == expected_element_base &&
            expected_element_base < saved_elements &&
            !byte_address_value[32] &&
            byte_address_value >= {1'b0, saved_base} &&
            byte_address_value + 33'd16 <= {1'b0, saved_limit};
    end

    always_comb begin : mixed_row_mapping
        integer row;
        integer column;
        integer segment;
        integer physical_row;
        integer physical_batch;
        integer row_in_batch;
        integer storage_group;
        integer compute_group;
        integer pe_slot;
        logic [2:0] macro_index;
        logic [1:0] row_mode;
        logic row_segment_found;
        logic [7:0] used_macros;
        logic [32:0] batch_byte_offset;
        logic [32:0] group_byte_offset;
        logic [32:0] byte_address_value;
        logic [24:0] compact_group_base;
        logic [14:0] compact_word_index;
        logic compact_phase;
        logic [ADDR_WIDTH-1:0] row_byte_address;
        logic [63:0] row_a8_data;
        logic [31:0] row_a4_data;

        mixed_next_write_slot_valid = '0;
        mixed_next_write_byte_address = '0;
        mixed_next_write_data = '0;
        mixed_next_write_byte_enable = '0;
        mixed_selected_rows = '0;
        mixed_input_mask_ok = 1'b1;
        mixed_input_rows_found = 1'b1;
        mixed_input_address_ok = 1'b1;
        mixed_input_row_mode = '0;
        mixed_input_byte_address = '0;
        used_macros = '0;

        for (row = 0; row < 8; row = row + 1) begin
            physical_row = integer'(expected_row_base) + row;
            row_mode = MODE_W4A4;
            row_segment_found = 1'b0;
            compute_group = 0;
            pe_slot = 0;
            compact_phase = 1'b0;
            if (saved_mixed_group_enable &&
                physical_row >= integer'(saved_row_base) &&
                physical_row < integer'(saved_row_base) +
                    integer'(saved_row_count)) begin
                row_mode = saved_row_precision_a8[physical_row] ?
                    MODE_W4A8 : MODE_W4A4;
                compute_group = integer'(saved_row_compute_group[
                    physical_row*6 +: 6]);
                pe_slot = integer'(saved_row_pe_slot[
                    physical_row*3 +: 3]);
                compact_phase = saved_row_precision_a8[physical_row] ?
                    in_element_base[4] :
                    saved_row_phase_mask[physical_row*2+1];
                row_segment_found = compute_group <
                    integer'(saved_compute_group_count);
            end else begin
                for (segment = 0; segment < 3; segment = segment + 1) begin
                    if (segment < saved_segment_count &&
                        physical_row >= integer'(saved_segment_row_base[
                            segment*6 +: 6]) &&
                        physical_row < integer'(saved_segment_row_base[
                            segment*6 +: 6]) + integer'(saved_segment_row_count[
                            segment*6 +: 6])) begin
                        row_mode = saved_segment_mode[segment*2 +: 2];
                        row_segment_found = 1'b1;
                    end
                end
            end
            mixed_input_row_mode[row*2 +: 2] = row_mode;
            if (row < batch_row_count) begin
                if (!row_segment_found || row_mode > MODE_W4A8)
                    mixed_input_rows_found = 1'b0;
                for (column = 0; column < 8; column = column + 1) begin
                    if (expected_element_base + 16'(column) < saved_elements) begin
                        if (!in_lane_mask[row*8 + column])
                            mixed_input_mask_ok = 1'b0;
                        if (row_mode == MODE_W4A4 &&
                            ($signed(in_values[(row*8+column)*8 +: 8]) < -8'sd7 ||
                             $signed(in_values[(row*8+column)*8 +: 8]) > 8'sd7))
                            mixed_input_mask_ok = 1'b0;
                        if (row_mode == MODE_W4A8 &&
                            $signed(in_values[(row*8+column)*8 +: 8]) == -8'sd128)
                            mixed_input_mask_ok = 1'b0;
                    end else if (in_lane_mask[row*8 + column]) begin
                        mixed_input_mask_ok = 1'b0;
                    end
                end

                if (row_segment_found && row_mode <= MODE_W4A8) begin
                    if (saved_mixed_group_enable) begin
                        compact_group_base = saved_elements == 16'd4096 ?
                            {8'd0, group_unit_base[compute_group], 11'd0} :
                            ({6'd0, group_unit_base[compute_group], 12'd0} +
                             {8'd0, group_unit_base[compute_group], 11'd0});
                        compact_word_index = 15'(compact_group_base >> 4) +
                            (15'(in_element_base[13:5]) << 4) +
                            (compact_phase ?
                                {11'd0, group_phase0_count[compute_group]} :
                                15'd0) + 15'(pe_slot);
                        byte_address_value = {1'b0, saved_base} +
                            ({18'd0, compact_word_index} << 4);
                    end else begin
                        physical_batch = physical_row >> 3;
                        row_in_batch = physical_row & 7;
                        case (physical_batch)
                            0: batch_byte_offset = '0;
                            1: batch_byte_offset = 33'(saved_elements) << 3;
                            2: batch_byte_offset = 33'(saved_elements) << 4;
                            3: batch_byte_offset = (33'(saved_elements) << 4) +
                                (33'(saved_elements) << 3);
                            4: batch_byte_offset = 33'(saved_elements) << 5;
                            5: batch_byte_offset = (33'(saved_elements) << 5) +
                                (33'(saved_elements) << 3);
                            default: batch_byte_offset = '0;
                        endcase
                        storage_group = row_mode == MODE_W4A4 ?
                            integer'(in_element_base) >> 5 :
                            integer'(in_element_base) >> 4;
                        group_byte_offset = 33'(storage_group) << 7;
                        byte_address_value = {1'b0, saved_base} +
                            batch_byte_offset + group_byte_offset +
                            (33'(row_in_batch) << 4);
                    end
                    mixed_input_byte_address[
                        row*ADDR_WIDTH +: ADDR_WIDTH] =
                        byte_address_value[ADDR_WIDTH-1:0];
                    if (byte_address_value[32] ||
                        byte_address_value < {1'b0, saved_base} ||
                        byte_address_value + 33'd16 > {1'b0, saved_limit})
                        mixed_input_address_ok = 1'b0;
                end else begin
                    mixed_input_address_ok = 1'b0;
                end
            end else if (|in_lane_mask[row*8 +: 8]) begin
                mixed_input_mask_ok = 1'b0;
            end
        end

        for (row = 0; row < 8; row = row + 1) begin
            row_mode = mixed_tile_row_mode[row*2 +: 2];
            row_byte_address = mixed_tile_byte_address[
                row*ADDR_WIDTH +: ADDR_WIDTH];
            macro_index = row_byte_address[6:4];

            row_a8_data = mixed_tile_values[row*64 +: 64];
            row_a4_data = {
                mixed_tile_values[(row*8+7)*8 +: 4],
                mixed_tile_values[(row*8+6)*8 +: 4],
                mixed_tile_values[(row*8+5)*8 +: 4],
                mixed_tile_values[(row*8+4)*8 +: 4],
                mixed_tile_values[(row*8+3)*8 +: 4],
                mixed_tile_values[(row*8+2)*8 +: 4],
                mixed_tile_values[(row*8+1)*8 +: 4],
                mixed_tile_values[(row*8+0)*8 +: 4]
            };
            if (mixed_pending_rows[row] && !used_macros[macro_index]) begin
                used_macros[macro_index] = 1'b1;
                mixed_selected_rows[row] = 1'b1;
                mixed_next_write_slot_valid[row] = 1'b1;
                mixed_next_write_byte_address[row*ADDR_WIDTH +: ADDR_WIDTH] =
                    row_byte_address;
                if (row_mode == MODE_W4A4) begin
                    case (mixed_tile_element_base[4:3])
                        2'd0: begin
                            mixed_next_write_data[row*128 +: 32] = row_a4_data;
                            mixed_next_write_byte_enable[row*16 +: 16] = 16'h000f;
                        end
                        2'd1: begin
                            mixed_next_write_data[row*128+32 +: 32] = row_a4_data;
                            mixed_next_write_byte_enable[row*16 +: 16] = 16'h00f0;
                        end
                        2'd2: begin
                            mixed_next_write_data[row*128+64 +: 32] = row_a4_data;
                            mixed_next_write_byte_enable[row*16 +: 16] = 16'h0f00;
                        end
                        default: begin
                            mixed_next_write_data[row*128+96 +: 32] = row_a4_data;
                            mixed_next_write_byte_enable[row*16 +: 16] = 16'hf000;
                        end
                    endcase
                end else if (mixed_tile_element_base[3]) begin
                    mixed_next_write_data[row*128+64 +: 64] = row_a8_data;
                    mixed_next_write_byte_enable[row*16 +: 16] = 16'hff00;
                end else begin
                    mixed_next_write_data[row*128 +: 64] = row_a8_data;
                    mixed_next_write_byte_enable[row*16 +: 16] = 16'h00ff;
                end
            end
        end
    end

    generate
        for (genvar write_slot = 0; write_slot < 8; write_slot = write_slot + 1) begin : fixed_write_slots
            logic slot_valid;
            logic [ADDR_WIDTH-1:0] slot_byte_address;
            logic [127:0] slot_data;
            logic [15:0] slot_byte_enable;
            logic [63:0] row_a8_data;
            logic [31:0] row_w4_data;
            logic p0_low_valid, p0_high_valid;
            logic p8_low_valid, p8_high_valid;
            logic [63:0] p0_low_data, p0_high_data;
            logic [63:0] p8_low_data, p8_high_data;

            assign row_a8_data = in_values[write_slot*64 +: 64];
            assign row_w4_data = {
                in_values[(write_slot*8+7)*8 +: 4],
                in_values[(write_slot*8+6)*8 +: 4],
                in_values[(write_slot*8+5)*8 +: 4],
                in_values[(write_slot*8+4)*8 +: 4],
                in_values[(write_slot*8+3)*8 +: 4],
                in_values[(write_slot*8+2)*8 +: 4],
                in_values[(write_slot*8+1)*8 +: 4],
                in_values[(write_slot*8+0)*8 +: 4]
            };

            if (write_slot < 4) begin : phase0_rows
                assign p0_low_valid = batch_row_count > 2*write_slot;
                assign p0_high_valid = batch_row_count > 2*write_slot+1;
                assign p0_low_data = in_values[(2*write_slot)*64 +: 64];
                assign p0_high_data = in_values[(2*write_slot+1)*64 +: 64];
            end else begin : no_phase0_rows
                assign p0_low_valid = 1'b0;
                assign p0_high_valid = 1'b0;
                assign p0_low_data = 64'd0;
                assign p0_high_data = 64'd0;
            end

            if (write_slot == 0) begin : phase8_first_row
                assign p8_low_valid = 1'b0;
                assign p8_high_valid = batch_row_count > 0;
                assign p8_low_data = 64'd0;
                assign p8_high_data = in_values[0 +: 64];
            end else if (write_slot < 4) begin : phase8_middle_rows
                assign p8_low_valid = batch_row_count > 2*write_slot-1;
                assign p8_high_valid = batch_row_count > 2*write_slot;
                assign p8_low_data = in_values[(2*write_slot-1)*64 +: 64];
                assign p8_high_data = in_values[(2*write_slot)*64 +: 64];
            end else if (write_slot == 4) begin : phase8_last_row
                assign p8_low_valid = batch_row_count > 7;
                assign p8_high_valid = 1'b0;
                assign p8_low_data = in_values[7*64 +: 64];
                assign p8_high_data = 64'd0;
            end else begin : no_phase8_rows
                assign p8_low_valid = 1'b0;
                assign p8_high_valid = 1'b0;
                assign p8_low_data = 64'd0;
                assign p8_high_data = 64'd0;
            end

            always_comb begin
                slot_valid = 1'b0;
                slot_byte_address = '0;
                slot_data = '0;
                slot_byte_enable = '0;
                case (saved_mode)
                    MODE_W8A8: begin
                        slot_byte_address = issue_aligned_address +
                            ADDR_WIDTH'(write_slot*16);
                        if (!issue_byte_address[3]) begin
                            if (p0_low_valid) begin
                                slot_valid = 1'b1;
                                slot_data[63:0] = p0_low_data;
                                slot_byte_enable[7:0] = 8'hff;
                            end
                            if (p0_high_valid) begin
                                slot_valid = 1'b1;
                                slot_data[127:64] = p0_high_data;
                                slot_byte_enable[15:8] = 8'hff;
                            end
                        end else begin
                            if (p8_low_valid) begin
                                slot_valid = 1'b1;
                                slot_data[63:0] = p8_low_data;
                                slot_byte_enable[7:0] = 8'hff;
                            end
                            if (p8_high_valid) begin
                                slot_valid = 1'b1;
                                slot_data[127:64] = p8_high_data;
                                slot_byte_enable[15:8] = 8'hff;
                            end
                        end
                    end
                    MODE_W4A8: begin
                        if (batch_row_count > write_slot) begin
                            slot_valid = 1'b1;
                            slot_byte_address = issue_byte_address +
                                ADDR_WIDTH'(write_slot*16);
                            if (expected_element_base[3]) begin
                                slot_data[127:64] = row_a8_data;
                                slot_byte_enable = 16'hff00;
                            end else begin
                                slot_data[63:0] = row_a8_data;
                                slot_byte_enable = 16'h00ff;
                            end
                        end
                    end
                    MODE_W4A4: begin
                        if (batch_row_count > write_slot) begin
                            slot_valid = 1'b1;
                            slot_byte_address = issue_byte_address +
                                ADDR_WIDTH'(write_slot*16);
                            case (expected_element_base[4:3])
                                2'd0: begin
                                    slot_data[31:0] = row_w4_data;
                                    slot_byte_enable = 16'h000f;
                                end
                                2'd1: begin
                                    slot_data[63:32] = row_w4_data;
                                    slot_byte_enable = 16'h00f0;
                                end
                                2'd2: begin
                                    slot_data[95:64] = row_w4_data;
                                    slot_byte_enable = 16'h0f00;
                                end
                                default: begin
                                    slot_data[127:96] = row_w4_data;
                                    slot_byte_enable = 16'hf000;
                                end
                            endcase
                        end
                    end
                    default: begin
                        slot_valid = 1'b0;
                    end
                endcase
            end

            assign next_write_slot_valid[write_slot] = slot_valid;
            assign next_write_byte_address[write_slot*ADDR_WIDTH +: ADDR_WIDTH] =
                slot_byte_address;
            assign next_write_data[write_slot*128 +: 128] = slot_data;
            assign next_write_byte_enable[write_slot*16 +: 16] = slot_byte_enable;
        end
    endgenerate

    assign cfg_ready = !abort_request && !active && !write_valid &&
        !scale_write_valid;
    assign scale_ready = !abort_request && active && !scale_received &&
        !scale_write_valid && scale_row_base == expected_row_base;
    assign in_ready = !abort_request && active && scale_received &&
        (!saved_mixed_group_enable || group_layout_ready) &&
        (saved_mixed_rows && !saved_row_partial ?
                            (!mixed_tile_valid && !write_valid) :
                            (!write_valid || write_ready));
    assign write_fire = write_valid && write_ready;
    assign scale_write_fire = scale_write_valid && scale_write_ready;
    assign mixed_bundle_launch = saved_mixed_rows && !saved_row_partial &&
        mixed_tile_valid &&
        |mixed_pending_rows && (!write_valid || write_ready);
    assign expected_scale_mask = batch_row_count == 8 ? 8'hff :
        8'((9'd1 << batch_row_count) - 1'b1);

    always_ff @(posedge clk) begin
        if (rst || abort_request) begin
            active <= 1'b0;
            final_tile_pending <= 1'b0;
            group_layout_ready <= 1'b0;
            group_build_row <= '0;
            group_build_last <= 6'h3f;
            expected_row_base <= '0;
            expected_element_base <= '0;
            batch_offset <= '0;
            issue_offset <= '0;
            partial_block_offset <= '0;
            partial_row_index <= '0;
            scale_received <= 1'b0;
            mixed_tile_valid <= 1'b0;
            mixed_pending_rows <= '0;
            mixed_write_finishes_tile <= 1'b0;
            scale_write_valid <= 1'b0;
            write_valid <= 1'b0;
            done_pulse <= 1'b0;
            error <= 1'b0;
        end else begin
            done_pulse <= 1'b0;
            if (scale_write_fire)
                scale_write_valid <= 1'b0;
            if (write_fire)
                write_valid <= 1'b0;
            if (final_tile_pending && !write_valid && !scale_write_valid) begin
                active <= 1'b0;
                final_tile_pending <= 1'b0;
                done_pulse <= 1'b1;
            end

            if (cfg_valid && cfg_ready) begin
                if (cfg_mode > MODE_W8A8 || cfg_row_count == 0 ||
                    {1'b0, cfg_physical_row_base} + {1'b0, cfg_row_count} >
                        7'd48 ||
                    cfg_elements_per_row == 0 || cfg_elements_per_row[2:0] != 0 ||
                    (cfg_row_partial && cfg_elements_per_row[5:0] != 0) ||
                    cfg_activation_base_byte_offset[3:0] != 0 || cfg_activation_limit_byte_offset <= cfg_activation_base_byte_offset ||
                    (cfg_mixed_rows && !cfg_mixed_group_enable &&
                     cfg_segment_count == 0) ||
                    (cfg_mixed_group_enable &&
                     (cfg_compute_group_count == 0 ||
                      cfg_compute_group_count > 6'd4 ||
                      (cfg_elements_per_row != 16'd4096 &&
                       cfg_elements_per_row != 16'd12288)))) begin
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    active <= 1'b1;
                    saved_mode <= cfg_mode;
                    saved_row_base <= cfg_physical_row_base;
                    saved_row_count <= cfg_row_count;
                    saved_elements <= cfg_elements_per_row;
                    saved_base <= cfg_activation_base_byte_offset;
                    saved_limit <= cfg_activation_limit_byte_offset;
                    saved_mixed_rows <= cfg_mixed_rows;
                    saved_row_partial <= cfg_row_partial;
                    saved_segment_count <= cfg_segment_count;
                    saved_segment_mode <= cfg_segment_mode;
                    saved_segment_row_base <= cfg_segment_row_base;
                    saved_segment_row_count <= cfg_segment_row_count;
                    saved_mixed_group_enable <= cfg_mixed_group_enable;
                    saved_compute_group_count <= cfg_compute_group_count;
                    saved_row_precision_a8 <= cfg_row_precision_a8;
                    saved_row_compute_group <= cfg_row_compute_group;
                    saved_row_pe_slot <= cfg_row_pe_slot;
                    saved_row_phase_mask <= cfg_row_phase_mask;
                    group_layout_ready <= !cfg_mixed_group_enable;
                    group_build_row <= '0;
                    group_build_last <= 6'h3f;
                    expected_row_base <= cfg_physical_row_base;
                    expected_element_base <= 16'd0;
                    batch_offset <= 33'd0;
                    issue_offset <= 33'd0;
                    partial_block_offset <= 33'd0;
                    partial_row_index <= 4'd0;
                    scale_received <= 1'b0;
                    mixed_tile_valid <= 1'b0;
                    mixed_pending_rows <= '0;
                    mixed_write_finishes_tile <= 1'b0;
                    error <= 1'b0;
                end
            end

            if (active && saved_mixed_group_enable && !group_layout_ready) begin
                logic [5:0] build_group;
                logic [1:0] build_phase_mask;
                build_group = saved_row_compute_group[group_build_row*6 +: 6];
                build_phase_mask = saved_row_phase_mask[group_build_row*2 +: 2];
                if (build_phase_mask != 0 &&
                    build_group != group_build_last) begin
                    group_unit_base[build_group] <=
                        {build_group[1:0], 4'b0000};
                    group_phase0_count[build_group] <=
                        build_phase_mask[0] ? 4'd1 : 4'd0;
                    group_build_last <= build_group;
                end else if (build_phase_mask != 0) begin
                    if (build_phase_mask[0])
                        group_phase0_count[build_group] <=
                            group_phase0_count[build_group] + 4'd1;
                end
                if (group_build_row == 6'd47)
                    group_layout_ready <= 1'b1;
                else
                    group_build_row <= group_build_row + 6'd1;
            end

            if (scale_valid && scale_ready) begin
                if (scale_row_mask != expected_scale_mask) begin
                    active <= 1'b0;
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    scale_received <= 1'b1;
                    scale_write_valid <= 1'b1;
                    scale_write_row_base <= scale_row_base;
                    scale_write_row_mask <= scale_row_mask;
                    scale_write_values_bf16 <= scale_values_bf16;
                end
            end

            if (in_valid && in_ready && saved_row_partial) begin
                if (!input_metadata_ok) begin
                    active <= 1'b0;
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    write_valid <= 1'b1;
                    write_slot_valid <= partial_next_write_slot_valid;
                    write_byte_address <= partial_next_write_byte_address;
                    write_data <= partial_next_write_data;
                    write_byte_enable <= partial_next_write_byte_enable;
                    write_tag <= in_tag;
                    if (in_element_base[5:0] == 6'd56) begin
                        if (6'(partial_row_index) + 6'd1 < batch_row_count) begin
                            partial_row_index <= partial_row_index + 1'b1;
                            expected_element_base <=
                                {in_element_base[15:6], 6'b000000};
                        end else begin
                            partial_row_index <= '0;
                            if (in_element_base + 16'd8 >= saved_elements) begin
                                if (expected_row_base + batch_row_count >=
                                    saved_row_base + saved_row_count) begin
                                    final_tile_pending <= 1'b1;
                                end else begin
                                    expected_row_base <=
                                        expected_row_base + batch_row_count;
                                    expected_element_base <= 16'd0;
                                    batch_offset <= batch_offset +
                                        (saved_mixed_rows ?
                                            (33'(saved_elements) << 3) :
                                            ({17'd0, row_bytes} << 3));
                                    partial_block_offset <= '0;
                                    scale_received <= 1'b0;
                                end
                            end else begin
                                expected_element_base <=
                                    {in_element_base[15:6], 6'b000000} + 16'd64;
                                partial_block_offset <=
                                    partial_block_offset + partial_block_bytes;
                            end
                        end
                    end else begin
                        expected_element_base <= expected_element_base + 16'd8;
                    end
                end
            end else if (in_valid && in_ready && saved_mixed_rows) begin
                if (!input_metadata_ok) begin
                    active <= 1'b0;
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    mixed_tile_valid <= 1'b1;
                    mixed_tile_row_base <= in_physical_row_base;
                    mixed_tile_element_base <= in_element_base;
                    mixed_tile_values <= in_values;
                    mixed_tile_row_mode <= mixed_input_row_mode;
                    mixed_tile_byte_address <= mixed_input_byte_address;
                    mixed_tile_tag <= in_tag;
                    mixed_pending_rows <= expected_scale_mask;
                end
            end else if (in_valid && in_ready) begin
                if (!input_metadata_ok) begin
                    active <= 1'b0;
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    write_valid <= 1'b1;
                    write_slot_valid <= next_write_slot_valid;
                    write_byte_address <= next_write_byte_address;
                    write_data <= next_write_data;
                    write_byte_enable <= next_write_byte_enable;
                    write_tag <= in_tag;
                    if (input_is_last_element && input_is_last_batch) begin
                        final_tile_pending <= 1'b1;
                    end else if (input_is_last_element) begin
                        expected_row_base <= expected_row_base + batch_row_count;
                        expected_element_base <= 16'd0;
                        batch_offset <= batch_offset + {14'd0, row_bytes, 3'b000};
                        issue_offset <= batch_offset + {14'd0, row_bytes, 3'b000};
                        scale_received <= 1'b0;
                    end else begin
                        expected_element_base <= expected_element_base + 16'd8;
                        if (input_completes_storage_group)
                            issue_offset <= issue_offset + 33'(issue_bytes);
                    end
                end
            end

            if (mixed_bundle_launch) begin
                if (mixed_selected_rows == 0) begin
                    active <= 1'b0;
                    mixed_tile_valid <= 1'b0;
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                end else begin
                    write_valid <= 1'b1;
                    write_slot_valid <= mixed_next_write_slot_valid;
                    write_byte_address <= mixed_next_write_byte_address;
                    write_data <= mixed_next_write_data;
                    write_byte_enable <= mixed_next_write_byte_enable;
                    write_tag <= mixed_tile_tag;
                    mixed_pending_rows <=
                        mixed_pending_rows & ~mixed_selected_rows;
                    mixed_write_finishes_tile <=
                        (mixed_pending_rows & ~mixed_selected_rows) == 0;
                end
            end

            if (write_fire && saved_mixed_rows && mixed_write_finishes_tile) begin
                mixed_tile_valid <= 1'b0;
                mixed_write_finishes_tile <= 1'b0;
                if (mixed_tile_element_base + 16'd8 >= saved_elements &&
                    mixed_tile_row_base + batch_row_count >=
                        saved_row_base + saved_row_count) begin
                    final_tile_pending <= 1'b1;
                end else if (mixed_tile_element_base + 16'd8 >= saved_elements) begin
                    expected_row_base <= mixed_tile_row_base + batch_row_count;
                    expected_element_base <= 16'd0;
                    scale_received <= 1'b0;
                end else begin
                    expected_element_base <= mixed_tile_element_base + 16'd8;
                end
            end
        end
    end

    initial begin
        if (ADDR_WIDTH != 32 || TAG_WIDTH < 1)
            $error("activation_quantized_writer requires ADDR_WIDTH=32 and positive TAG_WIDTH");
    end

`ifndef SYNTHESIS
    logic stalled_write;
    logic [8+8*ADDR_WIDTH+8*128+8*16+TAG_WIDTH-1:0] held_write;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_write <= 1'b0;
            held_write <= '0;
        end else begin
            if (stalled_write && !abort_request)
                assert (write_valid &&
                    {write_slot_valid, write_byte_address, write_data,
                     write_byte_enable, write_tag} == held_write)
                    else $error("activation_quantized_writer changed a stalled bundle");
            if (!abort_request && write_valid && saved_mixed_rows)
                for (integer left = 0; left < 8; left = left + 1)
                    for (integer right = left + 1; right < 8;
                         right = right + 1)
                        if (write_slot_valid[left] && write_slot_valid[right])
                            assert (write_byte_address[
                                        left*ADDR_WIDTH + 4 +: 3] !=
                                    write_byte_address[
                                        right*ADDR_WIDTH + 4 +: 3])
                                else $error("activation_quantized_writer emitted two writes to one SRAM macro");
            if ($past(!rst && !abort_request && mixed_tile_valid) &&
                mixed_tile_valid &&
                !$past(in_valid && in_ready && saved_mixed_rows))
                assert ((mixed_pending_rows & ~$past(mixed_pending_rows)) == 0)
                    else $error("activation_quantized_writer restored a completed mixed row");
            stalled_write <= write_valid && !write_ready && !abort_request;
            if (write_valid && !write_ready)
                held_write <= {write_slot_valid, write_byte_address, write_data,
                               write_byte_enable, write_tag};
        end
    end
`endif
endmodule

`default_nettype wire
