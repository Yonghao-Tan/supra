`default_nettype none

// Loads one K token panel or one V channel stripe through the shared 256-bit
// DMA stream. Two 1 KiB group buffers overlap DDR fill with paired bank writes.
module attention_cache_loader #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     abort_request,
    output logic                     abort_ack,

    input  logic                     start_valid,
    output logic                     start_ready,
    input  logic                     start_is_v,
    input  logic [5:0]               start_head,
    input  logic [10:0]              start_token_base,
    input  logic [11:0]              start_token_count,
    input  logic [3:0]               start_channel_tile,
    input  logic                     start_target_panel,
    input  logic                     start_target_quarter,
    input  logic [TAG_WIDTH-1:0]     start_tag,
    output logic                     busy,
    output logic                     done_pulse,
    output logic                     error,
    output logic [3:0]               error_id,

    output logic                     cache_lookup_req_valid,
    input  logic                     cache_lookup_req_ready,
    output logic [5:0]               cache_lookup_req_head,
    output logic [10:0]              cache_lookup_req_key_group,
    output logic [4:0]               cache_lookup_req_chunk,
    output logic [TAG_WIDTH-1:0]     cache_lookup_req_tag,
    input  logic                     cache_lookup_rsp_valid,
    output logic                     cache_lookup_rsp_ready,
    input  logic [10:0]              cache_lookup_rsp_key_group,
    input  logic [7:0]               cache_lookup_rsp_current_mask,
    input  logic [7:0]               cache_lookup_rsp_retained_mask,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_k_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_v_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_current_k_scale_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_k_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_v_address,
    input  logic [ADDR_WIDTH-1:0]    cache_lookup_rsp_retained_k_scale_address,
    input  logic [TAG_WIDTH-1:0]     cache_lookup_rsp_tag,
    output logic                     cache_lookup_abort_request,
    input  logic                     cache_lookup_abort_ack,

    output logic                     dma_start_valid,
    input  logic                     dma_start_ready,
    output logic [ADDR_WIDTH-1:0]    dma_start_address,
    output logic [31:0]              dma_start_bytes,
    input  logic                     dma_data_valid,
    output logic                     dma_data_ready,
    input  logic [255:0]             dma_data,
    input  logic [31:0]              dma_data_byte_enable,
    input  logic [31:0]              dma_data_offset,
    input  logic                     dma_done,
    input  logic                     dma_error,
    output logic                     dma_abort_request,
    input  logic                     dma_abort_ack,

    output logic                     panel_write_valid,
    input  logic                     panel_write_ready,
    output logic                     panel_write_target_panel,
    output logic                     panel_write_target_quarter,
    output logic                     panel_write_bank,
    output logic [9:0]               panel_write_address,
    output logic [255:0]             panel_write_data,
    output logic [31:0]              panel_write_byte_enable,
    output logic [TAG_WIDTH-1:0]     panel_write_tag,

    output logic                     scale_write_valid,
    input  logic                     scale_write_ready,
    output logic                     scale_write_target_panel,
    output logic [10:0]              scale_write_token_base,
    output logic [127:0]             scale_write_data,
    output logic [15:0]              scale_write_byte_enable,
    output logic [TAG_WIDTH-1:0]     scale_write_tag,

    output logic [63:0]              accepted_cache_lookup_request_count,
    output logic [63:0]              accepted_dma_request_count,
    output logic [63:0]              accepted_dma_byte_count,
    output logic [63:0]              accepted_panel_write_count,
    output logic [63:0]              accepted_scale_write_count,
    output logic [63:0]              load_compute_overlap_cycle_count
);
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_CACHE_LOOKUP = 4'h2;
    localparam logic [3:0] ERROR_DMA = 4'h3;
    localparam logic [3:0] ERROR_DMA_PAYLOAD = 4'h4;

    typedef enum logic [3:0] {
        FILL_IDLE, FILL_WAIT_BUFFER, FILL_CACHE_LOOKUP_REQUEST,
        FILL_CACHE_LOOKUP_RESPONSE, FILL_FIND_RUN, FILL_EXTEND_RUN,
        FILL_DMA_REQUEST, FILL_DMA_STREAM, FILL_COMMIT_GROUP,
        FILL_WAIT_BASE_DRAIN, FILL_FINISHED, FILL_ERROR_DRAIN,
        FILL_ABORT_DRAIN
    } fill_state_t;

    // The same 1 KiB per buffer stores complete 64-bit DMA units. K uses the
    // first index for one eight-channel unit and the second for one token;
    // V uses the same bits as groups of eight consecutive tokens. This keeps
    // DMA writes full-width and performs the K transpose only during drain.
    // Four banks keep the same two 1 KiB buffers. For one group, units u and
    // u+4 share one 128-bit word in bank (group+u) mod 4. Four consecutive
    // DMA units therefore target four different banks, while drain reads one
    // word from every bank to recover all eight units.
    logic [127:0] group_bank_words [0:3][0:31];
    logic [3:0] group_bank_write_valid;
    logic [4:0] group_bank_write_index [0:3];
    logic group_bank_write_upper [0:3];
    logic [63:0] group_bank_write_data [0:3];
    logic [15:0] group_scale_words [0:1][0:7];
    logic buffer_valid [0:1];
    logic buffer_is_v [0:1];
    logic buffer_target_panel [0:1];
    logic buffer_target_quarter [0:1];
    logic [7:0] buffer_group_index [0:1];
    logic [7:0] buffer_group_tokens [0:1];
    logic [7:0] buffer_token_mask [0:1];
    logic buffer_sparse_overlay [0:1];
    logic [10:0] buffer_token_base [0:1];
    logic [TAG_WIDTH-1:0] buffer_tag [0:1];

    fill_state_t fill_state;
    logic is_v;
    logic [5:0] head;
    logic [10:0] token_base;
    logic [11:0] token_count;
    logic [3:0] channel_tile;
    logic target_panel;
    logic target_quarter;
    logic [TAG_WIDTH-1:0] operation_tag;
    logic [11:0] group_offset;
    logic [3:0] group_token_count;
    logic [7:0] group_current_mask;
    logic [ADDR_WIDTH-1:0] group_current_data_address;
    logic [ADDR_WIDTH-1:0] group_retained_data_address;
    logic [ADDR_WIDTH-1:0] group_current_scale_address;
    logic [ADDR_WIDTH-1:0] group_retained_scale_address;
    logic fill_buffer;
    logic loading_scale;
    logic [3:0] scan_token;
    logic run_current;
    logic [7:0] run_start_token;
    logic [7:0] run_token_count;
    logic coalesced_run_active;
    logic coalesced_run_dma;
    logic coalesced_run_current;
    logic [11:0] coalesced_run_start_offset;
    logic [11:0] coalesced_run_token_count;
    logic [ADDR_WIDTH-1:0] coalesced_run_base_address;
    logic sparse_overlay_pending;
    logic sparse_overlay_base_dma;
    logic sparse_overlay_current_pass;
    logic sparse_overlay_decided;
    logic [11:0] sparse_overlay_start_offset;
    logic [ADDR_WIDTH-1:0] sparse_overlay_retained_base_address;
    logic dma_owned;
    logic cache_lookup_outstanding;
    logic dma_done_seen;
    logic [31:0] dma_expected_bytes;
    logic [31:0] dma_received_bytes;
    logic terminal_error;
    logic [3:0] terminal_error_id;

    logic panel_scale_loaded;
    logic panel_scale_dma;
    logic scale_stream_valid;
    logic scale_stream_upper;
    logic [255:0] scale_stream_data;
    logic [31:0] scale_stream_byte_enable;
    logic [31:0] scale_stream_offset;

    logic drain_active;
    logic drain_buffer;
    logic [6:0] drain_write_index;
    logic drain_word_pending;
    logic [255:0] drain_word_data;
    logic [31:0] drain_word_byte_enable;
    logic drain_word_bank;
    logic [9:0] drain_word_address;
    logic drain_scale_pending;

    logic cache_lookup_req_fire, cache_lookup_rsp_fire;
    logic dma_start_fire, dma_data_fire, panel_write_fire, scale_write_fire;
    logic configuration_valid;
    logic buffer_available;
    logic selected_available_buffer;
    logic [31:0] current_run_bytes;
    logic [ADDR_WIDTH-1:0] selected_run_base;
    logic run_has_next_token;
    logic [7:0] run_next_token;
    logic run_next_same_region;
    logic [7:0] cache_lookup_valid_mask;
    logic [7:0] cache_lookup_current_valid_mask;
    logic cache_lookup_group_all_current;
    logic cache_lookup_group_all_retained;
    logic cache_lookup_group_uniform;
    logic cache_lookup_group_current;
    logic cache_lookup_group_joins_run;
    logic cache_lookup_group_sparse_current;
    logic sparse_overlay_can_replace_run;
    logic coalesced_run_accept_next_lookup;
    logic coalescing_allowed;
    logic [ADDR_WIDTH-1:0] cache_lookup_selected_data_address;
    logic [ADDR_WIDTH-1:0] coalesced_run_next_address;
    logic [11:0] cache_lookup_request_offset;
    logic [3:0] cache_lookup_request_tokens;
    logic dma_transfer_complete;
    logic dma_payload_error_this_cycle;
    logic dma_response_error_this_cycle;
    logic all_buffers_empty;
    logic drain_enabled;
    logic [6:0] drain_word_count;
    logic drain_last_word;
    logic [255:0] selected_drain_word_pair;
    logic [127:0] selected_group_bank_word [0:3];
    logic [511:0] selected_drain_group;
    logic [8:0] drain_v_token_offset;
    logic [8:0] drain_v_remaining_tokens;
    logic [8:0] v_storage_token [0:3];
    logic [4:0] k_storage_group [0:3];
    logic [2:0] k_storage_unit;
    logic [2:0] scale_storage_unit [0:7];
    logic [5:0] dma_fire_bytes;
    logic dma_data_offset_in_range;
    logic [31:0] dma_bytes_remaining_at_offset;
    logic [4:0] dma_stream_group;
    logic [11:0] dma_stream_token_offset;
    logic [11:0] dma_stream_tokens_remaining;
    logic [7:0] dma_stream_group_tokens;
    logic dma_stream_group_complete;

    function automatic [7:0] valid_token_mask(input [3:0] count);
        begin
            valid_token_mask = count >= 8 ? 8'hff :
                (8'h01 << count) - 8'h01;
        end
    endfunction

    function automatic [5:0] count_enabled_bytes(input [31:0] enables);
        begin
            count_enabled_bytes = 6'($countones(enables));
        end
    endfunction

    function automatic [15:0] two_byte_token_mask(input [7:0] token_mask);
        begin
            two_byte_token_mask = '0;
            for (integer token = 0; token < 8; token = token + 1)
                two_byte_token_mask[token*2 +: 2] = {2{token_mask[token]}};
        end
    endfunction

    function automatic [31:0] v_token_byte_mask(
        input [7:0] token_mask,
        input [2:0] token_offset
    );
        begin
            v_token_byte_mask = '0;
            for (integer token = 0; token < 4; token = token + 1)
                if (token_offset + token < 8)
                    v_token_byte_mask[token*8 +: 8] =
                        {8{token_mask[token_offset + token]}};
        end
    endfunction

    // Compute each beat lane's destination once, then decode fixed storage
    // entries. Repeating this arithmetic inside every entry creates thousands
    // of identical adders even though only four data units arrive per beat.
    always_comb begin : group_bank_write_decode
        logic [4:0] lane_group;
        logic [2:0] lane_unit;
        logic [1:0] lane_bank;

        lane_group = '0;
        lane_unit = '0;
        lane_bank = '0;
        group_bank_write_valid = '0;
        for (integer bank = 0; bank < 4; bank++) begin
            group_bank_write_index[bank] = '0;
            group_bank_write_upper[bank] = 1'b0;
            group_bank_write_data[bank] = '0;
        end
        if (dma_data_fire && !loading_scale) begin
            for (integer data_lane = 0; data_lane < 4; data_lane++) begin
                lane_group = is_v ?
                    {1'b0, v_storage_token[data_lane][6:3]} :
                    k_storage_group[data_lane];
                lane_unit = is_v ? v_storage_token[data_lane][2:0] :
                    k_storage_unit;
                lane_bank = lane_group[1:0] + lane_unit[1:0];
                if (&dma_data_byte_enable[data_lane*8 +: 8]) begin
                    group_bank_write_valid[lane_bank] = 1'b1;
                    group_bank_write_index[lane_bank] =
                        {fill_buffer, lane_group[3:0]};
                    group_bank_write_upper[lane_bank] = lane_unit[2];
                    group_bank_write_data[lane_bank] =
                        dma_data[data_lane*64 +: 64];
                end
            end
        end
    end

    generate
        for (genvar data_lane = 0; data_lane < 4;
             data_lane = data_lane + 1) begin : storage_data_lane_decode
            // A buffer group is 1 KiB. Longer contiguous reads reuse the low
            // offset bits while backpressure protects the other full buffer.
            assign v_storage_token[data_lane] = {1'b0, run_start_token} +
                {2'b00, dma_data_offset[9:3]} + 9'(data_lane);
            assign k_storage_group[data_lane] =
                {1'b0, dma_data_offset[6:3]} + 5'(data_lane);
        end
        assign k_storage_unit = run_start_token[2:0] + dma_data_offset[9:7];
        for (genvar scale_lane = 0; scale_lane < 8;
             scale_lane = scale_lane + 1) begin : storage_scale_lane_decode
            assign scale_storage_unit[scale_lane] = run_start_token[2:0] +
                dma_data_offset[3:1] + 3'(scale_lane);
        end

        for (genvar storage_bank = 0; storage_bank < 4;
             storage_bank = storage_bank + 1) begin : unit_storage_banks
            for (genvar storage_entry = 0; storage_entry < 32;
                 storage_entry = storage_entry + 1) begin : unit_storage_entries
                always_ff @(posedge clk) begin
                    if (group_bank_write_valid[storage_bank] &&
                        group_bank_write_index[storage_bank] == storage_entry) begin
                        if (group_bank_write_upper[storage_bank])
                            group_bank_words[storage_bank][storage_entry][127:64] <=
                                group_bank_write_data[storage_bank];
                        else
                            group_bank_words[storage_bank][storage_entry][63:0] <=
                                group_bank_write_data[storage_bank];
                    end
                end
            end
        end
        for (genvar storage_buffer = 0; storage_buffer < 2;
             storage_buffer = storage_buffer + 1) begin : scale_storage_buffers
            for (genvar storage_scale = 0; storage_scale < 8;
                 storage_scale = storage_scale + 1) begin : scale_storage_entries
                always_ff @(posedge clk) begin
                    if (dma_data_fire && loading_scale && !panel_scale_dma &&
                        fill_buffer == storage_buffer[0]) begin
                        for (integer scale_lane = 0; scale_lane < 8;
                             scale_lane = scale_lane + 1) begin
                            if (&dma_data_byte_enable[scale_lane*2 +: 2] &&
                                scale_storage_unit[scale_lane] == storage_scale)
                                group_scale_words[storage_buffer][storage_scale] <=
                                    dma_data[scale_lane*16 +: 16];
                        end
                    end
                end
            end
        end
    endgenerate

    assign configuration_valid = start_token_count != 0 &&
        {1'b0, start_token_base} + start_token_count <= 13'd2048 &&
        start_head < 6'd32 &&
        start_token_base[2:0] == 3'd0 &&
        (start_is_v || start_token_count <= 12'd256);
    assign start_ready = fill_state == FILL_IDLE && !abort_request;
    assign busy = fill_state != FILL_IDLE;
    assign done_pulse = fill_state == FILL_FINISHED && all_buffers_empty &&
        !sparse_overlay_pending && !sparse_overlay_base_dma &&
        !sparse_overlay_current_pass;
    assign error = done_pulse && terminal_error;
    assign error_id = terminal_error_id;

    assign buffer_available = !buffer_valid[0] || !buffer_valid[1];
    assign selected_available_buffer = !buffer_valid[0] ? 1'b0 : 1'b1;
    assign current_run_bytes = coalesced_run_dma ?
        (is_v ? {17'd0, coalesced_run_token_count, 3'b000} :
                {13'd0, coalesced_run_token_count, 7'b0000000}) :
        loading_scale ? {23'd0, run_token_count, 1'b0} :
        (is_v ? {21'd0, run_token_count, 3'b000} :
                {17'd0, run_token_count, 7'b0000000});
    assign selected_run_base = coalesced_run_dma ? coalesced_run_base_address :
        (loading_scale ?
            (run_current ? group_current_scale_address : group_retained_scale_address) :
            (run_current ? group_current_data_address : group_retained_data_address));
    assign run_next_token = run_start_token + run_token_count;
    assign run_has_next_token = run_next_token < {4'd0, group_token_count};
    // An aligned V group spans two DDR beats. Read holes between selected
    // tokens once; buffer_token_mask prevents overwriting retained values.
    assign run_next_same_region = run_has_next_token &&
        ((sparse_overlay_current_pass && is_v &&
          group_current_data_address[4:0] == 5'd0) ?
            (|(group_current_mask >> run_next_token[2:0])) :
            group_current_mask[run_next_token[2:0]] == run_current);

    assign cache_lookup_valid_mask = valid_token_mask(group_token_count);
    assign cache_lookup_current_valid_mask = cache_lookup_rsp_current_mask &
        cache_lookup_valid_mask;
    assign cache_lookup_group_all_current =
        cache_lookup_current_valid_mask == cache_lookup_valid_mask;
    assign cache_lookup_group_all_retained = cache_lookup_current_valid_mask == 8'h00;
    assign cache_lookup_group_uniform = cache_lookup_group_all_current ||
        cache_lookup_group_all_retained;
    assign cache_lookup_group_current = cache_lookup_group_all_current;
    assign cache_lookup_group_sparse_current =
        !cache_lookup_group_uniform &&
        $countones(cache_lookup_current_valid_mask) <= 32'd2;
    assign coalescing_allowed = is_v || panel_scale_loaded;
    assign cache_lookup_selected_data_address = cache_lookup_group_current ?
        (is_v ? cache_lookup_rsp_current_v_address :
                cache_lookup_rsp_current_k_address) :
        (is_v ? cache_lookup_rsp_retained_v_address :
                cache_lookup_rsp_retained_k_address);
    assign coalesced_run_next_address = coalesced_run_base_address +
        (is_v ? (ADDR_WIDTH'(coalesced_run_token_count) << 3) :
                (ADDR_WIDTH'(coalesced_run_token_count) << 7));
    assign cache_lookup_group_joins_run = cache_lookup_group_uniform &&
        cache_lookup_group_current == coalesced_run_current &&
        cache_lookup_selected_data_address == coalesced_run_next_address;
    assign sparse_overlay_can_replace_run = coalesced_run_active &&
        !coalesced_run_current && !cache_lookup_group_uniform &&
        !sparse_overlay_decided && cache_lookup_group_sparse_current &&
        (is_v || panel_scale_loaded);

    assign coalesced_run_accept_next_lookup =
        fill_state == FILL_CACHE_LOOKUP_RESPONSE &&
        cache_lookup_rsp_valid && cache_lookup_rsp_ready &&
        ((!sparse_overlay_current_pass && coalescing_allowed &&
          cache_lookup_group_uniform) ||
         (sparse_overlay_current_pass && cache_lookup_group_all_current)) &&
        group_offset + 12'(group_token_count) < token_count;
    assign cache_lookup_request_offset = coalesced_run_accept_next_lookup ?
        group_offset + 12'(group_token_count) : group_offset;
    assign cache_lookup_request_tokens = token_count - cache_lookup_request_offset >=
        12'd8 ? 4'd8 : 4'(token_count - cache_lookup_request_offset);
    assign cache_lookup_req_valid =
        (fill_state == FILL_CACHE_LOOKUP_REQUEST ||
         coalesced_run_accept_next_lookup) && !abort_request;
    assign cache_lookup_req_head = head;
    assign cache_lookup_req_key_group =
        token_base + 11'(cache_lookup_request_offset);
    assign cache_lookup_req_chunk = is_v ? {1'b0, channel_tile} : 5'd0;
    assign cache_lookup_req_tag = operation_tag;
    assign cache_lookup_rsp_ready = cache_lookup_outstanding &&
        ((fill_state == FILL_CACHE_LOOKUP_RESPONSE &&
          (!coalesced_run_active || cache_lookup_group_joins_run ||
           sparse_overlay_can_replace_run)) ||
         fill_state == FILL_ERROR_DRAIN || fill_state == FILL_ABORT_DRAIN);
    assign cache_lookup_req_fire = cache_lookup_req_valid && cache_lookup_req_ready;
    assign cache_lookup_rsp_fire = cache_lookup_rsp_valid && cache_lookup_rsp_ready;

    assign dma_start_valid = fill_state == FILL_DMA_REQUEST && !abort_request;
    assign dma_start_address = panel_scale_dma ? group_current_scale_address :
        selected_run_base + (loading_scale ? (ADDR_WIDTH'(run_start_token) << 1) :
         is_v ? (ADDR_WIDTH'(run_start_token) << 3) :
                (ADDR_WIDTH'(run_start_token) << 7));
    assign dma_start_bytes = panel_scale_dma ?
        {19'd0, token_count, 1'b0} : current_run_bytes;
    assign dma_start_fire = dma_start_valid && dma_start_ready;
    assign dma_fire_bytes = count_enabled_bytes(dma_data_byte_enable);
    assign dma_data_offset_in_range = dma_data_offset < dma_expected_bytes;
    assign dma_bytes_remaining_at_offset =
        dma_expected_bytes - dma_data_offset;
    assign dma_data_ready = fill_state == FILL_DMA_STREAM && dma_owned &&
        (!panel_scale_dma || !scale_stream_valid) &&
        (!coalesced_run_dma || !buffer_valid[fill_buffer]);
    assign dma_data_fire = dma_data_valid && dma_data_ready;
    assign dma_transfer_complete = dma_owned && (dma_done_seen || dma_done) &&
        dma_received_bytes + (dma_data_fire ?
            32'(count_enabled_bytes(dma_data_byte_enable)) : 32'd0) ==
        dma_expected_bytes && (!panel_scale_dma ||
            (!scale_stream_valid && !dma_data_fire));
    assign dma_stream_group = dma_data_offset[14:10];
    assign dma_stream_token_offset = is_v ?
        {dma_stream_group, 7'b0000000} :
        {4'd0, dma_stream_group, 3'b000};
    assign dma_stream_tokens_remaining = coalesced_run_token_count -
        dma_stream_token_offset;
    assign dma_stream_group_tokens = is_v ?
        (dma_stream_tokens_remaining >= 12'd128 ? 8'd128 :
            8'(dma_stream_tokens_remaining)) :
        (dma_stream_tokens_remaining >= 12'd8 ? 8'd8 :
            8'(dma_stream_tokens_remaining));
    assign dma_stream_group_complete = coalesced_run_dma && dma_data_fire &&
        (({22'd0, dma_data_offset[9:0]} + 32'(dma_fire_bytes) == 32'd1024) ||
         (dma_data_offset + 32'(dma_fire_bytes) == dma_expected_bytes));
    always_comb begin
        dma_payload_error_this_cycle = 1'b0;
        if (dma_data_fire) begin
            for (integer byte_lane = 0; byte_lane < 32; byte_lane = byte_lane + 1) begin
                if (dma_data_byte_enable[byte_lane] &&
                    (!dma_data_offset_in_range ||
                     (dma_bytes_remaining_at_offset < 32 &&
                      5'(byte_lane) >= dma_bytes_remaining_at_offset[4:0])))
                    dma_payload_error_this_cycle = 1'b1;
            end
            if (dma_received_bytes + 32'(count_enabled_bytes(dma_data_byte_enable)) >
                dma_expected_bytes)
                dma_payload_error_this_cycle = 1'b1;
            if (loading_scale) begin
                if (dma_data_offset[0] != 1'b0)
                    dma_payload_error_this_cycle = 1'b1;
                for (integer scale_lane = 0; scale_lane < 16;
                     scale_lane = scale_lane + 1) begin
                    if (|dma_data_byte_enable[scale_lane*2 +: 2] &&
                        !(&dma_data_byte_enable[scale_lane*2 +: 2]))
                        dma_payload_error_this_cycle = 1'b1;
                end
            end else if (is_v) begin
                if (dma_data_offset[2:0] != 3'd0)
                    dma_payload_error_this_cycle = 1'b1;
                for (integer token_lane = 0; token_lane < 4;
                    token_lane = token_lane + 1) begin
                    if (|dma_data_byte_enable[token_lane*8 +: 8] &&
                        !(&dma_data_byte_enable[token_lane*8 +: 8]))
                        dma_payload_error_this_cycle = 1'b1;
                end
            end else begin
                if (dma_data_offset[2:0] != 3'd0)
                    dma_payload_error_this_cycle = 1'b1;
                for (integer unit_lane = 0; unit_lane < 4;
                     unit_lane = unit_lane + 1) begin
                    if (|dma_data_byte_enable[unit_lane*8 +: 8] &&
                        !(&dma_data_byte_enable[unit_lane*8 +: 8]))
                        dma_payload_error_this_cycle = 1'b1;
                end
            end
        end
    end
    assign dma_response_error_this_cycle = dma_owned && dma_error;

    assign cache_lookup_abort_request = cache_lookup_outstanding &&
        (fill_state == FILL_ABORT_DRAIN || fill_state == FILL_ERROR_DRAIN);
    assign dma_abort_request = (fill_state == FILL_ABORT_DRAIN ||
        fill_state == FILL_ERROR_DRAIN) && dma_owned;

    assign drain_enabled = fill_state != FILL_IDLE &&
        fill_state != FILL_ERROR_DRAIN && fill_state != FILL_ABORT_DRAIN &&
        !abort_request;
    assign panel_write_valid = drain_enabled && drain_word_pending;
    assign panel_write_target_panel = buffer_target_panel[drain_buffer];
    assign panel_write_target_quarter = buffer_target_quarter[drain_buffer];
    assign panel_write_bank = drain_word_bank;
    assign panel_write_address = drain_word_address;
    assign panel_write_data = drain_word_data;
    assign panel_write_byte_enable = drain_word_byte_enable;
    assign panel_write_tag = buffer_tag[drain_buffer];
    assign panel_write_fire = panel_write_valid && panel_write_ready;

    assign scale_write_valid = drain_enabled &&
        (panel_scale_dma ? scale_stream_valid : drain_scale_pending);
    assign scale_write_target_panel = panel_scale_dma ? target_panel :
        buffer_target_panel[drain_buffer];
    assign scale_write_token_base = panel_scale_dma ?
        token_base + 11'(scale_stream_offset[11:1]) +
            (scale_stream_upper ? 11'd8 : 11'd0) :
        buffer_token_base[drain_buffer];
    always_comb begin
        for (integer bank = 0; bank < 4; bank++)
            selected_group_bank_word[bank] =
                group_bank_words[bank][{drain_buffer, drain_write_index[5:2]}];
        selected_drain_group = '0;
        for (integer unit = 0; unit < 8; unit++) begin
            integer bank;
            bank = (integer'(drain_write_index[3:2]) + unit) & 3;
            selected_drain_group[unit*64 +: 64] =
                unit >= 4 ? selected_group_bank_word[bank][127:64] :
                            selected_group_bank_word[bank][63:0];
        end
        selected_drain_word_pair = '0;
        if (buffer_is_v[drain_buffer]) begin
            for (integer token_lane = 0; token_lane < 4;
                 token_lane = token_lane + 1) begin
                selected_drain_word_pair[token_lane*64 +: 64] =
                    selected_drain_group[
                        ({drain_write_index[1], 2'b00} + 3'(token_lane))*64 +: 64];
            end
        end else begin
            for (integer token_lane = 0; token_lane < 8;
                 token_lane = token_lane + 1) begin
                selected_drain_word_pair[token_lane*16 +: 16] =
                    selected_drain_group[
                        token_lane*64 + drain_write_index[1]*32 +: 16];
                selected_drain_word_pair[128 + token_lane*16 +: 16] =
                    selected_drain_group[
                        token_lane*64 + drain_write_index[1]*32 + 16 +: 16];
            end
        end
        if (panel_scale_dma) begin
            scale_write_data = scale_stream_upper ?
                scale_stream_data[255:128] : scale_stream_data[127:0];
        end else begin
            scale_write_data = '0;
            for (integer scale_lane = 0; scale_lane < 8;
                 scale_lane = scale_lane + 1)
                scale_write_data[scale_lane*16 +: 16] =
                    group_scale_words[drain_buffer][scale_lane];
        end
    end
    assign scale_write_byte_enable = panel_scale_dma ?
        (scale_stream_upper ? scale_stream_byte_enable[31:16] :
                              scale_stream_byte_enable[15:0]) :
        16'((17'h1 << (buffer_group_tokens[drain_buffer] * 2)) - 1'b1);
    assign scale_write_tag = panel_scale_dma ? operation_tag :
        buffer_tag[drain_buffer];
    assign scale_write_fire = scale_write_valid && scale_write_ready;
    assign all_buffers_empty = !buffer_valid[0] && !buffer_valid[1] &&
        !drain_active && !drain_word_pending && !drain_scale_pending;
    assign drain_word_count = buffer_is_v[drain_buffer] ?
        7'((buffer_group_tokens[drain_buffer] + 8'd1) >> 1) : 7'd64;
    assign drain_last_word = drain_write_index + 7'd2 >= drain_word_count;
    assign drain_v_token_offset = {2'd0, drain_write_index} << 1;
    assign drain_v_remaining_tokens =
        {1'b0, buffer_group_tokens[drain_buffer]} > drain_v_token_offset ?
        {1'b0, buffer_group_tokens[drain_buffer]} - drain_v_token_offset : 9'd0;

`ifdef SYNTHESIS
    always_comb begin
        accepted_cache_lookup_request_count = '0;
        accepted_dma_request_count = '0;
        accepted_dma_byte_count = '0;
        accepted_panel_write_count = '0;
        accepted_scale_write_count = '0;
        load_compute_overlap_cycle_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            fill_state <= FILL_IDLE;
            is_v <= 1'b0;
            head <= '0;
            token_base <= '0;
            token_count <= '0;
            channel_tile <= '0;
            target_panel <= 1'b0;
            target_quarter <= 1'b0;
            operation_tag <= '0;
            group_offset <= '0;
            group_token_count <= '0;
            group_current_mask <= '0;
            group_current_data_address <= '0;
            group_retained_data_address <= '0;
            group_current_scale_address <= '0;
            group_retained_scale_address <= '0;
            fill_buffer <= 1'b0;
            loading_scale <= 1'b0;
            scan_token <= '0;
            run_current <= 1'b0;
            run_start_token <= '0;
            run_token_count <= '0;
            coalesced_run_active <= 1'b0;
            coalesced_run_dma <= 1'b0;
            coalesced_run_current <= 1'b0;
            coalesced_run_start_offset <= '0;
            coalesced_run_token_count <= '0;
            coalesced_run_base_address <= '0;
            sparse_overlay_pending <= 1'b0;
            sparse_overlay_base_dma <= 1'b0;
            sparse_overlay_current_pass <= 1'b0;
            sparse_overlay_decided <= 1'b0;
            sparse_overlay_start_offset <= '0;
            sparse_overlay_retained_base_address <= '0;
            dma_owned <= 1'b0;
            cache_lookup_outstanding <= 1'b0;
            dma_done_seen <= 1'b0;
            dma_expected_bytes <= '0;
            dma_received_bytes <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            panel_scale_loaded <= 1'b0;
            panel_scale_dma <= 1'b0;
            scale_stream_valid <= 1'b0;
            scale_stream_upper <= 1'b0;
            scale_stream_data <= '0;
            scale_stream_byte_enable <= '0;
            scale_stream_offset <= '0;
            buffer_valid[0] <= 1'b0;
            buffer_valid[1] <= 1'b0;
            buffer_token_mask[0] <= '0;
            buffer_token_mask[1] <= '0;
            buffer_sparse_overlay[0] <= 1'b0;
            buffer_sparse_overlay[1] <= 1'b0;
            drain_active <= 1'b0;
            drain_buffer <= 1'b0;
            drain_write_index <= '0;
            drain_word_pending <= 1'b0;
            drain_word_data <= '0;
            drain_word_byte_enable <= '0;
            drain_word_bank <= 1'b0;
            drain_word_address <= '0;
            drain_scale_pending <= 1'b0;
            abort_ack <= 1'b0;
`ifndef SYNTHESIS
            accepted_cache_lookup_request_count <= '0;
            accepted_dma_request_count <= '0;
            accepted_dma_byte_count <= '0;
            accepted_panel_write_count <= '0;
            accepted_scale_write_count <= '0;
            load_compute_overlap_cycle_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;

            if (start_valid && start_ready) begin
                is_v <= start_is_v;
                head <= start_head;
                token_base <= start_token_base;
                token_count <= start_token_count;
                channel_tile <= start_channel_tile;
                target_panel <= start_target_panel;
                target_quarter <= start_target_quarter;
                operation_tag <= start_tag;
                group_offset <= 12'd0;
                loading_scale <= 1'b0;
                coalesced_run_active <= 1'b0;
                coalesced_run_dma <= 1'b0;
                coalesced_run_token_count <= '0;
                sparse_overlay_pending <= 1'b0;
                sparse_overlay_base_dma <= 1'b0;
                sparse_overlay_current_pass <= 1'b0;
                sparse_overlay_decided <= 1'b0;
                sparse_overlay_start_offset <= '0;
                sparse_overlay_retained_base_address <= '0;
                panel_scale_loaded <= 1'b0;
                panel_scale_dma <= 1'b0;
                scale_stream_valid <= 1'b0;
                scale_stream_upper <= 1'b0;
                terminal_error <= !configuration_valid;
                terminal_error_id <= configuration_valid ? 4'd0 : ERROR_CONFIGURATION;
                fill_state <= configuration_valid ? FILL_WAIT_BUFFER : FILL_FINISHED;
            end

            if (cache_lookup_req_fire) begin
                cache_lookup_outstanding <= 1'b1;
                group_token_count <= cache_lookup_request_tokens;
`ifndef SYNTHESIS
                accepted_cache_lookup_request_count <=
                    accepted_cache_lookup_request_count + 64'd1;
`endif
                fill_state <= FILL_CACHE_LOOKUP_RESPONSE;
            end

            if (cache_lookup_rsp_fire) begin
                cache_lookup_outstanding <= cache_lookup_req_fire;
                if (cache_lookup_rsp_key_group != token_base + 11'(group_offset) ||
                    cache_lookup_rsp_tag != operation_tag ||
                    (cache_lookup_rsp_current_mask |
                     cache_lookup_rsp_retained_mask) != 8'hff ||
                    (cache_lookup_rsp_current_mask &
                     cache_lookup_rsp_retained_mask) != 8'h00) begin
                    terminal_error <= 1'b1;
                    terminal_error_id <= ERROR_CACHE_LOOKUP;
                    fill_state <= FILL_ERROR_DRAIN;
                end else if ((!sparse_overlay_current_pass &&
                              coalescing_allowed &&
                              cache_lookup_group_uniform) ||
                             (sparse_overlay_current_pass &&
                              cache_lookup_group_all_current)) begin
                    if (!coalesced_run_active) begin
                        coalesced_run_active <= 1'b1;
                        coalesced_run_current <= cache_lookup_group_current;
                        coalesced_run_start_offset <= group_offset;
                        coalesced_run_token_count <=
                            {8'd0, group_token_count};
                        coalesced_run_base_address <=
                            cache_lookup_selected_data_address;
                    end else begin
                        coalesced_run_token_count <=
                            coalesced_run_token_count +
                            12'(group_token_count);
                    end
                    group_offset <= group_offset + 12'(group_token_count);
                    if (group_offset + 12'(group_token_count) >=
                        token_count) begin
                        run_current <= coalesced_run_active ?
                            coalesced_run_current : cache_lookup_group_current;
                        run_start_token <= 8'd0;
                        coalesced_run_token_count <=
                            (coalesced_run_active ?
                                coalesced_run_token_count : 12'd0) +
                            12'(group_token_count);
                        coalesced_run_dma <= 1'b1;
                        fill_state <= FILL_DMA_REQUEST;
                    end else
                        fill_state <= cache_lookup_req_fire ?
                            FILL_CACHE_LOOKUP_RESPONSE :
                            FILL_CACHE_LOOKUP_REQUEST;
                end else if (sparse_overlay_current_pass) begin
                    group_current_mask <= cache_lookup_current_valid_mask;
                    group_current_data_address <= is_v ?
                        cache_lookup_rsp_current_v_address :
                        cache_lookup_rsp_current_k_address;
                    group_retained_data_address <= is_v ?
                        cache_lookup_rsp_retained_v_address :
                        cache_lookup_rsp_retained_k_address;
                    group_current_scale_address <=
                        cache_lookup_rsp_current_k_scale_address;
                    group_retained_scale_address <=
                        cache_lookup_rsp_retained_k_scale_address;
                    loading_scale <= 1'b0;
                    if (cache_lookup_current_valid_mask == 8'h00) begin
                        group_offset <= group_offset + 12'(group_token_count);
                        if (group_offset + 12'(group_token_count) >=
                            token_count) begin
                            sparse_overlay_current_pass <= 1'b0;
                            fill_state <= FILL_FINISHED;
                        end else begin
                            fill_state <= FILL_CACHE_LOOKUP_REQUEST;
                        end
                    end else begin
                        scan_token <= 4'd0;
                        fill_state <= FILL_FIND_RUN;
                    end
                end else begin
                    if (!cache_lookup_group_uniform)
                        sparse_overlay_decided <= 1'b1;
                    group_current_mask <= cache_lookup_rsp_current_mask &
                        valid_token_mask(group_token_count);
                    group_current_data_address <= is_v ?
                        cache_lookup_rsp_current_v_address :
                        cache_lookup_rsp_current_k_address;
                    group_retained_data_address <= is_v ?
                        cache_lookup_rsp_retained_v_address :
                        cache_lookup_rsp_retained_k_address;
                    group_current_scale_address <=
                        cache_lookup_rsp_current_k_scale_address;
                    group_retained_scale_address <=
                        cache_lookup_rsp_retained_k_scale_address;
                    sparse_overlay_start_offset <=
                        sparse_overlay_can_replace_run ?
                            coalesced_run_start_offset : group_offset;
                    sparse_overlay_retained_base_address <=
                        sparse_overlay_can_replace_run ?
                            coalesced_run_base_address :
                            (is_v ? cache_lookup_rsp_retained_v_address :
                                    cache_lookup_rsp_retained_k_address);
                    if (!is_v && group_offset == 12'd0 &&
                        cache_lookup_rsp_current_k_scale_address ==
                            cache_lookup_rsp_retained_k_scale_address) begin
                        sparse_overlay_pending <=
                            !sparse_overlay_decided &&
                            cache_lookup_group_sparse_current;
                        loading_scale <= 1'b1;
                        panel_scale_dma <= 1'b1;
                        fill_state <= FILL_DMA_REQUEST;
                    end else if (!sparse_overlay_decided &&
                                cache_lookup_group_sparse_current &&
                                (is_v || panel_scale_loaded)) begin
                        loading_scale <= 1'b0;
                        sparse_overlay_base_dma <= 1'b1;
                        coalesced_run_active <= 1'b1;
                        coalesced_run_dma <= 1'b1;
                        coalesced_run_current <= 1'b0;
                        coalesced_run_start_offset <=
                            sparse_overlay_can_replace_run ?
                                coalesced_run_start_offset : group_offset;
                        coalesced_run_token_count <= token_count -
                            (sparse_overlay_can_replace_run ?
                                coalesced_run_start_offset : group_offset);
                        coalesced_run_base_address <=
                            sparse_overlay_can_replace_run ?
                                coalesced_run_base_address :
                                (is_v ? cache_lookup_rsp_retained_v_address :
                                        cache_lookup_rsp_retained_k_address);
                        run_current <= 1'b0;
                        run_start_token <= 8'd0;
                        group_offset <= token_count;
                        fill_state <= FILL_DMA_REQUEST;
                    end else begin
                        loading_scale <= 1'b0;
                        scan_token <= 4'd0;
                        fill_state <= FILL_FIND_RUN;
                    end
                end
            end

            if (fill_state == FILL_CACHE_LOOKUP_RESPONSE &&
                cache_lookup_rsp_valid && cache_lookup_outstanding &&
                coalesced_run_active && !cache_lookup_group_joins_run) begin
                run_current <= coalesced_run_current;
                run_start_token <= 8'd0;
                coalesced_run_dma <= 1'b1;
                fill_state <= FILL_DMA_REQUEST;
            end

            case (fill_state)
                FILL_IDLE: begin end
                FILL_WAIT_BUFFER: if (buffer_available) begin
                    fill_buffer <= selected_available_buffer;
                    fill_state <= cache_lookup_outstanding ?
                        FILL_CACHE_LOOKUP_RESPONSE : FILL_CACHE_LOOKUP_REQUEST;
                end
                FILL_CACHE_LOOKUP_REQUEST: begin end
                FILL_CACHE_LOOKUP_RESPONSE: begin end
                FILL_FIND_RUN: begin
                    if (scan_token >= group_token_count) begin
                        if (!is_v && !loading_scale && !panel_scale_loaded) begin
                            loading_scale <= 1'b1;
                            scan_token <= 4'd0;
                        end else begin
                            fill_state <= FILL_COMMIT_GROUP;
                        end
                    end else if (sparse_overlay_current_pass &&
                                 !group_current_mask[scan_token[2:0]]) begin
                        scan_token <= scan_token + 4'd1;
                    end else begin
                        run_current <= sparse_overlay_current_pass ? 1'b1 :
                            group_current_mask[scan_token[2:0]];
                        run_start_token <= {4'd0, scan_token};
                        run_token_count <= 8'd1;
                        fill_state <= FILL_EXTEND_RUN;
                    end
                end
                FILL_EXTEND_RUN: begin
                    if (run_has_next_token && run_next_same_region) begin
                        run_token_count <= run_token_count + 8'd1;
                    end else begin
                        fill_state <= FILL_DMA_REQUEST;
                    end
                end
                FILL_DMA_REQUEST: begin end
                FILL_DMA_STREAM: begin end
                FILL_COMMIT_GROUP: begin
                    buffer_valid[fill_buffer] <= 1'b1;
                    buffer_is_v[fill_buffer] <= is_v;
                    buffer_target_panel[fill_buffer] <= target_panel;
                    buffer_target_quarter[fill_buffer] <= target_quarter;
                    buffer_group_index[fill_buffer] <= 8'(group_offset >> 3);
                    buffer_group_tokens[fill_buffer] <=
                        {4'd0, group_token_count};
                    buffer_token_mask[fill_buffer] <=
                        sparse_overlay_current_pass ? group_current_mask :
                            valid_token_mask(group_token_count);
                    buffer_sparse_overlay[fill_buffer] <=
                        sparse_overlay_current_pass;
                    buffer_token_base[fill_buffer] <= token_base +
                        11'(group_offset);
                    buffer_tag[fill_buffer] <= operation_tag;
                    if (group_offset + 12'd8 >= token_count) begin
                        sparse_overlay_current_pass <= 1'b0;
                        fill_state <= FILL_FINISHED;
                    end else begin
                        group_offset <= group_offset + 12'd8;
                        fill_state <= FILL_WAIT_BUFFER;
                    end
                end
                FILL_WAIT_BASE_DRAIN: if (all_buffers_empty) begin
                    sparse_overlay_current_pass <= 1'b1;
                    group_offset <= sparse_overlay_start_offset;
                    fill_state <= FILL_CACHE_LOOKUP_REQUEST;
                end
                FILL_FINISHED: if (all_buffers_empty)
                    fill_state <= FILL_IDLE;
                FILL_ERROR_DRAIN: begin
                    if (!dma_owned && !cache_lookup_outstanding) begin
                        buffer_valid[0] <= 1'b0;
                        buffer_valid[1] <= 1'b0;
                        buffer_token_mask[0] <= '0;
                        buffer_token_mask[1] <= '0;
                        buffer_sparse_overlay[0] <= 1'b0;
                        buffer_sparse_overlay[1] <= 1'b0;
                        drain_active <= 1'b0;
                        drain_word_pending <= 1'b0;
                        drain_scale_pending <= 1'b0;
                        panel_scale_dma <= 1'b0;
                        scale_stream_valid <= 1'b0;
                        sparse_overlay_pending <= 1'b0;
                        sparse_overlay_base_dma <= 1'b0;
                        sparse_overlay_current_pass <= 1'b0;
                        sparse_overlay_decided <= 1'b0;
                        fill_state <= FILL_FINISHED;
                    end
                end
                FILL_ABORT_DRAIN: begin
                    if (!dma_owned && !cache_lookup_outstanding) begin
                        buffer_valid[0] <= 1'b0;
                        buffer_valid[1] <= 1'b0;
                        buffer_token_mask[0] <= '0;
                        buffer_token_mask[1] <= '0;
                        buffer_sparse_overlay[0] <= 1'b0;
                        buffer_sparse_overlay[1] <= 1'b0;
                        drain_active <= 1'b0;
                        drain_word_pending <= 1'b0;
                        drain_scale_pending <= 1'b0;
                        panel_scale_dma <= 1'b0;
                        scale_stream_valid <= 1'b0;
                        sparse_overlay_pending <= 1'b0;
                        sparse_overlay_base_dma <= 1'b0;
                        sparse_overlay_current_pass <= 1'b0;
                        sparse_overlay_decided <= 1'b0;
                        abort_ack <= 1'b1;
                        fill_state <= FILL_IDLE;
                    end
                end
                default: fill_state <= FILL_IDLE;
            endcase

            if (dma_start_fire) begin
                dma_owned <= 1'b1;
                dma_done_seen <= 1'b0;
                dma_expected_bytes <= dma_start_bytes;
                dma_received_bytes <= 32'd0;
`ifndef SYNTHESIS
                accepted_dma_request_count <= accepted_dma_request_count + 64'd1;
`endif
                fill_state <= FILL_DMA_STREAM;
            end

            if (dma_data_fire) begin
`ifndef SYNTHESIS
                accepted_dma_byte_count <= accepted_dma_byte_count +
                    64'(count_enabled_bytes(dma_data_byte_enable));
`endif
                dma_received_bytes <= dma_received_bytes +
                    32'(count_enabled_bytes(dma_data_byte_enable));
                if (panel_scale_dma) begin
                    scale_stream_valid <= 1'b1;
                    scale_stream_upper <= 1'b0;
                    scale_stream_data <= dma_data;
                    scale_stream_byte_enable <= dma_data_byte_enable;
                    scale_stream_offset <= dma_data_offset;
                end
                if (dma_stream_group_complete) begin
                    buffer_valid[fill_buffer] <= 1'b1;
                    buffer_is_v[fill_buffer] <= is_v;
                    buffer_target_panel[fill_buffer] <= target_panel;
                    buffer_target_quarter[fill_buffer] <= target_quarter;
                    buffer_group_index[fill_buffer] <= 8'(
                        (coalesced_run_start_offset +
                         dma_stream_token_offset) >> 3);
                    buffer_group_tokens[fill_buffer] <=
                        dma_stream_group_tokens;
                    buffer_token_mask[fill_buffer] <=
                        '0;
                    buffer_sparse_overlay[fill_buffer] <= 1'b0;
                    buffer_token_base[fill_buffer] <= token_base + 11'(
                        coalesced_run_start_offset +
                        dma_stream_token_offset);
                    buffer_tag[fill_buffer] <= operation_tag;
                    fill_buffer <= !fill_buffer;
                end
            end
            if (dma_done && dma_owned)
                dma_done_seen <= 1'b1;
            if (dma_response_error_this_cycle || dma_payload_error_this_cycle) begin
                terminal_error <= 1'b1;
                terminal_error_id <= dma_response_error_this_cycle ?
                    ERROR_DMA : ERROR_DMA_PAYLOAD;
                fill_state <= FILL_ERROR_DRAIN;
                if (dma_done)
                    dma_owned <= 1'b0;
            end
            if (dma_transfer_complete) begin
                dma_owned <= 1'b0;
                dma_done_seen <= 1'b0;
                if (terminal_error || dma_response_error_this_cycle ||
                    dma_payload_error_this_cycle) begin
                    fill_state <= FILL_ERROR_DRAIN;
                end else if (panel_scale_dma) begin
                    panel_scale_dma <= 1'b0;
                    panel_scale_loaded <= 1'b1;
                    loading_scale <= 1'b0;
                    if (sparse_overlay_pending) begin
                        sparse_overlay_pending <= 1'b0;
                        sparse_overlay_base_dma <= 1'b1;
                        coalesced_run_active <= 1'b1;
                        coalesced_run_dma <= 1'b1;
                        coalesced_run_current <= 1'b0;
                        coalesced_run_start_offset <=
                            sparse_overlay_start_offset;
                        coalesced_run_token_count <= token_count -
                            sparse_overlay_start_offset;
                        coalesced_run_base_address <=
                            sparse_overlay_retained_base_address;
                        run_current <= 1'b0;
                        run_start_token <= 8'd0;
                        group_offset <= token_count;
                        fill_state <= FILL_DMA_REQUEST;
                    end else if (group_current_mask ==
                            valid_token_mask(group_token_count) ||
                        group_current_mask == 8'h00) begin
                        coalesced_run_active <= 1'b1;
                        coalesced_run_current <= group_current_mask != 8'h00;
                        coalesced_run_start_offset <= group_offset;
                        coalesced_run_token_count <=
                            {8'd0, group_token_count};
                        coalesced_run_base_address <=
                            group_current_mask != 8'h00 ?
                                group_current_data_address :
                                group_retained_data_address;
                        run_current <= group_current_mask != 8'h00;
                        run_start_token <= 8'd0;
                        group_offset <= group_offset +
                            12'(group_token_count);
                        if (group_offset + 12'(group_token_count) >=
                            token_count) begin
                            coalesced_run_dma <= 1'b1;
                            fill_state <= FILL_DMA_REQUEST;
                        end else begin
                            fill_state <= FILL_CACHE_LOOKUP_REQUEST;
                        end
                    end else begin
                        scan_token <= 4'd0;
                        fill_state <= FILL_FIND_RUN;
                    end
                end else if (coalesced_run_dma && sparse_overlay_base_dma) begin
                    sparse_overlay_base_dma <= 1'b0;
                    coalesced_run_active <= 1'b0;
                    coalesced_run_dma <= 1'b0;
                    coalesced_run_token_count <= '0;
                    fill_state <= FILL_WAIT_BASE_DRAIN;
                end else if (coalesced_run_dma) begin
                    coalesced_run_active <= 1'b0;
                    coalesced_run_dma <= 1'b0;
                    coalesced_run_token_count <= '0;
                    if (group_offset >= token_count &&
                        !cache_lookup_outstanding) begin
                        if (sparse_overlay_current_pass)
                            sparse_overlay_current_pass <= 1'b0;
                        fill_state <= FILL_FINISHED;
                    end else begin
                        fill_state <= FILL_WAIT_BUFFER;
                    end
                end else begin
                    scan_token <= 4'(run_start_token + run_token_count);
                    fill_state <= FILL_FIND_RUN;
                end
            end
            if ((fill_state == FILL_ABORT_DRAIN ||
                 fill_state == FILL_ERROR_DRAIN) && dma_abort_ack)
                dma_owned <= 1'b0;
            if ((fill_state == FILL_ABORT_DRAIN ||
                 fill_state == FILL_ERROR_DRAIN) && cache_lookup_abort_ack)
                cache_lookup_outstanding <= 1'b0;

            if (drain_enabled && !drain_active && !drain_word_pending && !drain_scale_pending &&
                (buffer_valid[0] || buffer_valid[1])) begin
                drain_active <= 1'b1;
                drain_buffer <= buffer_valid[0] ? 1'b0 : 1'b1;
                drain_write_index <= 7'd0;
            end

            if (drain_enabled && drain_active && !drain_word_pending &&
                !drain_scale_pending) begin
                drain_word_pending <= 1'b1;
                if (buffer_is_v[drain_buffer]) begin
                    drain_word_data[127:0] <= selected_drain_word_pair[127:0];
                    drain_word_data[255:128] <=
                        drain_write_index + 7'd1 < drain_word_count ?
                        selected_drain_word_pair[255:128] :
                        128'd0;
                    if (buffer_sparse_overlay[drain_buffer]) begin
                        drain_word_byte_enable <= v_token_byte_mask(
                            buffer_token_mask[drain_buffer],
                            drain_v_token_offset[2:0]);
                    end else begin
                        drain_word_byte_enable[15:0] <=
                            drain_v_remaining_tokens >= 9'd2 ? 16'hffff :
                            drain_v_remaining_tokens == 9'd1 ?
                                16'h00ff : 16'h0000;
                        drain_word_byte_enable[31:16] <=
                            drain_v_remaining_tokens >= 9'd4 ? 16'hffff :
                            drain_v_remaining_tokens == 9'd3 ?
                                16'h00ff : 16'h0000;
                    end
                    drain_word_bank <= drain_write_index[0];
                    drain_word_address <=
                        (10'(buffer_group_index[drain_buffer]) << 1) +
                        10'(drain_write_index[6:1]);
                end else begin
                    drain_word_data <= selected_drain_word_pair;
                    drain_word_byte_enable <= buffer_sparse_overlay[drain_buffer] ?
                        {2{two_byte_token_mask(
                            buffer_token_mask[drain_buffer])}} :
                        {2{16'((17'h1 <<
                            (buffer_group_tokens[drain_buffer]*2)) - 1'b1)}};
                    drain_word_bank <= drain_write_index[0];
                    drain_word_address <=
                        (10'(buffer_group_index[drain_buffer]) << 5) +
                        (10'(drain_write_index[6:2]) << 1) +
                        10'(drain_write_index[1]);
                end
            end

            if (panel_write_fire) begin
`ifndef SYNTHESIS
                accepted_panel_write_count <= accepted_panel_write_count + 64'd1;
`endif
                drain_word_pending <= 1'b0;
                if (drain_last_word) begin
                    if (buffer_is_v[drain_buffer]) begin
                        buffer_valid[drain_buffer] <= 1'b0;
                        buffer_token_mask[drain_buffer] <= '0;
                        buffer_sparse_overlay[drain_buffer] <= 1'b0;
                        drain_active <= 1'b0;
                    end else if (panel_scale_loaded) begin
                        buffer_valid[drain_buffer] <= 1'b0;
                        buffer_token_mask[drain_buffer] <= '0;
                        buffer_sparse_overlay[drain_buffer] <= 1'b0;
                        drain_active <= 1'b0;
                    end else begin
                        drain_scale_pending <= 1'b1;
                    end
                end else begin
                    drain_write_index <= drain_write_index + 7'd2;
                end
            end

            if (scale_write_fire) begin
`ifndef SYNTHESIS
                accepted_scale_write_count <= accepted_scale_write_count + 64'd1;
`endif
                if (panel_scale_dma) begin
                    if (!scale_stream_upper &&
                        |scale_stream_byte_enable[31:16]) begin
                        scale_stream_upper <= 1'b1;
                    end else begin
                        scale_stream_valid <= 1'b0;
                        scale_stream_upper <= 1'b0;
                    end
                end else begin
                    drain_scale_pending <= 1'b0;
                    buffer_valid[drain_buffer] <= 1'b0;
                    buffer_token_mask[drain_buffer] <= '0;
                    buffer_sparse_overlay[drain_buffer] <= 1'b0;
                    drain_active <= 1'b0;
                end
            end

`ifndef SYNTHESIS
            if (dma_owned && drain_active)
                load_compute_overlap_cycle_count <=
                    load_compute_overlap_cycle_count + 64'd1;
`endif

            if (abort_request && fill_state != FILL_IDLE &&
                fill_state != FILL_ABORT_DRAIN)
                fill_state <= FILL_ABORT_DRAIN;
        end
    end

    initial begin
        if (ADDR_WIDTH != 64 || TAG_WIDTH < 16)
            $error("attention_cache_loader requires 64-bit addresses and TAG_WIDTH >= 16");
    end

`ifndef SYNTHESIS
    logic held_panel_write;
    logic [1+10+256+32+TAG_WIDTH-1:0] held_panel_payload;
    logic previous_sparse_overlay_current_pass;
    logic previous_wait_base_drain;
    logic previous_all_buffers_empty;
    always_ff @(posedge clk) begin
        if (rst) begin
            held_panel_write <= 1'b0;
            held_panel_payload <= '0;
            previous_sparse_overlay_current_pass <= 1'b0;
            previous_wait_base_drain <= 1'b0;
            previous_all_buffers_empty <= 1'b0;
        end else begin
            if (held_panel_write)
                assert (panel_write_valid &&
                    {panel_write_bank, panel_write_address, panel_write_data,
                     panel_write_byte_enable, panel_write_tag} == held_panel_payload)
                    else $error("attention cache loader changed a stalled panel write");
            held_panel_write <= panel_write_valid && !panel_write_ready;
            held_panel_payload <= {panel_write_bank, panel_write_address,
                panel_write_data, panel_write_byte_enable, panel_write_tag};
            assert (!(buffer_valid[0] && buffer_valid[1] && !drain_active))
                else $error("attention cache loader left two full buffers without a drain");
            if (sparse_overlay_current_pass &&
                !previous_sparse_overlay_current_pass)
                assert (previous_wait_base_drain &&
                        previous_all_buffers_empty)
                    else $error("attention cache loader started current-token overlay before retained-base buffers drained");
            if (panel_write_fire &&
                buffer_sparse_overlay[drain_buffer]) begin
                if (buffer_is_v[drain_buffer])
                    assert ((panel_write_byte_enable &
                        ~v_token_byte_mask(buffer_token_mask[drain_buffer],
                            drain_v_token_offset[2:0])) == 32'd0)
                        else $error("attention cache loader V overlay wrote a byte outside the current-token mask");
                else
                    assert ((panel_write_byte_enable &
                        ~{2{two_byte_token_mask(
                            buffer_token_mask[drain_buffer])}}) == 32'd0)
                        else $error("attention cache loader K overlay wrote a byte outside the current-token mask");
            end
            if (done_pulse)
                assert (all_buffers_empty && !sparse_overlay_pending &&
                    !sparse_overlay_base_dma &&
                    !sparse_overlay_current_pass)
                    else $error("attention cache loader completed with sparse overlay work pending");
            previous_sparse_overlay_current_pass <=
                sparse_overlay_current_pass;
            previous_wait_base_drain <=
                fill_state == FILL_WAIT_BASE_DRAIN;
            previous_all_buffers_empty <= all_buffers_empty;
        end
    end
`endif
endmodule

`default_nettype wire
