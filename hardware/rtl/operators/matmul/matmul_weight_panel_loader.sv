`default_nettype none

module matmul_weight_panel_loader #(
    parameter integer OPERATOR_ID_WIDTH = 8,
    parameter integer STRIPE_ID_WIDTH = 16
) (
    input  logic                            clk,
    input  logic                            rst,
    input  logic                            abort_request,
    output logic                            abort_ack,

    input  logic                            load_req_valid,
    output logic                            load_req_ready,
    input  logic [OPERATOR_ID_WIDTH-1:0]    load_req_operator_id,
    input  logic [STRIPE_ID_WIDTH-1:0]      load_req_stripe_id,
    input  logic [31:0]                     load_req_input_features,
    input  logic [63:0]                     load_req_base_addr,
    input  logic [63:0]                     load_req_enhancement_addr,
    input  logic [63:0]                     load_req_scale_addr,
    input  logic                            load_req_operator_has_w8,
    input  logic                            load_req_qkv_layout,
    input  logic                            load_req_allow_deferred,
    input  logic                            load_req_block_panel1,

    output logic                            load_done_valid,
    input  logic                            load_done_ready,
    output logic                            load_done_panel,
    output logic [3:0]                      load_done_error,

    output logic                            dma_req_valid,
    input  logic                            dma_req_ready,
    output logic [63:0]                     dma_req_addr,
    output logic [31:0]                     dma_req_bytes,
    output logic [1:0]                      dma_req_plane,
    output logic                            dma_req_panel,
    output logic                            dma_req_last_for_plane,
    input  logic                            dma_req_done,
    input  logic                            dma_req_error,

    input  logic                            dma_rsp_valid,
    output logic                            dma_rsp_ready,
    input  logic [255:0]                    dma_rsp_data,
    input  logic [31:0]                     dma_rsp_byte_enable,
    input  logic                            dma_rsp_error,
    input  logic                            dma_rsp_last,
    input  logic                            dma_abort_ack,

    output logic                            panel_write_valid,
    input  logic                            panel_write_ready,
    output logic                            panel_write_panel,
    output logic [1:0]                      panel_write_plane,
    output logic [2:0]                      panel_write_start_bank,
    output logic [9:0]                      panel_write_word_row,
    output logic [255:0]                    panel_write_data,
    output logic [31:0]                     panel_write_byte_enable,

    output logic [127:0]                    panel0_scale_base,
    output logic                            panel0_scale_valid,
    output logic [127:0]                    panel1_scale_base,
    output logic                            panel1_scale_valid,

    input  logic [OPERATOR_ID_WIDTH-1:0]    compute_request_operator_id,
    input  logic [STRIPE_ID_WIDTH-1:0]      compute_request_stripe_id,
    output logic                            compute_acquire_valid,
    input  logic                            compute_acquire_ready,
    output logic                            compute_acquire_panel,
    output logic [OPERATOR_ID_WIDTH-1:0]    compute_acquire_operator_id,
    output logic [STRIPE_ID_WIDTH-1:0]      compute_acquire_stripe_id,
    output logic                            compute_acquire_has_enhancement,
    input  logic                            compute_release_valid,
    output logic                            compute_release_ready,
    input  logic                            compute_release_panel,

    output logic [1:0]                      panel_valid,
    output logic [1:0]                      panel_loading,
    output logic [1:0]                      panel_computing,
    output logic [2*OPERATOR_ID_WIDTH-1:0]  panel_operator_id,
    output logic [2*STRIPE_ID_WIDTH-1:0]    panel_stripe_id,
    output logic [1:0]                      panel_has_enhancement,

    output logic [63:0]                     accepted_load_count,
    output logic [63:0]                     accepted_dma_request_count,
    output logic [63:0]                     accepted_base_request_count,
    output logic [63:0]                     accepted_enhancement_request_count,
    output logic [63:0]                     accepted_base_request_bytes,
    output logic [63:0]                     accepted_enhancement_request_bytes,
    output logic [63:0]                     accepted_scale_request_count,
    output logic [63:0]                     accepted_scale_request_bytes,
    output logic [63:0]                     accepted_dma_response_count,
    output logic [63:0]                     accepted_dma_response_bytes,
    output logic [63:0]                     discarded_dma_response_bytes,
    output logic [63:0]                     four_kib_split_count,
    output logic [63:0]                     dma_error_count
);
    localparam logic [3:0] LOAD_OK = 4'h0;
    localparam logic [3:0] LOAD_BASE_RANGE_ERROR = 4'h1;
    localparam logic [3:0] LOAD_ENH_RANGE_ERROR = 4'h2;
    localparam logic [3:0] LOAD_DMA_ERROR = 4'h3;
    localparam logic [3:0] LOAD_DMA_PROTOCOL_ERROR = 4'h4;
    localparam logic [3:0] LOAD_SCALE_RANGE_ERROR = 4'h5;

    logic [OPERATOR_ID_WIDTH-1:0] panel_operator [0:1];
    logic [STRIPE_ID_WIDTH-1:0] panel_stripe [0:1];
    logic [1:0] panel_enhancement;
    logic next_panel;
    logic compute_next_panel;

    logic active_load;
    logic active_panel;
    logic active_wait_for_panel;
    logic [OPERATOR_ID_WIDTH-1:0] active_operator_id;
    logic [STRIPE_ID_WIDTH-1:0] active_stripe_id;
    logic active_has_w8;
    logic [1:0] active_plane;
    logic [63:0] active_base_addr;
    logic [63:0] active_enhancement_addr;
    logic [63:0] active_scale_addr;
    logic [31:0] stripe_bytes;
    logic dma_request_outstanding;
    logic dma_payload_complete;
    logic [31:0] response_remaining;
    logic [2:0] active_bank_count;
    logic [9:0] active_plane_rows;
    logic [2:0] response_panel_bank;
    logic [9:0] response_panel_row;
    logic load_failed;
    logic [3:0] load_error;
    logic abort_pending;

    // 128 bits x 8 entries x 2 lines = 256 B; one serial DMA refill.
    // Each line supplies one 128-bit scale entry to the existing panel register.
    logic [127:0] scale_cache_entry0 [0:1];
    logic [127:0] scale_cache_entry1 [0:1];
    logic [127:0] scale_cache_entry2 [0:1];
    logic [127:0] scale_cache_entry3 [0:1];
    logic [127:0] scale_cache_entry4 [0:1];
    logic [127:0] scale_cache_entry5 [0:1];
    logic [127:0] scale_cache_entry6 [0:1];
    logic [127:0] scale_cache_entry7 [0:1];
    logic [63:0] scale_cache_line_addr [0:1];
    logic [1:0] scale_cache_valid;
    logic scale_cache_replace;
    logic [1:0] request_scale_cache_hit;
    logic request_scale_cache_slot;
    logic active_scale_cached;
    logic active_scale_cache_slot;
    logic [63:0] active_scale_line_addr;
    logic active_scale_line_eligible;
    logic active_scale_cache_hit;
    logic [127:0] active_cached_scale;

    logic selected_empty_panel;
    logic any_empty_panel;
    logic deferred_load_available;
    logic selected_load_panel;
    logic selected_load_deferred;
    logic [64:0] requested_stripe_bytes;
    logic [64:0] requested_base_end;
    logic [64:0] requested_enhancement_end;
    logic [64:0] requested_scale_end;
    logic request_base_range_ok;
    logic request_enhancement_range_ok;
    logic request_scale_range_ok;

    logic [5:0] expected_response_bytes;
    logic [31:0] expected_response_byte_enable;
    logic response_has_second_word;
    logic response_format_ok;
    logic response_accepted;
    logic response_bad;
    logic final_payload_accepted;
    logic completed_payload_available;
    logic [12:0] requested_plane_words;
    logic [12:0] requested_plane_rows;

    integer panel_index;

    assign any_empty_panel = (!panel_valid[0] && !panel_loading[0] && !panel_computing[0]) ||
                             (!load_req_block_panel1 && !panel_valid[1] &&
                              !panel_loading[1] && !panel_computing[1]);
    assign selected_empty_panel =
        load_req_block_panel1 ? 1'b0 :
        ((!panel_valid[next_panel] && !panel_loading[next_panel] && !panel_computing[next_panel]) ?
         next_panel : ~next_panel);
    assign deferred_load_available = panel_valid == 2'b11 &&
        panel_loading == 2'b00 &&
        (panel_computing == 2'b01 || panel_computing == 2'b10) &&
        load_req_allow_deferred && !load_req_block_panel1 &&
        !compute_release_valid;
    assign selected_load_deferred = !any_empty_panel && deferred_load_available;
    assign selected_load_panel = selected_load_deferred ?
        panel_computing[1] : selected_empty_panel;

    assign requested_stripe_bytes = {31'd0, load_req_input_features, 2'b00};
    assign requested_base_end = {1'b0, load_req_base_addr} + requested_stripe_bytes;
    assign requested_enhancement_end = {1'b0, load_req_enhancement_addr} + requested_stripe_bytes;
    assign requested_scale_end = {1'b0, load_req_scale_addr} + 65'd16;
    assign request_base_range_ok = load_req_input_features != 0 &&
        load_req_input_features <= 32'd12288 &&
        load_req_input_features[1:0] == 0 && load_req_base_addr[3:0] == 0 &&
        !requested_stripe_bytes[64] && !requested_base_end[64] &&
        (!load_req_qkv_layout ||
         (load_req_operator_has_w8 && load_req_input_features == 32'd4096));
    assign request_enhancement_range_ok = !load_req_operator_has_w8 ||
        (load_req_enhancement_addr[3:0] == 0 &&
         !requested_stripe_bytes[64] && !requested_enhancement_end[64]);
    assign request_scale_range_ok = load_req_scale_addr[3:0] == 0 &&
        !requested_scale_end[64];
    assign requested_plane_words = requested_stripe_bytes[16:4];
    assign requested_plane_rows = load_req_qkv_layout ?
        (requested_plane_words + 13'd1) >> 1 :
        load_req_operator_has_w8 &&
        load_req_input_features == 32'd12288 ?
        13'd512 :
        (requested_plane_words + 13'd3) >> 2;

    assign load_req_ready = !abort_request && !abort_pending && !active_load &&
                            !load_done_valid &&
                            (any_empty_panel || deferred_load_available);

    assign active_scale_line_addr = {active_scale_addr[63:7], 7'd0};
    assign active_scale_line_eligible = !active_has_w8 &&
        active_scale_addr[63:7] != {57{1'b1}};
    assign active_scale_cache_hit = active_plane == 2'd2 &&
        active_scale_cached;
    assign request_scale_cache_hit[0] = !load_req_operator_has_w8 &&
        load_req_scale_addr[63:7] != {57{1'b1}} && scale_cache_valid[0] &&
        scale_cache_line_addr[0] == {load_req_scale_addr[63:7], 7'd0};
    assign request_scale_cache_hit[1] = !load_req_operator_has_w8 &&
        load_req_scale_addr[63:7] != {57{1'b1}} && scale_cache_valid[1] &&
        scale_cache_line_addr[1] == {load_req_scale_addr[63:7], 7'd0};
    assign request_scale_cache_slot = |request_scale_cache_hit ?
        request_scale_cache_hit[1] :
        (&scale_cache_valid ? scale_cache_replace : scale_cache_valid[0]);

    always_comb begin
        case (active_scale_addr[6:4])
            3'd0: active_cached_scale = scale_cache_entry0[active_scale_cache_slot];
            3'd1: active_cached_scale = scale_cache_entry1[active_scale_cache_slot];
            3'd2: active_cached_scale = scale_cache_entry2[active_scale_cache_slot];
            3'd3: active_cached_scale = scale_cache_entry3[active_scale_cache_slot];
            3'd4: active_cached_scale = scale_cache_entry4[active_scale_cache_slot];
            3'd5: active_cached_scale = scale_cache_entry5[active_scale_cache_slot];
            3'd6: active_cached_scale = scale_cache_entry6[active_scale_cache_slot];
            default: active_cached_scale = scale_cache_entry7[active_scale_cache_slot];
        endcase
    end

    // Submit one logical request per plane. memory_controller owns 256-byte
    // burst generation, 4 KiB splitting, and the eight-burst AXI read window.
    assign dma_req_valid = active_load && !dma_request_outstanding &&
                           !load_failed && !abort_pending && !abort_request &&
                           !active_scale_cache_hit;
    assign dma_req_addr = active_plane == 2'd0 ? active_base_addr :
                          active_plane == 2'd1 ? active_enhancement_addr :
                          active_scale_line_eligible ? active_scale_line_addr :
                          active_scale_addr;
    assign dma_req_bytes = active_plane == 2'd2 ?
                           (active_scale_line_eligible ? 32'd128 : 32'd16) :
                           stripe_bytes[31:0];
    assign dma_req_plane = active_plane;
    assign dma_req_panel = active_panel;
    assign dma_req_last_for_plane = 1'b1;

    always_comb begin
        if (response_remaining >= 32)
            expected_response_bytes = 6'd32;
        else
            expected_response_bytes = {1'b0, response_remaining[4:0]};
        if (expected_response_bytes == 32)
            expected_response_byte_enable = 32'hffff_ffff;
        else
            expected_response_byte_enable =
                32'hffff_ffff >> (32 - expected_response_bytes);
    end

    assign response_has_second_word = |expected_response_byte_enable[31:16];

    assign response_format_ok = dma_rsp_byte_enable == expected_response_byte_enable &&
                                dma_rsp_last == (response_remaining <= 32);
    assign response_bad = dma_rsp_error || !response_format_ok;
    assign panel_write_valid = dma_rsp_valid && dma_request_outstanding &&
                               active_plane != 2'd2 &&
                               !active_wait_for_panel &&
                               !abort_pending &&
                               !abort_request && !load_failed && !response_bad;
    assign panel_write_panel = active_panel;
    assign panel_write_plane = active_plane;
    assign panel_write_start_bank = response_panel_bank;
    assign panel_write_word_row = response_panel_row;
    assign panel_write_data = dma_rsp_data;
    assign panel_write_byte_enable = dma_rsp_byte_enable;
    assign dma_rsp_ready = dma_request_outstanding &&
        ((abort_pending || abort_request || load_failed ||
          (dma_rsp_valid && response_bad) || active_plane == 2'd2) ?
         1'b1 : (!active_wait_for_panel && panel_write_ready));
    assign response_accepted = dma_rsp_valid && dma_rsp_ready;
    assign final_payload_accepted = response_accepted && !response_bad &&
                                    response_remaining <= 32;
    assign completed_payload_available = dma_payload_complete ||
                                         final_payload_accepted;

    always_comb begin
        compute_acquire_valid = 1'b0;
        compute_acquire_panel = 1'b0;
        compute_acquire_operator_id = '0;
        compute_acquire_stripe_id = '0;
        compute_acquire_has_enhancement = 1'b0;
        if (!abort_pending && !abort_request && panel_computing == 2'b00) begin
            if (panel_valid[compute_next_panel] &&
                panel_operator[compute_next_panel] == compute_request_operator_id &&
                panel_stripe[compute_next_panel] == compute_request_stripe_id) begin
                compute_acquire_valid = 1'b1;
                compute_acquire_panel = compute_next_panel;
                compute_acquire_operator_id = panel_operator[compute_next_panel];
                compute_acquire_stripe_id = panel_stripe[compute_next_panel];
                compute_acquire_has_enhancement = panel_enhancement[compute_next_panel];
            end else if (panel_valid[~compute_next_panel] &&
                         panel_operator[~compute_next_panel] ==
                            compute_request_operator_id &&
                         panel_stripe[~compute_next_panel] ==
                            compute_request_stripe_id) begin
                compute_acquire_valid = 1'b1;
                compute_acquire_panel = ~compute_next_panel;
                compute_acquire_operator_id = panel_operator[~compute_next_panel];
                compute_acquire_stripe_id = panel_stripe[~compute_next_panel];
                compute_acquire_has_enhancement = panel_enhancement[~compute_next_panel];
            end
        end
    end

    assign compute_release_ready = panel_computing[compute_release_panel];
    assign panel_operator_id = {panel_operator[1], panel_operator[0]};
    assign panel_stripe_id = {panel_stripe[1], panel_stripe[0]};
    assign panel_has_enhancement = panel_enhancement;

`ifdef SYNTHESIS
    always_comb begin
        accepted_load_count = '0;
        accepted_dma_request_count = '0;
        accepted_base_request_count = '0;
        accepted_enhancement_request_count = '0;
        accepted_base_request_bytes = '0;
        accepted_enhancement_request_bytes = '0;
        accepted_scale_request_count = '0;
        accepted_scale_request_bytes = '0;
        accepted_dma_response_count = '0;
        accepted_dma_response_bytes = '0;
        discarded_dma_response_bytes = '0;
        four_kib_split_count = '0;
        dma_error_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            abort_ack <= 1'b0;
            load_done_valid <= 1'b0;
            load_done_panel <= 1'b0;
            load_done_error <= LOAD_OK;
            panel_valid <= 2'b00;
            panel_loading <= 2'b00;
            panel_computing <= 2'b00;
            panel_enhancement <= 2'b00;
            panel_operator[0] <= '0;
            panel_operator[1] <= '0;
            panel_stripe[0] <= '0;
            panel_stripe[1] <= '0;
            next_panel <= 1'b0;
            compute_next_panel <= 1'b0;
            active_load <= 1'b0;
            active_panel <= 1'b0;
            active_wait_for_panel <= 1'b0;
            active_operator_id <= '0;
            active_stripe_id <= '0;
            active_has_w8 <= 1'b0;
            active_plane <= 2'd0;
            active_base_addr <= 64'd0;
            active_enhancement_addr <= 64'd0;
            active_scale_addr <= 64'd0;
            stripe_bytes <= 32'd0;
            dma_request_outstanding <= 1'b0;
            dma_payload_complete <= 1'b0;
            response_remaining <= 32'd0;
            active_bank_count <= 3'd4;
            active_plane_rows <= 10'd0;
            response_panel_bank <= 3'd0;
            response_panel_row <= 10'd0;
            panel0_scale_base <= '0;
            panel0_scale_valid <= 1'b0;
            panel1_scale_base <= '0;
            panel1_scale_valid <= 1'b0;
            load_failed <= 1'b0;
            load_error <= LOAD_OK;
            abort_pending <= 1'b0;
            scale_cache_line_addr[0] <= 64'd0;
            scale_cache_line_addr[1] <= 64'd0;
            scale_cache_valid <= 2'b00;
            scale_cache_replace <= 1'b0;
            active_scale_cached <= 1'b0;
            active_scale_cache_slot <= 1'b0;
`ifndef SYNTHESIS
            accepted_load_count <= 64'd0;
            accepted_dma_request_count <= 64'd0;
            accepted_base_request_count <= 64'd0;
            accepted_enhancement_request_count <= 64'd0;
            accepted_base_request_bytes <= 64'd0;
            accepted_enhancement_request_bytes <= 64'd0;
            accepted_scale_request_count <= 64'd0;
            accepted_scale_request_bytes <= 64'd0;
            accepted_dma_response_count <= 64'd0;
            accepted_dma_response_bytes <= 64'd0;
            discarded_dma_response_bytes <= 64'd0;
            four_kib_split_count <= 64'd0;
            dma_error_count <= 64'd0;
`endif
        end else begin
            abort_ack <= 1'b0;

            if (load_done_valid && load_done_ready)
                load_done_valid <= 1'b0;

            if (active_load && active_scale_cache_hit &&
                !abort_pending && !abort_request) begin
                scale_cache_replace <= ~active_scale_cache_slot;
                active_load <= 1'b0;
                panel_loading[active_panel] <= 1'b0;
                panel_valid[active_panel] <= 1'b1;
                load_done_valid <= 1'b1;
                load_done_panel <= active_panel;
                load_done_error <= LOAD_OK;
                next_panel <= ~active_panel;
                if (active_panel) begin
                    panel1_scale_base <= active_cached_scale;
                    panel1_scale_valid <= 1'b1;
                end else begin
                    panel0_scale_base <= active_cached_scale;
                    panel0_scale_valid <= 1'b1;
                end
            end

            if (load_req_valid && load_req_ready) begin
`ifndef SYNTHESIS
                accepted_load_count <= accepted_load_count + 64'd1;
`endif
                active_panel <= selected_load_panel;
                active_wait_for_panel <= selected_load_deferred;
                active_operator_id <= load_req_operator_id;
                active_stripe_id <= load_req_stripe_id;
                active_has_w8 <= load_req_operator_has_w8;
                active_plane <= 2'd0;
                active_base_addr <= load_req_base_addr;
                active_enhancement_addr <= load_req_enhancement_addr;
                active_scale_addr <= load_req_scale_addr;
                active_scale_cached <= |request_scale_cache_hit;
                active_scale_cache_slot <= request_scale_cache_slot;
                stripe_bytes <= requested_stripe_bytes[31:0];
                active_bank_count <= load_req_qkv_layout ? 3'd2 :
                    load_req_operator_has_w8 &&
                    load_req_input_features == 32'd12288 ? 3'd6 : 3'd4;
                active_plane_rows <= requested_plane_rows[9:0];
                response_panel_bank <= 3'd0;
                response_panel_row <= 10'd0;
                load_failed <= 1'b0;
                load_error <= LOAD_OK;
                load_done_panel <= selected_load_panel;
                if (!request_base_range_ok) begin
                    active_wait_for_panel <= 1'b0;
                    load_done_valid <= 1'b1;
                    load_done_error <= LOAD_BASE_RANGE_ERROR;
                end else if (!request_enhancement_range_ok) begin
                    active_wait_for_panel <= 1'b0;
                    load_done_valid <= 1'b1;
                    load_done_error <= LOAD_ENH_RANGE_ERROR;
                end else if (!request_scale_range_ok) begin
                    active_wait_for_panel <= 1'b0;
                    load_done_valid <= 1'b1;
                    load_done_error <= LOAD_SCALE_RANGE_ERROR;
                end else begin
                    active_load <= 1'b1;
                    if (!selected_load_deferred) begin
                        panel_operator[selected_load_panel] <= load_req_operator_id;
                        panel_stripe[selected_load_panel] <= load_req_stripe_id;
                        panel_enhancement[selected_load_panel] <=
                            load_req_operator_has_w8;
                        panel_loading[selected_load_panel] <= 1'b1;
                        if (selected_load_panel)
                            panel1_scale_valid <= 1'b0;
                        else
                            panel0_scale_valid <= 1'b0;
                    end
                end
            end

            if (dma_req_valid && dma_req_ready) begin
`ifndef SYNTHESIS
                accepted_dma_request_count <= accepted_dma_request_count + 64'd1;
`endif
                dma_request_outstanding <= 1'b1;
                dma_payload_complete <= 1'b0;
                response_remaining <= dma_req_bytes;
                if (active_plane == 2'd2 && active_scale_line_eligible) begin
                    scale_cache_line_addr[active_scale_cache_slot] <= active_scale_line_addr;
                    scale_cache_valid[active_scale_cache_slot] <= 1'b0;
                end
                if (active_plane == 2'd0) begin
`ifndef SYNTHESIS
                    accepted_base_request_count <= accepted_base_request_count + 64'd1;
                    accepted_base_request_bytes <= accepted_base_request_bytes +
                        {32'd0, dma_req_bytes};
`endif
                end else if (active_plane == 2'd1) begin
`ifndef SYNTHESIS
                    accepted_enhancement_request_count <= accepted_enhancement_request_count + 64'd1;
                    accepted_enhancement_request_bytes <=
                        accepted_enhancement_request_bytes + {32'd0, dma_req_bytes};
`endif
                end else begin
`ifndef SYNTHESIS
                    accepted_scale_request_count <= accepted_scale_request_count + 64'd1;
                    accepted_scale_request_bytes <= accepted_scale_request_bytes +
                        {32'd0, dma_req_bytes};
`endif
                end
            end

            if (response_accepted) begin
`ifndef SYNTHESIS
                accepted_dma_response_count <= accepted_dma_response_count + 64'd1;
                accepted_dma_response_bytes <= accepted_dma_response_bytes +
                    {58'd0, expected_response_bytes};
                if (abort_pending || abort_request || load_failed || response_bad)
                    discarded_dma_response_bytes <=
                        discarded_dma_response_bytes +
                        {58'd0, expected_response_bytes};
`endif
                if (response_bad) begin
                    load_failed <= 1'b1;
                    load_error <= dma_rsp_error ? LOAD_DMA_ERROR : LOAD_DMA_PROTOCOL_ERROR;
`ifndef SYNTHESIS
                    if (!load_failed)
                        dma_error_count <= dma_error_count + 64'd1;
`endif
                end
                if (active_plane == 2'd2 && !response_bad && !load_failed &&
                    !abort_pending && !abort_request) begin
                    if (active_scale_line_eligible) begin
                        case (response_remaining)
                            32'd128: begin
                                scale_cache_entry0[active_scale_cache_slot] <= dma_rsp_data[127:0];
                                scale_cache_entry1[active_scale_cache_slot] <= dma_rsp_data[255:128];
                            end
                            32'd96: begin
                                scale_cache_entry2[active_scale_cache_slot] <= dma_rsp_data[127:0];
                                scale_cache_entry3[active_scale_cache_slot] <= dma_rsp_data[255:128];
                            end
                            32'd64: begin
                                scale_cache_entry4[active_scale_cache_slot] <= dma_rsp_data[127:0];
                                scale_cache_entry5[active_scale_cache_slot] <= dma_rsp_data[255:128];
                            end
                            default: begin
                                scale_cache_entry6[active_scale_cache_slot] <= dma_rsp_data[127:0];
                                scale_cache_entry7[active_scale_cache_slot] <= dma_rsp_data[255:128];
                            end
                        endcase
                        case (active_scale_addr[6:4])
                            3'd0: if (response_remaining == 32'd128)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[127:0];
                                else
                                    panel0_scale_base <= dma_rsp_data[127:0];
                            3'd1: if (response_remaining == 32'd128)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[255:128];
                                else
                                    panel0_scale_base <= dma_rsp_data[255:128];
                            3'd2: if (response_remaining == 32'd96)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[127:0];
                                else
                                    panel0_scale_base <= dma_rsp_data[127:0];
                            3'd3: if (response_remaining == 32'd96)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[255:128];
                                else
                                    panel0_scale_base <= dma_rsp_data[255:128];
                            3'd4: if (response_remaining == 32'd64)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[127:0];
                                else
                                    panel0_scale_base <= dma_rsp_data[127:0];
                            3'd5: if (response_remaining == 32'd64)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[255:128];
                                else
                                    panel0_scale_base <= dma_rsp_data[255:128];
                            3'd6: if (response_remaining == 32'd32)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[127:0];
                                else
                                    panel0_scale_base <= dma_rsp_data[127:0];
                            default: if (response_remaining == 32'd32)
                                if (active_panel)
                                    panel1_scale_base <= dma_rsp_data[255:128];
                                else
                                    panel0_scale_base <= dma_rsp_data[255:128];
                        endcase
                    end else if (active_panel) begin
                        panel1_scale_base <= dma_rsp_data[127:0];
                    end else begin
                        panel0_scale_base <= dma_rsp_data[127:0];
                    end
                end
                if (response_panel_bank +
                        (response_has_second_word ? 3'd2 : 3'd1) >=
                    active_bank_count) begin
                    response_panel_bank <= response_panel_bank +
                        (response_has_second_word ? 3'd2 : 3'd1) -
                        active_bank_count;
                    response_panel_row <= response_panel_row + 10'd1;
                end else begin
                    response_panel_bank <= response_panel_bank +
                        (response_has_second_word ? 3'd2 : 3'd1);
                end
                response_remaining <= response_remaining -
                    {26'd0, expected_response_bytes};
                if (response_remaining <= 32)
                    dma_payload_complete <= 1'b1;
            end

            // Request completion, not the last payload beat, advances a plane.
            // This also handles terminal AXI errors with no final data beat.
            if (dma_req_done && dma_request_outstanding) begin
                dma_request_outstanding <= 1'b0;
                dma_payload_complete <= 1'b0;
                active_wait_for_panel <= 1'b0;
                if (abort_pending || abort_request) begin
                    active_load <= 1'b0;
                    panel_loading[active_panel] <= 1'b0;
                    panel_valid[active_panel] <= 1'b0;
                    panel_enhancement[active_panel] <= 1'b0;
                    if (active_panel)
                        panel1_scale_valid <= 1'b0;
                    else
                        panel0_scale_valid <= 1'b0;
                end else if (dma_req_error || load_failed ||
                             (response_accepted && response_bad) ||
                             !completed_payload_available) begin
                    active_load <= 1'b0;
                    panel_loading[active_panel] <= 1'b0;
                    panel_valid[active_panel] <= 1'b0;
                    panel_enhancement[active_panel] <= 1'b0;
                    if (active_panel)
                        panel1_scale_valid <= 1'b0;
                    else
                        panel0_scale_valid <= 1'b0;
                    load_done_valid <= 1'b1;
                    load_done_panel <= active_panel;
                    load_done_error <= dma_req_error ? LOAD_DMA_ERROR :
                        load_failed ? load_error : LOAD_DMA_PROTOCOL_ERROR;
`ifndef SYNTHESIS
                    if (!load_failed && !(response_accepted && response_bad))
                        dma_error_count <= dma_error_count + 64'd1;
`endif
                end else if (active_plane == 2'd0 && active_has_w8) begin
                    active_plane <= 2'd1;
                    response_panel_bank <= 3'd0;
                    response_panel_row <= active_plane_rows;
                end else if (active_plane != 2'd2) begin
                    active_plane <= 2'd2;
                    response_panel_bank <= 3'd0;
                    response_panel_row <= 10'd0;
                end else begin
                    active_load <= 1'b0;
                    panel_loading[active_panel] <= 1'b0;
                    panel_valid[active_panel] <= 1'b1;
                    load_done_valid <= 1'b1;
                    load_done_panel <= active_panel;
                    load_done_error <= LOAD_OK;
                    next_panel <= ~active_panel;
                    if (active_panel)
                        panel1_scale_valid <= 1'b1;
                    else
                        panel0_scale_valid <= 1'b1;
                    if (active_scale_line_eligible) begin
                        scale_cache_valid[active_scale_cache_slot] <= 1'b1;
                        scale_cache_replace <= ~active_scale_cache_slot;
                    end
                end
            end

            if (compute_acquire_valid && compute_acquire_ready) begin
                panel_computing[compute_acquire_panel] <= 1'b1;
                compute_next_panel <= ~compute_acquire_panel;
            end

            if (compute_release_valid && compute_release_ready) begin
                panel_computing[compute_release_panel] <= 1'b0;
                panel_valid[compute_release_panel] <= 1'b0;
                panel_enhancement[compute_release_panel] <= 1'b0;
                if (compute_release_panel)
                    panel1_scale_valid <= 1'b0;
                else
                    panel0_scale_valid <= 1'b0;
                if (active_wait_for_panel &&
                    compute_release_panel == active_panel) begin
                    active_wait_for_panel <= 1'b0;
                    panel_loading[active_panel] <= 1'b1;
                    panel_operator[active_panel] <= active_operator_id;
                    panel_stripe[active_panel] <= active_stripe_id;
                    panel_enhancement[active_panel] <= active_has_w8;
                end
            end

            if (abort_request)
                abort_pending <= 1'b1;

            if (abort_request) begin
                scale_cache_valid <= 2'b00;
                scale_cache_replace <= 1'b0;
            end

            if ((abort_pending || abort_request) && active_load &&
                !dma_request_outstanding) begin
                active_load <= 1'b0;
                active_wait_for_panel <= 1'b0;
                if (!active_wait_for_panel) begin
                    panel_loading[active_panel] <= 1'b0;
                    panel_enhancement[active_panel] <= 1'b0;
                end
            end

            if (abort_pending || abort_request) begin
                load_done_valid <= 1'b0;
                for (panel_index = 0; panel_index < 2; panel_index = panel_index + 1) begin
                    if (panel_valid[panel_index] && !panel_computing[panel_index]) begin
                        panel_valid[panel_index] <= 1'b0;
                        panel_enhancement[panel_index] <= 1'b0;
                    end
                end
            end

            if (dma_abort_ack && dma_request_outstanding) begin
                dma_request_outstanding <= 1'b0;
                dma_payload_complete <= 1'b0;
                active_load <= 1'b0;
                active_wait_for_panel <= 1'b0;
                if (!active_wait_for_panel) begin
                    panel_loading[active_panel] <= 1'b0;
                    panel_enhancement[active_panel] <= 1'b0;
                end
            end

            if (abort_pending && !active_load && !dma_request_outstanding &&
                panel_computing == 2'b00) begin
                abort_pending <= 1'b0;
                abort_ack <= 1'b1;
                panel_valid <= 2'b00;
                panel_loading <= 2'b00;
                panel_enhancement <= 2'b00;
                panel0_scale_valid <= 1'b0;
                panel1_scale_valid <= 1'b0;
                next_panel <= 1'b0;
                compute_next_panel <= 1'b0;
                active_wait_for_panel <= 1'b0;
            end
        end
    end

`ifndef SYNTHESIS
    logic held_dma_request;
    logic [63:0] held_dma_addr;
    logic [31:0] held_dma_bytes;
    logic [1:0] held_dma_plane;
    logic held_dma_panel;
    logic held_dma_last;

    always_ff @(posedge clk) begin
        if (rst) begin
            held_dma_request <= 1'b0;
            held_dma_addr <= 64'd0;
            held_dma_bytes <= 32'd0;
            held_dma_plane <= 2'd0;
            held_dma_panel <= 1'b0;
            held_dma_last <= 1'b0;
        end else begin
            if (dma_req_valid && !dma_req_ready && !abort_request) begin
                if (held_dma_request) begin
                    assert (dma_req_addr == held_dma_addr && dma_req_bytes == held_dma_bytes &&
                            dma_req_plane == held_dma_plane && dma_req_panel == held_dma_panel &&
                            dma_req_last_for_plane == held_dma_last)
                        else $error("matmul_weight_panel_loader changed a stalled DMA request");
                end
                held_dma_request <= 1'b1;
                held_dma_addr <= dma_req_addr;
                held_dma_bytes <= dma_req_bytes;
                held_dma_plane <= dma_req_plane;
                held_dma_panel <= dma_req_panel;
                held_dma_last <= dma_req_last_for_plane;
            end else begin
                held_dma_request <= 1'b0;
            end

            assert ((panel_valid & panel_loading) == 2'b00)
                else $error("matmul_weight_panel_loader panel cannot be valid and loading");
            assert ((panel_loading & panel_computing) == 2'b00)
                else $error("matmul_weight_panel_loader cannot load a computing panel");
            if (active_wait_for_panel)
                assert (active_load && panel_loading[active_panel] == 1'b0 &&
                        panel_computing[active_panel])
                    else $error("matmul_weight_panel_loader deferred load lost its computing target");
            if (compute_acquire_valid) begin
                assert (compute_acquire_operator_id == compute_request_operator_id &&
                        compute_acquire_stripe_id == compute_request_stripe_id)
                    else $error("matmul_weight_panel_loader offered a panel for the wrong compute request");
            end
            if (dma_req_valid) begin
                assert (dma_req_bytes != 0 && dma_req_bytes <= 32'd49152)
                    else $error("matmul_weight_panel_loader emitted an illegal DMA byte count");
            end
            if (panel_write_valid)
                assert (panel_write_start_bank < active_bank_count &&
                        panel_write_word_row <= 10'd1023)
                    else $error("matmul_weight_panel_loader emitted an out-of-range panel write");
            if (dma_rsp_valid && !dma_request_outstanding)
                $error("matmul_weight_panel_loader received a response without an accepted request");
            if (dma_payload_complete && dma_rsp_valid)
                $error("matmul_weight_panel_loader received data after the logical payload completed");
            if (dma_req_done && !dma_request_outstanding)
                $error("matmul_weight_panel_loader received completion without an accepted request");
            if (dma_req_error && !dma_req_done)
                $error("matmul_weight_panel_loader received request error without completion");
            if (compute_release_valid && !compute_release_ready)
                $error("matmul_weight_panel_loader release targeted a non-computing panel");
        end
    end
`endif
endmodule

`default_nettype wire
