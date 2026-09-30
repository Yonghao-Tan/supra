`default_nettype none

module kv_cache_access_controller #(
    parameter integer MAX_CURRENT_ROWS = 48,
    parameter integer MAX_SEQUENCE = 2048,
    parameter integer LOGICAL_WIDTH = 11,
    parameter integer ADDR_WIDTH = 64,
    parameter bit TRUSTED_START_CONFIGURATION = 1'b0
) (
    input  logic clk,
    input  logic rst,
    input  logic abort_request,
    output logic abort_ack,

    input  logic start_valid,
    output logic start_ready,
    input  logic [5:0] start_current_rows,
    input  logic [5:0] start_token_batch_index,
    input  logic [47:0] start_kv_write_disable,
    input  logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] start_token_position,
    input  logic start_pair_enable,
    input  logic [5:0] start_second_rows,
    input  logic [47:0] start_second_kv_write_disable,
    input  logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] start_second_token_position,
    input  hardware_types_pkg::attention_batch_group_config_t start_batch_group,
    input  logic [ADDR_WIDTH-1:0] start_current_k_base,
    input  logic [ADDR_WIDTH-1:0] start_current_v_base,
    input  logic [ADDR_WIDTH-1:0] start_current_k_scale_base,
    input  logic [ADDR_WIDTH-1:0] start_retained_k_base,
    input  logic [ADDR_WIDTH-1:0] start_retained_v_base,
    input  logic [ADDR_WIDTH-1:0] start_retained_k_scale_base,
    input  logic [ADDR_WIDTH-1:0] start_head_stride,
    input  logic [ADDR_WIDTH-1:0] start_token_stride,
    input  logic [ADDR_WIDTH-1:0] start_k_scale_head_stride,

    output logic config_valid,
    input  logic config_release,
    output logic config_done_pulse,
    output logic error,
    output logic [3:0] error_id,

    input  logic current_write_valid,
    output logic current_write_ready,
    input  logic [5:0] current_write_head,
    input  logic [5:0] current_write_physical_row,
    input  logic current_write_second_batch,
    input  logic [2:0] current_write_batch_index,
    input  logic [LOGICAL_WIDTH-1:0] current_write_logical_slot,
    input  logic [4:0] current_write_chunk,
    output logic current_write_accepted,
    output logic [ADDR_WIDTH-1:0] current_k_write_address,
    output logic [ADDR_WIDTH-1:0] current_v_write_address,
    output logic [ADDR_WIDTH-1:0] current_k_scale_write_address,
    output logic [5:0] current_write_physical_tag,
    output logic [LOGICAL_WIDTH-1:0] current_write_logical_tag,

    input  logic read_req_valid,
    output logic read_req_ready,
    input  logic [5:0] read_req_head,
    input  logic [LOGICAL_WIDTH-1:0] read_req_key_group,
    input  logic [4:0] read_req_chunk,
    input  logic [15:0] read_req_tag,
    output logic read_rsp_valid,
    input  logic read_rsp_ready,
    output logic [LOGICAL_WIDTH-1:0] read_rsp_key_group,
    output logic [7:0] read_rsp_current_mask,
    output logic [7:0] read_rsp_retained_mask,
    output logic [ADDR_WIDTH-1:0] read_rsp_current_k_address,
    output logic [ADDR_WIDTH-1:0] read_rsp_current_v_address,
    output logic [ADDR_WIDTH-1:0] read_rsp_current_k_scale_address,
    output logic [ADDR_WIDTH-1:0] read_rsp_retained_k_address,
    output logic [ADDR_WIDTH-1:0] read_rsp_retained_v_address,
    output logic [ADDR_WIDTH-1:0] read_rsp_retained_k_scale_address,
    output logic [15:0] read_rsp_tag,

    output logic [63:0] accepted_current_write_count,
    output logic [63:0] accepted_read_request_count,
    output logic [63:0] completed_read_response_count,
    output logic [63:0] retained_write_count
);
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_CURRENT_WRITE = 4'h2;
    localparam logic [3:0] ERROR_READ_REQUEST = 4'h3;

    logic [5:0] current_rows;
    logic [47:0] kv_write_disable;
    logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] token_position;
    logic pair_enable, config_second_batch;
    logic [5:0] second_rows;
    logic [47:0] second_kv_write_disable;
    logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] second_token_position;
    hardware_types_pkg::attention_batch_group_config_t batch_group;
    hardware_types_pkg::attention_batch_group_config_t visibility_batch_group;
    logic batch_group_enable;
    logic [2:0] config_batch_index, config_batch_count;
    logic [5:0] active_config_rows, next_config_rows;
    logic [47:0] active_config_disable, next_config_disable;
    logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0]
        active_config_position, next_config_position;
    logic [5:0] write_config_rows;
    logic [47:0] write_config_disable;
    logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] write_config_position;
    logic config_batch_last;
    logic [7:0] visibility_words [0:15][0:15];
    logic [7:0] visibility_bank_word [0:15];
    logic [LOGICAL_WIDTH-1:0] config_position [0:1];
    logic [1:0] config_position_valid;
    logic start_configuration_error;
    logic [ADDR_WIDTH-1:0] current_k_base;
    logic [ADDR_WIDTH-1:0] current_v_base;
    logic [ADDR_WIDTH-1:0] current_k_scale_base;
    logic [ADDR_WIDTH-1:0] retained_k_base;
    logic [ADDR_WIDTH-1:0] retained_v_base;
    logic [ADDR_WIDTH-1:0] retained_k_scale_base;
    logic [ADDR_WIDTH-1:0] head_stride;
    logic [ADDR_WIDTH-1:0] k_scale_head_stride;
    logic config_building;
    logic [4:0] config_pair;
    logic response_pending;
    logic [5:0] response_head;
    logic [LOGICAL_WIDTH-1:0] response_key_group;
    logic [4:0] response_chunk;
    logic [15:0] response_tag;
    logic [7:0] response_current_mask;
    logic [LOGICAL_WIDTH-1:0] current_write_expected_slot;
    logic [7:0] requested_current_mask;
    logic [ADDR_WIDTH-1:0] current_write_head_offset;
    logic [ADDR_WIDTH-1:0] current_write_k_scale_head_offset;
    logic [ADDR_WIDTH-1:0] response_head_offset;
    logic [ADDR_WIDTH-1:0] response_k_scale_head_offset;

    function automatic logic [LOGICAL_WIDTH-1:0] select_token_position(
        input logic [MAX_CURRENT_ROWS*LOGICAL_WIDTH-1:0] positions,
        input logic [5:0] physical_row
    );
        if (physical_row < 6'(MAX_CURRENT_ROWS))
            select_token_position = positions[
                integer'(physical_row)*LOGICAL_WIDTH +: LOGICAL_WIDTH];
        else
            select_token_position = '0;
    endfunction

    assign visibility_batch_group = TRUSTED_START_CONFIGURATION ?
        start_batch_group : batch_group;

    always_comb begin
        active_config_rows = current_rows;
        active_config_disable = kv_write_disable;
        active_config_position = token_position;
        next_config_rows = second_rows;
        next_config_disable = second_kv_write_disable;
        next_config_position = second_token_position;
        write_config_rows = current_write_second_batch ? second_rows : current_rows;
        write_config_disable = current_write_second_batch ?
            second_kv_write_disable : kv_write_disable;
        write_config_position = current_write_second_batch ?
            second_token_position : token_position;
        if (config_second_batch) begin
            active_config_rows = second_rows;
            active_config_disable = second_kv_write_disable;
            active_config_position = second_token_position;
        end
        if (batch_group_enable) begin
            case (config_batch_index)
                3'd0: begin active_config_rows = visibility_batch_group.row_count[0 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[0 +: 48]; active_config_position = visibility_batch_group.token_position[0 +: 528]; end
                3'd1: begin active_config_rows = visibility_batch_group.row_count[6 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[48 +: 48]; active_config_position = visibility_batch_group.token_position[528 +: 528]; end
                3'd2: begin active_config_rows = visibility_batch_group.row_count[12 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[96 +: 48]; active_config_position = visibility_batch_group.token_position[1056 +: 528]; end
                3'd3: begin active_config_rows = visibility_batch_group.row_count[18 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[144 +: 48]; active_config_position = visibility_batch_group.token_position[1584 +: 528]; end
                3'd4: begin active_config_rows = visibility_batch_group.row_count[24 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[192 +: 48]; active_config_position = visibility_batch_group.token_position[2112 +: 528]; end
                default: begin active_config_rows = visibility_batch_group.row_count[30 +: 6]; active_config_disable = visibility_batch_group.kv_write_disable[240 +: 48]; active_config_position = visibility_batch_group.token_position[2640 +: 528]; end
            endcase
            case (config_batch_index + 3'd1)
                3'd1: begin next_config_rows = visibility_batch_group.row_count[6 +: 6]; next_config_disable = visibility_batch_group.kv_write_disable[48 +: 48]; next_config_position = visibility_batch_group.token_position[528 +: 528]; end
                3'd2: begin next_config_rows = visibility_batch_group.row_count[12 +: 6]; next_config_disable = visibility_batch_group.kv_write_disable[96 +: 48]; next_config_position = visibility_batch_group.token_position[1056 +: 528]; end
                3'd3: begin next_config_rows = visibility_batch_group.row_count[18 +: 6]; next_config_disable = visibility_batch_group.kv_write_disable[144 +: 48]; next_config_position = visibility_batch_group.token_position[1584 +: 528]; end
                3'd4: begin next_config_rows = visibility_batch_group.row_count[24 +: 6]; next_config_disable = visibility_batch_group.kv_write_disable[192 +: 48]; next_config_position = visibility_batch_group.token_position[2112 +: 528]; end
                default: begin next_config_rows = visibility_batch_group.row_count[30 +: 6]; next_config_disable = visibility_batch_group.kv_write_disable[240 +: 48]; next_config_position = visibility_batch_group.token_position[2640 +: 528]; end
            endcase
            case (current_write_batch_index)
                3'd0: begin write_config_rows = visibility_batch_group.row_count[0 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[0 +: 48]; write_config_position = visibility_batch_group.token_position[0 +: 528]; end
                3'd1: begin write_config_rows = visibility_batch_group.row_count[6 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[48 +: 48]; write_config_position = visibility_batch_group.token_position[528 +: 528]; end
                3'd2: begin write_config_rows = visibility_batch_group.row_count[12 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[96 +: 48]; write_config_position = visibility_batch_group.token_position[1056 +: 528]; end
                3'd3: begin write_config_rows = visibility_batch_group.row_count[18 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[144 +: 48]; write_config_position = visibility_batch_group.token_position[1584 +: 528]; end
                3'd4: begin write_config_rows = visibility_batch_group.row_count[24 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[192 +: 48]; write_config_position = visibility_batch_group.token_position[2112 +: 528]; end
                default: begin write_config_rows = visibility_batch_group.row_count[30 +: 6]; write_config_disable = visibility_batch_group.kv_write_disable[240 +: 48]; write_config_position = visibility_batch_group.token_position[2640 +: 528]; end
            endcase
        end
    end
    assign current_write_expected_slot =
        select_token_position(write_config_position, current_write_physical_row);
    assign config_batch_last = {1'b0, config_pair, 1'b0} + 7'd2 >=
        {1'b0, active_config_rows};
    assign start_ready = !config_valid && !config_building && !abort_request;
    assign start_configuration_error = start_current_rows == 6'd0 ||
        start_current_rows > 6'(MAX_CURRENT_ROWS) ||
        (start_pair_enable && !start_batch_group.enable &&
         (start_second_rows == 0 || start_second_rows > 6'(MAX_CURRENT_ROWS))) ||
        (start_batch_group.enable &&
         (start_batch_group.batch_count < 3'd2 ||
          start_batch_group.batch_count > 3'd6)) ||
        start_token_stride != ADDR_WIDTH'(128) ||
        start_head_stride < (start_token_stride << 11) ||
        start_k_scale_head_stride < ADDR_WIDTH'(4096);
    for (genvar slot = 0; slot < 2; slot++) begin : g_config_positions
        logic [6:0] next_config_position_index;
        assign next_config_position_index = {1'b0, config_pair, 1'b0} + 7'(slot + 2);
        always_ff @(posedge clk) begin
            if (rst) begin
                config_position[slot] <= '0;
                config_position_valid[slot] <= 1'b0;
            end else if (start_valid && start_ready && !start_configuration_error) begin
                config_position[slot] <= start_token_position[slot*LOGICAL_WIDTH +: LOGICAL_WIDTH];
                config_position_valid[slot] <= 6'(slot) < start_current_rows &&
                    !start_kv_write_disable[slot];
            end else if (config_building) begin
                if (config_batch_last && config_batch_index + 3'd1 <
                    config_batch_count) begin
                    config_position[slot] <= next_config_position[
                        slot*LOGICAL_WIDTH +: LOGICAL_WIDTH];
                    config_position_valid[slot] <= 6'(slot) < next_config_rows &&
                        !next_config_disable[slot];
                end else begin
                    config_position[slot] <= select_token_position(
                        active_config_position, next_config_position_index[5:0]);
                    config_position_valid[slot] <= next_config_position_index <
                        {1'b0, active_config_rows} &&
                        !active_config_disable[next_config_position_index[5:0]];
                end
            end
        end
    end
    assign current_write_head_offset =
        ADDR_WIDTH'(current_write_head[4:0]) * head_stride;
    assign current_write_k_scale_head_offset =
        ADDR_WIDTH'(current_write_head[4:0]) * k_scale_head_stride;
    assign response_head_offset =
        ADDR_WIDTH'(response_head[4:0]) * head_stride;
    assign response_k_scale_head_offset =
        ADDR_WIDTH'(response_head[4:0]) * k_scale_head_stride;

    wire current_write_request = current_write_valid && current_write_ready;
    wire current_write_legal = current_write_head < 6'd32 &&
        current_write_batch_index < config_batch_count &&
        current_write_chunk < 5'd16 &&
        (TRUSTED_START_CONFIGURATION ?
            current_write_physical_row < 6'(MAX_CURRENT_ROWS) :
            (current_write_physical_row < write_config_rows &&
             !write_config_disable[current_write_physical_row] &&
             current_write_logical_slot == current_write_expected_slot));

    assign current_write_ready = config_valid && !config_release && !abort_request;
    assign current_write_accepted = current_write_request && current_write_legal;
    assign current_k_write_address = current_k_base +
        current_write_head_offset +
        (ADDR_WIDTH'(current_write_logical_slot) << 7) +
        ADDR_WIDTH'(current_write_chunk * 8);
    assign current_v_write_address = current_v_base +
        current_write_head_offset +
        ADDR_WIDTH'(current_write_chunk) * ADDR_WIDTH'(MAX_SEQUENCE * 8) +
        ADDR_WIDTH'(current_write_logical_slot) * ADDR_WIDTH'(8);
    assign current_k_scale_write_address = current_k_scale_base +
        current_write_k_scale_head_offset +
        ADDR_WIDTH'(current_write_logical_slot) * ADDR_WIDTH'(2);
    assign current_write_physical_tag = current_write_physical_row;
    assign current_write_logical_tag = current_write_logical_slot;

    wire read_request_legal = read_req_head < 6'd32 && read_req_chunk < 5'd16 &&
        read_req_key_group[2:0] == 3'd0 && read_req_key_group <= 11'd2040;
    assign read_req_ready = config_valid && !config_release &&
        (!response_pending || read_rsp_ready) && !abort_request &&
        read_request_legal;
    assign read_rsp_valid = response_pending;
    assign read_rsp_key_group = response_key_group;
    assign read_rsp_current_mask = response_current_mask;
    assign read_rsp_retained_mask = ~response_current_mask;
    assign read_rsp_current_k_address = current_k_base +
        response_head_offset + (ADDR_WIDTH'(response_key_group) << 7) +
        ADDR_WIDTH'(response_chunk * 8);
    assign read_rsp_current_v_address = current_v_base +
        response_head_offset +
        ADDR_WIDTH'(response_chunk) * ADDR_WIDTH'(MAX_SEQUENCE * 8) +
        ADDR_WIDTH'(response_key_group) * ADDR_WIDTH'(8);
    assign read_rsp_current_k_scale_address = current_k_scale_base +
        response_k_scale_head_offset +
        ADDR_WIDTH'(response_key_group) * ADDR_WIDTH'(2);
    assign read_rsp_retained_k_address = retained_k_base +
        response_head_offset + (ADDR_WIDTH'(response_key_group) << 7) +
        ADDR_WIDTH'(response_chunk * 8);
    assign read_rsp_retained_v_address = retained_v_base +
        response_head_offset +
        ADDR_WIDTH'(response_chunk) * ADDR_WIDTH'(16384) +
        ADDR_WIDTH'(response_key_group) * ADDR_WIDTH'(8);
    assign read_rsp_retained_k_scale_address = retained_k_scale_base +
        response_k_scale_head_offset +
        ADDR_WIDTH'(response_key_group) * ADDR_WIDTH'(2);
    assign read_rsp_tag = response_tag;

    // Two positions are inserted per cycle; both can share one bitmap word.
    // A 256-byte bitmap records membership across rounds.
    for (genvar bank = 0; bank < 16; bank++) begin : g_visibility_bank
        assign visibility_bank_word[bank] = visibility_words[bank][read_req_key_group[6:3]];
        for (genvar word = 0; word < 16; word++) begin : g_word
            logic [7:0] set_mask;
            always_comb begin
                set_mask = 8'd0;
                for (integer slot = 0; slot < 2; slot++) begin
                    if (config_position_valid[slot] &&
                        config_position[slot][10:7] == 4'(bank) &&
                        config_position[slot][6:3] == 4'(word))
                        set_mask = set_mask | (8'd1 << config_position[slot][2:0]);
                end
            end
            always_ff @(posedge clk) begin
                if (rst || (start_valid && start_ready && !start_configuration_error &&
                            start_token_batch_index == 6'd0))
                    visibility_words[bank][word] <= 8'd0;
                else if (config_building && !abort_request && |set_mask)
                    visibility_words[bank][word] <= visibility_words[bank][word] | set_mask;
            end
        end
    end
    assign requested_current_mask = visibility_bank_word[read_req_key_group[10:7]];

`ifdef SYNTHESIS
    always_comb begin
        accepted_current_write_count = '0;
        accepted_read_request_count = '0;
        completed_read_response_count = '0;
        retained_write_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            current_rows <= '0;
            kv_write_disable <= '0;
            token_position <= '0;
            pair_enable <= 1'b0;
            batch_group <= '0;
            batch_group_enable <= 1'b0;
            config_batch_index <= 3'd0;
            config_batch_count <= 3'd1;
            config_second_batch <= 1'b0;
            second_rows <= '0;
            second_kv_write_disable <= '0;
            second_token_position <= '0;
            current_k_base <= '0;
            current_v_base <= '0;
            current_k_scale_base <= '0;
            retained_k_base <= '0;
            retained_v_base <= '0;
            retained_k_scale_base <= '0;
            head_stride <= '0;
            k_scale_head_stride <= '0;
            config_building <= 1'b0;
            config_pair <= '0;
            config_valid <= 1'b0;
            config_done_pulse <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
            abort_ack <= 1'b0;
            response_pending <= 1'b0;
            response_head <= '0;
            response_key_group <= '0;
            response_chunk <= '0;
            response_tag <= '0;
            response_current_mask <= '0;
`ifndef SYNTHESIS
            accepted_current_write_count <= '0;
            accepted_read_request_count <= '0;
            completed_read_response_count <= '0;
            retained_write_count <= '0;
`endif
        end else begin
            config_done_pulse <= 1'b0;
            abort_ack <= 1'b0;

            if (start_valid && start_ready) begin
                error <= 1'b0;
                error_id <= '0;
                if (start_configuration_error) begin
                    error <= 1'b1;
                    error_id <= ERROR_CONFIGURATION;
                    config_done_pulse <= 1'b1;
                end else begin
                    current_rows <= start_current_rows;
                    kv_write_disable <= start_kv_write_disable;
                    token_position <= start_token_position;
                    pair_enable <= start_pair_enable || start_batch_group.enable;
                    if (!TRUSTED_START_CONFIGURATION)
                        batch_group <= start_batch_group;
                    batch_group_enable <= start_batch_group.enable;
                    config_batch_index <= 3'd0;
                    config_batch_count <= start_batch_group.enable ?
                        start_batch_group.batch_count :
                        (start_pair_enable ? 3'd2 : 3'd1);
                    config_second_batch <= 1'b0;
                    second_rows <= start_second_rows;
                    second_kv_write_disable <= start_second_kv_write_disable;
                    second_token_position <= start_second_token_position;
                    current_k_base <= start_current_k_base;
                    current_v_base <= start_current_v_base;
                    current_k_scale_base <= start_current_k_scale_base;
                    retained_k_base <= start_retained_k_base;
                    retained_v_base <= start_retained_v_base;
                    retained_k_scale_base <= start_retained_k_scale_base;
                    head_stride <= start_head_stride;
                    k_scale_head_stride <= start_k_scale_head_stride;
                    config_building <= 1'b1;
                    config_pair <= 5'd0;
                end
            end

            if (config_building) begin
                if (config_batch_last) begin
                    if (config_batch_index + 3'd1 < config_batch_count) begin
                        config_batch_index <= config_batch_index + 3'd1;
                        config_second_batch <= 1'b1;
                        config_pair <= 5'd0;
                    end else begin
                        config_building <= 1'b0;
                        config_valid <= 1'b1;
                        config_done_pulse <= 1'b1;
                    end
                end else begin
                    config_pair <= config_pair + 1'b1;
                end
            end

`ifndef SYNTHESIS
            if (current_write_accepted)
                accepted_current_write_count <= accepted_current_write_count + 1'b1;
`endif

            if (current_write_request && !current_write_legal) begin
                error <= 1'b1;
                if (error_id == 0)
                    error_id <= ERROR_CURRENT_WRITE;
            end

            if (read_req_valid && read_req_ready) begin
                response_pending <= 1'b1;
                response_head <= read_req_head;
                response_key_group <= read_req_key_group;
                response_chunk <= read_req_chunk;
                response_tag <= read_req_tag;
                response_current_mask <= requested_current_mask;
`ifndef SYNTHESIS
                accepted_read_request_count <= accepted_read_request_count + 1'b1;
`endif
            end
            if (read_req_valid && config_valid && !config_release &&
                (!response_pending || read_rsp_ready) && !abort_request &&
                !read_request_legal) begin
                error <= 1'b1;
                if (error_id == 0)
                    error_id <= ERROR_READ_REQUEST;
            end

            if (read_rsp_valid && read_rsp_ready &&
                !(read_req_valid && read_req_ready)) begin
                response_pending <= 1'b0;
            end
`ifndef SYNTHESIS
            if (read_rsp_valid && read_rsp_ready)
                completed_read_response_count <= completed_read_response_count + 1'b1;
`endif

            if (config_valid && config_release) begin
                config_valid <= 1'b0;
                response_pending <= 1'b0;
            end

            if (abort_request) begin
                config_valid <= 1'b0;
                config_building <= 1'b0;
                response_pending <= 1'b0;
                abort_ack <= 1'b1;
            end
        end
    end

    initial begin
        if (MAX_CURRENT_ROWS != 48 || MAX_SEQUENCE != 2048 || LOGICAL_WIDTH != 11 ||
            ADDR_WIDTH != 64)
            $error("kv_cache_access_controller requires 48 current tokens, Smax 2048 and 64-bit byte addresses");
    end

`ifndef SYNTHESIS
    logic previous_read_stall;
    logic [6*ADDR_WIDTH+LOGICAL_WIDTH+31:0] previous_read_payload;

    always_ff @(posedge clk) begin
        if (rst) begin
            previous_read_stall <= 1'b0;
            previous_read_payload <= '0;
        end else begin
            if (TRUSTED_START_CONFIGURATION && config_building &&
                batch_group_enable)
                assert ($stable(start_batch_group))
                    else $error("kv_cache_access grouped configuration changed while visibility was being built");
            assert (retained_write_count == 0)
                else $error("kv_cache_access state committed current data to retained cache");
            if (previous_read_stall && !abort_request)
                assert (read_rsp_valid &&
                    {read_rsp_key_group, read_rsp_current_mask, read_rsp_retained_mask,
                     read_rsp_current_k_address, read_rsp_current_v_address,
                     read_rsp_current_k_scale_address, read_rsp_retained_k_address,
                     read_rsp_retained_v_address, read_rsp_retained_k_scale_address,
                     read_rsp_tag} ==
                    previous_read_payload)
                    else $error("kv_cache_access state changed a stalled read response");
            previous_read_stall <= read_rsp_valid && !read_rsp_ready && !abort_request;
            previous_read_payload <= {
                read_rsp_key_group, read_rsp_current_mask, read_rsp_retained_mask,
                read_rsp_current_k_address, read_rsp_current_v_address,
                read_rsp_current_k_scale_address, read_rsp_retained_k_address,
                read_rsp_retained_v_address, read_rsp_retained_k_scale_address,
                read_rsp_tag
            };
        end
    end
`endif
endmodule

`default_nettype wire
