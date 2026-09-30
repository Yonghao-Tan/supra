`default_nettype none

// Moves the fixed 4096-channel hidden tensor between DDR and 16 hidden stripes.
// Execution transfers are logical-token-major; FFN residual transfers are
// N8-stripe-major. Up to two SRAM reads cover the two-cycle macro response;
// one logical DMA transfer remains active until its data and response drain.
module hidden_transfer #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 8,
    parameter logic [TAG_WIDTH-1:0] DMA_TAG = TAG_WIDTH'(8'h8f)
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    abort_request,
    output logic                    abort_ack,

    input  logic                    start_valid,
    output logic                    start_ready,
    input  logic [1:0]              start_operation,
    input  logic [5:0]              start_active_rows,
    input  logic [287:0]            start_physical_to_token_ordinal,
    input  logic                    start_direct_row_index_enable,
    input  logic [527:0]            start_direct_row_index,
    input  logic                    start_source_descriptor_enable,
    input  logic [815:0]            start_source_index,
    input  logic [47:0]             start_source_embedding,
    input  logic [ADDR_WIDTH-1:0]   start_embedding_base,
    input  logic [ADDR_WIDTH-1:0]   start_embedding_limit,
    input  logic [ADDR_WIDTH-1:0]   start_ddr_base,
    input  logic [ADDR_WIDTH-1:0]   start_ddr_limit,
    input  logic                    start_for_ffn,
    output logic                    busy,
    output logic                    done_valid,
    input  logic                    done_ready,
    output logic                    done_for_ffn,
    output logic                    error,
    output logic [3:0]              error_id,

    output logic                    read_request_valid,
    input  logic                    read_request_ready,
    output logic [ADDR_WIDTH-1:0]   read_request_address,
    output logic [31:0]             read_request_bytes,
    output logic [TAG_WIDTH-1:0]    read_request_tag,
    input  logic                    read_request_done,
    input  logic                    read_request_error,
    input  logic                    read_data_valid,
    output logic                    read_data_ready,
    input  logic [127:0]            read_data,
    input  logic [15:0]             read_byte_enable,
    input  logic                    read_data_last,
    input  logic [TAG_WIDTH-1:0]    read_data_tag,

    output logic                    write_request_valid,
    input  logic                    write_request_ready,
    output logic [ADDR_WIDTH-1:0]   write_request_address,
    output logic [31:0]             write_request_bytes,
    output logic [TAG_WIDTH-1:0]    write_request_tag,
    input  logic                    write_request_done,
    input  logic                    write_request_error,
    output logic                    write_data_valid,
    input  logic                    write_data_ready,
    output logic [127:0]            write_data,
    output logic [15:0]             write_byte_enable,
    output logic                    write_data_last,

    output logic                    local_write_valid,
    input  logic                    local_write_ready,
    output logic [5:0]              local_write_physical_row,
    output logic [8:0]              local_write_channel_word,
    output logic [127:0]            local_write_data,
    output logic [15:0]             local_write_byte_enable,
    output logic                    local_read_valid,
    input  logic                    local_read_ready,
    output logic [5:0]              local_read_physical_row,
    output logic [8:0]              local_read_channel_word,
    input  logic                    local_read_response_valid,
    input  logic [127:0]            local_read_data,

    output logic [63:0]             accepted_dma_request_count,
    output logic [63:0]             accepted_dma_data_count,
    output logic [63:0]             accepted_local_request_count
);
`ifdef SYNTHESIS
    always_comb begin
        accepted_dma_request_count = '0;
        accepted_dma_data_count = '0;
        accepted_local_request_count = '0;
    end
`endif
    import local_memory_layout_pkg::*;

    localparam logic [1:0] OP_EXECUTION_INPUT = 2'd0;
    localparam logic [1:0] OP_EXECUTION_OUTPUT = 2'd1;
    localparam logic [1:0] OP_RESIDUAL_REFILL = 2'd3;
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_PERMUTATION = 4'h2;
    localparam logic [3:0] ERROR_DMA_PROTOCOL = 4'h3;
    localparam logic [3:0] ERROR_DMA_RESPONSE = 4'h4;

    typedef enum logic [3:0] {
        IDLE,
        BUILD_INVERSE,
        PREPARE_DIRECT_ROW,
        ISSUE_DMA,
        READ_STREAM,
        WRITE_STREAM,
        WAIT_DMA_DONE,
        COMPLETE,
        ABORT_WAIT_LOW
    } state_t;

    state_t state;
    logic [1:0] operation;
    logic [5:0] active_token_count;
    logic [ADDR_WIDTH-1:0] ddr_base;
    logic [31:0] transfer_bytes;
    logic [14:0] total_beats;
    logic [14:0] completed_beats;
    logic [8:0] channel_chunk;
    logic [8:0] n8_stripe;
    logic [5:0] traversal_row;
    logic [5:0] permutation_index;
    logic [47:0] permutation_seen;
    logic [5:0] inverse_row [0:47];
    logic [287:0] inverse_row_values;
    logic dma_terminal_seen;
    logic dma_error_seen;
    logic local_read_holding_valid;
    logic [127:0] local_read_holding_data;
    logic local_read_skid_valid;
    logic [127:0] local_read_skid_data;
    logic [1:0] local_read_pending_count;
    logic [14:0] issued_read_beats;
    logic [8:0] issue_channel_chunk;
    logic [8:0] issue_n8_stripe;
    logic [5:0] issue_traversal_row;
    logic [5:0] issue_mapped_physical_row;
    logic write_data_fire;
    logic local_read_fire;
    logic [2:0] buffered_write_beats;
    logic abort_pending;
    logic direction_read;
    logic execution_layout;
    logic final_beat;
    logic [5:0] permutation_value;
    logic [5:0] mapped_physical_row;
    logic [ADDR_WIDTH:0] configured_end;
    logic [ADDR_WIDTH-1:0] ddr_region_base;
    logic [ADDR_WIDTH-1:0] ddr_limit;
    logic [ADDR_WIDTH-1:0] embedding_base;
    logic [ADDR_WIDTH-1:0] embedding_limit;
    logic [527:0] direct_row_indices;
    logic direct_row_mode;
    logic source_descriptor_mode;
    logic [5:0] direct_row;
    logic [10:0] direct_row_index_value;
    logic [16:0] direct_row_source_index;
    logic direct_row_embedding;
    logic [ADDR_WIDTH-1:0] direct_row_region_base;
    logic [ADDR_WIDTH-1:0] direct_row_region_limit;
    logic [ADDR_WIDTH:0] direct_row_start;
    logic [ADDR_WIDTH:0] direct_row_end;
    logic permutation_value_seen;
    logic for_ffn;

    function automatic logic [5:0] inverse_row_at(
        input logic [287:0] values, input logic [5:0] row
    );
        if (row < 6'd48)
            inverse_row_at = values[integer'(row)*6 +: 6];
        else
            inverse_row_at = 6'd0;
    endfunction

    function automatic logic [16:0] source_index_at(
        input logic [815:0] values,
        input logic [5:0] row
    );
        case (row)
            6'd0: source_index_at = values[0*17 +: 17];
            6'd1: source_index_at = values[1*17 +: 17];
            6'd2: source_index_at = values[2*17 +: 17];
            6'd3: source_index_at = values[3*17 +: 17];
            6'd4: source_index_at = values[4*17 +: 17];
            6'd5: source_index_at = values[5*17 +: 17];
            6'd6: source_index_at = values[6*17 +: 17];
            6'd7: source_index_at = values[7*17 +: 17];
            6'd8: source_index_at = values[8*17 +: 17];
            6'd9: source_index_at = values[9*17 +: 17];
            6'd10: source_index_at = values[10*17 +: 17];
            6'd11: source_index_at = values[11*17 +: 17];
            6'd12: source_index_at = values[12*17 +: 17];
            6'd13: source_index_at = values[13*17 +: 17];
            6'd14: source_index_at = values[14*17 +: 17];
            6'd15: source_index_at = values[15*17 +: 17];
            6'd16: source_index_at = values[16*17 +: 17];
            6'd17: source_index_at = values[17*17 +: 17];
            6'd18: source_index_at = values[18*17 +: 17];
            6'd19: source_index_at = values[19*17 +: 17];
            6'd20: source_index_at = values[20*17 +: 17];
            6'd21: source_index_at = values[21*17 +: 17];
            6'd22: source_index_at = values[22*17 +: 17];
            6'd23: source_index_at = values[23*17 +: 17];
            6'd24: source_index_at = values[24*17 +: 17];
            6'd25: source_index_at = values[25*17 +: 17];
            6'd26: source_index_at = values[26*17 +: 17];
            6'd27: source_index_at = values[27*17 +: 17];
            6'd28: source_index_at = values[28*17 +: 17];
            6'd29: source_index_at = values[29*17 +: 17];
            6'd30: source_index_at = values[30*17 +: 17];
            6'd31: source_index_at = values[31*17 +: 17];
            6'd32: source_index_at = values[32*17 +: 17];
            6'd33: source_index_at = values[33*17 +: 17];
            6'd34: source_index_at = values[34*17 +: 17];
            6'd35: source_index_at = values[35*17 +: 17];
            6'd36: source_index_at = values[36*17 +: 17];
            6'd37: source_index_at = values[37*17 +: 17];
            6'd38: source_index_at = values[38*17 +: 17];
            6'd39: source_index_at = values[39*17 +: 17];
            6'd40: source_index_at = values[40*17 +: 17];
            6'd41: source_index_at = values[41*17 +: 17];
            6'd42: source_index_at = values[42*17 +: 17];
            6'd43: source_index_at = values[43*17 +: 17];
            6'd44: source_index_at = values[44*17 +: 17];
            6'd45: source_index_at = values[45*17 +: 17];
            6'd46: source_index_at = values[46*17 +: 17];
            6'd47: source_index_at = values[47*17 +: 17];
            default: source_index_at = 17'd0;
        endcase
    endfunction

    for (genvar row = 0; row < 48; row++) begin : g_inverse_row_values
        assign inverse_row_values[row*6 +: 6] = inverse_row[row];
    end

    assign start_ready = state == IDLE && !abort_request;
    assign busy = state != IDLE && state != ABORT_WAIT_LOW;
    assign done_valid = state == COMPLETE;
    assign done_for_ffn = for_ffn;
    assign direction_read = operation == OP_EXECUTION_INPUT ||
        operation == OP_RESIDUAL_REFILL;
    assign execution_layout = operation == OP_EXECUTION_INPUT || operation == OP_EXECUTION_OUTPUT;
    assign final_beat = completed_beats + 1'b1 == total_beats;
    assign permutation_value =
        start_physical_to_token_ordinal[permutation_index*6 +: 6];
    assign permutation_value_seen = permutation_value < 6'd48 ?
        permutation_seen[permutation_value] : 1'b1;
    assign mapped_physical_row = direct_row_mode ? traversal_row :
        (execution_layout ? inverse_row_at(inverse_row_values, traversal_row) :
         traversal_row);
    assign issue_mapped_physical_row = direct_row_mode ? issue_traversal_row :
        (execution_layout ? inverse_row_at(inverse_row_values, issue_traversal_row) :
         issue_traversal_row);
    assign configured_end = {1'b0, start_ddr_base} +
        (ADDR_WIDTH+1)'(start_active_rows * 32'd8192);
    assign direct_row_index_value = direct_row_indices[
        integer'(direct_row)*11 +: 11];
    assign direct_row_source_index = source_descriptor_mode ?
        source_index_at(start_source_index, direct_row) :
        {6'd0, direct_row_index_value};
    assign direct_row_embedding = source_descriptor_mode &&
        start_source_embedding[direct_row];
    assign direct_row_region_base = direct_row_embedding ?
        embedding_base : ddr_region_base;
    assign direct_row_region_limit = direct_row_embedding ?
        embedding_limit : ddr_limit;
    assign direct_row_start = {1'b0, direct_row_region_base} +
        ((ADDR_WIDTH+1)'(direct_row_source_index) << 13);
    assign direct_row_end = direct_row_start + (ADDR_WIDTH+1)'(8192);

    assign read_request_valid = state == ISSUE_DMA && direction_read &&
        !abort_pending && !abort_request;
    assign read_request_address = ddr_base;
    assign read_request_bytes = transfer_bytes;
    assign read_request_tag = DMA_TAG;
    assign write_request_valid = state == ISSUE_DMA && !direction_read &&
        !abort_pending && !abort_request;
    assign write_request_address = ddr_base;
    assign write_request_bytes = transfer_bytes;
    assign write_request_tag = DMA_TAG;

    assign local_write_valid = state == READ_STREAM && read_data_valid &&
        !abort_pending;
    assign local_write_physical_row = mapped_physical_row;
    assign local_write_channel_word = execution_layout ? channel_chunk : n8_stripe;
    assign local_write_data = read_data;
    assign local_write_byte_enable = read_byte_enable;
    assign read_data_ready = state == READ_STREAM &&
        (abort_pending || local_write_ready);

    // An accepted logical write still needs its complete W payload during
    // abort drain, so abort cannot suppress the remaining SRAM reads.
    assign write_data_fire = write_data_valid && write_data_ready;
    assign buffered_write_beats = {2'd0, local_read_holding_valid} +
        {2'd0, local_read_skid_valid} +
        {1'b0, local_read_pending_count};
    assign local_read_valid = state == WRITE_STREAM &&
        !dma_error_seen && !write_request_error &&
        issued_read_beats < total_beats &&
        (buffered_write_beats < 3'd2 || write_data_fire);
    assign local_read_fire = local_read_valid && local_read_ready;
    assign local_read_physical_row = issue_mapped_physical_row;
    assign local_read_channel_word = execution_layout ?
        issue_channel_chunk : issue_n8_stripe;
    assign write_data_valid = state == WRITE_STREAM && local_read_holding_valid &&
        !dma_error_seen && !write_request_error;
    assign write_data = local_read_holding_data;
    assign write_byte_enable = 16'hffff;
    assign write_data_last = final_beat;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            operation <= OP_EXECUTION_INPUT;
            active_token_count <= '0;
            ddr_base <= '0;
            ddr_region_base <= '0;
            ddr_limit <= '0;
            embedding_base <= '0;
            embedding_limit <= '0;
            direct_row_indices <= '0;
            direct_row_mode <= 1'b0;
            source_descriptor_mode <= 1'b0;
            direct_row <= '0;
            transfer_bytes <= '0;
            total_beats <= '0;
            completed_beats <= '0;
            channel_chunk <= '0;
            n8_stripe <= '0;
            traversal_row <= '0;
            permutation_index <= '0;
            permutation_seen <= '0;
            dma_terminal_seen <= 1'b0;
            dma_error_seen <= 1'b0;
            local_read_holding_valid <= 1'b0;
            local_read_holding_data <= '0;
            local_read_skid_valid <= 1'b0;
            local_read_skid_data <= '0;
            local_read_pending_count <= '0;
            issued_read_beats <= '0;
            issue_channel_chunk <= '0;
            issue_n8_stripe <= '0;
            issue_traversal_row <= '0;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            for_ffn <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
`ifndef SYNTHESIS
            accepted_dma_request_count <= '0;
            accepted_dma_data_count <= '0;
            accepted_local_request_count <= '0;
`endif
            for (integer row = 0; row < 48; row = row + 1)
                inverse_row[row] <= '0;
        end else begin
            abort_ack <= 1'b0;
            if (abort_request && state != IDLE && state != COMPLETE &&
                state != ABORT_WAIT_LOW)
                abort_pending <= 1'b1;

            if (read_request_done || write_request_done)
                dma_terminal_seen <= 1'b1;
            if (read_request_error || write_request_error) begin
                dma_terminal_seen <= 1'b1;
                dma_error_seen <= 1'b1;
                if (!error)
                    error_id <= ERROR_DMA_RESPONSE;
                error <= 1'b1;
            end

            case (state)
                IDLE: if (start_valid && start_ready) begin
                    error <= 1'b0;
                    error_id <= '0;
                    abort_pending <= 1'b0;
                    dma_terminal_seen <= 1'b0;
                    dma_error_seen <= 1'b0;
                    local_read_holding_valid <= 1'b0;
                    local_read_skid_valid <= 1'b0;
                    local_read_pending_count <= '0;
                    issued_read_beats <= '0;
                    operation <= start_operation;
                    active_token_count <= start_active_rows;
                    ddr_base <= start_ddr_base;
                    ddr_region_base <= start_ddr_base;
                    ddr_limit <= start_ddr_limit;
                    embedding_base <= start_embedding_base;
                    embedding_limit <= start_embedding_limit;
                    direct_row_indices <= start_direct_row_index;
                    direct_row_mode <= start_direct_row_index_enable;
                    source_descriptor_mode <= start_source_descriptor_enable;
                    direct_row <= 6'd0;
                    transfer_bytes <= start_direct_row_index_enable ?
                        32'd8192 : start_active_rows * 32'd8192;
                    total_beats <= start_direct_row_index_enable ?
                        15'd512 : start_active_rows * 15'd512;
                    completed_beats <= '0;
                    channel_chunk <= '0;
                    n8_stripe <= '0;
                    traversal_row <= '0;
                    issue_channel_chunk <= '0;
                    issue_n8_stripe <= '0;
                    issue_traversal_row <= '0;
                    permutation_index <= '0;
                    permutation_seen <= '0;
                    for_ffn <= start_for_ffn;
                    if (start_active_rows == 0 || start_active_rows > 6'd48 ||
                        start_ddr_base[7:0] != 0 ||
                        start_ddr_limit <= start_ddr_base ||
                        (start_direct_row_index_enable &&
                         start_operation != OP_EXECUTION_INPUT &&
                         start_operation != OP_EXECUTION_OUTPUT) ||
                        (start_source_descriptor_enable &&
                         (!start_direct_row_index_enable ||
                          start_operation != OP_EXECUTION_INPUT ||
                          start_embedding_base[12:0] != 0 ||
                          start_embedding_limit <= start_embedding_base)) ||
                        (!start_direct_row_index_enable &&
                         (configured_end[ADDR_WIDTH] ||
                          configured_end > {1'b0, start_ddr_limit}))) begin
                        error <= 1'b1;
                        error_id <= ERROR_CONFIGURATION;
                        state <= COMPLETE;
                    end else if (start_direct_row_index_enable) begin
                        state <= PREPARE_DIRECT_ROW;
                    end else if (start_operation == OP_EXECUTION_INPUT ||
                                 start_operation == OP_EXECUTION_OUTPUT) begin
                        state <= BUILD_INVERSE;
                    end else begin
                        state <= ISSUE_DMA;
                    end
                end

                BUILD_INVERSE: begin
                    if (permutation_value >= active_token_count ||
                        permutation_value_seen) begin
                        error <= 1'b1;
                        error_id <= ERROR_PERMUTATION;
                        state <= COMPLETE;
                    end else begin
                        inverse_row[permutation_value] <= permutation_index;
                        permutation_seen[permutation_value] <= 1'b1;
                        if (permutation_index + 1'b1 == active_token_count) begin
                            state <= ISSUE_DMA;
                        end else begin
                            permutation_index <= permutation_index + 1'b1;
                        end
                    end
                end

                PREPARE_DIRECT_ROW: begin
                    if ((source_descriptor_mode && direct_row_embedding &&
                         direct_row_source_index >= 17'd126464) ||
                        (source_descriptor_mode && !direct_row_embedding &&
                         direct_row_source_index >= 17'd2048) ||
                        direct_row_start[ADDR_WIDTH] ||
                        direct_row_end[ADDR_WIDTH] ||
                        direct_row_end > {1'b0, direct_row_region_limit}) begin
                        error <= 1'b1;
                        error_id <= ERROR_CONFIGURATION;
                        state <= COMPLETE;
                    end else begin
                        ddr_base <= direct_row_start[ADDR_WIDTH-1:0];
                        transfer_bytes <= 32'd8192;
                        total_beats <= 15'd512;
                        completed_beats <= '0;
                        issued_read_beats <= '0;
                        channel_chunk <= '0;
                        issue_channel_chunk <= '0;
                        traversal_row <= direct_row;
                        issue_traversal_row <= direct_row;
                        dma_terminal_seen <= 1'b0;
                        dma_error_seen <= 1'b0;
                        local_read_holding_valid <= 1'b0;
                        local_read_skid_valid <= 1'b0;
                        local_read_pending_count <= '0;
                        state <= ISSUE_DMA;
                    end
                end

                ISSUE_DMA: begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if ((direction_read && read_request_valid && read_request_ready) ||
                                 (!direction_read && write_request_valid &&
                                  write_request_ready)) begin
`ifndef SYNTHESIS
                        accepted_dma_request_count <= accepted_dma_request_count + 1'b1;
`endif
                        state <= direction_read ? READ_STREAM : WRITE_STREAM;
                    end
                end

                READ_STREAM: begin
                    if (read_data_valid && read_data_ready) begin
`ifndef SYNTHESIS
                        accepted_dma_data_count <= accepted_dma_data_count + 1'b1;
`endif
                        if (read_data_tag != DMA_TAG || read_byte_enable != 16'hffff ||
                            read_data_last != final_beat) begin
                            if (!error)
                                error_id <= ERROR_DMA_PROTOCOL;
                            error <= 1'b1;
                        end
`ifndef SYNTHESIS
                        if (!abort_pending)
                            accepted_local_request_count <=
                                accepted_local_request_count + 1'b1;
`endif
                        completed_beats <= completed_beats + 1'b1;
                        if (execution_layout) begin
                            if (channel_chunk == 9'd511) begin
                                channel_chunk <= '0;
                                traversal_row <= traversal_row + 1'b1;
                            end else begin
                                channel_chunk <= channel_chunk + 1'b1;
                            end
                        end else if (traversal_row + 1'b1 == active_token_count) begin
                            traversal_row <= '0;
                            n8_stripe <= n8_stripe + 1'b1;
                        end else begin
                            traversal_row <= traversal_row + 1'b1;
                        end
                        if (final_beat)
                            state <= WAIT_DMA_DONE;
                    end else if (dma_error_seen || read_request_error) begin
                        state <= WAIT_DMA_DONE;
                    end
                end

                WRITE_STREAM: begin
                    if (local_read_fire) begin
                        issued_read_beats <= issued_read_beats + 1'b1;
`ifndef SYNTHESIS
                        accepted_local_request_count <=
                            accepted_local_request_count + 1'b1;
`endif
                        if (execution_layout) begin
                            if (issue_channel_chunk == 9'd511) begin
                                issue_channel_chunk <= '0;
                                issue_traversal_row <= issue_traversal_row + 1'b1;
                            end else begin
                                issue_channel_chunk <= issue_channel_chunk + 1'b1;
                            end
                        end else if (issue_traversal_row + 1'b1 == active_token_count) begin
                            issue_traversal_row <= '0;
                            issue_n8_stripe <= issue_n8_stripe + 1'b1;
                        end else begin
                            issue_traversal_row <= issue_traversal_row + 1'b1;
                        end
                    end

                    if (write_data_fire) begin
                        if (local_read_skid_valid) begin
                            local_read_holding_data <= local_read_skid_data;
                            local_read_holding_valid <= 1'b1;
                            local_read_skid_valid <= 1'b0;
                        end else begin
                            local_read_holding_valid <= 1'b0;
                        end
`ifndef SYNTHESIS
                        accepted_dma_data_count <= accepted_dma_data_count + 1'b1;
`endif
                        completed_beats <= completed_beats + 1'b1;
                        if (execution_layout) begin
                            if (channel_chunk == 9'd511) begin
                                channel_chunk <= '0;
                                traversal_row <= traversal_row + 1'b1;
                            end else begin
                                channel_chunk <= channel_chunk + 1'b1;
                            end
                        end else if (traversal_row + 1'b1 == active_token_count) begin
                            traversal_row <= '0;
                            n8_stripe <= n8_stripe + 1'b1;
                        end else begin
                            traversal_row <= traversal_row + 1'b1;
                        end
                        if (final_beat)
                            state <= WAIT_DMA_DONE;
                    end

                    if (local_read_response_valid) begin
                        if (!local_read_holding_valid ||
                            (write_data_fire && !local_read_skid_valid)) begin
                            local_read_holding_data <= local_read_data;
                            local_read_holding_valid <= 1'b1;
                        end else begin
                            local_read_skid_data <= local_read_data;
                            local_read_skid_valid <= 1'b1;
                        end
                    end
                    case ({local_read_fire, local_read_response_valid})
                        2'b10: local_read_pending_count <=
                            local_read_pending_count + 1'b1;
                        2'b01: local_read_pending_count <=
                            local_read_pending_count - 1'b1;
                        default: local_read_pending_count <=
                            local_read_pending_count;
                    endcase
                    // The DMA terminal error means its AXI transactions have
                    // drained. Discard the unsent payload, but wait for every
                    // accepted SRAM read before reporting completion.
                    if (dma_error_seen || write_request_error) begin
                        local_read_holding_valid <= 1'b0;
                        local_read_skid_valid <= 1'b0;
                        if (local_read_pending_count == 0 ||
                            (local_read_pending_count == 1 &&
                             local_read_response_valid))
                            state <= WAIT_DMA_DONE;
                    end
                end

                WAIT_DMA_DONE: if (dma_terminal_seen || read_request_done ||
                                      read_request_error || write_request_done ||
                                      write_request_error) begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end else if (dma_error_seen || read_request_error ||
                                 write_request_error) begin
                        state <= COMPLETE;
                    end else if (direct_row_mode &&
                                 direct_row + 6'd1 < active_token_count) begin
                        direct_row <= direct_row + 6'd1;
                        state <= PREPARE_DIRECT_ROW;
                    end else begin
                        state <= COMPLETE;
                    end
                end

                COMPLETE: if (done_ready)
                    state <= IDLE;

                ABORT_WAIT_LOW: if (!abort_request) begin
                    abort_pending <= 1'b0;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (ADDR_WIDTH < 32 || TAG_WIDTH < 4)
            $error("hidden_transfer requires ADDR_WIDTH>=32 and TAG_WIDTH>=4");
    end

`ifndef SYNTHESIS
    logic write_was_stalled;
    logic [144:0] stalled_write_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            write_was_stalled <= 1'b0;
            stalled_write_payload <= '0;
        end else begin
            assert (!(read_request_valid && write_request_valid))
                else $error("hidden_transfer issued read and write together");
            assert (!(local_write_valid && local_read_valid))
                else $error("hidden_transfer used both hidden SRAM directions together");
            if (local_read_response_valid)
                assert (state == WRITE_STREAM && local_read_pending_count != 0)
                    else $error("hidden_transfer received an unexpected SRAM response");
            assert (buffered_write_beats <= 3'd2)
                else $error("hidden_transfer write pipeline overflow");
            if (write_was_stalled && !dma_error_seen && !write_request_error)
                assert (write_data_valid &&
                    {write_data, write_byte_enable, write_data_last} ==
                    stalled_write_payload)
                    else $error("hidden_transfer changed stalled write data");
            if ($past(done_valid && !done_ready))
                assert (done_valid && $stable({done_for_ffn, error, error_id}))
                    else $error("hidden_transfer changed a stalled completion");
            write_was_stalled <= write_data_valid && !write_data_ready &&
                !dma_error_seen && !write_request_error;
            if (write_data_valid && !write_data_ready)
                stalled_write_payload <=
                    {write_data, write_byte_enable, write_data_last};
        end
    end
`endif
endmodule

`default_nettype wire
