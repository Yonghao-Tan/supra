`default_nettype none

package local_memory_layout_pkg;
    localparam int unsigned LOCAL_SRAM_MACRO_COUNT = 26;
    localparam int unsigned LOCAL_SRAM_2048_MACRO_COUNT = 14;
    localparam int unsigned LOCAL_SRAM_1024_MACRO_COUNT = 12;
    localparam int unsigned LOCAL_SRAM_MACRO_PORTS = 2;
    localparam int unsigned LOCAL_SRAM_PHYSICAL_PORT_COUNT =
        LOCAL_SRAM_MACRO_COUNT * LOCAL_SRAM_MACRO_PORTS;
    localparam int unsigned LOCAL_SRAM_ROW_BITS = 128;
    localparam int unsigned LOCAL_SRAM_MAXIMUM_ROWS = 2048;
    localparam int unsigned LOCAL_SRAM_TOTAL_BYTES = 655360;
    localparam int unsigned LOCAL_SRAM_REGION_ROWS = 1024;
    localparam int unsigned LOCAL_SRAM_REGION_BYTES = 16 * 1024;

    typedef enum logic [3:0] {
        LOCAL_MEMORY_PHASE_IDLE = 4'd0,
        LOCAL_MEMORY_PHASE_MATMUL = 4'd1,
        LOCAL_MEMORY_PHASE_QKV_PREPARATION = 4'd2,
        LOCAL_MEMORY_PHASE_ATTENTION_QK = 4'd3,
        LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX = 4'd4,
        LOCAL_MEMORY_PHASE_ATTENTION_PV = 4'd5,
        LOCAL_MEMORY_PHASE_RMSNORM = 4'd6,
        LOCAL_MEMORY_PHASE_ELEMENTWISE = 4'd7,
        LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN = 4'd8,
        LOCAL_MEMORY_PHASE_FINAL_OUTPUT = 4'd9,
        LOCAL_MEMORY_PHASE_R4 = 4'd10,
        LOCAL_MEMORY_PHASE_R4_TWO_TOKENS = 4'd11
    } local_memory_phase_t;

    typedef enum logic [1:0] {
        LOCAL_PANEL_LAYOUT_NONE = 2'd0,
        LOCAL_PANEL_LAYOUT_QKV = 2'd1,
        LOCAL_PANEL_LAYOUT_MIXED = 2'd2,
        LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL = 2'd3
    } local_panel_layout_t;

    typedef struct packed {
        logic kv_pair_enable;
        logic kv_second_batch;
        logic [5:0] kv_first_batch_tokens;
        logic attention_pair_enable;
        logic attention_second_batch;
        logic [1:0] attention_batch_index;
        logic [7:0] attention_score_words;
        local_memory_phase_t phase;
        logic active_panel;
        logic qkv_head_staging;
        logic q_head_active;
        logic q_head_slot;
        local_panel_layout_t panel_layout;
    } layout_requirement_t;

    typedef struct packed {
        logic [4:0] macro_id;
        logic [11:0] row_base;
        logic primary_port;
    } local_sram_region_t;

    localparam int unsigned MAXIMUM_A8_ACTIVATION_BYTES = 384 * 1024;
    localparam int unsigned MAXIMUM_W8_PANEL_BYTES = 96 * 1024;
    localparam int unsigned WRITE_COMBINING_BYTES = 8 * 1024;
    localparam int unsigned QKV_ROPE_CONSTANT_WORD_COUNT = 512;
    localparam int unsigned QKV_MATMUL_COMBINE_WORD_BASE = 512;
    localparam int unsigned QKV_MATMUL_COMBINE_WORD_COUNT = 96;
    localparam int unsigned QKV_PANEL_HALF_WORD_COUNT = 512;
    localparam int unsigned QKV_W4_WEIGHT_WORD_COUNT = 256;
    localparam int unsigned QKV_HEAD_STAGING_WORD_BASE =
        QKV_W4_WEIGHT_WORD_COUNT;
    localparam int unsigned QKV_MAXIMUM_ROW_COUNT = 48;
    localparam int unsigned QKV_SCALE_ROW_GROUP_COUNT =
        QKV_MAXIMUM_ROW_COUNT / 8;
    localparam int unsigned QKV_PROBABILITY_SCALE_WORD_BASE =
        32 * QKV_SCALE_ROW_GROUP_COUNT;
    localparam int unsigned CANDIDATE_LANE_STATE_BITS = 48;
    localparam int unsigned CANDIDATE_LANES_PER_BLOCK = 8;
    localparam int unsigned CANDIDATE_STATE_WORDS_PER_BLOCK = 3;
    localparam int unsigned CANDIDATE_STATE_ROW_BASE = 512;
    localparam int unsigned CANDIDATE_STATE_MAXIMUM_ROW = 895;
    localparam int unsigned POST_STATE_ENTRY_COUNT = 128;
    localparam int unsigned POST_STATE_WORDS_PER_ENTRY = 2;
    localparam int unsigned POST_STATE_WORD_COUNT =
        POST_STATE_ENTRY_COUNT * POST_STATE_WORDS_PER_ENTRY;
    localparam int unsigned POST_SUPPRESSED_TOKEN_COUNT = 255;
    localparam int unsigned POST_SUPPRESSED_ROW_BASE = POST_STATE_WORD_COUNT;
    localparam int unsigned POST_SUPPRESSED_ROW_COUNT = 64;

    function automatic logic [9:0] qkv_query_scale_word(
        input logic [5:0] head,
        input logic [5:0] physical_row
    );
        qkv_query_scale_word = 10'(
            integer'(head) * QKV_SCALE_ROW_GROUP_COUNT +
            integer'(physical_row[5:3]));
    endfunction

    function automatic logic [9:0] qkv_probability_scale_word(
        input logic [5:0] head,
        input logic [5:0] physical_row
    );
        qkv_probability_scale_word = 10'(
            QKV_PROBABILITY_SCALE_WORD_BASE +
            integer'(head) * QKV_SCALE_ROW_GROUP_COUNT +
            integer'(physical_row[5:3]));
    endfunction

    function automatic local_sram_region_t local_region(
        input logic [4:0] macro_id,
        input logic [11:0] row_base,
        input logic primary_port
    );
        local_region = '{macro_id: macro_id, row_base: row_base,
            primary_port: primary_port};
    endfunction

    localparam int unsigned ATTENTION_PAIR_ACTIVATION_WORD_COUNT = 12288;
    localparam int unsigned ATTENTION_PAIR_SECOND_ACTIVATION_WORD = 6144;
    localparam int unsigned ATTENTION_PAIR_MAXIMUM_SEQUENCE_LENGTH = 1296;
    localparam int unsigned ATTENTION_PAIR_QUERY_ROW_BASE = 1952;
    localparam int unsigned ATTENTION_PAIR_HEAD_WORD_COUNT = 768;
    localparam int unsigned ATTENTION_PAIR_Q_WEIGHT_WORDS_PER_MACRO = 256;
    localparam int unsigned ATTENTION_PAIR_HEAD_STAGING_WORDS_PER_MACRO = 192;
    localparam int unsigned ATTENTION_PAIR_K_PANEL_WORDS_PER_MACRO = 512;
    localparam int unsigned ATTENTION_PAIR_SCALE_WORDS_PER_BATCH = 8;
    localparam int unsigned ATTENTION_PAIR_K_SCALE_WORD_COUNT = 32;
    localparam int unsigned ATTENTION_PAIR_SOFTMAX_SCRATCH_WORD_COUNT = 256;

    localparam local_sram_region_t ATTENTION_PAIR_HEAD_SOURCE_REGION =
        '{macro_id: 5'd12, row_base: 12'd0, primary_port: 1'b0};
    localparam local_sram_region_t ATTENTION_PAIR_HEAD_DESTINATION_REGION =
        '{macro_id: 5'd13, row_base: 12'd0, primary_port: 1'b0};
    localparam local_sram_region_t ATTENTION_PAIR_ROPE_CONSTANT_REGION =
        '{macro_id: 5'd8, row_base: 12'd512, primary_port: 1'b0};
    localparam local_sram_region_t ATTENTION_PAIR_K_SCALE_REGION =
        '{macro_id: 5'd13, row_base: 12'd768, primary_port: 1'b0};

    function automatic logic attention_pair_sequence_supported(
        input logic [11:0] sequence_length
    );
        attention_pair_sequence_supported = sequence_length != 0 &&
            sequence_length <= 12'(ATTENTION_PAIR_MAXIMUM_SEQUENCE_LENGTH);
    endfunction

    // Keep the full result for unsupported lengths; only supported strides
    // fit the eight-bit layout field and leave both Q8 images untouched.
    function automatic logic [9:0] attention_pair_score_stride(
        input logic [11:0] sequence_length
    );
        logic [12:0] rounded_length;
        logic [9:0] score_words;
        begin
            rounded_length = {1'b0, sequence_length} + 13'd15;
            score_words = {rounded_length[12:4], 1'b0};
            attention_pair_score_stride = score_words < 10'd32 ?
                10'd32 : score_words;
        end
    endfunction

    // The explicit activation address already includes the second batch.
    function automatic logic [4:0] attention_pair_activation_macro(
        input logic [13:0] word_index
    );
        attention_pair_activation_macro = 5'd14 +
            {1'b0, word_index[13:12], 2'b00} + {3'b000, word_index[2:1]};
    endfunction

    function automatic logic [11:0] attention_pair_activation_row(
        input logic [13:0] word_index
    );
        attention_pair_activation_row = {2'b00, word_index[11:3], word_index[0]};
    endfunction

    function automatic local_sram_region_t attention_pair_row_region(
        input logic [1:0] batch_index,
        input logic [5:0] physical_row,
        input logic [7:0] score_words
    );
        logic [11:0] stride;
        logic [11:0] row_base;
        begin
            stride = {4'b0000, score_words};
            case (physical_row[5:3])
                3'd0: row_base = 12'd0;
                3'd1: row_base = stride;
                3'd2: row_base = stride << 1;
                3'd3: row_base = (stride << 1) + stride;
                3'd4: row_base = stride << 2;
                default: row_base = (stride << 2) + stride;
            endcase
            if (batch_index == 2'd1)
                row_base = row_base + (stride << 2) + (stride << 1);
            attention_pair_row_region = local_region(
                batch_index == 2'd2 ?
                    5'd14 + {2'b00, physical_row[2:0]} :
                    {2'b00, physical_row[2:0]}, row_base, 1'b0);
        end
    endfunction

    function automatic local_sram_region_t attention_pair_context_region(
        input logic [1:0] batch_index,
        input logic [5:0] physical_row,
        input logic [7:0] score_words
    );
        local_sram_region_t region;
        begin
            region = attention_pair_row_region(batch_index, physical_row, score_words);
            region.row_base = region.row_base + {5'b00000, score_words[7:1]};
            region.primary_port = 1'b1;
            attention_pair_context_region = region;
        end
    endfunction

    function automatic local_sram_region_t attention_pair_query_region(
        input logic [1:0] batch_index,
        input logic [5:0] physical_row
    );
        attention_pair_query_region = batch_index == 2'd2 ?
            local_region(5'd14 + {2'b00, physical_row[2:0]},
                12'd976 + {6'b000000, physical_row[5:3], 3'b000}, 1'b0) :
            local_region({2'b00, physical_row[2:0]},
                12'(ATTENTION_PAIR_QUERY_ROW_BASE) +
                (batch_index == 2'd1 ? 12'd48 : 12'd0) +
                {6'b000000, physical_row[5:3], 3'b000}, 1'b0);
    endfunction

    // Q weights remain below staging through both Q postprocessing passes.
    // K and then V reuse the panel only after the preceding readers drain.
    function automatic local_sram_region_t attention_pair_panel_region(
        input logic [1:0] stripe
    );
        attention_pair_panel_region = local_region(5'd8 + {3'b000, stripe},
            12'd0, 1'b0);
    endfunction

    function automatic local_sram_region_t attention_pair_rope_constant_region(
        input logic [1:0] batch_index
    );
        case (batch_index)
            2'd1: attention_pair_rope_constant_region =
                local_region(5'd8, 12'd1280, 1'b0);
            2'd2: attention_pair_rope_constant_region =
                local_region(5'd9, 12'd512, 1'b0);
            default: attention_pair_rope_constant_region =
                ATTENTION_PAIR_ROPE_CONSTANT_REGION;
        endcase
    endfunction

    function automatic local_sram_region_t attention_pair_head_staging_region(
        input logic [1:0] stripe
    );
        attention_pair_head_staging_region = local_region(
            5'd8 + {3'b000, stripe}, 12'd256, 1'b0);
    endfunction

    function automatic local_sram_region_t attention_pair_scale_region(
        input logic probability_scale,
        input logic [1:0] batch_index
    );
        attention_pair_scale_region = local_region(5'd12,
            batch_index == 2'd2 ?
                (probability_scale ? 12'd808 : 12'd800) :
                12'd768 + (probability_scale ? 12'd16 : 12'd0) +
                    (batch_index == 2'd1 ? 12'd8 : 12'd0), 1'b0);
    endfunction

    function automatic local_sram_region_t attention_pair_scratch_region(
        input logic stripe
    );
        attention_pair_scratch_region = local_region(
            stripe ? 5'd13 : 5'd12, 12'd1024, 1'b0);
    endfunction

    function automatic logic [5:0] local_region_port_index(
        input local_sram_region_t region,
        input logic secondary_port
    );
        local_region_port_index = {region.macro_id, 1'b0} +
            6'(region.primary_port ^ secondary_port);
    endfunction

    function automatic logic [11:0] local_region_row(
        input local_sram_region_t region,
        input logic [9:0] word_row
    );
        local_region_row = region.row_base + {2'b00, word_row};
    endfunction

    function automatic logic [9:0] attention_row_group_offset(
        input logic [1:0] row_group,
        input logic [9:0] rows_per_group
    );
        case (row_group)
            2'd0: attention_row_group_offset = 10'd0;
            2'd1: attention_row_group_offset = rows_per_group;
            2'd2: attention_row_group_offset = rows_per_group << 1;
            default: attention_row_group_offset =
                (rows_per_group << 1) + rows_per_group;
        endcase
    endfunction

    function automatic logic [12:0] local_sram_macro_depth(
        input logic [4:0] macro_id
    );
        if (macro_id < LOCAL_SRAM_2048_MACRO_COUNT)
            local_sram_macro_depth = 13'd2048;
        else if (macro_id < LOCAL_SRAM_MACRO_COUNT)
            local_sram_macro_depth = 13'd1024;
        else
            local_sram_macro_depth = 13'd0;
    endfunction

    function automatic logic [4:0] full_matmul_activation_stripe(
        input logic [14:0] word_index
    );
        full_matmul_activation_stripe = {word_index[14:13], word_index[2:0]};
    endfunction

    function automatic logic [9:0] full_matmul_activation_word_row(
        input logic [14:0] word_index
    );
        full_matmul_activation_word_row = word_index[12:3];
    endfunction

    // Six 48-token A4 batches use all rows of macros 0..21. Each adjacent
    // activation lane pair uses the two ports of one selected macro.
    function automatic logic [4:0] expanded_matmul_activation_macro(
        input logic [15:0] word_index
    );
        integer offset;
        begin
            offset = integer'(word_index >> 3);
            case (word_index[2:1])
                2'd0: begin
                    if (offset < 1024) expanded_matmul_activation_macro = 5'd0;
                    else if (offset < 2048) expanded_matmul_activation_macro = 5'd1;
                    else if (offset < 3072) expanded_matmul_activation_macro = 5'd2;
                    else if (offset < 4096) expanded_matmul_activation_macro = 5'd3;
                    else expanded_matmul_activation_macro = 5'd14;
                end
                2'd1: begin
                    if (offset < 1024) expanded_matmul_activation_macro = 5'd4;
                    else if (offset < 2048) expanded_matmul_activation_macro = 5'd5;
                    else if (offset < 3072) expanded_matmul_activation_macro = 5'd6;
                    else if (offset < 4096) expanded_matmul_activation_macro = 5'd7;
                    else expanded_matmul_activation_macro = 5'd15;
                end
                2'd2: begin
                    if (offset < 1024) expanded_matmul_activation_macro = 5'd8;
                    else if (offset < 2048) expanded_matmul_activation_macro = 5'd9;
                    else if (offset < 3072) expanded_matmul_activation_macro = 5'd10;
                    else if (offset < 3584) expanded_matmul_activation_macro = 5'd16;
                    else if (offset < 4096) expanded_matmul_activation_macro = 5'd17;
                    else expanded_matmul_activation_macro = 5'd18;
                end
                default: begin
                    if (offset < 1024) expanded_matmul_activation_macro = 5'd11;
                    else if (offset < 2048) expanded_matmul_activation_macro = 5'd12;
                    else if (offset < 3072) expanded_matmul_activation_macro = 5'd13;
                    else if (offset < 3584) expanded_matmul_activation_macro = 5'd19;
                    else if (offset < 4096) expanded_matmul_activation_macro = 5'd20;
                    else expanded_matmul_activation_macro = 5'd21;
                end
            endcase
        end
    endfunction

    function automatic logic [11:0] expanded_matmul_activation_row(
        input logic [15:0] word_index
    );
        integer offset;
        begin
            offset = integer'(word_index >> 3);
            case (word_index[2:1])
                2'd0, 2'd1: begin
                    if (offset >= 4096) offset = offset - 4096;
                    else if (offset >= 3072) offset = offset - 3072;
                    else if (offset >= 2048) offset = offset - 2048;
                    else if (offset >= 1024) offset = offset - 1024;
                end
                default: begin
                    if (offset >= 4096) offset = offset - 4096;
                    else if (offset >= 3584) offset = offset - 3584;
                    else if (offset >= 3072) offset = offset - 3072;
                    else if (offset >= 2048) offset = offset - 2048;
                    else if (offset >= 1024) offset = offset - 1024;
                end
            endcase
            expanded_matmul_activation_row = 12'((offset << 1) |
                integer'(word_index[0]));
        end
    endfunction

    function automatic local_sram_region_t expanded_matmul_panel_region(
        input logic panel,
        input logic [1:0] stripe
    );
        expanded_matmul_panel_region = local_region(
            5'(22 + integer'(stripe)), panel ? 12'd256 : 12'd0, 1'b0);
    endfunction

    // Hidden-token data retains the same physical macro and SRAM-row base from execution input
    // through Attention and all Matmul phases that preserve the residual.
    function automatic local_sram_region_t hidden_region(input logic [3:0] stripe);
        case (stripe)
            4'd0: hidden_region = local_region(5'd4, 12'd0, 1'b1);
            4'd1: hidden_region = local_region(5'd17, 12'd0, 1'b1);
            4'd2: hidden_region = local_region(5'd9, 12'd0, 1'b1);
            4'd3: hidden_region = local_region(5'd2, 12'd1024, 1'b1);
            4'd4: hidden_region = local_region(5'd13, 12'd1024, 1'b1);
            4'd5: hidden_region = local_region(5'd6, 12'd0, 1'b1);
            4'd6: hidden_region = local_region(5'd5, 12'd0, 1'b1);
            4'd7: hidden_region = local_region(5'd8, 12'd1024, 1'b1);
            4'd8: hidden_region = local_region(5'd5, 12'd1024, 1'b1);
            4'd9: hidden_region = local_region(5'd12, 12'd0, 1'b1);
            4'd10: hidden_region = local_region(5'd2, 12'd0, 1'b1);
            4'd11: hidden_region = local_region(5'd1, 12'd0, 1'b1);
            4'd12: hidden_region = local_region(5'd9, 12'd1024, 1'b1);
            4'd13: hidden_region = local_region(5'd8, 12'd0, 1'b1);
            4'd14: hidden_region = local_region(5'd6, 12'd1024, 1'b1);
            default: hidden_region = local_region(5'd3, 12'd0, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t resident_extra_region(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: resident_extra_region = local_region(5'd0, 12'd1024, 1'b0);
            3'd1: resident_extra_region = local_region(5'd12, 12'd1024, 1'b1);
            3'd2: resident_extra_region = local_region(5'd3, 12'd1024, 1'b1);
            3'd3: resident_extra_region = local_region(5'd22, 12'd0, 1'b1);
            3'd4: resident_extra_region = local_region(5'd7, 12'd1024, 1'b1);
            3'd5: resident_extra_region = local_region(5'd10, 12'd0, 1'b1);
            3'd6: resident_extra_region = local_region(5'd18, 12'd0, 1'b1);
            default: resident_extra_region = local_region(5'd23, 12'd0, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t resident_region(
        input logic [5:0] physical_row
    );
        if (physical_row < 6'd32)
            resident_region = hidden_region(physical_row[3:0]);
        else
            resident_region = resident_extra_region(physical_row[2:0]);
    endfunction

    function automatic logic [9:0] resident_word_row(
        input logic [5:0] physical_row,
        input logic [8:0] channel_word
    );
        if ((physical_row >= 6'd16 && physical_row < 6'd32) ||
            physical_row >= 6'd40)
            resident_word_row = {1'b1, channel_word};
        else
            resident_word_row = {1'b0, channel_word};
    endfunction

    function automatic logic [3:0] hidden_row_stripe(
        input logic [5:0] physical_row
    );
        hidden_row_stripe = physical_row[3:0];
    endfunction

    function automatic logic [9:0] hidden_row_word(
        input logic [5:0] physical_row,
        input logic [8:0] channel_word
    );
        hidden_row_word = resident_word_row(physical_row, channel_word);
    endfunction

    // During Attention RMSNorm these regions first contain normalized BF16.
    // After each source batch reaches last-use they contain compact activation.
    function automatic local_sram_region_t qkv_activation_region(input logic [2:0] stripe);
        case (stripe)
            3'd0: qkv_activation_region = local_region(5'd19, 12'd0, 1'b1);
            3'd1: qkv_activation_region = local_region(5'd10, 12'd1024, 1'b1);
            3'd2: qkv_activation_region = local_region(5'd7, 12'd0, 1'b1);
            3'd3: qkv_activation_region = local_region(5'd13, 12'd0, 1'b1);
            3'd4: qkv_activation_region = local_region(5'd11, 12'd1024, 1'b1);
            3'd5: qkv_activation_region = local_region(5'd24, 12'd0, 1'b1);
            3'd6: qkv_activation_region = local_region(5'd4, 12'd1024, 1'b1);
            default: qkv_activation_region = local_region(5'd1, 12'd1024, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t q_quantized_region(input logic [2:0] stripe);
        case (stripe)
            3'd0: q_quantized_region = local_region(5'd22, 12'd0, 1'b0);
            3'd1: q_quantized_region = local_region(5'd12, 12'd1024, 1'b1);
            3'd2: q_quantized_region = local_region(5'd0, 12'd1024, 1'b1);
            3'd3: q_quantized_region = local_region(5'd7, 12'd1024, 1'b1);
            3'd4: q_quantized_region = local_region(5'd10, 12'd0, 1'b1);
            3'd5: q_quantized_region = local_region(5'd11, 12'd0, 1'b1);
            3'd6: q_quantized_region = local_region(5'd3, 12'd1024, 1'b1);
            default: q_quantized_region = local_region(5'd18, 12'd0, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t q_quantized_extra_region(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: q_quantized_extra_region = local_region(5'd23, 12'd896, 1'b0);
            2'd1: q_quantized_extra_region = local_region(5'd12, 12'd0, 1'b0);
            2'd2: q_quantized_extra_region = local_region(5'd23, 12'd64, 1'b0);
            default: q_quantized_extra_region = local_region(5'd23, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t q_quantized_row_region(
        input logic [5:0] physical_row
    );
        if (physical_row < 6'd32)
            q_quantized_row_region = q_quantized_region(physical_row[2:0]);
        else
            q_quantized_row_region =
                q_quantized_extra_region(physical_row[1:0]);
    endfunction

    function automatic logic q_quantized_row_secondary_port(
        input logic [5:0] physical_row
    );
        q_quantized_row_secondary_port =
            physical_row >= 6'd32 && physical_row[2];
    endfunction

    function automatic logic [9:0] q_quantized_row_word(
        input logic [5:0] physical_row,
        input logic [7:0] row_word
    );
        if (physical_row < 6'd32)
            q_quantized_row_word =
                {physical_row[4:3], 8'b0} + {2'b00, row_word};
        else
            q_quantized_row_word =
                {physical_row[3:2], 8'b0} + {2'b00, row_word};
    endfunction

    // QKV panel storage is reused as the four-bank BF16 head staging area and
    // then retained as the two K/V ping-pong panels used by Attention.
    function automatic local_sram_region_t qkv_panel_region(input logic [1:0] stripe);
        case (stripe)
            2'd0: qkv_panel_region = local_region(5'd20, 12'd0, 1'b0);
            2'd1: qkv_panel_region = local_region(5'd21, 12'd0, 1'b0);
            2'd2: qkv_panel_region = local_region(5'd15, 12'd0, 1'b0);
            default: qkv_panel_region = local_region(5'd25, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t q_work_panel_region(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: q_work_panel_region = local_region(5'd18, 12'd0, 1'b1);
            2'd1: q_work_panel_region = local_region(5'd0, 12'd896, 1'b0);
            2'd2: q_work_panel_region = local_region(5'd22, 12'd0, 1'b1);
            default: q_work_panel_region = local_region(5'd17, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t attention_kv_panel_region(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: attention_kv_panel_region = local_region(5'd1, 12'd0, 1'b1);
            2'd1: attention_kv_panel_region = local_region(5'd10, 12'd0, 1'b1);
            2'd2: attention_kv_panel_region = local_region(5'd20, 12'd0, 1'b1);
            default: attention_kv_panel_region = local_region(5'd11, 12'd0, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t q_head_slot_region(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: q_head_slot_region = local_region(5'd23, 12'd0, 1'b0);
            3'd1: q_head_slot_region = local_region(5'd2, 12'd1792, 1'b0);
            3'd2: q_head_slot_region = local_region(5'd21, 12'd0, 1'b0);
            3'd3: q_head_slot_region = local_region(5'd3, 12'd768, 1'b0);
            3'd4: q_head_slot_region = local_region(5'd16, 12'd0, 1'b1);
            3'd5: q_head_slot_region = local_region(5'd6, 12'd0, 1'b1);
            3'd6: q_head_slot_region = local_region(5'd5, 12'd1792, 1'b0);
            default: q_head_slot_region = local_region(5'd9, 12'd1792, 1'b1);
        endcase
    endfunction

    function automatic logic [9:0] q_head_slot_word(
        input logic slot,
        input logic [5:0] physical_row,
        input logic [3:0] head_word
    );
        q_head_slot_word = 10'({slot, 6'b0}) +
            10'({physical_row[5:3], 3'b0}) + {6'b0, head_word};
    endfunction

    function automatic logic [9:0] q_head_slot_scale_word(
        input logic slot,
        input logic [5:0] physical_row
    );
        q_head_slot_scale_word = 10'({slot, 3'b0}) +
            {7'b0, physical_row[5:3]};
    endfunction

    function automatic local_sram_region_t qkv_head_source_region();
        qkv_head_source_region = local_region(5'd16, 12'd128, 1'b1);
    endfunction

    function automatic local_sram_region_t qkv_head_destination_region();
        qkv_head_destination_region = local_region(5'd23, 12'd128, 1'b0);
    endfunction

    function automatic local_sram_region_t qkv_constant_region();
        qkv_constant_region = local_region(5'd0, 12'd128, 1'b0);
    endfunction

    function automatic local_sram_region_t qkv_attention_metadata_region();
        qkv_attention_metadata_region = local_region(5'd14, 12'd512, 1'b1);
    endfunction

    // K-panel scales use a panel-specific region so the inactive panel can be
    // filled through its primary port while QK reads the active panel through
    // the secondary port. The regions occupy rows above Softmax scratch data.
    function automatic local_sram_region_t attention_panel_metadata_region(
        input logic panel
    );
        case (panel)
            1'b0: attention_panel_metadata_region =
                local_region(5'd13, 12'd1024, 1'b0);
            default: attention_panel_metadata_region =
                local_region(5'd7, 12'd1024, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t attention_score_region(input logic [2:0] stripe);
        case (stripe)
            3'd0: attention_score_region = local_region(5'd6, 12'd896, 1'b1);
            3'd1: attention_score_region = local_region(5'd15, 12'd0, 1'b0);
            3'd2: attention_score_region = local_region(5'd12, 12'd0, 1'b0);
            3'd3: attention_score_region = local_region(5'd3, 12'd896, 1'b0);
            3'd4: attention_score_region = local_region(5'd8, 12'd896, 1'b1);
            3'd5: attention_score_region = local_region(5'd9, 12'd0, 1'b1);
            3'd6: attention_score_region = local_region(5'd5, 12'd768, 1'b0);
            default: attention_score_region = local_region(5'd2, 12'd768, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t attention_score_extra_region(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: attention_score_extra_region = local_region(5'd2, 12'd0, 1'b0);
            3'd1: attention_score_extra_region = local_region(5'd8, 12'd128, 1'b1);
            3'd2: attention_score_extra_region = local_region(5'd6, 12'd128, 1'b1);
            3'd3: attention_score_extra_region = local_region(5'd25, 12'd0, 1'b0);
            3'd4: attention_score_extra_region = local_region(5'd12, 12'd1024, 1'b0);
            3'd5: attention_score_extra_region = local_region(5'd3, 12'd0, 1'b0);
            3'd6: attention_score_extra_region = local_region(5'd9, 12'd1024, 1'b1);
            default: attention_score_extra_region = local_region(5'd5, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t attention_score_row_region(
        input logic [5:0] physical_row
    );
        if (physical_row < 6'd32)
            attention_score_row_region =
                attention_score_region(physical_row[2:0]);
        else
            attention_score_row_region =
                attention_score_extra_region(physical_row[2:0]);
    endfunction

    function automatic logic [9:0] attention_score_row_word(
        input logic [5:0] physical_row,
        input logic [9:0] rows_per_group,
        input logic [7:0] key_word
    );
        if (physical_row < 6'd32)
            attention_score_row_word = attention_row_group_offset(
                physical_row[4:3], rows_per_group) + {2'b00, key_word};
        else
            attention_score_row_word =
                (physical_row[3] ? 10'd512 : 10'd0) +
                {2'b00, key_word};
    endfunction

    function automatic local_sram_region_t attention_scratch_region(input logic stripe);
        case (stripe)
            1'b0: attention_scratch_region = local_region(5'd21, 12'd128, 1'b0);
            default: attention_scratch_region = local_region(5'd8, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t attention_context_region();
        attention_context_region = local_region(5'd4, 12'd0, 1'b0);
    endfunction

    function automatic local_sram_region_t rms_scratch_region(input logic stripe);
        case (stripe)
            1'b0: rms_scratch_region = local_region(5'd14, 12'd0, 1'b0);
            default: rms_scratch_region = local_region(5'd16, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t rms_gamma_region(
        input logic ffn_command
    );
        // Gamma is read while the same RMSNorm cycle reads the source tile and
        // can write the preceding output tile. These fixed regions avoid every
        // source, output, scratch, and prefetched Matmul-panel port in that phase.
        rms_gamma_region = ffn_command ?
            local_region(5'd21, 12'd0, 1'b0) :
            local_region(5'd14, 12'd512, 1'b1);
    endfunction

    function automatic local_sram_region_t preserved_matmul_activation_region(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: preserved_matmul_activation_region = local_region(5'd20, 12'd0, 1'b1);
            3'd1: preserved_matmul_activation_region = local_region(5'd0, 12'd0, 1'b0);
            3'd2: preserved_matmul_activation_region = local_region(5'd19, 12'd0, 1'b1);
            3'd3: preserved_matmul_activation_region = local_region(5'd11, 12'd0, 1'b1);
            3'd4: preserved_matmul_activation_region = local_region(5'd21, 12'd0, 1'b1);
            3'd5: preserved_matmul_activation_region = local_region(5'd14, 12'd0, 1'b1);
            3'd6: preserved_matmul_activation_region = local_region(5'd11, 12'd1024, 1'b0);
            default: preserved_matmul_activation_region = local_region(5'd15, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t full_matmul_activation_region(
        input logic [4:0] stripe
    );
        case (stripe)
            5'd0: full_matmul_activation_region = local_region(5'd0, 12'd1024, 1'b0);
            5'd1: full_matmul_activation_region = local_region(5'd6, 12'd1024, 1'b1);
            5'd2: full_matmul_activation_region = local_region(5'd9, 12'd0, 1'b1);
            5'd3: full_matmul_activation_region = local_region(5'd0, 12'd0, 1'b1);
            5'd4: full_matmul_activation_region = local_region(5'd4, 12'd0, 1'b1);
            5'd5: full_matmul_activation_region = local_region(5'd3, 12'd0, 1'b1);
            5'd6: full_matmul_activation_region = local_region(5'd11, 12'd1024, 1'b1);
            5'd7: full_matmul_activation_region = local_region(5'd3, 12'd1024, 1'b0);
            5'd8: full_matmul_activation_region = local_region(5'd21, 12'd0, 1'b1);
            5'd9: full_matmul_activation_region = local_region(5'd20, 12'd0, 1'b1);
            5'd10: full_matmul_activation_region = local_region(5'd5, 12'd1024, 1'b1);
            5'd11: full_matmul_activation_region = local_region(5'd1, 12'd0, 1'b0);
            5'd12: full_matmul_activation_region = local_region(5'd13, 12'd0, 1'b1);
            5'd13: full_matmul_activation_region = local_region(5'd14, 12'd0, 1'b1);
            5'd14: full_matmul_activation_region = local_region(5'd2, 12'd0, 1'b1);
            5'd15: full_matmul_activation_region = local_region(5'd17, 12'd0, 1'b1);
            // A quantized row batch can span stripes 14 and 16 together.
            // Use opposite macro-2 ports for its lower and upper halves.
            5'd16: full_matmul_activation_region = local_region(5'd2, 12'd1024, 1'b0);
            5'd17: full_matmul_activation_region = local_region(5'd9, 12'd1024, 1'b1);
            5'd18: full_matmul_activation_region = local_region(5'd19, 12'd0, 1'b1);
            5'd19: full_matmul_activation_region = local_region(5'd12, 12'd1024, 1'b1);
            5'd20: full_matmul_activation_region = local_region(5'd8, 12'd1024, 1'b1);
            5'd21: full_matmul_activation_region = local_region(5'd6, 12'd0, 1'b1);
            5'd22: full_matmul_activation_region = local_region(5'd15, 12'd0, 1'b1);
            default: full_matmul_activation_region = local_region(5'd13, 12'd1024, 1'b1);
        endcase
    endfunction

    function automatic local_sram_region_t matmul_panel_region(
        input logic panel,
        input logic [2:0] stripe
    );
        if (!panel) begin
            case (stripe)
                3'd0: matmul_panel_region = local_region(5'd24, 12'd0, 1'b1);
                3'd1: matmul_panel_region = local_region(5'd16, 12'd0, 1'b1);
                3'd2: matmul_panel_region = local_region(5'd25, 12'd0, 1'b1);
                3'd3: matmul_panel_region = local_region(5'd7, 12'd0, 1'b1);
                3'd4: matmul_panel_region = local_region(5'd4, 12'd1024, 1'b0);
                default: matmul_panel_region = local_region(5'd7, 12'd1024, 1'b0);
            endcase
        end else begin
            case (stripe)
                3'd0: matmul_panel_region = local_region(5'd22, 12'd0, 1'b0);
                3'd1: matmul_panel_region = local_region(5'd10, 12'd1024, 1'b0);
                3'd2: matmul_panel_region = local_region(5'd23, 12'd0, 1'b0);
                3'd3: matmul_panel_region = local_region(5'd18, 12'd0, 1'b0);
                3'd4: matmul_panel_region = local_region(5'd1, 12'd1024, 1'b0);
                default: matmul_panel_region = local_region(5'd10, 12'd0, 1'b0);
            endcase
        end
    endfunction

    // K4096 panel 1 must remain separate from all 48 resident hidden rows.
    function automatic local_sram_region_t preserved_matmul_panel1_region(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: preserved_matmul_panel1_region = local_region(5'd13, 12'd0, 1'b1);
            2'd1: preserved_matmul_panel1_region = local_region(5'd10, 12'd1024, 1'b1);
            2'd2: preserved_matmul_panel1_region = local_region(5'd4, 12'd1024, 1'b1);
            default: preserved_matmul_panel1_region = local_region(5'd1, 12'd1024, 1'b1);
        endcase
    endfunction

    // Eight panel regions plus four idle hidden regions hold one 8-row R4 batch.
    function automatic local_sram_region_t r4_scratch_region(input logic [3:0] stripe);
        case (stripe)
            4'd0: r4_scratch_region = local_region(5'd24, 12'd0, 1'b1);
            4'd1: r4_scratch_region = local_region(5'd16, 12'd0, 1'b1);
            4'd2: r4_scratch_region = local_region(5'd25, 12'd0, 1'b1);
            4'd3: r4_scratch_region = local_region(5'd7, 12'd0, 1'b1);
            4'd4: r4_scratch_region = local_region(5'd22, 12'd0, 1'b0);
            4'd5: r4_scratch_region = local_region(5'd10, 12'd1024, 1'b0);
            4'd6: r4_scratch_region = local_region(5'd23, 12'd0, 1'b0);
            4'd7: r4_scratch_region = local_region(5'd18, 12'd0, 1'b0);
            4'd8: r4_scratch_region = local_region(5'd5, 12'd0, 1'b1);
            4'd9: r4_scratch_region = local_region(5'd8, 12'd0, 1'b1);
            4'd10: r4_scratch_region = local_region(5'd11, 12'd0, 1'b1);
            default: r4_scratch_region = local_region(5'd12, 12'd0, 1'b1);
        endcase
    endfunction

    // Before LM-head panel loading, these eight panel regions hold one local
    // 8-row BF16 tile. Each row uses 512 SRAM words for 4096 BF16 elements.
    function automatic local_sram_region_t final_output_workspace_region(
        input logic [2:0] local_row
    );
        case (local_row)
            3'd0: final_output_workspace_region = local_region(5'd24, 12'd0, 1'b1);
            3'd1: final_output_workspace_region = local_region(5'd16, 12'd0, 1'b1);
            3'd2: final_output_workspace_region = local_region(5'd25, 12'd0, 1'b1);
            3'd3: final_output_workspace_region = local_region(5'd7, 12'd0, 1'b1);
            3'd4: final_output_workspace_region = local_region(5'd22, 12'd0, 1'b0);
            3'd5: final_output_workspace_region = local_region(5'd10, 12'd1024, 1'b0);
            3'd6: final_output_workspace_region = local_region(5'd23, 12'd0, 1'b0);
            default: final_output_workspace_region = local_region(5'd18, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t candidate_state_region(
        input logic panel,
        input logic [1:0] stripe
    );
        if (!panel) begin
            case (stripe)
                2'd0: candidate_state_region = local_region(5'd24, 12'd0, 1'b1);
                2'd1: candidate_state_region = local_region(5'd16, 12'd0, 1'b1);
                2'd2: candidate_state_region = local_region(5'd25, 12'd0, 1'b1);
                default: candidate_state_region = local_region(5'd7, 12'd0, 1'b1);
            endcase
        end else begin
            case (stripe)
                2'd0: candidate_state_region = local_region(5'd22, 12'd0, 1'b0);
                2'd1: candidate_state_region = local_region(5'd10, 12'd1024, 1'b0);
                2'd2: candidate_state_region = local_region(5'd23, 12'd0, 1'b0);
                default: candidate_state_region = local_region(5'd18, 12'd0, 1'b0);
            endcase
        end
    endfunction

    function automatic logic [9:0] candidate_state_word_row(
        input logic [3:0] row_group,
        input logic row_half,
        input logic [2:0] lane_block,
        input logic [1:0] state_word
    );
        candidate_state_word_row = 10'(
            CANDIDATE_STATE_ROW_BASE + integer'(row_group) * 24 +
            integer'(row_half) * 12 + integer'(lane_block[2:1]) * 3 +
            integer'(state_word));
    endfunction

    function automatic local_sram_region_t matmul_staging_region(input logic stripe);
        case (stripe)
            1'b0: matmul_staging_region = local_region(5'd14, 12'd0, 1'b0);
            default: matmul_staging_region = local_region(5'd15, 12'd0, 1'b0);
        endcase
    endfunction

    function automatic local_sram_region_t lm_head_scale_buffer_region();
        lm_head_scale_buffer_region = local_region(5'd12, 12'd0, 1'b0);
    endfunction

    // Final norm retains up to 96 quantized rows across successive 8-row
    // tiles. Generic Matmul staging overlaps their activation stripes 13/22.
    // W8-head K4096 uses only panel stripes 0..3, leaving stripe 4 of each
    // panel free throughout normalization, quantization and head compute.
    function automatic local_sram_region_t final_norm_scratch_region(input logic bank);
        final_norm_scratch_region = matmul_panel_region(bank, 3'd4);
    endfunction

    // These tables reuse macro 12 port A only after LM-head scale reads drain.
    function automatic local_sram_region_t post_state_table_region();
        post_state_table_region = local_region(5'd12, 12'd0, 1'b0);
    endfunction

    function automatic local_sram_region_t post_suppressed_table_region();
        post_suppressed_table_region = local_region(
            5'd12, 12'(POST_SUPPRESSED_ROW_BASE), 1'b0);
    endfunction
endpackage

`default_nettype wire
