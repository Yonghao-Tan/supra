`timescale 1ns/1ps
`default_nettype none

// Selects one read client and one write client from the fixed schedule.
// Each selection is held until its transaction completes.
module dma_stream_router (
    input  logic clk,
    input  logic rst,
    input  logic [2:0] execution_memory_stage,
    input  logic [3:0] execution_layer_operator_index,
    input  logic [1:0] qkv_memory_stage,
    input  logic [1:0] matmul_memory_stage,
    input  logic rms_gamma_loaded,
    input  hardware_types_pkg::attention_read_source_t attention_read_source,
    output logic read_stream_idle,
    input  logic qkv_residual_spill_active,

    input  logic [2:0] execution_read_request_valid,
    output logic [2:0] execution_read_request_ready,
    input  hardware_types_pkg::dma_read_request_t execution_read_request [0:2],
    output logic [2:0] execution_read_data_valid,
    input  logic [2:0] execution_read_data_ready,
    output hardware_types_pkg::dma_read_beat_t execution_read_data [0:2],
    output hardware_types_pkg::dma_completion_t execution_read_completion [0:2],

    input  logic [6:0] operator_read_request_valid,
    output logic [6:0] operator_read_request_ready,
    input  hardware_types_pkg::dma_read_request_t operator_read_request [0:6],
    input  logic elementwise_read_second_span_valid,
    input  logic [63:0] elementwise_read_second_span_address,
    input  logic [31:0] elementwise_read_pair_stride,
    input  logic [10:0] elementwise_read_pair_count,
    input  logic matmul_aux_read_second_span_valid,
    input  logic [63:0] matmul_aux_read_second_span_address,
    input  logic [31:0] matmul_aux_read_pair_stride,
    input  logic [10:0] matmul_aux_read_pair_count,
    output logic [6:0] operator_read_data_valid,
    input  logic [6:0] operator_read_data_ready,
    output hardware_types_pkg::dma_read_beat_t operator_read_data [0:6],
    output hardware_types_pkg::dma_wide_read_beat_t operator_wide_read_data [0:6],
    output hardware_types_pkg::dma_completion_t operator_read_completion [0:6],
    output logic elementwise_read_data_span,

    input  logic [3:0] operator_write_request_valid,
    output logic [3:0] operator_write_request_ready,
    input  hardware_types_pkg::dma_write_request_t operator_write_request [0:3],
    input  logic [3:0] operator_write_data_valid,
    output logic [3:0] operator_write_data_ready,
    input  hardware_types_pkg::dma_write_beat_t operator_write_data [0:3],
    output hardware_types_pkg::dma_completion_t operator_write_completion [0:3],

    output logic memory_read_request_valid,
    input  logic memory_read_request_ready,
    output hardware_types_pkg::dma_read_request_t memory_read_request,
    output logic memory_read_second_span_valid,
    output logic [63:0] memory_read_second_span_address,
    output logic [31:0] memory_read_pair_stride,
    output logic [10:0] memory_read_pair_count,
    input  hardware_types_pkg::dma_completion_t memory_read_completion,
    input  logic memory_read_data_valid,
    output logic memory_read_data_ready,
    input  hardware_types_pkg::dma_read_beat_t memory_read_data,
    input  logic memory_read_data_span,
    input  logic memory_wide_read_data_valid,
    output logic memory_wide_read_data_ready,
    input  hardware_types_pkg::dma_wide_read_beat_t memory_wide_read_data,

    output logic memory_write_request_valid,
    input  logic memory_write_request_ready,
    output hardware_types_pkg::dma_write_request_t memory_write_request,
    input  hardware_types_pkg::dma_completion_t memory_write_completion,
    output logic memory_write_data_valid,
    input  logic memory_write_data_ready,
    output hardware_types_pkg::dma_write_beat_t memory_write_data
);
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_RMSNORM;
    import layer_schedule_pkg::LAYER_OPERATOR_QKV_PREPARATION;
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION;
    import layer_schedule_pkg::LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_RESIDUAL_SPILL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_RMSNORM;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_GATE_MATMUL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_UP_MATMUL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL;
    import layer_schedule_pkg::LAYER_OPERATOR_FFN_HIDDEN_REFILL;

    localparam logic [2:0] EXEC_MEMORY_CONFIGURATION = 3'd1;
    localparam logic [2:0] EXEC_MEMORY_METADATA = 3'd2;
    localparam logic [2:0] EXEC_MEMORY_HIDDEN_READ = 3'd3;
    localparam logic [2:0] EXEC_MEMORY_HIDDEN_WRITE = 3'd4;
    localparam logic [2:0] EXEC_MEMORY_FORWARD_POSTPROCESS = 3'd5;
    localparam logic [1:0] QKV_MEMORY_METADATA = 2'd1;
    localparam logic [1:0] QKV_MEMORY_MATMUL = 2'd2;
    localparam logic [1:0] MATMUL_MEMORY_AUXILIARY = 2'd2;

    localparam int unsigned EXEC_READ_CONFIGURATION = 0;
    localparam int unsigned EXEC_READ_METADATA = 1;
    localparam int unsigned EXEC_READ_LAYER = 2;
    localparam int unsigned OP_READ_HIDDEN = 0;
    localparam int unsigned OP_READ_RMS_GAMMA = 1;
    localparam int unsigned OP_READ_QKV_METADATA = 2;
    localparam int unsigned OP_READ_MATMUL = 3;
    localparam int unsigned OP_READ_MATMUL_AUX = 4;
    localparam int unsigned OP_READ_ATTENTION = 5;
    localparam int unsigned OP_READ_ELEMENTWISE = 6;
    localparam int unsigned OP_WRITE_HIDDEN = 0;
    localparam int unsigned OP_WRITE_QKV_CACHE = 1;
    localparam int unsigned OP_WRITE_ATTENTION = 2;
    localparam int unsigned OP_WRITE_MATMUL = 3;

    typedef enum logic [3:0] {
        READ_CLIENT_NONE,
        READ_CLIENT_CONFIGURATION,
        READ_CLIENT_METADATA,
        READ_CLIENT_LAYER,
        READ_CLIENT_HIDDEN,
        READ_CLIENT_GAMMA,
        READ_CLIENT_QKV_METADATA,
        READ_CLIENT_MATMUL,
        READ_CLIENT_MATMUL_AUX,
        READ_CLIENT_ATTENTION,
        READ_CLIENT_ELEMENTWISE
    } read_active_client_t;

    typedef enum logic [2:0] {
        WRITE_SOURCE_NONE,
        WRITE_SOURCE_HIDDEN,
        WRITE_SOURCE_QKV_CACHE,
        WRITE_SOURCE_ATTENTION,
        WRITE_SOURCE_MATMUL
    } write_source_t;

    read_active_client_t read_active_client;
    read_active_client_t stalled_read_source;
    read_active_client_t selected_read_active_client;
    write_source_t write_active_client;
    write_source_t selected_write_source;
    logic [9:0] read_request_vector;
    logic [9:0] read_allowed_mask;
    logic [9:0] read_start_vector;
    logic [3:0] write_request_vector;
    logic [3:0] write_allowed_mask;
    logic [3:0] write_start_vector;
    logic [4:0] write_request_outstanding;
    logic write_request_fire;
    logic write_completion_event;

    assign read_request_vector = {
        operator_read_request_valid,
        execution_read_request_valid
    };
    assign read_start_vector = read_request_vector & read_allowed_mask;
    assign write_request_vector = operator_write_request_valid;
    assign write_start_vector = write_request_vector & write_allowed_mask;
    assign write_request_fire = memory_write_request_valid &&
        memory_write_request_ready;
    assign write_completion_event = memory_write_completion.done_pulse ||
        memory_write_completion.error;
    assign read_stream_idle = read_active_client == READ_CLIENT_NONE && stalled_read_source == READ_CLIENT_NONE;

    // A phase can enable future producers, but only one may request in a cycle.
    always_comb begin
        read_allowed_mask = 10'd0;
        case (execution_memory_stage)
            EXEC_MEMORY_CONFIGURATION:
                read_allowed_mask[0] = 1'b1;
            EXEC_MEMORY_METADATA:
                read_allowed_mask[1] = 1'b1;
            EXEC_MEMORY_HIDDEN_READ:
                read_allowed_mask[3] = 1'b1;
            EXEC_MEMORY_FORWARD_POSTPROCESS: begin
                read_allowed_mask[3] = 1'b1;
                read_allowed_mask[5] = 1'b1;
                read_allowed_mask[7] = 1'b1;
            end
            default: begin
                read_allowed_mask[2] = 1'b1;
                case (execution_layer_operator_index)
                    LAYER_OPERATOR_ATTENTION_RMSNORM,
                    LAYER_OPERATOR_FFN_RMSNORM: begin
                        if (rms_gamma_loaded)
                            read_allowed_mask[6] = 1'b1;
                        else
                            read_allowed_mask[4] = 1'b1;
                    end
                    LAYER_OPERATOR_QKV_PREPARATION: begin
                        if (qkv_memory_stage == QKV_MEMORY_METADATA)
                            read_allowed_mask[5] = 1'b1;
                        else if (qkv_memory_stage == QKV_MEMORY_MATMUL) begin
                            if (matmul_memory_stage == MATMUL_MEMORY_AUXILIARY)
                                read_allowed_mask[7] = 1'b1;
                            else
                                read_allowed_mask[6] = 1'b1;
                        end
                    end
                    LAYER_OPERATOR_ATTENTION: begin
                        if (qkv_memory_stage == QKV_MEMORY_METADATA) begin
                            read_allowed_mask[5] = 1'b1;
                        end else if (attention_read_source ==
                            hardware_types_pkg::ATTENTION_READ_Q_MATMUL) begin
                            if (matmul_memory_stage == MATMUL_MEMORY_AUXILIARY)
                                read_allowed_mask[7] = 1'b1;
                            else
                                read_allowed_mask[6] = 1'b1;
                        end else if (attention_read_source ==
                                     hardware_types_pkg::ATTENTION_READ_CACHE) begin
                            read_allowed_mask[8] = 1'b1;
                        end
                    end
                    LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
                    LAYER_OPERATOR_FFN_GATE_MATMUL,
                    LAYER_OPERATOR_FFN_UP_MATMUL,
                    LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL: begin
                        if (matmul_memory_stage == MATMUL_MEMORY_AUXILIARY)
                            read_allowed_mask[7] = 1'b1;
                        else
                            read_allowed_mask[6] = 1'b1;
                    end
                    LAYER_OPERATOR_FFN_HIDDEN_REFILL:
                        read_allowed_mask[3] = 1'b1;
                    LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE: begin
                        read_allowed_mask[6] = 1'b1;
                        read_allowed_mask[9] = 1'b1;
                    end
                    default:
                        read_allowed_mask[6] = 1'b1;
                endcase
            end
        endcase
    end

    always_comb begin
        selected_read_active_client = READ_CLIENT_NONE;
        if (read_active_client == READ_CLIENT_NONE) begin
            if (stalled_read_source != READ_CLIENT_NONE)
                selected_read_active_client = stalled_read_source;
            else begin
            case (read_start_vector)
                10'b0000000001: selected_read_active_client = READ_CLIENT_CONFIGURATION;
                10'b0000000010: selected_read_active_client = READ_CLIENT_METADATA;
                10'b0000000100: selected_read_active_client = READ_CLIENT_LAYER;
                10'b0000001000: selected_read_active_client = READ_CLIENT_HIDDEN;
                10'b0000010000: selected_read_active_client = READ_CLIENT_GAMMA;
                10'b0000100000: selected_read_active_client = READ_CLIENT_QKV_METADATA;
                10'b0001000000: selected_read_active_client = READ_CLIENT_MATMUL;
                10'b0010000000: selected_read_active_client = READ_CLIENT_MATMUL_AUX;
                10'b0100000000: selected_read_active_client = READ_CLIENT_ATTENTION;
                10'b1000000000: selected_read_active_client = READ_CLIENT_ELEMENTWISE;
                default: begin end
            endcase
            end
        end
    end

    always_comb begin
        memory_read_request_valid = 1'b0;
        memory_read_request = '0;
        memory_read_second_span_valid = 1'b0;
        memory_read_second_span_address = '0;
        memory_read_pair_stride = '0;
        memory_read_pair_count = 11'd1;
        execution_read_request_ready = '0;
        operator_read_request_ready = '0;

        case (selected_read_active_client)
            READ_CLIENT_CONFIGURATION: begin
                memory_read_request_valid =
                    execution_read_request_valid[EXEC_READ_CONFIGURATION];
                memory_read_request = execution_read_request[EXEC_READ_CONFIGURATION];
                execution_read_request_ready[EXEC_READ_CONFIGURATION] =
                    memory_read_request_ready;
            end
            READ_CLIENT_METADATA: begin
                memory_read_request_valid =
                    execution_read_request_valid[EXEC_READ_METADATA];
                memory_read_request = execution_read_request[EXEC_READ_METADATA];
                execution_read_request_ready[EXEC_READ_METADATA] =
                    memory_read_request_ready;
            end
            READ_CLIENT_LAYER: begin
                memory_read_request_valid =
                    execution_read_request_valid[EXEC_READ_LAYER];
                memory_read_request = execution_read_request[EXEC_READ_LAYER];
                execution_read_request_ready[EXEC_READ_LAYER] =
                    memory_read_request_ready;
            end
            READ_CLIENT_HIDDEN: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_HIDDEN];
                memory_read_request = operator_read_request[OP_READ_HIDDEN];
                operator_read_request_ready[OP_READ_HIDDEN] =
                    memory_read_request_ready;
            end
            READ_CLIENT_GAMMA: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_RMS_GAMMA];
                memory_read_request = operator_read_request[OP_READ_RMS_GAMMA];
                operator_read_request_ready[OP_READ_RMS_GAMMA] =
                    memory_read_request_ready;
            end
            READ_CLIENT_QKV_METADATA: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_QKV_METADATA];
                memory_read_request = operator_read_request[OP_READ_QKV_METADATA];
                operator_read_request_ready[OP_READ_QKV_METADATA] =
                    memory_read_request_ready;
            end
            READ_CLIENT_MATMUL: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_MATMUL];
                memory_read_request = operator_read_request[OP_READ_MATMUL];
                operator_read_request_ready[OP_READ_MATMUL] =
                    memory_read_request_ready;
            end
            READ_CLIENT_MATMUL_AUX: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_MATMUL_AUX];
                memory_read_request = operator_read_request[OP_READ_MATMUL_AUX];
                memory_read_second_span_valid =
                    matmul_aux_read_second_span_valid;
                memory_read_second_span_address =
                    matmul_aux_read_second_span_address;
                memory_read_pair_stride = matmul_aux_read_pair_stride;
                memory_read_pair_count = matmul_aux_read_pair_count;
                operator_read_request_ready[OP_READ_MATMUL_AUX] =
                    memory_read_request_ready;
            end
            READ_CLIENT_ATTENTION: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_ATTENTION];
                memory_read_request = operator_read_request[OP_READ_ATTENTION];
                operator_read_request_ready[OP_READ_ATTENTION] =
                    memory_read_request_ready;
            end
            READ_CLIENT_ELEMENTWISE: begin
                memory_read_request_valid =
                    operator_read_request_valid[OP_READ_ELEMENTWISE];
                memory_read_request = operator_read_request[OP_READ_ELEMENTWISE];
                memory_read_second_span_valid =
                    elementwise_read_second_span_valid;
                memory_read_second_span_address =
                    elementwise_read_second_span_address;
                memory_read_pair_stride = elementwise_read_pair_stride;
                memory_read_pair_count = elementwise_read_pair_count;
                operator_read_request_ready[OP_READ_ELEMENTWISE] =
                    memory_read_request_ready;
            end
            default: begin end
        endcase
    end

    always_comb begin
        execution_read_data_valid = '0;
        operator_read_data_valid = '0;
        memory_read_data_ready = 1'b0;
        memory_wide_read_data_ready = 1'b0;
        elementwise_read_data_span = 1'b0;
        for (integer client = 0; client < 3; client++) begin
            execution_read_data[client] = '0;
            execution_read_completion[client] = '0;
        end
        for (integer client = 0; client < 7; client++) begin
            operator_read_data[client] = '0;
            operator_wide_read_data[client] = '0;
            operator_read_completion[client] = '0;
        end

        case (read_active_client)
            READ_CLIENT_CONFIGURATION: begin
                execution_read_data_valid[EXEC_READ_CONFIGURATION] = memory_read_data_valid;
                execution_read_data[EXEC_READ_CONFIGURATION] = memory_read_data;
                execution_read_completion[EXEC_READ_CONFIGURATION] = memory_read_completion;
                memory_read_data_ready = execution_read_data_ready[EXEC_READ_CONFIGURATION];
            end
            READ_CLIENT_METADATA: begin
                execution_read_data_valid[EXEC_READ_METADATA] = memory_read_data_valid;
                execution_read_data[EXEC_READ_METADATA] = memory_read_data;
                execution_read_completion[EXEC_READ_METADATA] = memory_read_completion;
                memory_read_data_ready = execution_read_data_ready[EXEC_READ_METADATA];
            end
            READ_CLIENT_LAYER: begin
                execution_read_data_valid[EXEC_READ_LAYER] = memory_read_data_valid;
                execution_read_data[EXEC_READ_LAYER] = memory_read_data;
                execution_read_completion[EXEC_READ_LAYER] = memory_read_completion;
                memory_read_data_ready = execution_read_data_ready[EXEC_READ_LAYER];
            end
            READ_CLIENT_HIDDEN: begin
                operator_read_data_valid[OP_READ_HIDDEN] = memory_read_data_valid;
                operator_read_data[OP_READ_HIDDEN] = memory_read_data;
                operator_read_completion[OP_READ_HIDDEN] = memory_read_completion;
                memory_read_data_ready = operator_read_data_ready[OP_READ_HIDDEN];
            end
            READ_CLIENT_GAMMA: begin
                operator_read_data_valid[OP_READ_RMS_GAMMA] = memory_read_data_valid;
                operator_read_data[OP_READ_RMS_GAMMA] = memory_read_data;
                operator_read_completion[OP_READ_RMS_GAMMA] = memory_read_completion;
                memory_read_data_ready = operator_read_data_ready[OP_READ_RMS_GAMMA];
            end
            READ_CLIENT_QKV_METADATA: begin
                operator_read_data_valid[OP_READ_QKV_METADATA] = memory_read_data_valid;
                operator_read_data[OP_READ_QKV_METADATA] = memory_read_data;
                operator_read_completion[OP_READ_QKV_METADATA] = memory_read_completion;
                memory_read_data_ready = operator_read_data_ready[OP_READ_QKV_METADATA];
            end
            READ_CLIENT_MATMUL: begin
                operator_read_data_valid[OP_READ_MATMUL] = memory_wide_read_data_valid;
                operator_wide_read_data[OP_READ_MATMUL] = memory_wide_read_data;
                operator_read_completion[OP_READ_MATMUL] = memory_read_completion;
                memory_wide_read_data_ready = operator_read_data_ready[OP_READ_MATMUL];
            end
            READ_CLIENT_MATMUL_AUX: begin
                operator_read_data_valid[OP_READ_MATMUL_AUX] = memory_read_data_valid;
                operator_read_data[OP_READ_MATMUL_AUX] = memory_read_data;
                operator_read_completion[OP_READ_MATMUL_AUX] = memory_read_completion;
                memory_read_data_ready = operator_read_data_ready[OP_READ_MATMUL_AUX];
            end
            READ_CLIENT_ATTENTION: begin
                operator_read_data_valid[OP_READ_ATTENTION] = memory_wide_read_data_valid;
                operator_wide_read_data[OP_READ_ATTENTION] = memory_wide_read_data;
                operator_read_completion[OP_READ_ATTENTION] = memory_read_completion;
                memory_wide_read_data_ready = operator_read_data_ready[OP_READ_ATTENTION];
            end
            READ_CLIENT_ELEMENTWISE: begin
                operator_read_data_valid[OP_READ_ELEMENTWISE] = memory_read_data_valid;
                operator_read_data[OP_READ_ELEMENTWISE] = memory_read_data;
                operator_read_completion[OP_READ_ELEMENTWISE] = memory_read_completion;
                elementwise_read_data_span = memory_read_data_span;
                memory_read_data_ready = operator_read_data_ready[OP_READ_ELEMENTWISE];
            end
            default: begin end
        endcase
    end

    always_comb begin
        write_allowed_mask = 4'd0;
        if (execution_memory_stage == EXEC_MEMORY_HIDDEN_WRITE) begin
            write_allowed_mask[OP_WRITE_HIDDEN] = 1'b1;
        end else if (execution_memory_stage == EXEC_MEMORY_FORWARD_POSTPROCESS) begin
            write_allowed_mask[OP_WRITE_MATMUL] = 1'b1;
        end else if (qkv_residual_spill_active) begin
            write_allowed_mask[OP_WRITE_HIDDEN] = 1'b1;
        end else begin
            case (execution_layer_operator_index)
                LAYER_OPERATOR_FFN_RESIDUAL_SPILL:
                    write_allowed_mask[OP_WRITE_HIDDEN] = 1'b1;
                LAYER_OPERATOR_QKV_PREPARATION: begin
                    write_allowed_mask[OP_WRITE_QKV_CACHE] = 1'b1;
                    write_allowed_mask[OP_WRITE_MATMUL] = 1'b1;
                end
                LAYER_OPERATOR_ATTENTION: begin
                    write_allowed_mask[OP_WRITE_ATTENTION] = 1'b1;
                    write_allowed_mask[OP_WRITE_MATMUL] = 1'b1;
                end
                LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL,
                LAYER_OPERATOR_FFN_GATE_MATMUL,
                LAYER_OPERATOR_FFN_UP_MATMUL,
                LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL:
                    write_allowed_mask[OP_WRITE_MATMUL] = 1'b1;
                default: begin end
            endcase
        end
    end

    always_comb begin
        selected_write_source = write_active_client;
        if (write_active_client == WRITE_SOURCE_NONE) begin
            case (write_start_vector)
                4'b0001: selected_write_source = WRITE_SOURCE_HIDDEN;
                4'b0010: selected_write_source = WRITE_SOURCE_QKV_CACHE;
                4'b0100: selected_write_source = WRITE_SOURCE_ATTENTION;
                4'b1000: selected_write_source = WRITE_SOURCE_MATMUL;
                default: selected_write_source = WRITE_SOURCE_NONE;
            endcase
        end
    end

    always_comb begin
        memory_write_request_valid = 1'b0;
        memory_write_request = '0;
        memory_write_data_valid = 1'b0;
        memory_write_data = '0;
        operator_write_request_ready = '0;
        operator_write_data_ready = '0;

        case (selected_write_source)
            WRITE_SOURCE_HIDDEN: begin
                memory_write_request_valid = operator_write_request_valid[OP_WRITE_HIDDEN];
                memory_write_request = operator_write_request[OP_WRITE_HIDDEN];
                memory_write_data_valid = operator_write_data_valid[OP_WRITE_HIDDEN];
                memory_write_data = operator_write_data[OP_WRITE_HIDDEN];
                operator_write_request_ready[OP_WRITE_HIDDEN] = memory_write_request_ready;
                operator_write_data_ready[OP_WRITE_HIDDEN] = memory_write_data_ready;
            end
            WRITE_SOURCE_QKV_CACHE: begin
                memory_write_request_valid = operator_write_request_valid[OP_WRITE_QKV_CACHE];
                memory_write_request = operator_write_request[OP_WRITE_QKV_CACHE];
                memory_write_data_valid = operator_write_data_valid[OP_WRITE_QKV_CACHE];
                memory_write_data = operator_write_data[OP_WRITE_QKV_CACHE];
                operator_write_request_ready[OP_WRITE_QKV_CACHE] = memory_write_request_ready;
                operator_write_data_ready[OP_WRITE_QKV_CACHE] = memory_write_data_ready;
            end
            WRITE_SOURCE_ATTENTION: begin
                memory_write_request_valid = operator_write_request_valid[OP_WRITE_ATTENTION];
                memory_write_request = operator_write_request[OP_WRITE_ATTENTION];
                memory_write_data_valid = operator_write_data_valid[OP_WRITE_ATTENTION];
                memory_write_data = operator_write_data[OP_WRITE_ATTENTION];
                operator_write_request_ready[OP_WRITE_ATTENTION] = memory_write_request_ready;
                operator_write_data_ready[OP_WRITE_ATTENTION] = memory_write_data_ready;
            end
            WRITE_SOURCE_MATMUL: begin
                memory_write_request_valid = operator_write_request_valid[OP_WRITE_MATMUL];
                memory_write_request = operator_write_request[OP_WRITE_MATMUL];
                memory_write_data_valid = operator_write_data_valid[OP_WRITE_MATMUL];
                memory_write_data = operator_write_data[OP_WRITE_MATMUL];
                operator_write_request_ready[OP_WRITE_MATMUL] = memory_write_request_ready;
                operator_write_data_ready[OP_WRITE_MATMUL] = memory_write_data_ready;
            end
            default: begin end
        endcase
    end

    // Responses follow the registered client, independent of new requests.
    always_comb begin
        for (integer client = 0; client < 4; client++)
            operator_write_completion[client] = '0;
        case (write_active_client)
            WRITE_SOURCE_HIDDEN:
                operator_write_completion[OP_WRITE_HIDDEN] = memory_write_completion;
            WRITE_SOURCE_QKV_CACHE:
                operator_write_completion[OP_WRITE_QKV_CACHE] = memory_write_completion;
            WRITE_SOURCE_ATTENTION:
                operator_write_completion[OP_WRITE_ATTENTION] = memory_write_completion;
            WRITE_SOURCE_MATMUL:
                operator_write_completion[OP_WRITE_MATMUL] = memory_write_completion;
            default: begin end
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            read_active_client <= READ_CLIENT_NONE;
            stalled_read_source <= READ_CLIENT_NONE;
            write_active_client <= WRITE_SOURCE_NONE;
            write_request_outstanding <= 5'd0;
        end else begin
            if (memory_read_request_valid && !memory_read_request_ready)
                stalled_read_source <= selected_read_active_client;
            else if (memory_read_request_valid && memory_read_request_ready)
                stalled_read_source <= READ_CLIENT_NONE;
            if (read_active_client == READ_CLIENT_NONE &&
                memory_read_request_valid && memory_read_request_ready)
                read_active_client <= selected_read_active_client;
            else if (read_active_client != READ_CLIENT_NONE &&
                     (memory_read_completion.done_pulse || memory_read_completion.error))
                read_active_client <= READ_CLIENT_NONE;

            case ({write_request_fire, write_completion_event})
                2'b10: begin
                    if (write_active_client == WRITE_SOURCE_NONE)
                        write_active_client <= selected_write_source;
                    write_request_outstanding <= write_request_outstanding + 5'd1;
                end
                2'b01: begin
                    write_request_outstanding <= write_request_outstanding - 5'd1;
                    if (write_request_outstanding == 5'd1)
                        write_active_client <= WRITE_SOURCE_NONE;
                end
                default: begin end
            endcase
        end
    end

`ifndef SYNTHESIS
    logic stalled_memory_read_request;
    hardware_types_pkg::dma_read_request_t held_memory_read_request;
    logic held_memory_second_span_valid;
    logic [63:0] held_memory_second_span_address;
    logic [31:0] held_memory_pair_stride;
    logic [10:0] held_memory_pair_count;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_memory_read_request <= 1'b0;
        end else begin
            if (stalled_memory_read_request)
                assert (memory_read_request_valid &&
                        memory_read_request == held_memory_read_request &&
                        memory_read_second_span_valid ==
                            held_memory_second_span_valid &&
                        memory_read_second_span_address ==
                            held_memory_second_span_address &&
                        memory_read_pair_stride == held_memory_pair_stride &&
                        memory_read_pair_count == held_memory_pair_count)
                    else $error("DMA read router changed a stalled memory request");
            stalled_memory_read_request <= memory_read_request_valid &&
                !memory_read_request_ready;
            held_memory_read_request <= memory_read_request;
            held_memory_second_span_valid <= memory_read_second_span_valid;
            held_memory_second_span_address <= memory_read_second_span_address;
            held_memory_pair_stride <= memory_read_pair_stride;
            held_memory_pair_count <= memory_read_pair_count;
            if (read_active_client != READ_CLIENT_NONE)
                assert (selected_read_active_client == READ_CLIENT_NONE)
                    else $error("DMA read router selected a client while another read was active");
            if (read_active_client == READ_CLIENT_NONE)
                assert ($onehot0(read_start_vector))
                    else $error("DMA read router concurrent requests request=%b allowed=%b start=%b command=%0d memory_stage=%0d qkv_stage=%0d matmul_stage=%0d",
                        read_request_vector, read_allowed_mask,
                        read_start_vector, execution_layer_operator_index,
                        execution_memory_stage, qkv_memory_stage,
                        matmul_memory_stage);
            if (write_active_client != WRITE_SOURCE_NONE)
                assert (selected_write_source == write_active_client)
                    else $error("DMA write router changed active client before completion");
            if (write_active_client == WRITE_SOURCE_NONE)
                assert ($onehot0(write_start_vector))
                    else $error("DMA write router concurrent requests request=%b allowed=%b start=%b command=%0d memory_stage=%0d",
                        write_request_vector, write_allowed_mask,
                        write_start_vector, execution_layer_operator_index,
                        execution_memory_stage);
            assert ((write_active_client == WRITE_SOURCE_NONE) ==
                    (write_request_outstanding == 5'd0))
                else $error("DMA write router active client and outstanding count disagree");
            assert (!(write_completion_event && write_request_outstanding == 5'd0))
                else $error("DMA write router received completion without an outstanding request");
            assert (!(write_request_fire && write_request_outstanding == 5'd31))
                else $error("DMA write router outstanding counter overflow");
        end
    end
`endif
endmodule

`default_nettype wire
