`default_nettype none

package layer_schedule_pkg;
    localparam int unsigned LAYER_OPERATOR_COUNT = 11;
    typedef enum logic [3:0] {
        LAYER_OPERATOR_ATTENTION_RMSNORM = 4'd0,
        LAYER_OPERATOR_QKV_PREPARATION = 4'd1,
        LAYER_OPERATOR_ATTENTION = 4'd2,
        LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL = 4'd3,
        LAYER_OPERATOR_FFN_RESIDUAL_SPILL = 4'd4,
        LAYER_OPERATOR_FFN_RMSNORM = 4'd5,
        LAYER_OPERATOR_FFN_GATE_MATMUL = 4'd6,
        LAYER_OPERATOR_FFN_UP_MATMUL = 4'd7,
        LAYER_OPERATOR_FFN_SILU_MULTIPLY_QUANTIZE = 4'd8,
        LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL = 4'd9,
        LAYER_OPERATOR_FFN_HIDDEN_REFILL = 4'd10
    } layer_operator_t;
endpackage

`default_nettype wire
