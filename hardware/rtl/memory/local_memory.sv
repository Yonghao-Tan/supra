`default_nettype none

// The 640 KiB local memory consists of 14 2048x128 and 12 1024x128
// true-dual-port SRAM macros. Every production client enters through a fixed
// logical bundle. A phase changes only after requests and responses drain.
module local_memory (
    input  logic          clk,
    input  logic          rst,

    input  logic          layout_update_valid,
    output logic          layout_update_ready,
    input  local_memory_layout_pkg::layout_requirement_t next_layout,
    output local_memory_layout_pkg::layout_requirement_t current_layout,
    input  logic          qkv_matmul_overlap_active,

    input  logic [5:0]    active_row_count,
    input  logic          rms_active,
    input  logic          rms_ffn_command,
    input  logic          attention_normalized_capture,
    input  logic          rms_gamma_bypass,
    input  logic          rms_gamma_stage_valid,
    output logic          rms_gamma_stage_ready,
    input  hardware_types_pkg::rms_gamma_write_t rms_gamma_write,
    input  logic          rms_tile_read_req_valid,
    output logic          rms_tile_read_req_ready,
    input  hardware_types_pkg::rms_tile_read_request_t rms_tile_read_req,
    output logic          rms_tile_read_rsp_valid,
    input  logic          rms_tile_read_rsp_ready,
    output hardware_types_pkg::rms_tile_read_response_t rms_tile_read_rsp,
    input  logic          rms_scratch_read_valid,
    output logic          rms_scratch_read_ready,
    input  hardware_types_pkg::local_pair_read_request_t rms_scratch_read_req,
    output logic          rms_scratch_read_rsp_valid,
    input  logic          rms_scratch_read_rsp_ready,
    output hardware_types_pkg::local_pair_read_response_t rms_scratch_read_rsp,
    input  logic          rms_scratch_write_valid,
    output logic          rms_scratch_write_ready,
    input  hardware_types_pkg::local_banked_write_t rms_scratch_write,
    input  logic          refresh_metadata_scratch_req_valid,
    output logic          refresh_metadata_scratch_req_ready,
    input  logic          refresh_metadata_scratch_write,
    input  logic [8:0]    refresh_metadata_scratch_address,
    input  logic [58:0]   refresh_metadata_scratch_write_data,
    input  logic [58:0]   refresh_metadata_scratch_write_enable,
    output logic          refresh_metadata_scratch_rsp_valid,
    output logic [58:0]   refresh_metadata_scratch_rsp_data,
    input  logic          refresh_metadata_scratch_aux_write_valid,
    output logic          refresh_metadata_scratch_aux_write_ready,
    input  logic [8:0]    refresh_metadata_scratch_aux_write_address,
    input  logic [19:0]   refresh_metadata_scratch_aux_write_data,
    input  logic [19:0]   refresh_metadata_scratch_aux_write_enable,
    input  logic          rms_norm_write_valid,
    output logic          rms_norm_write_ready,
    input  hardware_types_pkg::rms_norm_write_t rms_norm_write,

    input  logic          hidden_write_valid,
    output logic          hidden_write_ready,
    input  hardware_types_pkg::hidden_local_write_t hidden_write_req,
    input  logic          hidden_read_valid,
    output logic          hidden_read_ready,
    input  hardware_types_pkg::hidden_local_read_request_t hidden_read_req,
    output logic          hidden_read_response_valid,
    output hardware_types_pkg::local_128b_read_response_t hidden_read_rsp,

    input  logic          context_write_valid,
    output logic          context_write_ready,
    input  hardware_types_pkg::local_128b_write_t context_write_req,
    input  logic          context_read_req_valid,
    output logic          context_read_req_ready,
    input  hardware_types_pkg::tagged_local_read_request_t context_read_req,
    output logic          context_read_rsp_valid,
    input  logic          context_read_rsp_ready,
    output hardware_types_pkg::tagged_local_read_response_t context_read_rsp,

    input  logic          attention_scratch_read_valid,
    output logic          attention_scratch_read_ready,
    input  hardware_types_pkg::local_pair_read_request_t
        attention_scratch_read_req,
    output logic          attention_scratch_rsp_valid,
    input  logic          attention_scratch_rsp_ready,
    output hardware_types_pkg::local_pair_read_response_t attention_scratch_rsp,
    input  logic          attention_scratch_write_valid,
    output logic          attention_scratch_write_ready,
    input  hardware_types_pkg::local_banked_write_t attention_scratch_write_req,

    input  logic          qkv_head_read_req_valid,
    output logic          qkv_head_read_req_ready,
    input  hardware_types_pkg::qkv_head_read_request_t qkv_head_read_req,
    output logic          qkv_head_read_rsp_valid,
    input  logic          qkv_head_read_rsp_ready,
    output hardware_types_pkg::qkv_head_read_response_t qkv_head_read_rsp,

    input  logic          rope_source_read_req_valid,
    output logic          rope_source_read_req_ready,
    input  hardware_types_pkg::rope_source_read_request_t rope_source_read_req,
    output logic          rope_source_read_rsp_valid,
    input  logic          rope_source_read_rsp_ready,
    output hardware_types_pkg::rope_source_read_response_t rope_source_read_rsp,
    input  logic          rope_cos_read_req_valid,
    output logic          rope_cos_read_req_ready,
    input  hardware_types_pkg::rope_constant_read_request_t
        rope_constant_read_req,
    output logic          rope_cos_read_rsp_valid,
    input  logic          rope_cos_read_rsp_ready,
    output hardware_types_pkg::rope_constant_read_response_t rope_cos_read_rsp,
    input  logic          rope_sin_read_req_valid,
    output logic          rope_sin_read_req_ready,
    output logic          rope_sin_read_rsp_valid,
    input  logic          rope_sin_read_rsp_ready,
    output hardware_types_pkg::rope_constant_read_response_t rope_sin_read_rsp,
    input  logic          rope_destination_write_valid,
    output logic          rope_destination_write_ready,
    input  hardware_types_pkg::rope_destination_write_t rope_destination_write,

    input  logic          qkv_head_stage_write_valid,
    output logic          qkv_head_stage_write_ready,
    input  hardware_types_pkg::qkv_head_stage_write_t qkv_head_stage_write,
    input  logic          qkv_head_tile_read_req_valid,
    output logic          qkv_head_tile_read_req_ready,
    input  hardware_types_pkg::qkv_head_tile_read_request_t
        qkv_head_tile_read_req,
    output logic          qkv_head_tile_read_rsp_valid,
    output hardware_types_pkg::qkv_head_tile_read_response_t
        qkv_head_tile_read_rsp,

    input  logic          qkv_constant_write_valid,
    output logic          qkv_constant_write_ready,
    input  hardware_types_pkg::qkv_constant_write_t qkv_constant_write,
    input  logic          qkv_q_write_valid,
    output logic          qkv_q_write_ready,
    input  hardware_types_pkg::qkv_q_write_t qkv_q_write,
    input  logic          attention_panel_write_valid,
    output logic          attention_panel_write_ready,
    input  hardware_types_pkg::attention_panel_write_t attention_panel_write,
    input  logic          attention_panel_scale_write_valid,
    output logic          attention_panel_scale_write_ready,
    input  hardware_types_pkg::attention_panel_scale_write_t
        attention_panel_scale_write,

    input  logic          attention_operand_req_valid,
    output logic          attention_operand_req_ready,
    input  hardware_types_pkg::attention_operand_request_t attention_operand_req,
    output logic          attention_operand_rsp_valid,
    output hardware_types_pkg::attention_operand_response_t
        attention_operand_rsp,
    input  logic          attention_operand_abort_request,
    output logic          attention_operand_abort_ack,

    input  logic          attention_score_write_valid,
    output logic          attention_score_write_ready,
    input  hardware_types_pkg::attention_score_write_t attention_score_write,
    input  logic          attention_score_read_req_valid,
    output logic          attention_score_read_req_ready,
    input  logic          attention_score_read_slot_ready,
    input  hardware_types_pkg::attention_score_read_request_t
        attention_score_read_req,
    output logic          attention_score_read_rsp_valid,
    output hardware_types_pkg::attention_score_read_response_t
        attention_score_read_rsp,

    input  logic          attention_probability_write_valid,
    output logic          attention_probability_write_ready,
    input  hardware_types_pkg::attention_probability_write_t
        attention_probability_write,
    input  logic          attention_probability_quantized_write_valid,
    output logic          attention_probability_quantized_write_ready,
    input  hardware_types_pkg::attention_probability_quantized_write_t
        attention_probability_quantized_write,
    input  logic          attention_probability_scale_write_valid,
    output logic          attention_probability_scale_write_ready,
    input  hardware_types_pkg::attention_probability_scale_write_t
        attention_probability_scale_write,

    input  logic          matmul_local_source_req_valid,
    output logic          matmul_local_source_req_ready,
    input  hardware_types_pkg::matmul_source_request_t matmul_local_source_req,
    output logic          matmul_local_source_rsp_valid,
    input  logic          matmul_local_source_rsp_ready,
    output hardware_types_pkg::matmul_source_response_t matmul_local_source_rsp,
    input  logic          matmul_activation_write_valid,
    output logic          matmul_activation_write_ready,
    input  hardware_types_pkg::matmul_activation_write_t matmul_activation_write,
    input  logic          matmul_read_bundle_valid,
    output logic          matmul_read_bundle_ready,
    input  hardware_types_pkg::matmul_read_request_t matmul_read_req,
    output logic          matmul_read_response_valid,
    output hardware_types_pkg::matmul_read_response_t matmul_read_rsp,
    input  logic          matmul_panel_write_valid,
    output logic          matmul_panel_write_ready,
    input  hardware_types_pkg::matmul_panel_write_t matmul_panel_write,
    input  logic          lm_head_scale_write_valid,
    output logic          lm_head_scale_write_ready,
    input  logic [6:0]    lm_head_scale_write_address,
    input  logic [127:0]  lm_head_scale_write_data,
    input  logic          lm_head_scale_read_valid,
    output logic          lm_head_scale_read_ready,
    input  logic [6:0]    lm_head_scale_read_address,
    output logic          lm_head_scale_read_rsp_valid,
    output logic [127:0]  lm_head_scale_read_rsp_data,
    input  logic          post_table_access_valid,
    output logic          post_table_access_ready,
    input  logic          post_table_access_enable,
    input  logic          post_state_write_valid,
    output logic          post_state_write_ready,
    input  logic [9:0]    post_state_write_word_address,
    input  logic [127:0]  post_state_write_data,
    input  logic          post_state_read_valid,
    output logic          post_state_read_ready,
    input  logic [9:0]    post_state_read_word_address,
    input  logic [15:0]   post_state_read_tag,
    output logic          post_state_read_rsp_valid,
    output logic [127:0]  post_state_read_rsp_data,
    output logic [15:0]   post_state_read_rsp_tag,
    input  logic          post_suppressed_write_valid,
    output logic          post_suppressed_write_ready,
    input  logic [5:0]    post_suppressed_write_address,
    input  logic [127:0]  post_suppressed_write_data,
    input  logic          post_suppressed_read_valid,
    output logic          post_suppressed_read_ready,
    input  logic [7:0]    post_suppressed_read_index,
    output logic          post_suppressed_read_rsp_valid,
    output logic [16:0]   post_suppressed_read_rsp_token_id,
    input  logic          candidate_state_read_valid,
    output logic          candidate_state_read_ready,
    input  logic [3:0]    candidate_state_read_row_group,
    input  logic [2:0]    candidate_state_read_lane_block,
    input  logic          candidate_state_read_row_half,
    input  logic [1:0]    candidate_state_read_word,
    input  logic [3:0]    candidate_state_read_row_mask,
    input  logic [15:0]   candidate_state_read_tag,
    output logic          candidate_state_read_rsp_valid,
    input  logic          candidate_state_read_rsp_ready,
    output logic [511:0]  candidate_state_read_rsp_data,
    output logic [3:0]    candidate_state_read_rsp_row_mask,
    output logic [15:0]   candidate_state_read_rsp_tag,
    input  logic          candidate_state_write_valid,
    output logic          candidate_state_write_ready,
    input  logic [3:0]    candidate_state_write_row_group,
    input  logic [2:0]    candidate_state_write_lane_block,
    input  logic          candidate_state_write_row_half,
    input  logic [1:0]    candidate_state_write_word,
    input  logic [3:0]    candidate_state_write_row_mask,
    input  logic [511:0]  candidate_state_write_data,
    input  logic          matmul_combine_write_valid,
    output logic          matmul_combine_write_ready,
    input  hardware_types_pkg::local_128b_write_t matmul_combine_write,
    input  logic          matmul_combine_read_valid,
    output logic          matmul_combine_read_ready,
    input  hardware_types_pkg::local_128b_read_request_t matmul_combine_read_req,
    output logic          matmul_combine_read_response_valid,
    output hardware_types_pkg::local_128b_read_response_t matmul_combine_read_rsp,
    input  logic          matmul_local_output_write_valid,
    output logic          matmul_local_output_write_ready,
    input  hardware_types_pkg::matmul_output_write_t matmul_local_output_write,
    input  logic          elementwise_product_req_valid,
    output logic          elementwise_product_req_ready,
    input  hardware_types_pkg::elementwise_product_request_t
                          elementwise_product_req,
    output logic          elementwise_product_rsp_valid,
    input  logic          elementwise_product_rsp_ready,
    output hardware_types_pkg::elementwise_product_response_t
                          elementwise_product_rsp,
    input  logic          matmul_residual_read_valid,
    output logic          matmul_residual_read_ready,
    input  hardware_types_pkg::matmul_residual_read_request_t
        matmul_residual_read_req,
    output logic          matmul_residual_read_rsp_valid,
    output hardware_types_pkg::local_128b_read_response_t
        matmul_residual_read_rsp,

    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        port_req_ready,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        port_read_valid,
    output logic [local_memory_layout_pkg::LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        port_read_data
);
    import local_memory_layout_pkg::*;

    function automatic logic [127:0] bf16_lane_write_mask_n(
        input logic [7:0] lane_mask
    );
        for (integer lane = 0; lane < 8; lane++)
            bf16_lane_write_mask_n[lane*16 +: 16] =
                {16{!lane_mask[lane]}};
    endfunction

    function automatic logic [63:0] int8_lane_write_mask_n(
        input logic [7:0] lane_mask
    );
        for (integer lane = 0; lane < 8; lane++)
            int8_lane_write_mask_n[lane*8 +: 8] =
                {8{!lane_mask[lane]}};
    endfunction

    function automatic logic [127:0] byte_write_mask_n(
        input logic [15:0] byte_enable
    );
        for (integer byte_lane = 0; byte_lane < 16; byte_lane++)
            byte_write_mask_n[byte_lane*8 +: 8] =
                {8{!byte_enable[byte_lane]}};
    endfunction

    typedef struct packed {
        logic write;
        logic [11:0] address;
        logic [127:0] write_data;
        logic [127:0] write_bit_enable;
    } endpoint_request_payload_t;

    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] port_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] client_req_ready;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] port_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] port_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] port_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] port_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] port_double_drive;
    logic [LOCAL_SRAM_MACRO_COUNT-1:0] macro_same_address_conflict;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] rms_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] rms_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] rms_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] rms_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] rms_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        refresh_metadata_scratch_selected;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        refresh_metadata_scratch_aux_selected;
    logic [1:0] refresh_metadata_scratch_read_pending;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] hidden_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] hidden_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] hidden_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] hidden_port_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] hidden_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] context_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] context_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] context_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] context_port_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] context_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] attention_scratch_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] attention_scratch_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] attention_scratch_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_scratch_port_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_scratch_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] qkv_head_read_port_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] qkv_head_read_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] rope_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] rope_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] rope_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] rope_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] rope_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] qkv_stage_write_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] qkv_stage_write_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        qkv_stage_write_port_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        qkv_stage_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] qkv_tile_read_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] qkv_tile_read_address;
    logic qkv_tile_matmul_conflict;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] attention_operand_port_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        attention_operand_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] attention_score_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] attention_score_port_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] attention_score_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_score_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_score_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        attention_probability_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        attention_probability_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_probability_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        attention_probability_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] cache_write_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] cache_write_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] cache_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] cache_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] matmul_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] matmul_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] matmul_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] candidate_state_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] candidate_state_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] candidate_state_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        candidate_state_port_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        candidate_state_port_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_source_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] matmul_source_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_operand_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] matmul_operand_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_prefetch_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] matmul_prefetch_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        matmul_prefetch_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        matmul_prefetch_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_other_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] matmul_other_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] matmul_other_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        matmul_other_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        matmul_other_write_mask_n;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        elementwise_product_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        elementwise_product_selected_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0]
        elementwise_product_port_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0]
        elementwise_product_port_data;
    logic [5:0] elementwise_product_port [0:3];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        elementwise_product_macro_onehot [0:3];
    logic elementwise_product_port_side [0:3];
    logic [11:0] elementwise_product_address [0:3];
    logic [2:0] elementwise_product_response_group;
    logic [1:0] elementwise_product_read_pending;
    logic [2:0] elementwise_product_pending_group [0:1];
    logic elementwise_product_pending_half [0:1];
    logic elementwise_product_pending_physical_half [0:1];
    logic [15:0] elementwise_product_pending_tag [0:1];
    logic elementwise_product_fifo_input_valid;
    logic elementwise_product_fifo_input_ready;
    logic [528:0] elementwise_product_fifo_input_data;
    logic [2:0] elementwise_product_fifo_occupancy;
    logic [1:0] matmul_source_read_pending;
    logic [5:0] matmul_source_pending_row [0:1];
    logic [3:0] matmul_source_pending_row_count [0:1];
    logic [15:0] matmul_source_pending_element [0:1];
    logic [4:0] matmul_source_pending_bank_base [0:1];
    logic [1:0] matmul_operand_read_pending;
    logic [15:0] matmul_pending_activation_word_index [0:1];
    logic [3:0] matmul_pending_activation_word_count [0:1];
    logic matmul_pending_qkv_padded_layout [0:1];
    logic [5:0] matmul_pending_physical_row_base [0:1];
    logic matmul_pending_explicit_activation_rows [0:1];
    logic [47:0] matmul_pending_activation_physical_row [0:1];
    logic [127:0] matmul_pending_activation_lane_word_index [0:1];
    logic matmul_pending_panel [0:1];
    logic [11:0] matmul_pending_panel_mask [0:1];
    logic matmul_pending_q_work [0:1];
    logic [1:0] lm_head_scale_read_pending;
    logic post_table_access_enabled;
    logic post_table_request_exclusive;
    logic post_scale_request_valid;
    logic post_state_request_valid;
    logic post_suppressed_request_valid;
    logic [1:0] post_state_read_pending;
    logic [15:0] post_state_read_pending_tag [0:1];
    logic [1:0] post_suppressed_read_pending;
    logic [1:0] post_suppressed_read_pending_lane [0:1];
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        candidate_state_competing_req_valid;
    logic [11:0] candidate_state_selected_address;
    logic candidate_state_selected_panel;
    logic final_output_compute_panel;
    logic candidate_state_compute_panel;
    logic candidate_state_use_secondary;
    logic [3:0] candidate_state_selected_row_mask;
    logic [3:0] candidate_state_available_mask;
    logic [3:0] candidate_state_primary_available_mask;
    logic [3:0] candidate_state_secondary_available_mask;
    logic candidate_state_shape_valid;
    logic candidate_state_read_fire, candidate_state_write_fire;
    logic [1:0] candidate_state_read_pending;
    logic candidate_state_read_pending_panel [0:1];
    logic candidate_state_read_pending_secondary [0:1];
    logic [3:0] candidate_state_read_pending_mask [0:1];
    logic [15:0] candidate_state_read_pending_tag [0:1];
    logic candidate_state_fifo_input_valid;
    logic candidate_state_fifo_input_ready;
    logic [531:0] candidate_state_fifo_input_data;
    logic [2:0] candidate_state_fifo_occupancy;
    logic [2:0] candidate_state_read_reserved_count;
    logic candidate_state_read_reservation_available;
    localparam int unsigned MATMUL_STAGING_DEPTH = 96;
    localparam int unsigned MATMUL_GATE_SCRATCH_DEPTH = 288;
    localparam int unsigned MATMUL_GATE_SCRATCH_PORT = 45;
    localparam int unsigned MATMUL_GATE_SCRATCH_ROW_BASE = 512;
    localparam int unsigned MATMUL_COMBINE_DEPTH =
        MATMUL_STAGING_DEPTH + MATMUL_GATE_SCRATCH_DEPTH;
    logic [127:0] matmul_staging_memory [0:MATMUL_STAGING_DEPTH-1];
    logic [127:0] matmul_staging_read_data [0:1];
    logic [1:0] matmul_combine_read_pending;
    logic [1:0] matmul_combine_pending_sram;
    logic matmul_gate_scratch_layout;
    logic matmul_gate_scratch_port_available;
    logic [1:0] matmul_residual_read_pending;
    logic [5:0] matmul_residual_pending_row [0:1];
    logic [128:0] matmul_aligned_hidden_response [0:7];
    logic [5:0] matmul_source_port [0:7];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        matmul_source_macro_onehot [0:7];
    logic matmul_source_port_side [0:7];
    logic [11:0] matmul_source_word_address [0:7];
    logic [5:0] matmul_activation_write_port [0:7];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        matmul_activation_write_macro_onehot [0:7];
    logic matmul_activation_write_port_side [0:7];
    logic [11:0] matmul_activation_write_word_address [0:7];
    logic [5:0] matmul_activation_read_port [0:7];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        matmul_activation_read_macro_onehot [0:7];
    logic matmul_activation_read_port_side [0:7];
    logic [11:0] matmul_activation_read_word_address [0:7];
    logic [5:0] matmul_panel_read_port [0:11];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        matmul_panel_read_macro_onehot [0:11];
    logic matmul_panel_read_port_side [0:11];
    logic [11:0] matmul_panel_read_word_address [0:11];
    logic [5:0] matmul_panel_write_port [0:1];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        matmul_panel_write_macro_onehot [0:1];
    logic matmul_panel_write_port_side [0:1];
    logic [11:0] matmul_panel_write_word_address [0:1];
    // SRAM endpoint IDs for trace probes; 6'h3f denotes register staging.
    logic [5:0] matmul_combine_write_port;
    logic [5:0] matmul_combine_read_port;
    logic [5:0] matmul_local_output_port;
    logic [11:0] matmul_local_output_word_address;
    logic [5:0] matmul_residual_port;
    logic matmul_local_layout;
    logic matmul_expanded_layout;
    logic matmul_preserved_panel_layout;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] logical_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] macro_req_valid;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] macro_req_ready;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] macro_write;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*12-1:0] macro_address;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] macro_write_data;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT*128-1:0] macro_write_mask_n;
    logic [5:0] hidden_write_port;
    logic [5:0] hidden_read_port;
    logic [LOCAL_SRAM_MACRO_COUNT-1:0] hidden_read_macro_onehot;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] hidden_write_port_mask;
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0] hidden_read_port_mask;
    logic [11:0] hidden_write_address;
    logic [11:0] hidden_read_address;
    logic hidden_read_cache_conflict;
    logic [127:0] hidden_write_mask_word;
    logic [1:0] hidden_read_pending;
    logic [5:0] hidden_read_pending_port [0:1];
    logic [127:0] context_write_mask_word;
    logic context_read_phase_active;
    logic [1:0] context_read_pending;
    logic [15:0] context_read_pending_tag [0:1];
    logic [127:0] attention_scratch_write_mask_word;
    logic [1:0] attention_scratch_read_pending;
    logic attention_scratch_read_pending_bank [0:1];
    logic [15:0] attention_scratch_read_pending_tag [0:1];
    logic attention_scratch_fifo_input_valid;
    logic attention_scratch_fifo_input_ready;
    logic [271:0] attention_scratch_fifo_input_data;
    logic [2:0] attention_scratch_fifo_occupancy;
    logic [3:0] attention_scratch_reserved_count;
    logic [1:0] qkv_head_read_pending;
    logic qkv_head_read_pending_destination [0:1];
    logic qkv_head_read_return_valid;
    logic [127:0] qkv_head_read_return_data;
    logic qkv_head_read_fifo_input_ready;
    logic [2:0] qkv_head_read_fifo_occupancy;
    logic [3:0] qkv_head_read_reserved_count;
    logic qkv_head_read_reservation_available;
    logic rope_source_read_pending;
    logic rope_cos_read_pending;
    logic rope_sin_read_pending;
    logic [9:0] rope_constant_word;
    logic [127:0] qkv_stage_write_mask_word;
    logic [7:0] qkv_stage_write_endpoint_select;
    logic [7:0] qkv_stage_write_endpoint_ready;
    logic qkv_stage_write_active;
    logic [1:0] qkv_tile_read_pending;
    logic [7:0] qkv_tile_read_pending_mask [0:1];
    logic qkv_tile_read_pending_q_work [0:1];
    logic [128:0] qkv_tile_read_response [0:7];
    logic [9:0] qkv_tile_read_primary_word;
    logic [9:0] qkv_tile_read_secondary_word;
    logic [9:0] attention_operand_activation_word;
    logic [9:0] attention_operand_panel_word;
    logic [1:0] attention_operand_read_pending;
    logic attention_operand_pending_is_pv [0:1];
    logic attention_operand_pending_extra [0:1];
    logic attention_operand_pending_panel [0:1];
    logic attention_q_slot_ports_ready;
    logic attention_panel_ports_ready;
    logic attention_panel_ports_valid;
    logic attention_panel_metadata_req_ready;
    logic attention_panel_metadata_rsp_valid;
    logic [1:0] attention_score_read_pending;
    logic [7:0] attention_score_read_row_mask;
    logic [7:0] attention_score_read_pending_row_mask [0:1];
    logic attention_score_read_pending_extra [0:1];
    logic rms_role_busy;
    logic hidden_role_busy;
    logic context_role_busy;
    logic attention_scratch_role_busy;
    logic qkv_head_role_busy;
    logic rope_role_busy;
    logic qkv_stage_role_busy;
    logic qkv_cache_write_busy;
    logic qkv_q_endpoint_ready;
    logic [5:0] q_head_slot_write_port;
    logic [5:0] q_metadata_write_port;
    logic attention_operand_role_busy;
    logic attention_score_role_busy;
    logic attention_probability_role_busy;
    logic attention_panel_write_busy;
    logic attention_panel_data_endpoint_ready;
    logic attention_panel_scale_endpoint_ready;
    logic matmul_non_panel_role_busy;
    logic elementwise_product_role_busy;
    logic matmul_panel_role_busy;
    logic current_attention_phase;
    logic next_attention_phase;
    logic panel_mapping_changed;
    logic phase_role_drained;
    logic qkv_staging_role_drained;
    logic panel_role_drained;
    logic attention_pair_mapping_changed;
    logic attention_pair_role_drained;
    logic single_score_write_ready;
    logic single_score_read_ready;
    logic single_score_response_valid;
    logic [1023:0] single_score_response_values;
    logic [7:0] pair_score_write_ready;
    logic [7:0] pair_score_read_ready;
    logic [7:0] pair_score_response_valid;
    logic [1023:0] pair_score_response_values;
    logic single_probability_write_ready;
    logic single_probability_quantized_ready;
    logic single_probability_scale_ready;
    logic [7:0] pair_probability_write_ready;
    logic [7:0] pair_probability_quantized_ready;
    logic [5:0] context_write_port;
    logic [5:0] context_read_port;
    logic [2:0] context_read_pending_macro [0:1];
    logic context_return_valid;
    logic [127:0] context_return_data;
    logic single_attention_operand_ready;
    local_sram_region_t q_source_region;
    local_sram_region_t q_post_source_region;
    local_sram_region_t q_destination_region;
    local_sram_region_t rope_table_region;
    logic projection_pair_layout;
    assign projection_pair_layout = current_layout.attention_pair_enable ||
        current_layout.kv_pair_enable;
    local_sram_region_t q_staging_regions [0:3];
    local_sram_region_t kv_staging_regions [0:3];
    for (genvar stripe = 0; stripe < 4; stripe++) begin : g_q_staging_regions
        assign q_staging_regions[stripe] = current_layout.attention_pair_enable ?
            attention_pair_head_staging_region(2'(stripe)) : q_work_panel_region(2'(stripe));
        // K/V staging requests already include the 256-word staging offset.
        assign kv_staging_regions[stripe] = current_layout.kv_pair_enable ?
            attention_pair_panel_region(2'(stripe)) : qkv_panel_region(2'(stripe));
    end
    local_sram_region_t attention_scratch_regions [0:1];
    assign attention_scratch_regions[0] = current_layout.attention_pair_enable ?
        attention_pair_scratch_region(1'b0) : attention_scratch_region(1'b0);
    assign attention_scratch_regions[1] = current_layout.attention_pair_enable ?
        attention_pair_scratch_region(1'b1) : attention_scratch_region(1'b1);
    assign q_source_region = projection_pair_layout ?
        ATTENTION_PAIR_HEAD_SOURCE_REGION : qkv_head_source_region();
    always_comb begin
        q_post_source_region = q_source_region;
        if (current_layout.kv_pair_enable && current_layout.kv_second_batch)
            q_post_source_region.row_base = q_source_region.row_base +
                {2'd0, current_layout.kv_first_batch_tokens, 4'd0};
    end
    assign q_destination_region = projection_pair_layout ?
        ATTENTION_PAIR_HEAD_DESTINATION_REGION : qkv_head_destination_region();
    assign rope_table_region = current_layout.attention_pair_enable ?
        attention_pair_rope_constant_region(
            current_layout.attention_batch_index) :
        current_layout.kv_pair_enable ?
            attention_pair_rope_constant_region(
                {1'b0, current_layout.kv_second_batch}) :
            qkv_constant_region();
    logic single_attention_operand_valid;
    logic [7:0] pair_attention_activation_ready;
    logic [7:0] pair_attention_activation_valid;
    logic [3:0] pair_attention_panel_ready;
    logic [3:0] pair_attention_panel_valid;

    localparam logic [5:0] CONTEXT_WRITE_PORT = local_region_port_index(
        attention_context_region(), 1'b1);
    localparam logic [5:0] CONTEXT_READ_PORT = local_region_port_index(
        attention_context_region(), 1'b0);
    logic [5:0] ATTENTION_SCRATCH0_PRIMARY_PORT;
    assign ATTENTION_SCRATCH0_PRIMARY_PORT = local_region_port_index(attention_scratch_regions[0], 1'b0);
    logic [5:0] ATTENTION_SCRATCH0_SECONDARY_PORT;
    assign ATTENTION_SCRATCH0_SECONDARY_PORT = local_region_port_index(attention_scratch_regions[0], 1'b1);
    logic [5:0] ATTENTION_SCRATCH1_PRIMARY_PORT;
    assign ATTENTION_SCRATCH1_PRIMARY_PORT = local_region_port_index(attention_scratch_regions[1], 1'b0);
    logic [5:0] ATTENTION_SCRATCH1_SECONDARY_PORT;
    assign ATTENTION_SCRATCH1_SECONDARY_PORT = local_region_port_index(attention_scratch_regions[1], 1'b1);
    logic [5:0] QKV_HEAD_SOURCE_PORT;
    assign QKV_HEAD_SOURCE_PORT = local_region_port_index(q_source_region, 1'b0);
    logic [5:0] QKV_HEAD_DESTINATION_PORT;
    assign QKV_HEAD_DESTINATION_PORT = local_region_port_index(q_destination_region, 1'b0);
    logic [5:0] ROPE_SOURCE_PRIMARY_PORT;
    assign ROPE_SOURCE_PRIMARY_PORT = local_region_port_index(q_source_region, 1'b0);
    logic [5:0] ROPE_SOURCE_SECONDARY_PORT;
    assign ROPE_SOURCE_SECONDARY_PORT = local_region_port_index(q_source_region, 1'b1);
    logic [5:0] ROPE_CONSTANT_PRIMARY_PORT;
    assign ROPE_CONSTANT_PRIMARY_PORT = local_region_port_index(rope_table_region, 1'b0);
    logic [5:0] ROPE_CONSTANT_SECONDARY_PORT;
    assign ROPE_CONSTANT_SECONDARY_PORT = local_region_port_index(rope_table_region, 1'b1);
    logic [5:0] ROPE_DESTINATION_PRIMARY_PORT;
    assign ROPE_DESTINATION_PRIMARY_PORT = local_region_port_index(q_destination_region, 1'b0);
    logic [5:0] ROPE_DESTINATION_SECONDARY_PORT;
    assign ROPE_DESTINATION_SECONDARY_PORT = local_region_port_index(q_destination_region, 1'b1);
    logic [5:0] QKV_STAGE0_PRIMARY_PORT;
    assign QKV_STAGE0_PRIMARY_PORT =
        local_region_port_index(kv_staging_regions[0], 1'b0);
    logic [5:0] QKV_STAGE0_SECONDARY_PORT;
    assign QKV_STAGE0_SECONDARY_PORT =
        local_region_port_index(kv_staging_regions[0], 1'b1);
    logic [5:0] QKV_STAGE1_PRIMARY_PORT;
    assign QKV_STAGE1_PRIMARY_PORT =
        local_region_port_index(kv_staging_regions[1], 1'b0);
    logic [5:0] QKV_STAGE1_SECONDARY_PORT;
    assign QKV_STAGE1_SECONDARY_PORT =
        local_region_port_index(kv_staging_regions[1], 1'b1);
    logic [5:0] QKV_STAGE2_PRIMARY_PORT;
    assign QKV_STAGE2_PRIMARY_PORT =
        local_region_port_index(kv_staging_regions[2], 1'b0);
    logic [5:0] QKV_STAGE2_SECONDARY_PORT;
    assign QKV_STAGE2_SECONDARY_PORT =
        local_region_port_index(kv_staging_regions[2], 1'b1);
    logic [5:0] QKV_STAGE3_PRIMARY_PORT;
    assign QKV_STAGE3_PRIMARY_PORT =
        local_region_port_index(kv_staging_regions[3], 1'b0);
    logic [5:0] QKV_STAGE3_SECONDARY_PORT;
    assign QKV_STAGE3_SECONDARY_PORT =
        local_region_port_index(kv_staging_regions[3], 1'b1);
    logic [5:0] Q_WORK0_PRIMARY_PORT;
    assign Q_WORK0_PRIMARY_PORT = local_region_port_index(q_staging_regions[0], 1'b0);
    logic [5:0] Q_WORK0_SECONDARY_PORT;
    assign Q_WORK0_SECONDARY_PORT = local_region_port_index(q_staging_regions[0], 1'b1);
    logic [5:0] Q_WORK1_PRIMARY_PORT;
    assign Q_WORK1_PRIMARY_PORT = local_region_port_index(q_staging_regions[1], 1'b0);
    logic [5:0] Q_WORK1_SECONDARY_PORT;
    assign Q_WORK1_SECONDARY_PORT = local_region_port_index(q_staging_regions[1], 1'b1);
    logic [5:0] Q_WORK2_PRIMARY_PORT;
    assign Q_WORK2_PRIMARY_PORT = local_region_port_index(q_staging_regions[2], 1'b0);
    logic [5:0] Q_WORK2_SECONDARY_PORT;
    assign Q_WORK2_SECONDARY_PORT = local_region_port_index(q_staging_regions[2], 1'b1);
    logic [5:0] Q_WORK3_PRIMARY_PORT;
    assign Q_WORK3_PRIMARY_PORT = local_region_port_index(q_staging_regions[3], 1'b0);
    logic [5:0] Q_WORK3_SECONDARY_PORT;
    assign Q_WORK3_SECONDARY_PORT = local_region_port_index(q_staging_regions[3], 1'b1);
    localparam logic [5:0] ATTENTION_PANEL0_PRIMARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd0), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL0_SECONDARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd0), 1'b1);
    localparam logic [5:0] ATTENTION_PANEL1_PRIMARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd1), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL1_SECONDARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd1), 1'b1);
    localparam logic [5:0] ATTENTION_PANEL2_PRIMARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd2), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL2_SECONDARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd2), 1'b1);
    localparam logic [5:0] ATTENTION_PANEL3_PRIMARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd3), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL3_SECONDARY_PORT =
        local_region_port_index(attention_kv_panel_region(2'd3), 1'b1);
    localparam logic [5:0] Q_HEAD_SLOT0_PORT =
        local_region_port_index(q_head_slot_region(3'd0), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT1_PORT =
        local_region_port_index(q_head_slot_region(3'd1), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT2_PORT =
        local_region_port_index(q_head_slot_region(3'd2), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT3_PORT =
        local_region_port_index(q_head_slot_region(3'd3), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT4_PORT =
        local_region_port_index(q_head_slot_region(3'd4), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT5_PORT =
        local_region_port_index(q_head_slot_region(3'd5), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT6_PORT =
        local_region_port_index(q_head_slot_region(3'd6), 1'b0);
    localparam logic [5:0] Q_HEAD_SLOT7_PORT =
        local_region_port_index(q_head_slot_region(3'd7), 1'b0);
    localparam logic [5:0] Q_CODE0_PORT =
        local_region_port_index(q_quantized_region(3'd0), 1'b0);
    localparam logic [5:0] Q_CODE1_PORT =
        local_region_port_index(q_quantized_region(3'd1), 1'b0);
    localparam logic [5:0] Q_CODE2_PORT =
        local_region_port_index(q_quantized_region(3'd2), 1'b0);
    localparam logic [5:0] Q_CODE3_PORT =
        local_region_port_index(q_quantized_region(3'd3), 1'b0);
    localparam logic [5:0] Q_CODE4_PORT =
        local_region_port_index(q_quantized_region(3'd4), 1'b0);
    localparam logic [5:0] Q_CODE5_PORT =
        local_region_port_index(q_quantized_region(3'd5), 1'b0);
    localparam logic [5:0] Q_CODE6_PORT =
        local_region_port_index(q_quantized_region(3'd6), 1'b0);
    localparam logic [5:0] Q_CODE7_PORT =
        local_region_port_index(q_quantized_region(3'd7), 1'b0);
    localparam logic [5:0] Q_CODE_EXTRA0_PRIMARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd0), 1'b0);
    localparam logic [5:0] Q_CODE_EXTRA0_SECONDARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd0), 1'b1);
    localparam logic [5:0] Q_CODE_EXTRA1_PRIMARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd1), 1'b0);
    localparam logic [5:0] Q_CODE_EXTRA1_SECONDARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd1), 1'b1);
    localparam logic [5:0] Q_CODE_EXTRA2_PRIMARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd2), 1'b0);
    localparam logic [5:0] Q_CODE_EXTRA2_SECONDARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd2), 1'b1);
    localparam logic [5:0] Q_CODE_EXTRA3_PRIMARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd3), 1'b0);
    localparam logic [5:0] Q_CODE_EXTRA3_SECONDARY_PORT =
        local_region_port_index(q_quantized_extra_region(2'd3), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE0_PORT =
        local_region_port_index(attention_score_region(3'd0), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE1_PORT =
        local_region_port_index(attention_score_region(3'd1), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE2_PORT =
        local_region_port_index(attention_score_region(3'd2), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE3_PORT =
        local_region_port_index(attention_score_region(3'd3), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE4_PORT =
        local_region_port_index(attention_score_region(3'd4), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE5_PORT =
        local_region_port_index(attention_score_region(3'd5), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE6_PORT =
        local_region_port_index(attention_score_region(3'd6), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE7_PORT =
        local_region_port_index(attention_score_region(3'd7), 1'b0);
    localparam logic [5:0] ATTENTION_METADATA_PORT =
        local_region_port_index(qkv_attention_metadata_region(), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL_METADATA0_PORT =
        local_region_port_index(attention_panel_metadata_region(1'b0), 1'b1);
    localparam logic [5:0] ATTENTION_PANEL_METADATA1_PORT =
        local_region_port_index(attention_panel_metadata_region(1'b1), 1'b1);
    localparam logic [5:0] ATTENTION_PANEL_METADATA0_WRITE_PORT =
        local_region_port_index(attention_panel_metadata_region(1'b0), 1'b0);
    localparam logic [5:0] ATTENTION_PANEL_METADATA1_WRITE_PORT =
        local_region_port_index(attention_panel_metadata_region(1'b1), 1'b0);
    localparam logic [5:0] ATTENTION_METADATA_WRITE_PORT =
        local_region_port_index(qkv_attention_metadata_region(), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE0_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd0), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE1_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd1), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE2_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd2), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE3_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd3), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE4_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd4), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE5_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd5), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE6_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd6), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE7_WRITE_PORT =
        local_region_port_index(attention_score_region(3'd7), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA0_PORT =
        local_region_port_index(attention_score_extra_region(3'd0), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA1_PORT =
        local_region_port_index(attention_score_extra_region(3'd1), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA2_PORT =
        local_region_port_index(attention_score_extra_region(3'd2), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA3_PORT =
        local_region_port_index(attention_score_extra_region(3'd3), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA4_PORT =
        local_region_port_index(attention_score_extra_region(3'd4), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA5_PORT =
        local_region_port_index(attention_score_extra_region(3'd5), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA6_PORT =
        local_region_port_index(attention_score_extra_region(3'd6), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA7_PORT =
        local_region_port_index(attention_score_extra_region(3'd7), 1'b0);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA0_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd0), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA1_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd1), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA2_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd2), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA3_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd3), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA4_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd4), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA5_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd5), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA6_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd6), 1'b1);
    localparam logic [5:0] ATTENTION_SCORE_EXTRA7_WRITE_PORT =
        local_region_port_index(attention_score_extra_region(3'd7), 1'b1);
    localparam logic [5:0] RMS_ATTENTION_GAMMA_PORT =
        local_region_port_index(rms_gamma_region(1'b0), 1'b0);
    localparam logic [5:0] RMS_FFN_GAMMA_PORT =
        local_region_port_index(rms_gamma_region(1'b1), 1'b0);
    localparam logic [5:0] LM_HEAD_SCALE_PORT =
        local_region_port_index(lm_head_scale_buffer_region(), 1'b0);
    localparam logic [5:0] CANDIDATE_P0_S0_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd0), 1'b0);
    localparam logic [5:0] CANDIDATE_P0_S1_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd1), 1'b0);
    localparam logic [5:0] CANDIDATE_P0_S2_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd2), 1'b0);
    localparam logic [5:0] CANDIDATE_P0_S3_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd3), 1'b0);
    localparam logic [5:0] CANDIDATE_P0_S0_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd0), 1'b1);
    localparam logic [5:0] CANDIDATE_P0_S1_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd1), 1'b1);
    localparam logic [5:0] CANDIDATE_P0_S2_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd2), 1'b1);
    localparam logic [5:0] CANDIDATE_P0_S3_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b0, 2'd3), 1'b1);
    localparam logic [5:0] CANDIDATE_P1_S0_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd0), 1'b0);
    localparam logic [5:0] CANDIDATE_P1_S1_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd1), 1'b0);
    localparam logic [5:0] CANDIDATE_P1_S2_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd2), 1'b0);
    localparam logic [5:0] CANDIDATE_P1_S3_PRIMARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd3), 1'b0);
    localparam logic [5:0] CANDIDATE_P1_S0_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd0), 1'b1);
    localparam logic [5:0] CANDIDATE_P1_S1_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd1), 1'b1);
    localparam logic [5:0] CANDIDATE_P1_S2_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd2), 1'b1);
    localparam logic [5:0] CANDIDATE_P1_S3_SECONDARY_PORT =
        local_region_port_index(candidate_state_region(1'b1, 2'd3), 1'b1);

    assign q_head_slot_write_port = local_region_port_index(
        q_head_slot_region(qkv_q_write.physical_row[2:0]),
        1'b1);
    assign q_metadata_write_port = local_region_port_index(
        qkv_attention_metadata_region(),
        current_layout.q_head_active);
    localparam logic [5:0] RMS_SCRATCH0_PRIMARY_PORT =
        local_region_port_index(rms_scratch_region(1'b0), 1'b0);
    localparam logic [5:0] RMS_SCRATCH0_SECONDARY_PORT =
        local_region_port_index(rms_scratch_region(1'b0), 1'b1);
    localparam logic [5:0] RMS_SCRATCH1_PRIMARY_PORT =
        local_region_port_index(rms_scratch_region(1'b1), 1'b0);
    localparam logic [5:0] RMS_SCRATCH1_SECONDARY_PORT =
        local_region_port_index(rms_scratch_region(1'b1), 1'b1);
    localparam logic [5:0] RMS_FFN_SCRATCH0_PRIMARY_PORT =
        local_region_port_index(matmul_staging_region(1'b0), 1'b0);
    localparam logic [5:0] RMS_FFN_SCRATCH0_SECONDARY_PORT =
        local_region_port_index(matmul_staging_region(1'b0), 1'b1);
    localparam logic [5:0] RMS_FFN_SCRATCH1_PRIMARY_PORT =
        local_region_port_index(matmul_staging_region(1'b1), 1'b0);
    localparam logic [5:0] RMS_FFN_SCRATCH1_SECONDARY_PORT =
        local_region_port_index(matmul_staging_region(1'b1), 1'b1);
    localparam logic [5:0] RMS_FINAL_SCRATCH0_PRIMARY_PORT =
        local_region_port_index(final_norm_scratch_region(1'b0), 1'b0);
    localparam logic [5:0] RMS_FINAL_SCRATCH0_SECONDARY_PORT =
        local_region_port_index(final_norm_scratch_region(1'b0), 1'b1);
    localparam logic [5:0] RMS_FINAL_SCRATCH1_PRIMARY_PORT =
        local_region_port_index(final_norm_scratch_region(1'b1), 1'b0);
    localparam logic [5:0] RMS_FINAL_SCRATCH1_SECONDARY_PORT =
        local_region_port_index(final_norm_scratch_region(1'b1), 1'b1);

    // Each selector below names only the endpoints that the corresponding
    // logical Matmul role can use. The complete 52-port SRAM request vector is not a
    // selector input.
    function automatic logic [128:0] matmul_hidden_response(
        input logic [5:0] physical_row
    );
        if (physical_row >= 6'd32) begin
            case (physical_row[2:0])
                3'd0: matmul_hidden_response = {port_read_valid[0], port_read_data[0*128 +: 128]};
                3'd1: matmul_hidden_response = {port_read_valid[25], port_read_data[25*128 +: 128]};
                3'd2: matmul_hidden_response = {port_read_valid[7], port_read_data[7*128 +: 128]};
                3'd3: matmul_hidden_response = {port_read_valid[45], port_read_data[45*128 +: 128]};
                3'd4: matmul_hidden_response = {port_read_valid[15], port_read_data[15*128 +: 128]};
                3'd5: matmul_hidden_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
                3'd6: matmul_hidden_response = {port_read_valid[37], port_read_data[37*128 +: 128]};
                default: matmul_hidden_response = {port_read_valid[47], port_read_data[47*128 +: 128]};
            endcase
        end else case (physical_row[3:0])
            4'd0: matmul_hidden_response = {port_read_valid[9], port_read_data[9*128 +: 128]};
            4'd1: matmul_hidden_response = {port_read_valid[35], port_read_data[35*128 +: 128]};
            4'd2: matmul_hidden_response = {port_read_valid[19], port_read_data[19*128 +: 128]};
            4'd3: matmul_hidden_response = {port_read_valid[5], port_read_data[5*128 +: 128]};
            4'd4: matmul_hidden_response = {port_read_valid[27], port_read_data[27*128 +: 128]};
            4'd5: matmul_hidden_response = {port_read_valid[13], port_read_data[13*128 +: 128]};
            4'd6: matmul_hidden_response = {port_read_valid[11], port_read_data[11*128 +: 128]};
            4'd7: matmul_hidden_response = {port_read_valid[17], port_read_data[17*128 +: 128]};
            4'd8: matmul_hidden_response = {port_read_valid[11], port_read_data[11*128 +: 128]};
            4'd9: matmul_hidden_response = {port_read_valid[25], port_read_data[25*128 +: 128]};
            4'd10: matmul_hidden_response = {port_read_valid[5], port_read_data[5*128 +: 128]};
            4'd11: matmul_hidden_response = {port_read_valid[3], port_read_data[3*128 +: 128]};
            4'd12: matmul_hidden_response = {port_read_valid[19], port_read_data[19*128 +: 128]};
            4'd13: matmul_hidden_response = {port_read_valid[17], port_read_data[17*128 +: 128]};
            4'd14: matmul_hidden_response = {port_read_valid[13], port_read_data[13*128 +: 128]};
            default: matmul_hidden_response = {port_read_valid[7], port_read_data[7*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_qkv_activation_response(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: matmul_qkv_activation_response = {port_read_valid[39], port_read_data[39*128 +: 128]};
            3'd1: matmul_qkv_activation_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
            3'd2: matmul_qkv_activation_response = {port_read_valid[15], port_read_data[15*128 +: 128]};
            3'd3: matmul_qkv_activation_response = {port_read_valid[27], port_read_data[27*128 +: 128]};
            3'd4: matmul_qkv_activation_response = {port_read_valid[23], port_read_data[23*128 +: 128]};
            3'd5: matmul_qkv_activation_response = {port_read_valid[49], port_read_data[49*128 +: 128]};
            3'd6: matmul_qkv_activation_response = {port_read_valid[9], port_read_data[9*128 +: 128]};
            default: matmul_qkv_activation_response = {port_read_valid[3], port_read_data[3*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_q_quantized_response(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: matmul_q_quantized_response = {port_read_valid[44], port_read_data[44*128 +: 128]};
            3'd1: matmul_q_quantized_response = {port_read_valid[25], port_read_data[25*128 +: 128]};
            3'd2: matmul_q_quantized_response = {port_read_valid[1], port_read_data[1*128 +: 128]};
            3'd3: matmul_q_quantized_response = {port_read_valid[15], port_read_data[15*128 +: 128]};
            3'd4: matmul_q_quantized_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
            3'd5: matmul_q_quantized_response = {port_read_valid[23], port_read_data[23*128 +: 128]};
            3'd6: matmul_q_quantized_response = {port_read_valid[7], port_read_data[7*128 +: 128]};
            default: matmul_q_quantized_response = {port_read_valid[37], port_read_data[37*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_preserved_activation_response(
        input logic [2:0] stripe
    );
        case (stripe)
            3'd0: matmul_preserved_activation_response = {port_read_valid[41], port_read_data[41*128 +: 128]};
            3'd1: matmul_preserved_activation_response = {port_read_valid[0], port_read_data[0*128 +: 128]};
            3'd2: matmul_preserved_activation_response = {port_read_valid[39], port_read_data[39*128 +: 128]};
            3'd3: matmul_preserved_activation_response = {port_read_valid[23], port_read_data[23*128 +: 128]};
            3'd4: matmul_preserved_activation_response = {port_read_valid[43], port_read_data[43*128 +: 128]};
            3'd5: matmul_preserved_activation_response = {port_read_valid[29], port_read_data[29*128 +: 128]};
            3'd6: matmul_preserved_activation_response = {port_read_valid[22], port_read_data[22*128 +: 128]};
            default: matmul_preserved_activation_response = {port_read_valid[30], port_read_data[30*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_full_activation_response(
        input logic [4:0] stripe
    );
        case (stripe)
            5'd0: matmul_full_activation_response = {port_read_valid[0], port_read_data[0*128 +: 128]};
            5'd1: matmul_full_activation_response = {port_read_valid[13], port_read_data[13*128 +: 128]};
            5'd2: matmul_full_activation_response = {port_read_valid[19], port_read_data[19*128 +: 128]};
            5'd3: matmul_full_activation_response = {port_read_valid[1], port_read_data[1*128 +: 128]};
            5'd4: matmul_full_activation_response = {port_read_valid[9], port_read_data[9*128 +: 128]};
            5'd5: matmul_full_activation_response = {port_read_valid[7], port_read_data[7*128 +: 128]};
            5'd6: matmul_full_activation_response = {port_read_valid[23], port_read_data[23*128 +: 128]};
            5'd7: matmul_full_activation_response = {port_read_valid[6], port_read_data[6*128 +: 128]};
            5'd8: matmul_full_activation_response = {port_read_valid[43], port_read_data[43*128 +: 128]};
            5'd9: matmul_full_activation_response = {port_read_valid[41], port_read_data[41*128 +: 128]};
            5'd10: matmul_full_activation_response = {port_read_valid[11], port_read_data[11*128 +: 128]};
            5'd11: matmul_full_activation_response = {port_read_valid[2], port_read_data[2*128 +: 128]};
            5'd12: matmul_full_activation_response = {port_read_valid[27], port_read_data[27*128 +: 128]};
            5'd13: matmul_full_activation_response = {port_read_valid[29], port_read_data[29*128 +: 128]};
            5'd14: matmul_full_activation_response = {port_read_valid[5], port_read_data[5*128 +: 128]};
            5'd15: matmul_full_activation_response = {port_read_valid[35], port_read_data[35*128 +: 128]};
            5'd16: matmul_full_activation_response = {port_read_valid[4], port_read_data[4*128 +: 128]};
            5'd17: matmul_full_activation_response = {port_read_valid[19], port_read_data[19*128 +: 128]};
            5'd18: matmul_full_activation_response = {port_read_valid[39], port_read_data[39*128 +: 128]};
            5'd19: matmul_full_activation_response = {port_read_valid[25], port_read_data[25*128 +: 128]};
            5'd20: matmul_full_activation_response = {port_read_valid[17], port_read_data[17*128 +: 128]};
            5'd21: matmul_full_activation_response = {port_read_valid[13], port_read_data[13*128 +: 128]};
            5'd22: matmul_full_activation_response = {port_read_valid[31], port_read_data[31*128 +: 128]};
            default: matmul_full_activation_response = {port_read_valid[27], port_read_data[27*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_expanded_activation_response(
        input logic [15:0] word_index,
        input logic attention_pair
    );
        logic [4:0] macro_id;
        begin
            macro_id = attention_pair ? attention_pair_activation_macro(word_index[13:0]) :
                expanded_matmul_activation_macro(word_index);
            case (macro_id)
                5'd0: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[1], port_read_data[1*128 +: 128]} : {port_read_valid[0], port_read_data[0*128 +: 128]};
                5'd1: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[3], port_read_data[3*128 +: 128]} : {port_read_valid[2], port_read_data[2*128 +: 128]};
                5'd2: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[5], port_read_data[5*128 +: 128]} : {port_read_valid[4], port_read_data[4*128 +: 128]};
                5'd3: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[7], port_read_data[7*128 +: 128]} : {port_read_valid[6], port_read_data[6*128 +: 128]};
                5'd4: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[9], port_read_data[9*128 +: 128]} : {port_read_valid[8], port_read_data[8*128 +: 128]};
                5'd5: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[11], port_read_data[11*128 +: 128]} : {port_read_valid[10], port_read_data[10*128 +: 128]};
                5'd6: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[13], port_read_data[13*128 +: 128]} : {port_read_valid[12], port_read_data[12*128 +: 128]};
                5'd7: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[15], port_read_data[15*128 +: 128]} : {port_read_valid[14], port_read_data[14*128 +: 128]};
                5'd8: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[17], port_read_data[17*128 +: 128]} : {port_read_valid[16], port_read_data[16*128 +: 128]};
                5'd9: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[19], port_read_data[19*128 +: 128]} : {port_read_valid[18], port_read_data[18*128 +: 128]};
                5'd10: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[21], port_read_data[21*128 +: 128]} : {port_read_valid[20], port_read_data[20*128 +: 128]};
                5'd11: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[23], port_read_data[23*128 +: 128]} : {port_read_valid[22], port_read_data[22*128 +: 128]};
                5'd12: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[25], port_read_data[25*128 +: 128]} : {port_read_valid[24], port_read_data[24*128 +: 128]};
                5'd13: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[27], port_read_data[27*128 +: 128]} : {port_read_valid[26], port_read_data[26*128 +: 128]};
                5'd14: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[29], port_read_data[29*128 +: 128]} : {port_read_valid[28], port_read_data[28*128 +: 128]};
                5'd15: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[31], port_read_data[31*128 +: 128]} : {port_read_valid[30], port_read_data[30*128 +: 128]};
                5'd16: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[33], port_read_data[33*128 +: 128]} : {port_read_valid[32], port_read_data[32*128 +: 128]};
                5'd17: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[35], port_read_data[35*128 +: 128]} : {port_read_valid[34], port_read_data[34*128 +: 128]};
                5'd18: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[37], port_read_data[37*128 +: 128]} : {port_read_valid[36], port_read_data[36*128 +: 128]};
                5'd19: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[39], port_read_data[39*128 +: 128]} : {port_read_valid[38], port_read_data[38*128 +: 128]};
                5'd20: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[41], port_read_data[41*128 +: 128]} : {port_read_valid[40], port_read_data[40*128 +: 128]};
                5'd21: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[43], port_read_data[43*128 +: 128]} : {port_read_valid[42], port_read_data[42*128 +: 128]};
                5'd22: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[45], port_read_data[45*128 +: 128]} : {port_read_valid[44], port_read_data[44*128 +: 128]};
                5'd23: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[47], port_read_data[47*128 +: 128]} : {port_read_valid[46], port_read_data[46*128 +: 128]};
                5'd24: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[49], port_read_data[49*128 +: 128]} : {port_read_valid[48], port_read_data[48*128 +: 128]};
                5'd25: matmul_expanded_activation_response = word_index[0] ? {port_read_valid[51], port_read_data[51*128 +: 128]} : {port_read_valid[50], port_read_data[50*128 +: 128]};
                default: matmul_expanded_activation_response = '0;
            endcase
        end
    endfunction

    function automatic logic [128:0] matmul_preserved_panel1_response(
        input logic secondary,
        input logic [1:0] stripe
    );
        case ({secondary, stripe})
            3'd0: matmul_preserved_panel1_response = {port_read_valid[27], port_read_data[27*128 +: 128]};
            3'd1: matmul_preserved_panel1_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
            3'd2: matmul_preserved_panel1_response = {port_read_valid[9], port_read_data[9*128 +: 128]};
            3'd3: matmul_preserved_panel1_response = {port_read_valid[3], port_read_data[3*128 +: 128]};
            3'd4: matmul_preserved_panel1_response = {port_read_valid[26], port_read_data[26*128 +: 128]};
            3'd5: matmul_preserved_panel1_response = {port_read_valid[20], port_read_data[20*128 +: 128]};
            3'd6: matmul_preserved_panel1_response = {port_read_valid[8], port_read_data[8*128 +: 128]};
            default: matmul_preserved_panel1_response = {port_read_valid[2], port_read_data[2*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_panel_response(
        input logic panel,
        input logic secondary,
        input logic [2:0] stripe
    );
        case ({panel, secondary, stripe})
            5'b00000: matmul_panel_response = {port_read_valid[49], port_read_data[49*128 +: 128]};
            5'b00001: matmul_panel_response = {port_read_valid[33], port_read_data[33*128 +: 128]};
            5'b00010: matmul_panel_response = {port_read_valid[51], port_read_data[51*128 +: 128]};
            5'b00011: matmul_panel_response = {port_read_valid[15], port_read_data[15*128 +: 128]};
            5'b00100: matmul_panel_response = {port_read_valid[8], port_read_data[8*128 +: 128]};
            5'b00101: matmul_panel_response = {port_read_valid[14], port_read_data[14*128 +: 128]};
            5'b01000: matmul_panel_response = {port_read_valid[48], port_read_data[48*128 +: 128]};
            5'b01001: matmul_panel_response = {port_read_valid[32], port_read_data[32*128 +: 128]};
            5'b01010: matmul_panel_response = {port_read_valid[50], port_read_data[50*128 +: 128]};
            5'b01011: matmul_panel_response = {port_read_valid[14], port_read_data[14*128 +: 128]};
            5'b01100: matmul_panel_response = {port_read_valid[9], port_read_data[9*128 +: 128]};
            5'b01101: matmul_panel_response = {port_read_valid[15], port_read_data[15*128 +: 128]};
            5'b10000: matmul_panel_response = {port_read_valid[44], port_read_data[44*128 +: 128]};
            5'b10001: matmul_panel_response = {port_read_valid[20], port_read_data[20*128 +: 128]};
            5'b10010: matmul_panel_response = {port_read_valid[46], port_read_data[46*128 +: 128]};
            5'b10011: matmul_panel_response = {port_read_valid[36], port_read_data[36*128 +: 128]};
            5'b10100: matmul_panel_response = {port_read_valid[2], port_read_data[2*128 +: 128]};
            5'b10101: matmul_panel_response = {port_read_valid[20], port_read_data[20*128 +: 128]};
            5'b11000: matmul_panel_response = {port_read_valid[45], port_read_data[45*128 +: 128]};
            5'b11001: matmul_panel_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
            5'b11010: matmul_panel_response = {port_read_valid[47], port_read_data[47*128 +: 128]};
            5'b11011: matmul_panel_response = {port_read_valid[37], port_read_data[37*128 +: 128]};
            5'b11100: matmul_panel_response = {port_read_valid[3], port_read_data[3*128 +: 128]};
            default: matmul_panel_response = {port_read_valid[21], port_read_data[21*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_expanded_panel_response(
        input logic [3:0] lane
    );
        case (lane)
            4'd0: matmul_expanded_panel_response = {port_read_valid[44], port_read_data[44*128 +: 128]};
            4'd1: matmul_expanded_panel_response = {port_read_valid[46], port_read_data[46*128 +: 128]};
            4'd2: matmul_expanded_panel_response = {port_read_valid[48], port_read_data[48*128 +: 128]};
            4'd3: matmul_expanded_panel_response = {port_read_valid[50], port_read_data[50*128 +: 128]};
            4'd6: matmul_expanded_panel_response = {port_read_valid[45], port_read_data[45*128 +: 128]};
            4'd7: matmul_expanded_panel_response = {port_read_valid[47], port_read_data[47*128 +: 128]};
            4'd8: matmul_expanded_panel_response = {port_read_valid[49], port_read_data[49*128 +: 128]};
            4'd9: matmul_expanded_panel_response = {port_read_valid[51], port_read_data[51*128 +: 128]};
            default: matmul_expanded_panel_response = '0;
        endcase
    endfunction

    function automatic logic [512:0] r4_product_response_bundle(
        input logic group,
        input logic half
    );
        case ({group, half})
            2'b00: r4_product_response_bundle = {
                port_read_valid[49] && port_read_valid[33] && port_read_valid[51] && port_read_valid[15],
                port_read_data[15*128 +: 128],
                port_read_data[51*128 +: 128],
                port_read_data[33*128 +: 128],
                port_read_data[49*128 +: 128]};
            2'b01: r4_product_response_bundle = {
                port_read_valid[44] && port_read_valid[20] && port_read_valid[46] && port_read_valid[36],
                port_read_data[36*128 +: 128],
                port_read_data[46*128 +: 128],
                port_read_data[20*128 +: 128],
                port_read_data[44*128 +: 128]};
            default: r4_product_response_bundle = {
                port_read_valid[11] && port_read_valid[17] && port_read_valid[23] && port_read_valid[25],
                port_read_data[25*128 +: 128],
                port_read_data[23*128 +: 128],
                port_read_data[17*128 +: 128],
                port_read_data[11*128 +: 128]};
        endcase
    endfunction

    function automatic logic [512:0] elementwise_product_response_bundle(
        input logic [2:0] group,
        input logic half
    );
        case ({group, half})
            4'b000_0: elementwise_product_response_bundle = {
                port_read_valid[0] && port_read_valid[13] && port_read_valid[19] && port_read_valid[1],
                port_read_data[1*128 +: 128],
                port_read_data[19*128 +: 128],
                port_read_data[13*128 +: 128],
                port_read_data[0*128 +: 128]};
            4'b000_1: elementwise_product_response_bundle = {
                port_read_valid[9] && port_read_valid[7] && port_read_valid[23] && port_read_valid[6],
                port_read_data[6*128 +: 128],
                port_read_data[23*128 +: 128],
                port_read_data[7*128 +: 128],
                port_read_data[9*128 +: 128]};
            4'b001_0: elementwise_product_response_bundle = {
                port_read_valid[43] && port_read_valid[41] && port_read_valid[11] && port_read_valid[2],
                port_read_data[2*128 +: 128],
                port_read_data[11*128 +: 128],
                port_read_data[41*128 +: 128],
                port_read_data[43*128 +: 128]};
            4'b001_1: elementwise_product_response_bundle = {
                port_read_valid[27] && port_read_valid[29] && port_read_valid[5] && port_read_valid[35],
                port_read_data[35*128 +: 128],
                port_read_data[5*128 +: 128],
                port_read_data[29*128 +: 128],
                port_read_data[27*128 +: 128]};
            4'b010_0: elementwise_product_response_bundle = {
                port_read_valid[4] && port_read_valid[19] && port_read_valid[39] && port_read_valid[25],
                port_read_data[25*128 +: 128],
                port_read_data[39*128 +: 128],
                port_read_data[19*128 +: 128],
                port_read_data[4*128 +: 128]};
            4'b010_1: elementwise_product_response_bundle = {
                port_read_valid[17] && port_read_valid[13] && port_read_valid[31] && port_read_valid[27],
                port_read_data[27*128 +: 128],
                port_read_data[31*128 +: 128],
                port_read_data[13*128 +: 128],
                port_read_data[17*128 +: 128]};
            4'b011_0, 4'b011_1: elementwise_product_response_bundle = {
                port_read_valid[49] && port_read_valid[33] && port_read_valid[51] && port_read_valid[15],
                port_read_data[15*128 +: 128],
                port_read_data[51*128 +: 128],
                port_read_data[33*128 +: 128],
                port_read_data[49*128 +: 128]};
            4'b100_0, 4'b100_1: elementwise_product_response_bundle = {
                port_read_valid[44] && port_read_valid[20] && port_read_valid[46] && port_read_valid[36],
                port_read_data[36*128 +: 128],
                port_read_data[46*128 +: 128],
                port_read_data[20*128 +: 128],
                port_read_data[44*128 +: 128]};
            default: elementwise_product_response_bundle = {
                port_read_valid[8] && port_read_valid[14] && port_read_valid[2] && port_read_valid[20],
                port_read_data[20*128 +: 128],
                port_read_data[2*128 +: 128],
                port_read_data[14*128 +: 128],
                port_read_data[8*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_qkv_panel_response(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: matmul_qkv_panel_response = {port_read_valid[40], port_read_data[40*128 +: 128]};
            2'd1: matmul_qkv_panel_response = {port_read_valid[42], port_read_data[42*128 +: 128]};
            2'd2: matmul_qkv_panel_response = {port_read_valid[30], port_read_data[30*128 +: 128]};
            default: matmul_qkv_panel_response = {port_read_valid[50], port_read_data[50*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_qkv_panel_secondary_response(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: matmul_qkv_panel_secondary_response = {port_read_valid[41], port_read_data[41*128 +: 128]};
            2'd1: matmul_qkv_panel_secondary_response = {port_read_valid[43], port_read_data[43*128 +: 128]};
            2'd2: matmul_qkv_panel_secondary_response = {port_read_valid[31], port_read_data[31*128 +: 128]};
            default: matmul_qkv_panel_secondary_response = {port_read_valid[51], port_read_data[51*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_q_work_panel_response(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: matmul_q_work_panel_response =
                {port_read_valid[Q_WORK0_PRIMARY_PORT],
                 port_read_data[Q_WORK0_PRIMARY_PORT*128 +: 128]};
            2'd1: matmul_q_work_panel_response =
                {port_read_valid[Q_WORK1_PRIMARY_PORT],
                 port_read_data[Q_WORK1_PRIMARY_PORT*128 +: 128]};
            2'd2: matmul_q_work_panel_response =
                {port_read_valid[Q_WORK2_PRIMARY_PORT],
                 port_read_data[Q_WORK2_PRIMARY_PORT*128 +: 128]};
            default: matmul_q_work_panel_response =
                {port_read_valid[Q_WORK3_PRIMARY_PORT],
                 port_read_data[Q_WORK3_PRIMARY_PORT*128 +: 128]};
        endcase
    endfunction

    function automatic logic [128:0] matmul_q_work_panel_secondary_response(
        input logic [1:0] stripe
    );
        case (stripe)
            2'd0: matmul_q_work_panel_secondary_response =
                {port_read_valid[Q_WORK0_SECONDARY_PORT],
                 port_read_data[Q_WORK0_SECONDARY_PORT*128 +: 128]};
            2'd1: matmul_q_work_panel_secondary_response =
                {port_read_valid[Q_WORK1_SECONDARY_PORT],
                 port_read_data[Q_WORK1_SECONDARY_PORT*128 +: 128]};
            2'd2: matmul_q_work_panel_secondary_response =
                {port_read_valid[Q_WORK2_SECONDARY_PORT],
                 port_read_data[Q_WORK2_SECONDARY_PORT*128 +: 128]};
            default: matmul_q_work_panel_secondary_response =
                {port_read_valid[Q_WORK3_SECONDARY_PORT],
                 port_read_data[Q_WORK3_SECONDARY_PORT*128 +: 128]};
        endcase
    endfunction

    logic [1:0] rms_tile_read_pending;
    logic [1:0] rms_tile_read_pending_gamma_bypass;
    logic [5:0] rms_tile_read_pending_row_base [0:1];
    logic [15:0] rms_tile_read_pending_tag [0:1];
    logic [63:0] rms_tile_read_pending_mask [0:1];
    logic rms_tile_response_complete;
    logic rms_tile_fifo_input_valid;
    logic rms_tile_fifo_input_ready;
    logic [1231:0] rms_tile_fifo_input_data;
    logic [128:0] rms_aligned_hidden_response [0:7];
    logic [2:0] rms_tile_fifo_occupancy;
    logic [3:0] rms_tile_reserved_count;
    local_sram_region_t rms_gamma_region_value;
    logic [11:0] rms_gamma_address;
    logic [11:0] rms_gamma_read_address;
    logic [127:0] rms_gamma_write_mask_n;
    logic [7:0] rms_tile_lane_valid;
    local_sram_region_t rms_tile_region [0:7];
    logic [5:0] rms_tile_physical_row [0:7];
    logic [5:0] rms_tile_port [0:7];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0] rms_tile_macro_onehot [0:7];
    logic rms_tile_port_side [0:7];
    logic [11:0] rms_tile_address [0:7];

    logic [1:0] rms_scratch_read_pending;
    logic rms_scratch_read_pending_bank [0:1];
    logic [15:0] rms_scratch_read_pending_tag [0:1];
    logic rms_scratch_response_complete;
    logic rms_scratch_fifo_input_valid;
    logic rms_scratch_fifo_input_ready;
    logic [271:0] rms_scratch_fifo_input_data;
    logic [2:0] rms_scratch_fifo_occupancy;
    logic [3:0] rms_scratch_reserved_count;
    local_sram_region_t rms_scratch_read_region_value;
    local_sram_region_t rms_scratch_write_region_value;
    logic [11:0] rms_scratch_read_left_row;
    logic [11:0] rms_scratch_read_right_row;
    logic [11:0] rms_scratch_write_row;
    logic [5:0] rms_scratch_read_primary_port;
    logic [5:0] rms_scratch_read_secondary_port;
    logic [5:0] rms_scratch_write_port;
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        rms_scratch_read_primary_macro_onehot;
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        rms_scratch_read_secondary_macro_onehot;
    logic [LOCAL_SRAM_MACRO_COUNT-1:0] rms_scratch_write_macro_onehot;
    logic [127:0] rms_scratch_write_mask_word;
    logic rms_phase_active;
    logic [7:0] rms_norm_write_lane_valid;
    local_sram_region_t rms_norm_write_region [0:7];
    logic [5:0] rms_norm_physical_row [0:7];
    logic [5:0] rms_norm_write_port [0:7];
    logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        rms_norm_write_macro_onehot [0:7];
    logic rms_norm_write_port_side [0:7];
    logic [11:0] rms_norm_write_address [0:7];
    logic [127:0] rms_norm_write_lane_mask_n [0:7];

    assign rms_role_busy = rms_gamma_stage_valid || rms_tile_read_req_valid ||
        rms_scratch_read_valid || rms_scratch_write_valid ||
        rms_norm_write_valid || (|rms_tile_read_pending) ||
        (|rms_scratch_read_pending) || rms_tile_fifo_occupancy != 3'd0 ||
        rms_scratch_fifo_occupancy != 3'd0;
    assign hidden_role_busy = hidden_write_valid || hidden_read_valid ||
        (|hidden_read_pending);
    assign context_role_busy = context_write_valid || context_read_req_valid ||
        (|context_read_pending) || context_read_rsp_valid;
    assign attention_scratch_role_busy = attention_scratch_read_valid ||
        attention_scratch_write_valid ||
        (|attention_scratch_read_pending) ||
        attention_scratch_fifo_occupancy != 3'd0;
    assign qkv_head_role_busy = qkv_head_read_req_valid ||
        (|qkv_head_read_pending) || qkv_head_read_fifo_occupancy != 3'd0;
    assign rope_role_busy = rope_source_read_req_valid ||
        rope_cos_read_req_valid || rope_sin_read_req_valid ||
        rope_destination_write_valid || rope_source_read_pending ||
        rope_cos_read_pending || rope_sin_read_pending;
    assign qkv_stage_role_busy = qkv_head_stage_write_valid ||
        qkv_head_tile_read_req_valid || (|qkv_tile_read_pending);
    assign qkv_cache_write_busy = qkv_constant_write_valid || qkv_q_write_valid;
    assign attention_operand_role_busy = attention_operand_req_valid ||
        (|attention_operand_read_pending);
    assign attention_score_role_busy = attention_score_write_valid ||
        attention_score_read_req_valid || (|attention_score_read_pending);
    assign attention_probability_role_busy =
        attention_probability_write_valid ||
        attention_probability_quantized_write_valid ||
        attention_probability_scale_write_valid;
    assign attention_panel_write_busy = attention_panel_write_valid ||
        attention_panel_scale_write_valid;
    assign matmul_non_panel_role_busy = matmul_local_source_req_valid ||
        matmul_activation_write_valid || matmul_read_bundle_valid ||
        matmul_combine_write_valid || matmul_combine_read_valid ||
        lm_head_scale_write_valid || lm_head_scale_read_valid ||
        post_table_access_valid || post_state_write_valid ||
        post_state_read_valid || post_suppressed_write_valid ||
        post_suppressed_read_valid || (|post_state_read_pending) ||
        (|post_suppressed_read_pending) ||
        candidate_state_read_valid || candidate_state_write_valid ||
        matmul_local_output_write_valid || matmul_residual_read_valid ||
        (|matmul_source_read_pending) || (|matmul_operand_read_pending) ||
        (|matmul_combine_read_pending) ||
        (|lm_head_scale_read_pending) ||
        candidate_state_read_reserved_count != 3'd0 ||
        (|matmul_residual_read_pending);
    assign matmul_panel_role_busy = matmul_panel_write_valid;
    assign elementwise_product_role_busy = elementwise_product_req_valid ||
        (|elementwise_product_read_pending) ||
        elementwise_product_fifo_occupancy != 0;

    assign current_attention_phase =
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK ||
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX ||
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV;
    assign next_attention_phase =
        next_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK ||
        next_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX ||
        next_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV;
    assign panel_mapping_changed =
        current_layout.panel_layout != next_layout.panel_layout;
    assign attention_pair_mapping_changed =
        current_layout.kv_pair_enable != next_layout.kv_pair_enable ||
        current_layout.kv_second_batch != next_layout.kv_second_batch ||
        current_layout.kv_first_batch_tokens != next_layout.kv_first_batch_tokens ||
        current_layout.attention_pair_enable != next_layout.attention_pair_enable ||
        current_layout.attention_second_batch != next_layout.attention_second_batch ||
        current_layout.attention_batch_index != next_layout.attention_batch_index ||
        current_layout.attention_score_words != next_layout.attention_score_words;
    assign attention_pair_role_drained = !attention_pair_mapping_changed ||
        (!qkv_head_role_busy && !rope_role_busy && !qkv_stage_role_busy &&
         !qkv_cache_write_busy && !attention_operand_role_busy &&
         !attention_score_role_busy && !attention_probability_role_busy &&
         !attention_scratch_role_busy && !attention_panel_write_busy &&
         !context_role_busy && !matmul_non_panel_role_busy &&
         !matmul_panel_role_busy && !hidden_role_busy && !rms_role_busy &&
         !elementwise_product_role_busy);

    always_comb begin : layout_phase_drain
        phase_role_drained = 1'b1;
        if (current_layout.phase != next_layout.phase) begin
            case (current_layout.phase)
                LOCAL_MEMORY_PHASE_RMSNORM:
                    phase_role_drained = !rms_role_busy && !hidden_role_busy;
                LOCAL_MEMORY_PHASE_FINAL_OUTPUT:
                    phase_role_drained = !rms_role_busy && !hidden_role_busy &&
                        !matmul_non_panel_role_busy;
                LOCAL_MEMORY_PHASE_QKV_PREPARATION:
                    phase_role_drained = !qkv_head_role_busy &&
                        !rope_role_busy && !qkv_stage_role_busy &&
                        !qkv_cache_write_busy &&
                        !matmul_non_panel_role_busy && !matmul_panel_role_busy;
                LOCAL_MEMORY_PHASE_ATTENTION_QK:
                    phase_role_drained = !attention_operand_role_busy &&
                        !attention_score_role_busy;
                LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX:
                    phase_role_drained = !attention_scratch_role_busy &&
                        !attention_score_role_busy &&
                        !attention_probability_role_busy;
                LOCAL_MEMORY_PHASE_ATTENTION_PV:
                    phase_role_drained = !attention_operand_role_busy;
                LOCAL_MEMORY_PHASE_MATMUL,
                LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN,
                LOCAL_MEMORY_PHASE_R4,
                LOCAL_MEMORY_PHASE_R4_TWO_TOKENS,
                LOCAL_MEMORY_PHASE_ELEMENTWISE:
                    phase_role_drained = !matmul_non_panel_role_busy &&
                        !elementwise_product_role_busy;
                default: phase_role_drained = 1'b1;
            endcase
        end
        if (current_attention_phase && !next_attention_phase)
            phase_role_drained &= !context_role_busy &&
                !attention_panel_write_busy;
    end

    assign qkv_staging_role_drained =
        (current_layout.qkv_head_staging == next_layout.qkv_head_staging &&
        current_layout.q_head_active == next_layout.q_head_active &&
        current_layout.q_head_slot == next_layout.q_head_slot) ||
        (!qkv_head_role_busy && !rope_role_busy && !qkv_stage_role_busy &&
         !qkv_cache_write_busy && !matmul_non_panel_role_busy &&
         !matmul_panel_role_busy);
    assign panel_role_drained = !panel_mapping_changed ||
        (!qkv_head_role_busy && !rope_role_busy && !qkv_stage_role_busy &&
         !qkv_cache_write_busy && !attention_panel_write_busy &&
         !matmul_non_panel_role_busy && !matmul_panel_role_busy &&
         !elementwise_product_role_busy);
    assign layout_update_ready = phase_role_drained &&
        qkv_staging_role_drained && panel_role_drained &&
        attention_pair_role_drained;

    assign hidden_write_port = local_region_port_index(
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            final_output_workspace_region(hidden_write_req.physical_row[2:0]) :
            resident_region(hidden_write_req.physical_row), 1'b0);
    assign hidden_read_port = local_region_port_index(
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            final_output_workspace_region(hidden_read_req.physical_row[2:0]) :
            resident_region(hidden_read_req.physical_row), 1'b0);
    assign hidden_write_address = local_region_row(
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            final_output_workspace_region(hidden_write_req.physical_row[2:0]) :
            resident_region(hidden_write_req.physical_row),
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            {1'b0, hidden_write_req.channel_word} :
            resident_word_row(hidden_write_req.physical_row,
                hidden_write_req.channel_word));
    assign hidden_read_address = local_region_row(
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            final_output_workspace_region(hidden_read_req.physical_row[2:0]) :
            resident_region(hidden_read_req.physical_row),
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
            {1'b0, hidden_read_req.channel_word} :
            resident_word_row(hidden_read_req.physical_row,
                hidden_read_req.channel_word));

    assign hidden_read_macro_onehot =
        physical_port_macro_onehot(hidden_read_port);

    always_comb begin : hidden_cache_write_conflict
        hidden_read_cache_conflict = 1'b0;
        for (integer cache_macro = 0;
             cache_macro < LOCAL_SRAM_MACRO_COUNT; cache_macro++) begin
            if (hidden_read_macro_onehot[cache_macro] &&
                ((cache_write_req_valid[cache_macro*2] &&
                  (!hidden_read_port[0] ||
                   hidden_read_address ==
                       cache_write_address[cache_macro*24 +: 12])) ||
                 (cache_write_req_valid[cache_macro*2+1] &&
                  (hidden_read_port[0] ||
                   hidden_read_address ==
                       cache_write_address[cache_macro*24+12 +: 12]))))
                hidden_read_cache_conflict = 1'b1;
        end
    end

    // Hidden transfer has one logical lane. Decode the resident row into the
    // fixed primary endpoints before driving constant physical-port slices.
    always_comb begin : hidden_endpoint_decode
        hidden_write_port_mask = '0;
        hidden_read_port_mask = '0;
        case (hidden_write_port)
            6'd0: hidden_write_port_mask[0] = 1'b1;
            6'd3: hidden_write_port_mask[3] = 1'b1;
            6'd5: hidden_write_port_mask[5] = 1'b1;
            6'd7: hidden_write_port_mask[7] = 1'b1;
            6'd9: hidden_write_port_mask[9] = 1'b1;
            6'd11: hidden_write_port_mask[11] = 1'b1;
            6'd13: hidden_write_port_mask[13] = 1'b1;
            6'd15: hidden_write_port_mask[15] = 1'b1;
            6'd17: hidden_write_port_mask[17] = 1'b1;
            6'd19: hidden_write_port_mask[19] = 1'b1;
            6'd20: hidden_write_port_mask[20] = 1'b1;
            6'd21: hidden_write_port_mask[21] = 1'b1;
            6'd25: hidden_write_port_mask[25] = 1'b1;
            6'd27: hidden_write_port_mask[27] = 1'b1;
            6'd33: hidden_write_port_mask[33] = 1'b1;
            6'd35: hidden_write_port_mask[35] = 1'b1;
            6'd36: hidden_write_port_mask[36] = 1'b1;
            6'd37: hidden_write_port_mask[37] = 1'b1;
            6'd44: hidden_write_port_mask[44] = 1'b1;
            6'd45: hidden_write_port_mask[45] = 1'b1;
            6'd46: hidden_write_port_mask[46] = 1'b1;
            6'd47: hidden_write_port_mask[47] = 1'b1;
            6'd49: hidden_write_port_mask[49] = 1'b1;
            6'd51: hidden_write_port_mask[51] = 1'b1;
            default: begin end
        endcase
        case (hidden_read_port)
            6'd0: hidden_read_port_mask[0] = 1'b1;
            6'd3: hidden_read_port_mask[3] = 1'b1;
            6'd5: hidden_read_port_mask[5] = 1'b1;
            6'd7: hidden_read_port_mask[7] = 1'b1;
            6'd9: hidden_read_port_mask[9] = 1'b1;
            6'd11: hidden_read_port_mask[11] = 1'b1;
            6'd13: hidden_read_port_mask[13] = 1'b1;
            6'd15: hidden_read_port_mask[15] = 1'b1;
            6'd17: hidden_read_port_mask[17] = 1'b1;
            6'd19: hidden_read_port_mask[19] = 1'b1;
            6'd20: hidden_read_port_mask[20] = 1'b1;
            6'd21: hidden_read_port_mask[21] = 1'b1;
            6'd25: hidden_read_port_mask[25] = 1'b1;
            6'd27: hidden_read_port_mask[27] = 1'b1;
            6'd33: hidden_read_port_mask[33] = 1'b1;
            6'd35: hidden_read_port_mask[35] = 1'b1;
            6'd36: hidden_read_port_mask[36] = 1'b1;
            6'd37: hidden_read_port_mask[37] = 1'b1;
            6'd44: hidden_read_port_mask[44] = 1'b1;
            6'd45: hidden_read_port_mask[45] = 1'b1;
            6'd46: hidden_read_port_mask[46] = 1'b1;
            6'd47: hidden_read_port_mask[47] = 1'b1;
            6'd49: hidden_read_port_mask[49] = 1'b1;
            6'd51: hidden_read_port_mask[51] = 1'b1;
            default: begin end
        endcase
    end

    assign hidden_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM ||
         current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT) &&
        |(hidden_write_port_mask & client_req_ready);
    assign hidden_read_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM ||
         current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT) &&
        !hidden_read_cache_conflict &&
        |(hidden_read_port_mask & client_req_ready);

    for (genvar byte_number = 0; byte_number < 16; byte_number++) begin : g_hidden_mask
        assign hidden_write_mask_word[byte_number*8 +: 8] =
            {8{!hidden_write_req.byte_enable[byte_number]}};
    end

    for (genvar port_number = 0;
         port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         port_number++) begin : g_hidden_port
        assign hidden_req_valid[port_number] = !rst &&
            (current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM ||
             (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION &&
              !hidden_write_valid) ||
             current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT) &&
            ((hidden_write_valid && hidden_write_port_mask[port_number]) ||
             (!hidden_write_valid && hidden_read_valid &&
                !hidden_read_cache_conflict &&
                hidden_read_port_mask[port_number]));
        assign hidden_write[port_number] = hidden_write_valid &&
            hidden_write_port_mask[port_number];
        assign hidden_address[port_number*12 +: 12] =
            hidden_write[port_number] ? hidden_write_address :
                hidden_read_address;
        assign hidden_port_write_data[port_number*128 +: 128] =
            hidden_write[port_number] ? hidden_write_req.data : '0;
        assign hidden_write_mask_n[port_number*128 +: 128] =
            hidden_write[port_number] ? hidden_write_mask_word : '1;
    end

    always_ff @(posedge clk) begin : hidden_response_metadata
        if (rst) begin
            hidden_read_pending <= '0;
            hidden_read_pending_port[0] <= '0;
            hidden_read_pending_port[1] <= '0;
        end else begin
            hidden_read_pending <= {hidden_read_pending[0],
                hidden_read_valid && hidden_read_ready};
            hidden_read_pending_port[1] <= hidden_read_pending_port[0];
            if (hidden_read_valid && hidden_read_ready)
                hidden_read_pending_port[0] <= hidden_read_port;
        end
    end

    always_comb begin : hidden_response_mapping
        hidden_read_response_valid = 1'b0;
        hidden_read_rsp.data = '0;
        case (hidden_read_pending_port[1])
            6'd0: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[0];
                hidden_read_rsp.data = port_read_data[0*128 +: 128];
            end
            6'd3: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[3];
                hidden_read_rsp.data = port_read_data[3*128 +: 128];
            end
            6'd5: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[5];
                hidden_read_rsp.data = port_read_data[5*128 +: 128];
            end
            6'd7: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[7];
                hidden_read_rsp.data = port_read_data[7*128 +: 128];
            end
            6'd9: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[9];
                hidden_read_rsp.data = port_read_data[9*128 +: 128];
            end
            6'd11: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[11];
                hidden_read_rsp.data = port_read_data[11*128 +: 128];
            end
            6'd13: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[13];
                hidden_read_rsp.data = port_read_data[13*128 +: 128];
            end
            6'd15: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[15];
                hidden_read_rsp.data = port_read_data[15*128 +: 128];
            end
            6'd17: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[17];
                hidden_read_rsp.data = port_read_data[17*128 +: 128];
            end
            6'd19: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[19];
                hidden_read_rsp.data = port_read_data[19*128 +: 128];
            end
            6'd20: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[20];
                hidden_read_rsp.data = port_read_data[20*128 +: 128];
            end
            6'd21: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[21];
                hidden_read_rsp.data = port_read_data[21*128 +: 128];
            end
            6'd25: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[25];
                hidden_read_rsp.data = port_read_data[25*128 +: 128];
            end
            6'd27: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[27];
                hidden_read_rsp.data = port_read_data[27*128 +: 128];
            end
            6'd33: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[33];
                hidden_read_rsp.data = port_read_data[33*128 +: 128];
            end
            6'd35: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[35];
                hidden_read_rsp.data = port_read_data[35*128 +: 128];
            end
            6'd36: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[36];
                hidden_read_rsp.data = port_read_data[36*128 +: 128];
            end
            6'd37: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[37];
                hidden_read_rsp.data = port_read_data[37*128 +: 128];
            end
            6'd44: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[44];
                hidden_read_rsp.data = port_read_data[44*128 +: 128];
            end
            6'd45: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[45];
                hidden_read_rsp.data = port_read_data[45*128 +: 128];
            end
            6'd46: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[46];
                hidden_read_rsp.data = port_read_data[46*128 +: 128];
            end
            6'd47: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[47];
                hidden_read_rsp.data = port_read_data[47*128 +: 128];
            end
            6'd49: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[49];
                hidden_read_rsp.data = port_read_data[49*128 +: 128];
            end
            6'd51: begin
                hidden_read_response_valid = hidden_read_pending[1] && port_read_valid[51];
                hidden_read_rsp.data = port_read_data[51*128 +: 128];
            end
            default: begin end
        endcase
    end

    assign context_write_port = current_layout.attention_pair_enable ?
        (current_layout.attention_batch_index == 2'd2 ?
            6'd29 + {2'd0, context_write_req.address[6:4], 1'b0} :
            {2'd0, context_write_req.address[6:4], 1'b1}) : CONTEXT_WRITE_PORT;
    assign context_read_port = current_layout.attention_pair_enable ?
        (current_layout.attention_batch_index == 2'd2 ?
            6'd29 + {2'd0, context_read_req.address[6:4], 1'b0} :
            {2'd0, context_read_req.address[6:4], 1'b1}) : CONTEXT_READ_PORT;
    assign context_write_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV &&
        client_req_ready[context_write_port];
    assign context_read_phase_active =
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK ||
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX ||
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV;
    assign context_read_req_ready = !rst &&
        context_read_phase_active &&
        !(|context_read_pending) && !context_read_rsp_valid &&
        (!current_layout.attention_pair_enable || !context_write_valid) &&
        client_req_ready[context_read_port];

    for (genvar byte_number = 0; byte_number < 16; byte_number++) begin : g_context_mask
        assign context_write_mask_word[byte_number*8 +: 8] =
            {8{!context_write_req.byte_enable[byte_number]}};
    end

    for (genvar port_number = 0;
         port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         port_number++) begin : g_context_port
        assign context_req_valid[port_number] = !rst &&
            ((current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV &&
                port_number == context_write_port && context_write_valid) ||
             (context_read_phase_active && port_number == context_read_port &&
                context_read_req_valid &&
                (!current_layout.attention_pair_enable || !context_write_valid) &&
                !(|context_read_pending) && !context_read_rsp_valid));
        assign context_write[port_number] =
            port_number == context_write_port && context_write_valid;
        assign context_address[port_number*12 +: 12] =
            current_layout.attention_pair_enable ?
                local_region_row(attention_pair_context_region(
                    current_layout.attention_batch_index,
                    context_write_valid ? context_write_req.address[9:4] :
                        context_read_req.address[9:4], current_layout.attention_score_words),
                    {6'd0, context_write_valid ? context_write_req.address[3:0] :
                        context_read_req.address[3:0]}) :
            port_number == context_write_port ?
                local_region_row(attention_context_region(),
                    context_write_req.address) :
                local_region_row(attention_context_region(),
                    context_read_req.address);
        assign context_port_write_data[port_number*128 +: 128] =
            port_number == context_write_port ? context_write_req.data : '0;
        assign context_write_mask_n[port_number*128 +: 128] =
            port_number == context_write_port ? context_write_mask_word : '1;
    end

    // Fixed eight-macro selection feeds the existing registered response.
    always_comb begin
        context_return_valid = port_read_valid[CONTEXT_READ_PORT];
        context_return_data = port_read_data[CONTEXT_READ_PORT*128 +: 128];
        if (current_layout.attention_pair_enable) begin
            context_return_valid = 1'b0;
            context_return_data = '0;
            for (integer lane = 0; lane < 8; lane++) begin
                if (context_read_pending_macro[1] == 3'(lane)) begin
                    if (current_layout.attention_batch_index == 2'd2) begin
                        context_return_valid = port_read_valid[29+lane*2];
                        context_return_data = port_read_data[(29+lane*2)*128 +: 128];
                    end else begin
                        context_return_valid = port_read_valid[lane*2+1];
                        context_return_data = port_read_data[(lane*2+1)*128 +: 128];
                    end
                end
            end
        end
    end

    always_ff @(posedge clk) begin : context_response
        if (rst) begin
            context_read_pending <= '0;
            context_read_pending_tag[0] <= '0;
            context_read_pending_tag[1] <= '0;
            context_read_pending_macro[0] <= '0;
            context_read_pending_macro[1] <= '0;
            context_read_rsp_valid <= 1'b0;
            context_read_rsp <= '0;
        end else begin
            context_read_pending <= {context_read_pending[0],
                context_read_req_valid && context_read_req_ready};
            context_read_pending_tag[1] <= context_read_pending_tag[0];
            context_read_pending_macro[1] <= context_read_pending_macro[0];
            if (context_read_req_valid && context_read_req_ready)
                context_read_pending_macro[0] <= context_read_req.address[6:4];
            if (context_read_req_valid && context_read_req_ready)
                context_read_pending_tag[0] <= context_read_req.tag;
            if (context_read_rsp_valid && context_read_rsp_ready)
                context_read_rsp_valid <= 1'b0;
            if (context_read_pending[1] && context_return_valid) begin
                context_read_rsp_valid <= 1'b1;
                context_read_rsp.data <= context_return_data;
                context_read_rsp.tag <= context_read_pending_tag[1];
            end
        end
    end

    always_comb begin : attention_scratch_ready_mapping
        attention_scratch_read_ready = 1'b0;
        attention_scratch_write_ready = 1'b0;
        if (!rst && current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX) begin
            if (attention_scratch_read_req.bank == 1'b0)
                attention_scratch_read_ready =
                    client_req_ready[ATTENTION_SCRATCH0_PRIMARY_PORT] &&
                    client_req_ready[ATTENTION_SCRATCH0_SECONDARY_PORT];
            else
                attention_scratch_read_ready =
                    client_req_ready[ATTENTION_SCRATCH1_PRIMARY_PORT] &&
                    client_req_ready[ATTENTION_SCRATCH1_SECONDARY_PORT];
            if (attention_scratch_write_req.bank == 1'b0)
                attention_scratch_write_ready =
                    client_req_ready[ATTENTION_SCRATCH0_PRIMARY_PORT];
            else
                attention_scratch_write_ready =
                    client_req_ready[ATTENTION_SCRATCH1_PRIMARY_PORT];
        end
        attention_scratch_read_ready &=
            attention_scratch_reserved_count < 4'd3 ||
            (attention_scratch_rsp_valid && attention_scratch_rsp_ready);
    end

    for (genvar byte_number = 0; byte_number < 16; byte_number++) begin : g_attention_scratch_mask
        assign attention_scratch_write_mask_word[byte_number*8 +: 8] =
            {8{!attention_scratch_write_req.byte_enable[byte_number]}};
    end

    for (genvar port_number = 0;
         port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         port_number++) begin : g_attention_scratch_port
        logic read_port_selected;
        logic write_port_selected;
        logic secondary_read_port;

        assign read_port_selected = attention_scratch_read_req.bank == 1'b0 ?
            (port_number == ATTENTION_SCRATCH0_PRIMARY_PORT ||
             port_number == ATTENTION_SCRATCH0_SECONDARY_PORT) :
            (port_number == ATTENTION_SCRATCH1_PRIMARY_PORT ||
             port_number == ATTENTION_SCRATCH1_SECONDARY_PORT);
        assign write_port_selected = attention_scratch_write_req.bank == 1'b0 ?
            port_number == ATTENTION_SCRATCH0_PRIMARY_PORT :
            port_number == ATTENTION_SCRATCH1_PRIMARY_PORT;
        assign secondary_read_port =
            port_number == ATTENTION_SCRATCH0_SECONDARY_PORT ||
            port_number == ATTENTION_SCRATCH1_SECONDARY_PORT;
        assign attention_scratch_req_valid[port_number] = !rst &&
            current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
            ((attention_scratch_write_valid && write_port_selected) ||
             (attention_scratch_read_valid &&
                (attention_scratch_reserved_count < 4'd3 ||
                 (attention_scratch_rsp_valid &&
                    attention_scratch_rsp_ready)) && read_port_selected));
        assign attention_scratch_write[port_number] =
            attention_scratch_write_valid && write_port_selected;
        assign attention_scratch_address[port_number*12 +: 12] =
            attention_scratch_write[port_number] ?
                local_region_row(attention_scratch_regions[
                    attention_scratch_write_req.bank],
                    attention_scratch_write_req.address) :
                local_region_row(attention_scratch_regions[
                    attention_scratch_read_req.bank], secondary_read_port ?
                        attention_scratch_read_req.right_address :
                        attention_scratch_read_req.left_address);
        assign attention_scratch_port_write_data[port_number*128 +: 128] =
            attention_scratch_write[port_number] ?
                attention_scratch_write_req.data : '0;
        assign attention_scratch_write_mask_n[port_number*128 +: 128] =
            attention_scratch_write[port_number] ?
                attention_scratch_write_mask_word : '1;
    end

    always_ff @(posedge clk) begin : attention_scratch_response
        if (rst) begin
            attention_scratch_read_pending <= '0;
            attention_scratch_read_pending_bank[0] <= 1'b0;
            attention_scratch_read_pending_bank[1] <= 1'b0;
            attention_scratch_read_pending_tag[0] <= '0;
            attention_scratch_read_pending_tag[1] <= '0;
        end else begin
            attention_scratch_read_pending <= {
                attention_scratch_read_pending[0],
                attention_scratch_read_valid &&
                    attention_scratch_read_ready};
            attention_scratch_read_pending_bank[1] <=
                attention_scratch_read_pending_bank[0];
            attention_scratch_read_pending_tag[1] <=
                attention_scratch_read_pending_tag[0];
            if (attention_scratch_read_valid &&
                attention_scratch_read_ready) begin
                attention_scratch_read_pending_bank[0] <=
                    attention_scratch_read_req.bank;
                attention_scratch_read_pending_tag[0] <=
                    attention_scratch_read_req.tag;
            end
        end
    end

    assign attention_scratch_reserved_count =
        {1'b0, attention_scratch_fifo_occupancy} +
        4'($countones(attention_scratch_read_pending));
    assign attention_scratch_fifo_input_valid =
        attention_scratch_read_pending[1];
    assign attention_scratch_fifo_input_data =
        attention_scratch_read_pending_bank[1] ? {
            attention_scratch_read_pending_tag[1],
            port_read_data[ATTENTION_SCRATCH1_SECONDARY_PORT*128 +: 128],
            port_read_data[ATTENTION_SCRATCH1_PRIMARY_PORT*128 +: 128]
        } : {
            attention_scratch_read_pending_tag[1],
            port_read_data[ATTENTION_SCRATCH0_SECONDARY_PORT*128 +: 128],
            port_read_data[ATTENTION_SCRATCH0_PRIMARY_PORT*128 +: 128]
        };

    ready_valid_fifo #(.DATA_WIDTH(272), .DEPTH(3))
        attention_scratch_response_fifo (
        .clk(clk), .rst(rst),
        .input_valid(attention_scratch_fifo_input_valid),
        .input_ready(attention_scratch_fifo_input_ready),
        .input_data(attention_scratch_fifo_input_data),
        .output_valid(attention_scratch_rsp_valid),
        .output_ready(attention_scratch_rsp_ready),
        .output_data({attention_scratch_rsp.tag,
            attention_scratch_rsp.right_data,
            attention_scratch_rsp.left_data}),
        .occupancy(attention_scratch_fifo_occupancy)
    );

    // QKV reads one 128-bit word from either the fixed pre-RoPE source or
    // post-RoPE destination. Three total reservations cover the two-cycle
    // macro response and one consumer-held return while preserving II=1.
    always_comb begin : qkv_head_read_mapping
        qkv_head_read_port_valid = '0;
        qkv_head_read_address = '0;
        if (!rst && (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
                     current_layout.q_head_active) &&
            qkv_head_read_req_valid && qkv_head_read_reservation_available) begin
            if (qkv_head_read_req.rope_destination) begin
                qkv_head_read_port_valid[QKV_HEAD_DESTINATION_PORT] = 1'b1;
                qkv_head_read_address[
                    QKV_HEAD_DESTINATION_PORT*12 +: 12] = local_region_row(
                        q_destination_region,
                        {qkv_head_read_req.physical_row, qkv_head_read_req.word});
            end else begin
                qkv_head_read_port_valid[QKV_HEAD_SOURCE_PORT] = 1'b1;
                qkv_head_read_address[QKV_HEAD_SOURCE_PORT*12 +: 12] =
                    local_region_row(q_post_source_region,
                        {qkv_head_read_req.physical_row, qkv_head_read_req.word});
            end
        end
    end

    assign qkv_head_read_reserved_count =
        4'(qkv_head_read_fifo_occupancy) +
        4'($countones(qkv_head_read_pending));
    assign qkv_head_read_reservation_available =
        qkv_head_read_reserved_count < 4'd3 ||
        (qkv_head_read_rsp_valid && qkv_head_read_rsp_ready);
    assign qkv_head_read_req_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        qkv_head_read_reservation_available &&
        (qkv_head_read_req.rope_destination ?
            client_req_ready[QKV_HEAD_DESTINATION_PORT] :
            client_req_ready[QKV_HEAD_SOURCE_PORT]);
    assign qkv_head_read_return_valid = qkv_head_read_pending[1] &&
        (qkv_head_read_pending_destination[1] ?
            port_read_valid[QKV_HEAD_DESTINATION_PORT] :
            port_read_valid[QKV_HEAD_SOURCE_PORT]);
    assign qkv_head_read_return_data = qkv_head_read_pending_destination[1] ?
        port_read_data[QKV_HEAD_DESTINATION_PORT*128 +: 128] :
        port_read_data[QKV_HEAD_SOURCE_PORT*128 +: 128];
    assign qkv_head_read_rsp.lane_mask = 8'hff;

    ready_valid_fifo #(.DATA_WIDTH(128), .DEPTH(3)) qkv_head_read_response_fifo (
        .clk(clk), .rst(rst),
        .input_valid(qkv_head_read_return_valid),
        .input_ready(qkv_head_read_fifo_input_ready),
        .input_data(qkv_head_read_return_data),
        .output_valid(qkv_head_read_rsp_valid),
        .output_ready(qkv_head_read_rsp_ready),
        .output_data(qkv_head_read_rsp.data),
        .occupancy(qkv_head_read_fifo_occupancy)
    );

    always_ff @(posedge clk) begin : qkv_head_read_response
        if (rst) begin
            qkv_head_read_pending <= '0;
            qkv_head_read_pending_destination[0] <= 1'b0;
            qkv_head_read_pending_destination[1] <= 1'b0;
        end else begin
            qkv_head_read_pending <= {qkv_head_read_pending[0],
                qkv_head_read_req_valid && qkv_head_read_req_ready};
            qkv_head_read_pending_destination[1] <=
                qkv_head_read_pending_destination[0];
            if (qkv_head_read_req_valid && qkv_head_read_req_ready)
                qkv_head_read_pending_destination[0] <=
                    qkv_head_read_req.rope_destination;
        end
    end

    always_comb begin : rope_port_mapping
        rope_req_valid = '0;
        rope_write = '0;
        rope_address = '0;
        rope_write_data = '0;
        rope_write_mask_n = '1;
        rope_constant_word = {rope_constant_read_req.physical_row, 4'b0000} +
            {6'd0, rope_constant_read_req.pair_base[6:3]};

        if (!rst && (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
                     current_layout.q_head_active)) begin
            if (rope_source_read_req_valid && !rope_source_read_pending) begin
                rope_req_valid[ROPE_SOURCE_PRIMARY_PORT] = 1'b1;
                rope_req_valid[ROPE_SOURCE_SECONDARY_PORT] = 1'b1;
                rope_address[ROPE_SOURCE_PRIMARY_PORT*12 +: 12] =
                    local_region_row(q_post_source_region,
                        10'(rope_source_read_req.lane_address[17:0] >> 3));
                rope_address[ROPE_SOURCE_SECONDARY_PORT*12 +: 12] =
                    local_region_row(q_post_source_region,
                        10'(rope_source_read_req.lane_address[8*18 +: 18] >> 3));
            end
            if (rope_cos_read_req_valid && !rope_cos_read_pending) begin
                rope_req_valid[ROPE_CONSTANT_PRIMARY_PORT] = 1'b1;
                rope_address[ROPE_CONSTANT_PRIMARY_PORT*12 +: 12] =
                    local_region_row(rope_table_region, rope_constant_word);
            end
            if (rope_sin_read_req_valid && !rope_sin_read_pending) begin
                rope_req_valid[ROPE_CONSTANT_SECONDARY_PORT] = 1'b1;
                rope_address[ROPE_CONSTANT_SECONDARY_PORT*12 +: 12] =
                    local_region_row(rope_table_region,
                        rope_constant_word + 10'd8);
            end
            if (rope_destination_write_valid) begin
                rope_req_valid[ROPE_DESTINATION_PRIMARY_PORT] = 1'b1;
                rope_req_valid[ROPE_DESTINATION_SECONDARY_PORT] = 1'b1;
                rope_write[ROPE_DESTINATION_PRIMARY_PORT] = 1'b1;
                rope_write[ROPE_DESTINATION_SECONDARY_PORT] = 1'b1;
                rope_address[ROPE_DESTINATION_PRIMARY_PORT*12 +: 12] =
                    local_region_row(q_destination_region,
                        10'(rope_destination_write.lane_address[17:0] >> 3));
                rope_address[ROPE_DESTINATION_SECONDARY_PORT*12 +: 12] =
                    local_region_row(q_destination_region,
                        10'(rope_destination_write.lane_address[
                            8*18 +: 18] >> 3));
                rope_write_data[ROPE_DESTINATION_PRIMARY_PORT*128 +: 128] =
                    rope_destination_write.lane_data[127:0];
                rope_write_data[ROPE_DESTINATION_SECONDARY_PORT*128 +: 128] =
                    rope_destination_write.lane_data[255:128];
                rope_write_mask_n[
                    ROPE_DESTINATION_PRIMARY_PORT*128 +: 128] = '0;
                rope_write_mask_n[
                    ROPE_DESTINATION_SECONDARY_PORT*128 +: 128] = '0;
            end
        end
    end

    assign rope_source_read_req_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        !rope_source_read_pending &&
        client_req_ready[ROPE_SOURCE_PRIMARY_PORT] &&
        client_req_ready[ROPE_SOURCE_SECONDARY_PORT];
    assign rope_cos_read_req_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        !rope_cos_read_pending && client_req_ready[ROPE_CONSTANT_PRIMARY_PORT];
    assign rope_sin_read_req_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        !rope_sin_read_pending &&
        client_req_ready[ROPE_CONSTANT_SECONDARY_PORT];
    assign rope_destination_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        client_req_ready[ROPE_DESTINATION_PRIMARY_PORT] &&
        client_req_ready[ROPE_DESTINATION_SECONDARY_PORT];

    assign rope_source_read_rsp_valid = rope_source_read_pending &&
        port_read_valid[ROPE_SOURCE_PRIMARY_PORT] &&
        port_read_valid[ROPE_SOURCE_SECONDARY_PORT];
    assign rope_source_read_rsp.lane_mask = 16'hffff;
    assign rope_source_read_rsp.data = {
        port_read_data[ROPE_SOURCE_SECONDARY_PORT*128 +: 128],
        port_read_data[ROPE_SOURCE_PRIMARY_PORT*128 +: 128]
    };
    assign rope_cos_read_rsp_valid = rope_cos_read_pending &&
        port_read_valid[ROPE_CONSTANT_PRIMARY_PORT];
    assign rope_cos_read_rsp.lane_mask = 8'hff;
    assign rope_cos_read_rsp.data =
        port_read_data[ROPE_CONSTANT_PRIMARY_PORT*128 +: 128];
    assign rope_sin_read_rsp_valid = rope_sin_read_pending &&
        port_read_valid[ROPE_CONSTANT_SECONDARY_PORT];
    assign rope_sin_read_rsp.lane_mask = 8'hff;
    assign rope_sin_read_rsp.data =
        port_read_data[ROPE_CONSTANT_SECONDARY_PORT*128 +: 128];

    always_ff @(posedge clk) begin : rope_response_state
        if (rst) begin
            rope_source_read_pending <= 1'b0;
            rope_cos_read_pending <= 1'b0;
            rope_sin_read_pending <= 1'b0;
        end else begin
            if (rope_source_read_req_valid && rope_source_read_req_ready)
                rope_source_read_pending <= 1'b1;
            if (rope_source_read_rsp_valid && rope_source_read_rsp_ready)
                rope_source_read_pending <= 1'b0;
            if (rope_cos_read_req_valid && rope_cos_read_req_ready)
                rope_cos_read_pending <= 1'b1;
            if (rope_cos_read_rsp_valid && rope_cos_read_rsp_ready)
                rope_cos_read_pending <= 1'b0;
            if (rope_sin_read_req_valid && rope_sin_read_req_ready)
                rope_sin_read_pending <= 1'b1;
            if (rope_sin_read_rsp_valid && rope_sin_read_rsp_ready)
                rope_sin_read_pending <= 1'b0;
        end
    end

    for (genvar byte_number = 0; byte_number < 16;
         byte_number++) begin : g_qkv_stage_write_mask
        assign qkv_stage_write_mask_word[byte_number*8 +: 8] =
            {8{!qkv_head_stage_write.byte_enable[byte_number]}};
    end

    assign qkv_stage_write_endpoint_select =
        qkv_head_stage_write.source ? 8'd0 :
            8'b0000_0001 << {qkv_head_stage_write.bank,
                qkv_head_stage_write.port};
    assign qkv_stage_write_endpoint_ready = current_layout.q_head_active ? {
        client_req_ready[Q_WORK3_SECONDARY_PORT],
        client_req_ready[Q_WORK3_PRIMARY_PORT],
        client_req_ready[Q_WORK2_SECONDARY_PORT],
        client_req_ready[Q_WORK2_PRIMARY_PORT],
        client_req_ready[Q_WORK1_SECONDARY_PORT],
        client_req_ready[Q_WORK1_PRIMARY_PORT],
        client_req_ready[Q_WORK0_SECONDARY_PORT],
        client_req_ready[Q_WORK0_PRIMARY_PORT]
    } : {
        client_req_ready[QKV_STAGE3_SECONDARY_PORT],
        client_req_ready[QKV_STAGE3_PRIMARY_PORT],
        client_req_ready[QKV_STAGE2_SECONDARY_PORT],
        client_req_ready[QKV_STAGE2_PRIMARY_PORT],
        client_req_ready[QKV_STAGE1_SECONDARY_PORT],
        client_req_ready[QKV_STAGE1_PRIMARY_PORT],
        client_req_ready[QKV_STAGE0_SECONDARY_PORT],
        client_req_ready[QKV_STAGE0_PRIMARY_PORT]
    };
    assign qkv_stage_write_active = !rst &&
        ((current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION &&
          current_layout.qkv_head_staging) || current_layout.q_head_active);
    assign qkv_head_stage_write_ready = qkv_stage_write_active &&
        (qkv_head_stage_write.source ?
            client_req_ready[QKV_HEAD_SOURCE_PORT] :
            |(qkv_stage_write_endpoint_select & qkv_stage_write_endpoint_ready));

    always_comb begin : qkv_stage_write_mapping
        qkv_stage_write_req_valid = '0;
        qkv_stage_write_address = '0;
        qkv_stage_write_port_data = '0;
        qkv_stage_write_mask_n = '1;
        qkv_stage_write_req_valid[QKV_HEAD_SOURCE_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            qkv_head_stage_write.source;
        qkv_stage_write_address[QKV_HEAD_SOURCE_PORT*12 +: 12] =
            local_region_row(q_post_source_region, qkv_head_stage_write.word);
        qkv_stage_write_port_data[QKV_HEAD_SOURCE_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_mask_n[QKV_HEAD_SOURCE_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_req_valid[QKV_STAGE0_PRIMARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[0];
        qkv_stage_write_req_valid[QKV_STAGE0_SECONDARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[1];
        qkv_stage_write_req_valid[QKV_STAGE1_PRIMARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[2];
        qkv_stage_write_req_valid[QKV_STAGE1_SECONDARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[3];
        qkv_stage_write_req_valid[QKV_STAGE2_PRIMARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[4];
        qkv_stage_write_req_valid[QKV_STAGE2_SECONDARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[5];
        qkv_stage_write_req_valid[QKV_STAGE3_PRIMARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[6];
        qkv_stage_write_req_valid[QKV_STAGE3_SECONDARY_PORT] =
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            !current_layout.q_head_active &&
            qkv_stage_write_endpoint_select[7];
        qkv_stage_write_req_valid[Q_WORK0_PRIMARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[0];
        qkv_stage_write_req_valid[Q_WORK0_SECONDARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[1];
        qkv_stage_write_req_valid[Q_WORK1_PRIMARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[2];
        qkv_stage_write_req_valid[Q_WORK1_SECONDARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[3];
        qkv_stage_write_req_valid[Q_WORK2_PRIMARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[4];
        qkv_stage_write_req_valid[Q_WORK2_SECONDARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[5];
        qkv_stage_write_req_valid[Q_WORK3_PRIMARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[6];
        qkv_stage_write_req_valid[Q_WORK3_SECONDARY_PORT] |=
            qkv_head_stage_write_valid && qkv_stage_write_active &&
            current_layout.q_head_active && qkv_stage_write_endpoint_select[7];
        qkv_stage_write_address[QKV_STAGE0_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[0], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE0_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[0], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE1_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[1], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE1_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[1], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE2_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[2], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE2_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[2], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE3_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[3], qkv_head_stage_write.word);
        qkv_stage_write_address[QKV_STAGE3_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[3], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK0_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[0], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK0_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[0], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK1_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[1], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK1_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[1], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK2_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[2], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK2_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[2], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK3_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[3], qkv_head_stage_write.word);
        qkv_stage_write_address[Q_WORK3_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[3], qkv_head_stage_write.word);
        qkv_stage_write_port_data[QKV_STAGE0_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE0_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE1_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE1_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE2_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE2_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE3_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[QKV_STAGE3_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK0_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK0_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK1_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK1_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK2_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK2_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK3_PRIMARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_port_data[Q_WORK3_SECONDARY_PORT*128 +: 128] =
            qkv_head_stage_write.data;
        qkv_stage_write_mask_n[QKV_STAGE0_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE0_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE1_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE1_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE2_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE2_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE3_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[QKV_STAGE3_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK0_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK0_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK1_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK1_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK2_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK2_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK3_PRIMARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
        qkv_stage_write_mask_n[Q_WORK3_SECONDARY_PORT*128 +: 128] =
            qkv_stage_write_mask_word;
    end

    assign qkv_tile_read_primary_word =
        (current_layout.q_head_active ? 10'd0 :
            10'(QKV_HEAD_STAGING_WORD_BASE)) +
        {2'b00, qkv_head_tile_read_req.row_base[5:2], 4'b0000} +
        {6'd0, qkv_head_tile_read_req.word};
    assign qkv_tile_read_secondary_word = qkv_tile_read_primary_word + 10'd16;
    assign qkv_tile_matmul_conflict = current_layout.q_head_active ?
        ((qkv_head_tile_read_req.row_mask[0] &&
            matmul_req_valid[Q_WORK0_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[1] &&
            matmul_req_valid[Q_WORK1_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[2] &&
            matmul_req_valid[Q_WORK2_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[3] &&
            matmul_req_valid[Q_WORK3_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[4] &&
            matmul_req_valid[Q_WORK0_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[5] &&
            matmul_req_valid[Q_WORK1_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[6] &&
            matmul_req_valid[Q_WORK2_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[7] &&
            matmul_req_valid[Q_WORK3_SECONDARY_PORT])) :
        ((qkv_head_tile_read_req.row_mask[0] &&
            matmul_req_valid[QKV_STAGE0_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[1] &&
            matmul_req_valid[QKV_STAGE1_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[2] &&
            matmul_req_valid[QKV_STAGE2_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[3] &&
            matmul_req_valid[QKV_STAGE3_PRIMARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[4] &&
            matmul_req_valid[QKV_STAGE0_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[5] &&
            matmul_req_valid[QKV_STAGE1_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[6] &&
            matmul_req_valid[QKV_STAGE2_SECONDARY_PORT]) ||
         (qkv_head_tile_read_req.row_mask[7] &&
            matmul_req_valid[QKV_STAGE3_SECONDARY_PORT]));
    assign qkv_head_tile_read_req_ready = !rst &&
        ((current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION &&
          (current_layout.qkv_head_staging || qkv_matmul_overlap_active)) ||
         current_layout.q_head_active) &&
        !qkv_tile_matmul_conflict &&
        (qkv_head_tile_read_req.row_mask &
          (current_layout.q_head_active ? {
              client_req_ready[Q_WORK3_SECONDARY_PORT],
              client_req_ready[Q_WORK2_SECONDARY_PORT],
              client_req_ready[Q_WORK1_SECONDARY_PORT],
              client_req_ready[Q_WORK0_SECONDARY_PORT],
              client_req_ready[Q_WORK3_PRIMARY_PORT],
              client_req_ready[Q_WORK2_PRIMARY_PORT],
              client_req_ready[Q_WORK1_PRIMARY_PORT],
              client_req_ready[Q_WORK0_PRIMARY_PORT]} : {
              client_req_ready[QKV_STAGE3_SECONDARY_PORT],
              client_req_ready[QKV_STAGE2_SECONDARY_PORT],
              client_req_ready[QKV_STAGE1_SECONDARY_PORT],
              client_req_ready[QKV_STAGE0_SECONDARY_PORT],
              client_req_ready[QKV_STAGE3_PRIMARY_PORT],
              client_req_ready[QKV_STAGE2_PRIMARY_PORT],
              client_req_ready[QKV_STAGE1_PRIMARY_PORT],
              client_req_ready[QKV_STAGE0_PRIMARY_PORT]})) ==
        qkv_head_tile_read_req.row_mask;

    always_comb begin : qkv_tile_read_mapping
        qkv_tile_read_req_valid = '0;
        qkv_tile_read_address = '0;
        qkv_tile_read_req_valid[QKV_STAGE0_PRIMARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[0];
        qkv_tile_read_req_valid[QKV_STAGE1_PRIMARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[1];
        qkv_tile_read_req_valid[QKV_STAGE2_PRIMARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[2];
        qkv_tile_read_req_valid[QKV_STAGE3_PRIMARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[3];
        qkv_tile_read_req_valid[QKV_STAGE0_SECONDARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[4];
        qkv_tile_read_req_valid[QKV_STAGE1_SECONDARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[5];
        qkv_tile_read_req_valid[QKV_STAGE2_SECONDARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[6];
        qkv_tile_read_req_valid[QKV_STAGE3_SECONDARY_PORT] =
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            !current_layout.q_head_active &&
            qkv_head_tile_read_req.row_mask[7];
        qkv_tile_read_req_valid[Q_WORK0_PRIMARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[0];
        qkv_tile_read_req_valid[Q_WORK1_PRIMARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[1];
        qkv_tile_read_req_valid[Q_WORK2_PRIMARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[2];
        qkv_tile_read_req_valid[Q_WORK3_PRIMARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[3];
        qkv_tile_read_req_valid[Q_WORK0_SECONDARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[4];
        qkv_tile_read_req_valid[Q_WORK1_SECONDARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[5];
        qkv_tile_read_req_valid[Q_WORK2_SECONDARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[6];
        qkv_tile_read_req_valid[Q_WORK3_SECONDARY_PORT] |=
            qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready &&
            current_layout.q_head_active && qkv_head_tile_read_req.row_mask[7];
        qkv_tile_read_address[QKV_STAGE0_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[0], qkv_tile_read_primary_word);
        qkv_tile_read_address[QKV_STAGE1_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[1], qkv_tile_read_primary_word);
        qkv_tile_read_address[QKV_STAGE2_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[2], qkv_tile_read_primary_word);
        qkv_tile_read_address[QKV_STAGE3_PRIMARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[3], qkv_tile_read_primary_word);
        qkv_tile_read_address[QKV_STAGE0_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[0], qkv_tile_read_secondary_word);
        qkv_tile_read_address[QKV_STAGE1_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[1], qkv_tile_read_secondary_word);
        qkv_tile_read_address[QKV_STAGE2_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[2], qkv_tile_read_secondary_word);
        qkv_tile_read_address[QKV_STAGE3_SECONDARY_PORT*12 +: 12] =
            local_region_row(kv_staging_regions[3], qkv_tile_read_secondary_word);
        qkv_tile_read_address[Q_WORK0_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[0],
                qkv_tile_read_primary_word);
        qkv_tile_read_address[Q_WORK1_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[1],
                qkv_tile_read_primary_word);
        qkv_tile_read_address[Q_WORK2_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[2],
                qkv_tile_read_primary_word);
        qkv_tile_read_address[Q_WORK3_PRIMARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[3],
                qkv_tile_read_primary_word);
        qkv_tile_read_address[Q_WORK0_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[0],
                qkv_tile_read_secondary_word);
        qkv_tile_read_address[Q_WORK1_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[1],
                qkv_tile_read_secondary_word);
        qkv_tile_read_address[Q_WORK2_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[2],
                qkv_tile_read_secondary_word);
        qkv_tile_read_address[Q_WORK3_SECONDARY_PORT*12 +: 12] =
            local_region_row(q_staging_regions[3],
                qkv_tile_read_secondary_word);
    end

    assign qkv_head_tile_read_rsp_valid = qkv_tile_read_pending[1] &&
        (!qkv_tile_read_pending_mask[1][0] || qkv_tile_read_response[0][128]) &&
        (!qkv_tile_read_pending_mask[1][1] || qkv_tile_read_response[1][128]) &&
        (!qkv_tile_read_pending_mask[1][2] || qkv_tile_read_response[2][128]) &&
        (!qkv_tile_read_pending_mask[1][3] || qkv_tile_read_response[3][128]) &&
        (!qkv_tile_read_pending_mask[1][4] || qkv_tile_read_response[4][128]) &&
        (!qkv_tile_read_pending_mask[1][5] || qkv_tile_read_response[5][128]) &&
        (!qkv_tile_read_pending_mask[1][6] || qkv_tile_read_response[6][128]) &&
        (!qkv_tile_read_pending_mask[1][7] || qkv_tile_read_response[7][128]);
    assign qkv_head_tile_read_rsp.data = {
        qkv_tile_read_pending_mask[1][7] ?
            qkv_tile_read_response[7][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][6] ?
            qkv_tile_read_response[6][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][5] ?
            qkv_tile_read_response[5][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][4] ?
            qkv_tile_read_response[4][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][3] ?
            qkv_tile_read_response[3][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][2] ?
            qkv_tile_read_response[2][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][1] ?
            qkv_tile_read_response[1][127:0] : 128'd0,
        qkv_tile_read_pending_mask[1][0] ?
            qkv_tile_read_response[0][127:0] : 128'd0
    };
    assign qkv_tile_read_response[0] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK0_PRIMARY_PORT],
         port_read_data[Q_WORK0_PRIMARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE0_PRIMARY_PORT],
         port_read_data[QKV_STAGE0_PRIMARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[1] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK1_PRIMARY_PORT],
         port_read_data[Q_WORK1_PRIMARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE1_PRIMARY_PORT],
         port_read_data[QKV_STAGE1_PRIMARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[2] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK2_PRIMARY_PORT],
         port_read_data[Q_WORK2_PRIMARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE2_PRIMARY_PORT],
         port_read_data[QKV_STAGE2_PRIMARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[3] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK3_PRIMARY_PORT],
         port_read_data[Q_WORK3_PRIMARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE3_PRIMARY_PORT],
         port_read_data[QKV_STAGE3_PRIMARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[4] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK0_SECONDARY_PORT],
         port_read_data[Q_WORK0_SECONDARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE0_SECONDARY_PORT],
         port_read_data[QKV_STAGE0_SECONDARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[5] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK1_SECONDARY_PORT],
         port_read_data[Q_WORK1_SECONDARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE1_SECONDARY_PORT],
         port_read_data[QKV_STAGE1_SECONDARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[6] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK2_SECONDARY_PORT],
         port_read_data[Q_WORK2_SECONDARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE2_SECONDARY_PORT],
         port_read_data[QKV_STAGE2_SECONDARY_PORT*128 +: 128]};
    assign qkv_tile_read_response[7] = qkv_tile_read_pending_q_work[1] ?
        {port_read_valid[Q_WORK3_SECONDARY_PORT],
         port_read_data[Q_WORK3_SECONDARY_PORT*128 +: 128]} :
        {port_read_valid[QKV_STAGE3_SECONDARY_PORT],
         port_read_data[QKV_STAGE3_SECONDARY_PORT*128 +: 128]};

    for (genvar qkv_tile_row = 0; qkv_tile_row < 8;
         qkv_tile_row++) begin : g_qkv_tile_lane_mask
        assign qkv_head_tile_read_rsp.lane_mask[qkv_tile_row*8 +: 8] =
            {8{qkv_tile_read_pending_mask[1][qkv_tile_row]}};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            qkv_tile_read_pending <= '0;
            qkv_tile_read_pending_mask[0] <= '0;
            qkv_tile_read_pending_mask[1] <= '0;
            qkv_tile_read_pending_q_work[0] <= 1'b0;
            qkv_tile_read_pending_q_work[1] <= 1'b0;
        end else begin
            qkv_tile_read_pending <= {qkv_tile_read_pending[0],
                qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready};
            qkv_tile_read_pending_mask[1] <= qkv_tile_read_pending_mask[0];
            qkv_tile_read_pending_q_work[1] <=
                qkv_tile_read_pending_q_work[0];
            if (qkv_head_tile_read_req_valid && qkv_head_tile_read_req_ready) begin
                qkv_tile_read_pending_mask[0] <= qkv_head_tile_read_req.row_mask;
                qkv_tile_read_pending_q_work[0] <=
                    current_layout.q_head_active;
            end
        end
    end

    always_comb begin : attention_operand_request_mapping
        logic request_fire;
        logic [11:0] activation_address;
        logic [11:0] panel_address;
        logic [11:0] metadata_address;
        logic [11:0] panel_metadata_address;

        attention_operand_port_valid = '0;
        attention_operand_address = '0;
        request_fire = attention_operand_req_valid &&
            attention_operand_req_ready;
        activation_address = attention_operand_req.is_pv ?
            local_region_row(attention_score_row_region(
                attention_operand_req.query_base),
                attention_operand_activation_word) :
            local_region_row(q_head_slot_region(3'd0),
                attention_operand_activation_word);
        panel_address = local_region_row(attention_kv_panel_region(2'd0),
            attention_operand_panel_word +
            (attention_operand_req.active_panel ? 10'd512 : 10'd0) +
            (attention_operand_req.is_pv && attention_operand_req.active_quarter ?
                10'd256 : 10'd0));
        metadata_address = local_region_row(qkv_attention_metadata_region(),
            attention_operand_req.is_pv ?
                qkv_probability_scale_word(
                    attention_operand_req.active_head,
                    attention_operand_req.query_base) :
                q_head_slot_scale_word(
                    attention_operand_req.q_slot,
                    attention_operand_req.query_base));
        panel_metadata_address = local_region_row(
            attention_panel_metadata_region(
                attention_operand_req.active_panel),
            10'(128 + integer'(attention_operand_req.active_panel) * 32 +
                integer'(attention_operand_req.output_base[7:3])));

        if (request_fire && attention_operand_req.is_pv &&
            !attention_operand_req.query_base[5]) begin
            attention_operand_port_valid[ATTENTION_SCORE0_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE1_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE2_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE3_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE4_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE5_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE6_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE7_PORT] = 1'b1;
            attention_operand_address[ATTENTION_SCORE0_PORT*12 +: 12] =
                activation_address;
            attention_operand_address[ATTENTION_SCORE1_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd1),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE2_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd2),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE3_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd3),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE4_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd4),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE5_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd5),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE6_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd6),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE7_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd7),
                    attention_operand_activation_word);
        end else if (request_fire && attention_operand_req.is_pv) begin
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA0_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA1_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA2_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA3_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA4_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA5_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA6_PORT] = 1'b1;
            attention_operand_port_valid[ATTENTION_SCORE_EXTRA7_PORT] = 1'b1;
            attention_operand_address[ATTENTION_SCORE_EXTRA0_PORT*12 +: 12] =
                activation_address;
            attention_operand_address[ATTENTION_SCORE_EXTRA1_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd1),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA2_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd2),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA3_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd3),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA4_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd4),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA5_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd5),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA6_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd6),
                    attention_operand_activation_word);
            attention_operand_address[ATTENTION_SCORE_EXTRA7_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd7),
                    attention_operand_activation_word);
        end else if (request_fire) begin
            attention_operand_port_valid[Q_HEAD_SLOT0_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT1_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT2_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT3_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT4_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT5_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT6_PORT] = 1'b1;
            attention_operand_port_valid[Q_HEAD_SLOT7_PORT] = 1'b1;
            attention_operand_address[Q_HEAD_SLOT0_PORT*12 +: 12] =
                activation_address;
            attention_operand_address[Q_HEAD_SLOT1_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd1),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT2_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd2),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT3_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd3),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT4_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd4),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT5_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd5),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT6_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd6),
                    attention_operand_activation_word);
            attention_operand_address[Q_HEAD_SLOT7_PORT*12 +: 12] =
                local_region_row(q_head_slot_region(3'd7),
                    attention_operand_activation_word);
        end

        attention_operand_port_valid[ATTENTION_PANEL0_PRIMARY_PORT] = request_fire;
        attention_operand_port_valid[ATTENTION_PANEL1_PRIMARY_PORT] = request_fire;
        attention_operand_port_valid[ATTENTION_PANEL2_PRIMARY_PORT] = request_fire;
        attention_operand_port_valid[ATTENTION_PANEL3_PRIMARY_PORT] = request_fire;
        attention_operand_address[ATTENTION_PANEL0_PRIMARY_PORT*12 +: 12] =
            panel_address;
        attention_operand_address[ATTENTION_PANEL1_PRIMARY_PORT*12 +: 12] =
            local_region_row(attention_kv_panel_region(2'd1),
                attention_operand_panel_word +
                (attention_operand_req.active_panel ? 10'd512 : 10'd0) +
                (attention_operand_req.is_pv &&
                 attention_operand_req.active_quarter ? 10'd256 : 10'd0));
        attention_operand_address[ATTENTION_PANEL2_PRIMARY_PORT*12 +: 12] =
            local_region_row(attention_kv_panel_region(2'd2),
                attention_operand_panel_word +
                (attention_operand_req.active_panel ? 10'd512 : 10'd0) +
                (attention_operand_req.is_pv &&
                 attention_operand_req.active_quarter ? 10'd256 : 10'd0));
        attention_operand_address[ATTENTION_PANEL3_PRIMARY_PORT*12 +: 12] =
            local_region_row(attention_kv_panel_region(2'd3),
                attention_operand_panel_word +
                (attention_operand_req.active_panel ? 10'd512 : 10'd0) +
                (attention_operand_req.is_pv &&
                 attention_operand_req.active_quarter ? 10'd256 : 10'd0));
        attention_operand_port_valid[ATTENTION_METADATA_PORT] = request_fire;
        attention_operand_address[ATTENTION_METADATA_PORT*12 +: 12] =
            metadata_address;
        if (!attention_operand_req.is_pv) begin
            if (!attention_operand_req.active_panel) begin
                attention_operand_port_valid[ATTENTION_PANEL_METADATA0_PORT] =
                    request_fire;
                attention_operand_address[
                    ATTENTION_PANEL_METADATA0_PORT*12 +: 12] =
                    panel_metadata_address;
            end else begin
                attention_operand_port_valid[ATTENTION_PANEL_METADATA1_PORT] =
                    request_fire;
                attention_operand_address[
                    ATTENTION_PANEL_METADATA1_PORT*12 +: 12] =
                    panel_metadata_address;
            end
        end
        if (current_layout.attention_pair_enable) begin
            attention_operand_port_valid = '0;
            attention_operand_address = '0;
            for (integer lane = 0; lane < 8; lane++) begin
                if (current_layout.attention_batch_index == 2'd2) begin
                    attention_operand_port_valid[28+lane*2] = request_fire;
                    attention_operand_address[(28+lane*2)*12 +: 12] = local_region_row(
                        attention_operand_req.is_pv ?
                            attention_pair_row_region(current_layout.attention_batch_index,
                                attention_operand_req.query_base, current_layout.attention_score_words) :
                            attention_pair_query_region(current_layout.attention_batch_index,
                                attention_operand_req.query_base),
                        10'(attention_operand_req.k_base >> 4));
                end else begin
                    attention_operand_port_valid[lane*2] = request_fire;
                    attention_operand_address[lane*2*12 +: 12] = local_region_row(
                        attention_operand_req.is_pv ?
                            attention_pair_row_region(current_layout.attention_batch_index,
                                attention_operand_req.query_base, current_layout.attention_score_words) :
                            attention_pair_query_region(current_layout.attention_batch_index,
                                attention_operand_req.query_base),
                        10'(attention_operand_req.k_base >> 4));
                end
            end
            for (integer lane = 0; lane < 4; lane++) begin
                attention_operand_port_valid[16+lane*2] = request_fire;
                attention_operand_address[(16+lane*2)*12 +: 12] =
                    local_region_row(attention_pair_panel_region(2'(lane)),
                        attention_operand_panel_word);
            end
            attention_operand_port_valid[24] = request_fire;
            attention_operand_address[24*12 +: 12] = local_region_row(
                attention_pair_scale_region(attention_operand_req.is_pv,
                    current_layout.attention_batch_index),
                {7'd0, attention_operand_req.query_base[5:3]});
            attention_operand_port_valid[26] = request_fire && !attention_operand_req.is_pv;
            attention_operand_address[26*12 +: 12] = local_region_row(
                ATTENTION_PAIR_K_SCALE_REGION,
                {5'd0, attention_operand_req.output_base[7:3]});
        end
    end

    always_comb begin : attention_operand_response_mapping
        if (attention_operand_pending_is_pv[1] &&
            attention_operand_pending_extra[1])
            attention_operand_rsp.activation_rows = {
                port_read_data[ATTENTION_SCORE_EXTRA7_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA6_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA5_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA4_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA3_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA2_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA1_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE_EXTRA0_PORT*128 +: 128]
            };
        else if (attention_operand_pending_is_pv[1])
            attention_operand_rsp.activation_rows =
            {
                port_read_data[ATTENTION_SCORE7_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE6_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE5_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE4_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE3_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE2_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE1_PORT*128 +: 128],
                port_read_data[ATTENTION_SCORE0_PORT*128 +: 128]
            };
        else
            attention_operand_rsp.activation_rows = {
                port_read_data[Q_HEAD_SLOT7_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT6_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT5_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT4_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT3_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT2_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT1_PORT*128 +: 128],
                port_read_data[Q_HEAD_SLOT0_PORT*128 +: 128]
            };
        attention_operand_rsp.panel_rows = {
            port_read_data[ATTENTION_PANEL3_PRIMARY_PORT*128 +: 128],
            port_read_data[ATTENTION_PANEL2_PRIMARY_PORT*128 +: 128],
            port_read_data[ATTENTION_PANEL1_PRIMARY_PORT*128 +: 128],
            port_read_data[ATTENTION_PANEL0_PRIMARY_PORT*128 +: 128]
        };
        attention_operand_rsp.metadata =
            port_read_data[ATTENTION_METADATA_PORT*128 +: 128];
        attention_operand_rsp.panel_metadata = attention_operand_pending_panel[1] ?
            port_read_data[ATTENTION_PANEL_METADATA1_PORT*128 +: 128] :
            port_read_data[ATTENTION_PANEL_METADATA0_PORT*128 +: 128];
        if (current_layout.attention_pair_enable) begin
            for (integer lane = 0; lane < 8; lane++) begin
                attention_operand_rsp.activation_rows[lane*128 +: 128] =
                    current_layout.attention_batch_index == 2'd2 ?
                        port_read_data[(28+lane*2)*128 +: 128] :
                        port_read_data[lane*2*128 +: 128];
            end
            for (integer lane = 0; lane < 4; lane++)
                attention_operand_rsp.panel_rows[lane*128 +: 128] =
                    port_read_data[(16+lane*2)*128 +: 128];
            attention_operand_rsp.metadata = port_read_data[24*128 +: 128];
            attention_operand_rsp.panel_metadata = port_read_data[26*128 +: 128];
        end
    end

    assign attention_operand_activation_word = attention_operand_req.is_pv ?
        attention_score_row_word(attention_operand_req.query_base,
            10'((attention_operand_req.sequence_length + 12'd15) >> 4),
            8'(attention_operand_req.k_base >> 4)) :
        q_head_slot_word(attention_operand_req.q_slot,
            attention_operand_req.query_base,
            attention_operand_req.k_base[7:4]);
    assign attention_operand_panel_word = attention_operand_req.is_pv ?
        10'(integer'(attention_operand_req.k_base) >> 3) :
        10'(integer'(attention_operand_req.output_base[7:3]) * 16 +
            (integer'(attention_operand_req.k_base) >> 3));
    assign attention_q_slot_ports_ready =
        client_req_ready[Q_HEAD_SLOT0_PORT] &&
        client_req_ready[Q_HEAD_SLOT1_PORT] &&
        client_req_ready[Q_HEAD_SLOT2_PORT] &&
        client_req_ready[Q_HEAD_SLOT3_PORT] &&
        client_req_ready[Q_HEAD_SLOT4_PORT] &&
        client_req_ready[Q_HEAD_SLOT5_PORT] &&
        client_req_ready[Q_HEAD_SLOT6_PORT] &&
        client_req_ready[Q_HEAD_SLOT7_PORT];
    assign attention_panel_ports_ready =
        client_req_ready[ATTENTION_PANEL0_PRIMARY_PORT] &&
        client_req_ready[ATTENTION_PANEL1_PRIMARY_PORT] &&
        client_req_ready[ATTENTION_PANEL2_PRIMARY_PORT] &&
        client_req_ready[ATTENTION_PANEL3_PRIMARY_PORT];
    assign attention_panel_ports_valid =
        port_read_valid[ATTENTION_PANEL0_PRIMARY_PORT] &&
        port_read_valid[ATTENTION_PANEL1_PRIMARY_PORT] &&
        port_read_valid[ATTENTION_PANEL2_PRIMARY_PORT] &&
        port_read_valid[ATTENTION_PANEL3_PRIMARY_PORT];
    assign attention_panel_metadata_req_ready =
        attention_operand_req.active_panel ?
            client_req_ready[ATTENTION_PANEL_METADATA1_PORT] :
            client_req_ready[ATTENTION_PANEL_METADATA0_PORT];
    assign attention_panel_metadata_rsp_valid =
        attention_operand_pending_panel[1] ?
            port_read_valid[ATTENTION_PANEL_METADATA1_PORT] :
            port_read_valid[ATTENTION_PANEL_METADATA0_PORT];

    for (genvar lane = 0; lane < 8; lane++) begin : g_pair_operand_activation_ports
        assign pair_attention_activation_ready[lane] =
            current_layout.attention_batch_index == 2'd2 ?
                client_req_ready[28+lane*2] : client_req_ready[lane*2];
        assign pair_attention_activation_valid[lane] =
            current_layout.attention_batch_index == 2'd2 ?
                port_read_valid[28+lane*2] : port_read_valid[lane*2];
    end
    for (genvar lane = 0; lane < 4; lane++) begin : g_pair_operand_panel_ports
        assign pair_attention_panel_ready[lane] = client_req_ready[16+lane*2];
        assign pair_attention_panel_valid[lane] = port_read_valid[16+lane*2];
    end
    assign attention_operand_req_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid && !attention_operand_abort_request &&
         current_layout.phase == (attention_operand_req.is_pv ?
             LOCAL_MEMORY_PHASE_ATTENTION_PV : LOCAL_MEMORY_PHASE_ATTENTION_QK) &&
         (&pair_attention_activation_ready) && (&pair_attention_panel_ready) &&
         client_req_ready[24] && (attention_operand_req.is_pv || client_req_ready[26])) :
        single_attention_operand_ready;
    assign attention_operand_rsp_valid = current_layout.attention_pair_enable ?
        (attention_operand_read_pending[1] && (&pair_attention_activation_valid) &&
         (&pair_attention_panel_valid) && port_read_valid[24] &&
         (attention_operand_pending_is_pv[1] || port_read_valid[26])) :
        single_attention_operand_valid;

    assign single_attention_operand_ready = !rst && !layout_update_valid &&
        !attention_operand_abort_request &&
        current_layout.phase == (attention_operand_req.is_pv ?
            LOCAL_MEMORY_PHASE_ATTENTION_PV :
            LOCAL_MEMORY_PHASE_ATTENTION_QK) &&
        attention_panel_ports_ready &&
        client_req_ready[ATTENTION_METADATA_PORT] &&
        (attention_operand_req.is_pv && attention_operand_req.query_base[5] ?
            (client_req_ready[ATTENTION_SCORE_EXTRA0_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA1_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA2_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA3_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA4_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA5_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA6_PORT] &&
             client_req_ready[ATTENTION_SCORE_EXTRA7_PORT]) :
         attention_operand_req.is_pv ?
            (client_req_ready[ATTENTION_SCORE0_PORT] &&
             client_req_ready[ATTENTION_SCORE1_PORT] &&
             client_req_ready[ATTENTION_SCORE2_PORT] &&
             client_req_ready[ATTENTION_SCORE3_PORT] &&
             client_req_ready[ATTENTION_SCORE4_PORT] &&
             client_req_ready[ATTENTION_SCORE5_PORT] &&
             client_req_ready[ATTENTION_SCORE6_PORT] &&
             client_req_ready[ATTENTION_SCORE7_PORT]) :
            (attention_q_slot_ports_ready &&
             attention_panel_metadata_req_ready));
    assign single_attention_operand_valid = attention_operand_read_pending[1] &&
        attention_panel_ports_valid &&
        port_read_valid[ATTENTION_METADATA_PORT] &&
        (attention_operand_pending_is_pv[1] &&
         attention_operand_pending_extra[1] ?
            (port_read_valid[ATTENTION_SCORE_EXTRA0_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA1_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA2_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA3_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA4_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA5_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA6_PORT] &&
             port_read_valid[ATTENTION_SCORE_EXTRA7_PORT]) :
         attention_operand_pending_is_pv[1] ?
            (port_read_valid[ATTENTION_SCORE0_PORT] &&
             port_read_valid[ATTENTION_SCORE1_PORT] &&
             port_read_valid[ATTENTION_SCORE2_PORT] &&
             port_read_valid[ATTENTION_SCORE3_PORT] &&
             port_read_valid[ATTENTION_SCORE4_PORT] &&
             port_read_valid[ATTENTION_SCORE5_PORT] &&
             port_read_valid[ATTENTION_SCORE6_PORT] &&
             port_read_valid[ATTENTION_SCORE7_PORT]) :
            (port_read_valid[Q_HEAD_SLOT0_PORT] &&
             port_read_valid[Q_HEAD_SLOT1_PORT] &&
             port_read_valid[Q_HEAD_SLOT2_PORT] &&
             port_read_valid[Q_HEAD_SLOT3_PORT] &&
             port_read_valid[Q_HEAD_SLOT4_PORT] &&
             port_read_valid[Q_HEAD_SLOT5_PORT] &&
             port_read_valid[Q_HEAD_SLOT6_PORT] &&
             port_read_valid[Q_HEAD_SLOT7_PORT] &&
             attention_panel_metadata_rsp_valid));
    assign attention_operand_abort_ack = attention_operand_abort_request &&
        !(|attention_operand_read_pending);

    always_ff @(posedge clk) begin
        if (rst) begin
            attention_operand_read_pending <= '0;
            attention_operand_pending_is_pv[0] <= 1'b0;
            attention_operand_pending_is_pv[1] <= 1'b0;
            attention_operand_pending_extra[0] <= 1'b0;
            attention_operand_pending_extra[1] <= 1'b0;
            attention_operand_pending_panel[0] <= 1'b0;
            attention_operand_pending_panel[1] <= 1'b0;
        end else begin
            attention_operand_read_pending <= {
                attention_operand_read_pending[0],
                attention_operand_req_valid && attention_operand_req_ready};
            attention_operand_pending_is_pv[1] <=
                attention_operand_pending_is_pv[0];
            attention_operand_pending_extra[1] <=
                attention_operand_pending_extra[0];
            attention_operand_pending_panel[1] <=
                attention_operand_pending_panel[0];
            if (attention_operand_req_valid && attention_operand_req_ready) begin
                attention_operand_pending_is_pv[0] <=
                    attention_operand_req.is_pv;
                attention_operand_pending_extra[0] <=
                    attention_operand_req.query_base[5];
                attention_operand_pending_panel[0] <=
                    attention_operand_req.active_panel;
            end
            if (attention_operand_abort_request)
                attention_operand_read_pending <= '0;
        end
    end

    for (genvar score_row = 0; score_row < 8; score_row++) begin : g_score_row_mask
        assign attention_score_read_row_mask[score_row] =
            {1'b0, attention_score_read_req.row_base} + 7'(score_row) <
            {1'b0, attention_score_read_req.active_row_count};
    end

    always_comb begin : attention_score_request_mapping
        logic score_write_fire;
        logic score_read_fire;
        logic [9:0] score_write_word;
        logic [9:0] score_read_word;
        logic [11:0] score_write_address;
        logic [11:0] score_read_address;

        attention_score_req_valid = '0;
        attention_score_port_write = '0;
        attention_score_address = '0;
        attention_score_write_data = '0;
        attention_score_write_mask_n = '1;
        score_write_fire = attention_score_write_valid &&
            attention_score_write_ready;
        score_read_fire = attention_score_read_req_valid &&
            attention_score_read_req_ready;
        score_write_word = attention_score_row_word(
            attention_score_write.row_base,
            10'((attention_score_write.sequence_length + 12'd7) >> 3),
            8'(attention_score_write.key_base >> 3));
        score_read_word = attention_score_row_word(
            attention_score_read_req.row_base,
            10'((attention_score_read_req.sequence_length + 12'd7) >> 3),
            8'(attention_score_read_req.key >> 3));
        score_write_address = local_region_row(
            attention_score_row_region(attention_score_write.row_base),
            score_write_word);
        score_read_address = local_region_row(
            attention_score_row_region(attention_score_read_req.row_base),
            score_read_word);

        attention_score_req_valid[ATTENTION_SCORE0_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[0*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE1_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[1*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE2_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[2*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE3_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[3*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE4_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[4*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE5_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[5*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE6_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[6*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE7_WRITE_PORT] =
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[7*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE0_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[0];
        attention_score_req_valid[ATTENTION_SCORE1_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[1];
        attention_score_req_valid[ATTENTION_SCORE2_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[2];
        attention_score_req_valid[ATTENTION_SCORE3_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[3];
        attention_score_req_valid[ATTENTION_SCORE4_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[4];
        attention_score_req_valid[ATTENTION_SCORE5_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[5];
        attention_score_req_valid[ATTENTION_SCORE6_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[6];
        attention_score_req_valid[ATTENTION_SCORE7_PORT] =
            score_read_fire && !attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[7];

        attention_score_req_valid[ATTENTION_SCORE_EXTRA0_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[0*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA1_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[1*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA2_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[2*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA3_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[3*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA4_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[4*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA5_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[5*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA6_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[6*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA7_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[7*8 +: 8];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA0_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[0];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA1_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[1];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA2_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[2];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA3_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[3];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA4_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[4];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA5_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[5];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA6_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[6];
        attention_score_req_valid[ATTENTION_SCORE_EXTRA7_PORT] |=
            score_read_fire && attention_score_read_req.row_base[5] &&
            attention_score_read_row_mask[7];

        attention_score_port_write[ATTENTION_SCORE0_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[0*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE1_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[1*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE2_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[2*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE3_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[3*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE4_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[4*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE5_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[5*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE6_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[6*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE7_WRITE_PORT] |=
            score_write_fire && !attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[7*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA0_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[0*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA1_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[1*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA2_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[2*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA3_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[3*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA4_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[4*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA5_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[5*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA6_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[6*8 +: 8];
        attention_score_port_write[ATTENTION_SCORE_EXTRA7_WRITE_PORT] |=
            score_write_fire && attention_score_write.row_base[5] &&
            |attention_score_write.lane_mask[7*8 +: 8];

        if (score_write_fire && !attention_score_write.row_base[5]) begin
            attention_score_address[ATTENTION_SCORE0_WRITE_PORT*12 +: 12] =
                score_write_address;
            attention_score_address[ATTENTION_SCORE1_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd1), score_write_word);
            attention_score_address[ATTENTION_SCORE2_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd2), score_write_word);
            attention_score_address[ATTENTION_SCORE3_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd3), score_write_word);
            attention_score_address[ATTENTION_SCORE4_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd4), score_write_word);
            attention_score_address[ATTENTION_SCORE5_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd5), score_write_word);
            attention_score_address[ATTENTION_SCORE6_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd6), score_write_word);
            attention_score_address[ATTENTION_SCORE7_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd7), score_write_word);
        end
        if (score_read_fire && !attention_score_read_req.row_base[5]) begin
            attention_score_address[ATTENTION_SCORE0_PORT*12 +: 12] =
                score_read_address;
            attention_score_address[ATTENTION_SCORE1_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd1), score_read_word);
            attention_score_address[ATTENTION_SCORE2_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd2), score_read_word);
            attention_score_address[ATTENTION_SCORE3_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd3), score_read_word);
            attention_score_address[ATTENTION_SCORE4_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd4), score_read_word);
            attention_score_address[ATTENTION_SCORE5_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd5), score_read_word);
            attention_score_address[ATTENTION_SCORE6_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd6), score_read_word);
            attention_score_address[ATTENTION_SCORE7_PORT*12 +: 12] =
                local_region_row(attention_score_region(3'd7), score_read_word);
        end
        if (score_write_fire && attention_score_write.row_base[5]) begin
            attention_score_address[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*12 +: 12] =
                score_write_address;
            attention_score_address[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd1),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd2),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd3),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd4),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd5),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd6),
                    score_write_word);
            attention_score_address[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd7),
                    score_write_word);
        end
        if (score_read_fire && attention_score_read_req.row_base[5]) begin
            attention_score_address[ATTENTION_SCORE_EXTRA0_PORT*12 +: 12] =
                score_read_address;
            attention_score_address[ATTENTION_SCORE_EXTRA1_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd1),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA2_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd2),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA3_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd3),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA4_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd4),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA5_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd5),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA6_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd6),
                    score_read_word);
            attention_score_address[ATTENTION_SCORE_EXTRA7_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd7),
                    score_read_word);
        end

        attention_score_write_data[ATTENTION_SCORE0_WRITE_PORT*128 +: 128] =
            attention_score_write.values[0*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE1_WRITE_PORT*128 +: 128] =
            attention_score_write.values[1*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE2_WRITE_PORT*128 +: 128] =
            attention_score_write.values[2*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE3_WRITE_PORT*128 +: 128] =
            attention_score_write.values[3*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE4_WRITE_PORT*128 +: 128] =
            attention_score_write.values[4*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE5_WRITE_PORT*128 +: 128] =
            attention_score_write.values[5*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE6_WRITE_PORT*128 +: 128] =
            attention_score_write.values[6*128 +: 128];
        attention_score_write_data[ATTENTION_SCORE7_WRITE_PORT*128 +: 128] =
            attention_score_write.values[7*128 +: 128];
        attention_score_write_mask_n[ATTENTION_SCORE0_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[0*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE1_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[1*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE2_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[2*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE3_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[3*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE4_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[4*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE5_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[5*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE6_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[6*8 +: 8]);
        attention_score_write_mask_n[ATTENTION_SCORE7_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(attention_score_write.lane_mask[7*8 +: 8]);
        if (attention_score_write.row_base[5]) begin
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*128 +: 128] =
                attention_score_write.values[0*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*128 +: 128] =
                attention_score_write.values[1*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*128 +: 128] =
                attention_score_write.values[2*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*128 +: 128] =
                attention_score_write.values[3*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*128 +: 128] =
                attention_score_write.values[4*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*128 +: 128] =
                attention_score_write.values[5*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*128 +: 128] =
                attention_score_write.values[6*128 +: 128];
            attention_score_write_data[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*128 +: 128] =
                attention_score_write.values[7*128 +: 128];
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[0*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[1*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[2*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[3*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[4*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[5*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[6*8 +: 8]);
            attention_score_write_mask_n[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*128 +: 128] =
                bf16_lane_write_mask_n(
                    attention_score_write.lane_mask[7*8 +: 8]);
        end
        if (current_layout.attention_pair_enable) begin
            attention_score_req_valid = '0;
            attention_score_port_write = '0;
            attention_score_address = '0;
            attention_score_write_data = '0;
            attention_score_write_mask_n = '1;
            for (integer lane = 0; lane < 8; lane++) begin
                if (current_layout.attention_batch_index == 2'd2) begin
                    attention_score_req_valid[29+lane*2] = score_write_fire &&
                        (|attention_score_write.lane_mask[lane*8 +: 8]);
                    attention_score_port_write[29+lane*2] = 1'b1;
                    attention_score_address[(29+lane*2)*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            attention_score_write.row_base, current_layout.attention_score_words),
                        10'(attention_score_write.key_base >> 3));
                    attention_score_write_data[(29+lane*2)*128 +: 128] =
                        attention_score_write.values[lane*128 +: 128];
                    attention_score_write_mask_n[(29+lane*2)*128 +: 128] =
                        bf16_lane_write_mask_n(attention_score_write.lane_mask[lane*8 +: 8]);
                    attention_score_req_valid[28+lane*2] = score_read_fire &&
                        attention_score_read_row_mask[lane];
                    attention_score_address[(28+lane*2)*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            attention_score_read_req.row_base, current_layout.attention_score_words),
                        10'(attention_score_read_req.key >> 3));
                end else begin
                    attention_score_req_valid[lane*2+1] = score_write_fire &&
                        (|attention_score_write.lane_mask[lane*8 +: 8]);
                    attention_score_port_write[lane*2+1] = 1'b1;
                    attention_score_address[(lane*2+1)*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            attention_score_write.row_base, current_layout.attention_score_words),
                        10'(attention_score_write.key_base >> 3));
                    attention_score_write_data[(lane*2+1)*128 +: 128] =
                        attention_score_write.values[lane*128 +: 128];
                    attention_score_write_mask_n[(lane*2+1)*128 +: 128] =
                        bf16_lane_write_mask_n(attention_score_write.lane_mask[lane*8 +: 8]);
                    attention_score_req_valid[lane*2] = score_read_fire &&
                        attention_score_read_row_mask[lane];
                    attention_score_address[lane*2*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            attention_score_read_req.row_base, current_layout.attention_score_words),
                        10'(attention_score_read_req.key >> 3));
                end
            end
        end
    end

    for (genvar lane = 0; lane < 8; lane++) begin : g_pair_score_ports
        assign pair_score_write_ready[lane] =
            !(|attention_score_write.lane_mask[lane*8 +: 8]) ||
            (current_layout.attention_batch_index == 2'd2 ?
                client_req_ready[29+lane*2] : client_req_ready[lane*2+1]);
        assign pair_score_read_ready[lane] =
            !attention_score_read_row_mask[lane] ||
            (current_layout.attention_batch_index == 2'd2 ?
                client_req_ready[28+lane*2] : client_req_ready[lane*2]);
        assign pair_score_response_valid[lane] =
            !attention_score_read_pending_row_mask[1][lane] ||
            (current_layout.attention_batch_index == 2'd2 ?
                port_read_valid[28+lane*2] : port_read_valid[lane*2]);
        assign pair_score_response_values[lane*128 +: 128] =
            attention_score_read_pending_row_mask[1][lane] ?
                (current_layout.attention_batch_index == 2'd2 ?
                    port_read_data[(28+lane*2)*128 +: 128] :
                    port_read_data[lane*2*128 +: 128]) : 128'd0;
    end
    assign attention_score_write_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid &&
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK &&
         (&pair_score_write_ready)) : single_score_write_ready;
    assign attention_score_read_req_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid && attention_score_read_slot_ready &&
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
         (&pair_score_read_ready)) : single_score_read_ready;
    assign attention_score_read_rsp_valid = current_layout.attention_pair_enable ?
        (attention_score_read_pending[1] && (&pair_score_response_valid)) :
        single_score_response_valid;
    assign attention_score_read_rsp.values = current_layout.attention_pair_enable ?
        pair_score_response_values : single_score_response_values;

    assign single_score_write_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK &&
        (!(|attention_score_write.lane_mask[0*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA0_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE0_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[1*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA1_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE1_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[2*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA2_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE2_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[3*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA3_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE3_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[4*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA4_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE4_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[5*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA5_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE5_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[6*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA6_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE6_WRITE_PORT])) &&
        (!(|attention_score_write.lane_mask[7*8 +: 8]) ||
            (attention_score_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA7_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE7_WRITE_PORT]));
    assign single_score_read_ready = !rst &&
        attention_score_read_slot_ready &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
        (!attention_score_read_row_mask[0] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA0_PORT] :
                client_req_ready[ATTENTION_SCORE0_PORT])) &&
        (!attention_score_read_row_mask[1] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA1_PORT] :
                client_req_ready[ATTENTION_SCORE1_PORT])) &&
        (!attention_score_read_row_mask[2] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA2_PORT] :
                client_req_ready[ATTENTION_SCORE2_PORT])) &&
        (!attention_score_read_row_mask[3] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA3_PORT] :
                client_req_ready[ATTENTION_SCORE3_PORT])) &&
        (!attention_score_read_row_mask[4] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA4_PORT] :
                client_req_ready[ATTENTION_SCORE4_PORT])) &&
        (!attention_score_read_row_mask[5] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA5_PORT] :
                client_req_ready[ATTENTION_SCORE5_PORT])) &&
        (!attention_score_read_row_mask[6] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA6_PORT] :
                client_req_ready[ATTENTION_SCORE6_PORT])) &&
        (!attention_score_read_row_mask[7] ||
            (attention_score_read_req.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA7_PORT] :
                client_req_ready[ATTENTION_SCORE7_PORT]));
    assign single_score_response_valid = attention_score_read_pending[1] &&
        (!attention_score_read_pending_row_mask[1][0] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA0_PORT] :
                port_read_valid[ATTENTION_SCORE0_PORT])) &&
        (!attention_score_read_pending_row_mask[1][1] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA1_PORT] :
                port_read_valid[ATTENTION_SCORE1_PORT])) &&
        (!attention_score_read_pending_row_mask[1][2] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA2_PORT] :
                port_read_valid[ATTENTION_SCORE2_PORT])) &&
        (!attention_score_read_pending_row_mask[1][3] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA3_PORT] :
                port_read_valid[ATTENTION_SCORE3_PORT])) &&
        (!attention_score_read_pending_row_mask[1][4] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA4_PORT] :
                port_read_valid[ATTENTION_SCORE4_PORT])) &&
        (!attention_score_read_pending_row_mask[1][5] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA5_PORT] :
                port_read_valid[ATTENTION_SCORE5_PORT])) &&
        (!attention_score_read_pending_row_mask[1][6] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA6_PORT] :
                port_read_valid[ATTENTION_SCORE6_PORT])) &&
        (!attention_score_read_pending_row_mask[1][7] ||
            (attention_score_read_pending_extra[1] ?
                port_read_valid[ATTENTION_SCORE_EXTRA7_PORT] :
                port_read_valid[ATTENTION_SCORE7_PORT]));
    assign single_score_response_values = {
        attention_score_read_pending_row_mask[1][7] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA7_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE7_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][6] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA6_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE6_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][5] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA5_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE5_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][4] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA4_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE4_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][3] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA3_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE3_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][2] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA2_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE2_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][1] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA1_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE1_PORT*128 +: 128]) : 128'd0,
        attention_score_read_pending_row_mask[1][0] ?
            (attention_score_read_pending_extra[1] ?
                port_read_data[ATTENTION_SCORE_EXTRA0_PORT*128 +: 128] :
                port_read_data[ATTENTION_SCORE0_PORT*128 +: 128]) : 128'd0
    };

    always_ff @(posedge clk) begin
        if (rst) begin
            attention_score_read_pending <= '0;
            attention_score_read_pending_row_mask[0] <= '0;
            attention_score_read_pending_row_mask[1] <= '0;
            attention_score_read_pending_extra[0] <= 1'b0;
            attention_score_read_pending_extra[1] <= 1'b0;
        end else begin
            attention_score_read_pending <= {
                attention_score_read_pending[0],
                attention_score_read_req_valid &&
                    attention_score_read_req_ready};
            attention_score_read_pending_row_mask[1] <=
                attention_score_read_pending_row_mask[0];
            attention_score_read_pending_extra[1] <=
                attention_score_read_pending_extra[0];
            if (attention_score_read_req_valid &&
                attention_score_read_req_ready) begin
                attention_score_read_pending_row_mask[0] <=
                    attention_score_read_row_mask;
                attention_score_read_pending_extra[0] <=
                    attention_score_read_req.row_base[5];
            end
        end
    end

    always_comb begin : attention_probability_write_mapping
        logic bf16_fire;
        logic quantized_fire;
        logic scale_fire;
        logic [9:0] bf16_word;
        logic [9:0] quantized_word;
        logic [9:0] scale_word;
        logic probability_extra;
        logic [127:0] row_data [0:7];
        logic [127:0] row_mask_n [0:7];

        attention_probability_req_valid = '0;
        attention_probability_address = '0;
        attention_probability_write_data = '0;
        attention_probability_write_mask_n = '1;
        bf16_fire = attention_probability_write_valid &&
            attention_probability_write_ready;
        quantized_fire = attention_probability_quantized_write_valid &&
            attention_probability_quantized_write_ready;
        scale_fire = attention_probability_scale_write_valid &&
            attention_probability_scale_write_ready;
        probability_extra = bf16_fire ?
            attention_probability_write.row_base[5] :
            attention_probability_quantized_write.row_base[5];
        bf16_word = attention_score_row_word(
            attention_probability_write.row_base,
            10'((attention_score_read_req.sequence_length + 12'd7) >> 3),
            8'(attention_probability_write.key >> 3));
        quantized_word = attention_score_row_word(
            attention_probability_quantized_write.row_base,
            10'((attention_score_read_req.sequence_length + 12'd15) >> 4),
            8'(attention_probability_quantized_write.key >> 4));
        scale_word = qkv_probability_scale_word(
            attention_probability_scale_write.head,
            attention_probability_scale_write.row_base);

        attention_probability_req_valid[ATTENTION_SCORE0_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[0*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[0*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE1_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[1*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[1*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE2_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[2*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[2*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE3_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[3*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[3*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE4_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[4*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[4*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE5_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[5*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[5*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE6_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[6*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[6*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE7_WRITE_PORT] =
            !probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[7*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[7*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA0_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[0*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[0*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA1_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[1*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[1*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA2_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[2*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[2*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA3_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[3*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[3*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA4_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[4*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[4*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA5_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[5*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[5*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA6_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[6*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[6*8 +: 8]));
        attention_probability_req_valid[ATTENTION_SCORE_EXTRA7_WRITE_PORT] |=
            probability_extra && ((bf16_fire &&
                |attention_probability_write.lane_mask[7*8 +: 8]) ||
                (quantized_fire &&
                |attention_probability_quantized_write.lane_mask[7*8 +: 8]));
        attention_probability_req_valid[ATTENTION_METADATA_WRITE_PORT] =
            scale_fire;

        attention_probability_address[ATTENTION_SCORE0_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd0),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE1_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd1),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE2_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd2),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE3_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd3),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE4_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd4),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE5_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd5),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE6_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd6),
                bf16_fire ? bf16_word : quantized_word);
        attention_probability_address[ATTENTION_SCORE7_WRITE_PORT*12 +: 12] =
            local_region_row(attention_score_region(3'd7),
                bf16_fire ? bf16_word : quantized_word);
        if (probability_extra) begin
            attention_probability_address[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd0),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd1),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd2),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd3),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd4),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd5),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd6),
                    bf16_fire ? bf16_word : quantized_word);
            attention_probability_address[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*12 +: 12] =
                local_region_row(attention_score_extra_region(3'd7),
                    bf16_fire ? bf16_word : quantized_word);
        end
        attention_probability_address[ATTENTION_METADATA_WRITE_PORT*12 +: 12] =
            local_region_row(qkv_attention_metadata_region(), scale_word);

        for (integer row = 0; row < 8; row++) begin
            row_data[row] = '0;
            row_mask_n[row] = '1;
            if (bf16_fire) begin
                row_data[row] =
                    attention_probability_write.values[row*128 +: 128];
                row_mask_n[row] =
                    bf16_lane_write_mask_n(
                        attention_probability_write.lane_mask[row*8 +: 8]);
            end else begin
                row_data[row][
                    (attention_probability_quantized_write.key[3] ? 64 : 0) +: 64] =
                    attention_probability_quantized_write.values[row*64 +: 64];
                row_mask_n[row][
                    (attention_probability_quantized_write.key[3] ? 64 : 0) +: 64] =
                    int8_lane_write_mask_n(
                        attention_probability_quantized_write.lane_mask[
                            row*8 +: 8]);
            end
        end
        attention_probability_write_data[
            ATTENTION_SCORE0_WRITE_PORT*128 +: 128] = row_data[0];
        attention_probability_write_data[
            ATTENTION_SCORE1_WRITE_PORT*128 +: 128] = row_data[1];
        attention_probability_write_data[
            ATTENTION_SCORE2_WRITE_PORT*128 +: 128] = row_data[2];
        attention_probability_write_data[
            ATTENTION_SCORE3_WRITE_PORT*128 +: 128] = row_data[3];
        attention_probability_write_data[
            ATTENTION_SCORE4_WRITE_PORT*128 +: 128] = row_data[4];
        attention_probability_write_data[
            ATTENTION_SCORE5_WRITE_PORT*128 +: 128] = row_data[5];
        attention_probability_write_data[
            ATTENTION_SCORE6_WRITE_PORT*128 +: 128] = row_data[6];
        attention_probability_write_data[
            ATTENTION_SCORE7_WRITE_PORT*128 +: 128] = row_data[7];
        attention_probability_write_mask_n[
            ATTENTION_SCORE0_WRITE_PORT*128 +: 128] = row_mask_n[0];
        attention_probability_write_mask_n[
            ATTENTION_SCORE1_WRITE_PORT*128 +: 128] = row_mask_n[1];
        attention_probability_write_mask_n[
            ATTENTION_SCORE2_WRITE_PORT*128 +: 128] = row_mask_n[2];
        attention_probability_write_mask_n[
            ATTENTION_SCORE3_WRITE_PORT*128 +: 128] = row_mask_n[3];
        attention_probability_write_mask_n[
            ATTENTION_SCORE4_WRITE_PORT*128 +: 128] = row_mask_n[4];
        attention_probability_write_mask_n[
            ATTENTION_SCORE5_WRITE_PORT*128 +: 128] = row_mask_n[5];
        attention_probability_write_mask_n[
            ATTENTION_SCORE6_WRITE_PORT*128 +: 128] = row_mask_n[6];
        attention_probability_write_mask_n[
            ATTENTION_SCORE7_WRITE_PORT*128 +: 128] = row_mask_n[7];
        if (probability_extra) begin
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*128 +: 128] = row_data[0];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*128 +: 128] = row_data[1];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*128 +: 128] = row_data[2];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*128 +: 128] = row_data[3];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*128 +: 128] = row_data[4];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*128 +: 128] = row_data[5];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*128 +: 128] = row_data[6];
            attention_probability_write_data[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*128 +: 128] = row_data[7];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA0_WRITE_PORT*128 +: 128] = row_mask_n[0];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA1_WRITE_PORT*128 +: 128] = row_mask_n[1];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA2_WRITE_PORT*128 +: 128] = row_mask_n[2];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA3_WRITE_PORT*128 +: 128] = row_mask_n[3];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA4_WRITE_PORT*128 +: 128] = row_mask_n[4];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA5_WRITE_PORT*128 +: 128] = row_mask_n[5];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA6_WRITE_PORT*128 +: 128] = row_mask_n[6];
            attention_probability_write_mask_n[
                ATTENTION_SCORE_EXTRA7_WRITE_PORT*128 +: 128] = row_mask_n[7];
        end
        attention_probability_write_data[
            ATTENTION_METADATA_WRITE_PORT*128 +: 128] =
            attention_probability_scale_write.values;
        attention_probability_write_mask_n[
            ATTENTION_METADATA_WRITE_PORT*128 +: 128] =
            bf16_lane_write_mask_n(
                attention_probability_scale_write.row_mask);
        if (current_layout.attention_pair_enable) begin
            attention_probability_req_valid = '0;
            attention_probability_address = '0;
            attention_probability_write_data = '0;
            attention_probability_write_mask_n = '1;
            for (integer lane = 0; lane < 8; lane++) begin
                if (current_layout.attention_batch_index == 2'd2) begin
                    attention_probability_req_valid[29+lane*2] =
                        (bf16_fire && |attention_probability_write.lane_mask[lane*8 +: 8]) ||
                        (quantized_fire && |attention_probability_quantized_write.lane_mask[lane*8 +: 8]);
                    attention_probability_address[(29+lane*2)*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            bf16_fire ? attention_probability_write.row_base :
                                attention_probability_quantized_write.row_base,
                            current_layout.attention_score_words),
                        bf16_fire ? 10'(attention_probability_write.key >> 3) :
                            10'(attention_probability_quantized_write.key >> 4));
                    attention_probability_write_data[(29+lane*2)*128 +: 128] = row_data[lane];
                    attention_probability_write_mask_n[(29+lane*2)*128 +: 128] = row_mask_n[lane];
                end else begin
                    attention_probability_req_valid[lane*2+1] =
                        (bf16_fire && |attention_probability_write.lane_mask[lane*8 +: 8]) ||
                        (quantized_fire && |attention_probability_quantized_write.lane_mask[lane*8 +: 8]);
                    attention_probability_address[(lane*2+1)*12 +: 12] = local_region_row(
                        attention_pair_row_region(current_layout.attention_batch_index,
                            bf16_fire ? attention_probability_write.row_base :
                                attention_probability_quantized_write.row_base,
                            current_layout.attention_score_words),
                        bf16_fire ? 10'(attention_probability_write.key >> 3) :
                            10'(attention_probability_quantized_write.key >> 4));
                    attention_probability_write_data[(lane*2+1)*128 +: 128] = row_data[lane];
                    attention_probability_write_mask_n[(lane*2+1)*128 +: 128] = row_mask_n[lane];
                end
            end
            attention_probability_req_valid[25] = scale_fire;
            attention_probability_address[25*12 +: 12] = local_region_row(
                attention_pair_scale_region(1'b1, current_layout.attention_batch_index),
                {7'd0, attention_probability_scale_write.row_base[5:3]});
            attention_probability_write_data[25*128 +: 128] =
                attention_probability_scale_write.values;
            attention_probability_write_mask_n[25*128 +: 128] =
                bf16_lane_write_mask_n(attention_probability_scale_write.row_mask);
        end
    end

    for (genvar lane = 0; lane < 8; lane++) begin : g_pair_probability_ports
        assign pair_probability_write_ready[lane] =
            !(|attention_probability_write.lane_mask[lane*8 +: 8]) ||
            (current_layout.attention_batch_index == 2'd2 ?
                client_req_ready[29+lane*2] : client_req_ready[lane*2+1]);
        assign pair_probability_quantized_ready[lane] =
            !(|attention_probability_quantized_write.lane_mask[lane*8 +: 8]) ||
            (current_layout.attention_batch_index == 2'd2 ?
                client_req_ready[29+lane*2] : client_req_ready[lane*2+1]);
    end
    assign attention_probability_write_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid &&
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
         (&pair_probability_write_ready)) : single_probability_write_ready;
    assign attention_probability_quantized_write_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid &&
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
         (&pair_probability_quantized_ready)) : single_probability_quantized_ready;
    assign attention_probability_scale_write_ready = current_layout.attention_pair_enable ?
        (!rst && !layout_update_valid &&
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
         client_req_ready[25]) : single_probability_scale_ready;

    assign single_probability_write_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
        (!(|attention_probability_write.lane_mask[0*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA0_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE0_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[1*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA1_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE1_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[2*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA2_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE2_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[3*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA3_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE3_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[4*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA4_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE4_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[5*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA5_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE5_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[6*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA6_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE6_WRITE_PORT])) &&
        (!(|attention_probability_write.lane_mask[7*8 +: 8]) ||
            (attention_probability_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA7_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE7_WRITE_PORT]));
    assign single_probability_quantized_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
        (!(|attention_probability_quantized_write.lane_mask[0*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA0_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE0_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[1*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA1_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE1_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[2*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA2_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE2_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[3*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA3_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE3_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[4*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA4_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE4_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[5*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA5_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE5_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[6*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA6_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE6_WRITE_PORT])) &&
        (!(|attention_probability_quantized_write.lane_mask[7*8 +: 8]) ||
            (attention_probability_quantized_write.row_base[5] ?
                client_req_ready[ATTENTION_SCORE_EXTRA7_WRITE_PORT] :
                client_req_ready[ATTENTION_SCORE7_WRITE_PORT]));
    assign single_probability_scale_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX &&
        client_req_ready[ATTENTION_METADATA_WRITE_PORT];

    always_comb begin : cache_write_mapping
        logic qkv_constant_fire;
        logic qkv_q_fire;
        logic panel_fire;
        logic panel_scale_fire;
        logic [9:0] panel_word;

        cache_write_req_valid = '0;
        cache_write_address = '0;
        cache_write_data = '0;
        cache_write_mask_n = '1;
        qkv_constant_fire = qkv_constant_write_valid &&
            qkv_constant_write_ready;
        qkv_q_fire = qkv_q_write_valid && qkv_q_write_ready;
        panel_fire = attention_panel_write_valid &&
            attention_panel_write_ready;
        panel_scale_fire = attention_panel_scale_write_valid &&
            attention_panel_scale_write_ready;
        panel_word = {1'b0, attention_panel_write.address[9:1]} +
            (attention_panel_write.target_panel ? 10'd512 : 10'd0) +
            (attention_panel_write.target_quarter ? 10'd256 : 10'd0);

        cache_write_req_valid[ROPE_CONSTANT_PRIMARY_PORT] =
            qkv_constant_fire;
        cache_write_address[ROPE_CONSTANT_PRIMARY_PORT*12 +: 12] =
            local_region_row(rope_table_region,
                qkv_constant_write.word);
        cache_write_data[ROPE_CONSTANT_PRIMARY_PORT*128 +: 128] =
            qkv_constant_write.data;
        cache_write_mask_n[ROPE_CONSTANT_PRIMARY_PORT*128 +: 128] =
            byte_write_mask_n(qkv_constant_write.byte_enable);

        if (qkv_q_fire) begin
            if (qkv_q_write.scale) begin
                cache_write_req_valid[q_metadata_write_port] = 1'b1;
                cache_write_address[q_metadata_write_port*12 +: 12] =
                    local_region_row(qkv_attention_metadata_region(),
                        qkv_q_write.word);
                cache_write_data[q_metadata_write_port*128 +: 128] =
                    qkv_q_write.data;
                cache_write_mask_n[q_metadata_write_port*128 +: 128] =
                    byte_write_mask_n(qkv_q_write.byte_enable);
            end else if (current_layout.q_head_active) begin
                cache_write_req_valid[q_head_slot_write_port] = 1'b1;
                cache_write_address[q_head_slot_write_port*12 +: 12] =
                    local_region_row(q_head_slot_region(
                        qkv_q_write.physical_row[2:0]),
                        q_head_slot_word(qkv_q_write.slot,
                            qkv_q_write.physical_row,
                            qkv_q_write.word[3:0]));
                cache_write_data[q_head_slot_write_port*128 +: 128] =
                    qkv_q_write.data;
                cache_write_mask_n[q_head_slot_write_port*128 +: 128] =
                    byte_write_mask_n(qkv_q_write.byte_enable);
            end else if (qkv_q_write.physical_row < 6'd32) begin
                case (qkv_q_write.physical_row[2:0])
                    3'd0: begin
                        cache_write_req_valid[Q_CODE0_PORT] = 1'b1;
                        cache_write_address[Q_CODE0_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd0),
                                q_quantized_row_word(
                                    qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE0_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE0_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd1: begin
                        cache_write_req_valid[Q_CODE1_PORT] = 1'b1;
                        cache_write_address[Q_CODE1_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd1),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE1_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE1_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd2: begin
                        cache_write_req_valid[Q_CODE2_PORT] = 1'b1;
                        cache_write_address[Q_CODE2_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd2),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE2_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE2_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd3: begin
                        cache_write_req_valid[Q_CODE3_PORT] = 1'b1;
                        cache_write_address[Q_CODE3_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd3),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE3_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE3_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd4: begin
                        cache_write_req_valid[Q_CODE4_PORT] = 1'b1;
                        cache_write_address[Q_CODE4_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd4),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE4_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE4_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd5: begin
                        cache_write_req_valid[Q_CODE5_PORT] = 1'b1;
                        cache_write_address[Q_CODE5_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd5),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE5_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE5_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd6: begin
                        cache_write_req_valid[Q_CODE6_PORT] = 1'b1;
                        cache_write_address[Q_CODE6_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd6),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE6_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE6_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    default: begin
                        cache_write_req_valid[Q_CODE7_PORT] = 1'b1;
                        cache_write_address[Q_CODE7_PORT*12 +: 12] =
                            local_region_row(q_quantized_region(3'd7),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE7_PORT*128 +: 128] = qkv_q_write.data;
                        cache_write_mask_n[Q_CODE7_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                endcase
            end else begin
                case (qkv_q_write.physical_row[2:0])
                    3'd0: begin
                        cache_write_req_valid[Q_CODE_EXTRA0_PRIMARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA0_PRIMARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd0),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA0_PRIMARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA0_PRIMARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd1: begin
                        cache_write_req_valid[Q_CODE_EXTRA1_PRIMARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA1_PRIMARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd1),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA1_PRIMARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA1_PRIMARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd2: begin
                        cache_write_req_valid[Q_CODE_EXTRA2_PRIMARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA2_PRIMARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd2),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA2_PRIMARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA2_PRIMARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd3: begin
                        cache_write_req_valid[Q_CODE_EXTRA3_PRIMARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA3_PRIMARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd3),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA3_PRIMARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA3_PRIMARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd4: begin
                        cache_write_req_valid[Q_CODE_EXTRA0_SECONDARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA0_SECONDARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd0),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA0_SECONDARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA0_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd5: begin
                        cache_write_req_valid[Q_CODE_EXTRA1_SECONDARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA1_SECONDARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd1),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA1_SECONDARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA1_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    3'd6: begin
                        cache_write_req_valid[Q_CODE_EXTRA2_SECONDARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA2_SECONDARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd2),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA2_SECONDARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA2_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                    default: begin
                        cache_write_req_valid[Q_CODE_EXTRA3_SECONDARY_PORT] = 1'b1;
                        cache_write_address[Q_CODE_EXTRA3_SECONDARY_PORT*12 +: 12] =
                            local_region_row(q_quantized_extra_region(2'd3),
                                q_quantized_row_word(qkv_q_write.physical_row,
                                    qkv_q_write.word[7:0]));
                        cache_write_data[Q_CODE_EXTRA3_SECONDARY_PORT*128 +: 128] =
                            qkv_q_write.data;
                        cache_write_mask_n[Q_CODE_EXTRA3_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(qkv_q_write.byte_enable);
                    end
                endcase
            end
        end

        if (panel_fire) begin
            case ({attention_panel_write.address[0],
                   attention_panel_write.bank})
                2'd0: begin
                    cache_write_req_valid[ATTENTION_PANEL0_SECONDARY_PORT] = 1'b1;
                    cache_write_address[ATTENTION_PANEL0_SECONDARY_PORT*12 +: 12] =
                        local_region_row(attention_kv_panel_region(2'd0), panel_word);
                    cache_write_data[ATTENTION_PANEL0_SECONDARY_PORT*128 +: 128] =
                        attention_panel_write.data[127:0];
                    cache_write_mask_n[ATTENTION_PANEL0_SECONDARY_PORT*128 +: 128] =
                        byte_write_mask_n(
                            attention_panel_write.byte_enable[15:0]);
                end
                2'd1: begin
                    cache_write_req_valid[ATTENTION_PANEL1_SECONDARY_PORT] = 1'b1;
                    cache_write_address[ATTENTION_PANEL1_SECONDARY_PORT*12 +: 12] =
                        local_region_row(attention_kv_panel_region(2'd1), panel_word);
                    cache_write_data[ATTENTION_PANEL1_SECONDARY_PORT*128 +: 128] =
                        attention_panel_write.data[127:0];
                    cache_write_mask_n[ATTENTION_PANEL1_SECONDARY_PORT*128 +: 128] =
                        byte_write_mask_n(
                            attention_panel_write.byte_enable[15:0]);
                end
                2'd2: begin
                    cache_write_req_valid[ATTENTION_PANEL2_SECONDARY_PORT] = 1'b1;
                    cache_write_address[ATTENTION_PANEL2_SECONDARY_PORT*12 +: 12] =
                        local_region_row(attention_kv_panel_region(2'd2), panel_word);
                    cache_write_data[ATTENTION_PANEL2_SECONDARY_PORT*128 +: 128] =
                        attention_panel_write.data[127:0];
                    cache_write_mask_n[ATTENTION_PANEL2_SECONDARY_PORT*128 +: 128] =
                        byte_write_mask_n(
                            attention_panel_write.byte_enable[15:0]);
                end
                default: begin
                    cache_write_req_valid[ATTENTION_PANEL3_SECONDARY_PORT] = 1'b1;
                    cache_write_address[ATTENTION_PANEL3_SECONDARY_PORT*12 +: 12] =
                        local_region_row(attention_kv_panel_region(2'd3), panel_word);
                    cache_write_data[ATTENTION_PANEL3_SECONDARY_PORT*128 +: 128] =
                        attention_panel_write.data[127:0];
                    cache_write_mask_n[ATTENTION_PANEL3_SECONDARY_PORT*128 +: 128] =
                        byte_write_mask_n(
                            attention_panel_write.byte_enable[15:0]);
                end
            endcase
            if (|attention_panel_write.byte_enable[31:16]) begin
                case ({attention_panel_write.address[0],
                       !attention_panel_write.bank})
                    2'd0: begin
                        cache_write_req_valid[ATTENTION_PANEL0_SECONDARY_PORT] = 1'b1;
                        cache_write_address[ATTENTION_PANEL0_SECONDARY_PORT*12 +: 12] =
                            local_region_row(attention_kv_panel_region(2'd0), panel_word);
                        cache_write_data[ATTENTION_PANEL0_SECONDARY_PORT*128 +: 128] =
                            attention_panel_write.data[255:128];
                        cache_write_mask_n[ATTENTION_PANEL0_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(
                                attention_panel_write.byte_enable[31:16]);
                    end
                    2'd1: begin
                        cache_write_req_valid[ATTENTION_PANEL1_SECONDARY_PORT] = 1'b1;
                        cache_write_address[ATTENTION_PANEL1_SECONDARY_PORT*12 +: 12] =
                            local_region_row(attention_kv_panel_region(2'd1), panel_word);
                        cache_write_data[ATTENTION_PANEL1_SECONDARY_PORT*128 +: 128] =
                            attention_panel_write.data[255:128];
                        cache_write_mask_n[ATTENTION_PANEL1_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(
                                attention_panel_write.byte_enable[31:16]);
                    end
                    2'd2: begin
                        cache_write_req_valid[ATTENTION_PANEL2_SECONDARY_PORT] = 1'b1;
                        cache_write_address[ATTENTION_PANEL2_SECONDARY_PORT*12 +: 12] =
                            local_region_row(attention_kv_panel_region(2'd2), panel_word);
                        cache_write_data[ATTENTION_PANEL2_SECONDARY_PORT*128 +: 128] =
                            attention_panel_write.data[255:128];
                        cache_write_mask_n[ATTENTION_PANEL2_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(
                                attention_panel_write.byte_enable[31:16]);
                    end
                    default: begin
                        cache_write_req_valid[ATTENTION_PANEL3_SECONDARY_PORT] = 1'b1;
                        cache_write_address[ATTENTION_PANEL3_SECONDARY_PORT*12 +: 12] =
                            local_region_row(attention_kv_panel_region(2'd3), panel_word);
                        cache_write_data[ATTENTION_PANEL3_SECONDARY_PORT*128 +: 128] =
                            attention_panel_write.data[255:128];
                        cache_write_mask_n[ATTENTION_PANEL3_SECONDARY_PORT*128 +: 128] =
                            byte_write_mask_n(
                                attention_panel_write.byte_enable[31:16]);
                    end
                endcase
            end
        end

        if (!attention_panel_scale_write.target_panel) begin
            cache_write_req_valid[ATTENTION_PANEL_METADATA0_WRITE_PORT] =
                panel_scale_fire;
            cache_write_address[
                ATTENTION_PANEL_METADATA0_WRITE_PORT*12 +: 12] =
                local_region_row(attention_panel_metadata_region(1'b0),
                    10'(128 + integer'(
                        attention_panel_scale_write.token_base[7:3])));
            cache_write_data[
                ATTENTION_PANEL_METADATA0_WRITE_PORT*128 +: 128] =
                attention_panel_scale_write.data;
            cache_write_mask_n[
                ATTENTION_PANEL_METADATA0_WRITE_PORT*128 +: 128] =
                byte_write_mask_n(attention_panel_scale_write.byte_enable);
        end else begin
            cache_write_req_valid[ATTENTION_PANEL_METADATA1_WRITE_PORT] =
                panel_scale_fire;
            cache_write_address[
                ATTENTION_PANEL_METADATA1_WRITE_PORT*12 +: 12] =
                local_region_row(attention_panel_metadata_region(1'b1),
                    10'(160 + integer'(
                        attention_panel_scale_write.token_base[7:3])));
            cache_write_data[
                ATTENTION_PANEL_METADATA1_WRITE_PORT*128 +: 128] =
                attention_panel_scale_write.data;
            cache_write_mask_n[
                ATTENTION_PANEL_METADATA1_WRITE_PORT*128 +: 128] =
                byte_write_mask_n(attention_panel_scale_write.byte_enable);
        end
        if (current_layout.attention_pair_enable) begin
            cache_write_req_valid = '0;
            cache_write_address = '0;
            cache_write_data = '0;
            cache_write_mask_n = '1;
            cache_write_req_valid[ROPE_CONSTANT_PRIMARY_PORT] = qkv_constant_fire;
            cache_write_address[ROPE_CONSTANT_PRIMARY_PORT*12 +: 12] = local_region_row(
                rope_table_region, qkv_constant_write.word);
            cache_write_data[ROPE_CONSTANT_PRIMARY_PORT*128 +: 128] =
                qkv_constant_write.data;
            cache_write_mask_n[ROPE_CONSTANT_PRIMARY_PORT*128 +: 128] =
                byte_write_mask_n(qkv_constant_write.byte_enable);
            cache_write_req_valid[25] = qkv_q_fire && qkv_q_write.scale;
            cache_write_address[25*12 +: 12] = local_region_row(
                attention_pair_scale_region(1'b0, current_layout.attention_batch_index),
                {7'd0, qkv_q_write.physical_row[5:3]});
            cache_write_data[25*128 +: 128] = qkv_q_write.data;
            cache_write_mask_n[25*128 +: 128] = byte_write_mask_n(qkv_q_write.byte_enable);
            for (integer lane = 0; lane < 8; lane++) begin
                if (current_layout.attention_batch_index == 2'd2) begin
                    cache_write_req_valid[29+lane*2] = qkv_q_fire && !qkv_q_write.scale &&
                        qkv_q_write.physical_row[2:0] == 3'(lane);
                    cache_write_address[(29+lane*2)*12 +: 12] = local_region_row(
                        attention_pair_query_region(current_layout.attention_batch_index,
                            qkv_q_write.physical_row), {7'd0, qkv_q_write.word[2:0]});
                    cache_write_data[(29+lane*2)*128 +: 128] = qkv_q_write.data;
                    cache_write_mask_n[(29+lane*2)*128 +: 128] = byte_write_mask_n(qkv_q_write.byte_enable);
                end else begin
                    cache_write_req_valid[lane*2+1] = qkv_q_fire && !qkv_q_write.scale &&
                        qkv_q_write.physical_row[2:0] == 3'(lane);
                    cache_write_address[(lane*2+1)*12 +: 12] = local_region_row(
                        attention_pair_query_region(current_layout.attention_batch_index,
                            qkv_q_write.physical_row), {7'd0, qkv_q_write.word[2:0]});
                    cache_write_data[(lane*2+1)*128 +: 128] = qkv_q_write.data;
                    cache_write_mask_n[(lane*2+1)*128 +: 128] = byte_write_mask_n(qkv_q_write.byte_enable);
                end
            end
            for (integer lane = 0; lane < 4; lane++) begin
                if ({attention_panel_write.address[0], attention_panel_write.bank} == 2'(lane)) begin
                    cache_write_req_valid[17+lane*2] = panel_fire;
                    cache_write_data[(17+lane*2)*128 +: 128] = attention_panel_write.data[127:0];
                    cache_write_mask_n[(17+lane*2)*128 +: 128] =
                        byte_write_mask_n(attention_panel_write.byte_enable[15:0]);
                end else if ({attention_panel_write.address[0], !attention_panel_write.bank} == 2'(lane)) begin
                    cache_write_req_valid[17+lane*2] = panel_fire &&
                        (|attention_panel_write.byte_enable[31:16]);
                    cache_write_data[(17+lane*2)*128 +: 128] = attention_panel_write.data[255:128];
                    cache_write_mask_n[(17+lane*2)*128 +: 128] =
                        byte_write_mask_n(attention_panel_write.byte_enable[31:16]);
                end
                cache_write_address[(17+lane*2)*12 +: 12] =
                    {3'd0, attention_panel_write.address[9:1]};
            end
            cache_write_req_valid[27] = panel_scale_fire;
            cache_write_address[27*12 +: 12] = local_region_row(
                ATTENTION_PAIR_K_SCALE_REGION, {5'd0, attention_panel_scale_write.token_base[7:3]});
            cache_write_data[27*128 +: 128] = attention_panel_scale_write.data;
            cache_write_mask_n[27*128 +: 128] = byte_write_mask_n(attention_panel_scale_write.byte_enable);
        end
    end

    assign matmul_expanded_layout =
        current_layout.panel_layout == LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL;
    assign matmul_local_layout =
        current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED ||
        matmul_expanded_layout ||
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT;
    // Gate prefetch during residual spill/RMS uses the same banks as Gate/Up.
    assign matmul_preserved_panel_layout = matmul_local_layout &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN ||
         current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM);

    function automatic logic [LOCAL_SRAM_MACRO_COUNT-1:0]
        physical_port_macro_onehot(input logic [5:0] port_index);
        physical_port_macro_onehot = '0;
        if (port_index < 6'(LOCAL_SRAM_PHYSICAL_PORT_COUNT))
            physical_port_macro_onehot[port_index[5:1]] = 1'b1;
    endfunction

    for (genvar lane = 0; lane < 8; lane++) begin : g_matmul_lane_port_decode
        assign matmul_source_macro_onehot[lane] =
            physical_port_macro_onehot(matmul_source_port[lane]);
        assign matmul_source_port_side[lane] = matmul_source_port[lane][0];
        assign matmul_activation_write_macro_onehot[lane] =
            physical_port_macro_onehot(matmul_activation_write_port[lane]);
        assign matmul_activation_write_port_side[lane] =
            matmul_activation_write_port[lane][0];
        assign matmul_activation_read_macro_onehot[lane] =
            physical_port_macro_onehot(matmul_activation_read_port[lane]);
        assign matmul_activation_read_port_side[lane] =
            matmul_activation_read_port[lane][0];
    end
    for (genvar lane = 0; lane < 12; lane++) begin : g_matmul_panel_read_decode
        assign matmul_panel_read_macro_onehot[lane] =
            physical_port_macro_onehot(matmul_panel_read_port[lane]);
        assign matmul_panel_read_port_side[lane] =
            matmul_panel_read_port[lane][0];
    end
    for (genvar lane = 0; lane < 2; lane++) begin : g_matmul_panel_write_decode
        assign matmul_panel_write_macro_onehot[lane] =
            physical_port_macro_onehot(matmul_panel_write_port[lane]);
        assign matmul_panel_write_port_side[lane] =
            matmul_panel_write_port[lane][0];
    end
    for (genvar lane = 0; lane < 4; lane++) begin : g_elementwise_port_decode
        assign elementwise_product_macro_onehot[lane] =
            current_layout.phase == LOCAL_MEMORY_PHASE_R4_TWO_TOKENS && lane >= 2 ?
                '0 : physical_port_macro_onehot(elementwise_product_port[lane]);
        assign elementwise_product_port_side[lane] =
            elementwise_product_port[lane][0];
    end

    always_comb begin : elementwise_product_mapping
        local_sram_region_t region;
        integer word_index;
        integer panel_word;
        integer panel_tile;
        integer panel_row;

        elementwise_product_response_group = '0;
        for (integer lane = 0; lane < 4; lane++) begin
            word_index = integer'(elementwise_product_req.word_index) + lane;
            region = full_matmul_activation_region(5'd0);
            panel_row = 0;
            if (current_layout.phase == LOCAL_MEMORY_PHASE_R4_TWO_TOKENS) begin
                // R4 retains its eight-word virtual stripe. Only the first
                // two token words are stored; unused arithmetic lanes read zero.
                region = local_region(5'd22 +
                    {3'd0, elementwise_product_req.word_index[3], 1'b0} +
                    5'(lane & 1), 12'd0, 1'b1);
                elementwise_product_address[lane] =
                    {1'b0, elementwise_product_req.word_index[14:4]};
                elementwise_product_response_group =
                    {2'd0, elementwise_product_req.word_index[3]};
            end else if (current_layout.phase == LOCAL_MEMORY_PHASE_R4) begin
                if (word_index < 8192) begin
                    region = r4_scratch_region(4'(word_index & 7));
                    panel_row = word_index >> 3;
                    elementwise_product_response_group = 3'd0;
                end else begin
                    region = r4_scratch_region(4'd8 + 4'(word_index & 3));
                    panel_row = (word_index - 8192) >> 2;
                    elementwise_product_response_group = 3'd1;
                end
                elementwise_product_address[lane] = local_region_row(region, 10'(panel_row));
            end else if (word_index < 24 * 1024) begin
                region = full_matmul_activation_region(
                    full_matmul_activation_stripe(15'(word_index)));
                elementwise_product_address[lane] = local_region_row(region,
                    full_matmul_activation_word_row(15'(word_index)));
                elementwise_product_response_group = 3'(word_index >> 13);
            end else begin
                panel_word = word_index - 24 * 1024;
                panel_tile = panel_word >> 3;
                if (panel_tile < 128) begin
                    region = matmul_panel_region(1'b0, 3'(panel_word & 3));
                    panel_row = 768 + (panel_tile << 1) +
                        ((panel_word >> 2) & 1);
                    elementwise_product_response_group = 3'd3;
                end else if (panel_tile < 256) begin
                    region = matmul_panel_region(1'b1, 3'(panel_word & 3));
                    panel_row = 768 + ((panel_tile - 128) << 1) +
                        ((panel_word >> 2) & 1);
                    elementwise_product_response_group = 3'd4;
                end else begin
                    case (panel_word & 3)
                        0: region = matmul_panel_region(1'b0, 3'd4);
                        1: region = matmul_panel_region(1'b0, 3'd5);
                        2: region = matmul_panel_region(1'b1, 3'd4);
                        default: region = matmul_panel_region(1'b1, 3'd5);
                    endcase
                    panel_row = ((panel_tile - 256) << 1) +
                        ((panel_word >> 2) & 1);
                    elementwise_product_response_group = 3'd5;
                end
                elementwise_product_address[lane] =
                    local_region_row(region, 10'(panel_row));
            end
            elementwise_product_port[lane] =
                local_region_port_index(region, 1'b0);
        end
    end

    always_comb begin : matmul_lane_mapping
        local_sram_region_t region;
        logic [5:0] physical_row;
        logic qkv_w4_panel_access;
        integer word_index;
        integer word_row;
        integer output_word;
        integer output_row;

        qkv_w4_panel_access = !matmul_local_layout &&
            matmul_read_req.panel_port_mask[2];

        for (integer lane = 0; lane < 8; lane++) begin
            physical_row = matmul_local_source_req.row + 6'(lane);
            if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN)
                region = resident_region(physical_row);
            else if (matmul_local_layout)
                region = full_matmul_activation_region(5'(
                    integer'(matmul_local_source_req.bank_base) +
                    integer'(hidden_row_stripe(physical_row))));
            else if (active_row_count > 6'd32)
                region = resident_region(physical_row);
            else if (physical_row[3])
                region = q_quantized_region(physical_row[2:0]);
            else
                region = qkv_activation_region(physical_row[2:0]);
            matmul_source_port[lane] = local_region_port_index(region, 1'b0);
            matmul_source_word_address[lane] = local_region_row(region,
                hidden_row_word(physical_row,
                    9'(integer'(matmul_local_source_req.element) >> 3)));

            word_index = integer'(matmul_activation_write.byte_address[
                lane*32 +: 32] >> 4);
            if (projection_pair_layout) begin
                matmul_activation_write_port[lane] = {
                    attention_pair_activation_macro(14'(word_index)), 1'b0} +
                    6'(word_index & 1);
                matmul_activation_write_word_address[lane] =
                    attention_pair_activation_row(14'(word_index));
            end else if (matmul_expanded_layout) begin
                matmul_activation_write_port[lane] = {
                    expanded_matmul_activation_macro(16'(word_index)), 1'b0} +
                    6'(word_index & 1);
                matmul_activation_write_word_address[lane] =
                    expanded_matmul_activation_row(16'(word_index));
            end else begin
                if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN) begin
                    region = preserved_matmul_activation_region(3'(word_index));
                    word_row = word_index >> 3;
                end else if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL ||
                             current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ||
                             current_layout.phase == LOCAL_MEMORY_PHASE_R4 ||
                             current_layout.phase == LOCAL_MEMORY_PHASE_ELEMENTWISE) begin
                    region = full_matmul_activation_region(
                        full_matmul_activation_stripe(15'(word_index)));
                    word_row = full_matmul_activation_word_row(15'(word_index));
                end else begin
                    region = qkv_activation_region(3'(word_index));
                    word_row = word_index >> 3;
                end
                matmul_activation_write_port[lane] =
                    local_region_port_index(region, 1'b1);
                matmul_activation_write_word_address[lane] =
                    local_region_row(region, 10'(word_row));
            end

            physical_row = matmul_read_req.explicit_activation_rows ?
                matmul_read_req.activation_physical_row[lane*6 +: 6] :
                matmul_read_req.physical_row_base + 6'(lane);
            if (matmul_read_req.explicit_activation_rows) begin
                word_index = integer'(matmul_read_req.activation_lane_word_index[
                    lane*16 +: 16]);
            end else if (matmul_read_req.qkv_padded_layout) begin
                word_index = (integer'(physical_row) >> 3) << 11;
                word_index = word_index +
                    (integer'(matmul_read_req.activation_word_index) & 32'h0000_07f8) +
                    (integer'(physical_row) & 7);
            end else begin
                word_index = integer'(matmul_read_req.activation_word_index) + lane;
            end
            if (projection_pair_layout) begin
                matmul_activation_read_port[lane] = {
                    attention_pair_activation_macro(14'(word_index)), 1'b0} +
                    6'(word_index & 1);
                matmul_activation_read_word_address[lane] =
                    attention_pair_activation_row(14'(word_index));
            end else if (matmul_expanded_layout) begin
                matmul_activation_read_port[lane] = {
                    expanded_matmul_activation_macro(16'(word_index)), 1'b0} +
                    6'(word_index & 1);
                matmul_activation_read_word_address[lane] =
                    expanded_matmul_activation_row(16'(word_index));
            end else begin
                if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN) begin
                    region = preserved_matmul_activation_region(3'(word_index));
                    word_row = word_index >> 3;
                end else if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL ||
                             current_layout.phase == LOCAL_MEMORY_PHASE_R4 ||
                             current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT) begin
                    region = full_matmul_activation_region(
                        full_matmul_activation_stripe(15'(word_index)));
                    word_row = full_matmul_activation_word_row(15'(word_index));
                end else begin
                    region = qkv_activation_region(3'(word_index));
                    word_row = word_index >> 3;
                end
                matmul_activation_read_port[lane] =
                    local_region_port_index(region, 1'b0);
                matmul_activation_read_word_address[lane] =
                    local_region_row(region, 10'(word_row));
            end
        end

        for (integer lane = 0; lane < 12; lane++) begin
            if (projection_pair_layout) begin
                region = attention_pair_panel_region(2'(lane < 6 ? lane : lane - 6));
                matmul_panel_read_port[lane] = local_region_port_index(region, lane >= 6);
                word_row = integer'(matmul_read_req.panel_address[lane*10 +: 10]);
            end else if (matmul_expanded_layout) begin
                region = expanded_matmul_panel_region(
                    matmul_read_req.panel_select,
                    2'(lane < 6 ? lane : lane - 6));
                matmul_panel_read_port[lane] = local_region_port_index(
                    region, lane >= 6);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
            end else if (current_layout.q_head_active) begin
                region = q_work_panel_region(2'(
                    lane < 6 ? lane : lane - 6));
                matmul_panel_read_port[lane] = local_region_port_index(
                    region, lane >= 6);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
                word_row = word_row +
                    (matmul_read_req.panel_select ? 512 : 0);
            end else if (qkv_w4_panel_access) begin
                region = matmul_read_req.panel_select ?
                    q_work_panel_region(2'(lane < 6 ? lane : lane - 6)) :
                    qkv_panel_region(2'(lane < 6 ? lane : lane - 6));
                matmul_panel_read_port[lane] = local_region_port_index(
                    region, lane >= 6);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
            end else if (matmul_preserved_panel_layout &&
                         matmul_read_req.panel_select) begin
                region = preserved_matmul_panel1_region(
                    2'(lane < 6 ? lane : lane - 6));
                matmul_panel_read_port[lane] =
                    local_region_port_index(region, lane >= 6);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
            end else if (lane < 6) begin
                region = matmul_local_layout ?
                    matmul_panel_region(matmul_read_req.panel_select, 3'(lane)) :
                    (matmul_read_req.panel_select ?
                        q_work_panel_region(2'(lane)) :
                        qkv_panel_region(2'(lane)));
                matmul_panel_read_port[lane] =
                    local_region_port_index(region, 1'b0);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
            end else begin
                region = matmul_local_layout ?
                    matmul_panel_region(matmul_read_req.panel_select,
                        3'(lane - 6)) :
                    (matmul_read_req.panel_select ?
                        q_work_panel_region(2'(lane - 4)) :
                        qkv_panel_region(2'(lane - 4)));
                matmul_panel_read_port[lane] = local_region_port_index(
                    region, matmul_local_layout);
                word_row = matmul_read_req.panel_address[lane*10 +: 10];
                if (!matmul_local_layout)
                    word_row = word_row - 512;
            end
            matmul_panel_read_word_address[lane] =
                local_region_row(region, 10'(word_row));
        end

        for (integer lane = 0; lane < 2; lane++) begin
            if (projection_pair_layout) begin
                region = attention_pair_panel_region(2'(matmul_panel_write.start_bank + 3'(lane)));
                word_row = integer'(matmul_panel_write.word_row);
            end else if (matmul_expanded_layout) begin
                region = expanded_matmul_panel_region(
                    matmul_panel_write.panel,
                    2'(matmul_panel_write.start_bank + 3'(lane)));
                word_row = matmul_panel_write.word_row;
            end else if (matmul_preserved_panel_layout && matmul_panel_write.panel) begin
                region = preserved_matmul_panel1_region(
                    2'(matmul_panel_write.start_bank + 3'(lane)));
                word_row = matmul_panel_write.word_row;
            end else if (matmul_local_layout) begin
                region = matmul_panel_region(matmul_panel_write.panel,
                    matmul_panel_write.start_bank + 3'(lane));
                word_row = matmul_panel_write.word_row;
            end else if (current_layout.q_head_active &&
                         !matmul_panel_write.plane[0]) begin
                region = q_work_panel_region(2'(
                    integer'(matmul_panel_write.start_bank) + lane));
                word_row = integer'(matmul_panel_write.word_row) +
                    (matmul_panel_write.panel ? 512 : 0);
            end else if (current_layout.q_head_active) begin
                region = q_work_panel_region({matmul_panel_write.plane[0],
                    matmul_panel_write.start_bank[0] ^ lane[0]});
                word_row = integer'(matmul_panel_write.word_row) -
                    (matmul_panel_write.plane[0] ? 512 : 0) +
                    (matmul_panel_write.panel ? 512 : 0);
            end else if (!matmul_panel_write.plane[0]) begin
                region = matmul_panel_write.panel ?
                    q_work_panel_region(2'(
                        integer'(matmul_panel_write.start_bank) + lane)) :
                    qkv_panel_region(2'(
                        integer'(matmul_panel_write.start_bank) + lane));
                word_row = integer'(matmul_panel_write.word_row);
            end else begin
                region = matmul_panel_write.panel ?
                    q_work_panel_region({matmul_panel_write.plane[0],
                        matmul_panel_write.start_bank[0] ^ lane[0]}) :
                    qkv_panel_region({matmul_panel_write.plane[0],
                        matmul_panel_write.start_bank[0] ^ lane[0]});
                word_row = integer'(matmul_panel_write.word_row) - 512;
            end
            matmul_panel_write_port[lane] =
                local_region_port_index(region, 1'b1);
            matmul_panel_write_word_address[lane] =
                local_region_row(region, 10'(word_row));
        end

        matmul_combine_write_port =
            matmul_combine_write.address < 10'(MATMUL_STAGING_DEPTH) ?
                6'h3f : 6'(MATMUL_GATE_SCRATCH_PORT);
        matmul_combine_read_port =
            matmul_combine_read_req.address < 10'(MATMUL_STAGING_DEPTH) ?
                6'h3f : 6'(MATMUL_GATE_SCRATCH_PORT);
        output_word = integer'(matmul_local_output_write.byte_address >> 4);
        output_row = output_word / 512;
        region = matmul_local_layout ?
            resident_region(6'(output_row)) :
            q_source_region;
        matmul_local_output_port = local_region_port_index(region, 1'b1);
        matmul_local_output_word_address = current_layout.kv_pair_enable ?
            region.row_base + {1'b0, matmul_local_output_write.byte_address[14:4]} :
            local_region_row(region, matmul_local_layout ?
                hidden_row_word(6'(output_row), 9'(output_word % 512)) :
                matmul_local_output_write.byte_address[13:4]);
        region = resident_region(matmul_residual_read_req.row);
        matmul_residual_port = local_region_port_index(region, 1'b0);
    end

    for (genvar port_number = 0;
         port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         port_number++) begin : g_matmul_port
        localparam int unsigned PORT_MACRO = port_number / 2;
        localparam logic PORT_SIDE = 1'(port_number);

        assign elementwise_product_selected_mask[port_number] =
            (elementwise_product_macro_onehot[0][PORT_MACRO] &&
             elementwise_product_port_side[0] == PORT_SIDE) ||
            (elementwise_product_macro_onehot[1][PORT_MACRO] &&
             elementwise_product_port_side[1] == PORT_SIDE) ||
            (elementwise_product_macro_onehot[2][PORT_MACRO] &&
             elementwise_product_port_side[2] == PORT_SIDE) ||
            (elementwise_product_macro_onehot[3][PORT_MACRO] &&
             elementwise_product_port_side[3] == PORT_SIDE);

        always_comb begin
            matmul_source_port_mask[port_number] = 1'b0;
            matmul_source_address[port_number*12 +: 12] = '0;
            for (integer lane = 0; lane < 8; lane++) begin
                if (matmul_local_source_req_valid &&
                    lane < matmul_local_source_req.row_count &&
                    matmul_source_macro_onehot[lane][PORT_MACRO] &&
                    matmul_source_port_side[lane] == PORT_SIDE) begin
                    matmul_source_port_mask[port_number] = 1'b1;
                    matmul_source_address[port_number*12 +: 12] =
                        matmul_source_word_address[lane];
                end
            end
        end

        always_comb begin
            matmul_operand_port_mask[port_number] = 1'b0;
            matmul_operand_address[port_number*12 +: 12] = '0;
            for (integer lane = 0; lane < 8; lane++) begin
                if (matmul_read_bundle_valid &&
                    lane < matmul_read_req.activation_word_count &&
                    matmul_activation_read_macro_onehot[lane][PORT_MACRO] &&
                    matmul_activation_read_port_side[lane] == PORT_SIDE) begin
                    matmul_operand_port_mask[port_number] = 1'b1;
                    matmul_operand_address[port_number*12 +: 12] =
                        matmul_activation_read_word_address[lane];
                end
            end
            for (integer lane = 0; lane < 12; lane++) begin
                if (matmul_read_bundle_valid &&
                    matmul_read_req.panel_port_mask[lane] &&
                    matmul_panel_read_macro_onehot[lane][PORT_MACRO] &&
                    matmul_panel_read_port_side[lane] == PORT_SIDE) begin
                    matmul_operand_port_mask[port_number] = 1'b1;
                    matmul_operand_address[port_number*12 +: 12] =
                        matmul_panel_read_word_address[lane];
                end
            end
        end

        always_comb begin
            matmul_prefetch_port_mask[port_number] = 1'b0;
            matmul_prefetch_address[port_number*12 +: 12] = '0;
            matmul_prefetch_write_data[port_number*128 +: 128] = '0;
            matmul_prefetch_write_mask_n[port_number*128 +: 128] = '1;
            for (integer lane = 0; lane < 2; lane++) begin
                if (matmul_panel_write_valid && matmul_panel_write_ready &&
                    (lane == 0 ||
                     |matmul_panel_write.byte_enable[31:16]) &&
                    matmul_panel_write.start_bank + 3'(lane) <
                        (matmul_local_layout ? 3'd6 :
                         matmul_panel_write.plane[0] ? 3'd2 : 3'd4) &&
                    matmul_panel_write_macro_onehot[lane][PORT_MACRO] &&
                    matmul_panel_write_port_side[lane] == PORT_SIDE) begin
                    matmul_prefetch_port_mask[port_number] = 1'b1;
                    matmul_prefetch_address[port_number*12 +: 12] =
                        matmul_panel_write_word_address[lane];
                    matmul_prefetch_write_data[port_number*128 +: 128] =
                        matmul_panel_write.data[lane*128 +: 128];
                    matmul_prefetch_write_mask_n[
                        port_number*128 +: 128] = byte_write_mask_n(
                            matmul_panel_write.byte_enable[lane*16 +: 16]);
                end
            end
        end

        always_comb begin
            elementwise_product_port_mask[port_number] = 1'b0;
            elementwise_product_port_address[port_number*12 +: 12] = '0;
            elementwise_product_port_data[port_number*128 +: 128] = '0;
            if (elementwise_product_req_valid &&
                elementwise_product_req_ready) begin
                for (integer lane = 0; lane < 4; lane++) begin
                    if (elementwise_product_macro_onehot[lane][PORT_MACRO] &&
                        elementwise_product_port_side[lane] == PORT_SIDE) begin
                        elementwise_product_port_mask[port_number] = 1'b1;
                        elementwise_product_port_address[
                            port_number*12 +: 12] =
                            elementwise_product_address[lane];
                        elementwise_product_port_data[
                            port_number*128 +: 128] =
                            elementwise_product_req.data[lane*128 +: 128];
                    end
                end
            end
        end

        always_comb begin : other_requests
            local_sram_region_t region;
            logic [11:0] address_value;

            matmul_other_port_mask[port_number] = 1'b0;
            matmul_other_write[port_number] = 1'b0;
            matmul_other_address[port_number*12 +: 12] = '0;
            matmul_other_write_data[port_number*128 +: 128] = '0;
            matmul_other_write_mask_n[port_number*128 +: 128] = '1;
            // Gate BF16 words use macro 22's idle upper rows. One fixed port
            // serves reads and writes; panel operands precede this request,
            // while panel prefetch waits for the complete write bundle.
            if (port_number == MATMUL_GATE_SCRATCH_PORT) begin
                if (matmul_combine_write_valid &&
                    matmul_combine_write_ready &&
                    matmul_combine_write.address >= 10'(MATMUL_STAGING_DEPTH)) begin
                    matmul_other_port_mask[port_number] = 1'b1;
                    matmul_other_write[port_number] = 1'b1;
                    matmul_other_address[port_number*12 +: 12] =
                        12'(MATMUL_GATE_SCRATCH_ROW_BASE - MATMUL_STAGING_DEPTH) +
                        {2'b00, matmul_combine_write.address};
                    matmul_other_write_data[port_number*128 +: 128] =
                        matmul_combine_write.data;
                    matmul_other_write_mask_n[port_number*128 +: 128] =
                        byte_write_mask_n(matmul_combine_write.byte_enable);
                end else if (matmul_combine_read_valid &&
                    matmul_combine_read_ready &&
                    matmul_combine_read_req.address >= 10'(MATMUL_STAGING_DEPTH)) begin
                    matmul_other_port_mask[port_number] = 1'b1;
                    matmul_other_address[port_number*12 +: 12] =
                        12'(MATMUL_GATE_SCRATCH_ROW_BASE - MATMUL_STAGING_DEPTH) +
                        {2'b00, matmul_combine_read_req.address};
                end
            end
            for (integer lane = 0; lane < 8; lane++) begin
                if (matmul_activation_write_valid && matmul_activation_write_ready &&
                    matmul_activation_write.slot_valid[lane] &&
                    matmul_activation_write_macro_onehot[lane][PORT_MACRO] &&
                    matmul_activation_write_port_side[lane] == PORT_SIDE) begin
                    matmul_other_port_mask[port_number] = 1'b1;
                    matmul_other_write[port_number] = 1'b1;
                    matmul_other_address[port_number*12 +: 12] =
                        matmul_activation_write_word_address[lane];
                    matmul_other_write_data[port_number*128 +: 128] =
                        matmul_activation_write.data[lane*128 +: 128];
                    matmul_other_write_mask_n[port_number*128 +: 128] =
                        byte_write_mask_n(matmul_activation_write.byte_enable[
                            lane*16 +: 16]);
                end
            end
            address_value = matmul_local_output_word_address;
            if (matmul_local_output_write_valid &&
                matmul_local_output_write_ready &&
                matmul_local_output_port == 6'(port_number)) begin
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b1;
                matmul_other_address[port_number*12 +: 12] = address_value;
                matmul_other_write_data[port_number*128 +: 128] =
                    matmul_local_output_write.data;
                matmul_other_write_mask_n[port_number*128 +: 128] =
                    byte_write_mask_n(matmul_local_output_write.byte_enable);
            end
            region = resident_region(matmul_residual_read_req.row);
            address_value = local_region_row(region, hidden_row_word(
                matmul_residual_read_req.row,
                matmul_residual_read_req.channel_chunk));
            if (matmul_residual_read_valid && matmul_residual_read_ready &&
                matmul_residual_port == 6'(port_number)) begin
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b0;
                matmul_other_address[port_number*12 +: 12] = address_value;
            end
            if (lm_head_scale_write_valid &&
                current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
                !post_table_access_enabled && post_table_request_exclusive &&
                !lm_head_scale_read_valid &&
                LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = lm_head_scale_buffer_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b1;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region,
                        {3'd0, lm_head_scale_write_address});
                matmul_other_write_data[port_number*128 +: 128] =
                    lm_head_scale_write_data;
                matmul_other_write_mask_n[port_number*128 +: 128] = '0;
            end else if (lm_head_scale_read_valid &&
                         current_layout.phase ==
                            LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
                         !post_table_access_enabled &&
                         post_table_request_exclusive &&
                         !lm_head_scale_write_valid &&
                         LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = lm_head_scale_buffer_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b0;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region,
                        {3'd0, lm_head_scale_read_address});
            end else if (post_state_write_valid &&
                         (post_table_access_enabled ||
                          (post_state_write_word_address >= 10'd128 &&
                           post_state_write_word_address < 10'd480)) &&
                         post_table_request_exclusive &&
                         !post_state_read_valid &&
                         LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = post_state_table_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b1;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region, post_state_write_word_address);
                matmul_other_write_data[port_number*128 +: 128] =
                    post_state_write_data;
                matmul_other_write_mask_n[port_number*128 +: 128] = '0;
            end else if (post_state_read_valid &&
                         (post_table_access_enabled ||
                          (post_state_read_word_address >= 10'd128 &&
                           post_state_read_word_address < 10'd480)) &&
                         post_table_request_exclusive &&
                         !post_state_write_valid &&
                         LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = post_state_table_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b0;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region, post_state_read_word_address);
            end else if (post_suppressed_write_valid &&
                         post_table_access_enabled &&
                         post_table_request_exclusive &&
                         !post_suppressed_read_valid &&
                         LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = post_suppressed_table_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b1;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region,
                        {4'd0, post_suppressed_write_address});
                matmul_other_write_data[port_number*128 +: 128] =
                    post_suppressed_write_data;
                matmul_other_write_mask_n[port_number*128 +: 128] = '0;
            end else if (post_suppressed_read_valid &&
                         post_table_access_enabled &&
                         post_table_request_exclusive &&
                         !post_suppressed_write_valid &&
                         post_suppressed_read_index < 8'd255 &&
                         LM_HEAD_SCALE_PORT == 6'(port_number)) begin
                region = post_suppressed_table_region();
                matmul_other_port_mask[port_number] = 1'b1;
                matmul_other_write[port_number] = 1'b0;
                matmul_other_address[port_number*12 +: 12] =
                    local_region_row(region,
                        {4'd0, post_suppressed_read_index[7:2]});
            end
        end

        assign matmul_req_valid[port_number] =
            elementwise_product_port_mask[port_number] ||
            matmul_operand_port_mask[port_number] ||
            matmul_source_port_mask[port_number] ||
            matmul_prefetch_port_mask[port_number] ||
            matmul_other_port_mask[port_number];
        assign matmul_write[port_number] =
            (elementwise_product_port_mask[port_number] &&
                elementwise_product_req.write) ||
            matmul_prefetch_port_mask[port_number] ||
            (matmul_other_port_mask[port_number] &&
                matmul_other_write[port_number]);
        assign matmul_address[port_number*12 +: 12] =
            elementwise_product_port_mask[port_number] ?
                elementwise_product_port_address[
                    port_number*12 +: 12] :
            matmul_operand_port_mask[port_number] ?
                matmul_operand_address[port_number*12 +: 12] :
            (matmul_other_port_mask[port_number] ?
                matmul_other_address[port_number*12 +: 12] :
            (matmul_source_port_mask[port_number] ?
                matmul_source_address[port_number*12 +: 12] :
            (matmul_prefetch_port_mask[port_number] ?
                matmul_prefetch_address[port_number*12 +: 12] :
                '0)));
        assign matmul_write_data[port_number*128 +: 128] =
            elementwise_product_port_mask[port_number] ?
                elementwise_product_port_data[
                    port_number*128 +: 128] :
            matmul_prefetch_port_mask[port_number] ?
                matmul_prefetch_write_data[port_number*128 +: 128] :
                matmul_other_write_data[port_number*128 +: 128];
        assign matmul_write_mask_n[port_number*128 +: 128] =
            elementwise_product_port_mask[port_number] ? '0 :
            matmul_prefetch_port_mask[port_number] ?
                matmul_prefetch_write_mask_n[port_number*128 +: 128] :
                matmul_other_write_mask_n[port_number*128 +: 128];
    end

    always_comb begin : qkv_q_ready_mapping
        qkv_q_endpoint_ready = client_req_ready[q_metadata_write_port];
        if (!qkv_q_write.scale) begin
            if (current_layout.q_head_active) begin
                qkv_q_endpoint_ready =
                    client_req_ready[q_head_slot_write_port];
            end else if (qkv_q_write.physical_row < 6'd32) begin
                case (qkv_q_write.physical_row[2:0])
                    3'd0: qkv_q_endpoint_ready = client_req_ready[Q_CODE0_PORT];
                    3'd1: qkv_q_endpoint_ready = client_req_ready[Q_CODE1_PORT];
                    3'd2: qkv_q_endpoint_ready = client_req_ready[Q_CODE2_PORT];
                    3'd3: qkv_q_endpoint_ready = client_req_ready[Q_CODE3_PORT];
                    3'd4: qkv_q_endpoint_ready = client_req_ready[Q_CODE4_PORT];
                    3'd5: qkv_q_endpoint_ready = client_req_ready[Q_CODE5_PORT];
                    3'd6: qkv_q_endpoint_ready = client_req_ready[Q_CODE6_PORT];
                    default: qkv_q_endpoint_ready = client_req_ready[Q_CODE7_PORT];
                endcase
            end else begin
                case (qkv_q_write.physical_row[2:0])
                    3'd0: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA0_PRIMARY_PORT];
                    3'd1: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA1_PRIMARY_PORT];
                    3'd2: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA2_PRIMARY_PORT];
                    3'd3: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA3_PRIMARY_PORT];
                    3'd4: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA0_SECONDARY_PORT];
                    3'd5: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA1_SECONDARY_PORT];
                    3'd6: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA2_SECONDARY_PORT];
                    default: qkv_q_endpoint_ready =
                        client_req_ready[Q_CODE_EXTRA3_SECONDARY_PORT];
                endcase
            end
        end
    end

    assign qkv_constant_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        client_req_ready[ROPE_CONSTANT_PRIMARY_PORT];
    assign qkv_q_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION ||
         current_layout.q_head_active) &&
        (current_layout.attention_pair_enable ?
            (qkv_q_write.scale ? client_req_ready[25] :
                client_req_ready[{2'd0, qkv_q_write.physical_row[2:0], 1'b1}]) :
            qkv_q_endpoint_ready);
    always_comb begin : attention_panel_ready_mapping
        case ({attention_panel_write.address[0],
               attention_panel_write.bank})
            2'd0: attention_panel_data_endpoint_ready =
                client_req_ready[ATTENTION_PANEL0_SECONDARY_PORT];
            2'd1: attention_panel_data_endpoint_ready =
                client_req_ready[ATTENTION_PANEL1_SECONDARY_PORT];
            2'd2: attention_panel_data_endpoint_ready =
                client_req_ready[ATTENTION_PANEL2_SECONDARY_PORT];
            default: attention_panel_data_endpoint_ready =
                client_req_ready[ATTENTION_PANEL3_SECONDARY_PORT];
        endcase
        if (|attention_panel_write.byte_enable[31:16]) begin
            case ({attention_panel_write.address[0],
                   !attention_panel_write.bank})
                2'd0: attention_panel_data_endpoint_ready &=
                    client_req_ready[ATTENTION_PANEL0_SECONDARY_PORT];
                2'd1: attention_panel_data_endpoint_ready &=
                    client_req_ready[ATTENTION_PANEL1_SECONDARY_PORT];
                2'd2: attention_panel_data_endpoint_ready &=
                    client_req_ready[ATTENTION_PANEL2_SECONDARY_PORT];
                default: attention_panel_data_endpoint_ready &=
                    client_req_ready[ATTENTION_PANEL3_SECONDARY_PORT];
            endcase
        end
        attention_panel_scale_endpoint_ready =
            attention_panel_scale_write.target_panel ?
                client_req_ready[ATTENTION_PANEL_METADATA1_WRITE_PORT] :
                client_req_ready[ATTENTION_PANEL_METADATA0_WRITE_PORT];
    end

    assign attention_panel_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK ||
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX ||
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV) &&
        (current_layout.attention_pair_enable ?
            (client_req_ready[6'd17 + {3'd0, attention_panel_write.address[0], attention_panel_write.bank, 1'b0}] &&
             (!(|attention_panel_write.byte_enable[31:16]) ||
              client_req_ready[6'd17 + {3'd0, attention_panel_write.address[0], !attention_panel_write.bank, 1'b0}])) :
            attention_panel_data_endpoint_ready);
    assign attention_panel_scale_write_ready = !rst &&
        (current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_QK ||
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_SOFTMAX ||
         current_layout.phase == LOCAL_MEMORY_PHASE_ATTENTION_PV) &&
        (current_layout.attention_pair_enable ? client_req_ready[27] :
            attention_panel_scale_endpoint_ready);

    assign matmul_gate_scratch_layout =
        current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL &&
        (current_layout.panel_layout == LOCAL_PANEL_LAYOUT_MIXED ||
         current_layout.panel_layout == LOCAL_PANEL_LAYOUT_EXPANDED_MATMUL);

    always_comb begin : gate_scratch_ready
        matmul_gate_scratch_port_available = !rst &&
            matmul_gate_scratch_layout &&
            client_req_ready[MATMUL_GATE_SCRATCH_PORT] &&
            !matmul_operand_port_mask[MATMUL_GATE_SCRATCH_PORT] &&
            !matmul_source_port_mask[MATMUL_GATE_SCRATCH_PORT] &&
            !(matmul_local_output_write_valid &&
              matmul_local_output_port == 6'(MATMUL_GATE_SCRATCH_PORT)) &&
            !(matmul_residual_read_valid &&
              matmul_residual_port == 6'(MATMUL_GATE_SCRATCH_PORT));
        for (integer lane = 0; lane < 8; lane++) begin
            if (matmul_activation_write_valid && matmul_activation_write.slot_valid[lane] &&
                matmul_activation_write_port[lane] == 6'(MATMUL_GATE_SCRATCH_PORT))
                matmul_gate_scratch_port_available = 1'b0;
        end
    end

    always_comb begin : matmul_ready
        integer product_response_reserved;

        product_response_reserved = integer'(elementwise_product_fifo_occupancy) +
            integer'(elementwise_product_read_pending[0]) +
            integer'(elementwise_product_read_pending[1]);
        matmul_local_source_req_ready = !rst;
        matmul_read_bundle_ready = !rst;
        matmul_activation_write_ready = !rst;
        for (integer write_lane = 0; write_lane < 8; write_lane++) begin
            if (matmul_activation_write.slot_valid[write_lane]) begin
                for (integer product_lane = 0; product_lane < 4;
                     product_lane++) begin
                    if (elementwise_product_req_valid &&
                        (matmul_activation_write_port[write_lane] ==
                             elementwise_product_port[product_lane] ||
                         (matmul_activation_write_port[write_lane][5:1] ==
                              elementwise_product_port[product_lane][5:1] &&
                          matmul_activation_write_word_address[write_lane] ==
                              elementwise_product_address[product_lane])))
                        matmul_activation_write_ready = 1'b0;
                end
            end
        end
        matmul_combine_write_ready = !rst &&
            matmul_combine_write.address < 10'(MATMUL_COMBINE_DEPTH) &&
            (matmul_combine_write.address < 10'(MATMUL_STAGING_DEPTH) ||
             matmul_gate_scratch_port_available);
        matmul_combine_read_ready = !rst &&
            matmul_combine_read_req.address < 10'(MATMUL_COMBINE_DEPTH) &&
            (matmul_combine_read_req.address < 10'(MATMUL_STAGING_DEPTH) ||
             (matmul_gate_scratch_port_available &&
              !(matmul_combine_write_valid &&
                matmul_combine_write.address >= 10'(MATMUL_STAGING_DEPTH)))) &&
            !(|matmul_combine_read_pending);
        matmul_residual_read_ready = !rst &&
            !(|matmul_residual_read_pending);
        matmul_local_output_write_ready = !rst &&
            !matmul_operand_port_mask[matmul_local_output_port] &&
            !(elementwise_product_req_valid &&
              elementwise_product_selected_mask[matmul_local_output_port]);
        elementwise_product_req_ready = !rst &&
            (current_layout.phase == LOCAL_MEMORY_PHASE_ELEMENTWISE ||
             current_layout.phase == LOCAL_MEMORY_PHASE_R4 ||
             current_layout.phase == LOCAL_MEMORY_PHASE_R4_TWO_TOKENS) &&
            elementwise_product_req.word_index <=
                (current_layout.phase == LOCAL_MEMORY_PHASE_R4_TWO_TOKENS ? 15'd12280 :
                 current_layout.phase == LOCAL_MEMORY_PHASE_R4 ? 15'd12284 : 15'd30716) &&
            elementwise_product_req.word_index[1:0] == 2'b00 &&
            (current_layout.phase != LOCAL_MEMORY_PHASE_R4_TWO_TOKENS ||
             (matmul_expanded_layout && !elementwise_product_req.word_index[2])) &&
            (elementwise_product_req.write ||
             product_response_reserved < 3);
        for (integer port_number = 0;
             port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
             port_number++) begin
            if (matmul_source_port_mask[port_number] &&
                !client_req_ready[port_number])
                matmul_local_source_req_ready = 1'b0;
            if (matmul_operand_port_mask[port_number] &&
                !client_req_ready[port_number])
                matmul_read_bundle_ready = 1'b0;
            if (elementwise_product_selected_mask[port_number] &&
                (!client_req_ready[port_number] ||
                 matmul_source_port_mask[port_number] ||
                 matmul_operand_port_mask[port_number] ||
                 matmul_other_port_mask[port_number]))
                elementwise_product_req_ready = 1'b0;
        end

        matmul_panel_write_ready = !rst &&
            current_layout.phase != LOCAL_MEMORY_PHASE_R4_TWO_TOKENS &&
            matmul_panel_write.start_bank <
                (matmul_expanded_layout ? 3'd4 : matmul_local_layout ? 3'd6 :
                 matmul_panel_write.plane[0] ? 3'd2 : 3'd4) &&
            (!(|matmul_panel_write.byte_enable[31:16]) ||
             matmul_panel_write.start_bank + 1'b1 <
                (matmul_expanded_layout ? 3'd4 : matmul_local_layout ? 3'd6 :
                 matmul_panel_write.plane[0] ? 3'd2 : 3'd4));
        for (integer lane = 0; lane < 2; lane++) begin
            if ((lane == 0 || |matmul_panel_write.byte_enable[31:16]) &&
                (!client_req_ready[matmul_panel_write_port[lane]] ||
                 qkv_stage_write_req_valid[
                     matmul_panel_write_port[lane]] ||
                 matmul_source_port_mask[matmul_panel_write_port[lane]] ||
                 matmul_operand_port_mask[matmul_panel_write_port[lane]] ||
                 (elementwise_product_req_valid &&
                  elementwise_product_selected_mask[
                      matmul_panel_write_port[lane]]) ||
                 matmul_other_port_mask[matmul_panel_write_port[lane]]))
                matmul_panel_write_ready = 1'b0;
            if ((lane == 0 || |matmul_panel_write.byte_enable[31:16]) &&
                matmul_local_output_write_valid &&
                matmul_panel_write_port[lane][5:1] ==
                    matmul_local_output_port[5:1] &&
                matmul_panel_write_word_address[lane] ==
                    matmul_local_output_word_address)
                matmul_panel_write_ready = 1'b0;
        end
    end

    assign post_scale_request_valid =
        lm_head_scale_write_valid || lm_head_scale_read_valid;
    assign post_state_request_valid =
        post_state_write_valid || post_state_read_valid;
    assign post_suppressed_request_valid =
        post_suppressed_write_valid || post_suppressed_read_valid;
    assign post_table_request_exclusive =
        !(post_scale_request_valid && post_state_request_valid) &&
        !(post_scale_request_valid && post_suppressed_request_valid) &&
        !(post_state_request_valid && post_suppressed_request_valid);

    assign post_table_access_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !post_scale_request_valid && !post_state_request_valid &&
        !post_suppressed_request_valid && !(|lm_head_scale_read_pending) &&
        !(|post_state_read_pending) && !(|post_suppressed_read_pending);

    always_ff @(posedge clk) begin
        if (rst) begin
            post_table_access_enabled <= 1'b0;
        end else if (layout_update_valid && layout_update_ready &&
                     next_layout.phase != LOCAL_MEMORY_PHASE_FINAL_OUTPUT) begin
            post_table_access_enabled <= 1'b0;
        end else if (post_table_access_valid && post_table_access_ready) begin
            post_table_access_enabled <= post_table_access_enable;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            final_output_compute_panel <= 1'b0;
        end else if (layout_update_valid && layout_update_ready &&
                     next_layout.phase != LOCAL_MEMORY_PHASE_FINAL_OUTPUT) begin
            final_output_compute_panel <= 1'b0;
        end else if (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
                     matmul_read_bundle_valid && matmul_read_bundle_ready) begin
            final_output_compute_panel <= matmul_read_req.panel_select;
        end
    end

    assign lm_head_scale_write_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !post_table_access_enabled && post_table_request_exclusive &&
        !lm_head_scale_read_valid && client_req_ready[LM_HEAD_SCALE_PORT];
    assign lm_head_scale_read_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        !post_table_access_enabled && post_table_request_exclusive &&
        !lm_head_scale_write_valid && !(|lm_head_scale_read_pending) &&
        client_req_ready[LM_HEAD_SCALE_PORT];

    assign post_state_write_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        (post_table_access_enabled ||
         (post_state_write_word_address >= 10'd128 &&
          post_state_write_word_address < 10'd480)) &&
        post_table_request_exclusive && !post_state_read_valid &&
        client_req_ready[LM_HEAD_SCALE_PORT];
    assign post_state_read_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        (post_table_access_enabled ||
         (post_state_read_word_address >= 10'd128 &&
          post_state_read_word_address < 10'd480)) &&
        post_table_request_exclusive && !post_state_write_valid &&
        client_req_ready[LM_HEAD_SCALE_PORT];
    assign post_suppressed_write_ready = !rst && post_table_access_enabled &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        post_table_request_exclusive && !post_suppressed_read_valid &&
        client_req_ready[LM_HEAD_SCALE_PORT];
    assign post_suppressed_read_ready = !rst && post_table_access_enabled &&
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
        post_table_request_exclusive && !post_suppressed_write_valid &&
        post_suppressed_read_index < 8'd255 &&
        client_req_ready[LM_HEAD_SCALE_PORT];

    assign candidate_state_competing_req_valid = rms_req_valid |
        hidden_req_valid | context_req_valid | attention_scratch_req_valid |
        qkv_head_read_port_valid | rope_req_valid | qkv_stage_write_req_valid |
        qkv_tile_read_req_valid | attention_operand_port_valid |
        attention_score_req_valid | attention_probability_req_valid |
        cache_write_req_valid | matmul_req_valid;

    always_comb begin : candidate_state_endpoint
        logic [3:0] row_group;
        logic [2:0] lane_block;
        logic row_half;
        logic [1:0] state_word;

        row_group = candidate_state_write_valid ?
            candidate_state_write_row_group : candidate_state_read_row_group;
        lane_block = candidate_state_write_valid ?
            candidate_state_write_lane_block : candidate_state_read_lane_block;
        row_half = candidate_state_write_valid ?
            candidate_state_write_row_half : candidate_state_read_row_half;
        state_word = candidate_state_write_valid ?
            candidate_state_write_word : candidate_state_read_word;
        candidate_state_selected_row_mask = candidate_state_write_valid ?
            candidate_state_write_row_mask : candidate_state_read_row_mask;
        candidate_state_selected_panel = !lane_block[0];
        candidate_state_selected_address = {2'd0, candidate_state_word_row(
            row_group, row_half, lane_block, state_word)};
        candidate_state_shape_valid = state_word < 2'd3 &&
            candidate_state_selected_address >= 12'd512 &&
            candidate_state_selected_address <= 12'd895 &&
            |candidate_state_selected_row_mask;
        // Candidate uses port B only on the LM compute panel. It uses port A
        // on the other panel while the loader writes that panel through B.
        candidate_state_compute_panel = matmul_read_bundle_valid ?
            matmul_read_req.panel_select : final_output_compute_panel;
        candidate_state_use_secondary = candidate_state_selected_panel ==
            candidate_state_compute_panel;
        candidate_state_primary_available_mask = '0;
        candidate_state_secondary_available_mask = '0;
        if (!candidate_state_selected_panel) begin
            candidate_state_primary_available_mask = {
                !candidate_state_competing_req_valid[CANDIDATE_P0_S3_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S2_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S1_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S0_PRIMARY_PORT]};
            candidate_state_secondary_available_mask = {
                !candidate_state_competing_req_valid[CANDIDATE_P0_S3_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S2_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S1_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P0_S0_SECONDARY_PORT]};
        end else begin
            candidate_state_primary_available_mask = {
                !candidate_state_competing_req_valid[CANDIDATE_P1_S3_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S2_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S1_PRIMARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S0_PRIMARY_PORT]};
            candidate_state_secondary_available_mask = {
                !candidate_state_competing_req_valid[CANDIDATE_P1_S3_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S2_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S1_SECONDARY_PORT],
                !candidate_state_competing_req_valid[CANDIDATE_P1_S0_SECONDARY_PORT]};
        end
        candidate_state_available_mask = candidate_state_use_secondary ?
            candidate_state_secondary_available_mask :
            candidate_state_primary_available_mask;
        candidate_state_read_ready = !rst &&
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
            !candidate_state_write_valid && candidate_state_shape_valid &&
            candidate_state_read_reservation_available &&
            ((candidate_state_available_mask &
              candidate_state_selected_row_mask) ==
             candidate_state_selected_row_mask);
        candidate_state_write_ready = !rst &&
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
            !candidate_state_read_valid && candidate_state_shape_valid &&
            ((candidate_state_available_mask &
              candidate_state_selected_row_mask) ==
             candidate_state_selected_row_mask);
    end

    always_comb begin : candidate_state_request_drive
        candidate_state_req_valid = '0;
        candidate_state_write = '0;
        candidate_state_address = '0;
        candidate_state_port_write_data = '0;
        candidate_state_port_write_mask_n = '1;
        case ({candidate_state_selected_panel, candidate_state_use_secondary})
            2'b00: begin
                candidate_state_req_valid[CANDIDATE_P0_S0_PRIMARY_PORT] = candidate_state_selected_row_mask[0];
                candidate_state_req_valid[CANDIDATE_P0_S1_PRIMARY_PORT] = candidate_state_selected_row_mask[1];
                candidate_state_req_valid[CANDIDATE_P0_S2_PRIMARY_PORT] = candidate_state_selected_row_mask[2];
                candidate_state_req_valid[CANDIDATE_P0_S3_PRIMARY_PORT] = candidate_state_selected_row_mask[3];
            end
            2'b01: begin
                candidate_state_req_valid[CANDIDATE_P0_S0_SECONDARY_PORT] = candidate_state_selected_row_mask[0];
                candidate_state_req_valid[CANDIDATE_P0_S1_SECONDARY_PORT] = candidate_state_selected_row_mask[1];
                candidate_state_req_valid[CANDIDATE_P0_S2_SECONDARY_PORT] = candidate_state_selected_row_mask[2];
                candidate_state_req_valid[CANDIDATE_P0_S3_SECONDARY_PORT] = candidate_state_selected_row_mask[3];
            end
            2'b10: begin
                candidate_state_req_valid[CANDIDATE_P1_S0_PRIMARY_PORT] = candidate_state_selected_row_mask[0];
                candidate_state_req_valid[CANDIDATE_P1_S1_PRIMARY_PORT] = candidate_state_selected_row_mask[1];
                candidate_state_req_valid[CANDIDATE_P1_S2_PRIMARY_PORT] = candidate_state_selected_row_mask[2];
                candidate_state_req_valid[CANDIDATE_P1_S3_PRIMARY_PORT] = candidate_state_selected_row_mask[3];
            end
            default: begin
                candidate_state_req_valid[CANDIDATE_P1_S0_SECONDARY_PORT] = candidate_state_selected_row_mask[0];
                candidate_state_req_valid[CANDIDATE_P1_S1_SECONDARY_PORT] = candidate_state_selected_row_mask[1];
                candidate_state_req_valid[CANDIDATE_P1_S2_SECONDARY_PORT] = candidate_state_selected_row_mask[2];
                candidate_state_req_valid[CANDIDATE_P1_S3_SECONDARY_PORT] = candidate_state_selected_row_mask[3];
            end
        endcase
        candidate_state_req_valid &=
            {LOCAL_SRAM_PHYSICAL_PORT_COUNT{
                candidate_state_read_fire || candidate_state_write_fire}};
        for (integer port = 0; port < LOCAL_SRAM_PHYSICAL_PORT_COUNT; port++) begin
            candidate_state_write[port] = candidate_state_write_fire;
            candidate_state_address[port*12 +: 12] =
                candidate_state_selected_address;
            candidate_state_port_write_mask_n[port*128 +: 128] = '0;
        end
        candidate_state_port_write_data[CANDIDATE_P0_S0_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[0*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S1_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[1*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S2_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[2*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S3_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[3*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S0_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[0*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S1_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[1*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S2_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[2*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P0_S3_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[3*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S0_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[0*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S1_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[1*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S2_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[2*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S3_PRIMARY_PORT*128 +: 128] = candidate_state_write_data[3*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S0_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[0*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S1_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[1*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S2_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[2*128 +: 128];
        candidate_state_port_write_data[CANDIDATE_P1_S3_SECONDARY_PORT*128 +: 128] = candidate_state_write_data[3*128 +: 128];
    end
    assign candidate_state_read_fire = candidate_state_read_valid &&
        candidate_state_read_ready;
    assign candidate_state_write_fire = candidate_state_write_valid &&
        candidate_state_write_ready;
    assign candidate_state_read_reservation_available =
        candidate_state_read_reserved_count < 3'd4 ||
        (candidate_state_read_rsp_valid && candidate_state_read_rsp_ready);

    always_ff @(posedge clk) begin : candidate_state_response_metadata
        if (rst) begin
            candidate_state_read_pending <= '0;
            candidate_state_read_pending_panel[0] <= 1'b0;
            candidate_state_read_pending_panel[1] <= 1'b0;
            candidate_state_read_pending_secondary[0] <= 1'b0;
            candidate_state_read_pending_secondary[1] <= 1'b0;
            candidate_state_read_pending_mask[0] <= '0;
            candidate_state_read_pending_mask[1] <= '0;
            candidate_state_read_pending_tag[0] <= '0;
            candidate_state_read_pending_tag[1] <= '0;
            candidate_state_read_reserved_count <= '0;
        end else begin
            candidate_state_read_pending <= {
                candidate_state_read_pending[0], candidate_state_read_fire};
            candidate_state_read_pending_panel[1] <=
                candidate_state_read_pending_panel[0];
            candidate_state_read_pending_secondary[1] <=
                candidate_state_read_pending_secondary[0];
            candidate_state_read_pending_mask[1] <=
                candidate_state_read_pending_mask[0];
            candidate_state_read_pending_tag[1] <=
                candidate_state_read_pending_tag[0];
            if (candidate_state_read_fire) begin
                candidate_state_read_pending_panel[0] <=
                    !candidate_state_read_lane_block[0];
                candidate_state_read_pending_secondary[0] <=
                    candidate_state_use_secondary;
                candidate_state_read_pending_mask[0] <=
                    candidate_state_read_row_mask;
                candidate_state_read_pending_tag[0] <=
                    candidate_state_read_tag;
            end
            case ({candidate_state_read_fire,
                   candidate_state_read_rsp_valid &&
                       candidate_state_read_rsp_ready})
                2'b10: candidate_state_read_reserved_count <=
                    candidate_state_read_reserved_count + 3'd1;
                2'b01: candidate_state_read_reserved_count <=
                    candidate_state_read_reserved_count - 3'd1;
                default: begin end
            endcase
        end
    end

    always_comb begin : candidate_state_response_capture
        candidate_state_fifo_input_valid =
            candidate_state_read_pending[1];
        candidate_state_fifo_input_data = '0;
        candidate_state_fifo_input_data[531:516] =
            candidate_state_read_pending_tag[1];
        candidate_state_fifo_input_data[515:512] =
            candidate_state_read_pending_mask[1];
        case ({candidate_state_read_pending_panel[1],
               candidate_state_read_pending_secondary[1]})
            2'b00: begin
                candidate_state_fifo_input_data[0*128 +: 128] = port_read_data[CANDIDATE_P0_S0_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[1*128 +: 128] = port_read_data[CANDIDATE_P0_S1_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[2*128 +: 128] = port_read_data[CANDIDATE_P0_S2_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[3*128 +: 128] = port_read_data[CANDIDATE_P0_S3_PRIMARY_PORT*128 +: 128];
            end
            2'b01: begin
                candidate_state_fifo_input_data[0*128 +: 128] = port_read_data[CANDIDATE_P0_S0_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[1*128 +: 128] = port_read_data[CANDIDATE_P0_S1_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[2*128 +: 128] = port_read_data[CANDIDATE_P0_S2_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[3*128 +: 128] = port_read_data[CANDIDATE_P0_S3_SECONDARY_PORT*128 +: 128];
            end
            2'b10: begin
                candidate_state_fifo_input_data[0*128 +: 128] = port_read_data[CANDIDATE_P1_S0_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[1*128 +: 128] = port_read_data[CANDIDATE_P1_S1_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[2*128 +: 128] = port_read_data[CANDIDATE_P1_S2_PRIMARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[3*128 +: 128] = port_read_data[CANDIDATE_P1_S3_PRIMARY_PORT*128 +: 128];
            end
            default: begin
                candidate_state_fifo_input_data[0*128 +: 128] = port_read_data[CANDIDATE_P1_S0_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[1*128 +: 128] = port_read_data[CANDIDATE_P1_S1_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[2*128 +: 128] = port_read_data[CANDIDATE_P1_S2_SECONDARY_PORT*128 +: 128];
                candidate_state_fifo_input_data[3*128 +: 128] = port_read_data[CANDIDATE_P1_S3_SECONDARY_PORT*128 +: 128];
            end
        endcase
    end

    ready_valid_fifo #(.DATA_WIDTH(532), .DEPTH(4))
        candidate_state_response_fifo (
            .clk(clk), .rst(rst),
            .input_valid(candidate_state_fifo_input_valid),
            .input_ready(candidate_state_fifo_input_ready),
            .input_data(candidate_state_fifo_input_data),
            .output_valid(candidate_state_read_rsp_valid),
            .output_ready(candidate_state_read_rsp_ready),
            .output_data({candidate_state_read_rsp_tag,
                candidate_state_read_rsp_row_mask,
                candidate_state_read_rsp_data}),
            .occupancy(candidate_state_fifo_occupancy));

    always_ff @(posedge clk) begin : matmul_response_metadata
        if (rst) begin
            matmul_source_read_pending <= '0;
            matmul_operand_read_pending <= '0;
            matmul_combine_read_pending <= '0;
            matmul_combine_pending_sram <= '0;
            matmul_residual_read_pending <= '0;
            lm_head_scale_read_pending <= '0;
            post_state_read_pending <= '0;
            post_suppressed_read_pending <= '0;
            post_state_read_pending_tag[0] <= '0;
            post_state_read_pending_tag[1] <= '0;
            post_suppressed_read_pending_lane[0] <= '0;
            post_suppressed_read_pending_lane[1] <= '0;
            for (integer stage = 0; stage < 2; stage++) begin
                matmul_source_pending_row[stage] <= '0;
                matmul_source_pending_row_count[stage] <= '0;
                matmul_source_pending_element[stage] <= '0;
                matmul_source_pending_bank_base[stage] <= '0;
                matmul_pending_activation_word_index[stage] <= '0;
                matmul_pending_activation_word_count[stage] <= '0;
                matmul_pending_qkv_padded_layout[stage] <= 1'b0;
                matmul_pending_physical_row_base[stage] <= '0;
                matmul_pending_explicit_activation_rows[stage] <= 1'b0;
                matmul_pending_activation_physical_row[stage] <= '0;
                matmul_pending_activation_lane_word_index[stage] <= '0;
                matmul_pending_panel[stage] <= 1'b0;
                matmul_pending_panel_mask[stage] <= '0;
                matmul_pending_q_work[stage] <= 1'b0;
                matmul_residual_pending_row[stage] <= '0;
            end
        end else begin
            matmul_source_read_pending <= {matmul_source_read_pending[0],
                matmul_local_source_req_valid &&
                matmul_local_source_req_ready};
            matmul_operand_read_pending <= {matmul_operand_read_pending[0],
                matmul_read_bundle_valid && matmul_read_bundle_ready};
            matmul_combine_read_pending <= {matmul_combine_read_pending[0],
                matmul_combine_read_valid && matmul_combine_read_ready};
            matmul_combine_pending_sram <= {matmul_combine_pending_sram[0],
                matmul_combine_read_req.address >= 10'(MATMUL_STAGING_DEPTH)};
            matmul_residual_read_pending <= {matmul_residual_read_pending[0],
                matmul_residual_read_valid && matmul_residual_read_ready};
            lm_head_scale_read_pending <= {lm_head_scale_read_pending[0],
                lm_head_scale_read_valid && lm_head_scale_read_ready};
            post_state_read_pending <= {post_state_read_pending[0],
                post_state_read_valid && post_state_read_ready};
            post_suppressed_read_pending <= {
                post_suppressed_read_pending[0],
                post_suppressed_read_valid && post_suppressed_read_ready};
            post_state_read_pending_tag[1] <=
                post_state_read_pending_tag[0];
            post_suppressed_read_pending_lane[1] <=
                post_suppressed_read_pending_lane[0];
            if (post_state_read_valid && post_state_read_ready)
                post_state_read_pending_tag[0] <= post_state_read_tag;
            if (post_suppressed_read_valid && post_suppressed_read_ready)
                post_suppressed_read_pending_lane[0] <=
                    post_suppressed_read_index[1:0];
            matmul_source_pending_row[1] <= matmul_source_pending_row[0];
            matmul_source_pending_row_count[1] <=
                matmul_source_pending_row_count[0];
            matmul_source_pending_element[1] <=
                matmul_source_pending_element[0];
            matmul_source_pending_bank_base[1] <=
                matmul_source_pending_bank_base[0];
            matmul_pending_activation_word_index[1] <=
                matmul_pending_activation_word_index[0];
            matmul_pending_activation_word_count[1] <=
                matmul_pending_activation_word_count[0];
            matmul_pending_qkv_padded_layout[1] <=
                matmul_pending_qkv_padded_layout[0];
            matmul_pending_physical_row_base[1] <=
                matmul_pending_physical_row_base[0];
            matmul_pending_explicit_activation_rows[1] <=
                matmul_pending_explicit_activation_rows[0];
            matmul_pending_activation_physical_row[1] <=
                matmul_pending_activation_physical_row[0];
            matmul_pending_activation_lane_word_index[1] <=
                matmul_pending_activation_lane_word_index[0];
            matmul_pending_panel[1] <= matmul_pending_panel[0];
            matmul_pending_panel_mask[1] <= matmul_pending_panel_mask[0];
            matmul_pending_q_work[1] <= matmul_pending_q_work[0];
            matmul_residual_pending_row[1] <= matmul_residual_pending_row[0];
            if (matmul_local_source_req_valid &&
                matmul_local_source_req_ready) begin
                matmul_source_pending_row[0] <= matmul_local_source_req.row;
                matmul_source_pending_row_count[0] <=
                    matmul_local_source_req.row_count;
                matmul_source_pending_element[0] <=
                    matmul_local_source_req.element;
                matmul_source_pending_bank_base[0] <=
                    matmul_local_source_req.bank_base;
            end
            if (matmul_read_bundle_valid && matmul_read_bundle_ready) begin
                matmul_pending_activation_word_index[0] <=
                    matmul_read_req.activation_word_index;
                matmul_pending_activation_word_count[0] <=
                    matmul_read_req.activation_word_count;
                matmul_pending_qkv_padded_layout[0] <=
                    matmul_read_req.qkv_padded_layout;
                matmul_pending_physical_row_base[0] <=
                    matmul_read_req.physical_row_base;
                matmul_pending_explicit_activation_rows[0] <=
                    matmul_read_req.explicit_activation_rows;
                matmul_pending_activation_physical_row[0] <=
                    matmul_read_req.activation_physical_row;
                matmul_pending_activation_lane_word_index[0] <=
                    matmul_read_req.activation_lane_word_index;
                matmul_pending_panel[0] <= matmul_read_req.panel_select;
                matmul_pending_panel_mask[0] <= matmul_read_req.panel_port_mask;
                matmul_pending_q_work[0] <= current_layout.q_head_active ||
                    (current_layout.panel_layout == LOCAL_PANEL_LAYOUT_QKV &&
                     matmul_read_req.panel_select);
            end
            if (matmul_residual_read_valid && matmul_residual_read_ready)
                matmul_residual_pending_row[0] <= matmul_residual_read_req.row;
        end
    end

    always_ff @(posedge clk) begin : matmul_staging_storage
        if (rst) begin
            matmul_staging_read_data[0] <= '0;
            matmul_staging_read_data[1] <= '0;
        end else begin
            matmul_staging_read_data[1] <= matmul_staging_read_data[0];
            if (matmul_combine_read_valid &&
                matmul_combine_read_ready &&
                matmul_combine_read_req.address < 10'(MATMUL_STAGING_DEPTH))
                matmul_staging_read_data[0] <= matmul_staging_memory[
                    matmul_combine_read_req.address[6:0]];
            if (matmul_combine_write_valid &&
                matmul_combine_write_ready &&
                matmul_combine_write.address < 10'(MATMUL_STAGING_DEPTH)) begin
                for (integer byte_lane = 0; byte_lane < 16; byte_lane++) begin
                    if (matmul_combine_write.byte_enable[byte_lane])
                        matmul_staging_memory[
                            matmul_combine_write.address[6:0]]
                            [byte_lane*8 +: 8] <=
                            matmul_combine_write.data[byte_lane*8 +: 8];
                end
            end
        end
    end

    always_ff @(posedge clk) begin : elementwise_product_response_metadata
        if (rst) begin
            elementwise_product_read_pending <= '0;
            for (integer stage = 0; stage < 2; stage++) begin
                elementwise_product_pending_group[stage] <= '0;
                elementwise_product_pending_half[stage] <= 1'b0;
                elementwise_product_pending_physical_half[stage] <= 1'b0;
                elementwise_product_pending_tag[stage] <= '0;
            end
        end else begin
            elementwise_product_read_pending <= {
                elementwise_product_read_pending[0],
                elementwise_product_req_valid &&
                    elementwise_product_req_ready &&
                    !elementwise_product_req.write};
            elementwise_product_pending_group[1] <=
                elementwise_product_pending_group[0];
            elementwise_product_pending_half[1] <=
                elementwise_product_pending_half[0];
            elementwise_product_pending_physical_half[1] <=
                elementwise_product_pending_physical_half[0];
            elementwise_product_pending_tag[1] <=
                elementwise_product_pending_tag[0];
            if (elementwise_product_req_valid &&
                elementwise_product_req_ready &&
                !elementwise_product_req.write) begin
                elementwise_product_pending_group[0] <=
                    elementwise_product_response_group;
                elementwise_product_pending_half[0] <=
                    elementwise_product_req.half;
                elementwise_product_pending_physical_half[0] <=
                    elementwise_product_req.word_index[2];
                elementwise_product_pending_tag[0] <=
                    elementwise_product_req.tag;
            end
        end
    end

    always_comb begin : elementwise_product_response_mapping
        logic [512:0] response_bundle;

        response_bundle = current_layout.phase == LOCAL_MEMORY_PHASE_R4_TWO_TOKENS ?
            (elementwise_product_pending_group[1][0] ?
                {port_read_valid[49] && port_read_valid[51], 256'd0,
                 port_read_data[51*128 +: 128], port_read_data[49*128 +: 128]} :
                {port_read_valid[45] && port_read_valid[47], 256'd0,
                 port_read_data[47*128 +: 128], port_read_data[45*128 +: 128]}) :
            current_layout.phase == LOCAL_MEMORY_PHASE_R4 ?
            r4_product_response_bundle(elementwise_product_pending_group[1][0],
                elementwise_product_pending_physical_half[1]) :
            elementwise_product_response_bundle(elementwise_product_pending_group[1],
                elementwise_product_pending_physical_half[1]);
        elementwise_product_fifo_input_valid =
            elementwise_product_read_pending[1] && response_bundle[512];
        elementwise_product_fifo_input_data = {
            response_bundle[511:0],
            elementwise_product_pending_half[1],
            elementwise_product_pending_tag[1]};
    end

    ready_valid_fifo #(.DATA_WIDTH(529), .DEPTH(3))
        elementwise_product_response_fifo (
            .clk(clk), .rst(rst),
            .input_valid(elementwise_product_fifo_input_valid),
            .input_ready(elementwise_product_fifo_input_ready),
            .input_data(elementwise_product_fifo_input_data),
            .output_valid(elementwise_product_rsp_valid),
            .output_ready(elementwise_product_rsp_ready),
            .output_data({elementwise_product_rsp.data,
                elementwise_product_rsp.half,
                elementwise_product_rsp.tag}),
            .occupancy(elementwise_product_fifo_occupancy)
        );

    for (genvar hidden_response_lane = 0; hidden_response_lane < 8;
         hidden_response_lane++) begin : g_aligned_hidden_response
        localparam logic [5:0] LOWER_PORT = local_region_port_index(
            hidden_region(4'(hidden_response_lane)), 1'b0);
        localparam logic [5:0] UPPER_PORT = local_region_port_index(
            hidden_region(4'(hidden_response_lane + 8)), 1'b0);
        localparam logic [5:0] EXTRA_PORT = local_region_port_index(
            resident_extra_region(3'(hidden_response_lane)), 1'b0);
        localparam logic [5:0] FINAL_OUTPUT_PORT = local_region_port_index(
            final_output_workspace_region(3'(hidden_response_lane)), 1'b0);

        assign matmul_aligned_hidden_response[hidden_response_lane] =
            matmul_source_pending_row[1][5] ?
                {port_read_valid[EXTRA_PORT],
                 port_read_data[EXTRA_PORT*128 +: 128]} :
            matmul_source_pending_row[1][3] ?
                {port_read_valid[UPPER_PORT],
                 port_read_data[UPPER_PORT*128 +: 128]} :
                {port_read_valid[LOWER_PORT],
                 port_read_data[LOWER_PORT*128 +: 128]};
        assign rms_aligned_hidden_response[hidden_response_lane] =
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
                {port_read_valid[FINAL_OUTPUT_PORT],
                 port_read_data[FINAL_OUTPUT_PORT*128 +: 128]} :
            rms_tile_read_pending_row_base[1][5] ?
                {port_read_valid[EXTRA_PORT],
                 port_read_data[EXTRA_PORT*128 +: 128]} :
            rms_tile_read_pending_row_base[1][3] ?
                {port_read_valid[UPPER_PORT],
                 port_read_data[UPPER_PORT*128 +: 128]} :
                {port_read_valid[LOWER_PORT],
                 port_read_data[LOWER_PORT*128 +: 128]};
    end

    always_comb begin : matmul_response_mapping
        logic [128:0] response;
        logic [127:0] post_table_read_data;
        logic [5:0] physical_row;
        integer word_index;

        matmul_local_source_rsp_valid = matmul_source_read_pending[1];
        matmul_local_source_rsp = '0;
        for (integer lane = 0; lane < 8; lane++) begin
            physical_row = matmul_source_pending_row[1] + 6'(lane);
            if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN) begin
                if (matmul_source_pending_row_count[1] == 4'd1) begin
                    if (lane == 0)
                        response = matmul_hidden_response(
                            physical_row);
                    else
                        response = '0;
                end else begin
                    response = matmul_aligned_hidden_response[lane];
                end
            end else if (matmul_local_layout)
                response = matmul_full_activation_response(5'(
                    integer'(matmul_source_pending_bank_base[1]) +
                    integer'(hidden_row_stripe(physical_row))));
            else if (active_row_count > 6'd32)
                response = matmul_hidden_response(physical_row);
            else if (physical_row[3])
                response = matmul_q_quantized_response(physical_row[2:0]);
            else
                response = matmul_qkv_activation_response(physical_row[2:0]);
            if (lane < matmul_source_pending_row_count[1]) begin
                matmul_local_source_rsp_valid &= response[128];
                matmul_local_source_rsp.values[lane*128 +: 128] =
                    response[127:0];
                matmul_local_source_rsp.lane_mask[lane*8 +: 8] = 8'hff;
            end
        end

        matmul_read_response_valid = matmul_operand_read_pending[1];
        matmul_read_rsp = '0;
        for (integer lane = 0; lane < 8; lane++) begin
            physical_row = matmul_pending_explicit_activation_rows[1] ?
                matmul_pending_activation_physical_row[1][lane*6 +: 6] :
                matmul_pending_physical_row_base[1] + 6'(lane);
            if (matmul_pending_explicit_activation_rows[1]) begin
                word_index = integer'(
                    matmul_pending_activation_lane_word_index[1][lane*16 +: 16]);
            end else if (matmul_pending_qkv_padded_layout[1]) begin
                word_index = (integer'(physical_row) >> 3) << 11;
                word_index = word_index +
                    (integer'(matmul_pending_activation_word_index[1]) &
                     32'h0000_07f8) +
                    (integer'(physical_row) & 7);
            end else begin
                word_index = integer'(matmul_pending_activation_word_index[1]) + lane;
            end
            if (matmul_expanded_layout || projection_pair_layout)
                response = matmul_expanded_activation_response(16'(word_index),
                    projection_pair_layout);
            else if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL_PRESERVE_HIDDEN)
                response = matmul_preserved_activation_response(
                    3'(word_index));
            else if (current_layout.phase == LOCAL_MEMORY_PHASE_MATMUL ||
                     current_layout.phase == LOCAL_MEMORY_PHASE_R4 ||
                     current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT)
                response = matmul_full_activation_response(
                    full_matmul_activation_stripe(15'(word_index)));
            else
                response = matmul_qkv_activation_response(3'(word_index));
            matmul_read_rsp.activation_data[lane*128 +: 128] =
                response[127:0];
            if (lane < matmul_pending_activation_word_count[1])
                matmul_read_response_valid &= response[128];
        end
        for (integer lane = 0; lane < 12; lane++) begin
            if (projection_pair_layout)
                response = {port_read_valid[16+2*(lane < 6 ? lane : lane-6)+(lane >= 6)],
                    port_read_data[(16+2*(lane < 6 ? lane : lane-6)+(lane >= 6))*128 +: 128]};
            else if (matmul_expanded_layout)
                response = matmul_expanded_panel_response(4'(lane));
            else if (matmul_preserved_panel_layout && matmul_pending_panel[1])
                response = matmul_preserved_panel1_response(lane >= 6,
                    2'(lane < 6 ? lane : lane - 6));
            else if (matmul_local_layout)
                response = matmul_panel_response(matmul_pending_panel[1],
                    lane >= 6, 3'(lane < 6 ? lane : lane - 6));
            else if (matmul_pending_q_work[1] &&
                     matmul_pending_panel_mask[1][2] && lane >= 6)
                response = matmul_q_work_panel_secondary_response(
                    2'(lane - 6));
            else if (matmul_pending_q_work[1])
                response = matmul_q_work_panel_response(
                    2'(lane < 6 ? lane : lane - 4));
            else if (matmul_pending_panel_mask[1][2] && lane >= 6)
                response = matmul_qkv_panel_secondary_response(
                    2'(lane - 6));
            else
                response = matmul_qkv_panel_response(
                    2'(lane < 6 ? lane : lane - 4));
            matmul_read_rsp.panel_data[lane*128 +: 128] =
                response[127:0];
            if (matmul_pending_panel_mask[1][lane])
                matmul_read_response_valid &= response[128];
        end

        response = matmul_combine_pending_sram[1] ?
            {port_read_valid[MATMUL_GATE_SCRATCH_PORT],
             port_read_data[MATMUL_GATE_SCRATCH_PORT*128 +: 128]} :
            {matmul_combine_read_pending[1], matmul_staging_read_data[1]};
        matmul_combine_read_response_valid = matmul_combine_read_pending[1] &&
            response[128];
        matmul_combine_read_rsp.data = response[127:0];

        response = matmul_hidden_response(matmul_residual_pending_row[1]);
        matmul_residual_read_rsp_valid = matmul_residual_read_pending[1] &&
            response[128];
        matmul_residual_read_rsp.data = response[127:0];

        lm_head_scale_read_rsp_valid = lm_head_scale_read_pending[1] &&
            port_read_valid[LM_HEAD_SCALE_PORT];
        lm_head_scale_read_rsp_data =
            port_read_data[LM_HEAD_SCALE_PORT*128 +: 128];
        post_table_read_data =
            port_read_data[LM_HEAD_SCALE_PORT*128 +: 128];
        post_state_read_rsp_valid = post_state_read_pending[1] &&
            port_read_valid[LM_HEAD_SCALE_PORT];
        post_state_read_rsp_data = post_table_read_data;
        post_state_read_rsp_tag = post_state_read_pending_tag[1];
        post_suppressed_read_rsp_valid = post_suppressed_read_pending[1] &&
            port_read_valid[LM_HEAD_SCALE_PORT];
        case (post_suppressed_read_pending_lane[1])
            2'd0: post_suppressed_read_rsp_token_id =
                post_table_read_data[16:0];
            2'd1: post_suppressed_read_rsp_token_id =
                post_table_read_data[48:32];
            2'd2: post_suppressed_read_rsp_token_id =
                post_table_read_data[80:64];
            default: post_suppressed_read_rsp_token_id =
                post_table_read_data[112:96];
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            current_layout.phase <= LOCAL_MEMORY_PHASE_IDLE;
            current_layout.active_panel <= 1'b0;
            current_layout.qkv_head_staging <= 1'b0;
            current_layout.q_head_active <= 1'b0;
            current_layout.q_head_slot <= 1'b0;
            current_layout.panel_layout <= LOCAL_PANEL_LAYOUT_NONE;
            current_layout.attention_pair_enable <= 1'b0;
            current_layout.attention_second_batch <= 1'b0;
            current_layout.attention_batch_index <= 2'd0;
            current_layout.attention_score_words <= '0;
            current_layout.kv_pair_enable <= 1'b0;
            current_layout.kv_second_batch <= 1'b0;
            current_layout.kv_first_batch_tokens <= '0;
        end else if (layout_update_valid && layout_update_ready) begin
            current_layout.phase <= next_layout.phase;
            current_layout.active_panel <= next_layout.active_panel;
            current_layout.qkv_head_staging <= next_layout.qkv_head_staging;
            current_layout.q_head_active <= next_layout.q_head_active;
            current_layout.q_head_slot <= next_layout.q_head_slot;
            current_layout.panel_layout <= next_layout.panel_layout;
            current_layout.attention_pair_enable <= next_layout.attention_pair_enable;
            current_layout.attention_second_batch <= next_layout.attention_second_batch;
            current_layout.attention_batch_index <= next_layout.attention_batch_index;
            current_layout.attention_score_words <= next_layout.attention_score_words;
            current_layout.kv_pair_enable <= next_layout.kv_pair_enable;
            current_layout.kv_second_batch <= next_layout.kv_second_batch;
            current_layout.kv_first_batch_tokens <= next_layout.kv_first_batch_tokens;
        end
    end

    // RMSNorm can read a tile while writing scratch and the preceding result.
    // Describe those lanes first, then connect each fixed physical endpoint.
    assign rms_gamma_region_value = rms_gamma_region(rms_ffn_command);
    assign rms_phase_active = current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM ||
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT;
    assign rms_gamma_address = local_region_row(
        rms_gamma_region_value, rms_gamma_write.word);
    assign rms_gamma_read_address = local_region_row(
        rms_gamma_region_value,
        10'(integer'(rms_tile_read_req.element) / 8));
    assign rms_gamma_write_mask_n = byte_write_mask_n(
        rms_gamma_write.byte_enable);
    assign rms_gamma_stage_ready = !rst &&
        current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM;
    assign rms_tile_read_req_ready = !rst &&
        rms_phase_active &&
        (current_layout.phase != LOCAL_MEMORY_PHASE_FINAL_OUTPUT ||
         rms_gamma_bypass) &&
        (rms_tile_reserved_count < 4'd3 ||
         (rms_tile_read_rsp_valid && rms_tile_read_rsp_ready));

    for (genvar rms_tile_lane = 0; rms_tile_lane < 8;
         rms_tile_lane++) begin : g_rms_tile_lane
        assign rms_tile_lane_valid[rms_tile_lane] =
            rms_tile_read_req_valid && !rst &&
            rms_phase_active &&
            (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ||
             integer'(rms_tile_read_req.row_base) + rms_tile_lane <
                integer'(active_row_count));
        assign rms_tile_physical_row[rms_tile_lane] =
            rms_tile_read_req.row_base + 6'(rms_tile_lane);
        assign rms_tile_region[rms_tile_lane] =
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
                final_output_workspace_region(3'(rms_tile_lane)) :
                resident_region(rms_tile_physical_row[rms_tile_lane]);
        assign rms_tile_port[rms_tile_lane] = local_region_port_index(
            rms_tile_region[rms_tile_lane], 1'b0);
        assign rms_tile_macro_onehot[rms_tile_lane] =
            physical_port_macro_onehot(rms_tile_port[rms_tile_lane]);
        assign rms_tile_port_side[rms_tile_lane] =
            rms_tile_port[rms_tile_lane][0];
        assign rms_tile_address[rms_tile_lane] = local_region_row(
            rms_tile_region[rms_tile_lane],
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
                10'(integer'(rms_tile_read_req.element) / 8) :
                hidden_row_word(
                    rms_tile_read_req.row_base + 6'(rms_tile_lane),
                    9'(integer'(rms_tile_read_req.element) / 8)));
    end

    assign rms_scratch_read_region_value =
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
        final_norm_scratch_region(rms_scratch_read_req.bank) : rms_ffn_command ?
        matmul_staging_region(rms_scratch_read_req.bank) :
        rms_scratch_region(rms_scratch_read_req.bank);
    assign rms_scratch_read_left_row = local_region_row(
        rms_scratch_read_region_value, rms_scratch_read_req.left_address);
    assign rms_scratch_read_right_row = local_region_row(
        rms_scratch_read_region_value, rms_scratch_read_req.right_address);
    assign rms_scratch_read_primary_port = local_region_port_index(
        rms_scratch_read_region_value, 1'b0);
    assign rms_scratch_read_secondary_port = local_region_port_index(
        rms_scratch_read_region_value, 1'b1);
    assign rms_scratch_read_primary_macro_onehot =
        physical_port_macro_onehot(rms_scratch_read_primary_port);
    assign rms_scratch_read_secondary_macro_onehot =
        physical_port_macro_onehot(rms_scratch_read_secondary_port);
    assign rms_scratch_read_ready = !rst &&
        rms_phase_active &&
        (rms_scratch_reserved_count < 4'd3 ||
         (rms_scratch_read_rsp_valid && rms_scratch_read_rsp_ready));

    assign rms_scratch_write_region_value =
        current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
        final_norm_scratch_region(rms_scratch_write.bank) : rms_ffn_command ?
        matmul_staging_region(rms_scratch_write.bank) :
        rms_scratch_region(rms_scratch_write.bank);
    assign rms_scratch_write_row = local_region_row(
        rms_scratch_write_region_value, rms_scratch_write.address);
    assign rms_scratch_write_port = local_region_port_index(
        rms_scratch_write_region_value, 1'b0);
    assign rms_scratch_write_macro_onehot =
        physical_port_macro_onehot(rms_scratch_write_port);
    assign rms_scratch_write_mask_word = byte_write_mask_n(
        rms_scratch_write.byte_enable);
    assign rms_scratch_write_ready = !rst &&
        rms_phase_active;
    assign rms_norm_write_ready = !rst &&
        rms_phase_active;

    for (genvar rms_norm_lane = 0; rms_norm_lane < 8;
         rms_norm_lane++) begin : g_rms_norm_lane
        assign rms_norm_write_lane_valid[rms_norm_lane] =
            rms_norm_write_valid && !rst &&
            rms_phase_active &&
            |rms_norm_write.lane_mask[rms_norm_lane*8 +: 8];
        assign rms_norm_physical_row[rms_norm_lane] =
            rms_norm_write.row_base + 6'(rms_norm_lane);
        assign rms_norm_write_region[rms_norm_lane] =
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
                final_output_workspace_region(3'(rms_norm_lane)) :
            rms_ffn_command || attention_normalized_capture || active_row_count > 6'd32 ?
                resident_region(rms_norm_physical_row[rms_norm_lane]) :
                (rms_norm_physical_row[rms_norm_lane][3] ?
                    q_quantized_region(3'(rms_norm_lane)) :
                    qkv_activation_region(3'(rms_norm_lane)));
        assign rms_norm_write_port[rms_norm_lane] = local_region_port_index(
            rms_norm_write_region[rms_norm_lane], 1'b1);
        assign rms_norm_write_macro_onehot[rms_norm_lane] =
            physical_port_macro_onehot(rms_norm_write_port[rms_norm_lane]);
        assign rms_norm_write_port_side[rms_norm_lane] =
            rms_norm_write_port[rms_norm_lane][0];
        assign rms_norm_write_address[rms_norm_lane] = local_region_row(
            rms_norm_write_region[rms_norm_lane],
            current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT ?
                10'(integer'(rms_norm_write.element) / 8) :
                hidden_row_word(
                    rms_norm_write.row_base + 6'(rms_norm_lane),
                    9'(integer'(rms_norm_write.element) / 8)));
        assign rms_norm_write_lane_mask_n[rms_norm_lane] =
            bf16_lane_write_mask_n(
                rms_norm_write.lane_mask[rms_norm_lane*8 +: 8]);
    end

    for (genvar rms_port_number = 0;
         rms_port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         rms_port_number++) begin : g_rms_port
        localparam int unsigned PORT_MACRO = rms_port_number / 2;
        localparam logic PORT_SIDE = 1'(rms_port_number);
        logic endpoint_req_valid;
        logic endpoint_write;
        logic [11:0] endpoint_address;
        logic [127:0] endpoint_write_data;
        logic [127:0] endpoint_write_mask_n;
        logic [20:0] endpoint_source_match;
        endpoint_request_payload_t endpoint_source_payload [0:31];
        endpoint_request_payload_t endpoint_select_l0 [0:31];
        endpoint_request_payload_t endpoint_select_l1 [0:15];
        endpoint_request_payload_t endpoint_select_l2 [0:7];
        endpoint_request_payload_t endpoint_select_l3 [0:3];
        endpoint_request_payload_t endpoint_select_l4 [0:1];
        endpoint_request_payload_t endpoint_select;
        logic gamma_endpoint_match;
        logic scratch_read_primary_endpoint_match;
        logic scratch_read_secondary_endpoint_match;
        logic scratch_write_endpoint_match;
        logic [7:0] tile_lane_endpoint_match;
        logic [7:0] norm_lane_endpoint_match;

        assign gamma_endpoint_match =
            (!rms_ffn_command &&
             rms_port_number == RMS_ATTENTION_GAMMA_PORT) ||
            (rms_ffn_command && rms_port_number == RMS_FFN_GAMMA_PORT);
        assign scratch_read_primary_endpoint_match =
            rms_scratch_read_primary_macro_onehot[PORT_MACRO] &&
            rms_scratch_read_primary_port[0] == PORT_SIDE;
        assign scratch_read_secondary_endpoint_match =
            rms_scratch_read_secondary_macro_onehot[PORT_MACRO] &&
            rms_scratch_read_secondary_port[0] == PORT_SIDE;
        assign scratch_write_endpoint_match =
            rms_scratch_write_macro_onehot[PORT_MACRO] &&
            rms_scratch_write_port[0] == PORT_SIDE;
        for (genvar rms_endpoint_lane = 0; rms_endpoint_lane < 8;
             rms_endpoint_lane++) begin : g_endpoint_lane_match
            assign tile_lane_endpoint_match[rms_endpoint_lane] =
                rms_tile_macro_onehot[rms_endpoint_lane][PORT_MACRO] &&
                rms_tile_port_side[rms_endpoint_lane] == PORT_SIDE;
            assign norm_lane_endpoint_match[rms_endpoint_lane] =
                rms_norm_write_macro_onehot[rms_endpoint_lane][PORT_MACRO] &&
                rms_norm_write_port_side[rms_endpoint_lane] == PORT_SIDE;

            assign endpoint_source_match[rms_endpoint_lane+1] =
                rms_tile_lane_valid[rms_endpoint_lane] &&
                tile_lane_endpoint_match[rms_endpoint_lane];
            assign endpoint_source_payload[rms_endpoint_lane+1] =
                endpoint_source_match[rms_endpoint_lane+1] ?
                    {1'b0, rms_tile_address[rms_endpoint_lane],
                     128'd0, 128'd0} : '0;
            assign endpoint_source_match[rms_endpoint_lane+12] =
                rms_norm_write_lane_valid[rms_endpoint_lane] &&
                norm_lane_endpoint_match[rms_endpoint_lane];
            assign endpoint_source_payload[rms_endpoint_lane+12] =
                endpoint_source_match[rms_endpoint_lane+12] ?
                    {1'b1, rms_norm_write_address[rms_endpoint_lane],
                     rms_norm_write.data[rms_endpoint_lane*128 +: 128],
                     ~rms_norm_write_lane_mask_n[rms_endpoint_lane]} : '0;
        end

        assign endpoint_source_match[0] = rms_gamma_stage_valid && !rst &&
            current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM &&
            gamma_endpoint_match;
        assign endpoint_source_payload[0] = endpoint_source_match[0] ?
            {1'b1, rms_gamma_address, rms_gamma_write.data,
             ~rms_gamma_write_mask_n} : '0;
        assign endpoint_source_match[9] = rms_scratch_read_valid && !rst &&
            rms_phase_active &&
            scratch_read_primary_endpoint_match;
        assign endpoint_source_payload[9] = endpoint_source_match[9] ?
            {1'b0, rms_scratch_read_left_row, 128'd0, 128'd0} : '0;
        assign endpoint_source_match[10] = rms_scratch_read_valid && !rst &&
            rms_phase_active &&
            scratch_read_secondary_endpoint_match;
        assign endpoint_source_payload[10] = endpoint_source_match[10] ?
            {1'b0, rms_scratch_read_right_row, 128'd0, 128'd0} : '0;
        assign endpoint_source_match[11] = rms_scratch_write_valid && !rst &&
            rms_phase_active &&
            scratch_write_endpoint_match;
        assign endpoint_source_payload[11] = endpoint_source_match[11] ?
            {1'b1, rms_scratch_write_row, rms_scratch_write.data,
             ~rms_scratch_write_mask_word} : '0;
        assign endpoint_source_match[20] = rms_tile_read_req_valid && !rst &&
            current_layout.phase == LOCAL_MEMORY_PHASE_RMSNORM &&
            !rms_gamma_bypass &&
            gamma_endpoint_match;
        assign endpoint_source_payload[20] = endpoint_source_match[20] ?
            {1'b0, rms_gamma_read_address, 128'd0, 128'd0} : '0;

        for (genvar endpoint_source = 0; endpoint_source < 32;
             endpoint_source++) begin : g_endpoint_source
            if (endpoint_source < 21)
                assign endpoint_select_l0[endpoint_source] =
                    endpoint_source_payload[endpoint_source];
            else
                assign endpoint_select_l0[endpoint_source] = '0;
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 16;
             endpoint_tree_node++) begin : g_endpoint_select_l1
            assign endpoint_select_l1[endpoint_tree_node] =
                endpoint_select_l0[endpoint_tree_node*2] |
                endpoint_select_l0[endpoint_tree_node*2+1];
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 8;
             endpoint_tree_node++) begin : g_endpoint_select_l2
            assign endpoint_select_l2[endpoint_tree_node] =
                endpoint_select_l1[endpoint_tree_node*2] |
                endpoint_select_l1[endpoint_tree_node*2+1];
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 4;
             endpoint_tree_node++) begin : g_endpoint_select_l3
            assign endpoint_select_l3[endpoint_tree_node] =
                endpoint_select_l2[endpoint_tree_node*2] |
                endpoint_select_l2[endpoint_tree_node*2+1];
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 2;
             endpoint_tree_node++) begin : g_endpoint_select_l4
            assign endpoint_select_l4[endpoint_tree_node] =
                endpoint_select_l3[endpoint_tree_node*2] |
                endpoint_select_l3[endpoint_tree_node*2+1];
        end
        assign endpoint_select = endpoint_select_l4[0] |
            endpoint_select_l4[1];
        assign endpoint_req_valid = |endpoint_source_match;
        assign endpoint_write = endpoint_select.write;
        assign endpoint_address = endpoint_select.address;
        assign endpoint_write_data = endpoint_select.write_data;
        assign endpoint_write_mask_n = ~endpoint_select.write_bit_enable;

        assign rms_req_valid[rms_port_number] = endpoint_req_valid;
        assign rms_write[rms_port_number] = endpoint_write;
        assign rms_address[rms_port_number*12 +: 12] = endpoint_address;
        assign rms_write_data[rms_port_number*128 +: 128] =
            endpoint_write_data;
        assign rms_write_mask_n[rms_port_number*128 +: 128] =
            endpoint_write_mask_n;

`ifndef SYNTHESIS
        always_ff @(posedge clk) begin
            if (!rst)
                assert ($countones(endpoint_source_match) <= 1)
                    else $error("local_memory RMSNorm lanes selected one physical port");
        end
`endif
    end

    assign rms_tile_reserved_count = {1'b0, rms_tile_fifo_occupancy} +
        4'($countones(rms_tile_read_pending));
    assign rms_scratch_reserved_count = {1'b0, rms_scratch_fifo_occupancy} +
        4'($countones(rms_scratch_read_pending));

    always_comb begin : rmsnorm_response_mapping
        logic [128:0] response;

        if (rms_tile_read_pending_gamma_bypass[1]) begin
            response = {1'b1, {8{16'h3f80}}};
        end else begin
            case (rms_ffn_command)
                1'b0: response = {
                    port_read_valid[RMS_ATTENTION_GAMMA_PORT],
                    port_read_data[RMS_ATTENTION_GAMMA_PORT*128 +: 128]};
                default: response = {
                    port_read_valid[RMS_FFN_GAMMA_PORT],
                    port_read_data[RMS_FFN_GAMMA_PORT*128 +: 128]};
            endcase
        end
        rms_tile_response_complete = rms_tile_read_pending[1] && response[128];
        rms_tile_fifo_input_data = '0;
        rms_tile_fifo_input_data[1151:1024] = response[127:0];
        rms_tile_fifo_input_data[1215:1152] = rms_tile_read_pending_mask[1];
        rms_tile_fifo_input_data[1231:1216] = rms_tile_read_pending_tag[1];
        for (integer tile_row = 0; tile_row < 8; tile_row++) begin
            if (|rms_tile_read_pending_mask[1][tile_row*8 +: 8]) begin
                response = rms_aligned_hidden_response[tile_row];
                rms_tile_response_complete &= response[128];
                rms_tile_fifo_input_data[tile_row*128 +: 128] =
                    response[127:0];
            end
        end
        rms_tile_fifo_input_valid = rms_tile_response_complete;

        rms_scratch_fifo_input_data = '0;
        case ({current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT,
               rms_ffn_command && current_layout.phase != LOCAL_MEMORY_PHASE_FINAL_OUTPUT,
               rms_scratch_read_pending_bank[1]})
            3'b000: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_SCRATCH0_PRIMARY_PORT] &&
                    port_read_valid[RMS_SCRATCH0_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_SCRATCH0_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_SCRATCH0_PRIMARY_PORT*128 +: 128]};
            end
            3'b001: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_SCRATCH1_PRIMARY_PORT] &&
                    port_read_valid[RMS_SCRATCH1_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_SCRATCH1_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_SCRATCH1_PRIMARY_PORT*128 +: 128]};
            end
            3'b010: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_FFN_SCRATCH0_PRIMARY_PORT] &&
                    port_read_valid[RMS_FFN_SCRATCH0_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_FFN_SCRATCH0_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_FFN_SCRATCH0_PRIMARY_PORT*128 +: 128]};
            end
            3'b011: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_FFN_SCRATCH1_PRIMARY_PORT] &&
                    port_read_valid[RMS_FFN_SCRATCH1_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_FFN_SCRATCH1_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_FFN_SCRATCH1_PRIMARY_PORT*128 +: 128]};
            end
            3'b100: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_FINAL_SCRATCH0_PRIMARY_PORT] &&
                    port_read_valid[RMS_FINAL_SCRATCH0_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_FINAL_SCRATCH0_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_FINAL_SCRATCH0_PRIMARY_PORT*128 +: 128]};
            end
            default: begin
                rms_scratch_response_complete = rms_scratch_read_pending[1] &&
                    port_read_valid[RMS_FINAL_SCRATCH1_PRIMARY_PORT] &&
                    port_read_valid[RMS_FINAL_SCRATCH1_SECONDARY_PORT];
                rms_scratch_fifo_input_data = {
                    rms_scratch_read_pending_tag[1],
                    port_read_data[RMS_FINAL_SCRATCH1_SECONDARY_PORT*128 +: 128],
                    port_read_data[RMS_FINAL_SCRATCH1_PRIMARY_PORT*128 +: 128]};
            end
        endcase
        rms_scratch_fifo_input_valid = rms_scratch_response_complete;
    end

    always_ff @(posedge clk) begin : rmsnorm_response_metadata
        if (rst) begin
            rms_tile_read_pending <= '0;
            rms_tile_read_pending_gamma_bypass <= '0;
            rms_tile_read_pending_row_base[0] <= '0;
            rms_tile_read_pending_row_base[1] <= '0;
            rms_tile_read_pending_tag[0] <= '0;
            rms_tile_read_pending_tag[1] <= '0;
            rms_tile_read_pending_mask[0] <= '0;
            rms_tile_read_pending_mask[1] <= '0;
            rms_scratch_read_pending <= '0;
            rms_scratch_read_pending_bank[0] <= 1'b0;
            rms_scratch_read_pending_bank[1] <= 1'b0;
            rms_scratch_read_pending_tag[0] <= '0;
            rms_scratch_read_pending_tag[1] <= '0;
        end else begin
            rms_tile_read_pending <= {rms_tile_read_pending[0],
                rms_tile_read_req_valid && rms_tile_read_req_ready};
            rms_tile_read_pending_gamma_bypass <= {
                rms_tile_read_pending_gamma_bypass[0],
                rms_tile_read_req_valid && rms_tile_read_req_ready &&
                    rms_gamma_bypass};
            rms_tile_read_pending_row_base[1] <=
                rms_tile_read_pending_row_base[0];
            rms_tile_read_pending_tag[1] <= rms_tile_read_pending_tag[0];
            rms_tile_read_pending_mask[1] <= rms_tile_read_pending_mask[0];
            if (rms_tile_read_req_valid && rms_tile_read_req_ready) begin
                rms_tile_read_pending_row_base[0] <= rms_tile_read_req.row_base;
                rms_tile_read_pending_tag[0] <= rms_tile_read_req.tag;
                for (integer tile_row = 0; tile_row < 8; tile_row++)
                    for (integer tile_column = 0; tile_column < 8; tile_column++)
                        rms_tile_read_pending_mask[0][tile_row*8 + tile_column] <=
                            (current_layout.phase ==
                                LOCAL_MEMORY_PHASE_FINAL_OUTPUT ||
                             integer'(rms_tile_read_req.row_base) + tile_row <
                                integer'(active_row_count)) &&
                            integer'(rms_tile_read_req.element) + tile_column < 4096;
            end
            rms_scratch_read_pending <= {rms_scratch_read_pending[0],
                rms_scratch_read_valid && rms_scratch_read_ready};
            rms_scratch_read_pending_bank[1] <= rms_scratch_read_pending_bank[0];
            rms_scratch_read_pending_tag[1] <= rms_scratch_read_pending_tag[0];
            if (rms_scratch_read_valid && rms_scratch_read_ready) begin
                rms_scratch_read_pending_bank[0] <= rms_scratch_read_req.bank;
                rms_scratch_read_pending_tag[0] <= rms_scratch_read_req.tag;
            end
        end
    end

    ready_valid_fifo #(.DATA_WIDTH(1232), .DEPTH(3)) rms_tile_response_fifo (
        .clk(clk), .rst(rst), .input_valid(rms_tile_fifo_input_valid),
        .input_ready(rms_tile_fifo_input_ready),
        .input_data(rms_tile_fifo_input_data),
        .output_valid(rms_tile_read_rsp_valid),
        .output_ready(rms_tile_read_rsp_ready),
        .output_data({rms_tile_read_rsp.tag, rms_tile_read_rsp.lane_mask,
            rms_tile_read_rsp.gamma, rms_tile_read_rsp.values}),
        .occupancy(rms_tile_fifo_occupancy)
    );

    ready_valid_fifo #(.DATA_WIDTH(272), .DEPTH(3)) rms_scratch_response_fifo (
        .clk(clk), .rst(rst), .input_valid(rms_scratch_fifo_input_valid),
        .input_ready(rms_scratch_fifo_input_ready),
        .input_data(rms_scratch_fifo_input_data),
        .output_valid(rms_scratch_read_rsp_valid),
        .output_ready(rms_scratch_read_rsp_ready),
        .output_data({rms_scratch_read_rsp.tag, rms_scratch_read_rsp.right_data,
            rms_scratch_read_rsp.left_data}),
        .occupancy(rms_scratch_fifo_occupancy)
    );

    for (genvar port_number = 0;
         port_number < LOCAL_SRAM_PHYSICAL_PORT_COUNT;
         port_number++) begin : g_port
        logic [12:0] endpoint_source_valid;
        endpoint_request_payload_t endpoint_source_payload [0:15];
        endpoint_request_payload_t endpoint_select_l0 [0:15];
        endpoint_request_payload_t endpoint_select_l1 [0:7];
        endpoint_request_payload_t endpoint_select_l2 [0:3];
        endpoint_request_payload_t endpoint_select_l3 [0:1];
        endpoint_request_payload_t endpoint_select;

        assign endpoint_source_valid = {
            candidate_state_req_valid[port_number],
            matmul_req_valid[port_number],
            cache_write_req_valid[port_number],
            attention_probability_req_valid[port_number],
            attention_score_req_valid[port_number],
            attention_operand_port_valid[port_number],
            qkv_tile_read_req_valid[port_number],
            qkv_stage_write_req_valid[port_number],
            rope_req_valid[port_number],
            qkv_head_read_port_valid[port_number],
            attention_scratch_req_valid[port_number],
            context_req_valid[port_number],
            hidden_req_valid[port_number]
        };

        assign endpoint_source_payload[0] = endpoint_source_valid[0] ?
            {hidden_write[port_number], hidden_address[port_number*12 +: 12],
             hidden_port_write_data[port_number*128 +: 128],
             ~hidden_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[1] = endpoint_source_valid[1] ?
            {context_write[port_number], context_address[port_number*12 +: 12],
             context_port_write_data[port_number*128 +: 128],
             ~context_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[2] = endpoint_source_valid[2] ?
            {attention_scratch_write[port_number],
             attention_scratch_address[port_number*12 +: 12],
             attention_scratch_port_write_data[port_number*128 +: 128],
             ~attention_scratch_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[3] = endpoint_source_valid[3] ?
            {1'b0, qkv_head_read_address[port_number*12 +: 12],
             128'd0, 128'd0} : '0;
        assign endpoint_source_payload[4] = endpoint_source_valid[4] ?
            {rope_write[port_number], rope_address[port_number*12 +: 12],
             rope_write_data[port_number*128 +: 128],
             ~rope_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[5] = endpoint_source_valid[5] ?
            {1'b1, qkv_stage_write_address[port_number*12 +: 12],
             qkv_stage_write_port_data[port_number*128 +: 128],
             ~qkv_stage_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[6] = endpoint_source_valid[6] ?
            {1'b0, qkv_tile_read_address[port_number*12 +: 12],
             128'd0, 128'd0} : '0;
        assign endpoint_source_payload[7] = endpoint_source_valid[7] ?
            {1'b0, attention_operand_address[port_number*12 +: 12],
             128'd0, 128'd0} : '0;
        assign endpoint_source_payload[8] = endpoint_source_valid[8] ?
            {attention_score_port_write[port_number],
             attention_score_address[port_number*12 +: 12],
             attention_score_write_data[port_number*128 +: 128],
             ~attention_score_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[9] = endpoint_source_valid[9] ?
            {1'b1, attention_probability_address[port_number*12 +: 12],
             attention_probability_write_data[port_number*128 +: 128],
             ~attention_probability_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[10] = endpoint_source_valid[10] ?
            {1'b1, cache_write_address[port_number*12 +: 12],
             cache_write_data[port_number*128 +: 128],
             ~cache_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[11] = endpoint_source_valid[11] ?
            {matmul_write[port_number], matmul_address[port_number*12 +: 12],
             matmul_write_data[port_number*128 +: 128],
             ~matmul_write_mask_n[port_number*128 +: 128]} : '0;
        assign endpoint_source_payload[12] = endpoint_source_valid[12] ?
            {candidate_state_write[port_number],
             candidate_state_address[port_number*12 +: 12],
             candidate_state_port_write_data[port_number*128 +: 128],
             ~candidate_state_port_write_mask_n[port_number*128 +: 128]} : '0;

        for (genvar endpoint_source = 0; endpoint_source < 16;
             endpoint_source++) begin : g_endpoint_source
            if (endpoint_source < 13)
                assign endpoint_select_l0[endpoint_source] =
                    endpoint_source_payload[endpoint_source];
            else
                assign endpoint_select_l0[endpoint_source] = '0;
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 8;
             endpoint_tree_node++) begin : g_endpoint_select_l1
            assign endpoint_select_l1[endpoint_tree_node] =
                endpoint_select_l0[endpoint_tree_node*2] |
                endpoint_select_l0[endpoint_tree_node*2+1];
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 4;
             endpoint_tree_node++) begin : g_endpoint_select_l2
            assign endpoint_select_l2[endpoint_tree_node] =
                endpoint_select_l1[endpoint_tree_node*2] |
                endpoint_select_l1[endpoint_tree_node*2+1];
        end
        for (genvar endpoint_tree_node = 0; endpoint_tree_node < 2;
             endpoint_tree_node++) begin : g_endpoint_select_l3
            assign endpoint_select_l3[endpoint_tree_node] =
                endpoint_select_l2[endpoint_tree_node*2] |
                endpoint_select_l2[endpoint_tree_node*2+1];
        end
        assign endpoint_select = endpoint_select_l3[0] |
            endpoint_select_l3[1];

        assign logical_req_valid[port_number] = |endpoint_source_valid;
        assign port_req_valid[port_number] = logical_req_valid[port_number];
        assign port_write[port_number] = endpoint_select.write;
        assign port_address[port_number*12 +: 12] =
            endpoint_select.address;
        assign port_write_data[port_number*128 +: 128] =
            endpoint_select.write_data;
        assign port_write_mask_n[port_number*128 +: 128] =
            ~endpoint_select.write_bit_enable;
        assign port_double_drive[port_number] =
            port_req_valid[port_number] && client_req_ready[port_number] &&
            rms_req_valid[port_number];
        assign refresh_metadata_scratch_selected[port_number] =
            port_number == 50 && refresh_metadata_scratch_req_valid &&
            !rms_req_valid[port_number];
        assign refresh_metadata_scratch_aux_selected[port_number] =
            port_number == 51 && refresh_metadata_scratch_aux_write_valid &&
            !rms_req_valid[port_number];

`ifndef SYNTHESIS
        always_ff @(posedge clk) begin
            if (!rst)
                assert ($onehot0(endpoint_source_valid))
                    else $error(
                        "local_memory endpoint %0d had multiple logical owners 0x%0h",
                        port_number, endpoint_source_valid);
        end
`endif

        assign macro_req_valid[port_number] =
            port_req_valid[port_number] || rms_req_valid[port_number] ||
            refresh_metadata_scratch_selected[port_number] ||
            refresh_metadata_scratch_aux_selected[port_number];
        assign macro_write[port_number] = rms_req_valid[port_number] ?
            rms_write[port_number] :
            refresh_metadata_scratch_selected[port_number] ?
                refresh_metadata_scratch_write :
            refresh_metadata_scratch_aux_selected[port_number] ?
                1'b1 : port_write[port_number];
        assign macro_address[port_number*12 +: 12] = rms_req_valid[port_number] ?
            rms_address[port_number*12 +: 12] :
            refresh_metadata_scratch_selected[port_number] ?
                {3'd0, refresh_metadata_scratch_address} :
            refresh_metadata_scratch_aux_selected[port_number] ?
                {3'd0, refresh_metadata_scratch_aux_write_address} :
                port_address[port_number*12 +: 12];
        assign macro_write_data[port_number*128 +: 128] =
            rms_req_valid[port_number] ?
            rms_write_data[port_number*128 +: 128] :
            refresh_metadata_scratch_selected[port_number] ?
                {69'd0, refresh_metadata_scratch_write_data} :
            refresh_metadata_scratch_aux_selected[port_number] ?
                {69'd0, refresh_metadata_scratch_aux_write_data, 39'd0} :
                port_write_data[port_number*128 +: 128];
        assign macro_write_mask_n[port_number*128 +: 128] =
            rms_req_valid[port_number] ?
            rms_write_mask_n[port_number*128 +: 128] :
            refresh_metadata_scratch_selected[port_number] ?
                {{69{1'b1}},
                 ~refresh_metadata_scratch_write_enable} :
            refresh_metadata_scratch_aux_selected[port_number] ?
                {{69{1'b1}}, ~refresh_metadata_scratch_aux_write_enable,
                 {39{1'b1}}} :
                port_write_mask_n[port_number*128 +: 128];
        assign client_req_ready[port_number] = macro_req_ready[port_number] &&
            !rms_req_valid[port_number] &&
            !refresh_metadata_scratch_selected[port_number] &&
            !refresh_metadata_scratch_aux_selected[port_number];
        assign port_req_ready[port_number] = client_req_ready[port_number] &&
            !logical_req_valid[port_number];
    end

    assign refresh_metadata_scratch_req_ready =
        macro_req_ready[50] && !rms_req_valid[50];
    assign refresh_metadata_scratch_aux_write_ready =
        macro_req_ready[51] && !rms_req_valid[51];
    assign refresh_metadata_scratch_rsp_valid =
        refresh_metadata_scratch_read_pending[1];
    assign refresh_metadata_scratch_rsp_data = port_read_data[50*128 +: 59];

    always_ff @(posedge clk) begin
        if (rst)
            refresh_metadata_scratch_read_pending <= '0;
        else
            refresh_metadata_scratch_read_pending <= {
                refresh_metadata_scratch_read_pending[0],
                refresh_metadata_scratch_req_valid &&
                refresh_metadata_scratch_req_ready &&
                !refresh_metadata_scratch_write};
    end

    for (genvar macro_number = 0;
         macro_number < LOCAL_SRAM_MACRO_COUNT;
         macro_number++) begin : g_macro
        localparam int unsigned MACRO_DEPTH =
            macro_number < LOCAL_SRAM_2048_MACRO_COUNT ? 2048 : 1024;
        local_sram_macro #(.DEPTH(MACRO_DEPTH)) macro (
            .clk(clk), .rst(rst),
            .a_req_valid(macro_req_valid[macro_number*2]),
            .a_req_ready(macro_req_ready[macro_number*2]),
            .a_write(macro_write[macro_number*2]),
            .a_address(macro_address[macro_number*24 +: 12]),
            .a_write_data(macro_write_data[macro_number*256 +: 128]),
            .a_write_mask_n(macro_write_mask_n[macro_number*256 +: 128]),
            .a_read_valid(port_read_valid[macro_number*2]),
            .a_read_data(port_read_data[macro_number*256 +: 128]),
            .b_req_valid(macro_req_valid[macro_number*2+1]),
            .b_req_ready(macro_req_ready[macro_number*2+1]),
            .b_write(macro_write[macro_number*2+1]),
            .b_address(macro_address[macro_number*24+12 +: 12]),
            .b_write_data(macro_write_data[macro_number*256+128 +: 128]),
            .b_write_mask_n(macro_write_mask_n[macro_number*256+128 +: 128]),
            .b_read_valid(port_read_valid[macro_number*2+1]),
            .b_read_data(port_read_data[macro_number*256+128 +: 128]),
            .same_address_conflict(macro_same_address_conflict[macro_number])
        );
    end

`ifndef SYNTHESIS
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_attention_operand_request_mask [0:1];
    logic [LOCAL_SRAM_PHYSICAL_PORT_COUNT-1:0]
        debug_attention_operand_macro_read_mask [0:1];

    initial begin : check_candidate_state_fixed_ports
        assert (CANDIDATE_P0_S0_PRIMARY_PORT == 6'd49 &&
                CANDIDATE_P0_S0_SECONDARY_PORT == 6'd48 &&
                CANDIDATE_P1_S0_PRIMARY_PORT == 6'd44 &&
                CANDIDATE_P1_S0_SECONDARY_PORT == 6'd45 &&
                CANDIDATE_P0_S1_PRIMARY_PORT == 6'd33 &&
                CANDIDATE_P0_S1_SECONDARY_PORT == 6'd32 &&
                CANDIDATE_P1_S1_PRIMARY_PORT == 6'd20 &&
                CANDIDATE_P1_S1_SECONDARY_PORT == 6'd21 &&
                CANDIDATE_P0_S2_PRIMARY_PORT == 6'd51 &&
                CANDIDATE_P0_S2_SECONDARY_PORT == 6'd50 &&
                CANDIDATE_P1_S2_PRIMARY_PORT == 6'd46 &&
                CANDIDATE_P1_S2_SECONDARY_PORT == 6'd47 &&
                CANDIDATE_P0_S3_PRIMARY_PORT == 6'd15 &&
                CANDIDATE_P0_S3_SECONDARY_PORT == 6'd14 &&
                CANDIDATE_P1_S3_PRIMARY_PORT == 6'd36 &&
                CANDIDATE_P1_S3_SECONDARY_PORT == 6'd37)
            else $error("local_memory Candidate fixed ports do not match the configured SRAM placement");
    end

    always_ff @(posedge clk) begin
        if (!rst && refresh_metadata_scratch_req_valid) begin
            assert (refresh_metadata_scratch_address < 9'd432)
                else $error("local_memory refresh metadata scratch address out of range");
            assert (!rms_req_valid[50] && !logical_req_valid[50])
                else $error("local_memory refresh metadata scratch port 50 conflict");
            if (refresh_metadata_scratch_write)
                assert (refresh_metadata_scratch_write_enable != 59'd0)
                    else $error("local_memory refresh metadata scratch empty write");
        end
        if (!rst && refresh_metadata_scratch_aux_write_valid) begin
            assert (refresh_metadata_scratch_aux_write_address < 9'd432)
                else $error("local_memory refresh metadata scratch auxiliary address out of range");
            assert (!rms_req_valid[51] && !logical_req_valid[51])
                else $error("local_memory refresh metadata scratch port 51 conflict");
            assert (refresh_metadata_scratch_aux_write_enable != 20'd0)
                else $error("local_memory refresh metadata scratch auxiliary empty write");
        end
        if (rst) begin
            debug_attention_operand_request_mask[0] <= '0;
            debug_attention_operand_request_mask[1] <= '0;
            debug_attention_operand_macro_read_mask[0] <= '0;
            debug_attention_operand_macro_read_mask[1] <= '0;
        end else begin
            debug_attention_operand_request_mask[1] <=
                debug_attention_operand_request_mask[0];
            debug_attention_operand_request_mask[0] <=
                attention_operand_req_valid && attention_operand_req_ready ?
                    attention_operand_port_valid : '0;
            debug_attention_operand_macro_read_mask[1] <=
                debug_attention_operand_macro_read_mask[0];
            debug_attention_operand_macro_read_mask[0] <=
                attention_operand_req_valid && attention_operand_req_ready ?
                    (macro_req_valid & macro_req_ready & ~macro_write) : '0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(|port_double_drive))
                else $error("local_memory RMSNorm and top-level client drove one physical port");
            assert (!(|(hidden_req_valid & context_req_valid)))
                else $error("local_memory hidden and context bundles drove one physical port");
            assert (!(|(attention_scratch_req_valid &
                (hidden_req_valid | context_req_valid))))
                else $error("local_memory attention scratch overlapped another logical bundle");
            assert (!(|(qkv_head_read_port_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid))))
                else $error("local_memory QKV head read overlapped another logical bundle");
            assert (!(|(rope_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid))))
                else $error("local_memory RoPE bundle overlapped another logical bundle");
            assert (!(|(qkv_stage_write_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid))))
                else $error("local_memory QKV stage write overlapped another logical bundle");
            assert (!(|(qkv_tile_read_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid))))
                else $error("local_memory QKV tile read overlapped another logical bundle");
            assert (!(|(attention_operand_port_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid))))
                else $error("local_memory Attention operand overlapped another logical bundle");
            assert (!(|(attention_score_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid |
                attention_operand_port_valid))))
                else $error("local_memory Attention score overlap phase=%0d q_head=%0b score=%h hidden=%h context=%h scratch=%h head=%h rope=%h stage_w=%h tile_r=%h attn_operand=%h",
                    current_layout.phase, current_layout.q_head_active,
                    attention_score_req_valid, hidden_req_valid,
                    context_req_valid, attention_scratch_req_valid,
                    qkv_head_read_port_valid, rope_req_valid,
                    qkv_stage_write_req_valid, qkv_tile_read_req_valid,
                    attention_operand_port_valid);
            assert (!(|(attention_probability_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid |
                attention_operand_port_valid | attention_score_req_valid))))
                else $error("local_memory Attention probability overlap phase=%0d q_head=%0b probability=%h hidden=%h context=%h scratch=%h head=%h rope=%h stage_w=%h tile_r=%h attn_operand=%h score=%h",
                    current_layout.phase, current_layout.q_head_active,
                    attention_probability_req_valid, hidden_req_valid,
                    context_req_valid, attention_scratch_req_valid,
                    qkv_head_read_port_valid, rope_req_valid,
                    qkv_stage_write_req_valid, qkv_tile_read_req_valid,
                    attention_operand_port_valid, attention_score_req_valid);
            assert (!(|(candidate_state_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid |
                attention_operand_port_valid | attention_score_req_valid |
                attention_probability_req_valid | cache_write_req_valid |
                matmul_req_valid))))
                else $error("local_memory Candidate state overlapped another logical bundle");
            assert (!(candidate_state_read_valid && candidate_state_write_valid))
                else $error("local_memory Candidate state read and write were simultaneous");
            if (candidate_state_read_valid || candidate_state_write_valid)
                assert (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT)
                    else $error("local_memory Candidate state request was outside final-output phase");
            if (candidate_state_read_valid || candidate_state_write_valid)
                assert (candidate_state_shape_valid)
                    else $error("local_memory Candidate state request had an invalid address shape");
            assert (candidate_state_read_reserved_count <= 3'd4)
                else $error("local_memory Candidate state reserved more than four responses");
            if (candidate_state_fifo_input_valid)
                assert (candidate_state_fifo_input_ready)
                    else $error("local_memory Candidate state response FIFO overflow");
            if (candidate_state_read_pending[1]) begin
                for (integer candidate_row = 0; candidate_row < 4;
                     candidate_row++) begin
                    local_sram_region_t candidate_region;
                    logic [5:0] candidate_port;
                    candidate_region = candidate_state_region(
                        candidate_state_read_pending_panel[1],
                        2'(candidate_row));
                    candidate_port = local_region_port_index(
                        candidate_region,
                        candidate_state_read_pending_secondary[1]);
                    if (candidate_state_read_pending_mask[1][candidate_row])
                        assert (port_read_valid[candidate_port])
                            else $error("local_memory Candidate state response was incomplete");
                end
            end
            assert (!(|(cache_write_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid |
                attention_operand_port_valid | attention_score_req_valid |
                attention_probability_req_valid))))
                else $error("local_memory cache write overlap phase=%0d q_head=%0b cache=%h hidden=%h context=%h scratch=%h head=%h rope=%h stage_w=%h tile_r=%h attn_operand=%h score=%h probability=%h",
                    current_layout.phase, current_layout.q_head_active,
                    cache_write_req_valid, hidden_req_valid, context_req_valid,
                    attention_scratch_req_valid, qkv_head_read_port_valid,
                    rope_req_valid, qkv_stage_write_req_valid,
                    qkv_tile_read_req_valid, attention_operand_port_valid,
                    attention_score_req_valid,
                    attention_probability_req_valid);
            assert (!(|(matmul_req_valid & (hidden_req_valid |
                context_req_valid | attention_scratch_req_valid |
                qkv_head_read_port_valid | rope_req_valid |
                qkv_stage_write_req_valid | qkv_tile_read_req_valid |
                attention_operand_port_valid | attention_score_req_valid |
                attention_probability_req_valid | cache_write_req_valid))))
                else $error("local_memory Matmul overlap phase=%0d q_head=%0b matmul=%h hidden=%h context=%h scratch=%h head=%h rope=%h stage_w=%h tile_r=%h attn_operand=%h score=%h probability=%h cache=%h",
                    current_layout.phase, current_layout.q_head_active,
                    matmul_req_valid, hidden_req_valid, context_req_valid,
                    attention_scratch_req_valid, qkv_head_read_port_valid,
                    rope_req_valid, qkv_stage_write_req_valid,
                    qkv_tile_read_req_valid, attention_operand_port_valid,
                    attention_score_req_valid,
                    attention_probability_req_valid, cache_write_req_valid);
            assert (!(|(matmul_source_port_mask & matmul_prefetch_port_mask)))
                else $error("local_memory Matmul source and panel prefetch selected one endpoint");
            assert (!(|(matmul_source_port_mask &
                (matmul_operand_port_mask | matmul_other_port_mask))))
                else $error("local_memory Matmul source and compute request selected one endpoint");
            assert (!(|(matmul_operand_port_mask & matmul_prefetch_port_mask)))
                else $error("local_memory Matmul operand and panel prefetch selected one endpoint");
            assert (!(|(matmul_operand_port_mask & matmul_other_port_mask)))
                else $error("local_memory Matmul operand and auxiliary request selected one endpoint");
            assert (!(|(matmul_prefetch_port_mask & matmul_other_port_mask)))
                else $error("local_memory Matmul panel prefetch and auxiliary request selected one endpoint");
            if (matmul_local_source_req_valid)
                assert (matmul_local_source_req.row_count != 0 &&
                        matmul_local_source_req.row_count <= 8)
                    else $error("local_memory Matmul source row count was outside 1..8");
            if (matmul_local_source_req_valid &&
                matmul_local_source_req.row_count > 4'd1)
                assert (matmul_local_source_req.row[2:0] == 3'd0)
                    else $error("local_memory multi-row Matmul source was not M8-aligned");
            if (matmul_source_read_pending[1])
                assert (matmul_local_source_rsp_valid)
                    else $error("local_memory Matmul source response was incomplete");
            if (matmul_operand_read_pending[1])
                assert (matmul_read_response_valid)
                    else $error("local_memory Matmul operand response incomplete phase=%0d panel_layout=%0d projection_pair=%0b expanded=%0b port_valid=%013h word=%0d count=%0d padded=%0b explicit=%0b row_base=%0d rows=%012h lane_words=%032h panel=%0b panel_mask=%03h q_work=%0b",
                        current_layout.phase, current_layout.panel_layout,
                        projection_pair_layout, matmul_expanded_layout,
                        port_read_valid, matmul_pending_activation_word_index[1],
                        matmul_pending_activation_word_count[1],
                        matmul_pending_qkv_padded_layout[1],
                        matmul_pending_explicit_activation_rows[1],
                        matmul_pending_physical_row_base[1],
                        matmul_pending_activation_physical_row[1],
                        matmul_pending_activation_lane_word_index[1],
                        matmul_pending_panel[1], matmul_pending_panel_mask[1],
                        matmul_pending_q_work[1]);
            if (matmul_combine_read_pending[1])
                assert (matmul_combine_read_response_valid)
                    else $error("local_memory Matmul combine response was incomplete");
            if (matmul_combine_write_valid)
                assert (matmul_combine_write.address <
                        10'(MATMUL_COMBINE_DEPTH))
                    else $error("local_memory Matmul combine write address exceeded 383");
            if (matmul_combine_read_valid)
                assert (matmul_combine_read_req.address <
                        10'(MATMUL_COMBINE_DEPTH))
                    else $error("local_memory Matmul combine read address exceeded 383");
            if ((matmul_combine_write_valid &&
                 matmul_combine_write.address >= 10'(MATMUL_STAGING_DEPTH)) ||
                (matmul_combine_read_valid &&
                 matmul_combine_read_req.address >= 10'(MATMUL_STAGING_DEPTH)))
                assert (matmul_gate_scratch_layout)
                    else $error("local_memory Gate scratch requires Matmul MIXED/EXPANDED layout");
            if (matmul_combine_write_valid && matmul_combine_write_ready &&
                matmul_combine_read_valid && matmul_combine_read_ready)
                assert (matmul_combine_write.address !=
                        matmul_combine_read_req.address)
                    else $error("local_memory Matmul staging read and write selected one address");
            if (matmul_residual_read_pending[1])
                assert (matmul_residual_read_rsp_valid)
                    else $error("local_memory Matmul residual response was incomplete");
            assert (!(lm_head_scale_write_valid && lm_head_scale_read_valid))
                else $error("local_memory LM-head scale read and write were simultaneous");
            assert (post_table_request_exclusive)
                else $error("local_memory post scale, state and suppressed-token requests were simultaneous");
            assert (!(post_state_write_valid && post_state_read_valid))
                else $error("local_memory post state read and write were simultaneous");
            assert (!(post_suppressed_write_valid && post_suppressed_read_valid))
                else $error("local_memory suppressed-token read and write were simultaneous");
            if (lm_head_scale_write_valid || lm_head_scale_read_valid)
                assert (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT)
                    else $error("local_memory LM-head scale request was outside final-output phase");
            if (post_table_access_valid)
                assert (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT)
                    else $error("local_memory post table access changed outside final-output phase");
            if (post_state_write_valid || post_state_read_valid)
                assert (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
                        (post_table_access_enabled ||
                         (post_state_write_valid &&
                          post_state_write_word_address >= 10'd128 &&
                          post_state_write_word_address < 10'd480) ||
                         (post_state_read_valid &&
                          post_state_read_word_address >= 10'd128 &&
                          post_state_read_word_address < 10'd480)))
                    else $error("local_memory post table request used the wrong SRAM role");
            if (post_suppressed_write_valid || post_suppressed_read_valid)
                assert (current_layout.phase == LOCAL_MEMORY_PHASE_FINAL_OUTPUT &&
                        post_table_access_enabled)
                    else $error("local_memory suppressed-token request used the wrong SRAM role");
            if (post_suppressed_read_valid)
                assert (post_suppressed_read_index < 8'd255)
                    else $error("local_memory suppressed-token index exceeded 254");
            if (lm_head_scale_read_pending[1])
                assert (lm_head_scale_read_rsp_valid)
                    else $error("local_memory LM-head scale response was incomplete");
            if (post_state_read_pending[1])
                assert (post_state_read_rsp_valid)
                    else $error("local_memory post state response was incomplete");
            if (post_suppressed_read_pending[1])
                assert (post_suppressed_read_rsp_valid)
                    else $error("local_memory suppressed-token response was incomplete");
            assert ($onehot0(matmul_combine_read_pending))
                else $error("local_memory Matmul combine had more than one outstanding read");
            assert ($onehot0(matmul_residual_read_pending))
                else $error("local_memory Matmul residual had more than one outstanding read");
            assert ($onehot0(lm_head_scale_read_pending))
                else $error("local_memory LM-head scale buffer had more than one outstanding read");
            if (matmul_panel_write_valid &&
                |matmul_panel_write.byte_enable[31:16])
                assert (matmul_panel_write_port[0] !=
                        matmul_panel_write_port[1])
                    else $error("local_memory Matmul panel prefetch halves selected one endpoint");
            if (matmul_activation_write_valid && matmul_activation_write_ready)
                for (integer first_slot = 0; first_slot < 8; first_slot++)
                    for (integer second_slot = first_slot + 1; second_slot < 8; second_slot++)
                        if (matmul_activation_write.slot_valid[first_slot] && matmul_activation_write.slot_valid[second_slot])
                            assert (matmul_activation_write_port[first_slot] != matmul_activation_write_port[second_slot])
                                else $error("local_memory quantized write slots %0d/%0d selected the same SRAM port %0d",
                                    first_slot, second_slot, matmul_activation_write_port[first_slot]);
            if (matmul_panel_write_valid && !matmul_local_layout &&
                !current_layout.q_head_active) begin
                assert (current_layout.panel_layout == LOCAL_PANEL_LAYOUT_QKV)
                    else $error("local_memory nonlocal Matmul panel write was outside QKV layout");
                if (!matmul_panel_write.plane[0])
                    assert (matmul_panel_write.word_row < 10'd512)
                        else $error("local_memory QKV base panel write row exceeded 511");
                else
                    assert (matmul_panel_write.word_row >= 10'd512)
                        else $error("local_memory QKV enhancement panel write row was below 512");
            end
            if (matmul_read_bundle_valid) begin
                for (integer lane = 0; lane < 8; lane++)
                    if (lane < matmul_read_req.activation_word_count)
                        assert (matmul_activation_read_port[lane] <
                                6'(LOCAL_SRAM_PHYSICAL_PORT_COUNT))
                            else $error("local_memory Matmul activation port out of range lane=%0d port=%0d word=%0d phase=%0d panel_layout=%0d",
                                lane, matmul_activation_read_port[lane],
                                matmul_read_req.activation_lane_word_index[lane*16 +: 16],
                                current_layout.phase, current_layout.panel_layout);
                if (!matmul_local_layout && !current_layout.q_head_active)
                    assert (current_layout.panel_layout ==
                            LOCAL_PANEL_LAYOUT_QKV)
                        else $error("local_memory nonlocal Matmul panel read was outside QKV layout");
                for (integer first_lane = 0; first_lane < 8;
                     first_lane++) begin
                    for (integer second_lane = first_lane + 1;
                         second_lane < 8; second_lane++) begin
                        if (first_lane < matmul_read_req.activation_word_count &&
                            second_lane < matmul_read_req.activation_word_count)
                            assert (matmul_activation_read_port[first_lane] !=
                                    matmul_activation_read_port[second_lane])
                                else $error(
                                    "local_memory Matmul activation lanes selected one endpoint lanes=%0d/%0d ports=%0d/%0d word_indices=%0d/%0d",
                                    first_lane, second_lane,
                                    matmul_activation_read_port[first_lane],
                                    matmul_activation_read_port[second_lane],
                                    matmul_read_req.activation_lane_word_index[
                                        first_lane*16 +: 16],
                                    matmul_read_req.activation_lane_word_index[
                                        second_lane*16 +: 16]);
                    end
                    for (integer panel_lane = 0; panel_lane < 12;
                         panel_lane++) begin
                        if (first_lane < matmul_read_req.activation_word_count &&
                            matmul_read_req.panel_port_mask[panel_lane])
                            assert (matmul_activation_read_port[first_lane] !=
                                    matmul_panel_read_port[panel_lane])
                                else $error("local_memory Matmul activation and panel lanes selected one endpoint");
                    end
                end
                for (integer first_panel_lane = 0;
                     first_panel_lane < 12; first_panel_lane++) begin
                    for (integer second_panel_lane = first_panel_lane + 1;
                         second_panel_lane < 12; second_panel_lane++) begin
                        if (matmul_read_req.panel_port_mask[first_panel_lane] &&
                            matmul_read_req.panel_port_mask[second_panel_lane])
                            assert (matmul_panel_read_port[first_panel_lane] !=
                                    matmul_panel_read_port[second_panel_lane])
                                else $error("local_memory Matmul panel lanes selected one endpoint");
                    end
                end
            end
            if (qkv_tile_read_pending[1])
                assert (qkv_head_tile_read_rsp_valid)
                    else $error("local_memory QKV tile read response was incomplete");
            if (qkv_head_tile_read_req_valid) begin
                assert (qkv_head_tile_read_req.row_base[2:0] == 3'd0)
                    else $error("local_memory QKV tile read row base was not M8-aligned");
                if (qkv_matmul_overlap_active &&
                    qkv_head_tile_read_req_ready)
                    assert (!qkv_tile_matmul_conflict)
                        else $error("local_memory QKV tile read bypassed Matmul port priority");
            end
            if (attention_operand_req_valid) begin
                assert (attention_operand_req.query_base[2:0] == 3'd0)
                    else $error("local_memory Attention operand query base was not M8-aligned");
                assert (attention_operand_req.query_base <= 6'd40)
                    else $error("local_memory Attention operand query bundle exceeded 48 rows");
                assert (attention_operand_req.output_base[2:0] == 3'd0)
                    else $error("local_memory Attention operand output base was not M8-aligned");
                assert (attention_operand_req.k_base[2:0] == 3'd0)
                    else $error("local_memory Attention operand K base was not M8-aligned");
            end
            if (attention_operand_read_pending[1])
                assert (attention_operand_rsp_valid)
                    else $error("local_memory Attention operand response incomplete is_pv=%0b extra=%0b panel=%0b request=%013h macro_read=%013h port_read_valid=%013h q_slots=%013h panels=%013h metadata_port=%0d panel_metadata_port=%0d",
                        attention_operand_pending_is_pv[1],
                        attention_operand_pending_extra[1],
                        attention_operand_pending_panel[1],
                        debug_attention_operand_request_mask[1],
                        debug_attention_operand_macro_read_mask[1],
                        port_read_valid,
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT0_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT1_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT2_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT3_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT4_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT5_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT6_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            Q_HEAD_SLOT7_PORT),
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            ATTENTION_PANEL0_PRIMARY_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            ATTENTION_PANEL1_PRIMARY_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            ATTENTION_PANEL2_PRIMARY_PORT) |
                        (LOCAL_SRAM_PHYSICAL_PORT_COUNT'(1) <<
                            ATTENTION_PANEL3_PRIMARY_PORT),
                        ATTENTION_METADATA_PORT,
                        attention_operand_pending_panel[1] ?
                            ATTENTION_PANEL_METADATA1_PORT :
                            ATTENTION_PANEL_METADATA0_PORT);
            if (attention_score_write_valid)
                assert (attention_score_write.row_base[2:0] == 3'd0)
                    else $error("local_memory Attention score write row base was not M8-aligned");
            if (attention_score_write_valid)
                assert (attention_score_write.row_base <= 6'd40)
                    else $error("local_memory Attention score write bundle exceeded 48 rows");
            if (attention_score_read_req_valid) begin
                assert (attention_score_read_req.row_base[2:0] == 3'd0)
                    else $error("local_memory Attention score read row base was not M8-aligned");
                assert (attention_score_read_req.row_base <= 6'd40)
                    else $error("local_memory Attention score read bundle exceeded 48 rows");
            end
            if (attention_score_read_pending[1])
                assert (attention_score_read_rsp_valid)
                    else $error("local_memory Attention score response was incomplete pending_mask=0x%02h extra=%0b port_read_valid=0x%013h score_req_valid=0x%013h probability_req_valid=0x%013h macro_write=0x%013h",
                        attention_score_read_pending_row_mask[1],
                        attention_score_read_pending_extra[1],
                        port_read_valid, attention_score_req_valid,
                        attention_probability_req_valid, macro_write);
            assert ($onehot0({attention_probability_write_valid,
                attention_probability_quantized_write_valid,
                attention_probability_scale_write_valid}))
                else $error("local_memory Attention probability write channels were not mutually exclusive");
            if (attention_probability_write_valid)
                assert (attention_probability_write.row_base[2:0] == 3'd0)
                    else $error("local_memory Attention probability row base was not M8-aligned");
            if (attention_probability_quantized_write_valid)
                assert (attention_probability_quantized_write.row_base[2:0] == 3'd0)
                    else $error("local_memory Attention probability-quantized row base was not M8-aligned");
            if (attention_probability_scale_write_valid)
                assert (attention_probability_scale_write.row_base[2:0] == 3'd0)
                    else $error("local_memory Attention probability-scale row base was not M8-aligned");
            assert ($onehot0(
                {qkv_constant_write_valid, qkv_q_write_valid}))
                else $error("local_memory QKV constant and Q write channels were both valid");
            assert ($onehot0({attention_panel_write_valid,
                attention_panel_scale_write_valid}))
                else $error("local_memory Attention panel data and scale write channels were both valid");
            if (qkv_q_write_valid) begin
                assert (qkv_q_write.physical_row < 6'd48)
                    else $error("local_memory Q write physical row exceeded 47");
                if (!qkv_q_write.scale)
                    assert (qkv_q_write.word < 10'd256)
                        else $error("local_memory Q row word exceeded 255");
            end
            assert (!(hidden_write_valid && hidden_read_valid))
                else $error("local_memory hidden bundle issued read and write together");
            if (hidden_read_pending[1])
                assert (hidden_read_response_valid)
                    else $error("local_memory hidden read response was missing");
            if (context_read_pending[1])
                assert (context_return_valid)
                    else $error("local_memory context read response was missing");
            if (attention_scratch_read_pending[1]) begin
                assert (attention_scratch_read_pending_bank[1] ?
                    (port_read_valid[ATTENTION_SCRATCH1_PRIMARY_PORT] &&
                     port_read_valid[ATTENTION_SCRATCH1_SECONDARY_PORT]) :
                    (port_read_valid[ATTENTION_SCRATCH0_PRIMARY_PORT] &&
                     port_read_valid[ATTENTION_SCRATCH0_SECONDARY_PORT]))
                    else $error("local_memory attention scratch read response was missing");
                assert (attention_scratch_fifo_input_ready)
                    else $error("local_memory attention scratch response FIFO overflow");
            end
            assert (attention_scratch_reserved_count <= 4'd3)
                else $error("local_memory attention scratch reservation exceeded three responses");
            if (qkv_head_read_pending[1] &&
                (port_read_valid[QKV_HEAD_SOURCE_PORT] ||
                 port_read_valid[QKV_HEAD_DESTINATION_PORT]))
                assert (qkv_head_read_return_valid)
                    else $error("local_memory QKV head read response used the wrong endpoint");
            if (qkv_head_read_pending[1])
                assert (qkv_head_read_return_valid)
                    else $error("local_memory QKV head read response was missing");
            assert (qkv_head_read_reserved_count <= 4'd3)
                else $error("local_memory QKV head read reservation exceeded three responses");
            assert (!qkv_head_read_return_valid || qkv_head_read_fifo_input_ready)
                else $error("local_memory QKV head read response FIFO overflow");
            if (rope_source_read_pending &&
                (port_read_valid[ROPE_SOURCE_PRIMARY_PORT] ||
                 port_read_valid[ROPE_SOURCE_SECONDARY_PORT]))
                assert (rope_source_read_rsp_valid)
                    else $error("local_memory RoPE source response was incomplete");
            if (attention_scratch_read_valid &&
                attention_scratch_write_valid)
                assert (attention_scratch_read_req.bank !=
                    attention_scratch_write_req.bank)
                    else $error("local_memory attention scratch read and write selected one region");
            assert (!(|macro_same_address_conflict))
                else $error("local_memory macro ports accessed one row while writing");
            assert (!rms_tile_fifo_input_valid || rms_tile_fifo_input_ready)
                else $error("local_memory RMSNorm tile response FIFO overflow");
            assert (!rms_scratch_fifo_input_valid || rms_scratch_fifo_input_ready)
                else $error("local_memory RMSNorm scratch response FIFO overflow");
            if (rms_tile_read_req_valid)
                assert (rms_tile_read_req.row_base[2:0] == 3'd0)
                    else $error("local_memory RMSNorm tile read row base was not M8-aligned");
            if (rms_norm_write_valid)
                assert (rms_norm_write.row_base[2:0] == 3'd0)
                    else $error("local_memory RMSNorm tile write row base was not M8-aligned");
            if (rms_scratch_read_valid && rms_scratch_write_valid)
                assert (rms_scratch_read_req.bank != rms_scratch_write.bank)
                    else $error("local_memory RMSNorm scratch read and write selected one region");
        end
        if (!rst && layout_update_valid && layout_update_ready) begin
            assert (attention_pair_role_drained)
                else $error("local_memory changed projection/Attention batch mapping before responses drained");
            if (next_layout.kv_pair_enable)
                assert (!next_layout.attention_pair_enable &&
                    next_layout.phase == LOCAL_MEMORY_PHASE_QKV_PREPARATION &&
                    next_layout.kv_first_batch_tokens != 0 &&
                    next_layout.kv_first_batch_tokens <= 6'd48)
                    else $error("local_memory KV pair requires QKV phase and 1..48 first-batch tokens");
            if (next_layout.attention_pair_enable)
                assert (next_layout.attention_score_words >= 8'd32 &&
                    next_layout.attention_score_words <= 8'd162 &&
                    !next_layout.attention_score_words[0])
                    else $error("local_memory Attention pair score stride is out of range or odd");
            assert (phase_role_drained)
                else $error("local_memory changed phase %0d to %0d before the current role drained",
                    current_layout.phase, next_layout.phase);
            assert (qkv_staging_role_drained)
                else $error("local_memory changed QKV head staging %0d to %0d before its role drained",
                    current_layout.qkv_head_staging, next_layout.qkv_head_staging);
            assert (panel_role_drained)
                else $error("local_memory changed panel mapping %0d to %0d before its role drained",
                    current_layout.panel_layout, next_layout.panel_layout);
        end
    end
`endif

    logic unused_rms_active;
    assign unused_rms_active = rms_active;
endmodule

`default_nettype wire
