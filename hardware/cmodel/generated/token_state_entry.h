#ifndef TOKEN_STATE_ENTRY_H
#define TOKEN_STATE_ENTRY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define TOKEN_STATE_ENTRY_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define TOKEN_STATE_ENTRY_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define TOKEN_STATE_ENTRY_BYTES 32u
#define TOKEN_STATE_ENTRY_ALIGNMENT 16u
#define TOKEN_STATE_ENTRY_TOKEN_POSITION_OFFSET 0u
#define TOKEN_STATE_ENTRY_BLOCK_LOCAL_POSITION_OFFSET 2u
#define TOKEN_STATE_ENTRY_BLOCK_SLOT_OFFSET 3u
#define TOKEN_STATE_ENTRY_GLOBAL_BLOCK_ID_OFFSET 4u
#define TOKEN_STATE_ENTRY_STATE_OFFSET 5u
#define TOKEN_STATE_ENTRY_ORIGIN_OFFSET 6u
#define TOKEN_STATE_ENTRY_ACTIVATION_BITS_OFFSET 7u
#define TOKEN_STATE_ENTRY_TOKEN_ID_OFFSET 8u
#define TOKEN_STATE_ENTRY_LAST_TOP1_OFFSET 12u
#define TOKEN_STATE_ENTRY_PRECISION_AGE_OFFSET 16u
#define TOKEN_STATE_ENTRY_CACHE_VALID_OFFSET 18u
#define TOKEN_STATE_ENTRY_REFRESH_REQUIRED_OFFSET 19u
#define TOKEN_STATE_ENTRY_CAPTURE_INDEX_OFFSET 20u
#define TOKEN_STATE_ENTRY_PREDICTION_FLAG_OFFSET 24u
#define TOKEN_STATE_ENTRY_CHANGE_FLAGS_OFFSET 25u
#define TOKEN_STATE_ENTRY_CHANGE_CONFIDENCE_BF16_OFFSET 26u
#define TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_BF16_OFFSET 28u
#define TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_VALID_OFFSET 30u
#define TOKEN_STATE_ENTRY_SOURCE_A_PENDING_OFFSET 31u

struct token_state_entry {
    uint16_t token_position;
    uint8_t block_local_position;
    uint8_t block_slot;
    uint8_t global_block_id;
    uint8_t state;
    uint8_t origin;
    uint8_t activation_bits;
    uint32_t token_id;
    uint32_t last_top1;
    int16_t precision_age;
    uint8_t cache_valid;
    uint8_t refresh_required;
    uint32_t capture_index;
    uint8_t prediction_flag;
    uint8_t change_flags;
    uint16_t change_confidence_bf16;
    uint16_t last_action_confidence_bf16;
    uint8_t last_action_confidence_valid;
    uint8_t source_a_pending;
};

TOKEN_STATE_ENTRY_STATIC_ASSERT(sizeof(struct token_state_entry) == TOKEN_STATE_ENTRY_BYTES,
               "token_state_entry size mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, token_position) ==
               TOKEN_STATE_ENTRY_TOKEN_POSITION_OFFSET,
               "token_position offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, block_local_position) ==
               TOKEN_STATE_ENTRY_BLOCK_LOCAL_POSITION_OFFSET,
               "block_local_position offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, block_slot) ==
               TOKEN_STATE_ENTRY_BLOCK_SLOT_OFFSET,
               "block_slot offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, global_block_id) ==
               TOKEN_STATE_ENTRY_GLOBAL_BLOCK_ID_OFFSET,
               "global_block_id offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, state) ==
               TOKEN_STATE_ENTRY_STATE_OFFSET,
               "state offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, origin) ==
               TOKEN_STATE_ENTRY_ORIGIN_OFFSET,
               "origin offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, activation_bits) ==
               TOKEN_STATE_ENTRY_ACTIVATION_BITS_OFFSET,
               "activation_bits offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, token_id) ==
               TOKEN_STATE_ENTRY_TOKEN_ID_OFFSET,
               "token_id offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, last_top1) ==
               TOKEN_STATE_ENTRY_LAST_TOP1_OFFSET,
               "last_top1 offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, precision_age) ==
               TOKEN_STATE_ENTRY_PRECISION_AGE_OFFSET,
               "precision_age offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, cache_valid) ==
               TOKEN_STATE_ENTRY_CACHE_VALID_OFFSET,
               "cache_valid offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, refresh_required) ==
               TOKEN_STATE_ENTRY_REFRESH_REQUIRED_OFFSET,
               "refresh_required offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, capture_index) ==
               TOKEN_STATE_ENTRY_CAPTURE_INDEX_OFFSET,
               "capture_index offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, prediction_flag) ==
               TOKEN_STATE_ENTRY_PREDICTION_FLAG_OFFSET,
               "prediction_flag offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, change_flags) ==
               TOKEN_STATE_ENTRY_CHANGE_FLAGS_OFFSET,
               "change_flags offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, change_confidence_bf16) ==
               TOKEN_STATE_ENTRY_CHANGE_CONFIDENCE_BF16_OFFSET,
               "change_confidence_bf16 offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, last_action_confidence_bf16) ==
               TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_BF16_OFFSET,
               "last_action_confidence_bf16 offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, last_action_confidence_valid) ==
               TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_VALID_OFFSET,
               "last_action_confidence_valid offset mismatch");
TOKEN_STATE_ENTRY_STATIC_ASSERT(offsetof(struct token_state_entry, source_a_pending) ==
               TOKEN_STATE_ENTRY_SOURCE_A_PENDING_OFFSET,
               "source_a_pending offset mismatch");

#undef TOKEN_STATE_ENTRY_STATIC_ASSERT

#endif
