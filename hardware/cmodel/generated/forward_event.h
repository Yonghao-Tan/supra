#ifndef FORWARD_EVENT_H
#define FORWARD_EVENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define FORWARD_EVENT_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define FORWARD_EVENT_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define FORWARD_EVENT_BYTES 64u
#define FORWARD_EVENT_ALIGNMENT 16u
#define FORWARD_EVENT_CAPTURE_INDEX_OFFSET 0u
#define FORWARD_EVENT_BLOCK_SLOT_OFFSET 4u
#define FORWARD_EVENT_GLOBAL_BLOCK_ID_OFFSET 5u
#define FORWARD_EVENT_POSITION_COUNT_OFFSET 6u
#define FORWARD_EVENT_CONFIRMED_MASK_OFFSET 8u
#define FORWARD_EVENT_REMASKED_MASK_OFFSET 12u
#define FORWARD_EVENT_SELECTED_MASK_OFFSET 16u
#define FORWARD_EVENT_DIRECT_LOCKED_MASK_OFFSET 20u
#define FORWARD_EVENT_STABLE_TENTATIVE_MASK_OFFSET 24u
#define FORWARD_EVENT_FALLBACK_TENTATIVE_MASK_OFFSET 28u
#define FORWARD_EVENT_MANDATORY_REFRESH_MASK_OFFSET 32u
#define FORWARD_EVENT_CACHE_COMMIT_MASK_OFFSET 36u
#define FORWARD_EVENT_CACHE_INVALIDATE_MASK_OFFSET 40u
#define FORWARD_EVENT_CACHE_KEEP_MASK_OFFSET 44u
#define FORWARD_EVENT_TOKEN_CHANGED_MASK_OFFSET 48u
#define FORWARD_EVENT_TAIL_CLOSED_MASK_OFFSET 52u
#define FORWARD_EVENT_RESERVED0_OFFSET 56u
#define FORWARD_EVENT_RESERVED1_OFFSET 60u

struct forward_event {
    uint32_t capture_index;
    uint8_t block_slot;
    uint8_t global_block_id;
    uint16_t position_count;
    uint32_t confirmed_mask;
    uint32_t remasked_mask;
    uint32_t selected_mask;
    uint32_t direct_locked_mask;
    uint32_t stable_tentative_mask;
    uint32_t fallback_tentative_mask;
    uint32_t mandatory_refresh_mask;
    uint32_t cache_commit_mask;
    uint32_t cache_invalidate_mask;
    uint32_t cache_keep_mask;
    uint32_t token_changed_mask;
    uint32_t tail_closed_mask;
    uint32_t reserved0;
    uint32_t reserved1;
};

FORWARD_EVENT_STATIC_ASSERT(sizeof(struct forward_event) == FORWARD_EVENT_BYTES,
               "forward_event size mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, capture_index) ==
               FORWARD_EVENT_CAPTURE_INDEX_OFFSET,
               "capture_index offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, block_slot) ==
               FORWARD_EVENT_BLOCK_SLOT_OFFSET,
               "block_slot offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, global_block_id) ==
               FORWARD_EVENT_GLOBAL_BLOCK_ID_OFFSET,
               "global_block_id offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, position_count) ==
               FORWARD_EVENT_POSITION_COUNT_OFFSET,
               "position_count offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, confirmed_mask) ==
               FORWARD_EVENT_CONFIRMED_MASK_OFFSET,
               "confirmed_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, remasked_mask) ==
               FORWARD_EVENT_REMASKED_MASK_OFFSET,
               "remasked_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, selected_mask) ==
               FORWARD_EVENT_SELECTED_MASK_OFFSET,
               "selected_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, direct_locked_mask) ==
               FORWARD_EVENT_DIRECT_LOCKED_MASK_OFFSET,
               "direct_locked_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, stable_tentative_mask) ==
               FORWARD_EVENT_STABLE_TENTATIVE_MASK_OFFSET,
               "stable_tentative_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, fallback_tentative_mask) ==
               FORWARD_EVENT_FALLBACK_TENTATIVE_MASK_OFFSET,
               "fallback_tentative_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, mandatory_refresh_mask) ==
               FORWARD_EVENT_MANDATORY_REFRESH_MASK_OFFSET,
               "mandatory_refresh_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, cache_commit_mask) ==
               FORWARD_EVENT_CACHE_COMMIT_MASK_OFFSET,
               "cache_commit_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, cache_invalidate_mask) ==
               FORWARD_EVENT_CACHE_INVALIDATE_MASK_OFFSET,
               "cache_invalidate_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, cache_keep_mask) ==
               FORWARD_EVENT_CACHE_KEEP_MASK_OFFSET,
               "cache_keep_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, token_changed_mask) ==
               FORWARD_EVENT_TOKEN_CHANGED_MASK_OFFSET,
               "token_changed_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, tail_closed_mask) ==
               FORWARD_EVENT_TAIL_CLOSED_MASK_OFFSET,
               "tail_closed_mask offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, reserved0) ==
               FORWARD_EVENT_RESERVED0_OFFSET,
               "reserved0 offset mismatch");
FORWARD_EVENT_STATIC_ASSERT(offsetof(struct forward_event, reserved1) ==
               FORWARD_EVENT_RESERVED1_OFFSET,
               "reserved1 offset mismatch");

#undef FORWARD_EVENT_STATIC_ASSERT

#endif
