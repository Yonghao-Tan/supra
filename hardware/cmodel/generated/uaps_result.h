#ifndef UAPS_RESULT_H
#define UAPS_RESULT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define UAPS_RESULT_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define UAPS_RESULT_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define UAPS_RESULT_BYTES 16u
#define UAPS_RESULT_ALIGNMENT 16u
#define UAPS_RESULT_FUTURE_PREDICTION_MASK_OFFSET 0u
#define UAPS_RESULT_ADDED_FUTURE_TOKEN_MASK_OFFSET 4u
#define UAPS_RESULT_FUTURE_PREDICTION_COUNT_OFFSET 8u
#define UAPS_RESULT_ADDED_FUTURE_TOKEN_COUNT_OFFSET 9u
#define UAPS_RESULT_BASE_ACTIVATION_SLOTS_OFFSET 10u
#define UAPS_RESULT_JOINT_ACTIVATION_SLOTS_OFFSET 11u
#define UAPS_RESULT_NEXT_PASS_ACTIVATION_SLOTS_OFFSET 12u
#define UAPS_RESULT_RESERVED0_OFFSET 13u
#define UAPS_RESULT_RESERVED1_OFFSET 14u

struct uaps_result {
    uint32_t future_prediction_mask;
    uint32_t added_future_token_mask;
    uint8_t future_prediction_count;
    uint8_t added_future_token_count;
    uint8_t base_activation_slots;
    uint8_t joint_activation_slots;
    uint8_t next_pass_activation_slots;
    uint8_t reserved0;
    uint16_t reserved1;
};

UAPS_RESULT_STATIC_ASSERT(sizeof(struct uaps_result) == UAPS_RESULT_BYTES,
               "uaps_result size mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, future_prediction_mask) ==
               UAPS_RESULT_FUTURE_PREDICTION_MASK_OFFSET,
               "future_prediction_mask offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, added_future_token_mask) ==
               UAPS_RESULT_ADDED_FUTURE_TOKEN_MASK_OFFSET,
               "added_future_token_mask offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, future_prediction_count) ==
               UAPS_RESULT_FUTURE_PREDICTION_COUNT_OFFSET,
               "future_prediction_count offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, added_future_token_count) ==
               UAPS_RESULT_ADDED_FUTURE_TOKEN_COUNT_OFFSET,
               "added_future_token_count offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, base_activation_slots) ==
               UAPS_RESULT_BASE_ACTIVATION_SLOTS_OFFSET,
               "base_activation_slots offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, joint_activation_slots) ==
               UAPS_RESULT_JOINT_ACTIVATION_SLOTS_OFFSET,
               "joint_activation_slots offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, next_pass_activation_slots) ==
               UAPS_RESULT_NEXT_PASS_ACTIVATION_SLOTS_OFFSET,
               "next_pass_activation_slots offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, reserved0) ==
               UAPS_RESULT_RESERVED0_OFFSET,
               "reserved0 offset mismatch");
UAPS_RESULT_STATIC_ASSERT(offsetof(struct uaps_result, reserved1) ==
               UAPS_RESULT_RESERVED1_OFFSET,
               "reserved1 offset mismatch");

#undef UAPS_RESULT_STATIC_ASSERT

#endif
