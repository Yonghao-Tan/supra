#ifndef ATTENTION_DEPENDENCY_JOB_H
#define ATTENTION_DEPENDENCY_JOB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define ATTENTION_DEPENDENCY_JOB_BYTES 64u
#define ATTENTION_DEPENDENCY_JOB_ALIGNMENT 16u
#define ATTENTION_DEPENDENCY_JOB_SOURCE_BASE_OFFSET 0u
#define ATTENTION_DEPENDENCY_JOB_HEAD_STRIDE_OFFSET 8u
#define ATTENTION_DEPENDENCY_JOB_LANE_MASK_OFFSET 12u
#define ATTENTION_DEPENDENCY_JOB_OPERATION_OFFSET 13u
#define ATTENTION_DEPENDENCY_JOB_RESERVED0_OFFSET 14u
#define ATTENTION_DEPENDENCY_JOB_OUTPUT_BASE_OFFSET 16u
#define ATTENTION_DEPENDENCY_JOB_SOURCE_LIMIT_OFFSET 24u
#define ATTENTION_DEPENDENCY_JOB_OUTPUT_LIMIT_OFFSET 32u
#define ATTENTION_DEPENDENCY_JOB_RESERVED1_OFFSET 40u
#define ATTENTION_DEPENDENCY_JOB_RESERVED2_OFFSET 48u
#define ATTENTION_DEPENDENCY_JOB_RESERVED3_OFFSET 56u

struct attention_dependency_job {
    uint64_t source_base;
    uint32_t head_stride;
    uint8_t lane_mask;
    uint8_t operation;
    uint16_t reserved0;
    uint64_t output_base;
    uint64_t source_limit;
    uint64_t output_limit;
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
};

ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(sizeof(struct attention_dependency_job) == ATTENTION_DEPENDENCY_JOB_BYTES,
               "attention_dependency_job size mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, source_base) ==
               ATTENTION_DEPENDENCY_JOB_SOURCE_BASE_OFFSET,
               "source_base offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, head_stride) ==
               ATTENTION_DEPENDENCY_JOB_HEAD_STRIDE_OFFSET,
               "head_stride offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, lane_mask) ==
               ATTENTION_DEPENDENCY_JOB_LANE_MASK_OFFSET,
               "lane_mask offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, operation) ==
               ATTENTION_DEPENDENCY_JOB_OPERATION_OFFSET,
               "operation offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, reserved0) ==
               ATTENTION_DEPENDENCY_JOB_RESERVED0_OFFSET,
               "reserved0 offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, output_base) ==
               ATTENTION_DEPENDENCY_JOB_OUTPUT_BASE_OFFSET,
               "output_base offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, source_limit) ==
               ATTENTION_DEPENDENCY_JOB_SOURCE_LIMIT_OFFSET,
               "source_limit offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, output_limit) ==
               ATTENTION_DEPENDENCY_JOB_OUTPUT_LIMIT_OFFSET,
               "output_limit offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, reserved1) ==
               ATTENTION_DEPENDENCY_JOB_RESERVED1_OFFSET,
               "reserved1 offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, reserved2) ==
               ATTENTION_DEPENDENCY_JOB_RESERVED2_OFFSET,
               "reserved2 offset mismatch");
ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT(offsetof(struct attention_dependency_job, reserved3) ==
               ATTENTION_DEPENDENCY_JOB_RESERVED3_OFFSET,
               "reserved3 offset mismatch");

#undef ATTENTION_DEPENDENCY_JOB_STATIC_ASSERT

#endif
