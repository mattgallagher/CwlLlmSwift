#ifndef LLMAMXExports_h
#define LLMAMXExports_h

#include <stddef.h>
#include <stdint.h>

void amx_set(void);
void amx_clr(void);

void amx_ldx(uint64_t operand);
void amx_ldy(uint64_t operand);
void amx_ldz(uint64_t operand);
void amx_stz(uint64_t operand);
void amx_matfp(uint64_t operand);

void amx_pack_lhs_f32(
    const float *src,
    size_t row_start,
    size_t rows_in_block,
    size_t inner_count,
    size_t row_stride,
    size_t col_stride,
    float *dst
);

void amx_pack_rhs_f32(
    const float *src,
    size_t column_start,
    size_t columns_in_block,
    size_t inner_count,
    size_t row_stride,
    size_t col_stride,
    float *dst
);

void amx_scatter_output_f32_simd(
    const float *out_tiles,
    float *out_base,
    size_t out_row_stride,
    size_t row_start,
    size_t rows_in_block,
    size_t column_start,
    size_t columns_in_block,
    uint8_t accumulate
);

#endif
