#include <stdint.h>

void NCHW_to_NHWC_tensor8_vec_32xC(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W);
void NCHW_to_NHWC_tensor8(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W);

void NCHW_to_NHWC_tensor16_vec_32xC(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W);
void NCHW_to_NHWC_tensor16(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W);

void NCHW_to_NHWC_tensor32_vec_16xC(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W);
void NCHW_to_NHWC_tensor32(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W);

void NCHW_to_NHWC_tensor64_vec_8xC(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W);
void NCHW_to_NHWC_tensor64(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W);



