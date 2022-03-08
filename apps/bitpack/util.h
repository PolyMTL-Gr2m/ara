#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define for_endian(size) for (int i = 0; i < size; ++i)
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define for_endian(size) for (int i = size - 1; i >= 0; --i)
#else
#error "Endianness not detected"
#endif

#define printb(value)                                   \
({                                                      \
        typeof(value) _v = value;                       \
        __printb((typeof(_v) *) &_v, sizeof(_v));       \
})

#define MSB_MASK 1 << (CHAR_BIT - 1)

#define PRINT_INT 0
#define PRINT_BIN 1


void  __printb(void *value, size_t size);
void print_matrix(uint64_t *mat, int num_rows, int num_columns, int bin_print_format);
const uint64_t vshift[64] = {  0,  1,  2,  3,  4,  5,  6,  7, 
                               8,  9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 26, 27, 28, 29, 30, 31,
                              32, 33, 34, 35, 36, 37, 38, 39,
                              40, 41, 42, 43, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55,
                              56, 57, 58, 59, 60, 61, 62, 63};
const uint64_t vmask[64] = { 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1};
