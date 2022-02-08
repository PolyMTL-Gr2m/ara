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
