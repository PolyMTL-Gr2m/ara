#include "util.h"
#include "printf.h"

// Assumes little endian
void __printb(void *value, size_t size)
{
        unsigned char uc;
        unsigned char bits[CHAR_BIT + 1];

        bits[CHAR_BIT] = '\0';
        for_endian(size) {
                uc = ((unsigned char *) value)[i];
                memset(bits, '0', CHAR_BIT);
                for (int j = 0; uc && j < CHAR_BIT; ++j) {
                        if (uc & MSB_MASK)
                                bits[j] = '1';
                        uc <<= 1;
                }
                printf("%s", bits);
        }
        printf("\n");
}

void print_matrix(uint64_t *mat, int num_rows, int num_columns, int bin_print_format)
{
    for (int i=0; i<num_rows; i++)
    {
        for(int j=0; j<num_columns; j++)
        {
            if (bin_print_format==PRINT_BIN) {
                printb(mat[i * num_columns + j]);
            }else{
                printf("%llu ", mat[i * num_columns + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
