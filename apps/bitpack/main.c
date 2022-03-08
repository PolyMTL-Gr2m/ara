#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <riscv_vector.h>

#include "util.h"
#include "printf.h"
#include "runtime.h"

#define RISC_V_ARA 

void init_matrix(uint64_t* matrix, int num_rows, int num_columns, uint64_t MAX_VAL) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_columns; ++j) {
            matrix[i * num_columns + j] = (rand() % (MAX_VAL - 0 + 1)) + 0;
        }
    }
}

void transpose_matrix(uint64_t* matrix, int num_rows, int num_columns){
    uint64_t temp;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = i; j < num_columns; ++j) {
            temp = matrix[i * num_columns + j];
            matrix[i * num_columns + j] = matrix[j * num_columns + i];
            matrix[j * num_columns + i] = temp;
        }
    }
}

// Naive implementation of bit packing
void bitpack_naive(uint64_t* matrix, uint64_t* packed_data, int DATA_WIDTH, int dlen, int bitprec){
    // Packed data pointer
    uint64_t p_ptr = 0;
    for (int i=0; i<dlen; i+=DATA_WIDTH){
        for (int el=0; el<DATA_WIDTH; el++){
            for (int bit_pos=0; bit_pos<bitprec; bit_pos++){
                int bit_idx = p_ptr+bit_pos;
                uint64_t data = (matrix[i+el] >> bit_pos) & 0x1;
                packed_data[bit_idx] <<= 1;
                packed_data[bit_idx] = (packed_data[bit_idx] | data);
            }
        }
        p_ptr += bitprec;
    }
}

void bitpack_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
}

// Vectorized implementation of bit packing
void bitpack_vectorized(uint64_t* matrix, uint64_t* packed_data, int DATA_WIDTH, int dlen, int bitprec){
    // Initialize vector RF
    bitpack_init();
    size_t vl=0;
    uint64_t *p_ptr = matrix;
    // Load shift values into v0
    asm volatile("vle64.v v0, (%[A])" : : [A] "r" (vshift));
    // Load mask values into v1
    asm volatile("vle64.v v1, (%[A])" : : [A] "r" (vmask));
    for (size_t c_n_count=dlen; c_n_count; c_n_count -= vl){
        vl = vsetvl_e64m1(c_n_count);
        asm volatile("vle64.v v2, (%[A])" : : [A] "r" (matrix));
        asm volatile("vand.vv v3, v2, v1" ::);
        for (int bit_pos=0; bit_pos<bitprec; bit_pos++){

            int bit_idx = p_ptr+bit_pos;
            uint64_t data = (matrix[i+el] >> bit_pos) & 0x1;
            packed_data[bit_idx] <<= 1;
            packed_data[bit_idx] = (packed_data[bit_idx] | data);
        }
        p_ptr += bitprec;
    }
}

int main(){
    printf("bitpack init!\n");
    const uint64_t BITPREC=3;
    const uint64_t MAT_SIZE_W = 8;
    const uint64_t MAT_SIZE_H = 8;
    const uint64_t DATA_WIDTH = 8;
    assert((MAT_SIZE_W%DATA_WIDTH)==0 && "Matrix size must be multiple of data type");
    const uint64_t PACKED_MAT_SIZE = (MAT_SIZE_H*MAT_SIZE_W)/(DATA_WIDTH) * BITPREC;
    uint64_t tensor[MAT_SIZE_H*MAT_SIZE_W];
    uint64_t packed_data[PACKED_MAT_SIZE];
    uint64_t max_val = (1<<BITPREC) - 1;
    srand(0);

    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Bitpacking of a tensor [%dx%dx%dx%d] with %d precision\n", 1, 1, MAT_SIZE_H, MAT_SIZE_W, BITPREC);
    printf("------------------------------------------------------------\n");
    printf("\n");
    init_matrix(tensor, MAT_SIZE_H, MAT_SIZE_W, max_val);
    init_matrix(packed_data, 1, PACKED_MAT_SIZE, 0);
    // // transpose_matrix(tensor, MAT_SIZE_W, MAT_SIZE_H);

    // print_matrix(tensor, MAT_SIZE_H, MAT_SIZE_W, PRINT_INT);
    start_timer();
    bitpack_naive(tensor, packed_data, DATA_WIDTH, MAT_SIZE_H*MAT_SIZE_W, BITPREC);
    stop_timer();
    // Metrics
    int64_t runtime = get_timer();
    printf("bitpack_naive took %d cycles.\n", runtime);

    // print_matrix(packed_data, 1, PACKED_MAT_SIZE, PRINT_BIN);
    return 0;
}
