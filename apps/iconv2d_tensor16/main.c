// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti
//	- Modified by Théo Dupuis - Polytechnique Montréal (2022)

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iconv2d_tensor16.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd

// Matrices defined in data.S
extern int16_t i[] __attribute__((aligned(4 * NR_LANES))); 			// [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int16_t f[] __attribute__((aligned(4 * NR_LANES)));        // [ F*F ]
extern int16_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
extern int16_t golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth
extern int64_t F; // filter
extern int64_t K; // number of kernel to convolve

// Verify the matrices to implement on tensors
int verify_64btensor(int64_t *tensor1, int64_t *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if (tensor1[c + C * (r + d * R)] != tensor2[c + C * (r + d * R)]) {
  	      printf("Error: o[%d][%d][%d] = %ld, instead of %ld\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      return 1;
  	   }
return 0;
  		
}


// Verify the matrices to implement on tensors
int verify_16btensor(int16_t *tensor1, int16_t *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if (tensor1[c + C * (r + d * R)] != tensor2[c + C * (r + d * R)]) {
  	      printf("Error: o[%d][%d][%d] = %ld, instead of %ld\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      return 1;
  	   }
return 0;
  		
}


void print_64btensor(int64_t const *tensor, uint64_t num_rows,
                  uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10d ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}


void print_16btensor(uint16_t const *tensor, uint64_t num_rows,
                  uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10d ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}


int main() {
  printf("\n");
  printf("================\n");
  printf("=  CONV2D i16  =\n");
  printf("================\n");
  printf("\n");
  printf("\n");
  

  // Call the main kernel, and measure cycles
  start_timer();
  //if (F == 3)
    iconv2d_tensor16_3x3(o, i, f, M, N, L, F, K);
  /*else if (F == 5)
    conv2d_5x5(o, i, f, M, N, F);
  else if (F == 7)
    conv2d_7x7(o, i, f, M, N, F);
  else
    printf("Error: the filter size is different from 3 or 5 or 7.\n");
  */stop_timer();
  
  // Performance metrics
  int64_t runtime = get_timer();
  printf("The execution took %d cycles.\n", runtime);
    
   // Verify correctness
  printf("Verifying result...\n");
  
  int error = verify_16btensor(o, golden_o, M, N, K);
  if (error != 0) {
    printf("Fail.\n");
  } else {
    printf("Passed.\n");
  } 

  

  return 0; //error;
}
