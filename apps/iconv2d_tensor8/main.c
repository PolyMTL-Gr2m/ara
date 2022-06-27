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

#include "iconv2d_tensor8.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

//#define NAIVE
#define NHWC
//#define WIDE

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd
	// Matrices defined in data.S
	extern int8_t i[] __attribute__((
		 aligned(4 * NR_LANES))); // [ (M+floor(F/2)) * (N+floor(F/2)) ]
	extern int8_t f[] __attribute__((aligned(4 * NR_LANES)));        // [ F*F ]

	#ifdef WIDE
		extern int32_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
		extern int32_t golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
	#else
		extern int8_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
		extern int8_t golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
	#endif

// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth
extern int64_t F; // filter
extern int64_t K; // number of kernel to convolve


int verify_8btensor(int8_t *tensor1, int8_t *tensor2, int64_t R,
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

int verify_32btensor(int32_t *tensor1, int32_t *tensor2, int64_t R,
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


void print_8btensor(uint8_t const *tensor, uint64_t num_rows,
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

void print_32btensor(uint32_t const *tensor, uint64_t num_rows,
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
  printf("=  CONV2D int8 =\n");
  printf("================\n");
  printf("\n");
  #ifdef NHWC
  printf("Format is NHWC\n");
  #else
  printf("Format is NCHW\n");
  #endif
  printf("\n");
  printf("----------------------------------------------------------------\n");
  printf("Calculating convolution between \n");
  printf("Input of [1 x %i x %i x %i] and Filters of [%i x %i x %i x %i]  \n", L, (M + F -1), (N+F-1), K, L, F, F);
  printf("Result is an output of [1 x %i x %i x %i] \n", K, M, N);
  printf("----------------------------------------------------------------\n");

  // Call the main kernel, and measure cycles
  //start_timer();
  if ( F == 1 || F == 3 || F == 5 || F == 7){
#ifndef NHWC 
	#ifdef WIDE
	start_timer();
		#ifdef NAIVE
			iconv2d_tensor8_naive_wide(o, i, f, M, N, L, F, K);
		#else
			iconv2d_tensor8_wide(o, i, f, M, N, L, F, K);
		#endif
	#else
		iconv2d_tensor8(o, i, f, M, N, L, F, K);
	#endif
	
#else
	if (F == 1)
		iconv2d_tensor8_NHWC(o, i, f, M, N, L, F, K);
	else //if (F == 3)
		iconv2d_tensor8_NHWC(o, i, f, M, N, L, F, K);
		
#endif
	stop_timer();
	}
  else
    printf("Error: the filter size is different from 1 or 3 or 5 or 7.\n");	
  //stop_timer();
	
  // Performance metrics
  #ifndef SPIKE
  int64_t runtime = get_timer();
  float performance = 2.0 * K * L * F * F * M * N / runtime;
  #ifdef WIDE
  float utilization = 100 * performance / (2 * 2.0 * NR_LANES);
  #else
  float utilization = 100 * performance / (4 * 2.0 * NR_LANES);
  #endif
  
  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
  #endif
    
  
   // Verify correctness
  printf("Verifying result...\n");
  
#ifdef WIDE
  int error = verify_32btensor(o, golden_o, M, N, K);
#else
  int error = verify_8btensor(o, golden_o, M, N, K);
#endif

  if (error != 0) {
    printf("Fail.\n");
  } else {
    printf("Passed.\n");
  }
  
  //print_32btensor(o, M, N, K);  
  print_8btensor(o, M, N, K);  
  

  

  return 0; //error;
}
