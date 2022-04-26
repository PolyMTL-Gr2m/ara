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

#include "NCWH_to_NWHC.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd

// Matrices defined in data.S
extern int8_t  i8[] __attribute__((aligned(4 * NR_LANES))); 			// [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int16_t i16[] __attribute__((aligned(4 * NR_LANES))); 			// [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int32_t i32[] __attribute__((aligned(4 * NR_LANES))); 			// [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int64_t i64[] __attribute__((aligned(4 * NR_LANES))); 			// [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int8_t  o8[] __attribute__((aligned(4 * NR_LANES)));        	// 
extern int16_t o16[] __attribute__((aligned(4 * NR_LANES)));        	// 
extern int32_t o32[] __attribute__((aligned(4 * NR_LANES)));        	// 
extern int64_t o64[] __attribute__((aligned(4 * NR_LANES)));        	// 
extern int8_t  golden_o8[] __attribute__((aligned(4 * NR_LANES))); 	// 
extern int16_t golden_o16[] __attribute__((aligned(4 * NR_LANES))); 	// 
extern int32_t golden_o32[] __attribute__((aligned(4 * NR_LANES))); 	// 
extern int64_t golden_o64[] __attribute__((aligned(4 * NR_LANES))); 	// 

// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth


// Verify the matrices to implement on tensors
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


// Verify the matrices to implement on tensors
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


int main() {
  printf("\n");
  printf("=================================\n");
  printf("=  TRANSPOSE FROM NCWH to NWHC  =\n");
  printf("=================================\n");
  printf("\n");
  printf("\n");
  

  // Call the transpose function, and measure cycles
  start_timer();
  
  NCWH_to_NWHC_tensor8(o8, i8, M, N, L);
  stop_timer();
  int64_t runtime = get_timer();
  printf("The execution for 8b took %d cycles.\n", runtime);
  
  start_timer();
  
  NCWH_to_NWHC_tensor16(o16, i16, M, N, L);
  stop_timer();
  runtime = get_timer();
  printf("The execution for 16b took %d cycles.\n", runtime);
  
  start_timer();
  
  NCWH_to_NWHC_tensor32(o32, i32, M, N, L);
  stop_timer();
  runtime = get_timer();
  printf("The execution for 32b took %d cycles.\n", runtime);
  
  start_timer();
  
  NCWH_to_NWHC_tensor64(o64, i64, M, N, L);
  stop_timer();
  runtime = get_timer();
  printf("The execution for 64b took %d cycles.\n", runtime);
  
    
   // Verify correctness
  printf("Verifying result...\n");
  
  	int error = verify_8btensor(o8, golden_o8, M, N, L);
	error += verify_16btensor(o16, golden_o16, M, N, L);
	error += verify_32btensor(o32, golden_o32, M, N, L);
	error += verify_64btensor(o64, golden_o64, M, N, L);
  if (error != 0) {
    printf("Fail.\n");
  } else {
    printf("Passed.\n");
  }
  

  

  return 0; //error;
}
