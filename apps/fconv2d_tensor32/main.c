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

#include "fconv2d_tensor32.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

#define F_MAX 7
#define I_MAX 32

#define PRECISION_FLOAT 0.01

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[(M-F+1)x(N-F+1)xK]
// The filter is a cube tensor, and F is odd
// Tensors defined in data.S

// input and filters

extern float i[] __attribute__((aligned(4 * NR_LANES))); // [M x N x L]
extern float f[] __attribute__((aligned(4 * NR_LANES))); // [K x F x F x L]

// o points to a [(M + F - 1) x (N - F + 1) x K] memory vector which is set to 0 at the start
		
extern float o[] __attribute__((aligned(4 * NR_LANES)));

// expected output (golden_o) are size of [(M + F - 1) x (N - F + 1) x K]

extern float golden_o_1[] __attribute__((aligned(4 * NR_LANES)));
extern float golden_o_3[] __attribute__((aligned(4 * NR_LANES)));
extern float golden_o_5[] __attribute__((aligned(4 * NR_LANES)));
extern float golden_o_7[] __attribute__((aligned(4 * NR_LANES)));

// o points to a [(M + F - 1) x (N - F + 1) x K] memory vector which is set to 0 at the start

extern double o_wide[] __attribute__((aligned(4 * NR_LANES)));

// expected output (golden_o) are size of [(M + F - 1) x (N - F + 1) x K]

extern double golden_o_1_wide[] __attribute__((aligned(4 * NR_LANES))); 
extern double golden_o_3_wide[] __attribute__((aligned(4 * NR_LANES)));
extern double golden_o_5_wide[] __attribute__((aligned(4 * NR_LANES))); 
extern double golden_o_7_wide[] __attribute__((aligned(4 * NR_LANES))); 

// M, N, L, K defined in data.S

extern int64_t M; // number of input rows
extern int64_t N; // number of input column
extern int64_t L; // number of channels
extern int64_t K; // number of filters (1 by default)



//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////


int verify_f64tensor(double *tensor1, double *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if ((tensor1[c + C * (r + d * R)] < (tensor2[c + C * (r + d * R)] - PRECISION_FLOAT)) || (tensor1[c + C * (r + d * R)] > (tensor2[c + C * (r + d * R)] + PRECISION_FLOAT))){
  	      printf("Error: o[%d][%d][%d] = %f, instead of %f\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      return 1;
  	   }
return 0;
  		
}

int verify_f32tensor(float *tensor1, float *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if ((tensor1[c + C * (r + d * R)] < (tensor2[c + C * (r + d * R)] - PRECISION_FLOAT)) || (tensor1[c + C * (r + d * R)] > (tensor2[c + C * (r + d * R)] + PRECISION_FLOAT))){
  	      printf("Error: o[%d][%d][%d] = %f, instead of %f\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      return 1;
  	   }
return 0;
  		
}


void print_f64tensor(double *tensor, uint64_t num_rows,
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

void print_f32tensor(float *tensor, uint64_t num_rows,
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

printf("===================\n");
printf("=  CONV2D float32 =\n");
printf("===================\n");

	/////////////////////////////////
	// SAME SIZE OUTPUT 32b -> 32b //
	/////////////////////////////////

#ifndef WIDE_ONLY
for(int64_t F = 1 ; F <= F_MAX ; F += 2){

	float * golden_o;

	switch(F){
		case 1: golden_o = golden_o_1; break;
		case 3: golden_o = golden_o_3; break;
		case 5: golden_o = golden_o_5; break;
		case 7: golden_o = golden_o_7; break;
	}
 
	printf("\nfilter %ix%i \n", F, F);

	for(int size = 8 ; size <= I_MAX ; size = size * 2){

	printf("----------------------------------------------------------------\n");
	printf("Calculating convolution between \n");
	printf("Input of [1 x %i x %i x %i] and Filters of [%i x %i x %i x %i]  \n", L, size,  size, K, L, F, F);
	printf("Result (32b) is an output of [1 x %i x %i x %i] \n", K, size - F + 1, size - F + 1);
	printf("----------------------------------------------------------------\n");

	printf("Computing results...\n");

	int channels = L; // channel size is fixed for simplicity
	int width = size;
	int height = size;  

	int filters = K;
	  
	float input[width * height * channels];
	float filter[filters * F * F * channels];
	float output[width * height * filters];
	float golden_output[(width - F + 1) * (height- F + 1) * filters];

	////////////////////////////////////////////////
	// INPUT, FILTERS AND EXPECTED OUTPUT SLICING //
	////////////////////////////////////////////////
	
	for(int k = 0; k < filters ; k++)
		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < F ; y++)
				for(int x = 0 ; x < F ; x++)
		  			filter[x + F * (y + F * (z + k * channels))] = f[x + F_MAX * (y + F_MAX * (z + k * channels))];

	for(int z = 0; z < channels ; z++)
		for(int y = 0 ; y < height ; y++)
			for(int x = 0 ; x < width ; x++)
	  			input[x + width * (y + z * height)] = i[x + N * (y + z * M)];
	  			
	for(int z = 0; z < filters ; z++)
		for(int y = 0 ; y < (height - F + 1) ; y++)
			for(int x = 0 ; x < (width - F + 1) ; x++)
	  			golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = golden_o[x + (N - F + 1) * (y + z * (M - F + 1))];

	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////	
	#ifndef SPIKE
	start_timer();
	#endif			
	fconv2d_tensor32(output, input, filter, height, width, channels, F, filters);
	#ifndef SPIKE
	stop_timer();
	#endif
	
	//////////////////
	// VERIFICATION //
	//////////////////
		
	printf("Verifying results...\n");
		
	int error = verify_f32tensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
	
	//print_16btensor(input, height, width, channels);
	//print_32btensor(output, (height - F + 1), (width - F + 1), filters);

	/////////////
	// METRICS //
	/////////////
	
	#ifndef SPIKE
	int64_t runtime = get_timer();
	float performance = 2.0 * filters * channels * F * F * (height - F + 1) * (width - F + 1) / runtime;
	float utilization = 100 * performance / (2 * 2.0 * NR_LANES);
	#endif

	if (error != 0)
		 printf("Fail.\n");
	else {
		 printf("Passed.\n");
		 #ifndef SPIKE  
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  	 #endif
	  }
	}
}
	
#endif
	
	////////////////////////////
	// WIDE OUTPUT 32b -> 64b //
	////////////////////////////
	
#ifndef SIMPLE_ONLY

for(int64_t F = 1 ; F <= F_MAX ; F += 2){
	
	double * golden_o;

	switch(F){
		case 1: golden_o = golden_o_1_wide; break;
		case 3: golden_o = golden_o_3_wide; break;
		case 5: golden_o = golden_o_5_wide; break;
		case 7: golden_o = golden_o_7_wide; break;
	}
 
	printf("\nfilter %ix%i \n", F, F);

	for(int size = 8 ; size <= I_MAX ; size = size * 2){

	printf("----------------------------------------------------------------\n");
	printf("Calculating convolution between \n");
	printf("Input of [1 x %i x %i x %i] and Filters of [%i x %i x %i x %i]  \n", L, size,  size, K, L, F, F);
	printf("Result (64b) is an output of [1 x %i x %i x %i] \n", K, size - F + 1, size - F + 1);
	printf("----------------------------------------------------------------\n");

	printf("Computing results...\n");

	int channels = L;
	int width = size;
	int height = size;  

	int filters = K;
	  
	float input[width * height * channels];
	float filter[filters * F * F * channels];
	double output[width * height * filters]; // keep empty space at max size (just for the test)
	double golden_output[(width - F + 1) * (height - F + 1) * filters];
	
	////////////////////////////////////////////////
	// INPUT, FILTERS AND EXPECTED OUTPUT SLICING //
	////////////////////////////////////////////////
	
	for(int k = 0; k < filters ; k++)
		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < F ; y++)
				for(int x = 0 ; x < F ; x++)
		  			filter[x + F * (y + F * (z + k * channels))] = f[x + F_MAX * (y + F_MAX * (z + k * channels))];

	for(int z = 0; z < channels ; z++)
		for(int y = 0 ; y < height ; y++)
			for(int x = 0 ; x < width ; x++)
	  			input[x + width * (y + z * height)] = i[x + N * (y + z * M)];
	  			
	for(int z = 0; z < filters ; z++)
		for(int y = 0 ; y < (height - F + 1) ; y++)
			for(int x = 0 ; x < (width - F + 1) ; x++)
	  			golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = golden_o[x + (N - F + 1) * (y + z * (M - F + 1))];
	  			
	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////
	#ifndef SPIKE
	start_timer();			
	#endif
	fconv2d_tensor32_wide(output, input, filter, height, width, channels, F, filters);
	#ifndef SPIKE
	stop_timer();
	#endif
	
	//////////////////
	// VERIFICATION //
	//////////////////
		
	printf("Verifying results...\n");
		
	int error = verify_f64tensor(output, golden_output, (height - F + 1), (width - F + 1), filters);

	/////////////
	// METRICS //
	/////////////
	#ifndef SPIKE
	int64_t runtime = get_timer();
	float performance = 2.0 * filters * channels * F * F * (height - F + 1) * (width - F + 1) / runtime;
	float utilization = 100 * performance / (2.0 * NR_LANES); 
	#endif
	
	  if (error != 0) {
		 printf("Fail.\n");
	  } else {
		 printf("Passed.\n");
		 #ifndef SPIKE  
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  	 #endif
	  }
	}
}
#endif
  
  return 0; //error;
}
