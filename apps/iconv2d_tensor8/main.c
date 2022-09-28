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
#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif


// use to check if the results are correct
// since we compute the expectude results with
// scalar code and rolled loops, it had a significant
// amout of time on simulation

#define VERIF


#define F_MAX 		7		// Max size of the kernel F x F
#define C_in 		1		// Number of input channels 
#define C_out		1		// Number of filters (or output channels C_out)
#define I_MAX 		64		// Max H_in x W_in input size
#define I_START	8		// Start input size


int8_t i[I_MAX * I_MAX * C_in];

int8_t f[F_MAX * F_MAX * C_in * C_out];



//////////////////////////////////////
// initialisation functions         //
//////////////////////////////////////


void init_tensor(int8_t *tensor, int64_t R, int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	  {
  	  		tensor[c + C * (r + d * R)] = (1 + d + r + c) % 8;
  	  }
}

void init_tensor_wide(int32_t *tensor, int64_t R, int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	  {
  	  		tensor[c + C * (r + d * R)] = (1 + d + r + c) % 8;
  	  }
}



//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////

void iconv2d_tensor8_naive_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
int8_t (*i_)[R+F-1][C+F-1] = (int8_t (*)[R+F-1][C+F-1])i;
int8_t (*f_)[W][F][F] = (int8_t (*)[W][F][F])f;
int32_t (*o_)[R][C] = (int32_t (*)[R][C])o;
 
for(int k = 0 ; k < K ; k++) 
	for(int ch = 0 ; ch < W ; ch++)
		for(int r = 0 ; r < R ; r++)
			for(int c = 0 ; c < C ; c++)
				for(int fh = 0 ; fh < F ; fh++)
					for(int fw = 0 ; fw < F ; fw++) {
						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
					}
}

void iconv2d_tensor8_naive(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
int8_t (*i_)[R+F-1][C+F-1] = (int8_t (*)[R+F-1][C+F-1])i;
int8_t (*f_)[W][F][F] = (int8_t (*)[W][F][F])f;
int8_t (*o_)[R][C] = (int8_t (*)[R][C])o;
 
for(int k = 0 ; k < K ; k++) 
	for(int ch = 0 ; ch < W ; ch++)
		for(int r = 0 ; r < R ; r++)
			for(int c = 0 ; c < C ; c++)
				for(int fh = 0 ; fh < F ; fh++)
					for(int fw = 0 ; fw < F ; fw++) {
						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
					}
}


int verify_8btensor(uint8_t *tensor1, uint8_t *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if (tensor1[c + C * (r + d * R)] != tensor2[c + C * (r + d * R)]) {
  	      printf("Error: o[%d][%d][%d] = %ld, instead of %ld\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      //return 1;
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
  	      //return 1;
  	   }
return 0;
  		
}


int main() {

printf("===============\n");
printf("= CONV2D int8 =\n");
printf("===============\n");

	///////////////////////////////
	// SAME SIZE OUTPUT 8b -> 8b //
	///////////////////////////////
	

#ifndef WIDE_ONLY

printf("Filling the input tensor and filters...");
init_tensor(i, I_MAX, I_MAX, C_in);
init_tensor(f, F_MAX, F_MAX, C_in);
printf("   done\n");

for(int64_t F = 1 ; F <= F_MAX ; F += 2){
 
	printf("\nfilter %dx%d \n", F, F);

	for(int size = 16 ; size <= I_MAX ; size *= 2){

	printf("\n");
	printf("----------------------------------------------------------------\n");
	printf("Calculating convolution between \n");
	printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
	printf("Result (8b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
	printf("----------------------------------------------------------------\n");
	printf("\n");
	printf("\n");

	#ifdef VERIF
	printf("Formatting data and expected outputs...");
	#else
	printf("Formatting data...");
	#endif

	int64_t channels = C_in; // channel size is fixed for simplicity
	int64_t width = size;
	int64_t height = size;  

	int64_t filters = C_out;
	  
	int8_t input[width * height * channels];
	int8_t filter[filters * F * F * channels];
	int8_t output[width * height * filters];
	int8_t golden_output[(width - F + 1) * (height- F + 1) * filters];

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
	  			input[x + width * (y + z * height)] = i[x + I_MAX * (y + z * I_MAX)];
	  			
	for(int z = 0; z < filters ; z++)
		for(int y = 0 ; y < (height - F + 1) ; y++)
			for(int x = 0 ; x < (width - F + 1) ; x++)
			{
				output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
				golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
			}
	
	#ifdef VERIF
	//Compute the expected output
	iconv2d_tensor8_naive(golden_output, input, filter, (height - F + 1), (width - F + 1), channels, F, filters);	
	#endif	
	
	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////
	printf("   done\n");
	printf("Computing results...");
	start_timer();	
	iconv2d_tensor8(output, input, filter, height, width, channels, F, filters);
	stop_timer();
	printf("   done\n");
	//////////////////
	// VERIFICATION //
	//////////////////
	
	#ifdef VERIF	
	printf("Verifying results...");
	int error = verify_8btensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
	printf("   done\n");
	#else
	printf("-- Change macro to add verification step -- \n");
	int error = 0;
	#endif
	
	/////////////
	// METRICS //
	/////////////
	
	
	int64_t runtime = get_timer();
	float performance = 2.0 * filters * channels * F * F * (height - F + 1) * (width - F + 1) / runtime;
	float utilization = 100 * performance / (8 * 2.0 * NR_LANES);

	if (error != 0)
		 printf("Fail.\n");
	else {
		 printf("Passed.\n");
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  }
	}
}
	
#endif
	
	///////////////////////////
	// WIDE OUTPUT 8b -> 32b //
	///////////////////////////
	
#ifndef SIMPLE_ONLY

printf("Filling the input tensor and filters...");
init_tensor(i, I_MAX, I_MAX, C_in);
init_tensor(f, F_MAX, F_MAX, C_in);
printf("   done\n");

for(int64_t F = 1 ; F <= F_MAX ; F += 2){
 
	printf("\nfilter %dx%d \n", F, F);

	for(int size = 16 ; size <= I_MAX ; size *= 2){

	printf("\n");
	printf("----------------------------------------------------------------\n");
	printf("Calculating convolution between \n");
	printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
	printf("Result (32b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
	printf("----------------------------------------------------------------\n");
	printf("\n");
	printf("\n");
	
	#ifdef VERIF
	printf("Formatting data and expected outputs...");
	#else
	printf("Formatting data...");
	#endif

	int64_t channels = C_in; // channel size is fixed for simplicity
	int64_t width = size;
	int64_t height = size;  

	int64_t filters = C_out;

	  
	int8_t input[width * height * channels];
	int8_t filter[filters * F * F * channels];
	int32_t output[width * height * filters];
	int32_t golden_output[(width - F + 1) * (height- F + 1) * filters];

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
	  			input[x + width * (y + z * height)] = i[x + I_MAX * (y + z * I_MAX)];
	  			
	for(int z = 0; z < filters ; z++)
		for(int y = 0 ; y < (height - F + 1) ; y++)
			for(int x = 0 ; x < (width - F + 1) ; x++)
			{
				output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
				golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
			}
	
	#ifdef VERIF
	//Compute the expected output
	iconv2d_tensor8_naive_wide(golden_output, input, filter, (height - F + 1), (width - F + 1), channels, F, filters);	
	#endif	
	
	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////
	
	printf("   done\n");
	printf("Computing results...");
	start_timer();	
	iconv2d_tensor8_wide(output, input, filter, height, width, channels, F, filters);
	stop_timer();
	printf("   done\n");
	
	//////////////////
	// VERIFICATION //
	//////////////////
	
	#ifdef VERIF	
	printf("Verifying results...");
	int error = verify_32btensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
	printf("   done\n");
	#else
	printf("-- Change macro to add verification step -- \n");
	int error = 0;
	#endif
	
	/////////////
	// METRICS //
	/////////////
	
	
	int64_t runtime = get_timer();
	float performance = 2.0 * filters * channels * F * F * (height - F + 1) * (width - F + 1) / runtime;
	float utilization = 100 * performance / (2 * 2.0 * NR_LANES);

	if (error != 0)
		 printf("Fail.\n");
	else {
		 printf("Passed.\n");
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  }
	}
}
#endif
  
  return 0; //error;
}
