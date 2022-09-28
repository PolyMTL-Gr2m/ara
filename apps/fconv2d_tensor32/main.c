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

#define PRECISION_FLOAT 0.01

float i[I_MAX * I_MAX * C_in];

float f[F_MAX * F_MAX * C_in * C_out];

//////////////////////////////////////
// initialisation functions         //
//////////////////////////////////////

void fconv2d_tensor32_naive_wide(double *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
float (*i_)[R+F-1][C+F-1] = (float (*)[R+F-1][C+F-1])i;
float (*f_)[W][F][F] = (float (*)[W][F][F])f;
double (*o_)[R][C] = (double (*)[R][C])o;
 
for(int k = 0 ; k < K ; k++) 
	for(int ch = 0 ; ch < W ; ch++)
		for(int r = 0 ; r < R ; r++)
			for(int c = 0 ; c < C ; c++)
				for(int fh = 0 ; fh < F ; fh++)
					for(int fw = 0 ; fw < F ; fw++) {
						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
					}
}

void fconv2d_tensor32_naive(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
float (*i_)[R+F-1][C+F-1] = (float (*)[R+F-1][C+F-1])i;
float (*f_)[W][F][F] = (float (*)[W][F][F])f;
float (*o_)[R][C] = (float (*)[R][C])o;
 
for(int k = 0 ; k < K ; k++) 
	for(int ch = 0 ; ch < W ; ch++)
		for(int r = 0 ; r < R ; r++)
			for(int c = 0 ; c < C ; c++)
				for(int fh = 0 ; fh < F ; fh++)
					for(int fw = 0 ; fw < F ; fw++) {
						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
					}
}


void init_tensor(float *tensor, int64_t R, int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	  {
  	  		tensor[c + C * (r + d * R)] = (1 + d + r + c) / 8;
  	  }
}

void init_tensor_wide(float *tensor, int64_t R, int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	  {
  	  		tensor[c + C * (r + d * R)] = (1 + d + r + c) / 8;
  	  }
}



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

int main() {

printf("===================\n");
printf("= CONV2D f32 (SP) =\n");
printf("===================\n");

	///////////////////////////////
	// SAME SIZE OUTPUT 32b -> 32b //
	///////////////////////////////
	

#ifndef WIDE_ONLY

printf("Filling the input tensor and filters...");
init_tensor(i, I_MAX, I_MAX, C_in);
init_tensor(f, F_MAX, F_MAX, C_in);
printf("   done\n");

for(int64_t F = 1 ; F <= F_MAX ; F += 2){
 
	printf("\nfilter %dx%d \n", F, F);

	for(int size = I_START ; size <= I_MAX ; size *= 2){

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
	fconv2d_tensor32_naive(golden_output, input, filter, (height - F + 1), (width - F + 1), channels, F, filters);	
	#endif	
	
	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////
	
	printf("   done\n");
	printf("Computing results...");
	start_timer();	
	fconv2d_tensor32(output, input, filter, height, width, channels, F, filters);
	stop_timer();
	printf("   done\n");
	
	//////////////////
	// VERIFICATION //
	//////////////////
	
	#ifdef VERIF	
	printf("Verifying results...");
	int error = verify_f32tensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
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

	for(int size = I_START ; size <= I_MAX ; size *= 2){

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
	  
	float input[width * height * channels];
	float filter[filters * F * F * channels];
	double output[width * height * filters];
	double golden_output[(width - F + 1) * (height- F + 1) * filters];

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
	fconv2d_tensor32_naive_wide(golden_output, input, filter, (height - F + 1), (width - F + 1), channels, F, filters);	
	#endif	
	
	///////////////////////////
	// FONCTION TO BE TESTED //
	///////////////////////////
	
	printf("   done\n");
	printf("Computing results...");
	start_timer();	
	fconv2d_tensor32_wide(output, input, filter, height, width, channels, F, filters);
	stop_timer();
	printf("   done\n");
	
	//////////////////
	// VERIFICATION //
	//////////////////
	
	#ifdef VERIF	
	printf("Verifying results...");
	int error = verify_f64tensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
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
	float utilization = 100 * performance / (2.0 * NR_LANES);

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
