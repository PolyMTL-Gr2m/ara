#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ibsconv2d_tensor8.h"
#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif


// use to check if the results are correct
// since we compute the expectude results with
// scalar code and rolled loops, it had a significant
// amout of time on simulation

#define VERIF

#define PRECA_MAX	2
#define PRECW_MAX	2

#define F_MAX 		3		// Max size of the kernel F x F
#define C_in 		8		// Number of input channels 
#define C_out		1		// Number of filters (or output channels C_out)
#define I_MAX 		6		// Max H_in x W_in input size
#define I_START	6		// Start input size

int8_t i[I_MAX * I_MAX * C_in];

int8_t f[F_MAX * F_MAX * C_in * C_out];

//////////////////////////////////////
// initialisation functions         //
//////////////////////////////////////

void iconv2d_tensor64_naive(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

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


void init_tensor(int8_t *tensor, int64_t R, int64_t C, int64_t D, int64_t precision) {

int64_t const mask = (1 << precision) - 1;
int64_t const val_init = 37550;
int64_t val = 0;

for (int d = 0; d < D; ++d)   //depth
	for (int r = 0; r < R; ++r)  //rows
		for (int c = 0; c < C; ++c)//column
  	   {
  	   	val = (val + val_init) % 65535; //pseudo random generator
  	  		tensor[c + C * (r + d * R)] = val & mask;
  	   }
}


//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////


int verify_tensor(int8_t *tensor1, int8_t *tensor2, int64_t R,
                  int64_t C, int64_t D) {
  for (int d = 0; d < D; ++d)   //depth
  	for (int r = 0; r < R; ++r)  //rows
  	  for (int c = 0; c < C; ++c)//column
  	    if (tensor1[c + C * (r + d * R)] != tensor2[c + C * (r + d * R)]){
  	      printf("\nError: o[%d][%d][%d] = %d, instead of %d\n", d, r, c,
  	             tensor1[c + C * (r + d * R)], tensor2[c + C * (r + d * R)]);
  	      return 1;
  	   }
return 0;
  		
}


void print_tensor(uint8_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
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

printf("=============\n");
printf("= BSCONV2D8 =\n");
printf("=============\n");

	/////////////////////////////////
	// SAME SIZE OUTPUT 64b -> 64b //
	/////////////////////////////////
	
// for now, but need to do nested loop to check for precision from 1 -> 3 bits  

for(int64_t precA = 1; precA <= PRECA_MAX; precA++){
	for(int64_t precW = 1; precW <= PRECA_MAX; precW++){
		printf("\n");
		printf("************\n");
		printf("*** A%dW%d ***\n", precA, precW);
		printf("************\n");

		printf("\n");
		printf("Filling the input and filter tensors...\n");
		init_tensor(i, I_MAX, I_MAX, C_in, precA);
		init_tensor(f, F_MAX, F_MAX, C_in * C_out, precW);

			printf("                                                            done\n");

		for(int64_t F = 3 ; F <= F_MAX ; F += 2){

			int64_t channels = C_in; // channel size is fixed for simplicity
			int64_t filters = C_out;
			int8_t filter[filters * F * F * channels];

			for(int k = 0; k < filters ; k++)
				for(int z = 0; z < channels ; z++)
					for(int y = 0 ; y < F ; y++)
						for(int x = 0 ; x < F ; x++)
				  			filter[x + F * (y + F * (z + k * channels))] = f[x + F_MAX * (y + F_MAX * (z + k * channels))];
			
			#ifdef VERIF
			printf("Computing the expected outuput for this kernel size...\n");
			//Compute the expected output
			int8_t golden_o[(I_MAX - F + 1) * (I_MAX - F + 1) * C_out]; 
			
			for(int z = 0; z < filters ; z++)
				for(int y = 0 ; y < (I_MAX - F + 1) ; y++)
					for(int x = 0 ; x < (I_MAX - F + 1) ; x++)
					{
						golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))] = 0;
					}
			
			iconv2d_tensor64_naive(golden_o, i, f, (I_MAX - F + 1), (I_MAX - F + 1), channels, F, filters);	
			printf("                                                            done\n");
			#endif	
		 
			printf("\nfilter %dx%d \n", F, F);

			for(int size = I_START ; size <= I_MAX ; size *= 2){

				printf("\n");
				printf("----------------------------------------------------------------\n");
				printf("Calculating convolution between \n");
				printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
				printf("Activation precision of %d and Weights precision of %d  \n", precA, precW);
				printf("Result (8b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
				printf("----------------------------------------------------------------\n");
				printf("\n");
				
				#ifdef VERIF
				printf("Formatting data and expected outputs...\n");
				#else
				printf("Formatting data...\n");
				#endif


				int64_t width = size;
				int64_t height = size;  
				
				int8_t input[width * height * channels];
				int8_t output[width * height * filters];
				int8_t golden_output[(width - F + 1) * (height- F + 1) * filters];

				////////////////////////////////////////////////
				// INPUT, FILTERS AND EXPECTED OUTPUT SLICING //
				////////////////////////////////////////////////

				for(int z = 0; z < channels ; z++)
					for(int y = 0 ; y < height ; y++)
						for(int x = 0 ; x < width ; x++)
				  			input[x + width * (y + z * height)] = i[x + I_MAX * (y + z * I_MAX)];
				  			
				for(int z = 0; z < filters ; z++)
					for(int y = 0 ; y < (height - F + 1) ; y++)
						for(int x = 0 ; x < (width - F + 1) ; x++)
						{
							output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
							#ifdef VERIF
							golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))];
							#endif
						}
				
				///////////////////////////
				// FONCTION TO BE TESTED //
				///////////////////////////
				
				printf("                                                            done\n");
				printf("Computing results...\n");
				start_timer();	
				ibsconv2d_tensor8_3x3(output, input, filter, height, width, channels, F, filters, precA, precW);
				stop_timer();
				printf("                                                            done\n");
				
				//////////////////
				// VERIFICATION //
				//////////////////
				
				#ifdef VERIF	
				printf("Verifying results...\n");
				int error = verify_tensor(output, golden_output, (height - F + 1), (width - F + 1), filters);
				if (error == 0)
					printf("                                                            done\n");
				else
					printf("   ERROR\n");
				#else
				printf("-- Change macro to add verification step -- \n");
				int error = 0;
				#endif
				
				/////////////
				// METRICS //
				/////////////
				
				
				/*printf("OUT\n");
				print_tensor(output, (height - F + 1), (width - F + 1), C_out);
				printf("EXPECTED OUT\n");
				print_tensor(golden_output, (height - F + 1), (width - F + 1), C_out);*/
				
				
				int64_t runtime = get_timer();
				float performance = 3.0 * F * F * C_out * (size - F + 1) * (size - F + 1) * precA * precW / runtime;
				float utilization = 100 * performance / (8.0 * NR_LANES); 

				if (error != 0)
					 printf("Fail.\n");
				else {
					 printf("Passed.\n");
					 printf("The execution took %d cycles.\n", runtime);
				  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
				  }
				}
			}
		}
	}
}
