// Author : Theo Dupuis
// GR2M - 2022
// Polytechnique Montreal

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ibsconv2d_tensor32.h"
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
#define PRECW_MAX	1

#define F_MAX 		5		// Max size of the kernel F x F
#define C_in 		64		// Number of input input_channels 
#define C_out		2		// Number of output_channels (or output input_channels C_out)
#define I_MAX 		10		// Max H_in x W_in input size
#define I_START	10			// Start input size

int8_t i[I_MAX * I_MAX * C_in];

int8_t f     [F_MAX * F_MAX * C_in * C_out];
int8_t f_nhwc[F_MAX * F_MAX * C_in * C_out];


//////////////////////////////////////
//       utilities functions        //
//////////////////////////////////////

void iconv2d_tensor64_naive(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
int8_t (*i_)[R+F-1][C+F-1] = (int32_t (*)[R+F-1][C+F-1])i;
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

void NCHW_to_NHWC_8b(int8_t * NCHW_format, int8_t * NHWC_format, int64_t N, int64_t C, int64_t H, int64_t W){
	for(int k = 0; k < N ; k++)
		for(int z = 0; z < H ; z++)
			for(int y = 0 ; y < W ; y++)
				for(int x = 0 ; x < C ; x++)
					{
					NHWC_format[x + C * (y + H * (z + k * W))] = NCHW_format[y + H * (z + W * (x + k * C))];
					//printf("%10u ", y + H * (z + W * (x + k * C)));
					}
				printf("\n");
}

void NCHW_to_NHWC_32b(int32_t * NCHW_format, int32_t * NHWC_format, int64_t N, int64_t C, int64_t H, int64_t W){
	for(int k = 0; k < N ; k++)
		for(int z = 0; z < H ; z++)
			for(int y = 0 ; y < W ; y++)
				for(int x = 0 ; x < C ; x++)
					{
					NHWC_format[x + C * (y + H * (z + k * W))] = NCHW_format[y + H * (z + W * (x + k * C))];
					//printf("%10u ", y + H * (z + W * (x + k * C)));
					}
				
}

void init_tensor(int8_t *tensor, int64_t R, int64_t C, int64_t D, int64_t precision) {

int64_t const mask = (1 << precision) - 1;
int64_t const val_init = 8888888888888;//37550;
int64_t val = 0;

for (int d = 0; d < D; ++d)   //depth
	for (int r = 0; r < R; ++r)  //rows
		for (int c = 0; c < C; ++c)//column
  	   {
  	   	
  	   		val = (val + val_init) % 27;//65535; //pseudo random generator
  	   		tensor[c + C * (r + d * R)] = val & mask;
  	  			
  	   }		
}



//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////


int verify_tensor(int32_t *tensor1, int32_t *tensor2, int64_t height, int64_t width, int64_t channels) {
  for (int h = 0; h < height; ++h)   //depth
  	for (int w = 0; w < width; ++w)  //rows
  	  for (int c = 0; c < channels; ++c)//column			
  	    if (tensor1[w + width * (h + c * channels)] != tensor2[w + width * (h + c * channels)]){
  	      printf("\nError: o[%d][%d][%d] = %d, instead of %d\n", channels, height, width,
  	             tensor1[w + width * (h + c * channels)], tensor2[w + width * (h + c * channels)]);
  	      return 1;
  	   }
return 0;
  		
}

void print_tensor(uint8_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10u ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}

void print_tensor_32_(uint32_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10u ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}




int main() {



printf("==============\n");
printf("= BSCONV2D32 =\n");
printf("==============\n");

	/////////////////////////////////
	// SAME SIZE OUTPUT 64b -> 64b //
	/////////////////////////////////
	
for(int64_t precA = PRECA_MAX; precA <= PRECA_MAX; precA++){
	for(int64_t precW = PRECW_MAX; precW <= PRECW_MAX; precW++){
		printf("\n");
		printf("************\n");
		printf("*** A%dW%d ***\n", precA, precW);
		printf("************\n");

		printf("\n");
		printf("Filling the input and filter tensors...\n");

		init_tensor(i, I_MAX, I_MAX, C_in, precA);
		init_tensor(f, F_MAX, F_MAX, C_in * C_out, precW);

			printf("                                                            done\n");

		for(int64_t F = F_MAX ; F <= F_MAX ; F += 2){

			int64_t input_channels = C_in; // channel size is fixed for simplicity
			int64_t output_channels = C_out;
			int8_t filter[output_channels * F * F * input_channels];

			for(int k = 0; k < output_channels ; k++)
				for(int z = 0; z < input_channels ; z++)
					for(int y = 0 ; y < F ; y++)
						for(int x = 0 ; x < F ; x++)
				  			filter[x + F * (y + F * (z + k * input_channels))] = f[x + F_MAX * (y + F_MAX * (z + k * input_channels))];
			
			#ifdef VERIF
			printf("Computing the expected output for this kernel size...\n");
			//Compute the expected output
			int32_t golden_o[(I_MAX - F + 1) * (I_MAX - F + 1) * C_out]; 
			
			for(int z = 0; z < output_channels ; z++)
				for(int y = 0 ; y < (I_MAX - F + 1) ; y++)
					for(int x = 0 ; x < (I_MAX - F + 1) ; x++)
					{
						golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))] = 0;
					}
			
			iconv2d_tensor64_naive(golden_o, i, f, (I_MAX - F + 1), (I_MAX - F + 1), input_channels, F, output_channels);	
			
			printf("                                                            done\n");
			#endif	
			
			
			// FILTER TRANSPOSITION INTO NHWC format
			
			NCHW_to_NHWC_8b(f, f_nhwc, output_channels, input_channels, F, F);
	 
			printf("\nfilter %dx%d \n", F, F);

			for(int size = I_START ; size <= I_MAX ; size++ /**= 2*/){

				printf("\n");
				printf("----------------------------------------------------------------\n");
				printf("Calculating convolution between \n");
				printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
				printf("Activation precision of %d and Weights precision of %d  \n", precA, precW);
				printf("Result (32b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
				printf("----------------------------------------------------------------\n");
				printf("\n");
				
				#ifdef VERIF
				printf("Formatting data and expected outputs...\n");
				#else
				printf("Formatting data...\n");
				#endif


				int64_t width = size;
				int64_t height = size;  
				
				int8_t input  [width * height * input_channels];
				int8_t i_nhwc [width * height * input_channels];
				int32_t output[width * height * output_channels];
				int32_t golden_output[(width - F + 1) * (height- F + 1) * output_channels];

				////////////////////////////////////////////////
				// INPUT, FILTERS AND EXPECTED OUTPUT SLICING //
				////////////////////////////////////////////////

				for(int z = 0; z < input_channels ; z++)
					for(int y = 0 ; y < height ; y++)
						for(int x = 0 ; x < width ; x++)
				  			input[x + width * (y + z * height)] = i[x + I_MAX * (y + z * I_MAX)];
				  			
				for(int z = 0; z < output_channels ; z++)
					for(int y = 0 ; y < (height - F + 1) ; y++)
						for(int x = 0 ; x < (width - F + 1) ; x++)
						{
							output[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
							#ifdef VERIF
							golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))];
							#endif
						}
						
				NCHW_to_NHWC_8b(input, i_nhwc, 1, input_channels, height, width);
				
				///////////////////////////
				// FONCTION TO BE TESTED //
				///////////////////////////
				
				printf("                                                            done\n");
				printf("Computing results...\n");
				
				start_timer();
				
				ibsconv2d_tensor32_3x3(output, i_nhwc, f_nhwc, height, width, input_channels, F, output_channels, precA, precW);
			
				stop_timer();
				printf("                                                            done\n");                                
				
				//////////////////
				// VERIFICATION //
				//////////////////
				
				int32_t golden_output_nhwc[(I_MAX - F + 1) * (I_MAX - F + 1) * C_out];
				
				NCHW_to_NHWC_32b(golden_output, golden_output_nhwc, 1, C_out, I_MAX - F + 1, I_MAX - F + 1);
				
				#ifdef VERIF	
				printf("Verifying results...\n");
				int error = verify_tensor(output, golden_output_nhwc, (height - F + 1), (width - F + 1), output_channels);
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
				
				int64_t runtime = get_timer();
				float performance = 2.0 * C_out * C_in * F * F * (size - F + 1) * (size - F + 1) / runtime;
				float utilization = 100 * performance / (256 / (precA * precW)) * NR_LANES; 
				
				if (error != 0){
					 printf("Fail.\n");
					 printf("OUT NHWC\n");
					 print_tensor_32_(output, (height - F + 1), (width - F + 1), output_channels);
					 printf("EXPECTED OUT NHWC\n");
				    print_tensor_32_(golden_output_nhwc, (height - F + 1), (width - F + 1), output_channels);
					 printf("EXPECTED OUT\n");
				    print_tensor_32_(golden_output, (height - F + 1), (width - F + 1), output_channels);
				}
				else {
					 printf("Passed.\n");					 	
					 printf("The execution took %d cycles.\n", runtime);
				  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
				  	 #ifdef PERF
				  	 	printf("The execution of bit-serial packing took %d cycles.\n", runtime_bp);
				  	 	printf("The execution of conv2d took %d cycles.\n", runtime - runtime_bp);
				  	 #endif
				  }
				}
			}
		}
	}
}
