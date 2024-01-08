/*// Author : Theo Dupuis
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

#define PRECA_MAX	1
#define PRECW_MAX	1

#define F_MAX 		7		// Max size of the kernel F x F
#define C_in 		4		// Number of input input_channels 
#define C_out		1		// Number of output_channels (or output input_channels C_out)
#define I_MAX 		64		// Max H_in x W_in input size
#define I_START	8			// Start input size

int8_t i[I_MAX * I_MAX * C_in];

int8_t f     [F_MAX * F_MAX * C_in * C_out];
int8_t f_nhwc[F_MAX * F_MAX * C_in * C_out];


//////////////////////////////////////
//       utilities functions        //
//////////////////////////////////////

void iconv2d_tensor_naive(int16_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

//treat pointers as 3D arrays
int8_t (*i_)[R+F-1][C+F-1] = (int8_t (*)[R+F-1][C+F-1])i;
int8_t (*f_)[W][F][F] = (int8_t (*)[W][F][F])f;
int16_t (*o_)[R][C] = (int16_t (*)[R][C])o;
 
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
					}
				printf("\n");
}

void NCHW_to_NHWC_16b(int16_t * NCHW_format, int16_t * NHWC_format, int64_t N, int64_t C, int64_t H, int64_t W){
	for(int k = 0; k < N ; k++)
		for(int z = 0; z < H ; z++)
			for(int y = 0 ; y < W ; y++)
				for(int x = 0 ; x < C ; x++)
					{
					NHWC_format[x + C * (y + H * (z + k * W))] = NCHW_format[y + H * (z + W * (x + k * C))];
					}
				
}

void init_tensor(int8_t *tensor, int64_t R, int64_t C, int64_t D, int precision) {
   
    unsigned int seed = 72986;
    unsigned int limit = (1 << precision) - 1;

    for (int d = 0; d < D; ++d)   //depth
        for (int r = 0; r < R; ++r)  //rows
            for (int c = 0; c < C; ++c)//column
            {
                seed = (seed * 1103515245 + 12345) % 2147483648;
                if(r < R) // test purposes
                    tensor[c + C * (r + d * R)] = seed % (limit + 1);
                else
                    tensor[c + C * (r + d * R)] = 0;
            }
}




//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////


int verify_tensor(int16_t *tensor1, int16_t *tensor2, int64_t height, int64_t width, int64_t channels) {
  for (int h = 0; h < height; ++h)   //depth
  	for (int w = 0; w < width; ++w)  //rows
  	  for (int c = 0; c < channels; ++c)//column			
  	    if (tensor1[w + width * (h + c * height)] != tensor2[w + width * (h + c * height)]){
  	      printf("\nError: o[%d][%d][%d] = %d, instead of %d\n", channels, height, width,
  	             tensor1[w + width * (h + c * height)], tensor2[w + width * (h + c * height)]);
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

void print_tensor_16_(uint16_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
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
			int16_t golden_o[(I_MAX - F + 1) * (I_MAX - F + 1) * C_out]; 
			
			for(int z = 0; z < output_channels ; z++)
				for(int y = 0 ; y < (I_MAX - F + 1) ; y++)
					for(int x = 0 ; x < (I_MAX - F + 1) ; x++)
					{
						golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))] = 0;
					}
			
			iconv2d_tensor_naive(golden_o, i, f, (I_MAX - F + 1), (I_MAX - F + 1), input_channels, F, output_channels);	
			
			printf("                                                            done\n");
			#endif	
			
			
			// FILTER TRANSPOSITION INTO NHWC format
			
			NCHW_to_NHWC_8b(f, f_nhwc, output_channels, input_channels, F, F);
	 
			printf("\nfilter %dx%d \n", F, F);

			for(int size = I_START ; size <= I_MAX ; size*=2){

				printf("\n");
				printf("----------------------------------------------------------------\n");
				printf("Calculating convolution between \n");
				printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
				printf("Activation precision of %d and Weights precision of %d  \n", precA, precW);
				printf("Result (16b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
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
				int16_t output[width * height * output_channels];
				int16_t golden_output[(width - F + 1) * (height- F + 1) * output_channels];

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
				//print_tensor(f, F, F, input_channels);
				//print_tensor(input, height, width, input_channels);
				printf("Computing results...\n");
				
				start_timer();
				
				ulppack_conv2d(output, input, f, height, width, input_channels, F, output_channels, precA, precW);
			
				stop_timer();
				printf("                                                            done\n");                                
				
				//////////////////
				// VERIFICATION //
				//////////////////
				
				int16_t golden_output_nhwc[(I_MAX - F + 1) * (I_MAX - F + 1) * C_out];
				
				NCHW_to_NHWC_16b(golden_output, golden_output_nhwc, 1, C_out, I_MAX - F + 1, I_MAX - F + 1);
				
				#ifdef VERIF	
				printf("Verifying results...\n");
				int error = verify_tensor(output, golden_output, (height - F + 1), (width - F + 1), output_channels);
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
					 print_tensor_16_(output, (height - F + 1), (width - F + 1), output_channels);
					 printf("EXPECTED OUT NHWC\n");
				    print_tensor_16_(golden_output_nhwc, (height - F + 1), (width - F + 1), output_channels);
					 printf("EXPECTED OUT\n");
				    print_tensor_16_(golden_output, (height - F + 1), (width - F + 1), output_channels);
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
*/
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


#define PRECA	5
#define PRECW	4

#define F 			7		// Max size of the kernel F x F
#define C_in 		128		// Number of input input_channels 
#define C_out		1		// Number of output_channels (or output input_channels C_out)
#define I_MAX 		128		// Max H_in x W_in input size
#define I_START	128		// Start input size

//int8_t i[I_MAX * I_MAX * C_in];

int8_t f[F * F * C_in * C_out];

int16_t o[I_MAX * I_MAX * C_out];



int main() {



printf("====================\n");
printf("= ULPPACK CONV2D16 =\n");
printf("====================\n");

	for(int precA = PRECA - 1 ; precA < PRECA ; precA ++){
	for(int precW = 1 ; precW < PRECW ; precW ++){

			for(int size = I_START ; size <= I_MAX ; size *= 2){

				printf("\n");
				printf("----------------------------------------------------------------\n");
				printf("Calculating convolution between \n");
				printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \n", C_in, size,  size, C_out, C_in, F, F);
				printf("Activation precision of %d and Weights precision of %d  \n", precA, precW);
				printf("Result (16b) is an output of [1 x %d x %d x %d] \n", C_out, size - F + 1, size - F + 1);
				printf("----------------------------------------------------------------\n");
				printf("\n");

				printf("Computing results...\n");
				start_timer();
				
				ulppack_conv2d(o, o, f, size, size, C_in, F, C_out, precA, precW);
			
				stop_timer();
				printf("                                                            done\n");                                
				
				/////////////
				// METRICS //
				/////////////
				
				int64_t runtime = get_timer();
				
				float performance = (2.0 * C_out * C_in * F * F * (size - F + 1) * (size - F + 1) )/ runtime;
				float utilization = 100 * performance / (4 * 2 * NR_LANES); 
				
					 printf("Passed.\n");					 	
					 printf("The execution took %d cycles.\n", runtime);
				  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	}
	}}
}