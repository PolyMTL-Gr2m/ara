
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fReLu_tensor64.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

#define I_MAX 32

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd
	// Matrices defined in data.S
extern double i[] __attribute__((aligned(4 * NR_LANES))); // [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern double o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
extern double golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth

int verify_tensor(double *tensor1, double *tensor2, int64_t R,
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
  printf("=================\n");
  printf("=  ReLu float64 =\n");
  printf("=================\n");
  printf("\n");
	
  for(int size = 8 ; size <= I_MAX ; size = size * 2){
	  
	  printf("----------------------------------------------------------------\n");
	  printf("Calculating pooling between \n");
	  printf("Input of [1 x %i x %i x %i] \n", L, size, size);
	  printf("Result (64b) is an output of [1 x %i x %i x %i] \n", L, size, size);
	  printf("----------------------------------------------------------------\n");
	  
	  	int channels = L; // loop could be added to test different channel size but for now we stick to one value only 
		int width = size;
		int height = size;  
		  
		double input[width * height * channels];
		double output[width * height * channels]; // keep empty space at max size (just for the test)
		double golden_output[width * height  * channels];
		
		///////////////////////////////////////
		// INPUT AND EXPECTED OUTPUT SLICING //
		///////////////////////////////////////

		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < height ; y++)
				for(int x = 0 ; x < width ; x++){
		  			input[x + width * (y + z * height)] = i[x + N * (y + z * M)];
		  			golden_output[x + (width * (y + z * height ))] = golden_o[x + (N * (y + z * M ))];
		  		}
		  			
	  

	  // Call the main kernel, and measure cycles
	  
	  printf("Computing result...\n");
	  	start_timer();
		fReLu_tensor64(output, input, height, width, channels);
		stop_timer();
		
	  // Performance metrics
	  #ifndef SPIKE
	  int64_t runtime = get_timer();
	  float performance = 1.0 * channels * height * width / runtime;
	  float utilization = 100 * performance / (2.0 * NR_LANES);
	  #endif
	  
	  // Verify correctness
	  printf("Verifying result...\n");

	  int error = verify_tensor(output, golden_output, width, height, channels);


	  if (error != 0) {
		 printf("Fail.\n");
		 //return 0; //error;
	  } else {
		 printf("Passed.\n");
		 #ifndef SPIKE  
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  	 #endif
	  }
	}
  

  return 0; //error;
}
