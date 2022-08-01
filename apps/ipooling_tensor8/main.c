// Description : 
// Test and benchmark program meant for pooling functions on Ara VP

//	Author : Théo Dupuis - Polytechnique Montréal (2022)


#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ipooling_tensor8.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

#define I_MAX 32
#define F_MAX 3

#define AVG_POOL
#define MAX_POOL

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd
	// Matrices defined in data.S
extern int8_t i[] __attribute__((aligned(4 * NR_LANES))); // [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int8_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]

extern int8_t golden_o_max_2 [] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
extern int8_t golden_o_avg_2 [] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]

extern int8_t golden_o_max_3 [] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
extern int8_t golden_o_avg_3 [] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]


// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth


int verify_tensor(int8_t *tensor1, int8_t *tensor2, int64_t R,
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


void print_tensor(int8_t *tensor, uint64_t num_rows,
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

#ifdef MAX_POOL

  printf("\n");
  printf("================\n");
  printf("= MAXPOOL int8 =\n");
  printf("================\n");
  printf("\n");
  printf("Format is NCHW\n");
  printf("\n");
  
int8_t * golden_o_max;

for(int64_t F = 2 ; F <= F_MAX ; F += 1){

	if (F == 2)
		golden_o_max = golden_o_max_2;
	else if (F == 3)
		golden_o_max = golden_o_max_3;
		
printf("\nwindow is [%i x %i] \n", F, F);

  	for(int size = 8*F ; size <= I_MAX ; size = size * 2){
	  
	  printf("----------------------------------------------------------------\n");
	  printf("Calculating pooling between \n");
	  printf("Input of [1 x %i x %i x %i] \n", L, size, size);
	  printf("Result (8b) is an output of [1 x %i x %i x %i] \n", L, size / F, size / F);
	  printf("----------------------------------------------------------------\n");
	  
	  	int channels = L;
		int width = size;
		int height = size;  
		  
		int8_t input[width * height * channels];
		int8_t output[(width / F)* (height / F) * channels]; // keep empty space at max size (just for the test)
		int8_t golden_output[(width / F)* (height / F) * channels];
		
		///////////////////////////////////////
		// INPUT AND EXPECTED OUTPUT SLICING //
		///////////////////////////////////////

		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < height ; y++)
				for(int x = 0 ; x < width ; x++)
		  			input[x + width * (y + z * height)] = i[x + N * (y + z * M)];
		  			
		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < (height / F) ; y++)
				for(int x = 0 ; x < (width / F) ; x++)
		  			golden_output[x + (width / F) * (y + z * (height / F))] = golden_o_max[x + (N / F) * (y + z * (M / F))];
	  

	  // Call the main kernel, and measure cycles
	  
	  printf("Computing result...\n");
	  	start_timer();
		imax_pool8(output, input, height, width, channels, F);
		stop_timer();
		
		
	  // Performance metrics
	  #ifndef SPIKE
	  int64_t runtime = get_timer();
	  float performance = 1.0 * channels * height * width / runtime; // F * F * Cin * Hin / F * Win / F
	  float utilization = 100 * performance / (8 * 2.0 * NR_LANES);
	  #endif
	  
	  // Verify correctness
	  printf("Verifying result...\n");

	  int error = verify_tensor(output, golden_output, width / F, height / F, channels);


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
}

#endif

#ifdef AVG_POOL

  printf("\n");
  printf("================\n");
  printf("= AVGPOOL int8 =\n");
  printf("================\n");
  printf("\n");
  printf("Format is NCHW\n");
  printf("\n");
  
int8_t * golden_o_avg;
  
for(int64_t F = 2 ; F <= F_MAX ; F += 1){

	if (F == 2)
		golden_o_avg = golden_o_avg_2;
	else if (F == 3)
		golden_o_avg = golden_o_avg_3;
		
printf("\nwindow is [%i x %i] \n", F, F);
		
  	for(int size = 8*F ; size <= I_MAX ; size = size * 2){
	  
	  printf("----------------------------------------------------------------\n");
	  printf("Calculating pooling between \n");
	  printf("Input of [1 x %i x %i x %i] \n", L, size, size);
	  printf("Result (8b) is an output of [1 x %i x %i x %i] \n", L, size / F, size / F);
	  printf("----------------------------------------------------------------\n");
	  
	  	int channels = L;
		int width = size;
		int height = size;  
		  
		int8_t input[width * height * channels];
		int8_t output[(width / F)* (height / F) * channels]; // keep empty space at max size (just for the test)
		int8_t golden_output[(width / F)* (height / F) * channels];
		
		///////////////////////////////////////
		// INPUT AND EXPECTED OUTPUT SLICING //
		///////////////////////////////////////

		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < height ; y++)
				for(int x = 0 ; x < width ; x++)
		  			input[x + width * (y + z * height)] = i[x + N * (y + z * M)];
		  			
		for(int z = 0; z < channels ; z++)
			for(int y = 0 ; y < (height / F) ; y++)
				for(int x = 0 ; x < (width / F) ; x++)
		  			golden_output[x + (width / F) * (y + z * (height / F))] = golden_o_avg[x + (N / F) * (y + z * (M / F))];
	  

	  // Call the main kernel, and measure cycles
	  
	  printf("Computing result...\n");
	  	start_timer();
		iavg_pool8(output, input, height, width, channels, F);
		stop_timer();
		
		//if(F == 3)
		//print_tensor(output, height / F, width / F, channels);
		
	  // Performance metrics
	  #ifndef SPIKE
	  int64_t runtime = get_timer();
	  float performance = 2.0 * channels * height * width / runtime;
	  float utilization = 100 * performance / (8 * 2.0 * NR_LANES);
	  #endif
	  
	  // Verify correctness
	  printf("Verifying result...\n");

	  int error = verify_tensor(output, golden_output, width / F, height / F, channels);


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
}

#endif
  
  return 0; //error;
}
