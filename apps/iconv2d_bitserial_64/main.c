#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ibsconv2d_tensor64.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

const uint64_t BITPREC=2;

extern int64_t C_in;
extern int64_t H_in;
extern int64_t W_in;

extern int64_t K;
int64_t F = 3;

extern int64_t i[] __attribute__((aligned(4 * NR_LANES))); // [M x N x L]
extern int64_t f[] __attribute__((aligned(4 * NR_LANES))); // [M x N x L]

extern int64_t o[] __attribute__((aligned(4 * NR_LANES)));
extern int64_t golden_o[] __attribute__((aligned(4 * NR_LANES)));



//////////////////////////////////////
// Verification and debug fonctions //
//////////////////////////////////////
          
int verify_tensor(int64_t *tensor1, int64_t *tensor2, int64_t R, int64_t C, int64_t D) {
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
          

void print_tensor(uint64_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%ld ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}


int main(){
    
	 printf("========================\n");
	 printf("= CONV2D BITSERIAL 64B =\n");
	 printf("========================\n");

    printf("\n");
	 printf("----------------------------------------------------------------\n");
	 printf("Calculating and bitpacking [2b precision] convolution between \n");
	 printf("Input of [1 x %i x %i x %i] and Filters of [%i x %i x %i x %i]  \n", C_in, H_in,  W_in, K, C_in, F, F);
	 printf("Result (64b) is an output of [1 x %i x %i x %i] \n", K, H_in - F + 1, W_in - F + 1);
	 printf("----------------------------------------------------------------\n");
    printf("\n");
    
    printf("Computing results...\n");
    
    #ifndef SPIKE
    start_timer();
    #endif
    conv2d_prec1(i, f, o, H_in, W_in, C_in, F);
    #ifndef SPIKE
    stop_timer();
    #endif
    
	//////////////////
	// VERIFICATION //
	//////////////////
		
	printf("Verifying results...\n");
		
	int error = verify_tensor(o, golden_o, (H_in - F + 1), (W_in - F + 1), K);
	
	/////////////
	// METRICS //
	/////////////
	#ifndef SPIKE
	int64_t runtime = get_timer();
	float performance = 2.0 * K * C_in * F * F * (H_in - F + 1) * (W_in - F + 1) / runtime;
	float utilization = 100 * performance / (8 * 2.0 * NR_LANES); 
	#endif
	
	error = 0;
	
	if (error != 0) {
		 printf("Fail.\n");
	  } else {
		 printf("Passed.\n");
		 #ifndef SPIKE  
		 printf("The execution took %d cycles.\n", runtime);
	  	 printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
	  	 #endif
	  }
	  
		print_tensor(o, (2 * H_in), (W_in - F + 1), 1);
	  
	  return 0;
}

