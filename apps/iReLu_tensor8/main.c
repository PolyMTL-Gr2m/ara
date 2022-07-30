
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iReLu_tensor8.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

// Define Tensor dimensions:
// o = i ° f, with i=[MxNxL], f=[FxFxF], o=[MxNxL]
// The filter is a cube tensor, and F is odd
	// Matrices defined in data.S
extern int8_t i[] __attribute__((aligned(4 * NR_LANES))); // [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int8_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
extern int8_t golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
// M, N, F defined in data.S
extern int64_t M;
extern int64_t N; 
extern int64_t L; // depth

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


void print_8btensor(uint8_t const *tensor, uint64_t num_rows,
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
  printf("\n");
  printf("================\n");
  printf("=  ReLu   int8 =\n");
  printf("================\n");
  printf("\n");
  printf("------------------------------------------------\n");
  printf("Calculating activation (ReLu) of \n");
  printf("Input of [1 x %i x %i x %i]  \n", L, M, N);
  printf("------------------------------------------------\n");
	
  start_timer();
  iReLu_tensor8(o, i, M, N, L);
  stop_timer();
	
  // Performance metrics
  #ifndef SPIKE
  int64_t runtime = get_timer();
  float performance = 1.0 * L * M * N / runtime;
  float utilization = 100 * performance / (8 * 2.0 * NR_LANES);

  
  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f OP/cycle, the utilization is %f % \n", performance, utilization);
  #endif
    
  
   // Verify correctness
  printf("Verifying result...\n");

  int error = verify_8btensor(o, golden_o, M, N, L);

  if (error != 0) {
    printf("Fail.\n");
  } else {
    printf("Passed.\n");
  }
 
  
 //print_8btensor(o,M,N,L);
  

  return 0; //error;
}
