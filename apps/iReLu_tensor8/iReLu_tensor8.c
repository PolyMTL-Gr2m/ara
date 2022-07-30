#include "iReLu_tensor8.h"
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                            Description : Apply ReLu function to a tensor                                    //		
//                                                                                                             //				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o : tensor output pointer C_in x H_in x W_in
// *i : input tensor pointer  C_in x H_in x W_in
// H_in  : number of input rows
// W_in  : number of input column
// C_in  : number of input channels

void iReLu_tensor8(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W) {

  iReLu_tensor8_vec_8xC(o, i, R, C, W);
 
}

// Calculate 2 output matrix rows
void iReLu_tensor8_vec_8xC(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W) {
  
  // Helper variables
  int64_t const ld = TILE_SIZE; 					// increment adress on output stores and input loads
  int64_t const size = C * R * W % TILE_SIZE; 	// check for tiling

  // Compute on C elements
  
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(TILE_SIZE));
  
  for (int c = 0 ; c <= R * C * W - TILE_SIZE ; c+= TILE_SIZE){ // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
			  
	asm volatile("vle8.v v0,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ld));
	asm volatile("vmax.vx v0,  v0,  zero");
				  
	// STORE THE ACTIVATED REGISTERS OUTPUT

	asm volatile("vse8.v  v0,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ld));
			
	}
	
	if(size != 0){
	
		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(size));
		
		asm volatile("vle8.v v0,  (%0)": "+&r"(i));
		asm volatile("vmax.vx v0,  v0,  zero");
		asm volatile("vse8.v  v0,  (%0)" : "+&r"(o));
	}
	
	// LAST GROUP
}


