#include "fReLu_tensor64.h"
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


// Calculate 2 output matrix rows
void fReLu_tensor64(double *o, double *i, int64_t H_in, int64_t W_in, int64_t C_in) {

int64_t const size = H_in * W_in * C_in;

double comp = 0;

asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(TILE_SIZE));

	for (int c = 0 ; c < size ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
	
	  double *i_ = i + c;  // input pointer realtive to the tile (constant throughout the tile)
	  double *o_ = o + c;  // output pointer relative to the tile	
		
		
	  if(c > size - TILE_SIZE) 	// if we are at the right border of the input
				asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(size % TILE_SIZE));
	  
	  asm volatile("vle64.v v16,  (%0)" : "+&r"(i_));
	  
	  asm volatile("vfmax.vf v0,  v16,  %0" :: "f"(comp));
	  
	  asm volatile("vse64.v  v0,  (%0)" : "+&r"(o_));
	
	}

}


