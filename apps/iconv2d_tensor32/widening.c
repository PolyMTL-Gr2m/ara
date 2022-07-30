#include "iconv2d_tensor32.h"
#include <stdio.h>

#define next_plane_(a) ((R - a + 1)*C) << 2

#define block_size_wide_1x1 4
#define block_size_wide_3x3 4
#define block_size_wide_5x5 8
#define block_size_wide_7x7 6

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                    //			
//                          input (32b)            kernels (32b)             output (32b)                      //	
//																																					//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o    : tensor convolution output pointer 
// *i    : input tensor pointer
// *f    : kernel/filter tensor pointer
// Hin   : number of input cows
// Win   : number of input column
// Cin   : number of input channels 
// F     : size of the kernel/filter 
// Cout  : number of kernel/filter corresponding to the number of output channels


// In the functions, this notation is used:

// R is Rows (H_in)
// C is Column (W_in)
// W is Channels (C_in)
// F is Filter (F)

// 3x3, 5x5 and 7x7 kernel are tiled in the functions, the tile size is configurable in the header
// 1x1 kernel has its own tile size that is also configurable


void iconv2d_tensor32_wide(int64_t *o, int32_t *i, int32_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out) {
	
  int32_t *i_;
  int64_t *o_;
  int32_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c * (H_in - F + 1) * (W_in - F + 1);      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;

	  // Iterate over the output rows
	  

		switch (F){
			 case 1:
					iconv2d_tensor32_vec_4xC_1x1_wide(o_, i_, f_, H_in, W_in, C_in, F);
				break;

			 case 3:
					iconv2d_tensor32_vec_4xC_3x3_wide(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
			 case 5:
					iconv2d_tensor32_vec_8xC_5x5_wide(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
			 case 7:
					iconv2d_tensor32_vec_6xC_7x7_wide(o_, i_, f_, H_in, W_in, C_in, F);
				break;
		}
		 
		 
	}
}


//////////////////////////////////////////////////////////////////////////
//																								//
//											1x1 kernel										//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor32_vec_4xC_1x1_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
	int64_t const ldo = C << 3;
	int64_t const ldi = C << 2;

	int64_t vlen;

	int64_t const last_group = R % block_size_wide_1x1;
	
for (int c = 0 ; c < C ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	
		int32_t *f_ = f;
		int32_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
		int64_t *o_ = o + c;									// output pointer relative to the tile		
		
		int32_t t[1];
	

		
		if(c > C - TILE_SIZE) 	// if we are at the right border of the input
			vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
		else
			vlen = TILE_SIZE;						// else we go full length
		
		int32_t * i__ = i_;							// inside tile pointer (change at each load)
		  	  	
	
	for (int r = block_size_wide_1x1; r < R ; r += block_size_wide_1x1) // Loop until the last group
	{
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));


	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	asm volatile("vwmul.vx v0,  v16, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v4,  v18, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v8,  v20, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v12,  v22, %0" :: "r"(t[0]));


	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));

		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		asm volatile("vwmacc.vx v0,  %0, v16" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v18" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v22" ::"r"(t[0]));

		}

	asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen));

	i__ = i_ + r * C;
	f_ = f;
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));


	}
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vwmul.vx v0,  v16, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v4,  v18, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v8,  v20, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v12,  v22, %0" :: "r"(t[0]));




	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));

		if (last_group == 0)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vwmacc.vx v0,  %0, v16" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v18" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v22" ::"r"(t[0]));



		}

	asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v12, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse64.v v0, (%0)" : "+&r"(o_));

	}
	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                3x3 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor32_vec_4xC_3x3_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
	int64_t const ldo = (C - 2) << 3;
	int64_t const ldi = C << 2;

	int64_t vlen;

	int64_t const last_group = (R - F + 1) % block_size_wide_3x3;
	
for (int c = 0 ; c < (C - 2) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	
		int32_t *f_ = f;
		int32_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
		int64_t *o_ = o + c;									// output pointer relative to the tile		
		
		int32_t t[3];
	

		
		if(c > C - TILE_SIZE) 	// if we are at the right border of the input
			vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
		else
			vlen = TILE_SIZE;						// else we go full length
		
		int32_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 2;
		  	  	
	
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ -= 2; 


	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	asm volatile("vwmul.vx v0,  v24, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v4,  v26, %0" :: "r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));

	asm volatile("vslidedown.vi v26, v26, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ -= 2;


	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 3;

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	
	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));


	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ -= 2;

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));


		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ -= 2; 

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));

		asm volatile("vslidedown.vi v26, v26, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[1]));

		}

	i__ = i_ + 2 * C;
	f_ = f;
	
	for (int r = 2 + block_size_wide_3x3; r < R ; r += block_size_wide_3x3)
	{

		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ -= 5;

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v8,  v24, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v12,  v26, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		asm volatile("vwmul.vx v16,  v28, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmul.vx v20,  v30, %0" :: "r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ -= 5;

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));


		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));


		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));


		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 1;

			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 3;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 3;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ -= 5;

			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 3;
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 3;
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ -= 5;

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 3;
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 3;
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));


			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));


			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));


			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));


			}

		asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen_out));


			i__ = i_ + r * C;
			f_ = f;
		  	
		asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

		asm volatile("vmv.v.v v0, v16");
		asm volatile("vmv.v.v v4, v20");

		}

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 3;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ -= 5;

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	
	
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmul.vx v8,  v28, %0" :: "r"(t[2]));
	asm volatile("vwmul.vx v12,  v30, %0" :: "r"(t[2]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 3;

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ -= 5;

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 3;

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 3;

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));

	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));


	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ -= 5;

		if (last_group == 0)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ -= 5;

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 3;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 3;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[2]));

		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[1]));

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));

		}

	asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v12, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse64.v v0, (%0)" : "+&r"(o_));

	}
	}
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                5x5 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor32_vec_8xC_5x5_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
	int64_t const ldo = (C - 4) << 3;
	int64_t const ldi = C << 2;

	int64_t vlen;

	int64_t const last_group = (R - F + 1) % block_size_wide_5x5;
	
for (int c = 0 ; c < (C - 4) ; c += TILE_SIZE_WIDE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE_WIDE)
{

	
		int32_t *f_ = f;
		int32_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
		int64_t *o_ = o + c;									// output pointer relative to the tile		
		
		int32_t t[5];
	

		
		if(c > C - TILE_SIZE_WIDE) 	// if we are at the right border of the input
			vlen = C % TILE_SIZE_WIDE_OUT;		 	// we set the vector length to fit the last inputs
		else
			vlen = TILE_SIZE_WIDE;						// else we go full length
		
		int32_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 4;
		  	  	
	
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ -= 14;


	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

	asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	asm volatile("vwmul.vx v0,  v20, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v2,  v21, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v4,  v22, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v6,  v23, %0" :: "r"(t[0]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

	asm volatile("vslidedown.vi v21, v21, 1");

	asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

	asm volatile("vslidedown.vi v23, v23, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ -= 14;

	asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

	asm volatile("vslidedown.vi v21, v21, 1");

	asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

	asm volatile("vslidedown.vi v23, v23, 1");

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ -= 14;

	asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

	asm volatile("vslidedown.vi v21, v21, 1");

	asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));
	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

	asm volatile("vslidedown.vi v23, v23, 1");

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ -= 14;

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

	asm volatile("vslidedown.vi v21, v21, 1");

	asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));
	
	asm volatile("vslidedown.vi v23, v23, 1");

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;

	asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

	asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

	asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

	asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ -= 14;

		asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

		asm volatile("vslidedown.vi v21, v21, 1");

		asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

		asm volatile("vslidedown.vi v23, v23, 1");

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ -= 14;
		
		asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

		asm volatile("vslidedown.vi v21, v21, 1");

		asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));
		
		asm volatile("vslidedown.vi v23, v23, 1");

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ -= 14;

		asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

		asm volatile("vslidedown.vi v21, v21, 1");

		asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

		asm volatile("vslidedown.vi v23, v23, 1");

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ -= 14;

		asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));
		
		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

		asm volatile("vslidedown.vi v21, v21, 1");

		asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));

		asm volatile("vslidedown.vi v23, v23, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;


		asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t[0]));

		asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t[1]));

		asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t[2]));
		
		asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t[3]));


		}

	i__ = i_ + 4 * C;
	f_ = f;

	for (int r = 4 + block_size_wide_5x5; r < R ; r += block_size_wide_5x5)
	{

		asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v8,  v24, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v10,  v25, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v12,  v26, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v14,  v27, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v16,  v28, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v18,  v29, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vle32.v v31, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(8)));

		asm volatile("vwmul.vx v20,  v30, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmul.vx v22,  v31, %0" :: "r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v31, v31, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v31, v31, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v31, v31, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));

		asm volatile("vslidedown.vi v31, v31, 1");

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));


		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));


		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));


		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));


		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));


		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));


		asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));


		asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
		asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
		asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
		asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 1;

			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 5;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 5;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 5;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 5;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ -= 19;

			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vle32.v v31, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(8)));

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 5;
			asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 5;
			asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 5;
			asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 5;
			asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ -= 19;

			asm volatile("vslidedown.vi v31, v31, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 5;
			asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 5;
			asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 5;
			asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 5;
			asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ -= 19;

			asm volatile("vslidedown.vi v31, v31, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 5;
			asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 5;
			asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 5;
			asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 5;
			asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ -= 19;

			asm volatile("vslidedown.vi v31, v31, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 5;
			asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 5;
			asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 5;
			asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 5;
			asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));

			asm volatile("vslidedown.vi v31, v31, 1");

			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));


			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));


			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));


			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));


			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));


			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));


			asm volatile("vwmacc.vx v20,  %0, v30" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v30" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v30" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));


			asm volatile("vwmacc.vx v22,  %0, v31" ::"r"(t[0]));
			asm volatile("vwmacc.vx v20,  %0, v31" ::"r"(t[1]));
			asm volatile("vwmacc.vx v18,  %0, v31" ::"r"(t[2]));
			asm volatile("vwmacc.vx v16,  %0, v31" ::"r"(t[3]));
			asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));


			}

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(vlen_out));


			i__ = i_ + r * C;
			f_ = f;
		  	
		asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v14, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

		asm volatile("vmv.v.v v0, v16");
		asm volatile("vmv.v.v v2, v18");
		asm volatile("vmv.v.v v4, v20");
		asm volatile("vmv.v.v v6, v22");

		}

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ -= 19;

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v31, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(8)));

	}
	else if (last_group == 7)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(7)));

	}
	else if (last_group == 6)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	}
	else if (last_group == 5)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

	}
	else if (last_group == 4)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
	asm volatile("vwmul.vx v8,  v28, %0" :: "r"(t[4]));
	asm volatile("vwmul.vx v10,  v29, %0" :: "r"(t[4]));
	asm volatile("vwmul.vx v12,  v30, %0" :: "r"(t[4]));
	asm volatile("vwmul.vx v14,  v31, %0" :: "r"(t[4]));

	asm volatile("vslidedown.vi v31, v31, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
	asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ -= 19;

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
	asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

	asm volatile("vslidedown.vi v31, v31, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
	asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ -= 19;

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
	asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

	asm volatile("vslidedown.vi v31, v31, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
	asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ -= 19;

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
	asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

	asm volatile("vslidedown.vi v31, v31, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
	asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 5;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
	asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
	asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
	asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));




	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		if (last_group == 0)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v31, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(8)));

		}
		else if (last_group == 7)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(7)));

		}
		else if (last_group == 6)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		}
		else if (last_group == 5)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

		}
		else if (last_group == 4)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

		asm volatile("vslidedown.vi v31, v31, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

		asm volatile("vslidedown.vi v31, v31, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

		asm volatile("vslidedown.vi v31, v31, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ -= 19;

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

		asm volatile("vslidedown.vi v31, v31, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 5;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v30" ::"r"(t[4]));
		asm volatile("vwmacc.vx v14,  %0, v31" ::"r"(t[4]));

		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v30" ::"r"(t[3]));

		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[2]));

		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[1]));

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[0]));



		}

	asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v14, (%0)" : "+&r"(o_));

	}
	else if (last_group == 7)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v12, (%0)" : "+&r"(o_));

	}
	else if (last_group == 6)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v10, (%0)" : "+&r"(o_));

	}
	else if (last_group == 5)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 4)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse64.v v0, (%0)" : "+&r"(o_));

	}
	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                7x7 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor32_vec_6xC_7x7_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
	int64_t const ldo = (C - 6) << 3;
	int64_t const ldi = C << 2;

	int64_t vlen;

	int64_t const last_group = (R - F + 1) % block_size_wide_7x7;
	
for (int c = 0 ; c < (C - 6) ; c += TILE_SIZE_WIDE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE_WIDE)
{

	
		int32_t *f_ = f;
		int32_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
		int64_t *o_ = o + c;									// output pointer relative to the tile		
		
		int32_t t[7];
	

		
		if(c > C - TILE_SIZE_WIDE) 	// if we are at the right border of the input
			vlen = C % TILE_SIZE_WIDE_OUT;		 	// we set the vector length to fit the last inputs
		else
			vlen = TILE_SIZE_WIDE;						// else we go full length
		
		int32_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;
		  	  	
	
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;


	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

	asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	asm volatile("vwmul.vx v0,  v24, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v2,  v25, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v4,  v26, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v6,  v27, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v8,  v28, %0" :: "r"(t[0]));
	asm volatile("vwmul.vx v10,  v29, %0" :: "r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;
	
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ -= 34;
	
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vslidedown.vi v25, v25, 1");

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vslidedown.vi v27, v27, 1");

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vslidedown.vi v28, v28, 1");

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

	asm volatile("vslidedown.vi v29, v29, 1");


	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

	asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
	asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
	asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
	asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
	asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

	asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
	asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
	asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
	asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

	asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
	asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
	asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

	asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
	asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

	asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));



	for(int ch = 1 ; ch < W ; ch ++){
		
		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;
		
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;
		

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ -= 34;

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));

		asm volatile("vslidedown.vi v29, v29, 1");


		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[0]));

		asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v6,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v29" ::"r"(t[1]));

		asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v4,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v29" ::"r"(t[2]));

		asm volatile("vwmacc.vx v0,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v2,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v29" ::"r"(t[3]));

		asm volatile("vwmacc.vx v0,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v29" ::"r"(t[4]));

		asm volatile("vwmacc.vx v0,  %0, v29" ::"r"(t[5]));


		}
		
	i__ = i_ + 6 * C;
	f_ = f;


	for (int r = 6 + block_size_wide_7x7; r < R ; r += block_size_wide_7x7)
	{

		asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v12,  v24, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v14,  v25, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v16,  v26, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vwmul.vx v18,  v27, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vwmul.vx v20,  v28, %0" :: "r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmul.vx v22,  v29, %0" :: "r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

		asm volatile("vslidedown.vi v25, v25, 1");

		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

		asm volatile("vslidedown.vi v27, v27, 1");

		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");

		asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));


		asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
		asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));


		asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
		asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
		asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));


		asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
		asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
		asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
		asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));


		asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
		asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
		asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
		asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
		asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));


		asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
		asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
		asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
		asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
		asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
		asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 1;

			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
			f_ -= 41;

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));

			asm volatile("vslidedown.vi v25, v25, 1");

			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));

			asm volatile("vslidedown.vi v27, v27, 1");

			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
			f_ += 7;
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
			f_ += 7;
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
			f_ += 7;
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
			f_ += 7;
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
			f_ += 7;
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
			f_ += 7;
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));
			asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));

			asm volatile("vslidedown.vi v29, v29, 1");

			asm volatile("vwmacc.vx v12,  %0, v24" ::"r"(t[0]));
			asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));
			asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
			asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
			asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
			asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
			asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));


			asm volatile("vwmacc.vx v14,  %0, v25" ::"r"(t[0]));
			asm volatile("vwmacc.vx v12,  %0, v25" ::"r"(t[1]));
			asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));
			asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
			asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
			asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
			asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));


			asm volatile("vwmacc.vx v16,  %0, v26" ::"r"(t[0]));
			asm volatile("vwmacc.vx v14,  %0, v26" ::"r"(t[1]));
			asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t[2]));
			asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));
			asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
			asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
			asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));


			asm volatile("vwmacc.vx v18,  %0, v27" ::"r"(t[0]));
			asm volatile("vwmacc.vx v16,  %0, v27" ::"r"(t[1]));
			asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t[2]));
			asm volatile("vwmacc.vx v12,  %0, v27" ::"r"(t[3]));
			asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));
			asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
			asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));


			asm volatile("vwmacc.vx v20,  %0, v28" ::"r"(t[0]));
			asm volatile("vwmacc.vx v18,  %0, v28" ::"r"(t[1]));
			asm volatile("vwmacc.vx v16,  %0, v28" ::"r"(t[2]));
			asm volatile("vwmacc.vx v14,  %0, v28" ::"r"(t[3]));
			asm volatile("vwmacc.vx v12,  %0, v28" ::"r"(t[4]));
			asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));
			asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));


			asm volatile("vwmacc.vx v22,  %0, v29" ::"r"(t[0]));
			asm volatile("vwmacc.vx v20,  %0, v29" ::"r"(t[1]));
			asm volatile("vwmacc.vx v18,  %0, v29" ::"r"(t[2]));
			asm volatile("vwmacc.vx v16,  %0, v29" ::"r"(t[3]));
			asm volatile("vwmacc.vx v14,  %0, v29" ::"r"(t[4]));
			asm volatile("vwmacc.vx v12,  %0, v29" ::"r"(t[5]));
			asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));


			}

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(vlen_out));


			i__ = i_ + r * C;
			f_ = f;
		  	
		asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse64.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		
		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");
		asm volatile("vmv.v.v v4, v16");
		asm volatile("vmv.v.v v6, v18");
		asm volatile("vmv.v.v v8, v20");
		asm volatile("vmv.v.v v10, v22");


		}

	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	}
	else if (last_group == 5)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

	}
	else if (last_group == 4)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
	f_ -= 41;

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vslidedown.vi v29, v29, 1");
	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vslidedown.vi v27, v27, 1");
	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vslidedown.vi v25, v25, 1");
	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

	asm volatile("vslidedown.vi v24, v24, 1");

	asm volatile("vslidedown.vi v23, v23, 1");
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
	f_ += 7;
	asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
	asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
	asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
	asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
	asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
	asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

	asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
	asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
	asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
	asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
	asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

	asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
	asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
	asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
	asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

	asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
	asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
	asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

	asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
	asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

	asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));





	for(int c = 1 ; c < W ; c ++){

		f_ += 1;

		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		if (last_group == 0)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v29, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		}
		else if (last_group == 5)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

		}
		else if (last_group == 4)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle32.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));
		f_ -= 41;

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vslidedown.vi v29, v29, 1");
		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vslidedown.vi v27, v27, 1");
		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vslidedown.vi v25, v25, 1");
		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vslidedown.vi v23, v23, 1");
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[0]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[1]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[2]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[3]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[4]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[5]));
		f_ += 7;
		asm volatile("lw %1, (%0)" : "+&r"(f_), "=&r"(t[6]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t[6]));
		asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t[6]));
		asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t[6]));
		asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t[6]));
		asm volatile("vwmacc.vx v8,  %0, v28" ::"r"(t[6]));
		asm volatile("vwmacc.vx v10,  %0, v29" ::"r"(t[6]));

		asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t[5]));
		asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t[5]));
		asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t[5]));
		asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t[5]));
		asm volatile("vwmacc.vx v10,  %0, v28" ::"r"(t[5]));

		asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t[4]));
		asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t[4]));
		asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t[4]));
		asm volatile("vwmacc.vx v10,  %0, v27" ::"r"(t[4]));

		asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t[3]));
		asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t[3]));
		asm volatile("vwmacc.vx v10,  %0, v26" ::"r"(t[3]));

		asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t[2]));
		asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t[2]));

		asm volatile("vwmacc.vx v10,  %0, v24" ::"r"(t[1]));




		}

	asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v10, (%0)" : "+&r"(o_));

	}
	else if (last_group == 5)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 4)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v6, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v4, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse64.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse64.v v2, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse64.v v0, (%0)" : "+&r"(o_));

	}
	}
}

