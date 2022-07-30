#include "iconv2d_tensor8.h"
#include <stdio.h>

#define next_plane_(a) (R - a + 1)*C

#define TILE_SIZE_1x1 4096

#define block_size_3x3 6
#define block_size_5x5 6
#define block_size_7x7 4


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                      //			
//                          input              kernels                       output                            //	
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


void iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out) {
	
  int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c * (H_in - F + 1) * (W_in - F + 1);      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;

	  // Iterate over the output rows
	  

		switch (F){
			 case 1:
					iconv2d_tensor8_vec_1xC_1x1(o_, i_, f_, H_in, W_in, C_in, F);
				break;

			 case 3:
					iconv2d_tensor8_vec_4xC_3x3(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
			 case 5:
					iconv2d_tensor8_vec_6xC_5x5(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
			 case 7:
					iconv2d_tensor8_vec_4xC_7x7(o_, i_, f_, H_in, W_in, C_in, F);
				break;
		}
		 
		 
	}
}



//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                1x1 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// 1x1 convolution is straight forward
// Each line (per block of 16) is multiplied by the kernel value
// Then we switch to the next plane (regarding depth) and do the same
// ...
// When every plane has been computed, we go back to the first one and do the next 16 block

/*

	 🡕 depth direction 
	▣▣▣▣...▣▣▣▣		On the first plane, we load 16 rows and multiply them by the kernel
	...
	▣▣▣▣...▣▣▣▣	
	▣▣▣▣...▣▣▣▣			
	▢▢▢▢...▢▢▢▢		Then when we get back to the first plane, we start at this line
	▢▢▢▢...▢▢▢▢
	▢▢▢▢...▢▢▢▢
	...
	▢▢▢▢...▢▢▢▢↑ row direction (R)
	⟶ column (C)
	
*/

// Calculate 16 output matrix rows
void iconv2d_tensor8_vec_1xC_1x1(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{

int64_t vlen;

for (int c = 0 ; c < C * R ; c += TILE_SIZE_1x1) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *f_ = f;
	int8_t *o_ = o + c;									// output pointer relative to the tile		
	
	
	if(c > (C * R) - TILE_SIZE_1x1) 	// if we are at the right border of the input
	{
		vlen = (C * R) % TILE_SIZE_1x1;		 	// we set the vector length to fit the last inputs
	}
	else
	{
		vlen = TILE_SIZE_1x1;						// else we go full length
	}
	

	asm volatile("vsetvli zero, %0, e8, m8, ta, ma" ::"r"(vlen));
	
	asm volatile("vle8.v v16, (%0)" : "+&r"(i_));
		
	asm volatile("vmul.vx v0,  v16, %0" :: "r"(f_[0]));
	


	for(int ch = 1 ; ch < W ; ch ++){
		
		i_ += R*C;
		f_ += 1;
		
		asm volatile("vle8.v v16, (%0)" : "+&r"(i_));
		
		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[0]));

		}
	
	asm volatile("vse8.v v0,  (%0)" : "+&r"(o_));
	o_ += vlen;

	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                3x3 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor8_vec_4xC_3x3(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
int64_t const ldo = (C - 2);
int64_t const ldi = C;

int64_t vlen;

int64_t const last_group = (R - F + 1) % block_size_3x3;

for (int c = 0 ; c < (C - 2) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	
	int8_t *f_ = f;
	int8_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o + c;									// output pointer relative to the tile		


	
	if(c > C - TILE_SIZE) 	// if we are at the right border of the input
		vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length
	
	int8_t * i__ = i_;							// inside tile pointer (change at each load)
	
	int64_t vlen_out = vlen - 2;
	  	  	


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	asm volatile("vmul.vx v0,  v16, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v2,  v18, %0" :: "r"(f_[0]));

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[3]));

	asm volatile("vslidedown.vi v16, v16, 1");
	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[1]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[1]));

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[4]));

	asm volatile("vslidedown.vi v16, v16, 1");
	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[2]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[2]));

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[5]));

	for(int c = 1 ; c < W ; c ++){

		f_ += 9;

		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[0]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[0]));

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[3]));

		asm volatile("vslidedown.vi v16, v16, 1");
		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[1]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[1]));

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[4]));

		asm volatile("vslidedown.vi v16, v16, 1");
		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[2]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[2]));

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[5]));

		}


	i__ = i_ + 2 * C;
	f_ = f;
	
	
	for (int r = 2 + block_size_3x3; r < R ; r += block_size_3x3)
	{

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v4,  v16, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[3]));
		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v6,  v18, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[3]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v8,  v20, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v10,  v22, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[3]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vmul.vx v12,  v24, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[3]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmul.vx v14,  v26, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[4]));
		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[4]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[4]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[4]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[1]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[5]));
		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[8]));


		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[5]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[8]));


		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[8]));


		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[8]));


		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[8]));


		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[2]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[8]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 9;

			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[0]));
			asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[3]));
			asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v16, v16, 1");

			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[0]));
			asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[3]));
			asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v18, v18, 1");

			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[3]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[3]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[0]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[3]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[0]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[3]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[6]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
			asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[4]));
			asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v16, v16, 1");

			asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
			asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[4]));
			asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v18, v18, 1");

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[4]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[4]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[1]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[4]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[1]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[4]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[7]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
			asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[5]));
			asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[8]));


			asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
			asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[5]));
			asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[8]));


			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[8]));


			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[8]));


			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[2]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[8]));


			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[2]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[5]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[8]));


			}

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));


		i__ = i_ + r * C;
		f_ = f;
		  	
		asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");

		}


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	}
	else if (last_group == 5)
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

	}
	else if (last_group == 4)
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[6]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[6]));
	asm volatile("vmul.vx v4,  v20, %0" :: "r"(f_[6]));
	asm volatile("vmul.vx v6,  v22, %0" :: "r"(f_[6]));
	asm volatile("vmul.vx v8,  v24, %0" :: "r"(f_[6]));
	asm volatile("vmul.vx v10,  v26, %0" :: "r"(f_[6]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[3]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[3]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[3]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[3]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[3]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[0]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[0]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");
	asm volatile("vslidedown.vi v18, v18, 1");
	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[7]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[7]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[7]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[7]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[7]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[7]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[4]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[4]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[4]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[4]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[4]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");
	asm volatile("vslidedown.vi v18, v18, 1");
	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[8]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[8]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[8]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[8]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[8]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[8]));

	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[5]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[5]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));

	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));



	for(int c = 1 ; c < W ; c ++){

		f_ += 9;

		if (last_group == 0)
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		}
		else if (last_group == 5)
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

		}
		else if (last_group == 4)
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[6]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[6]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[6]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[6]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[3]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[3]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[3]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[0]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[0]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");
		asm volatile("vslidedown.vi v18, v18, 1");
		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[7]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[7]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[7]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[7]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[4]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[4]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[4]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");
		asm volatile("vslidedown.vi v18, v18, 1");
		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[8]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[8]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[8]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[8]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[8]));

		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[5]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[5]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));

		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));


		}

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v10, (%0)" : "+&r"(o_));

	}
	else if (last_group == 5)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v8, (%0)" : "+&r"(o_));

	}
	else if (last_group == 4)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse8.v v0, (%0)" : "+&r"(o_));

	}
	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                5x5 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor8_vec_6xC_5x5(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
int64_t const ldo = (C - 4);
int64_t const ldi = C;

int64_t vlen;

int64_t const last_group = (R - F + 1) % block_size_5x5;

for (int c = 0 ; c < (C - 4) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	
	int8_t *f_ = f;
	int8_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o + c;									// output pointer relative to the tile		


	
	if(c > C - TILE_SIZE) 	// if we are at the right border of the input
		vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length
	
	int8_t * i__ = i_;							// inside tile pointer (change at each load)
	
	int64_t vlen_out = vlen - 4;
	  	  	


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	asm volatile("vmul.vx v0,  v16, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v2,  v18, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v4,  v20, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v6,  v22, %0" :: "r"(f_[0]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[5]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[5]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[5]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[10]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[10]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[15]));

	asm volatile("vslidedown.vi v22, v22, 1");


	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[1]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[1]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[1]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[1]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[6]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[6]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[6]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[11]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[11]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[16]));

	asm volatile("vslidedown.vi v22, v22, 1");


	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[2]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[2]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[2]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[2]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[7]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[7]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[7]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[12]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[12]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[17]));

	asm volatile("vslidedown.vi v22, v22, 1");


	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[3]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[3]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[3]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[3]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[8]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[8]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[8]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[13]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[13]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[18]));

	asm volatile("vslidedown.vi v22, v22, 1");


	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vslidedown.vi v26, v26, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[4]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[4]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[4]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[4]));

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[9]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[9]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[9]));

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[14]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[14]));

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[19]));



	for(int c = 1 ; c < W ; c ++){

		f_ += 25;

		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[0]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[0]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[0]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[0]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[5]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[5]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[10]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[10]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[15]));

		asm volatile("vslidedown.vi v22, v22, 1");


		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[1]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[1]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[1]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[6]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[11]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[11]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[16]));

		asm volatile("vslidedown.vi v22, v22, 1");


		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[2]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[2]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[2]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[7]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[12]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[12]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[17]));

		asm volatile("vslidedown.vi v22, v22, 1");


		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[3]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[3]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[3]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[8]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[8]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[13]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[13]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[18]));

		asm volatile("vslidedown.vi v22, v22, 1");


		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[4]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[4]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[4]));

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[9]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[9]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[9]));

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[14]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[14]));

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[19]));


		}


	i__ = i_ + 4 * C;
	f_ = f;

	for (int r = 4 + block_size_5x5; r < R ; r += block_size_5x5)
	{

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v8,  v20, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[10]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[15]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v10,  v22, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[10]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[15]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v12,  v24, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[10]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[15]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v14,  v26, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[10]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[15]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vmul.vx v16,  v28, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[5]));
		asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[10]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[15]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vmul.vx v18,  v30, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[5]));
		asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[10]));
		asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[15]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[11]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[16]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[6]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[11]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[16]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[6]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[11]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[16]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[1]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[6]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[11]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[16]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[1]));
		asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[6]));
		asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[11]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[16]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[1]));
		asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[6]));
		asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[11]));
		asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[16]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[12]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[17]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[12]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[17]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[7]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[12]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[17]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[2]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[7]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[12]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[17]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[2]));
		asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[7]));
		asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[12]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[17]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[2]));
		asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[7]));
		asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[12]));
		asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[17]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[13]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[18]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[13]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[18]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[8]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[13]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[18]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[3]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[8]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[13]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[18]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[3]));
		asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[8]));
		asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[13]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[18]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v28, v28, 1");

		asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[3]));
		asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[8]));
		asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[13]));
		asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[18]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v30, v30, 1");

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[19]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[24]));


		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[19]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[24]));


		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[9]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[14]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[19]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[24]));


		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[4]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[9]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[14]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[19]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[24]));


		asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[4]));
		asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[9]));
		asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[14]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[19]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[24]));


		asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[4]));
		asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[9]));
		asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[14]));
		asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[19]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[24]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 25;

			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[10]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[15]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[10]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[15]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[0]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[10]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[15]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[0]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[5]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[10]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[15]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

			asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[0]));
			asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[5]));
			asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[10]));
			asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[15]));
			asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[0]));
			asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[5]));
			asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[10]));
			asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[15]));
			asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[20]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[6]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[11]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[16]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[6]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[11]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[16]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[1]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[6]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[11]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[16]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[1]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[6]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[11]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[16]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[1]));
			asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[6]));
			asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[11]));
			asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[16]));
			asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[1]));
			asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[6]));
			asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[11]));
			asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[16]));
			asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[21]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[12]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[17]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[12]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[17]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[2]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[7]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[12]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[17]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[2]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[7]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[12]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[17]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[2]));
			asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[7]));
			asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[12]));
			asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[17]));
			asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[2]));
			asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[7]));
			asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[12]));
			asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[17]));
			asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[22]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[13]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[18]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[13]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[18]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[3]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[8]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[13]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[18]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[3]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[8]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[13]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[18]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[3]));
			asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[8]));
			asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[13]));
			asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[18]));
			asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v28, v28, 1");

			asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[3]));
			asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[8]));
			asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[13]));
			asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[18]));
			asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[23]));

			asm volatile("vslidedown.vi v30, v30, 1");

			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[19]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[24]));


			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[19]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[24]));


			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[4]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[9]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[14]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[19]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[24]));


			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[4]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[9]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[14]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[19]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[24]));


			asm volatile("vmacc.vx v16,  %0, v28" ::"r"(f_[4]));
			asm volatile("vmacc.vx v14,  %0, v28" ::"r"(f_[9]));
			asm volatile("vmacc.vx v12,  %0, v28" ::"r"(f_[14]));
			asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[19]));
			asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[24]));


			asm volatile("vmacc.vx v18,  %0, v30" ::"r"(f_[4]));
			asm volatile("vmacc.vx v16,  %0, v30" ::"r"(f_[9]));
			asm volatile("vmacc.vx v14,  %0, v30" ::"r"(f_[14]));
			asm volatile("vmacc.vx v12,  %0, v30" ::"r"(f_[19]));
			asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[24]));


			}

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));


		i__ = i_ + r * C;
		f_ = f;
	  	
		asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");
		asm volatile("vmv.v.v v4, v16");
		asm volatile("vmv.v.v v6, v18");

		}


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	}
	else if (last_group == 5)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

	}
	else if (last_group == 4)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[20]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[20]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[20]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[20]));
	asm volatile("vmul.vx v8,  v28, %0" :: "r"(f_[20]));
	asm volatile("vmul.vx v10,  v30, %0" :: "r"(f_[20]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[15]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[15]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[15]));
	asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[15]));
	asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[15]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[10]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[10]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[10]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[10]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[21]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[21]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[21]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[21]));
	asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[21]));
	asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[21]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[16]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[16]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[16]));
	asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[16]));
	asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[16]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[11]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[11]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[11]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[11]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[6]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[6]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[6]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[22]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[22]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[22]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[22]));
	asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[22]));
	asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[22]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[17]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[17]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[17]));
	asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[17]));
	asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[17]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[12]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[12]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[12]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[12]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[7]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[23]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[23]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[23]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[23]));
	asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[23]));
	asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[23]));

	asm volatile("vslidedown.vi v30, v30, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[18]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[18]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[18]));
	asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[18]));
	asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[18]));

	asm volatile("vslidedown.vi v28, v28, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[13]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[13]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[13]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[13]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[8]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[24]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[24]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[24]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[24]));
	asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[24]));
	asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[24]));

	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[19]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[19]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[19]));
	asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[19]));
	asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[19]));

	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));
	asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[14]));
	asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[14]));

	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));
	asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[9]));

	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));



	for(int c = 1 ; c < W ; c ++){

		f_ += 25;

		if (last_group == 0)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		}
		else if (last_group == 5)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(5)));

		}
		else if (last_group == 4)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[20]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[20]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[20]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[20]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[20]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[20]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[15]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[15]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[15]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[15]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[15]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[10]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[10]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[10]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[10]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[5]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[21]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[21]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[21]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[21]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[21]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[16]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[16]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[16]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[16]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[16]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[11]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[11]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[11]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[11]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[6]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[6]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[22]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[22]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[22]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[22]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[22]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[17]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[17]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[17]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[17]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[17]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[12]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[12]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[12]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[12]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[23]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[23]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[23]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[23]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[23]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v30, v30, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[18]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[18]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[18]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[18]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[18]));

		asm volatile("vslidedown.vi v28, v28, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[13]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[13]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[13]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[13]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[8]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[24]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[24]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[24]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[24]));
		asm volatile("vmacc.vx v8,  %0, v28" ::"r"(f_[24]));
		asm volatile("vmacc.vx v10,  %0, v30" ::"r"(f_[24]));

		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[19]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[19]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[19]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[19]));
		asm volatile("vmacc.vx v10,  %0, v28" ::"r"(f_[19]));

		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[14]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[14]));

		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[9]));

		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));


		}

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	else if (last_group == 5)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v8, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	else if (last_group == 4)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	else if (last_group == 3)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	else if (last_group == 2)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	else
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));

	}
	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                7x7 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor8_vec_4xC_7x7(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F)
{
	
int64_t const ldo = (C - 6);
int64_t const ldi = C;

int64_t vlen;

int64_t const last_group = (R - F + 1) % block_size_7x7;

for (int c = 0 ; c < (C - 6) ; c += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE_WIDE)
{

	
	int8_t *f_ = f;
	int8_t *i_ = i + c; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o + c;									// output pointer relative to the tile		


	
	if(c > C - TILE_SIZE) 	// if we are at the right border of the input
		vlen = C % TILE_SIZE_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length
	
	int8_t * i__ = i_;							// inside tile pointer (change at each load)
	
	int64_t vlen_out = vlen - 6;
	  	  	


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	asm volatile("vle8.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

	asm volatile("vmul.vx v0,  v12, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v2,  v14, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v4,  v16, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v6,  v18, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v8,  v20, %0" :: "r"(f_[0]));
	asm volatile("vmul.vx v10,  v22, %0" :: "r"(f_[0]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[7]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[7]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[7]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[14]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[14]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[21]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[21]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[21]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[28]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[28]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[35]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[1]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[1]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[8]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[8]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[8]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[15]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[15]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[15]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[15]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[22]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[22]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[22]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[29]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[29]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[36]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[2]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[2]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[9]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[9]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[9]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[16]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[16]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[16]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[16]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[23]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[23]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[23]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[30]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[30]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[37]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[3]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[3]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[3]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[3]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[10]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[10]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[10]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[10]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[10]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[17]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[17]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[17]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[17]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[24]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[24]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[24]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[31]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[31]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[38]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[4]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[4]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[4]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[4]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[11]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[11]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[11]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[11]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[11]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[18]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[18]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[18]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[18]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[25]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[25]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[25]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[32]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[32]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[39]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[5]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[5]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[5]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[5]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[5]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[5]));

	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[12]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[12]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[12]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[12]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[12]));

	asm volatile("vslidedown.vi v14, v14, 1");

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[19]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[19]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[19]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[19]));

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[26]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[26]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[26]));

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[33]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[33]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[40]));

	asm volatile("vslidedown.vi v22, v22, 1");

	asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[6]));
	asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[6]));
	asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[6]));
	asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[6]));
	asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[6]));
	asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[6]));

	asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[13]));
	asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[13]));
	asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[13]));
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[13]));
	asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[13]));

	asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[20]));
	asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[20]));
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[20]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[20]));

	asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[27]));
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[27]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[27]));

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[34]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[34]));

	asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[41]));



	for(int c = 1 ; c < W ; c ++){

		f_ += 49;

		asm volatile("vle8.v v12, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v14, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(6)));

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[0]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[0]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[0]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[0]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[0]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[7]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[7]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[7]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[7]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[14]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[14]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[14]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[14]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[21]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[21]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[28]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[28]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[35]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[1]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[1]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[1]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[1]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[1]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[8]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[8]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[8]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[8]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[15]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[15]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[15]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[15]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[22]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[22]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[29]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[29]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[36]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[2]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[2]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[2]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[2]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[2]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[9]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[9]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[9]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[9]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[9]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[16]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[16]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[16]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[16]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[23]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[23]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[30]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[30]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[37]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[3]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[3]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[3]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[3]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[3]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[10]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[10]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[10]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[10]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[10]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[17]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[17]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[17]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[17]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[24]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[24]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[24]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[31]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[31]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[38]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[4]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[4]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[4]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[4]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[4]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[11]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[11]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[11]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[11]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[11]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[18]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[18]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[18]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[18]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[25]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[25]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[25]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[32]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[32]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[39]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[5]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[5]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[5]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[5]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[5]));

		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[12]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[12]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[12]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[12]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[12]));

		asm volatile("vslidedown.vi v14, v14, 1");

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[19]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[19]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[19]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[19]));

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[26]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[26]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[26]));

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[33]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[33]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[40]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v0,  %0, v12" ::"r"(f_[6]));
		asm volatile("vmacc.vx v2,  %0, v14" ::"r"(f_[6]));
		asm volatile("vmacc.vx v4,  %0, v16" ::"r"(f_[6]));
		asm volatile("vmacc.vx v6,  %0, v18" ::"r"(f_[6]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[6]));

		asm volatile("vmacc.vx v0,  %0, v14" ::"r"(f_[13]));
		asm volatile("vmacc.vx v2,  %0, v16" ::"r"(f_[13]));
		asm volatile("vmacc.vx v4,  %0, v18" ::"r"(f_[13]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[13]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[13]));

		asm volatile("vmacc.vx v0,  %0, v16" ::"r"(f_[20]));
		asm volatile("vmacc.vx v2,  %0, v18" ::"r"(f_[20]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[20]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[20]));

		asm volatile("vmacc.vx v0,  %0, v18" ::"r"(f_[27]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[27]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[27]));

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[34]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[34]));

		asm volatile("vmacc.vx v0,  %0, v22" ::"r"(f_[41]));


		}

	i__ = i_ + 6 * C;
	f_ = f;
	  	

	for (int r = 6 + block_size_7x7; r < R ; r += block_size_7x7)
	{

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v12,  v20, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[7]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[14]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[21]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[28]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[35]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[42]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

		asm volatile("vmul.vx v14,  v22, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[7]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[14]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[21]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[28]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[35]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[42]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		asm volatile("vmul.vx v16,  v24, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[7]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[14]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[21]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[28]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[35]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[42]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmul.vx v18,  v26, %0" :: "r"(f_[0]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[7]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[14]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[21]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[28]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[35]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[42]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[1]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[8]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[15]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[22]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[29]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[36]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[43]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[1]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[8]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[15]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[22]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[29]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[36]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[43]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[1]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[8]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[15]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[22]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[29]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[36]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[43]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[1]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[8]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[15]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[22]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[29]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[36]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[43]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[2]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[9]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[16]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[23]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[30]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[37]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[44]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[2]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[9]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[16]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[23]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[30]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[37]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[44]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[2]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[9]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[16]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[23]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[30]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[37]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[44]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[2]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[9]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[16]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[23]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[30]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[37]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[44]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[3]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[10]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[17]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[24]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[31]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[38]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[45]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[3]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[10]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[17]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[24]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[31]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[38]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[45]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[3]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[10]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[17]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[24]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[31]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[38]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[45]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[3]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[10]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[17]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[24]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[31]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[38]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[45]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[4]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[11]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[18]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[25]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[32]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[39]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[46]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[4]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[11]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[18]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[25]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[32]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[39]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[46]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[4]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[11]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[18]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[25]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[32]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[39]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[46]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[4]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[11]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[18]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[25]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[32]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[39]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[46]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[5]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[12]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[19]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[26]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[33]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[40]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[47]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[5]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[12]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[19]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[26]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[33]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[40]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[47]));

		asm volatile("vslidedown.vi v22, v22, 1");

		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[5]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[12]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[19]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[26]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[33]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[40]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[47]));

		asm volatile("vslidedown.vi v24, v24, 1");

		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[5]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[12]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[19]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[26]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[33]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[40]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[47]));

		asm volatile("vslidedown.vi v26, v26, 1");

		asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[6]));
		asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[13]));
		asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[20]));
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[27]));
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[34]));
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[41]));
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[48]));


		asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[6]));
		asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[13]));
		asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[20]));
		asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[27]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[34]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[41]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[48]));


		asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[6]));
		asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[13]));
		asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[20]));
		asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[27]));
		asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[34]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[41]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[48]));


		asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[6]));
		asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[13]));
		asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[20]));
		asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[27]));
		asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[34]));
		asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[41]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[48]));



		for(int c = 1 ; c < W ; c ++){

			f_ += 49;

			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[0]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[7]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[14]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[21]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[28]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[35]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[42]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[0]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[7]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[14]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[21]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[28]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[35]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[42]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[0]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[7]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[14]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[21]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[28]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[35]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[42]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[0]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[7]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[14]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[21]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[28]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[35]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[42]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[1]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[8]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[15]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[22]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[29]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[36]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[43]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[1]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[8]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[15]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[22]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[29]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[36]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[43]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[1]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[8]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[15]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[22]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[29]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[36]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[43]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[1]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[8]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[15]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[22]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[29]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[36]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[43]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[2]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[9]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[16]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[23]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[30]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[37]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[44]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[2]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[9]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[16]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[23]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[30]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[37]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[44]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[2]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[9]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[16]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[23]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[30]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[37]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[44]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[2]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[9]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[16]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[23]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[30]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[37]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[44]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[3]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[10]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[17]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[24]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[31]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[38]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[45]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[3]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[10]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[17]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[24]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[31]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[38]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[45]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[3]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[10]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[17]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[24]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[31]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[38]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[45]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[3]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[10]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[17]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[24]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[31]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[38]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[45]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[4]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[11]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[18]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[25]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[32]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[39]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[46]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[4]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[11]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[18]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[25]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[32]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[39]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[46]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[4]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[11]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[18]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[25]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[32]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[39]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[46]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[4]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[11]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[18]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[25]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[32]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[39]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[46]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[5]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[12]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[19]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[26]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[33]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[40]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[47]));

			asm volatile("vslidedown.vi v20, v20, 1");

			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[5]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[12]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[19]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[26]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[33]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[40]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[47]));

			asm volatile("vslidedown.vi v22, v22, 1");

			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[5]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[12]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[19]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[26]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[33]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[40]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[47]));

			asm volatile("vslidedown.vi v24, v24, 1");

			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[5]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[12]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[19]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[26]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[33]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[40]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[47]));

			asm volatile("vslidedown.vi v26, v26, 1");

			asm volatile("vmacc.vx v12,  %0, v20" ::"r"(f_[6]));
			asm volatile("vmacc.vx v10,  %0, v20" ::"r"(f_[13]));
			asm volatile("vmacc.vx v8,  %0, v20" ::"r"(f_[20]));
			asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[27]));
			asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[34]));
			asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[41]));
			asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[48]));


			asm volatile("vmacc.vx v14,  %0, v22" ::"r"(f_[6]));
			asm volatile("vmacc.vx v12,  %0, v22" ::"r"(f_[13]));
			asm volatile("vmacc.vx v10,  %0, v22" ::"r"(f_[20]));
			asm volatile("vmacc.vx v8,  %0, v22" ::"r"(f_[27]));
			asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[34]));
			asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[41]));
			asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[48]));


			asm volatile("vmacc.vx v16,  %0, v24" ::"r"(f_[6]));
			asm volatile("vmacc.vx v14,  %0, v24" ::"r"(f_[13]));
			asm volatile("vmacc.vx v12,  %0, v24" ::"r"(f_[20]));
			asm volatile("vmacc.vx v10,  %0, v24" ::"r"(f_[27]));
			asm volatile("vmacc.vx v8,  %0, v24" ::"r"(f_[34]));
			asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[41]));
			asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[48]));


			asm volatile("vmacc.vx v18,  %0, v26" ::"r"(f_[6]));
			asm volatile("vmacc.vx v16,  %0, v26" ::"r"(f_[13]));
			asm volatile("vmacc.vx v14,  %0, v26" ::"r"(f_[20]));
			asm volatile("vmacc.vx v12,  %0, v26" ::"r"(f_[27]));
			asm volatile("vmacc.vx v10,  %0, v26" ::"r"(f_[34]));
			asm volatile("vmacc.vx v8,  %0, v26" ::"r"(f_[41]));
			asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[48]));


			}

		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));


			i__ = i_ + r * C;
			f_ = f;
		  	
		asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		asm volatile("vse8.v v6, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
		
		asm volatile("vmv.v.v v0,  v8");
		asm volatile("vmv.v.v v2,  v10");
		asm volatile("vmv.v.v v4,  v12");
		asm volatile("vmv.v.v v6,  v14");
		asm volatile("vmv.v.v v8,  v16");
		asm volatile("vmv.v.v v10, v18");



		}


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

	if (last_group == 0)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

	}
	else if (last_group == 3)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

	}
	else if (last_group == 2)
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

	}
	else
	{
		asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

	}
	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[42]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[42]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[42]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[42]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[35]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[35]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[35]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[28]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[28]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[21]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[43]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[43]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[43]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[43]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[36]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[36]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[36]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[29]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[29]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[22]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[44]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[44]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[44]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[44]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[37]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[37]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[37]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[30]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[30]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[23]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[45]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[45]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[45]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[45]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[38]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[38]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[38]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[31]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[31]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[24]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[46]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[46]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[46]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[46]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[39]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[39]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[39]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[32]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[32]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[25]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[47]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[47]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[47]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[47]));

	asm volatile("vslidedown.vi v26, v26, 1");
	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[40]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[40]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[40]));

	asm volatile("vslidedown.vi v24, v24, 1");
	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[33]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[33]));

	asm volatile("vslidedown.vi v22, v22, 1");
	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[26]));

	asm volatile("vslidedown.vi v20, v20, 1");

	asm volatile("vslidedown.vi v18, v18, 1");

	asm volatile("vslidedown.vi v16, v16, 1");

	asm volatile("vslidedown.vi v14, v14, 1");
	asm volatile("vslidedown.vi v12, v12, 1");

	asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[48]));
	asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[48]));
	asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[48]));
	asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[48]));

	asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[41]));
	asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[41]));
	asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[41]));

	asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[34]));
	asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[34]));

	asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[27]));






	for(int c = 1 ; c < W ; c ++){

		f_ += 49;

		if (last_group == 0)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(4)));

		}
		else if (last_group == 3)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(3)));

		}
		else if (last_group == 2)
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(2)));

		}
		else
		{
			asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_(1)));

		}
		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[42]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[42]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[42]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[42]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[35]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[35]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[35]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[28]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[28]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[21]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[43]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[43]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[43]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[43]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[36]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[36]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[36]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[29]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[29]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[22]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[44]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[44]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[44]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[44]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[37]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[37]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[37]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[30]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[30]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[23]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[45]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[45]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[45]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[45]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[38]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[38]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[38]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[31]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[31]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[24]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[46]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[46]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[46]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[46]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[39]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[39]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[39]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[32]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[32]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[25]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[47]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[47]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[47]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[47]));

		asm volatile("vslidedown.vi v26, v26, 1");
		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[40]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[40]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[40]));

		asm volatile("vslidedown.vi v24, v24, 1");
		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[33]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[33]));

		asm volatile("vslidedown.vi v22, v22, 1");
		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[26]));

		asm volatile("vslidedown.vi v20, v20, 1");

		asm volatile("vslidedown.vi v18, v18, 1");

		asm volatile("vslidedown.vi v16, v16, 1");

		asm volatile("vslidedown.vi v14, v14, 1");
		asm volatile("vslidedown.vi v12, v12, 1");

		asm volatile("vmacc.vx v0,  %0, v20" ::"r"(f_[48]));
		asm volatile("vmacc.vx v2,  %0, v22" ::"r"(f_[48]));
		asm volatile("vmacc.vx v4,  %0, v24" ::"r"(f_[48]));
		asm volatile("vmacc.vx v6,  %0, v26" ::"r"(f_[48]));

		asm volatile("vmacc.vx v2,  %0, v20" ::"r"(f_[41]));
		asm volatile("vmacc.vx v4,  %0, v22" ::"r"(f_[41]));
		asm volatile("vmacc.vx v6,  %0, v24" ::"r"(f_[41]));

		asm volatile("vmacc.vx v4,  %0, v20" ::"r"(f_[34]));
		asm volatile("vmacc.vx v6,  %0, v22" ::"r"(f_[34]));

		asm volatile("vmacc.vx v6,  %0, v20" ::"r"(f_[27]));





		}

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen_out));

	if (last_group == 0)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v6, (%0)" : "+&r"(o_));

	}
	else if (last_group == 3)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v4, (%0)" : "+&r"(o_));

	}
	else if (last_group == 2)
	{
	asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	asm volatile("vse8.v v2, (%0)" : "+&r"(o_));

	}
	else
	{
	asm volatile("vse8.v v0, (%0)" : "+&r"(o_));

	}
	}
}
