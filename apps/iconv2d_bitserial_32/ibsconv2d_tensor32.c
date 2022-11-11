// Author : Theo Dupuis
// GR2M - 2022
// Polytechnique Montreal

#include "ibsconv2d_tensor32.h"
#include <stdio.h>


#define ENCOD_SIZE 32

#ifdef PERF
	uint64_t runtime_cv = 0;
	uint64_t runtime_bp = 0;
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                    //			
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

void ibsconv2d_tensor32_3x3(int32_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW) {
	
  int8_t *i_;
  int32_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c;      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;
		
		
    switch(F){
    	case 3:
		  // Iterate over the output rows
		  if(precA == 1 && precW == 1)
		  		ibsconv2d32_W1_A1_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F, C_out);	
		  else if(precA == 1 && precW == 2)
		  		ibsconv2d32_W2_A1_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F, C_out);	
		  else if(precA == 2 && precW == 2)
		  		ibsconv2d32_W2_A2_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F, C_out);	
		break;
		
		case 5:
			if(precA == 1 && precW == 1)
				ibsconv2d32_W1_A1_vec_5x5(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			else if(precA == 1 && precW == 2)
				ibsconv2d32_W2_A1_vec_5x5(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			
		break;
		
		case 7:
			if(precA == 1 && precW == 1)
				ibsconv2d32_W1_A1_vec_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			else if(precA == 1 && precW == 2)
				ibsconv2d32_W2_A1_vec_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			else if(precA == 2 && precW == 2)
				ibsconv2d32_W2_A2_vec_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
		break;
		
		default:
	 		return;	
	 	}
		 
		 
	}
}

void print_tensor_(uint32_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%lu ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}



//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                  (A1W1)                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



void ibsconv2d32_W1_A1_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - F + 1);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * (F*F + 1)* C_in / 32]
int32_t f_packed[(F * F + 1) * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_1_to_32(f_, f_packed, F*F, C_in);

if(H_in >= F && W_in >= F)
for (int width = 0 ; width < W_in - F + 1 ; width += TILE_SIZE_A1_3x3_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_3x3) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_3x3_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_3x3;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + channels * (F * F + 1);

		bitpack32_vec_1_to_32_2H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 3 ; f_w ++){		
			
			asm volatile("lw t0, (%0); addi %0, %0, 12"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, -8"   : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
				// v25 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				
				// no addition because values are already in the right vd
			}

			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			
			
			asm volatile("vadd.vv v25, v25, v11");
			
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;	
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 2 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + channels * (F * F + 1);

				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
				for(int f_w = 0; f_w < 3 ; f_w ++){		
				
					asm volatile("lw t0, (%0); addi %0, %0, 12"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 12"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, -20"  : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed[2])
					asm volatile(".byte  0xD7, 0xC5, 0x33, 0x06");
					// v12 = popcount(v3  & f_packed[1])
					asm volatile(".byte  0x57, 0x46, 0x33, 0x06");
					
					if(f_w > 0 || channels > 0){
					
						// v13 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xC6, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
					}
					else{
						// v23 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xCB, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						

					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed[2])
						asm volatile(".byte  0x57, 0xC6, 0x43, 0x06");
						// v13 = popcount(v4 & f_packed[1])
						asm volatile(".byte  0xD7, 0x46, 0x43, 0x06");


							
						if(f_w > 0 || channels > 0)
							// v14 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xC7, 0x42, 0x06");
						else
							// v24 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xCC, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v24, v24, v14");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed[2])
						asm volatile(".byte  0xD7, 0xC6, 0x53, 0x06");
						// v14 = popcount(v5  & f_packed[1])
						asm volatile(".byte  0x57, 0x47, 0x53, 0x06");

						
						if(f_w > 0 || channels > 0)
							// v15 = popcount(v5  & f_packed[0])
							asm volatile(".byte  0xD7, 0xC7, 0x52, 0x06");
						else
							// v25 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xCC, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v25, v25, v15");
					}
					
					if(height < H_in - 3){

						// v14 = popcount(v6  & f_packed[2])
						asm volatile(".byte  0x57, 0xC7, 0x63, 0x06");
						// v15 = popcount(v6  & f_packed[1])
						asm volatile(".byte  0xD7, 0x47, 0x63, 0x06");

						if(f_w > 0 || channels > 0)
							// v16 = popcount(v6  & f_packed[0])
							asm volatile(".byte  0x57, 0xC8, 0x62, 0x06");
						else
							// v26 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xCD, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v26, v26, v16");
					}
						
						
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ += 4 * C_in * W_in;	
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}





void ibsconv2d32_W1_A1_vec_5x5(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - 4);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * (F*F + 1)* C_in / 32]
int32_t f_packed[(F * F + 1) * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_1_to_32(f_, f_packed, F*F, C_in);

if(H_in >= 5 && W_in >= 5)
for (int width = 0 ; width < (W_in - 4) ; width += TILE_SIZE_A1_5x5_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_5x5) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_5x5_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_5x5;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + channels * (F * F + 1);

		bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 5 ; f_w ++){		
			
			asm volatile("lw t0, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, -56"  : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				// v13 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
				// v14 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				asm volatile("vadd.vv v27, v27, v13");
				asm volatile("vadd.vv v28, v28, v14");
				
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
				// v25 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				// v27 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");
				// v28 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");
				
				// no addition because values are already in the right vd
			}

			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			
			
			// v11 = popcount(v5  & f_packed[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			
			// v11 = popcount(v6  & f_packed[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");

			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v5, v5, 1");
				asm volatile("vslidedown.vi v6, v6, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;	
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 4 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + channels * (F * F + 1);

				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
				
				asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));

		
				for(int f_w = 0; f_w < 5 ; f_w ++){		
				
					asm volatile("lw t0, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 20"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, -76"  : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed[4])
					asm volatile(".byte  0xD7, 0xC5, 0x3E, 0x06");
					// v12 = popcount(v3  & f_packed[3])
					asm volatile(".byte  0x57, 0x46, 0x3E, 0x06");
					// v13 = popcount(v3  & f_packed[2])
					asm volatile(".byte  0xD7, 0xC6, 0x33, 0x06");
					// v14 = popcount(v3  & f_packed[1])
					asm volatile(".byte  0x57, 0x47, 0x33, 0x06");
					
					if(f_w > 0 || channels > 0){
					
						// v15 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xC7, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
					}
					else{
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						asm volatile("vadd.vv v23, v27, v13");
						asm volatile("vadd.vv v24, v28, v14");
						
						// v25 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed[4])
						asm volatile(".byte  0x57, 0xC6, 0x4E, 0x06");
						// v13 = popcount(v4 & f_packed[3])
						asm volatile(".byte  0xD7, 0x46, 0x4E, 0x06");
						// v14 = popcount(v4 & f_packed[2])
						asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
						// v15 = popcount(v4 & f_packed[1])
						asm volatile(".byte  0xD7, 0x47, 0x43, 0x06");

							
						if(f_w > 0 || channels > 0)
							// v16 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xC8, 0x42, 0x06");
						else
							// v26 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v26, v26, v16");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed[4])
						asm volatile(".byte  0xD7, 0xC6, 0x5E, 0x06");
						// v14 = popcount(v5  & f_packed[3])
						asm volatile(".byte  0x57, 0x47, 0x5E, 0x06");
						// v15 = popcount(v5  & f_packed[2])
						asm volatile(".byte  0xD7, 0xC7, 0x53, 0x06");
						// v16 = popcount(v5  & f_packed[1])
						asm volatile(".byte  0x57, 0x48, 0x53, 0x06");

						if(f_w > 0 || channels > 0)
							// v17 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xC8, 0x52, 0x06");
						else
							// v27 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v27, v27, v17");
					}
					
					if(height < H_in - 3){

						// v14 = popcount(v6  & f_packed[4])
						asm volatile(".byte  0x57, 0xC7, 0x6E, 0x06");
						// v15 = popcount(v6  & f_packed[3])
						asm volatile(".byte  0xD7, 0x47, 0x6E, 0x06");
						// v16 = popcount(v6  & f_packed[2])
						asm volatile(".byte  0x57, 0xC8, 0x63, 0x06");
						// v17 = popcount(v6  & f_packed[1])
						asm volatile(".byte  0xD7, 0x48, 0x63, 0x06");
						
						if(f_w > 0 || channels > 0)
							// v18 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xC9, 0x62, 0x06");
						else
							// v28 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v28, v28, v18");
					}
						
						
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ += (F - 1) * C_in * W_in;	
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}






void ibsconv2d32_W1_A1_vec_7x7(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - 6);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * (F*F + 1)* C_in / 32]
int32_t f_packed[50 * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_1_to_32(f_, f_packed, F*F, C_in);

if(H_in >= 7 && W_in >= 7)
for (int width = 0 ; width < (W_in - 6) ; width += TILE_SIZE_A1_7x7_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_7x7) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_7x7_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_7x7;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + channels * (F * F + 1);

		bitpack32_vec_1_to_32_6H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 7 ; f_w ++){		
			
			asm volatile("lw t0, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
			asm volatile("lw t4, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
			asm volatile("lw t5, (%0); addi %0, %0, -136" : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				// v13 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
				// v14 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
				// v15 = popcount(v7 & f_packed[0])
				asm volatile(".byte  0xD7, 0xC7, 0x72, 0x06");
				// v16 = popcount(v8 & f_packed[0])
				asm volatile(".byte  0x57, 0xC8, 0x82, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				asm volatile("vadd.vv v27, v27, v13");
				asm volatile("vadd.vv v28, v28, v14");
				asm volatile("vadd.vv v29, v29, v15");
				asm volatile("vadd.vv v30, v30, v16");
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
				// v25 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				// v27 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");
				// v28 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");
				// v29 = popcount(v7 & f_packed[0])
				asm volatile(".byte  0xD7, 0xCE, 0x72, 0x06");
				// v30 = popcount(v8 & f_packed[0])
				asm volatile(".byte  0x57, 0xCF, 0x82, 0x06");
				
				// no addition because values are already in the right vd
			}

			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			// v14 = popcount(v7  & f_packed[1])
			asm volatile(".byte  0x57, 0x47, 0x73, 0x06");
			// v15 = popcount(v8 & f_packed[1])
			asm volatile(".byte  0xD7, 0x47, 0x83, 0x06");
			
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			
			
			
			// v11 = popcount(v5  & f_packed[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
			// v13 = popcount(v7  & f_packed[2])
			asm volatile(".byte  0xD7, 0xC6, 0x73, 0x06");
			// v14 = popcount(v8  & f_packed[2])
			asm volatile(".byte  0x57, 0xC7, 0x83, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			
			// v11 = popcount(v6  & f_packed[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
			// v12 = popcount(v7  & f_packed[3])
			asm volatile(".byte  0x57, 0x46, 0x7E, 0x06");
			// v13 = popcount(v8  & f_packed[3])
			asm volatile(".byte  0xD7, 0x46, 0x8E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			
			// v11 = popcount(v7  & f_packed[4])
			asm volatile(".byte  0xD7, 0xC5, 0x7E, 0x06");
			// v12 = popcount(v8  & f_packed[4])
			asm volatile(".byte  0x57, 0xC6, 0x8E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			
			// v11 = popcount(v8  & f_packed[5])
			asm volatile(".byte  0xD7, 0x45, 0x8F, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v5, v5, 1");
				asm volatile("vslidedown.vi v6, v6, 1");
				asm volatile("vslidedown.vi v7, v7, 1");
				asm volatile("vslidedown.vi v8, v8, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;	
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 6 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + channels * (F * F + 1);

				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
				for(int f_w = 0; f_w < 7 ; f_w ++){		
				
					asm volatile("lw t0, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t5, (%0); addi %0, %0, 28"   : "+&r"(f_loop));
					asm volatile("lw t6, (%0); addi %0, %0, -164" : "+&r"(f_loop));
					
					
					// v11 = popcount(v3  & f_packed[6])
					asm volatile(".byte  0xD7, 0xC5, 0x3F, 0x06");
					// v12 = popcount(v3  & f_packed[5])
					asm volatile(".byte  0x57, 0x46, 0x3F, 0x06");
					// v13 = popcount(v3  & f_packed[4])
					asm volatile(".byte  0xD7, 0xC6, 0x3E, 0x06");
					// v14 = popcount(v3  & f_packed[3])
					asm volatile(".byte  0x57, 0x47, 0x3E, 0x06");
					// v15 = popcount(v3 & f_packed[2])
					asm volatile(".byte  0xD7, 0xC7, 0x33, 0x06");
					// v16 = popcount(v3 & f_packed[1])
					asm volatile(".byte  0x57, 0x48, 0x33, 0x06");
					if(f_w > 0 || channels > 0)
						// v17 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xC8, 0x32, 0x06");
					
					if(f_w > 0 || channels > 0){
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
					}
					else{
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						asm volatile("vadd.vv v23, v27, v13");
						asm volatile("vadd.vv v24, v28, v14");
						asm volatile("vadd.vv v25, v29, v15");
						asm volatile("vadd.vv v26, v30, v16");
						
						// v27 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xCD, 0x32, 0x06");
					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed[6])
						asm volatile(".byte  0x57, 0xC6, 0x4F, 0x06");
						// v13 = popcount(v4 & f_packed[5])
						asm volatile(".byte  0xD7, 0x46, 0x4F, 0x06");
						// v14 = popcount(v4 & f_packed[4])
						asm volatile(".byte  0x57, 0xC7, 0x4E, 0x06");
						// v15 = popcount(v4 & f_packed[3])
						asm volatile(".byte  0xD7, 0x47, 0x4E, 0x06");
						// v16 = popcount(v4 & f_packed[2])
						asm volatile(".byte  0x57, 0xC8, 0x43, 0x06");
						// v17 = popcount(v4 & f_packed[1])
						asm volatile(".byte  0xD7, 0x48, 0x43, 0x06");
							
						if(f_w > 0 || channels > 0)
							// v18 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
						else
							// v28 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xCE, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v28, v28, v18");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed[6])
						asm volatile(".byte  0xD7, 0xC6, 0x5F, 0x06");
						// v14 = popcount(v5  & f_packed[5])
						asm volatile(".byte  0x57, 0x47, 0x5F, 0x06");
						// v15 = popcount(v5  & f_packed[4])
						asm volatile(".byte  0xD7, 0xC7, 0x5E, 0x06");
						// v16 = popcount(v5  & f_packed[3])
						asm volatile(".byte  0x57, 0x48, 0x5E, 0x06");
						// v17 = popcount(v5 & f_packed[2])
						asm volatile(".byte  0xD7, 0xC8, 0x53, 0x06");
						// v18 = popcount(v5 & f_packed[1])
						asm volatile(".byte  0x57, 0x49, 0x53, 0x06");
						if(f_w > 0 || channels > 0)
							// v19 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xC9, 0x52, 0x06");
						else
							// v29 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xCE, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v29, v29, v19");
					}
					
					if(height < H_in - 3){
						// v14 = popcount(v6  & f_packed[6])
						asm volatile(".byte  0x57, 0xC7, 0x6F, 0x06");
						// v15 = popcount(v6  & f_packed[5])
						asm volatile(".byte  0xD7, 0x47, 0x6F, 0x06");
						// v16 = popcount(v6  & f_packed[4])
						asm volatile(".byte  0x57, 0xC8, 0x6E, 0x06");
						// v17 = popcount(v6  & f_packed[3])
						asm volatile(".byte  0xD7, 0x48, 0x6E, 0x06");
						// v18 = popcount(v6 & f_packed[2])
						asm volatile(".byte  0x57, 0xC9, 0x63, 0x06");
						// v19 = popcount(v6 & f_packed[1])
						asm volatile(".byte  0xD7, 0x49, 0x63, 0x06");
						if(f_w > 0 || channels > 0)
							// v20 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
						else
							// v30 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xCF, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v30, v30, v20");
					}
						
						
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ +=  4 * C_in * W_in;
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                 (A1W2)                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


void ibsconv2d32_W2_A1_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - F + 1);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * (F*F + 1)* C_in / 32]
int32_t f_packed[2 * F * F * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;

bitpack_filter32_vec_2_to_32(f_, f_packed, F*F, C_in);

if(H_in >= F && W_in >= F)
for (int width = 0 ; width < (W_in - F + 1) ; width += TILE_SIZE_A1_3x3_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_3x3) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_3x3_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_3x3;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
		
		f_loop = f_packed + 2 * channels * F * F;

		bitpack32_vec_1_to_32_2H(i__, vlen, C_in, W_in); 
		
		asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen));
		
		for(int f_w = 0; f_w < 3 ; f_w ++){		
			
			asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, -20"   : "+&r"(f_loop));
			
			// Weight LSB and Activation LSB

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				
			}
			else{  
				// v25 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				
			}
			
			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			
			asm volatile("vadd.vv v25, v25, v11");
			
			
			
			
			
			// Weight bit 1 and Activation LSB
			asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, -20"   : "+&r"(f_loop));
			
			// v11 = popcount(v3  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
			// v12 = popcount(v4  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
			
			
			////// REPLACE WITH 2 VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			//////////////////////////////////////
			
			// v11 = popcount(v4  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			#endif
			
			asm volatile("vadd.vv v25, v25, v11");
			/////////////////////////////////////


			
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;	
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 2 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + 2 * channels * F * F;
				
				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
				for(int f_w = 0; f_w < 3 ; f_w ++){		
				
					// Weight LSB and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, -44"  : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed_0[2])
					asm volatile(".byte  0xD7, 0xC5, 0x33, 0x06");
					// v12 = popcount(v3  & f_packed_0[1])
					asm volatile(".byte  0x57, 0x46, 0x33, 0x06");
					
					if(f_w > 0 || channels > 0){
					
						// v13 = popcount(v3 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xC6, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
					}
					else{
						// v23 = popcount(v3 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xCB, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						

					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC6, 0x43, 0x06");
						// v13 = popcount(v4 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x46, 0x43, 0x06");


							
						if(f_w > 0 || channels > 0)
							// v14 = popcount(v4 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xC7, 0x42, 0x06");
						else
							// v24 = popcount(v4 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCC, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v24, v24, v14");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC6, 0x53, 0x06");
						// v14 = popcount(v5  & f_packed_0[1])
						asm volatile(".byte  0x57, 0x47, 0x53, 0x06");

						
						if(f_w > 0 || channels > 0)
							// v15 = popcount(v5  & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xC7, 0x52, 0x06");
						else
							// v25 = popcount(v5 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCC, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v25, v25, v15");
					}
					
					if(height < H_in - 3){

						// v14 = popcount(v6  & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC7, 0x63, 0x06");
						// v15 = popcount(v6  & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x47, 0x63, 0x06");

						if(f_w > 0 || channels > 0)
							// v16 = popcount(v6  & f_packed_0[0])
							asm volatile(".byte  0x57, 0xC8, 0x62, 0x06");
						else
							// v26 = popcount(v6 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCD, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v26, v26, v16");
					}
						
						
					// Weight bit 1 and Activation LSB	
					asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, -44"  : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed_1[2])
					asm volatile(".byte  0xD7, 0xC5, 0x33, 0x06");
					// v12 = popcount(v3  & f_packed_1[1])
					asm volatile(".byte  0x57, 0x46, 0x33, 0x06");
					// v13 = popcount(v3 & f_packed_1[0])
					asm volatile(".byte  0xD7, 0xC6, 0x32, 0x06");

					////// REPLACE WITH 3 VSHACC   //////
					#ifndef PERF_VHSACC
					asm volatile("vsll.vi v11, v11, 1");
					asm volatile("vsll.vi v12, v12, 1");
					asm volatile("vsll.vi v13, v13, 1");
					#endif
					
					asm volatile("vadd.vv v21, v21, v11");
					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					/////////////////////////////////////
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed_1[2])
						asm volatile(".byte  0x57, 0xC6, 0x43, 0x06");
						// v13 = popcount(v4 & f_packed_1[1])
						asm volatile(".byte  0xD7, 0x46, 0x43, 0x06");
						// v14 = popcount(v4 & f_packed_1[0])
						asm volatile(".byte  0x57, 0xC7, 0x42, 0x06");

						////// REPLACE WITH 3 VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v12, v12, 1");
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						#endif
						
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						/////////////////////////////////////
					}
					
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed_1[2])
						asm volatile(".byte  0xD7, 0xC6, 0x53, 0x06");
						// v14 = popcount(v5  & f_packed_1[1])
						asm volatile(".byte  0x57, 0x47, 0x53, 0x06");
						// v15 = popcount(v5  & f_packed_1[0])
						asm volatile(".byte  0xD7, 0xC7, 0x52, 0x06");
						
						////// REPLACE WITH 3 VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						#endif
						
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						/////////////////////////////////////
					}
					
					if(height < H_in - 3){
						// v14 = popcount(v6  & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC7, 0x63, 0x06");
						// v15 = popcount(v6  & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x47, 0x63, 0x06");
						// v16 = popcount(v6  & f_packed_0[0])
						asm volatile(".byte  0x57, 0xC8, 0x62, 0x06");
						
						
						////// REPLACE WITH 3 VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						#endif
						
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						/////////////////////////////////////
					}
						
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ += 4 * C_in * W_in;	
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}

void ibsconv2d32_W2_A1_vec_5x5(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - 4);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * F * F * C_in / 32]
int32_t f_packed[2 * F * F * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_2_to_32(f_, f_packed, F*F, C_in);

if(H_in >= 5 && W_in >= 5)
for (int width = 0 ; width < (W_in - 4) ; width += TILE_SIZE_A1_5x5_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_5x5) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_5x5_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_5x5;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + 2 * channels * F * F;

		bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 5 ; f_w ++){		
			
			// Weight LSB and Activation LSB
			asm volatile("lw t0, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, -116" : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				// v13 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
				// v14 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				asm volatile("vadd.vv v27, v27, v13");
				asm volatile("vadd.vv v28, v28, v14");
				
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
				// v25 = popcount(v3  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				// v27 = popcount(v5  & f_packed[0])
				asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");
				// v28 = popcount(v6  & f_packed[0])
				asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");
				
				// no addition because values are already in the right vd
			}

			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			
			
			// v11 = popcount(v5  & f_packed[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			
			// v11 = popcount(v6  & f_packed[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			
			
			
			// Weight bit 1 and Activation LSB	
			asm volatile("lw t0, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, -116" : "+&r"(f_loop));

			// v11 = popcount(v3  & f_packed[0])
			asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
			// v12 = popcount(v4  & f_packed[0])
			asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
			// v13 = popcount(v5  & f_packed[0])
			asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
			// v14 = popcount(v6  & f_packed[0])
			asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			#endif
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			//////////////////////////////////////

			// v11 = popcount(v4  & f_packed[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			#endif
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			//////////////////////////////////////
			
			// v11 = popcount(v5  & f_packed[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			//////////////////////////////////////
			
			// v11 = popcount(v6  & f_packed[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			///////////////////////////////////////
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v5, v5, 1");
				asm volatile("vslidedown.vi v6, v6, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;	
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 4 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + 2 * channels * F * F;

				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 
		
				for(int f_w = 0; f_w < 5 ; f_w ++){		
				
					// Weight LSB and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, -156" : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed[4])
					asm volatile(".byte  0xD7, 0xC5, 0x3E, 0x06");
					// v12 = popcount(v3  & f_packed[3])
					asm volatile(".byte  0x57, 0x46, 0x3E, 0x06");
					// v13 = popcount(v3  & f_packed[2])
					asm volatile(".byte  0xD7, 0xC6, 0x33, 0x06");
					// v14 = popcount(v3  & f_packed[1])
					asm volatile(".byte  0x57, 0x47, 0x33, 0x06");
					
					if(f_w > 0 || channels > 0){
					
						// v15 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xC7, 0x32, 0x06");
						
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
					}
					else{
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						asm volatile("vadd.vv v23, v27, v13");
						asm volatile("vadd.vv v24, v28, v14");
						
						// v25 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed[4])
						asm volatile(".byte  0x57, 0xC6, 0x4E, 0x06");
						// v13 = popcount(v4 & f_packed[3])
						asm volatile(".byte  0xD7, 0x46, 0x4E, 0x06");
						// v14 = popcount(v4 & f_packed[2])
						asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
						// v15 = popcount(v4 & f_packed[1])
						asm volatile(".byte  0xD7, 0x47, 0x43, 0x06");

							
						if(f_w > 0 || channels > 0)
							// v16 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xC8, 0x42, 0x06");
						else
							// v26 = popcount(v4 & f_packed[0])
							asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v26, v26, v16");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed[4])
						asm volatile(".byte  0xD7, 0xC6, 0x5E, 0x06");
						// v14 = popcount(v5  & f_packed[3])
						asm volatile(".byte  0x57, 0x47, 0x5E, 0x06");
						// v15 = popcount(v5  & f_packed[2])
						asm volatile(".byte  0xD7, 0xC7, 0x53, 0x06");
						// v16 = popcount(v5  & f_packed[1])
						asm volatile(".byte  0x57, 0x48, 0x53, 0x06");

						if(f_w > 0 || channels > 0)
							// v17 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xC8, 0x52, 0x06");
						else
							// v27 = popcount(v5 & f_packed[0])
							asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v27, v27, v17");
					}
					
					if(height < H_in - 3){

						// v14 = popcount(v6  & f_packed[4])
						asm volatile(".byte  0x57, 0xC7, 0x6E, 0x06");
						// v15 = popcount(v6  & f_packed[3])
						asm volatile(".byte  0xD7, 0x47, 0x6E, 0x06");
						// v16 = popcount(v6  & f_packed[2])
						asm volatile(".byte  0x57, 0xC8, 0x63, 0x06");
						// v17 = popcount(v6  & f_packed[1])
						asm volatile(".byte  0xD7, 0x48, 0x63, 0x06");
						
						if(f_w > 0 || channels > 0)
							// v18 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xC9, 0x62, 0x06");
						else
							// v28 = popcount(v6 & f_packed[0])
							asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v28, v28, v18");
					}
					
					
					
					
					// Weight bit 1 and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 40"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, -156" : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed[4])
					asm volatile(".byte  0xD7, 0xC5, 0x3E, 0x06");
					// v12 = popcount(v3  & f_packed[3])
					asm volatile(".byte  0x57, 0x46, 0x3E, 0x06");
					// v13 = popcount(v3  & f_packed[2])
					asm volatile(".byte  0xD7, 0xC6, 0x33, 0x06");
					// v14 = popcount(v3  & f_packed[1])
					asm volatile(".byte  0x57, 0x47, 0x33, 0x06");
					// v15 = popcount(v3 & f_packed[0])
					asm volatile(".byte  0xD7, 0xC7, 0x32, 0x06");
					
					#ifndef PERF_VHSACC	
					asm volatile("vsll.vi v11, v11, 1");
					asm volatile("vsll.vi v12, v12, 1");
					asm volatile("vsll.vi v13, v13, 1");
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v15, v15, 1");
					#endif
						
					asm volatile("vadd.vv v21, v21, v11");
					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					asm volatile("vadd.vv v24, v24, v14");
					asm volatile("vadd.vv v25, v25, v15");

					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed[4])
						asm volatile(".byte  0x57, 0xC6, 0x4E, 0x06");
						// v13 = popcount(v4 & f_packed[3])
						asm volatile(".byte  0xD7, 0x46, 0x4E, 0x06");
						// v14 = popcount(v4 & f_packed[2])
						asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
						// v15 = popcount(v4 & f_packed[1])
						asm volatile(".byte  0xD7, 0x47, 0x43, 0x06");
						// v16 = popcount(v4 & f_packed[0])
						asm volatile(".byte  0x57, 0xC8, 0x42, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v12, v12, 1");
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						#endif
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed[4])
						asm volatile(".byte  0xD7, 0xC6, 0x5E, 0x06");
						// v14 = popcount(v5  & f_packed[3])
						asm volatile(".byte  0x57, 0x47, 0x5E, 0x06");
						// v15 = popcount(v5  & f_packed[2])
						asm volatile(".byte  0xD7, 0xC7, 0x53, 0x06");
						// v16 = popcount(v5  & f_packed[1])
						asm volatile(".byte  0x57, 0x48, 0x53, 0x06");
						// v17 = popcount(v5 & f_packed[0])
						asm volatile(".byte  0xD7, 0xC8, 0x52, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						#endif
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
					}
					
					if(height < H_in - 3){

						// v14 = popcount(v6  & f_packed[4])
						asm volatile(".byte  0x57, 0xC7, 0x6E, 0x06");
						// v15 = popcount(v6  & f_packed[3])
						asm volatile(".byte  0xD7, 0x47, 0x6E, 0x06");
						// v16 = popcount(v6  & f_packed[2])
						asm volatile(".byte  0x57, 0xC8, 0x63, 0x06");
						// v17 = popcount(v6  & f_packed[1])
						asm volatile(".byte  0xD7, 0x48, 0x63, 0x06");
						// v18 = popcount(v6 & f_packed[0])
						asm volatile(".byte  0x57, 0xC9, 0x62, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						#endif
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
					}
						
						
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ += (F - 1) * C_in * W_in;	
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}



void ibsconv2d32_W2_A1_vec_7x7(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){


int64_t const ldo = C_out * (W_in - 6);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [prec * (F*F + 1)* C_in / 32]
int32_t f_packed[2 * F * F * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;




bitpack_filter32_vec_2_to_32(f_, f_packed, F*F, C_in);

if(H_in >= 7 && W_in >= 7)
for (int width = 0 ; width < (W_in - 6) ; width += TILE_SIZE_A1_7x7_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A1_7x7) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A1_7x7_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A1_7x7;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + 2 * channels * F * F;

		bitpack32_vec_1_to_32_6H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 7 ; f_w ++){		
			
			// Weight LSB and Activation LSB
			asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t5, (%0); addi %0, %0, -276" : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				// v11 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
				// v12 = popcount(v4  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
				// v13 = popcount(v5  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
				// v14 = popcount(v6  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
				// v15 = popcount(v7 & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC7, 0x72, 0x06");
				// v16 = popcount(v8 & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC8, 0x82, 0x06");
				
				asm volatile("vadd.vv v25, v25, v11");
				asm volatile("vadd.vv v26, v26, v12");
				asm volatile("vadd.vv v27, v27, v13");
				asm volatile("vadd.vv v28, v28, v14");
				asm volatile("vadd.vv v29, v29, v15");
				asm volatile("vadd.vv v30, v30, v16");
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
				// v25 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCC, 0x32, 0x06");
				// v26 = popcount(v4  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCD, 0x42, 0x06");
				// v27 = popcount(v5  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCD, 0x52, 0x06");
				// v28 = popcount(v6  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCE, 0x62, 0x06");
				// v29 = popcount(v7 & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCE, 0x72, 0x06");
				// v30 = popcount(v8 & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCF, 0x82, 0x06");
				
				// no addition because values are already in the right vd
			}

			// v11 = popcount(v4  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			// v14 = popcount(v7  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x47, 0x73, 0x06");
			// v15 = popcount(v8 & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x47, 0x83, 0x06");
			
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			
			
			
			// v11 = popcount(v5  & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
			// v13 = popcount(v7  & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC6, 0x73, 0x06");
			// v14 = popcount(v8  & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC7, 0x83, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			
			// v11 = popcount(v6  & f_packed_0[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
			// v12 = popcount(v7  & f_packed_0[3])
			asm volatile(".byte  0x57, 0x46, 0x7E, 0x06");
			// v13 = popcount(v8  & f_packed_0[3])
			asm volatile(".byte  0xD7, 0x46, 0x8E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			
			// v11 = popcount(v7  & f_packed_0[4])
			asm volatile(".byte  0xD7, 0xC5, 0x7E, 0x06");
			// v12 = popcount(v8  & f_packed_0[4])
			asm volatile(".byte  0x57, 0xC6, 0x8E, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			
			// v11 = popcount(v8  & f_packed_0[5])
			asm volatile(".byte  0xD7, 0x45, 0x8F, 0x06");
				
			asm volatile("vadd.vv v25, v25, v11");
			
			
			
			// Weight bit 1 and Activation LSB	
			asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t5, (%0); addi %0, %0, -276" : "+&r"(f_loop));
			
			// v11 = popcount(v3  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC5, 0x32, 0x06");
			// v12 = popcount(v4  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC6, 0x42, 0x06");
			// v13 = popcount(v5  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC6, 0x52, 0x06");
			// v14 = popcount(v6  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC7, 0x62, 0x06");
			// v15 = popcount(v7 & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC7, 0x72, 0x06");
			// v16 = popcount(v8 & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC8, 0x82, 0x06");
			
			////// REPLACE WITH 6 VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			asm volatile("vsll.vi v16, v16, 1");
			#endif
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			/////////////////////////////////////
			
			// v11 = popcount(v4  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x45, 0x43, 0x06");
			// v12 = popcount(v5  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x46, 0x53, 0x06");
			// v13 = popcount(v6  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x46, 0x63, 0x06");
			// v14 = popcount(v7  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x47, 0x73, 0x06");
			// v15 = popcount(v8 & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x47, 0x83, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			#endif
			
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			//////////////////////////////////////
			
			// v11 = popcount(v5  & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC5, 0x53, 0x06");
			// v12 = popcount(v6  & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC6, 0x63, 0x06");
			// v13 = popcount(v7  & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC6, 0x73, 0x06");
			// v14 = popcount(v8  & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC7, 0x83, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			////////////////////////////////////
			
			// v11 = popcount(v6  & f_packed_1[3])
			asm volatile(".byte  0xD7, 0x45, 0x6E, 0x06");
			// v12 = popcount(v7  & f_packed_1[3])
			asm volatile(".byte  0x57, 0x46, 0x7E, 0x06");
			// v13 = popcount(v8  & f_packed_1[3])
			asm volatile(".byte  0xD7, 0x46, 0x8E, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			/////////////////////////////////////
			
			// v11 = popcount(v7  & f_packed_1[4])
			asm volatile(".byte  0xD7, 0xC5, 0x7E, 0x06");
			// v12 = popcount(v8  & f_packed_1[4])
			asm volatile(".byte  0x57, 0xC6, 0x8E, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			asm volatile("vsll.vi v12, v12, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			asm volatile("vadd.vv v26, v26, v12");
			/////////////////////////////////////
			
			// v11 = popcount(v8  & f_packed_1[5])
			asm volatile(".byte  0xD7, 0x45, 0x8F, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v11, v11, 1");
			#endif
				
			asm volatile("vadd.vv v25, v25, v11");
			///////////////////////////////////
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v5, v5, 1");
				asm volatile("vslidedown.vi v6, v6, 1");
				asm volatile("vslidedown.vi v7, v7, 1");
				asm volatile("vslidedown.vi v8, v8, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 6 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + 2 * channels * F * F;

				bitpack32_vec_1_to_32_4H(i__, vlen, C_in, W_in); 

				for(int f_w = 0; f_w < 7 ; f_w ++){		
				
					// Weight LSB and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t5, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t6, (%0); addi %0, %0, -332" : "+&r"(f_loop));
					
					
					// v11 = popcount(v3  & f_packed_0[6])
					asm volatile(".byte  0xD7, 0xC5, 0x3F, 0x06");
					// v12 = popcount(v3  & f_packed_0[5])
					asm volatile(".byte  0x57, 0x46, 0x3F, 0x06");
					// v13 = popcount(v3  & f_packed_0[4])
					asm volatile(".byte  0xD7, 0xC6, 0x3E, 0x06");
					// v14 = popcount(v3  & f_packed_0[3])
					asm volatile(".byte  0x57, 0x47, 0x3E, 0x06");
					// v15 = popcount(v3 & f_packed_0[2])
					asm volatile(".byte  0xD7, 0xC7, 0x33, 0x06");
					// v16 = popcount(v3 & f_packed_0[1])
					asm volatile(".byte  0x57, 0x48, 0x33, 0x06");
					if(f_w > 0 || channels > 0)
						// v17 = popcount(v3 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xC8, 0x32, 0x06");
					
					if(f_w > 0 || channels > 0){
						asm volatile("vadd.vv v21, v21, v11");
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
					}
					else{
						asm volatile("vadd.vv v21, v25, v11");
						asm volatile("vadd.vv v22, v26, v12");
						asm volatile("vadd.vv v23, v27, v13");
						asm volatile("vadd.vv v24, v28, v14");
						asm volatile("vadd.vv v25, v29, v15");
						asm volatile("vadd.vv v26, v30, v16");
						
						// v27 = popcount(v3 & f_packed[0])
						asm volatile(".byte  0xD7, 0xCD, 0x32, 0x06");
					}
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed_0[6])
						asm volatile(".byte  0x57, 0xC6, 0x4F, 0x06");
						// v13 = popcount(v4 & f_packed_0[5])
						asm volatile(".byte  0xD7, 0x46, 0x4F, 0x06");
						// v14 = popcount(v4 & f_packed_0[4])
						asm volatile(".byte  0x57, 0xC7, 0x4E, 0x06");
						// v15 = popcount(v4 & f_packed_0[3])
						asm volatile(".byte  0xD7, 0x47, 0x4E, 0x06");
						// v16 = popcount(v4 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC8, 0x43, 0x06");
						// v17 = popcount(v4 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x48, 0x43, 0x06");
							
						if(f_w > 0 || channels > 0)
							// v18 = popcount(v4 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
						else
							// v28 = popcount(v4 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCE, 0x42, 0x06");
					
					
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v28, v28, v18");
					}
						
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC6, 0x5F, 0x06");
						// v14 = popcount(v5  & f_packed_0[5])
						asm volatile(".byte  0x57, 0x47, 0x5F, 0x06");
						// v15 = popcount(v5  & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC7, 0x5E, 0x06");
						// v16 = popcount(v5  & f_packed_0[3])
						asm volatile(".byte  0x57, 0x48, 0x5E, 0x06");
						// v17 = popcount(v5 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC8, 0x53, 0x06");
						// v18 = popcount(v5 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x49, 0x53, 0x06");
						if(f_w > 0 || channels > 0)
							// v19 = popcount(v5 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xC9, 0x52, 0x06");
						else
							// v29 = popcount(v5 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCE, 0x52, 0x06");	
							
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v29, v29, v19");
					}
					
					if(height < H_in - 3){
						// v14 = popcount(v6  & f_packed_0[6])
						asm volatile(".byte  0x57, 0xC7, 0x6F, 0x06");
						// v15 = popcount(v6  & f_packed_0[5])
						asm volatile(".byte  0xD7, 0x47, 0x6F, 0x06");
						// v16 = popcount(v6  & f_packed_0[4])
						asm volatile(".byte  0x57, 0xC8, 0x6E, 0x06");
						// v17 = popcount(v6  & f_packed_0[3])
						asm volatile(".byte  0xD7, 0x48, 0x6E, 0x06");
						// v18 = popcount(v6 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC9, 0x63, 0x06");
						// v19 = popcount(v6 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x49, 0x63, 0x06");
						if(f_w > 0 || channels > 0)
							// v20 = popcount(v6 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
						else
							// v30 = popcount(v6 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCF, 0x62, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v30, v30, v20");
					}
					
					
					// Weight bit 1 and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t5, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t6, (%0); addi %0, %0, -332" : "+&r"(f_loop));
					
					// v11 = popcount(v3  & f_packed_1[6])
					asm volatile(".byte  0xD7, 0xC5, 0x3F, 0x06");
					// v12 = popcount(v3  & f_packed_1[5])
					asm volatile(".byte  0x57, 0x46, 0x3F, 0x06");
					// v13 = popcount(v3  & f_packed_1[4])
					asm volatile(".byte  0xD7, 0xC6, 0x3E, 0x06");
					// v14 = popcount(v3  & f_packed_1[3])
					asm volatile(".byte  0x57, 0x47, 0x3E, 0x06");
					// v15 = popcount(v3 & f_packed_1[2])
					asm volatile(".byte  0xD7, 0xC7, 0x33, 0x06");
					// v16 = popcount(v3 & f_packed_1[1])
					asm volatile(".byte  0x57, 0x48, 0x33, 0x06");
					// v17 = popcount(v3 & f_packed_1[0])
					asm volatile(".byte  0xD7, 0xC8, 0x32, 0x06");
					
					////// REPLACE WITH VSHACC   //////
					#ifndef PERF_VHSACC
					asm volatile("vsll.vi v11, v11, 1");
					asm volatile("vsll.vi v12, v12, 1");
					asm volatile("vsll.vi v13, v13, 1");
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v15, v15, 1");
					asm volatile("vsll.vi v16, v16, 1");
					asm volatile("vsll.vi v17, v17, 1");
					#endif
					
					asm volatile("vadd.vv v21, v21, v11");
					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					asm volatile("vadd.vv v24, v24, v14");
					asm volatile("vadd.vv v25, v25, v15");
					asm volatile("vadd.vv v26, v26, v16");
					asm volatile("vadd.vv v27, v27, v17");
					/////////////////////////////////////
					
					if(height < H_in - 1){
						// v12 = popcount(v4 & f_packed_1[6])
						asm volatile(".byte  0x57, 0xC6, 0x4F, 0x06");
						// v13 = popcount(v4 & f_packed_1[5])
						asm volatile(".byte  0xD7, 0x46, 0x4F, 0x06");
						// v14 = popcount(v4 & f_packed_1[4])
						asm volatile(".byte  0x57, 0xC7, 0x4E, 0x06");
						// v15 = popcount(v4 & f_packed_1[3])
						asm volatile(".byte  0xD7, 0x47, 0x4E, 0x06");
						// v16 = popcount(v4 & f_packed_1[2])
						asm volatile(".byte  0x57, 0xC8, 0x43, 0x06");
						// v17 = popcount(v4 & f_packed_1[1])
						asm volatile(".byte  0xD7, 0x48, 0x43, 0x06");
						// v18 = popcount(v4 & f_packed_1[0])
						asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
						
						////// REPLACE WITH VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v12, v12, 1");
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						#endif
						
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						/////////////////////////////////////
					}
					
					if(height < H_in - 2){
						// v13 = popcount(v5  & f_packed_1[6])
						asm volatile(".byte  0xD7, 0xC6, 0x5F, 0x06");
						// v14 = popcount(v5  & f_packed_1[5])
						asm volatile(".byte  0x57, 0x47, 0x5F, 0x06");
						// v15 = popcount(v5  & f_packed_1[4])
						asm volatile(".byte  0xD7, 0xC7, 0x5E, 0x06");
						// v16 = popcount(v5  & f_packed_1[3])
						asm volatile(".byte  0x57, 0x48, 0x5E, 0x06");
						// v17 = popcount(v5 & f_packed_1[2])
						asm volatile(".byte  0xD7, 0xC8, 0x53, 0x06");
						// v18 = popcount(v5 & f_packed_1[1])
						asm volatile(".byte  0x57, 0x49, 0x53, 0x06");
						// v19 = popcount(v5 & f_packed_1[0])
						asm volatile(".byte  0xD7, 0xC9, 0x52, 0x06");
						
						////// REPLACE WITH VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						#endif
								
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						/////////////////////////////////////
					}
					
					if(height < H_in - 3){
						// v14 = popcount(v6  & f_packed_1[6])
						asm volatile(".byte  0x57, 0xC7, 0x6F, 0x06");
						// v15 = popcount(v6  & f_packed_1[5])
						asm volatile(".byte  0xD7, 0x47, 0x6F, 0x06");
						// v16 = popcount(v6  & f_packed_1[4])
						asm volatile(".byte  0x57, 0xC8, 0x6E, 0x06");
						// v17 = popcount(v6  & f_packed_1[3])
						asm volatile(".byte  0xD7, 0x48, 0x6E, 0x06");
						// v18 = popcount(v6 & f_packed_1[2])
						asm volatile(".byte  0x57, 0xC9, 0x63, 0x06");
						// v19 = popcount(v6 & f_packed_1[1])
						asm volatile(".byte  0xD7, 0x49, 0x63, 0x06");
						// v20 = popcount(v6 & f_packed_1[0])
						asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");

						////// REPLACE WITH VSHACC   //////
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						asm volatile("vsll.vi v20, v20, 1");
						#endif
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
						/////////////////////////////////////
					}
									
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v6, v6, 1");	
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ +=  4 * C_in * W_in;
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v21, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                A2W1                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


// TBD

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                A2W2                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


void ibsconv2d32_W2_A2_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

// TO BE REDESIGNED


/*
int64_t const ch_loop = C_in >> 5;

int64_t const ldo = C_out * (W_in - 2);
uint64_t const stride_o = C_out << 2;

// number of elements is  [prec * F*F * C_in / 32]
int32_t f_packed[2 * 9 * ch_loop];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_2_to_32(f_, f_packed, F*F, C_in);




	//1ST LINE LSB = v2	MSB = v4
	//2ND LINE LSB = v6  MSB = v8
	//3RD LINE LSB = v10 MSB = v12


if(H_in >= 3 && W_in >= 3)
for (int width = 0 ; width < (W_in - 2) ; width += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	if(width > W_in - TILE_SIZE) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length


	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	
	
	int8_t *i__ = i_;
	
	for(int channels = 0 ; channels < ch_loop ; channels ++){
	
		bitpack32_vec_2_to_32_2H(i__ + 2 * C_in * W_in, vlen, C_in); 


		asm volatile("vmv.v.v v10, v2");
		asm volatile("vmv.v.v v12, v4");


		bitpack32_vec_2_to_32_2H(i__ + C_in * W_in, vlen, C_in); 
		

		asm volatile("vmv.v.v v6, v2");
		asm volatile("vmv.v.v v8, v4");


		bitpack32_vec_2_to_32_2H(i__, vlen, C_in); 

	
	
		for (int f_w = 0 ; f_w < F ; f_w += 1) { 
		
		// we have to iterate on a *2 index since the packed filter
		// alternate between LSB and MSB like this
		
		// the filter is packed in this way:
		//		32b	|	32b
		// [	0LSB, 	0MSB,		// initially filter[0] 
		//		1LSB,		1MSB,		// initially filter[1]
		//		...		...
		//		9LSB,		9MSB]		// initially filter[9]
		
		
		// since we call the va = popcount(vb & ra)
		// we need to use real registers
		// we use t0, t1, t2 for the 3 values at each iteration
		
			int32_t * f_loop = f_packed + (f_w << 1) + channels * 18;
		
		
			asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, -44"  : "+&r"(f_loop));
			
			
			// v14 = popcount(v2 & f_packed[1])
			asm volatile(".byte  0x57, 0xC7, 0x22, 0x06");
			// Pre-Calc
			// v20 = popcount(v6 & f_packed[1])
			asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
			// Pre-Calc
			// v24 = popcount(v10 & f_packed[1])
			asm volatile(".byte  0x57, 0xCC, 0xA2, 0x06");
			
			// v16 = popcount(v6 & f_packed[7])
			asm volatile(".byte  0x57, 0x48, 0x63, 0x06");
			// Pre-Calc
			// v22 = popcount(v10 & f_packed[7])
			asm volatile(".byte  0x57, 0x4B, 0xA3, 0x06");
			
			// v18 = popcount(v10 & f_packed[13])
			asm volatile(".byte  0x57, 0xC9, 0xA3, 0x06");
			

			if(f_w > 0 || channels > 0){
							
				asm volatile("vadd.vv v30, v30, v14");
				asm volatile("vadd.vv v28, v28, v20");
				asm volatile("vadd.vv v26, v26, v24");	
				
				asm volatile("vadd.vv v30, v30, v16");
				asm volatile("vadd.vv v28, v28, v22");
				
				asm volatile("vadd.vv v30, v30, v18");		
					
			}
			else
			{
				asm volatile("vadd.vv v30, v14, v16");
				asm volatile("vadd.vv v28, v20, v22");
				asm volatile("vmv.v.v v26, v24");	
				
				asm volatile("vadd.vv v30, v30, v18");

			}
				
				
			// LSB
			
			// v14 = popcount(v4 & f_packed[0])
			asm volatile(".byte  0x57, 0xC7, 0x42, 0x06");
			
			// Pre-Calc
			// v20 = popcount(v8 & f_packed[0])
			asm volatile(".byte  0x57, 0xca, 0x82, 0x06");
			// Pre-Calc
			// v24 = popcount(v12 & f_packed[0])
			asm volatile(".byte  0x57, 0xCC, 0xC2, 0x06");
			
			// v16 = popcount(v8 & f_packed[6])
			asm volatile(".byte  0x57, 0x48, 0x83, 0x06");
			// Pre-Calc
			// v22 = popcount(v12 & f_packed[6])
			asm volatile(".byte  0x57, 0x4B, 0xC3, 0x06");
			
			// v18 = popcount(v12 & f_packed[12])
			asm volatile(".byte  0x57, 0xC9, 0xC3, 0x06");
	
			
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v20, v20, 1");
			asm volatile("vsll.vi v24, v24, 1");
			
			asm volatile("vsll.vi v16, v16, 1");
			asm volatile("vsll.vi v22, v22, 1");
			
			asm volatile("vsll.vi v18, v18, 1");
			
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			asm volatile("vadd.vv v26, v26, v24");
			

			asm volatile("lw t0, (%0); addi %0, %0, 24" : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 24" : "+&r"(f_loop));
			asm volatile("lw t2, (%0);" 					  : "+&r"(f_loop));	
			
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v28, v28, v22");
				
			asm volatile("vadd.vv v30, v30, v18");	

			

			// MSB
			
			// v14 = popcount(v2 & f_packed[1])
			asm volatile(".byte  0x57, 0xC7, 0x22, 0x06");
			// Pre-Calc
			// v20 = popcount(v6 & f_packed[1])
			asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
			// Pre-Calc
			// v24 = popcount(v10 & f_packed[1])
			asm volatile(".byte  0x57, 0xCC, 0xA2, 0x06");
			
			// v16 = popcount(v6 & f_packed[7])
			asm volatile(".byte  0x57, 0x48, 0x63, 0x06");
			// Pre-Calc
			// v22 = popcount(v10 & f_packed[7])
			asm volatile(".byte  0x57, 0x4B, 0xA3, 0x06");
			
			// v18 = popcount(v10 & f_packed[13])
			asm volatile(".byte  0x57, 0xC9, 0xA3, 0x06");
			



			
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v20, v20, 1");
			asm volatile("vsll.vi v24, v24, 1");
			
			asm volatile("vsll.vi v16, v16, 1");
			asm volatile("vsll.vi v22, v22, 1");
			
			asm volatile("vsll.vi v18, v18, 1");
			
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			asm volatile("vadd.vv v26, v26, v24");	
			
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v28, v28, v22");
				
			asm volatile("vadd.vv v30, v30, v18");		
			
					
			// MSB
			
			// v14 = popcount(v4 & f_packed[0])
			asm volatile(".byte  0x57, 0xC7, 0x42, 0x06");
			
			// Pre-Calc
			// v20 = popcount(v8 & f_packed[0])
			asm volatile(".byte  0x57, 0xca, 0x82, 0x06");
			// Pre-Calc
			// v24 = popcount(v12 & f_packed[0])
			asm volatile(".byte  0x57, 0xCC, 0xC2, 0x06");
			

			// v16 = popcount(v8 & f_packed[6])
			asm volatile(".byte  0x57, 0x48, 0x83, 0x06");
			// Pre-Calc
			// v22 = popcount(v12 & f_packed[6])
			asm volatile(".byte  0x57, 0x4B, 0xC3, 0x06");
			

			// v18 = popcount(v12 & f_packed[12])
			asm volatile(".byte  0x57, 0xC9, 0xC3, 0x06");
			
			
			asm volatile("vsll.vi v14, v14, 2");
			asm volatile("vsll.vi v20, v20, 2");
			asm volatile("vsll.vi v24, v24, 2");
			
			asm volatile("vsll.vi v16, v16, 2");
			asm volatile("vsll.vi v22, v22, 2");
			
			asm volatile("vsll.vi v18, v18, 2");
			
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			asm volatile("vadd.vv v26, v26, v24");
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v2, v2, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v6, v6, 1");
				
				asm volatile("vslidedown.vi v8, v8, 1");
				asm volatile("vslidedown.vi v10, v10, 1");
				asm volatile("vslidedown.vi v12, v12, 1");
			}	
			
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v28, v28, v22");
				
			asm volatile("vadd.vv v30, v30, v18");
					
			
		}
			
			i__ += 32;
	}
		
		
				
	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen - 2));
	asm volatile("vsse32.v v30, (%0), %1" : "+&r"(o_) : "r"(stride_o));
		
	i_ += 3 * C_in * W_in;	
		

	// v26 (needs 2 more lines) and v28 (needs one more line) are used to store precalculated values		
		
	for (int height = 3 ; height < H_in - 2 ; height += 1){

			i__ = i_;
			o_ += ldo;
			
			for(int channels = 0 ; channels < ch_loop ; channels ++){

				bitpack32_vec_2_to_32_2H(i__, vlen, C_in); 
		
					
				for (int f_w = 0 ; f_w < F ; f_w += 1) {
				
					int32_t * f_loop = f_packed + (f_w << 1) + channels * 18;
				
				
					asm volatile("lw t0, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 24"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, -44" : "+&r"(f_loop));
				
					// LSB (activation bit 0)		
					// v14 = popcount(v2 & f_packed[12])
					asm volatile(".byte  0x57, 0xC7, 0x23, 0x06");
					// v16 = popcount(v2 & f_packed[6])
					asm volatile(".byte  0x57, 0x48, 0x23, 0x06");
					// v18 = popcount(v2 & f_packed[0])
					asm volatile(".byte  0x57, 0xC9, 0x22, 0x06");

					
					if(f_w > 0 || channels > 0){
						asm volatile("vadd.vv v30, v30, v14");
						asm volatile("vadd.vv v28, v28, v16");		
						asm volatile("vadd.vv v26, v26, v18");

					
					}
					else{
						asm volatile("vadd.vv v30, v28, v14");
						asm volatile("vadd.vv v28, v26, v16");		
						asm volatile("vmv.v.v v26, v18");
					}	
					
					
					// LSB (weights bit 0)
					// v14 = popcount(v4 & f_packed[12])
					asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
					// v16 = popcount(v4 & f_packed[6])
					asm volatile(".byte  0x57, 0x48, 0x43, 0x06");
					// v18 = popcount(v4 & f_packed[0])
					asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
				
				
					asm volatile("lw t0, (%0); addi %0, %0, 24" : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 24" : "+&r"(f_loop));
					asm volatile("lw t2, (%0);" 					  : "+&r"(f_loop));
			
			
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v16, v16, 1");
					asm volatile("vsll.vi v18, v18, 1");

					
					asm volatile("vadd.vv v30, v30, v14");
					asm volatile("vadd.vv v28, v28, v16");
					asm volatile("vadd.vv v26, v26, v18");	
													
							
							
					// MSB (activation bit 1)
					// v14 = popcount(v2 & f_packed[13])
					asm volatile(".byte  0x57, 0xC7, 0x23, 0x06");
					// v16 = popcount(v2 & f_packed[7])
					asm volatile(".byte  0x57, 0x48, 0x23, 0x06");
					// v18 = popcount(v2 & f_packed[1])
					asm volatile(".byte  0x57, 0xC9, 0x22, 0x06");
					
							
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v16, v16, 1");
					asm volatile("vsll.vi v18, v18, 1");

					
					asm volatile("vadd.vv v30, v30, v14");
					asm volatile("vadd.vv v28, v28, v16");
					asm volatile("vadd.vv v26, v26, v18");	
					
							
					// MSB (weights bit 1)
					// v14 = popcount(v4 & f_packed[13])
					asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
					// v16 = popcount(v4 & f_packed[7])
					asm volatile(".byte  0x57, 0x48, 0x43, 0x06");
					// v18 = popcount(v4 & f_packed[1])
					asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");

							
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v2, v2, 1");
						asm volatile("vslidedown.vi v4, v4, 1");
					}
						
							
					asm volatile("vsll.vi v14, v14, 2");
					asm volatile("vsll.vi v16, v16, 2");
					asm volatile("vsll.vi v18, v18, 2");

					
					asm volatile("vadd.vv v30, v30, v14");
					asm volatile("vadd.vv v28, v28, v16");
					asm volatile("vadd.vv v26, v26, v18");	
				
				
				}
				
				i__ += 32;	
			}
			
				
			i_ +=  C_in * W_in;
			
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vsse32.v v30, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			
			// last 2 lines (no need for pre calculation)
			
			if(H_in - 2 > 2){
				
				i__ = i_;
				o_ += ldo;
				
				for(int channels = 0 ; channels < ch_loop ; channels ++){
				
					bitpack32_vec_2_to_32_2H(i__, vlen, C_in); 					
					
					for (int f_w = 0 ; f_w < F ; f_w += 1) {
					
						int32_t * f_loop = f_packed + (f_w << 1) + 12 + channels * 18;
					
						asm volatile("lw t2, (%0); addi %0, %0, -24" : "+&r"(f_loop));
						asm volatile("lw t1, (%0); addi %0, %0, 28"  : "+&r"(f_loop));
					
						// LSB (activation bit 0)
						// v14 = popcount(v2 & f_packed[12])
						asm volatile(".byte  0x57, 0xC7, 0x23, 0x06");
						// v16 = popcount(v2 & f_packed[6])
						asm volatile(".byte  0x57, 0x48, 0x23, 0x06");
						
						// LSB (weights bit 0)
						// v18 = popcount(v4 & f_packed[12])
						asm volatile(".byte  0x57, 0xC9, 0x43, 0x06");
						// v20 = popcount(v4 & f_packed[6])
						asm volatile(".byte  0x57, 0x4A, 0x43, 0x06");
						
					
						asm volatile("lw t2, (%0); addi %0, %0, -24" : "+&r"(f_loop));
						asm volatile("lw t1, (%0)"  						: "+&r"(f_loop));
						
						if(f_w > 0 || channels > 0){
							asm volatile("vadd.vv v30, v30, v14");
							asm volatile("vadd.vv v28, v28, v16");
						}
						else{
							asm volatile("vadd.vv v30, v28, v14");
							asm volatile("vadd.vv v28, v26, v16");
						}
						
						
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v20, v20, 1");
						
						asm volatile("vadd.vv v30, v30, v18");
						asm volatile("vadd.vv v28, v28, v20");	
						
						
						
						// MSB (activation bit 1)
						// v14 = popcount(v2 & f_packed[13])
						asm volatile(".byte  0x57, 0xC7, 0x23, 0x06");
						// v16 = popcount(v2 & f_packed[7])
						asm volatile(".byte  0x57, 0x48, 0x23, 0x06"); 
						
						// MSB (weights bit 1)
						// v18 = popcount(v4 & f_packed[13])
						asm volatile(".byte  0x57, 0xC9, 0x43, 0x06");
						// v20 = popcount(v4 & f_packed[7])
						asm volatile(".byte  0x57, 0x4A, 0x43, 0x06"); 
						
						
						if(f_w < F - 1){
							asm volatile("vslidedown.vi v2, v2, 1");
							asm volatile("vslidedown.vi v4, v4, 1");
						}
						
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v16, v16, 1");
						
						asm volatile("vsll.vi v18, v18, 2");
						asm volatile("vsll.vi v20, v20, 2");
						
						asm volatile("vadd.vv v30, v30, v14");
						asm volatile("vadd.vv v28, v28, v16");
						
						asm volatile("vadd.vv v30, v30, v18");
						asm volatile("vadd.vv v28, v28, v20");	
							
					}
					i__ += 32;
				}
					
				i_ += C_in * W_in;
				
				asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen - 2));
					
				asm volatile("vsse32.v v30, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(H_in - 2 > 1){
			
							
				i__ = i_;
				o_ += ldo;
				
				for(int channels = 0 ; channels < ch_loop ; channels ++){
				
					bitpack32_vec_2_to_32_2H(i__, vlen, C_in); 
						
					for (int f_w = 0 ; f_w < F ; f_w += 1) { 
					
						int32_t * f_loop = f_packed + (f_w << 1) + 12 + channels * 18;
						
						asm volatile("lw t2, (%0); addi %0, %0, 4" : "+&r"(f_loop));
						asm volatile("lw t1, (%0)"  : "+&r"(f_loop));
					
						// LSB (activation bit 0)
						// v14 = popcount(v2 & f_packed[12])
						asm volatile(".byte  0x57, 0xC7, 0x23, 0x06");
						
						// LSB (weights bit 0)
						// v16 = popcount(v4 & f_packed[12])
						asm volatile(".byte  0x57, 0xC8, 0x43, 0x06"); 

						// MSB (activation bit 1)
						// v18 = popcount(v2 & f_packed[13])
						asm volatile(".byte  0x57, 0x49, 0x23, 0x06");

						// MSB (weights bit 1)				
						// v20 = popcount(v4 & f_packed[13])
						asm volatile(".byte  0x57, 0x4A, 0x43, 0x06");

						
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v30, v30, v14");
						else
							asm volatile("vadd.vv v30, v28, v14");
							
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v20, v20, 2");
						
						asm volatile("vadd.vv v30, v30, v16");
						
						if(f_w < F - 1){
							asm volatile("vslidedown.vi v2, v2, 1");
							asm volatile("vslidedown.vi v4, v4, 1");
						}
						
						asm volatile("vadd.vv v30, v30, v18");
						asm volatile("vadd.vv v30, v30, v20");	
					
						}	
					i__ += 32;
					}
									
				asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen - 2));
					
				asm volatile("vsse32.v v30, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}

	}*/

}


void ibsconv2d32_W2_A2_vec_7x7(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int64_t const ldo = C_out * (W_in - 6);
uint64_t const stride_o = C_out << 2;//C_out * (W_in - 2);

// number of elements is  [precW * (F*F + 1)* C_in / 32]
int32_t f_packed[2 * F * F * (C_in >> 5)];

int64_t vlen; //for now

int8_t * f_ = f_ptr;



bitpack_filter32_vec_2_to_32(f_, f_packed, F*F, C_in);

if(H_in >= 7 && W_in >= 7)
for (int width = 0 ; width < (W_in - 6) ; width += TILE_SIZE_A2_7x7_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t *i_ = i_ptr + width * C_in; 									// input pointer realtive to the tile (constant throughout the tile)
	int32_t *o_ = o_ptr + width * C_out;									// output pointer relative to the tile	

	if(width > W_in - TILE_SIZE_A2_7x7) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE_A2_7x7_OUT;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE_A2_7x7;						// else we go full length
	
	int8_t *i__ = i_;
	
	int32_t * f_loop = f_packed;
	
	for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
	
		f_loop = f_packed + 2 * channels * F * F;

		bitpack32_vec_2_to_32_6H(i__, vlen, C_in, W_in); 
		
		for(int f_w = 0; f_w < 7 ; f_w ++){		
			
			// Weight LSB and Activation LSB
			asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t5, (%0); addi %0, %0, -276" : "+&r"(f_loop));

			if(f_w > 0 || channels > 0){ 
				
				// v12 = popcount(v0  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC6, 0x02, 0x06");
				// v13 = popcount(v1  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC6, 0x12, 0x06");
				// v14 = popcount(v2  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC7, 0x22, 0x06");
				// v15 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC7, 0x32, 0x06");
				// v16 = popcount(v4 & f_packed_0[0])
				asm volatile(".byte  0x57, 0xC8, 0x42, 0x06");
				// v17 = popcount(v5 & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xC8, 0x52, 0x06"); 
				
				asm volatile("vadd.vv v26, v26, v12");
				asm volatile("vadd.vv v27, v27, v13");
				asm volatile("vadd.vv v28, v28, v14");
				asm volatile("vadd.vv v29, v29, v15");
				asm volatile("vadd.vv v30, v30, v16");
				asm volatile("vadd.vv v31, v31, v17");
			}
			else{  
			//	to place the first value in the right register on first execution
			// avoid the sum step and the reset of vectors
			
				// v26 = popcount(v0  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCD, 0x02, 0x06");
				// v27 = popcount(v1  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCD, 0x12, 0x06");
				// v28 = popcount(v2  & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCE, 0x22, 0x06");
				// v29 = popcount(v3  & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCE, 0x32, 0x06");
				// v30 = popcount(v4 & f_packed_0[0])
				asm volatile(".byte  0x57, 0xCF, 0x42, 0x06");
				// v31 = popcount(v5 & f_packed_0[0])
				asm volatile(".byte  0xD7, 0xCF, 0x52, 0x06"); 
				
				// no addition because values are already in the right vd
			}

			// v12 = popcount(v1  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x46, 0x13, 0x06");
			// v13 = popcount(v2  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x46, 0x23, 0x06");
			// v14 = popcount(v3  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x47, 0x33, 0x06");
			// v15 = popcount(v4  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x47, 0x43, 0x06");
			// v16 = popcount(v5  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x48, 0x53, 0x06"); 
			
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			
			
			// v12 = popcount(v2  & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC6, 0x23, 0x06");
			// v13 = popcount(v3  & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC6, 0x33, 0x06");
	   	// v14 = popcount(v4  & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
			// v15 = popcount(v5  & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC7, 0x53, 0x06");
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			
			// v12 = popcount(v3  & f_packed_0[3])
			asm volatile(".byte  0x57, 0x46, 0x3E, 0x06");
			// v13 = popcount(v4  & f_packed_0[3])
			asm volatile(".byte  0xD7, 0x46, 0x4E, 0x06");
			// v14 = popcount(v5  & f_packed_0[3])
			asm volatile(".byte  0x57, 0x47, 0x5E, 0x06");
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			
			// v12 = popcount(v4  & f_packed_0[4])
			asm volatile(".byte  0x57, 0xC6, 0x4E, 0x06");
			// v13 = popcount(v5  & f_packed_0[4])
			asm volatile(".byte  0xD7, 0xC6, 0x5E, 0x06");
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			
			// v12 = popcount(v5  & f_packed_0[5])
			asm volatile(".byte  0x57, 0x46, 0x5F, 0x06");
				
			asm volatile("vadd.vv v26, v26, v12");
			
			
			
			
			
			
			
			
			
			// Weight LSB and Activation bit 1
			// so we don't have to reload filter
			
			// v12 = popcount(v6  & f_packed_0[0])
			asm volatile(".byte  0x57, 0xC6, 0x62, 0x06");
			// v13 = popcount(v7  & f_packed_0[0])
			asm volatile(".byte  0xD7, 0xC6, 0x72, 0x06");
			// v14 = popcount(v8  & f_packed_0[0])
			asm volatile(".byte  0x57, 0xC7, 0x82, 0x06");
			// v15 = popcount(v9  & f_packed_0[0])
			asm volatile(".byte  0xD7, 0xC7, 0x92, 0x06");
			// v16 = popcount(v10 & f_packed_0[0])
			asm volatile(".byte  0x57, 0xC8, 0xA2, 0x06");
			// v17 = popcount(v11 & f_packed_0[0])
			asm volatile(".byte  0xD7, 0xC8, 0xB2, 0x06");
			
			////// REPLACE WITH 6 VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			asm volatile("vsll.vi v16, v16, 1");
			asm volatile("vsll.vi v17, v17, 1");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v31, v31, v17");
			/////////////////////////////////////
			
			// v12 = popcount(v7  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x46, 0x73, 0x06");
			// v13 = popcount(v8  & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x46, 0x83, 0x06");
			// v14 = popcount(v9  & f_packed_0[1])
			asm volatile(".byte  0x57, 0x47, 0x93, 0x06");
			// v15 = popcount(v10 & f_packed_0[1])
			asm volatile(".byte  0xD7, 0x47, 0xA3, 0x06");
			// v16 = popcount(v11 & f_packed_0[1])
			asm volatile(".byte  0x57, 0x48, 0xB3, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			asm volatile("vsll.vi v16, v16, 1");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			//////////////////////////////////////
			
			// v12 = popcount(v8  & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC6, 0x83, 0x06");
			// v13 = popcount(v9  & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC6, 0x93, 0x06");
			// v14 = popcount(v10 & f_packed_0[2])
			asm volatile(".byte  0x57, 0xC7, 0xA3, 0x06");
			// v15 = popcount(v11 & f_packed_0[2])
			asm volatile(".byte  0xD7, 0xC7, 0xB3, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			////////////////////////////////////
			
			// v12 = popcount(v9  & f_packed_0[3])
			asm volatile(".byte  0x57, 0x46, 0x9E, 0x06");
			// v13 = popcount(v10 & f_packed_0[3])
			asm volatile(".byte  0xD7, 0x46, 0xAE, 0x06");
			// v14 = popcount(v11 & f_packed_0[3])
			asm volatile(".byte  0x57, 0x47, 0xBE, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			/////////////////////////////////////
			
			// v12 = popcount(v10  & f_packed_0[4])
			asm volatile(".byte  0x57, 0xC6, 0xAE, 0x06");
			// v13 = popcount(v11  & f_packed_0[4])
			asm volatile(".byte  0xD7, 0xC6, 0xBE, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			/////////////////////////////////////
			
			// v12 = popcount(v11  & f_packed_0[5])
			asm volatile(".byte  0x57, 0x46, 0xBF, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			///////////////////////////////////
			
			

			
			
			
			
			
			
			
			
			// Weight bit 1 and Activation LSB	
			asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
			asm volatile("lw t5, (%0); addi %0, %0, -276" : "+&r"(f_loop));
			
			// v12 = popcount(v0  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC6, 0x02, 0x06");
			// v13 = popcount(v1  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC6, 0x12, 0x06");
			// v14 = popcount(v2  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC7, 0x22, 0x06");
			// v15 = popcount(v3  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC7, 0x32, 0x06");
			// v16 = popcount(v4  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC8, 0x42, 0x06");
			// v17 = popcount(v5  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC8, 0x52, 0x06");
			
			////// REPLACE WITH 6 VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			asm volatile("vsll.vi v16, v16, 1");
			asm volatile("vsll.vi v17, v17, 1");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v31, v31, v17");
			/////////////////////////////////////
			
			// v12 = popcount(v1  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x46, 0x13, 0x06");
			// v13 = popcount(v2  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x46, 0x23, 0x06");
			// v14 = popcount(v3  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x47, 0x33, 0x06");
			// v15 = popcount(v4  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x47, 0x43, 0x06");
			// v16 = popcount(v5  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x48, 0x53, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			asm volatile("vsll.vi v16, v16, 1");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			//////////////////////////////////////
			
			// v12 = popcount(v2  & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC6, 0x23, 0x06");
			// v13 = popcount(v3  & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC6, 0x33, 0x06");
			// v14 = popcount(v4  & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC7, 0x43, 0x06");
			// v15 = popcount(v5  & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC7, 0x53, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			asm volatile("vsll.vi v15, v15, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			////////////////////////////////////
			
			// v12 = popcount(v3  & f_packed_1[3])
			asm volatile(".byte  0x57, 0x46, 0x3E, 0x06");
			// v13 = popcount(v4  & f_packed_1[3])
			asm volatile(".byte  0xD7, 0x46, 0x4E, 0x06");
			// v14 = popcount(v5  & f_packed_1[3])
			asm volatile(".byte  0x57, 0x47, 0x5E, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			asm volatile("vsll.vi v14, v14, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			/////////////////////////////////////
			
			// v12 = popcount(v4  & f_packed_1[4])
			asm volatile(".byte  0x57, 0xC6, 0x4E, 0x06");
			// v13 = popcount(v5  & f_packed_1[4])
			asm volatile(".byte  0xD7, 0xC6, 0x5E, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			asm volatile("vsll.vi v13, v13, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			/////////////////////////////////////
			
			// v12 = popcount(v5  & f_packed_1[5])
			asm volatile(".byte  0x57, 0x46, 0x5F, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 1");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			///////////////////////////////////

			
			
			
			
			
			
			
			
			// Weight bit 1 and Activation bit 1
			// so we don't have to reload filter
			// v12 = popcount(v6  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC6, 0x62, 0x06");
			// v13 = popcount(v7  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC6, 0x72, 0x06");
			// v14 = popcount(v8  & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC7, 0x82, 0x06");
			// v15 = popcount(v9  & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC7, 0x92, 0x06");
			// v16 = popcount(v10 & f_packed_1[0])
			asm volatile(".byte  0x57, 0xC8, 0xA2, 0x06");
			// v17 = popcount(v11 & f_packed_1[0])
			asm volatile(".byte  0xD7, 0xC8, 0xB2, 0x06");
			
			////// REPLACE WITH 6 VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			asm volatile("vsll.vi v13, v13, 2");
			asm volatile("vsll.vi v14, v14, 2");
			asm volatile("vsll.vi v15, v15, 2");
			asm volatile("vsll.vi v16, v16, 2");
			asm volatile("vsll.vi v17, v17, 2");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			asm volatile("vadd.vv v31, v31, v17");
			/////////////////////////////////////
			
			// v12 = popcount(v7  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x46, 0x73, 0x06");
			// v13 = popcount(v8  & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x46, 0x83, 0x06");
			// v14 = popcount(v9  & f_packed_1[1])
			asm volatile(".byte  0x57, 0x47, 0x93, 0x06");
			// v15 = popcount(v10 & f_packed_1[1])
			asm volatile(".byte  0xD7, 0x47, 0xA3, 0x06");
			// v16 = popcount(v11 & f_packed_1[1])
			asm volatile(".byte  0x57, 0x48, 0xB3, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			asm volatile("vsll.vi v13, v13, 2");
			asm volatile("vsll.vi v14, v14, 2");
			asm volatile("vsll.vi v15, v15, 2");
			asm volatile("vsll.vi v16, v16, 2");
			#endif
			
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			asm volatile("vadd.vv v30, v30, v16");
			//////////////////////////////////////
			
			// v12 = popcount(v8  & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC6, 0x83, 0x06");
			// v13 = popcount(v9  & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC6, 0x93, 0x06");
			// v14 = popcount(v10 & f_packed_1[2])
			asm volatile(".byte  0x57, 0xC7, 0xA3, 0x06");
			// v15 = popcount(v11 & f_packed_1[2])
			asm volatile(".byte  0xD7, 0xC7, 0xB3, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			asm volatile("vsll.vi v13, v13, 2");
			asm volatile("vsll.vi v14, v14, 2");
			asm volatile("vsll.vi v15, v15, 2");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			asm volatile("vadd.vv v29, v29, v15");
			////////////////////////////////////
			
			// v12 = popcount(v9  & f_packed_1[3])
			asm volatile(".byte  0x57, 0x46, 0x9E, 0x06");
			// v13 = popcount(v10 & f_packed_1[3])
			asm volatile(".byte  0xD7, 0x46, 0xAE, 0x06");
			// v14 = popcount(v11 & f_packed_1[3])
			asm volatile(".byte  0x57, 0x47, 0xBE, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			asm volatile("vsll.vi v13, v13, 2");
			asm volatile("vsll.vi v14, v14, 2");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			asm volatile("vadd.vv v28, v28, v14");
			/////////////////////////////////////
			
			// v12 = popcount(v10  & f_packed_1[4])
			asm volatile(".byte  0x57, 0xC6, 0xAE, 0x06");
			// v13 = popcount(v11  & f_packed_1[4])
			asm volatile(".byte  0xD7, 0xC6, 0xBE, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			asm volatile("vsll.vi v13, v13, 2");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			asm volatile("vadd.vv v27, v27, v13");
			/////////////////////////////////////
			
			// v12 = popcount(v11  & f_packed_1[5])
			asm volatile(".byte  0x57, 0x46, 0xBF, 0x06");
			
			////// REPLACE WITH VSHACC   //////
			#ifndef PERF_VHSACC
			asm volatile("vsll.vi v12, v12, 2");
			#endif
				
			asm volatile("vadd.vv v26, v26, v12");
			///////////////////////////////////
			
			
						
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v0, v0, 1");
				asm volatile("vslidedown.vi v1, v1, 1");
				asm volatile("vslidedown.vi v2, v2, 1");
				asm volatile("vslidedown.vi v3, v3, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				asm volatile("vslidedown.vi v5, v5, 1");
				
				asm volatile("vslidedown.vi v6, v6, 1");
				asm volatile("vslidedown.vi v7, v7, 1");
				asm volatile("vslidedown.vi v8, v8, 1");
				asm volatile("vslidedown.vi v9, v9, 1");
				asm volatile("vslidedown.vi v10, v10, 1");
				asm volatile("vslidedown.vi v11, v11, 1");
			}

				
			}
			
			i__ += ENCOD_SIZE;
		}
		
		i_ += (F - 1) * C_in * W_in;	

		
		for (int height = 6 ; height < H_in ; height += 4){

			f_loop = f_packed;
			
			i__ = i_;
			
			
			for(int channels = 0 ; channels < (C_in >> 5) ; channels ++){
			
				f_loop = f_packed + 2 * channels * F * F;

				bitpack32_vec_2_to_32_4H(i__, vlen, C_in, W_in); 
				

				for(int f_w = 0; f_w < 7 ; f_w ++){		
				
					// Weight LSB and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t5, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t6, (%0); addi %0, %0, -332" : "+&r"(f_loop));
					
					
					// v12 = popcount(v0  & f_packed_0[6])
					asm volatile(".byte  0x57, 0xC6, 0x0F, 0x06");
					// v13 = popcount(v0  & f_packed_0[5])
					asm volatile(".byte  0xD7, 0x46, 0x0F, 0x06");
					// v14 = popcount(v0  & f_packed_0[4])
					asm volatile(".byte  0x57, 0xC7, 0x0E, 0x06");
					// v15 = popcount(v0  & f_packed_0[3])
					asm volatile(".byte  0xD7, 0x47, 0x0E, 0x06");
					// v16 = popcount(v0 & f_packed_0[2])
					asm volatile(".byte  0x57, 0xC8, 0x03, 0x06");
					// v17 = popcount(v0 & f_packed_0[1])
					asm volatile(".byte  0xD7, 0x48, 0x03, 0x06");
					if(f_w > 0 || channels > 0)
						// v18 = popcount(v0 & f_packed_0[0])
						asm volatile(".byte  0x57, 0xC9, 0x02, 0x06");
					
					if(f_w > 0 || channels > 0){
						asm volatile("vadd.vv v22, v22, v12");
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
					}
					else{
						asm volatile("vadd.vv v22, v26, v12");
						asm volatile("vadd.vv v23, v27, v13");
						asm volatile("vadd.vv v24, v28, v14");
						asm volatile("vadd.vv v25, v29, v15");
						asm volatile("vadd.vv v26, v30, v16");
						asm volatile("vadd.vv v27, v31, v17");
						
						// v28 = popcount(v0 & f_packed[0])
						asm volatile(".byte  0x57, 0xCE, 0x02, 0x06");
					}
					
					if(height < H_in - 1){
						// v13 = popcount(v1 & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC6, 0x1F, 0x06");
						// v14 = popcount(v1 & f_packed_0[5])
						asm volatile(".byte  0x57, 0x47, 0x1F, 0x06");
						// v15 = popcount(v1 & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC7, 0x1E, 0x06");
						// v16 = popcount(v1 & f_packed_0[3])
						asm volatile(".byte  0x57, 0x48, 0x1E, 0x06");
						// v17 = popcount(v1 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC8, 0x13, 0x06");
						// v18 = popcount(v1 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x49, 0x13, 0x06");
							
						if(f_w > 0 || channels > 0)
							// v19 = popcount(v1 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xC9, 0x12, 0x06");
						else
							// v29 = popcount(v1 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCE, 0x12, 0x06");
					
					
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v29, v29, v19");
					}
						
					if(height < H_in - 2){
						// v14 = popcount(v2  & f_packed_0[6])
						asm volatile(".byte  0x57, 0xC7, 0x2F, 0x06");
						// v15 = popcount(v2  & f_packed_0[5])
						asm volatile(".byte  0xD7, 0x47, 0x2F, 0x06");
						// v16 = popcount(v2  & f_packed_0[4])
						asm volatile(".byte  0x57, 0xC8, 0x2E, 0x06");
						// v17 = popcount(v2  & f_packed_0[3])
						asm volatile(".byte  0xD7, 0x48, 0x2E, 0x06");
						// v18 = popcount(v2 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC9, 0x23, 0x06");
						// v19 = popcount(v2 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x49, 0x23, 0x06");
						if(f_w > 0 || channels > 0)
							// v20 = popcount(v2 & f_packed_0[0])
							asm volatile(".byte  0x57, 0xCA, 0x22, 0x06");
						else
							// v30 = popcount(v2 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCF, 0x22, 0x06");	
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v30, v30, v20");
					}
					
					if(height < H_in - 3){
						// v15 = popcount(v3  & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC7, 0x3F, 0x06");
						// v16 = popcount(v3  & f_packed_0[5])
						asm volatile(".byte  0x57, 0x48, 0x3F, 0x06");
						// v17 = popcount(v3  & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC8, 0x3E, 0x06");
						// v18 = popcount(v3  & f_packed_0[3])
						asm volatile(".byte  0x57, 0x49, 0x3E, 0x06");
						// v19 = popcount(v3 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC9, 0x33, 0x06");
						// v20 = popcount(v3 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x4A, 0x33, 0x06");
						if(f_w > 0 || channels > 0)
							// v21 = popcount(v3 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCA, 0x32, 0x06");
						else
							// v31 = popcount(v3 & f_packed_0[0])
							asm volatile(".byte  0xD7, 0xCF, 0x32, 0x06");	
						
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
						if(f_w > 0 || channels > 0)
							asm volatile("vadd.vv v31, v31, v21");
					}
					
					
					
					
					
					
					
					
					// Weight LSB and Activation bit 1
					// v12 = popcount(v4  & f_packed_0[6])
					asm volatile(".byte  0x57, 0xC6, 0x4F, 0x06");
					// v13 = popcount(v4  & f_packed_0[5])
					asm volatile(".byte  0xD7, 0x46, 0x4F, 0x06");
					// v14 = popcount(v4  & f_packed_0[4])
					asm volatile(".byte  0x57, 0xC7, 0x4E, 0x06");
					// v15 = popcount(v4  & f_packed_0[3])
					asm volatile(".byte  0xD7, 0x47, 0x4E, 0x06");
					// v16 = popcount(v4 & f_packed_0[2])
					asm volatile(".byte  0x57, 0xC8, 0x43, 0x06");
					// v17 = popcount(v4 & f_packed_0[1])
					asm volatile(".byte  0xD7, 0x48, 0x43, 0x06");
					// v18 = popcount(v4 & f_packed_0[0])
					asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
					
					#ifndef PERF_VHSACC
					asm volatile("vsll.vi v12, v12, 1");
					asm volatile("vsll.vi v13, v13, 1");
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v15, v15, 1");
					asm volatile("vsll.vi v16, v16, 1");
					asm volatile("vsll.vi v17, v17, 1");
					asm volatile("vsll.vi v18, v18, 1");
					#endif

					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					asm volatile("vadd.vv v24, v24, v14");
					asm volatile("vadd.vv v25, v25, v15");
					asm volatile("vadd.vv v26, v26, v16");
					asm volatile("vadd.vv v27, v27, v17");
					asm volatile("vadd.vv v28, v28, v18");
					
					if(height < H_in - 1){
						// v13 = popcount(v5 & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC6, 0x5F, 0x06");
						// v14 = popcount(v5 & f_packed_0[5])
						asm volatile(".byte  0x57, 0x47, 0x5F, 0x06");
						// v15 = popcount(v5 & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC7, 0x5E, 0x06");
						// v16 = popcount(v5 & f_packed_0[3])
						asm volatile(".byte  0x57, 0x48, 0x5E, 0x06");
						// v17 = popcount(v5 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC8, 0x53, 0x06");
						// v18 = popcount(v5 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x49, 0x53, 0x06");
						// v19 = popcount(v5 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xC9, 0x52, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						#endif

						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						
					}
						
					if(height < H_in - 2){
						// v14 = popcount(v6  & f_packed_0[6])
						asm volatile(".byte  0x57, 0xC7, 0x6F, 0x06");
						// v15 = popcount(v6  & f_packed_0[5])
						asm volatile(".byte  0xD7, 0x47, 0x6F, 0x06");
						// v16 = popcount(v6  & f_packed_0[4])
						asm volatile(".byte  0x57, 0xC8, 0x6E, 0x06");
						// v17 = popcount(v6  & f_packed_0[3])
						asm volatile(".byte  0xD7, 0x48, 0x6E, 0x06");
						// v18 = popcount(v6 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC9, 0x63, 0x06");
						// v19 = popcount(v6 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x49, 0x63, 0x06");
						// v20 = popcount(v6 & f_packed_0[0])
						asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						asm volatile("vsll.vi v20, v20, 1");
						#endif
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
					}
					
					if(height < H_in - 3){
						// v15 = popcount(v7  & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC7, 0x7F, 0x06");
						// v16 = popcount(v7  & f_packed_0[5])
						asm volatile(".byte  0x57, 0x48, 0x7F, 0x06");
						// v17 = popcount(v7  & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC8, 0x7E, 0x06");
						// v18 = popcount(v7  & f_packed_0[3])
						asm volatile(".byte  0x57, 0x49, 0x7E, 0x06");
						// v19 = popcount(v7 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC9, 0x73, 0x06");
						// v20 = popcount(v7 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x4A, 0x73, 0x06");
						// v21 = popcount(v7 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xCA, 0x72, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						asm volatile("vsll.vi v20, v20, 1");
						asm volatile("vsll.vi v21, v21, 1");
						#endif
						
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
						asm volatile("vadd.vv v31, v31, v21");
					}
					
					
					
					
					
					
					
					// Weight LSB and Activation LSB
					asm volatile("lw t0, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t1, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t2, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t3, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t4, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t5, (%0); addi %0, %0, 56"   : "+&r"(f_loop));
					asm volatile("lw t6, (%0); addi %0, %0, -332" : "+&r"(f_loop));
					
					
					// v12 = popcount(v0  & f_packed_0[6])
					asm volatile(".byte  0x57, 0xC6, 0x0F, 0x06");
					// v13 = popcount(v0  & f_packed_0[5])
					asm volatile(".byte  0xD7, 0x46, 0x0F, 0x06");
					// v14 = popcount(v0  & f_packed_0[4])
					asm volatile(".byte  0x57, 0xC7, 0x0E, 0x06");
					// v15 = popcount(v0  & f_packed_0[3])
					asm volatile(".byte  0xD7, 0x47, 0x0E, 0x06");
					// v16 = popcount(v0 & f_packed_0[2])
					asm volatile(".byte  0x57, 0xC8, 0x03, 0x06");
					// v17 = popcount(v0 & f_packed_0[1])
					asm volatile(".byte  0xD7, 0x48, 0x03, 0x06");
					// v18 = popcount(v0 & f_packed_0[0])
					asm volatile(".byte  0x57, 0xC9, 0x02, 0x06");
					
					#ifndef PERF_VHSACC
					asm volatile("vsll.vi v12, v12, 1");
					asm volatile("vsll.vi v13, v13, 1");
					asm volatile("vsll.vi v14, v14, 1");
					asm volatile("vsll.vi v15, v15, 1");
					asm volatile("vsll.vi v16, v16, 1");
					asm volatile("vsll.vi v17, v17, 1");
					asm volatile("vsll.vi v18, v18, 1");
					#endif
					
					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					asm volatile("vadd.vv v24, v24, v14");
					asm volatile("vadd.vv v25, v25, v15");
					asm volatile("vadd.vv v26, v26, v16");
					asm volatile("vadd.vv v27, v27, v17");
					asm volatile("vadd.vv v28, v28, v18");
					
					if(height < H_in - 1){
						// v13 = popcount(v1 & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC6, 0x1F, 0x06");
						// v14 = popcount(v1 & f_packed_0[5])
						asm volatile(".byte  0x57, 0x47, 0x1F, 0x06");
						// v15 = popcount(v1 & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC7, 0x1E, 0x06");
						// v16 = popcount(v1 & f_packed_0[3])
						asm volatile(".byte  0x57, 0x48, 0x1E, 0x06");
						// v17 = popcount(v1 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC8, 0x13, 0x06");
						// v18 = popcount(v1 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x49, 0x13, 0x06");
						// v19 = popcount(v1 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xC9, 0x12, 0x06");

						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 1");
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						#endif
					
					
						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
					}
						
					if(height < H_in - 2){
						// v14 = popcount(v2  & f_packed_0[6])
						asm volatile(".byte  0x57, 0xC7, 0x2F, 0x06");
						// v15 = popcount(v2  & f_packed_0[5])
						asm volatile(".byte  0xD7, 0x47, 0x2F, 0x06");
						// v16 = popcount(v2  & f_packed_0[4])
						asm volatile(".byte  0x57, 0xC8, 0x2E, 0x06");
						// v17 = popcount(v2  & f_packed_0[3])
						asm volatile(".byte  0xD7, 0x48, 0x2E, 0x06");
						// v18 = popcount(v2 & f_packed_0[2])
						asm volatile(".byte  0x57, 0xC9, 0x23, 0x06");
						// v19 = popcount(v2 & f_packed_0[1])
						asm volatile(".byte  0xD7, 0x49, 0x23, 0x06");
						// v20 = popcount(v2 & f_packed_0[0])
						asm volatile(".byte  0x57, 0xCA, 0x22, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 1");
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						asm volatile("vsll.vi v20, v20, 1");
						#endif
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
					}
					
					if(height < H_in - 3){
						// v15 = popcount(v3  & f_packed_0[6])
						asm volatile(".byte  0xD7, 0xC7, 0x3F, 0x06");
						// v16 = popcount(v3  & f_packed_0[5])
						asm volatile(".byte  0x57, 0x48, 0x3F, 0x06");
						// v17 = popcount(v3  & f_packed_0[4])
						asm volatile(".byte  0xD7, 0xC8, 0x3E, 0x06");
						// v18 = popcount(v3  & f_packed_0[3])
						asm volatile(".byte  0x57, 0x49, 0x3E, 0x06");
						// v19 = popcount(v3 & f_packed_0[2])
						asm volatile(".byte  0xD7, 0xC9, 0x33, 0x06");
						// v20 = popcount(v3 & f_packed_0[1])
						asm volatile(".byte  0x57, 0x4A, 0x33, 0x06");
						// v21 = popcount(v3 & f_packed_0[0])
						asm volatile(".byte  0xD7, 0xCA, 0x32, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v15, v15, 1");
						asm volatile("vsll.vi v16, v16, 1");
						asm volatile("vsll.vi v17, v17, 1");
						asm volatile("vsll.vi v18, v18, 1");
						asm volatile("vsll.vi v19, v19, 1");
						asm volatile("vsll.vi v20, v20, 1");
						asm volatile("vsll.vi v21, v21, 1");
						#endif
						
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
						asm volatile("vadd.vv v31, v31, v21");
					}
						
						
						
						
						
						
					// Weight bit 1 and Activation bit 1
					// v12 = popcount(v4 & f_packed_1[6])
					asm volatile(".byte  0x57, 0xC6, 0x4F, 0x06");
					// v13 = popcount(v4 & f_packed_1[5])
					asm volatile(".byte  0xD7, 0x46, 0x4F, 0x06");
					// v14 = popcount(v4 & f_packed_1[4])
					asm volatile(".byte  0x57, 0xC7, 0x4E, 0x06");
					// v15 = popcount(v4 & f_packed_1[3])
					asm volatile(".byte  0xD7, 0x47, 0x4E, 0x06");
					// v16 = popcount(v4 & f_packed_1[2])
					asm volatile(".byte  0x57, 0xC8, 0x43, 0x06");
					// v17 = popcount(v4 & f_packed_1[1])
					asm volatile(".byte  0xD7, 0x48, 0x43, 0x06");
					// v18 = popcount(v4 & f_packed_1[0])
					asm volatile(".byte  0x57, 0xC9, 0x42, 0x06");
					
					#ifndef PERF_VHSACC
					asm volatile("vsll.vi v12, v12, 2");
					asm volatile("vsll.vi v13, v13, 2");
					asm volatile("vsll.vi v14, v14, 2");
					asm volatile("vsll.vi v15, v15, 2");
					asm volatile("vsll.vi v16, v16, 2");
					asm volatile("vsll.vi v17, v17, 2");
					asm volatile("vsll.vi v18, v18, 2");
					#endif

					asm volatile("vadd.vv v22, v22, v12");
					asm volatile("vadd.vv v23, v23, v13");
					asm volatile("vadd.vv v24, v24, v14");
					asm volatile("vadd.vv v25, v25, v15");
					asm volatile("vadd.vv v26, v26, v16");
					asm volatile("vadd.vv v27, v27, v17");
					asm volatile("vadd.vv v28, v28, v18");
					
					if(height < H_in - 1){
						// v13 = popcount(v5 & f_packed_1[6])
						asm volatile(".byte  0xD7, 0xC6, 0x5F, 0x06");
						// v14 = popcount(v5 & f_packed_1[5])
						asm volatile(".byte  0x57, 0x47, 0x5F, 0x06");
						// v15 = popcount(v5 & f_packed_1[4])
						asm volatile(".byte  0xD7, 0xC7, 0x5E, 0x06");
						// v16 = popcount(v5 & f_packed_1[3])
						asm volatile(".byte  0x57, 0x48, 0x5E, 0x06");
						// v17 = popcount(v5 & f_packed_1[2])
						asm volatile(".byte  0xD7, 0xC8, 0x53, 0x06");
						// v18 = popcount(v5 & f_packed_1[1])
						asm volatile(".byte  0x57, 0x49, 0x53, 0x06");
						// v19 = popcount(v5 & f_packed_1[0])
						asm volatile(".byte  0xD7, 0xC9, 0x52, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v13, v13, 2");
						asm volatile("vsll.vi v14, v14, 2");
						asm volatile("vsll.vi v15, v15, 2");
						asm volatile("vsll.vi v16, v16, 2");
						asm volatile("vsll.vi v17, v17, 2");
						asm volatile("vsll.vi v18, v18, 2");
						asm volatile("vsll.vi v19, v19, 2");
						#endif

						asm volatile("vadd.vv v23, v23, v13");
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						
					}
						
					if(height < H_in - 2){
						// v14 = popcount(v6 & f_packed_1[6])
						asm volatile(".byte  0x57, 0xC7, 0x6F, 0x06");
						// v15 = popcount(v6 & f_packed_1[5])
						asm volatile(".byte  0xD7, 0x47, 0x6F, 0x06");
						// v16 = popcount(v6 & f_packed_1[4])
						asm volatile(".byte  0x57, 0xC8, 0x6E, 0x06");
						// v17 = popcount(v6 & f_packed_1[3])
						asm volatile(".byte  0xD7, 0x48, 0x6E, 0x06");
						// v18 = popcount(v6 & f_packed_1[2])
						asm volatile(".byte  0x57, 0xC9, 0x63, 0x06");
						// v19 = popcount(v6 & f_packed_1[1])
						asm volatile(".byte  0xD7, 0x49, 0x63, 0x06");
						// v20 = popcount(v6 & f_packed_1[0])
						asm volatile(".byte  0x57, 0xCA, 0x62, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v14, v14, 2");
						asm volatile("vsll.vi v15, v15, 2");
						asm volatile("vsll.vi v16, v16, 2");
						asm volatile("vsll.vi v17, v17, 2");
						asm volatile("vsll.vi v18, v18, 2");
						asm volatile("vsll.vi v19, v19, 2");
						asm volatile("vsll.vi v20, v20, 2");
						#endif
							
						asm volatile("vadd.vv v24, v24, v14");
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
					}
					
					if(height < H_in - 3){
						// v15 = popcount(v7 & f_packed_1[6])
						asm volatile(".byte  0xD7, 0xC7, 0x7F, 0x06");
						// v16 = popcount(v7 & f_packed_1[5])
						asm volatile(".byte  0x57, 0x48, 0x7F, 0x06");
						// v17 = popcount(v7 & f_packed_1[4])
						asm volatile(".byte  0xD7, 0xC8, 0x7E, 0x06");
						// v18 = popcount(v7 & f_packed_1[3])
						asm volatile(".byte  0x57, 0x49, 0x7E, 0x06");
						// v19 = popcount(v7 & f_packed_1[2])
						asm volatile(".byte  0xD7, 0xC9, 0x73, 0x06");
						// v20 = popcount(v7 & f_packed_1[1])
						asm volatile(".byte  0x57, 0x4A, 0x73, 0x06");
						// v21 = popcount(v7 & f_packed_1[0])
						asm volatile(".byte  0xD7, 0xCA, 0x72, 0x06");
						
						#ifndef PERF_VHSACC
						asm volatile("vsll.vi v15, v15, 2");
						asm volatile("vsll.vi v16, v16, 2");
						asm volatile("vsll.vi v17, v17, 2");
						asm volatile("vsll.vi v18, v18, 2");
						asm volatile("vsll.vi v19, v19, 2");
						asm volatile("vsll.vi v20, v20, 2");
						asm volatile("vsll.vi v21, v21, 2");
						#endif
						
						asm volatile("vadd.vv v25, v25, v15");
						asm volatile("vadd.vv v26, v26, v16");
						asm volatile("vadd.vv v27, v27, v17");
						asm volatile("vadd.vv v28, v28, v18");
						asm volatile("vadd.vv v29, v29, v19");
						asm volatile("vadd.vv v30, v30, v20");
						asm volatile("vadd.vv v31, v31, v21");
					}
						
						
									
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v1, v1, 1");
							asm volatile("vslidedown.vi v5, v5, 1");
						if(height < H_in - 2)	
							asm volatile("vslidedown.vi v2, v2, 1");
							asm volatile("vslidedown.vi v6, v6, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v3, v3, 1");	
							asm volatile("vslidedown.vi v7, v7, 1");
					}
					
				}	
				
				i__ += ENCOD_SIZE;	
			}
			
			i_ +=  4 * C_in * W_in;
			
			asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlen - F + 1));
				
			asm volatile("vsse32.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			
			if(height < H_in - 1){
				o_ += ldo;	
				asm volatile("vsse32.v v23, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			if(height < H_in - 2){
				o_ += ldo;	
				asm volatile("vsse32.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));	
			}
			
			if(height < H_in - 3){
				o_ += ldo;	
				asm volatile("vsse32.v v25, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			}
			
			o_ += ldo;
			
			}
	}
}



