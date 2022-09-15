#include "ibsconv2d_tensor64.h"
#include <stdio.h>

#define VEC_SIZE 128

//#define DEBUG

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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


void iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out) {
	
  /*int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c * (H_in - F + 1) * (W_in - F + 1);      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;*/

	  // Iterate over the output rows
	  

		/*switch (F){
			 case 1:
					iconv2d_tensor8_vec_1xC_1x1(o_, i_, f_, H_in, W_in, C_in, F);
				break;

			 case 3:
					conv2d(o, i, f, H_in, W_in, C_in, F);
				break;
				
			 case 5:
					iconv2d_tensor8_vec_6xC_5x5(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
			 case 7:
					iconv2d_tensor8_vec_4xC_7x7(o_, i_, f_, H_in, W_in, C_in, F);
				break;
		}
		 
		 
	}*/
}




//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                1x1 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


void print_tensor_(uint64_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%lX ", tensor[(i+k*num_rows) * num_columns  + j ] >> 63);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}



//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                3x3 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


void conv2d_prec1(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
#endif

int64_t * o_ = o_ptr;
int64_t * i_ = i_ptr;
int64_t f_packed[9];

int64_t ldo = W_in - 2;
int64_t H_per_it = MIN(VEC_SIZE / W_in, H_in); 

bitpack_filter_1(f_ptr, f_packed, F*F);

int64_t nb_out;

for (int height = H_per_it ; height <= H_in ; height += H_per_it){
		

	if(height > H_per_it){
	
		asm volatile("vmv.v.i v0, 0");
		asm volatile("vmv.v.i v2, 0");
		asm volatile("vmv.v.i v4, 0");
		
		asm volatile("vmv.v.i v16, 0");

	
	
		nb_out = H_per_it;
	
		// we re-use the values already loaded 
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));

		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack_1(i_,  H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			sum_count = get_timer();
			cycle_count += sum_count;
		#endif
		
		asm volatile("vmv.v.i v8, 0");
		
		asm volatile("vmv.v.v v4, v0"); // à retirer non ?
		
		asm volatile("vslideup.vx v2, v0, %0" :: "r"(W_in));
		
		asm volatile("vslideup.vx v16, v0, %0" :: "r"(W_in << 1));
		
		
		asm volatile("vor.vv v0, v16, v26");
		
		
		asm volatile("vslidedown.vx v26, v26, %0" :: "r"(W_in));
		
		
		asm volatile("vor.vv v2, v2, v26");

		asm volatile("vslidedown.vx v26, v4, %0" :: "r"((H_per_it - 2) * W_in));

				
		i_ += H_per_it * W_in;

		
	}
	else{
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));
		
		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack_1(i_, H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			cycle_count = get_timer();
		#endif
		
		asm volatile("vmv.v.v v26, v0");
		
		asm volatile("vslidedown.vx v4, v0, %0" :: "r"(W_in << 1));
		
		asm volatile("vslidedown.vx v2, v0, %0" :: "r"(W_in));
		
		asm volatile("vslidedown.vx v26, v26, %0" :: "r"((H_per_it - 2) * W_in));
	
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"((H_per_it - 2) * W_in));
		
		
		i_ += H_per_it * W_in;
		
		nb_out = H_per_it - 2;
	
	}
	
	
	
	
		
	// LSB
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	//Pre-compute
	


			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		
	asm volatile("vadd.vv v30, v12, v14");				
	
	asm volatile("vslidedown.vi v0, v0, 1");				
	asm volatile("vslidedown.vi v2, v2, 1");				
	asm volatile("vslidedown.vi v4, v4, 1");
	
	asm volatile("vadd.vv v30, v30, v16");
	
		
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
	asm volatile("vadd.vv v12, v12, v14");
	asm volatile("vadd.vv v30, v30, v16");	
		
	asm volatile("vslidedown.vi v0, v0, 1");				
	asm volatile("vslidedown.vi v2, v2, 1");				
	asm volatile("vslidedown.vi v4, v4, 1");
							
	asm volatile("vadd.vv v30, v30, v12");
		
		
		
		
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		
	asm volatile("vadd.vv v30, v30, v16");

	asm volatile("vadd.vv v12, v12, v14");
	
	asm volatile("vadd.vv v30, v30, v12");
		

	for(int64_t h_i = 0 ; h_i < nb_out ; h_i ++){
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(ldo));
		
		asm volatile("vse64.v v30, (%0)" : "+&r"(o_));

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(nb_out * W_in));

		asm volatile("vslidedown.vx v30, v30, %0" :: "r"(W_in));			
		
		o_ += ldo;
		
		}
		
	}
	
	if(H_in % H_per_it > 0){
		
			nb_out = H_per_it - 2 ;
		
			// we re-use the values already loaded 

			asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));
			//start_timer();
			vbitpack_1(i_,  H_in * W_in); 
			//stop_timer();
			
			asm volatile("vmv.v.v v4, v0");
			
			asm volatile("vslideup.vx v2, v0, %0" :: "r"(W_in));
			
			asm volatile("vslideup.vx v16, v0, %0" :: "r"(W_in << 1));
			
			asm volatile("vor.vv v0, v16, v26");
			
			asm volatile("vslidedown.vx v26, v26, %0" :: "r"(W_in));
			
			asm volatile("vor.vv v2, v2, v26");

			asm volatile("vslidedown.vx v26, v0, %0" :: "r"((H_per_it - 2) * W_in));	
			
			asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));	
		
			
		// LSB
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				

				
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
			
		asm volatile("vadd.vv v30, v12, v14");				
		
		asm volatile("vslidedown.vi v0, v0, 1");				
		asm volatile("vslidedown.vi v2, v2, 1");				
		asm volatile("vslidedown.vi v4, v4, 1");
		
		asm volatile("vadd.vv v30, v30, v16");
		
			
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				

				
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
		asm volatile("vadd.vv v12, v12, v14");
		asm volatile("vadd.vv v30, v30, v16");	
			
		asm volatile("vslidedown.vi v0, v0, 1");				
		asm volatile("vslidedown.vi v2, v2, 1");				
		asm volatile("vslidedown.vi v4, v4, 1");
								
		asm volatile("vadd.vv v30, v30, v12");
			
			
			
			
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				

				
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
		asm volatile("vadd.vv v12, v12, v14");
		asm volatile("vadd.vv v30, v30, v16");	
		asm volatile("vadd.vv v30, v30, v12");
			

		for(int64_t h_i = 0 ; h_i < nb_out ; h_i ++){
			
			asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(ldo));
			
			asm volatile("vse64.v v30, (%0)" : "+&r"(o_));

			asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));

			asm volatile("vslidedown.vx v30, v30, %0" :: "r"(W_in));			
			
			o_ += ldo;
			
		}
	
	
	}
	
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
	#endif


}


// TO BE RE-DONE TO FIT THE 1b PREC (rn isn't working properly)


void conv2d(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
#endif

int64_t * o_ = o_ptr;
int64_t * i_ = i_ptr;
int64_t f_packed[18];

int64_t ldo = W_in - 2;
int64_t H_per_it = MIN(VEC_SIZE / W_in, H_in); 

bitpack_filter(f_ptr, f_packed, F*F);

int64_t nb_out;

for (int height = H_per_it ; height <= H_in ; height += H_per_it){

	if(height > H_per_it){
	
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));
	
		asm volatile("vmv.v.i v0, 0");
		asm volatile("vmv.v.i v2, 0");
		asm volatile("vmv.v.i v4, 0");
		asm volatile("vmv.v.i v6, 0");
		asm volatile("vmv.v.i v8, 0");
		asm volatile("vmv.v.i v10, 0");
		
		asm volatile("vmv.v.i v16, 0");
		asm volatile("vmv.v.i v18, 0");
		
	
	
		nb_out = H_per_it;
	
		// we re-use the values already loaded 
		
		printf("BITPACK...\n");

		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack(i_,  H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			sum_count = get_timer();
			cycle_count += sum_count;
		#endif
		
		printf("BITPACK DONE\n");
		
		asm volatile("vmv.v.i v8, 0");
		asm volatile("vmv.v.i v16, 0");
		
		
		asm volatile("vmv.v.v v4, v0");
		asm volatile("vmv.v.v v10, v6");
		
		asm volatile("vslideup.vx v2, v0, %0" :: "r"(W_in));
		asm volatile("vslideup.vx v8, v6, %0" :: "r"(W_in));
		
		asm volatile("vslideup.vx v16, v0, %0" :: "r"(W_in << 1));
		asm volatile("vslideup.vx v18, v6, %0" :: "r"(W_in << 1));
		
		
		asm volatile("vor.vv v0, v16, v26");
		asm volatile("vor.vv v6, v18, v28");
		
		
		
		
		printf("RE-USE ...\n");
		
		
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(W_in << 1)); //pretty bad solution, should be deleted later, but there might be a bug within slidedown instruction
		
		asm volatile("vslidedown.vx v26, v26, %0" :: "r"(W_in));
		asm volatile("vslidedown.vx v28, v28, %0" :: "r"(W_in));
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));
		
		printf("CA ME CASSE LES BONBONS\n");
		
		asm volatile("vor.vv v2, v2, v26");
		
		printf("QUOI1\n");
		asm volatile("vor.vv v8, v8, v28");
		printf("QUOI2\n");
		asm volatile("vmv.v.v v26, v4");
		printf("QUOI3\n");
		asm volatile("vmv.v.v v28, v10");
		
		printf("CA ME CASSE LES BONBONS LE RETOUR\n");

				
		i_ += H_per_it * W_in;
		
		asm volatile("vslidedown.vx v26, v26, %0" :: "r"((H_per_it - 2) * W_in));
		asm volatile("vslidedown.vx v28, v28, %0" :: "r"((H_per_it - 2) * W_in));
		
		
		printf("RE-USE DONE\n");
		

			
	}
	else{
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in));
		
		printf("BITPACK...\n");
		
		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack(i_, H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			cycle_count = get_timer();
		#endif
		
		printf("BITPACK DONE\n");
		
		asm volatile("vslidedown.vx v26, v0, %0" :: "r"((H_per_it - 2) * W_in));
		asm volatile("vslidedown.vx v28, v6, %0" :: "r"((H_per_it - 2) * W_in));
		
		asm volatile("vslidedown.vx v4, v0, %0" :: "r"(W_in << 1));
		asm volatile("vslidedown.vx v10, v6, %0" :: "r"(W_in << 1));
		
		asm volatile("vslidedown.vx v2, v0, %0" :: "r"(W_in));
		asm volatile("vslidedown.vx v8, v6, %0" :: "r"(W_in));


		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"((H_per_it - 2) * W_in));
		
		
		i_ += H_per_it * W_in;
		
		if(H_per_it - 2 >= 0)
			nb_out = H_per_it - 2;
		else
			nb_out = 0;
		
		printf("FIRST LOAD DONE \n");
	
	}
	
	/*printf("V0\n");
	asm volatile("vse64.v v0, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);

	printf("V2\n");
	asm volatile("vse64.v v2, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);*/
	
	
	//v14, v18, v20


		 
	for (int f_w = 0 ; f_w < F ; f_w ++) {
		
			// LSB
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// MSB
			
			asm volatile("vand.vx v18, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v20, v2, %0" :: "r"(f_packed[f_w + 12])); // This one is the problem
			
			asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel

			
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

			// v16 = popcount(v16)
			asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
			// v18 = popcount(v18)
			asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
			
			// v20 = popcount(v20)
			asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
			
			// v22 = popcount(v22)
			asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
			
			
			asm volatile("vsll.vi v18, v18, 1"); 
			asm volatile("vsll.vi v20, v20, 1"); 
			asm volatile("vsll.vi v22, v22, 1"); 
			
				
			if(f_w > 0){
				
				asm volatile("vadd.vv v12, v12, v14");
				asm volatile("vadd.vv v30, v30, v16");
				
				asm volatile("vslidedown.vi v0, v0, 1");
				
				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v18, v18, v20");
				
				asm volatile("vslidedown.vi v2, v2, 1");				

				asm volatile("vadd.vv v30, v30, v18");
				
				asm volatile("vslidedown.vi v4, v4, 1");
				
				asm volatile("vadd.vv v30, v30, v22");
				
			}
			
			else {
				asm volatile("vadd.vv v30, v12, v14");
				
				asm volatile("vadd.vv v18, v18, v20");
				
				asm volatile("vslidedown.vi v0, v0, 1");
				
				asm volatile("vadd.vv v30, v30, v16");
				
				asm volatile("vadd.vv v18, v18, v22");
				
				asm volatile("vslidedown.vi v2, v2, 1");
				asm volatile("vslidedown.vi v4, v4, 1");
				
				asm volatile("vadd.vv v30, v30, v18");
				
				
			}
			
			
			
			// LSB
			asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// MSB
			
			asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v20, v8, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel

			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

			// v16 = popcount(v16)
			asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
			// v18 = popcount(v18)
			asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
			
			// v20 = popcount(v20)
			asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
			
			// v22 = popcount(v22)
			asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
			
			
			
			asm volatile("vsll.vi v12, v12, 1"); 
			asm volatile("vsll.vi v14, v14, 1"); 
			asm volatile("vsll.vi v16, v16, 1"); 
			asm volatile("vsll.vi v18, v18, 2"); 
			asm volatile("vsll.vi v20, v20, 2"); 
			asm volatile("vsll.vi v22, v22, 2"); 



			asm volatile("vadd.vv v12, v12, v14");
			asm volatile("vadd.vv v30, v30, v16");
			
			asm volatile("vslidedown.vi v6, v6, 1");
			
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v18, v18, v20");
			
			asm volatile("vslidedown.vi v8, v8, 1");
			
			asm volatile("vadd.vv v30, v30, v18");
			
			asm volatile("vslidedown.vi v10, v10, 1");
			
			asm volatile("vadd.vv v30, v30, v22");	
			
		
	}
	
	/*printf("V12\n");
	asm volatile("vse64.v v12, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);

	printf("V14\n");
	asm volatile("vse64.v v14, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);

	printf("V16\n");
	asm volatile("vse64.v v16, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);
	
	printf("V18\n");
	asm volatile("vse64.v v18, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);

	printf("V20\n");
	asm volatile("vse64.v v20, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);

	printf("V22\n");
	asm volatile("vse64.v v22, (%0)" : "+&r"(o_));
	print_tensor_(o_, H_per_it, W_in, 1);*/


	printf("%i\n", nb_out);
	for(int64_t h_i = 0 ; h_i < nb_out ; h_i ++){
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(ldo));
		
		asm volatile("vse64.v v30, (%0)" : "+&r"(o_));

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(nb_out * W_in));

		asm volatile("vslidedown.vx v30, v30, %0" :: "r"(W_in));
		
		printf("STORE DONE \n");			
		
		o_ += ldo;
		
		}
	printf("ITERATION_DONE \n");
	}
	
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
	#endif

}





// TEST 

void conv2d_prec1_test(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
#endif

int64_t H_per_it;
int64_t W_in_tiled;

int64_t * o_ = o_ptr;
int64_t * i_ = i_ptr;
int64_t f_packed[9];

int64_t ldo = W_in - 2;

if(W_in >= VEC_SIZE){
	H_per_it = 2; 
	W_in_tiled = VEC_SIZE << 1;
	}
else
	{
	H_per_it = MIN(VEC_SIZE / W_in, H_in); 
	W_in_tiled = W_in;
	}

bitpack_filter_1(f_ptr, f_packed, F*F);

int64_t nb_out;

for (int height = H_per_it ; height <= H_in ; height += H_per_it){
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in_tiled));
	
	
	if (height == H_per_it){
		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack_1(i_, H_in * W_in_tiled); 
		#ifdef DEBUG
			stop_timer();
			cycle_count = get_timer();
		#endif
		
		
		asm volatile("vslidedown.vx v2, v0, %0" :: "r"(W_in_tiled));
		
		asm volatile("vslidedown.vx v4, v0, %0" :: "r"(W_in_tiled << 1));
		
		}
		
		asm volatile("vmv.v.v v6, v0");
		
		asm volatile("vslidedown.vx v6, v6, %0" :: "r"((H_per_it - 2) * W_in_tiled));
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(W_in_tiled << 1));
		
		asm volatile("vslidedown.vx v8, v6, %0" :: "r"(W_in_tiled));
		
	
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"((H_per_it - 2) * W_in_tiled));
		
		
		i_ += H_per_it * W_in_tiled;
		
		nb_out = H_per_it - 2;
	
	
		
	// LSB
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	//Pre-compute
	


			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		
	asm volatile("vadd.vv v30, v12, v14");				
	
	asm volatile("vslidedown.vi v0, v0, 1");				
	asm volatile("vslidedown.vi v2, v2, 1");				
	asm volatile("vslidedown.vi v4, v4, 1");
	
	asm volatile("vadd.vv v30, v30, v16");
	
		
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
	asm volatile("vadd.vv v12, v12, v14");
	asm volatile("vadd.vv v30, v30, v16");	
		
	asm volatile("vslidedown.vi v0, v0, 1");				
	asm volatile("vslidedown.vi v2, v2, 1");				
	asm volatile("vslidedown.vi v4, v4, 1");
							
	asm volatile("vadd.vv v30, v30, v12");
		
		
		
		
	asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
	// v16 = popcount(v16)
	asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		
	asm volatile("vadd.vv v30, v30, v16");

	asm volatile("vadd.vv v12, v12, v14");
	
	asm volatile("vadd.vv v30, v30, v12");
	
	
	
	for(int64_t h_i = 0 ; h_i < nb_out ; h_i ++){
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(ldo));
		
		asm volatile("vse64.v v30, (%0)" : "+&r"(o_));

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(nb_out * W_in_tiled));

		asm volatile("vslidedown.vx v30, v30, %0" :: "r"(W_in_tiled));			
		
		o_ += ldo;
		
		}
	
	
	/*printf("V6_1\n");
	asm volatile("vse64.v v6, (%0)" : "+&r"(o_));
	print_tensor_(o_, 2, W_in, 1);

	printf("V8_1\n");
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));
	print_tensor_(o_, 2, W_in, 1);*/
	
	
	
	
	if(height < H_in){
	///////////////////////
	// for the tail
	///////////////////////
	
	
	asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(W_in_tiled << 1));
	
	// LSB
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
		
	asm volatile("vadd.vv v10, v12, v14");				
	
	asm volatile("vslidedown.vi v6, v6, 1");				
	asm volatile("vslidedown.vi v8, v8, 1");				
	
		
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
	asm volatile("vadd.vv v12, v12, v14");
		
	asm volatile("vslidedown.vi v6, v6, 1");				
	asm volatile("vslidedown.vi v8, v8, 1");				
							
	asm volatile("vadd.vv v10, v10, v12");
		
		
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			

			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

	asm volatile("vadd.vv v12, v12, v14");
	
	asm volatile("vadd.vv v10, v10, v12");
			
	//	complete the joint between the tail and the overhead
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(H_per_it * W_in_tiled));
		
		#ifdef DEBUG
			start_timer();
		#endif
		vbitpack_1(i_, H_in * W_in_tiled); 
		#ifdef DEBUG
			stop_timer();
			cycle_count = get_timer();
		#endif
		
		asm volatile("vmv.v.i v6, 0");
		
		asm volatile("vmv.v.i v8, 0");
		
		
		asm volatile("vslidedown.vx v2, v0, %0" :: "r"(W_in_tiled));
		
		asm volatile("vslidedown.vx v4, v0, %0" :: "r"(W_in_tiled << 1));
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(W_in_tiled << 1));
		
		asm volatile("vslideup.vx v6, v0, %0" :: "r"(W_in_tiled));
		
		asm volatile("vmv.v.v v8, v0");
		
		
		//	complete the joint between the tail and the overhead
		
		nb_out = 2;
			
		
		
		
	asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(W_in_tiled << 1));
	
	// LSB
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
		
	asm volatile("vadd.vv v12, v12, v14");				
	
	asm volatile("vslidedown.vi v6, v6, 1");				
	asm volatile("vslidedown.vi v8, v8, 1");	
	
	asm volatile("vadd.vv v10, v10, v12");			
	
		
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
	asm volatile("vadd.vv v12, v12, v14");
		
	asm volatile("vslidedown.vi v6, v6, 1");				
	asm volatile("vslidedown.vi v8, v8, 1");				
							
	asm volatile("vadd.vv v10, v10, v12");
		
		
	asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
	
	asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			
	// v12 = popcount(v12)
	asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
	// v14 = popcount(v14)
	asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

	asm volatile("vadd.vv v12, v12, v14");
	
	asm volatile("vadd.vv v10, v10, v12");
	
	asm volatile("vmv.v.i v6, 0");
		
	asm volatile("vmv.v.i v8, 0");
	
	
	for(int64_t h_i = 0 ; h_i < nb_out ; h_i ++){
		
		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(ldo));
		
		asm volatile("vse64.v v10, (%0)" : "+&r"(o_));

		asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(nb_out * W_in_tiled));

		asm volatile("vslidedown.vx v10, v10, %0" :: "r"(W_in_tiled));			
		
		o_ += ldo;
		
		}
	
	}
	
	
	/*printf("V6_2\n");
	asm volatile("vse64.v v6, (%0)" : "+&r"(o_));
	print_tensor_(o_, 2, W_in, 1);


	printf("V8_2\n");
	asm volatile("vse64.v v8, (%0)" : "+&r"(o_));
	print_tensor_(o_, 2, W_in, 1);*/

		
	}
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
	#endif


}



