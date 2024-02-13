// Author: Elisabeth Humblet
// Based on the work by: Theo Dupuis
// GR2M - 2024
// Polytechnique Montréal

#include <stdint.h> 
#include <stdio.h>
#include <string.h> 

#include "ibsconv2d_tensor32.h"
#include "runtime.h"
#include "cache_metrics.h"

#include "util.h"

// =============================
// ====== UART FUNCTIONS =======
// =============================

#define UART_BASE 0xFFF0C2C000

#define UART_INTERRUPT_ENABLE UART_BASE + 1
#define UART_LINE_CONTROL UART_BASE + 3
#define UART_MODEM_CONTROL UART_BASE + 4
#define UART_LINE_STATUS UART_BASE + 5
#define UART_MODEM_STATUS UART_BASE + 6
#define UART_DLAB_LSB UART_BASE + 0
#define UART_DLAB_MSB UART_BASE + 1

void write_reg_u8(uintptr_t addr, uint8_t value)
{
    volatile uint8_t *loc_addr = (volatile uint8_t *)addr;
    *loc_addr = value;
}

void init_uart(uint32_t freq, uint32_t baud)
{
    uint32_t divisor = freq / (baud << 4);

    write_reg_u8(UART_INTERRUPT_ENABLE, 0x00); // Disable all interrupts
    write_reg_u8(UART_LINE_CONTROL, 0x80);     // Enable DLAB (set baud rate divisor)
    write_reg_u8(UART_DLAB_LSB, divisor);         // divisor (lo byte)
    write_reg_u8(UART_DLAB_MSB, (divisor >> 8) & 0xFF);  // divisor (hi byte)
    write_reg_u8(UART_LINE_CONTROL, 0x03);     // 8 bits, no parity, one stop bit
    write_reg_u8(UART_MODEM_CONTROL, 0x20);    // Autoflow mode
}

// =============================
// ==== MACROS DEFINITIONS =====
// =============================

// ---- Architecture ----
#define NR_LANES 4

// ---- Debug ----
//#define VERIF

// ---- Precisions ----
#define PRECA_MAX 1
#define PRECW_MAX 1

// ---- Tensors ----
#define F_MAX     7             // Max size of the kernel
#define C_IN      16            // Number of input channels
#define C_OUT     1             // Number of output channels
#define I_MAX     16            // Max H_in x W_in input size
#define I_START   16            // Start input size

int8_t i     [I_MAX * I_MAX * C_IN];
int8_t f     [F_MAX * F_MAX * C_IN * C_OUT];
int8_t f_nhwc[F_MAX * F_MAX * C_IN * C_OUT];
int16_t o    [(I_MAX - F_MAX + 1)*(I_MAX - F_MAX + 1) * C_OUT];

// =============================
// === MULTICORE DEFINITIONS ===
// =============================

volatile static uint32_t init_done = 0;
volatile static uint32_t conv_done = 0;

//#define INPUT_MULTICORE
#define OUTPUT_MULTICORE

// =============================
// ==== UTILITIES FUNCTIONS ====
// =============================

void iconv2d_tensor_naive(int16_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {

    //treat pointers as 3D arrays
    int8_t (*i_)[R+F-1][C+F-1] = (int8_t (*)[R+F-1][C+F-1])i;
    int8_t (*f_)[W][F][F] = (int8_t (*)[W][F][F])f;
    int16_t (*o_)[R][C] = (int16_t (*)[R][C])o;
    
    for(int k = 0 ; k < K ; k++) 
    	for(int ch = 0 ; ch < W ; ch++)
    		for(int r = 0 ; r < R ; r++)
    			for(int c = 0 ; c < C ; c++)
    				for(int fh = 0 ; fh < F ; fh++)
    					for(int fw = 0 ; fw < F ; fw++) {
    						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
    						if(get_hartid()==0){
    						printf("k  %d, ch %d, r %d, c %d, fh %d, fw %d\r\n", k, ch, r, c, fh, fw);
    						}
    					}
}

void NCHW_to_NHWC_8b(int8_t * NCHW_format, int8_t * NHWC_format, int64_t N, int64_t C, int64_t H, int64_t W){
	for(int k = 0; k < N ; k++)
		for(int z = 0; z < H ; z++)
			for(int y = 0 ; y < W ; y++)
				for(int x = 0 ; x < C ; x++)
					{
					NHWC_format[x + C * (y + H * (z + k * W))] = NCHW_format[y + H * (z + W * (x + k * C))];
					}
				printf("\n");
}

void NCHW_to_NHWC_16b(int16_t * NCHW_format, int16_t * NHWC_format, int64_t N, int64_t C, int64_t H, int64_t W){
	for(int k = 0; k < N ; k++)
		for(int z = 0; z < H ; z++)
			for(int y = 0 ; y < W ; y++)
				for(int x = 0 ; x < C ; x++)
					{
					NHWC_format[x + C * (y + H * (z + k * W))] = NCHW_format[y + H * (z + W * (x + k * C))];
					}
				
}

void init_tensor(int8_t *tensor, int64_t R, int64_t C, int64_t D, int precision) {
   
    unsigned int seed = 72986;
    unsigned int limit = (1 << precision) - 1;

    for (int d = 0; d < D; ++d)   //depth
        for (int r = 0; r < R; ++r)  //rows
            for (int c = 0; c < C; ++c)//column
            {
                seed = (seed * 1103515245 + 12345) % 2147483648;
                if(r < R) // test purposes
                    tensor[c + C * (r + d * R)] = seed % (limit + 1);
                else
                    tensor[c + C * (r + d * R)] = 0;
            }
}

// =============================
// === VERIF/DEBUG FUNCTIONS ===
// =============================

int verify_tensor(int16_t *tensor1, int16_t *tensor2, int64_t height, int64_t width, int64_t channels) {
  for (int h = 0; h < height; ++h)   //depth
  	for (int w = 0; w < width; ++w)  //rows
  	  for (int c = 0; c < channels; ++c)//column			
  	    if (tensor1[w + width * (h + c * height)] != tensor2[w + width * (h + c * height)]){
  	      printf("\nError: o[%d][%d][%d] = %d, instead of %d\n", channels, height, width,
  	             tensor1[w + width * (h + c * height)], tensor2[w + width * (h + c * height)]);
  	      return 1;
  	   }
return 0;
  		
}

void print_tensor(uint8_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
    printf("0x%8X\n", (uint64_t)tensor);
    for (uint64_t k = 0; k < num_depth; ++k) {
  	    for (uint64_t i = 0; i < num_rows; ++i) {
        	for (uint64_t j = 0; j < num_columns; ++j) {
                printf("%10u ", tensor[(i+k*num_rows) * num_columns  + j ]);
        	}
        	printf("\r\n");
  	    }
  	printf("\r\n");
    }
}

void print_tensor_16_(uint16_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
    printf("0x%8X\n", (uint64_t)tensor);
    for (uint64_t k = 0; k < num_depth; ++k) {
  	    for (uint64_t i = 0; i < num_rows; ++i) {
        	for (uint64_t j = 0; j < num_columns; ++j) {
                printf("%10u ", tensor[(i+k*num_rows) * num_columns  + j ]);
        	}
        	printf("\r\n");
  	    }
  	    printf("\r\n");
	}
}

void initialization(int64_t precA, int64_t precW, int64_t F, int64_t input_channels, int64_t output_channels, int8_t *filter, int size, int64_t width, int64_t height, int8_t *input, int8_t *i_nhwc, int16_t *output, int16_t *golden_output, int16_t *golden_o){
    printf("===============\r\n");
    printf("= ULPCONV2D16 =\r\n");
    printf("===============\r\n");

    printf("\r\n");
    printf("************\r\n");
    printf("*** A%dW%d ***\r\n", precA, precW);
    printf("************\r\n");


    // ==== Init tensors ====
    printf("\r\n");
    printf("Filling the input and filter tensors... \r\n");

    init_tensor(i, I_MAX, I_MAX, C_IN, precA);
    init_tensor(f, F_MAX, F_MAX, C_IN * C_OUT, precW);

    printf("                                              done\r\n");

    //for(int k = 0; k < output_channels ; k++)
	//	for(int z = 0; z < input_channels ; z++)
	//		for(int y = 0 ; y < F ; y++)
	//			for(int x = 0 ; x < F ; x++)
	//	  			filter[x + F * (y + F * (z + k * input_channels))] = f[x + F_MAX * (y + F_MAX * (z + k * input_channels))];

    #ifdef VERIF
        // ==== Expected output ====
        printf("Computing the expected  output for this kernel size... \r\n");
        for(int z = 0; z < output_channels ; z++)
			for(int y = 0 ; y < (I_MAX - F + 1) ; y++)
				for(int x = 0 ; x < (I_MAX - F + 1) ; x++)
					golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))] = 0;
        
        iconv2d_tensor_naive(golden_o, i, f, (I_MAX - F + 1), (I_MAX - F + 1), input_channels, F, output_channels);
        printf("                                              done\r\n");        
    #endif

    // ==== Transpose filter ====
    NCHW_to_NHWC_8b(f, f_nhwc, output_channels, input_channels, F, F);

    // ==== Information ====
    printf("\r\n");
    printf("----------------------------------------------------------------\r\n");
	printf("Calculating convolution between \r\n");
	printf("Input of [1 x %d x %d x %d] and Filters of [%d x %d x %d x %d]  \r\n", C_IN, size,  size, C_OUT, C_IN, F, F);
	printf("Activation precision of %d and Weights precision of %d  \r\n", precA, precW);
	printf("Result (16b) is an output of [1 x %d x %d x %d] \r\n", C_OUT, size - F + 1, size - F + 1);
	printf("----------------------------------------------------------------\r\n");
	printf("\r\n");

    #ifdef VERIF
    printf("Formatting data and expected outputs...\r\n");
    #else
    printf("Formatting data...\r\n");
    #endif

    // ==== Tensors slicing ====
    //for(int z = 0; z < input_channels; z++)
    //    for(int y = 0; y < height; y++)
    //        for(int x = 0; x < width; x++)
    //            input[x + width * (y + z * height)] = i[x + I_MAX * (y + z * I_MAX)];

    for(int z = 0; z < output_channels ; z++)
		for(int y = 0 ; y < (height - F + 1) ; y++)
			for(int x = 0 ; x < (width - F + 1) ; x++)
			{
				o[x + (width - F + 1) * (y + z * (height - F + 1))] = 0;
    //			#ifdef VERIF
    //			golden_output[x + (width - F + 1) * (y + z * (height - F + 1))] = golden_o[x + (I_MAX - F + 1) * (y + z * (I_MAX - F + 1))];
    //			#endif
			}

    NCHW_to_NHWC_8b(i, i_nhwc, 1, input_channels, height, width);
    printf("                                              done\r\n");

    // ==== INITIALIZATION DONE ====
    ATOMIC_OP(init_done, 1, add, w);
}

void ulppack_conv2d_msparq(int16_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW){
    int8_t *i_;
    int16_t *o_;
    int8_t *f_;

    for(int64_t c = 0; c < C_out; c++){
        #ifdef OUTPUT_MULTICORE
        if(get_hartid() == c){
        #endif
            o_ = o + c * (H_in - F + 1) *(W_in - F + 1);
            i_ = i;
            f_ = f + c * F * F * C_in;
            #ifdef VMACSR
                if (F == 7){
                    if((precA <= 2 && precW < 2) || (precA < 2 && precW <= 2)){
                        #ifdef INPUT_MULTICORE
                            ulppack_conv2d_vec8_7x7_tiling(o_, i_, f_, H_in, W_in, C_in, F, C_out);
                        #else 
                            ulppack_conv2d_vec8_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
                        #endif
                    } else {
                        ulppack_conv2d_vec16_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
                    }
                }
            #else
                if(F == 3){
			    	ulppack_conv2d_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			    }else 
			    if (F == 7){
			    	if (precA <= 1 && precW <= 1){
			    		ulppack_conv2d_vec_7x7_A1W1(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			    	}else if (precA <= 2 && precW <= 2){
			     		ulppack_conv2d_vec_7x7_A2W2(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			    	}else if (precA <= 3 && precW <= 3){
			     		ulppack_conv2d_vec_7x7_A3W3(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			    	}
			    }
            #endif
        #ifdef OUTPUT_MULTICORE
        }
        #endif
    }
    ATOMIC_OP(conv_done, 1, add, w);
}

int main(int argc, char** argv){
    init_uart(50000000, 115200);

    // ===== INIT DEFINES =====
    int64_t precA = PRECA_MAX;
    int64_t precW = PRECW_MAX;

    int64_t F = F_MAX;
    int64_t input_channels = C_IN;
    int64_t output_channels = C_OUT;
    int8_t filter[output_channels * F * F * input_channels];

    int size = I_START;
    int64_t width = size;
    int64_t height = size;

    int8_t input          [width * height * input_channels];
    int8_t i_nhwc         [width * height * input_channels];
    int16_t output        [width * height * output_channels];
    int16_t golden_output [(width - F + 1) * (width - F + 1) * output_channels];
    int16_t golden_o[(I_MAX - F + 1) * (I_MAX - F + 1) * C_OUT];


    // ===== INITIALIZATION =====

    if (argv[0][0] == 0){
        initialization(precA, precW, F, input_channels, output_channels, filter, size, width, height, input, i_nhwc, output, golden_output, golden_o);
    }
    while(init_done == 0);

    printf("Computing results...\r\n");

    // ===== COMPUTING RESULTS =====

    // --- Reset L2 metrics ---
    reset_L2_metrics(argv[0][0]);
    init_L2_metrics(argv[0][0]);

    // --- Timer and convolution ---
    start_timer();
    ulppack_conv2d_msparq(o, i, f, height, width, input_channels, F, output_channels, precA, precW);
    stop_timer();

    // --- Wait for end of convolution ---
    while(conv_done == 0);

    // --- Stop L2 metrics
    stop_L2_metrics(argv[0][0]);
    printf("                                              done\r\n");

    if(argv[0][0] == 0){
        // ===== VERIFICATION =====
        int16_t golden_output_nhwc[(I_MAX - F + 1) * (I_MAX - F + 1) * C_OUT];
        NCHW_to_NHWC_16b(golden_o, golden_output_nhwc, 1, C_OUT, I_MAX - F + 1, I_MAX - F + 1);

        #ifdef VERIF
            printf("Verifying results...\r\n");
            int error = verify_tensor(o, golden_o, (height - F + 1), (width - F + 1), output_channels);
            if (error == 0)
                printf("                                              done\r\n");
            else 
                printf("    ERROR\r\n");
        #else
            int error = 0;
        #endif

        // ===== METRICS =====
        int64_t runtime = get_timer();

        if(error != 0){
            printf("Fail.\r\n");
            printf("Output NHWC\r\n");
            print_tensor_16_(o, (height - F + 1), (width - F + 1), output_channels);
            printf("=========================================\r\n");
            printf("Expected output\r\n");
            print_tensor_16_(golden_o, (height - F + 1), (width - F + 1), output_channels);
        } else {
            printf("Passed.\r\n");
            printf("The execution took %d cycles.\r\n", runtime);
            print_L2_metrics(argv[0][0]);
        }
    }
}