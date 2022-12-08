/**
k2c_pooling_layers.c
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
 */


#include <math.h>
#include <string.h>
#include "k2c_include.h"
#include "fpooling_tensor32.h"
#include "ipooling_tensor8.h"

#include <stdint.h>
//#include "../../common/printf.h"
#ifndef SPIKE
#include "printf.h"
#endif

/**
 * Global max pooling.
 * works for 1D, 2D, or 3D inputs.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 */
void k2c_global_max_pooling(k2c_tensor* output, const k2c_tensor* input) {
    fmax_pool32(output->array, input->array, input->shape[0], input->shape[1], input->shape[2], 3);
    /* 
    const int8_t in_chan = input->shape[input->ndim-1];
    for (int8_t i=0; i<in_chan; ++i) {
        output->array[i] = input->array[i];
    }

    for (int8_t i=0; i<input->numel; i+=in_chan) {
        for (int8_t j=0; j<in_chan; ++j) {
            if (output->array[j]<input->array[i+j]) {
                output->array[j] = input->array[i+j];
            }
        }
    }
    */
}


/**
 * Global average pooling.
 * works for 1D, 2D, or 3D inputs.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 */
void k2c_global_avg_pooling(k2c_tensor* output, const k2c_tensor* input) {

    const int8_t in_chan = input->shape[input->ndim-1];
    memset(output->array,0,output->numel*sizeof(input->array[0]));
    const int8_t num_inv = 1.0f/(input->numel/in_chan);

    for (int8_t i=0; i<input->numel; i+=in_chan) {
        for (int8_t j=0; j<in_chan; ++j) {
            output->array[j] += input->array[i+j]*num_inv;
        }
    }
}


/**
 * Max pooling for 1D (temporal) data.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param pool_size: size of the max pooling window.
 * :param stride: factor by which to downscale.
 */
void k2c_maxpool1d(k2c_tensor* output, const k2c_tensor* input, const size_t pool_size,
                   const size_t stride) {
    const size_t channels = input->shape[1];

    for(size_t i=0; i<channels; ++i) {
        for (size_t j=0, k=0; j<output->shape[0]*channels; j+=channels, k+=stride*channels) {
            output->array[j+i] = input->array[k+i];
            for (size_t l=0; l<pool_size*channels; l+=channels) {
                if (output->array[j+i] < input->array[k+i+l]) {
                    output->array[j+i] = input->array[k+i+l];
                }
            }
        }
    }
}


/**
 * Max pooling for 2D (spatial) data.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param pool_size: array[2] size of the max pooling window. Order is {pool size dim 1, pool size dim 2}.
 * :param stride: array[2] factor by which to downscale. Order is {stride dim 1, stride dim 2}.
 */
void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                   const size_t * stride) {
    const size_t channels = input->shape[2];

	printf("in include/pooling_layers.c\n");
    // i,j,l output indices
    /// i, k, m input indices
    imax_pool8(output->array, input->array, input->shape[0], input->shape[1], input->shape[2], 3);

    // i,j,l output indices
    /// i, k, m input indices
    /*
    for (size_t i=0; i< channels; ++i) {
        for (size_t j=0, k=0; j<output->shape[1]*channels;
                j+=channels, k+=channels*stride[1]) {
            for (size_t l=0, m=0; l<output->numel; l+=channels*output->shape[1],
                    m+=channels*input->shape[1]*stride[0]) {
                output->array[l+j+i] = input->array[m+k+i];
                for (size_t n=0; n<pool_size[1]*channels; n+=channels) {
                    for (size_t p=0; p<pool_size[0]*channels*input->shape[1];
                            p+=channels*input->shape[1]) {
                        if (output->array[l+j+i] < input->array[m+k+i+n+p]) {
                            output->array[l+j+i] = input->array[m+k+i+n+p];
                        }
                    }
                }
            }
        }
    }
    */
    
}


/**
 * Average pooling for 1D (temporal) data.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param pool_size: size of the average pooling window.
 * :param stride: factor by which to downscale.
 */
void k2c_avgpool1d(k2c_tensor* output, const k2c_tensor* input, const size_t pool_size,
                   const size_t stride) {

    const size_t channels = input->shape[1];
    memset(output->array,0,output->numel*sizeof(output->array[0]));
    for(size_t i=0; i<channels; ++i) {
        for (size_t j=0, k=0; j<output->numel; j+=channels, k+=stride*channels) {
            int count = 0;
            for (size_t l=0; l<pool_size*channels; l+=channels) {
                if (input->array[k+i+l] > -HUGE_VALF) {
                    output->array[j+i] += input->array[k+i+l];
                    ++count;
                }
            }
            output->array[i+j] /= (size_t)count;
        }
    }
}


/**
 * Average pooling for 2D (spatial) data.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param pool_size: array[2] size of the average pooling window. Order is {pool size dim 1, pool size dim 2}.
 * :param stride: array[2] factor by which to downscale. Order is {stride dim 1, stride dim 2}.
 */
void k2c_avgpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                   const size_t * stride) {
    memset(output->array,0,output->numel*sizeof(output->array[0]));
    const int8_t channels = input->shape[2];
    // i,j,l output indices
    /// i, k, m input indices
    for (int8_t i=0; i< channels; ++i) {
        for (int8_t j=0, k=0; j<output->shape[1]*channels;
                j+=channels, k+=channels*stride[1]) {
            for (int8_t l=0, m=0; l<output->numel; l+=channels*output->shape[1],
                    m+=channels*input->shape[1]*stride[0]) {
                int8_t count = 0;
                for (int8_t n=0; n<pool_size[1]*channels; n+=channels) {
                    for (int8_t p=0; p<pool_size[0]*channels*input->shape[1];
                            p+=channels*input->shape[1]) {
                        if (-HUGE_VALF < input->array[m+k+i+n+p]) {
                            output->array[l+j+i] += input->array[m+k+i+n+p];
                            ++count;
                        }
                    }
                }
                output->array[l+j+i] /= (int8_t)count;
            }
        }
    }
}
