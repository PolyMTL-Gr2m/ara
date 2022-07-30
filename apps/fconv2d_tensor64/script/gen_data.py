#!/usr/bin/env python3

import numpy as np
import scipy.signal
import sys


def rand_tensor(O, L, N, M, seed):
	return np.arange(seed, seed+O*N*M*L, dtype=np.float64).reshape(O, L, N, M) * 3.141

def emit(name, array, alignment='3'):
	print(".global %s" % name)
	print(".align " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	while ((len(bs) % 4) != 0):
		bs += bytes([0]) 
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n] 
		print("    .word 0x%s" % s)
		

num_filter = 1

# Input image
M_pad = 32
N_pad = 32
L = 3


M = M_pad
N = N_pad
L_pad = L

# Generate a random float64 input padded image
tensor = rand_tensor(1, L_pad, M_pad, N_pad, 1).astype(np.float64) / 2**16
np.random.shuffle(tensor.flat)

# Generate a random float64 filter
gen_filter = rand_tensor(num_filter, L, 7, 7, -128).astype(np.float64) / 2**16
np.random.shuffle(gen_filter.flat)

filter_1 = gen_filter[:,:,:1,:1]
filter_3 = gen_filter[:,:,:3,:3]
filter_5 = gen_filter[:,:,:5,:5]
filter_7 = gen_filter

#transform to 32b for convolution (otherwise, it will lock at 8b and cause overflow)
tensor_conv = tensor.astype(np.float64)

filter_1_conv = filter_1.astype(np.float64)
filter_3_conv = filter_3.astype(np.float64)
filter_5_conv = filter_5.astype(np.float64)
filter_7_conv = filter_7.astype(np.float64)

# Create the empty o matrix
empty_o = np.zeros((num_filter, M_pad, N_pad)).astype(np.float64)

result_1 = np.zeros((num_filter, M, N)).astype(np.float64) #Num_filter x M x N
result_3 = np.zeros((num_filter, M-2, N-2)).astype(np.float64) #Num_filter x M x N
result_5 = np.zeros((num_filter, M-4, N-4)).astype(np.float64) #Num_filter x M x N
result_7 = np.zeros((num_filter, M-6, N-6)).astype(np.float64) #Num_filter x M x N

for num in range(num_filter):
	for plane in range(L):
		result_1[num,:,:] += scipy.signal.convolve2d(np.flip(filter_1_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid').astype(np.float64)
		result_3[num,:,:] += scipy.signal.convolve2d(np.flip(filter_3_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid').astype(np.float64)
		result_5[num,:,:] += scipy.signal.convolve2d(np.flip(filter_5_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid').astype(np.float64)
		result_7[num,:,:] += scipy.signal.convolve2d(np.flip(filter_7_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid').astype(np.float64)
	

# Calculate a checksum
checksum = np.sum(result_1, dtype=np.float64)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display
#vhex = np.vectorize(hex)

tensor = np.float64(tensor)
gen_filter = np.float64(gen_filter)

print("\n----------------------")
print("NCHW memory layout")
print("----------------------\n")

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M_pad, dtype=np.uint64))
emit("N", np.array(N_pad, dtype=np.uint64))
emit("L", np.array(L, dtype=np.uint64))
#emit("F", np.array(filter_size, dtype=np.uint64))
emit("K", np.array(num_filter, dtype=np.uint64))
emit("i", tensor, 'NR_LANES*4')
emit("f", gen_filter, 'NR_LANES*4')

emit("golden_o_1", result_1, 'NR_LANES*4')
emit("golden_o_3", result_3, 'NR_LANES*4')
emit("golden_o_5", result_5, 'NR_LANES*4')
emit("golden_o_7", result_7, 'NR_LANES*4')

emit("o", empty_o, 'NR_LANES*4')

emit("o_checksum", checksum)


