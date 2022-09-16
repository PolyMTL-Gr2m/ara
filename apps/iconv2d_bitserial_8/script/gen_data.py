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
M_pad = 512  	#rows
N_pad = 512		#column
L = 8				#channels

prec = 3


M = M_pad
N = N_pad
L_pad = L

# Generate a random int64 input padded image
tensor = np.around(rand_tensor(1, L_pad, M_pad, N_pad, 0)).astype(np.uint8) & prec
np.random.shuffle(tensor.flat)

tensor[:,1:,:,:] = 0;

# Generate a random int64 filter
gen_filter = np.around(rand_tensor(num_filter, L, 7, 7, 0)).astype(np.uint8) & prec
np.random.shuffle(gen_filter.flat)


filter_1 = gen_filter[:,:,:1,:1]
filter_3 = gen_filter[:,:,:3,:3]
filter_5 = gen_filter[:,:,:5,:5]
filter_7 = gen_filter

# if not wide input (8b) -> output (8b)

#transform to 16b for convolution (otherwise, it will lock at 8b and cause overflow)
tensor_conv = tensor.astype(np.int8)

filter_1_conv = filter_1.astype(np.int8)
filter_3_conv = filter_3.astype(np.int8)

	# Create the empty o matrix
empty_o = np.zeros((num_filter, M_pad, N_pad)).astype(np.int8)

result_1 = np.zeros((num_filter, M, N)).astype(np.int8) #Num_filter x M x N
result_3 = np.zeros((num_filter, M - 2, N - 2)).astype(np.int8) #Num_filter x M x N

for num in range(num_filter):
	for plane in range(L):
		result_1[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(filter_1_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid')).astype(np.uint8)
		result_3[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(filter_3_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid')).astype(np.uint8)


# Calculate a checksum
checksum = np.sum(result_1, dtype=np.uint64)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display
#vhex = np.vectorize(hex)
print("\n----------------------")
print("NCHW memory layout")
print("----------------------\n")

print("SINGLE SIZE\n")

print("INPUT\n")
print(tensor)
print("\n")
print("FILTERS\n")
print("1x1\n")
print(filter_1)
print("3x3\n")
print(filter_3)
print("\n")
print("RESULTS\n")
print("1x1\n")
print(result_1)
print("3x3\n")
print(result_3)

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("H_in", np.array(M_pad, dtype=np.uint64))
emit("W_in", np.array(N_pad, dtype=np.uint64))
emit("C_in", np.array(L, dtype=np.uint64))
#emit("F", np.array(filter_size, dtype=np.uint64))
emit("K", np.array(num_filter, dtype=np.uint64))
emit("i", tensor, 'NR_LANES*4')
emit("f", filter_3, 'NR_LANES*4')

#emit("golden_o_1", result_1, 'NR_LANES*4')
emit("golden_o", result_3, 'NR_LANES*4')

emit("o", empty_o, 'NR_LANES*4')
emit("o_checksum", checksum)


