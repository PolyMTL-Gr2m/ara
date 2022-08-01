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
		
def emit_64b(name, array, alignment='3'):
	print(".global %s" % name)
	print(".align " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n]
		print("    .word 0x%s" % s)



# Input image
M = 32
N = 32
L = 3

#assert(M % 8 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
#assert(N % 8 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
#assert(L / filter_size != 1), "Depth of tensor must be same depth as the filter"

# Generate a random int8 input image
tensor = rand_tensor(1, L, M, N, -128).astype(np.float64)
np.random.shuffle(tensor.flat)
	
# Create the empty o matrix
empty_o = np.zeros((L, M, N)).astype(np.float64)

# Calculate the output matrix		
result = np.maximum(tensor, empty_o)
	

# Calculate a checksum
checksum = np.sum(result, dtype=np.uint64)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display

print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
print(tensor)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(L, M, N))
print(result)
print("\n")
print(checksum)

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("L", np.array(L, dtype=np.uint64))
emit("i", tensor, 'NR_LANES*4')
emit("o", empty_o, 'NR_LANES*4')
emit("golden_o", result, 'NR_LANES*4')
emit("o_checksum", checksum)


