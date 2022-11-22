KERAS_INC=include/fmatmul.c include/fmatmul.h  include/imatmul.c include/imatmul.h  include/k2c_activations.c include/k2c_convolution_layers.c include/k2c_core_layers.c include/k2c_embedding_layers.c include/k2c_helper_functions.c include/k2c_merge_layers.c include/k2c_normalization_layers.c include/k2c_pooling_layers.c include/k2c_recurrent_layers.c  include/fconv2d7by7.c include/fpooling_tensor32.h include/fReLu_tensor32.h include/fpooling_tensor32.c include/fReLu_tensor32.c 

COMMON_INC= ./common/fconv2d_tensor32.h ../common/printf.c ../common/serial.c  ../riscv-tests/benchmarks/common/syscalls.c ../riscv-tests/benchmarks/common/crt.S -T ../riscv-tests/benchmarks/common/test.ld ./common/fconv2d_tensor32.c

GCC=../../install/riscv-gcc/bin/riscv64-unknown-elf-gcc

test:
	python -m keras2c example/model.h5 zanga -t 10
all:
	riscv64-unknown-elf-gcc -std=c99 -I./include/ -o zanga zanga.c zanga_test_suite.c  -L./include/ -l:libkeras2c.a -lm
ara:
	  $(GCC) data.S -I../../env -I../common  -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-math-errno  -DNR_LANES=4  main.c  zanga.c  helper_func.c $(COMMON_INC) $(KERAS_INC) ~/ara/install/riscv-llvm/riscv64-unknown-elf/lib/libm.a -lc -fno-math-errno -static -nostdlib -nostartfiles -I ./include -I . -o keras2c.riscv 
verilate:
	cp -f keras2c.riscv ../bin/; cd ../../hardware; app=keras2c.riscv make simv 
clean:
	rm -f zanga *.o ; rm -f keras2c.riscv*; rm -f ../bin/keras2c.riscv; cd include; rm -f *.o *.a;
