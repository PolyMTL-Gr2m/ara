KERAS_INC=include/k2c_activations.c include/k2c_convolution_layers.c include/k2c_core_layers.c include/k2c_embedding_layers.c include/k2c_helper_functions.c include/k2c_merge_layers.c include/k2c_normalization_layers.c include/k2c_pooling_layers.c include/k2c_recurrent_layers.c

GCC=../../install/riscv-gcc/bin/riscv64-unknown-elf-gcc

test:
	python -m keras2c example/model.h5 zanga -t 10
all:
	riscv64-unknown-elf-gcc -std=c99 -I./include/ -o zanga zanga.c zanga_test_suite.c  -L./include/ -l:libkeras2c.a -lm
ara:
	$(GCC) -I../../env -I../common  -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-math-errno  main.c  zanga.c printf.c  $(KERAS_INC) syscalls.c ./common/crt.S ~/ara/install/riscv-llvm/riscv64-unknown-elf/lib/libm.a -lc -fno-math-errno -static -nostdlib -nostartfiles -T ./common/test.ld -I ./include -I . -o keras2c.riscv
	
clean:
	rm -f zanga *.o ; rm -f keras2c.riscv; cd include; rm -f *.o *.a;
