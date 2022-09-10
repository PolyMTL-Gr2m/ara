##Keras2c
Keras2c is an open source code that transforms a Keras model into a C network. This version for Keras2c is edited to work on RISCV baremetal.

to build a the project you can use the following command
>make ara

to remove generated file use this command:

>make clean

you can also modify generate.py to download a custom network

for now the generated code *zanga.c*, *zanga.h* and *zanga_test_suit.h* can be used to generate a binary. However the code might need to be edited to work on baremeral, for example you need to add your own inlcude your own "printf.h"

you also might need to remove timinig function since timekeeping need OS to work, emulating using verilator can does the cycle count for you.

After using the both make command up there you can use:

>make verilate

this will copy the binary to the bin folder and then run verilator on the generated RISCV binary. Note that you need to have already ran throguht the ara installation first.

##Running verilator on the source code
------------------------------------------------
Ubuntu 20.04.4 LTS
Ara Dependencies:
GCC=9.4.0
CMake=3.16.3
Clang=10
Ninja=1.10.0
texinfo
flex
bison
autoconf
libelf-dev
gtkwave
------------------------------------------------
How to check waveform:
Add "trace=1" on the Makefile of "hardware" file
Then "make varilate"
And "app=** make simv" again
Final run"gtkwave **.fst"
------------------------------------------------
RISC-V "V" Vector Extension C Intrinsics can work
on Ara.
------------------------------------------------
