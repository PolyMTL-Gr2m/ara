##Keras2c
Keras2c is an open source code that transforms a Keras model into a C network. This version for Keras2c is edited to work on RISCV baremetal.

to build a the project you can use the following command
>make ara

to remove generated file use this command:

>make clean

you can also modify generate.py to download a custom network

for now the generated code *zanga.c*, *zanga.h* and *zanga_test_suit.h* can be used to generate a binary. However the code might need to be edited to work on baremeral, for example you need to add your own inlcude your own "printf.h"

you also might need to remove timinig function since timekeeping need OS to work, emulating using verilator can does the cycle count for you.

