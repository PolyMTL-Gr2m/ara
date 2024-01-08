Python gen_data_script is used to generate a file with multiple results for different kernel size (1,3,5,7).
It's meant to be used with the automation script presented in the main.c file.
It runs, tests and benchmarks a certain range input according to different kernel and input size.

By default, a the data.S file is already done for input up to 32x32 with 3 channels

The code has to be compiled and can be run using ModelSim or Verilator.

This can be achieved by following the configuration from :

https://github.com/pulp-platform/ara/blob/main/README.md