#include <stdint.h>
#include <string.h>
#include "printf.h"
#include "tiny_malloc.h"

// #include "parser.h"
// #include "utils.h"
// #include "cifar_cfg_test_str.h"

// extern void test_cifar(char *filename, char *weightfile);
#define MAT_SIZE 10000

int main()
{
    printf("Hello Wolrd from darknet!\n");
    char* buffer = malloc(MAT_SIZE);
    for (int i=0; i<MAT_SIZE; i++){
        buffer[i] = i;
    }
    for (int i=0; i<MAT_SIZE; i++){
        printf("wow %p\n", &buffer[i]);
    }
    // test_cifar(cifar_cfg_test_str, NULL);

    return 0;
}

