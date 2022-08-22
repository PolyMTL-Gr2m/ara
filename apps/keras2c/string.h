#include <stdio.h>
#ifndef STRING_H
#define STRING_H

void* memcpy(void* dest, const void* src, size_t len);
void* memset(void* dest, int byte, size_t len);
int strcmp(const char* s1, const char* s2);
size_t strlen(const char *s);
#endif
