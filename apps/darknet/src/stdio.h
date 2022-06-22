#ifndef __STDIO__
#define __STDIO__

#define HEAP_SIZE (1*1024*1024)

extern char *malloc();
// extern int printf(const char *format, ...);

// extern void *memcpy(void *dest, const void *src, long n);
// extern char *strcpy(char *dest, const char *src);
// extern int strcmp(const char *s1, const char *s2);

char heap_memory[HEAP_SIZE];
int heap_memory_used = 0;

#endif
