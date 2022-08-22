#include "helper_func.h"
void * Memcpy(void * dst,
		  const void * src, unsigned int cnt) {
	  char * pszDest = (char * ) dst;
	    const char * pszSource = (const char * ) src;
	      if ((pszDest != NULL) && (pszSource != NULL)) {
		          while (cnt) {
				        *(pszDest++) = * (pszSource++);
					      --cnt;
					          }
			    }
	        return dst;
}


void memset(void *s, int c,  unsigned int len)
{
	unsigned char* p=s;
      	while(len--){
         *p++ = (unsigned char)c; }
	return s;
}
