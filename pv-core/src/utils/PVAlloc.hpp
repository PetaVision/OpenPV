#ifndef _pv_alloc_h
#define _pv_alloc_h

#include <stdlib.h>
#include "utils/pv_log.h"

#define pvMalloc(size) pv_malloc(__FILE__, __LINE__, size)
#define pvCalloc(count, size) pv_calloc(__FILE__, __LINE__, count, size)
#define pvMallocError(size, fmt, ...) pv_malloc(__FILE__, __LINE__, size, fmt, ##__VA_ARGS__)
#define pvCallocError(count, size, fmt, ...) pv_calloc(__FILE__, __LINE__, count, size, fmt, ##__VA_ARGS__)
#define pvDelete(ptr) pv_delete(__FILE__, __LINE__, ptr)

void *pv_malloc(const char *file, int line, size_t size);
void *pv_malloc(const char *file, int line, size_t size, const char *fmt, ...);
void *pv_calloc(const char *file, int line, size_t count, size_t size);
void *pv_calloc(const char *file, int line, size_t count, size_t size, const char *fmt, ...);

template<typename T>
void pv_delete(const char *file, int line, T *ptr) {
   if (ptr != NULL) {
      delete ptr;
      ptr = NULL;
   }
}

#endif
