#ifndef _pv_alloc_h
#define _pv_alloc_h
/**
 * Memory allocation macros
 *
 * Provides a standardized method for calling malloc and calloc and handling
 * any errors that occur.
 *
 * These helpers were written due to the repeated code written to handle memory
 * allocation failures. In all observed instances, the program usually called
 * exit(EXIT_FAILURE).
 *
 * These macros check for error conditions, log the file and line number of the failed
 * call, and then exit.
 *
 * The goals are:
 *
 * . consistent error handling
 * . easier debugging
 * . provide an opportunity to handle sending errors to other MPI ranks
 */

#include "utils/PVLog.hpp"
#include <stdlib.h>

/**
 * pvMalloc(size_t size);
 *
 * replaces standard malloc()
 */
#define pvMalloc(size) PV::pv_malloc(__FILE__, __LINE__, size)
/**
 * pvCalloc(count, size)
 *
 * replaces standard calloc()
 */
#define pvCalloc(count, size) PV::pv_calloc(__FILE__, __LINE__, count, size)
/**
 * pvMallocError(size, fmt, ...)
 *
 * Adds an error message if the allocation fails.
 */
#define pvMallocError(size, fmt, ...) PV::pv_malloc(__FILE__, __LINE__, size, fmt, ##__VA_ARGS__)
/**
 * pvCallocError(count, size, fmt, ...)
 *
 * Adds an error message if the allocation fails.
 */
#define pvCallocError(count, size, fmt, ...)                                                       \
   PV::pv_calloc(__FILE__, __LINE__, count, size, fmt, ##__VA_ARGS__)
/**
 * Wraps a call to delete
 *
 * Verifies that the pointer is not NULL for calling delete.
 */
#define pvDelete(ptr) PV::pv_delete(__FILE__, __LINE__, ptr)

namespace PV {

void *pv_malloc(const char *file, int line, size_t size);
void *pv_malloc(const char *file, int line, size_t size, const char *fmt, ...);
void *pv_calloc(const char *file, int line, size_t count, size_t size);
void *pv_calloc(const char *file, int line, size_t count, size_t size, const char *fmt, ...);

template <typename T>
void pv_delete(const char *file, int line, T *ptr) {
   if (ptr != NULL) {
      delete ptr;
      ptr = NULL;
   }
}
}

#endif
