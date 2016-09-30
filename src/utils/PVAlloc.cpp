#include "PVAlloc.hpp"
#include "utils/PVLog.hpp"
#include <stdarg.h>
#include <stdio.h>

namespace PV {

void *pv_malloc(const char *file, int line, size_t size) {
   void *ptr = malloc(size);
   if (ptr == NULL) {
      pv_log_error(file, line, "malloc(%zu) failed\n", size);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

void *pv_calloc(const char *file, int line, size_t count, size_t size) {
   void *ptr = calloc(count, size);
   if (ptr == NULL) {
      pv_log_error(file, line, "calloc(%zu, %zu) failed\n", count, size);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

void *pv_malloc(const char *file, int line, size_t size, const char *fmt, ...) {
   void *ptr = malloc(size);
   if (ptr == NULL) {
      /* Build up custom error string */
      va_list args;
      va_start(args, fmt);
      static int buf_size = 1024;
      char msg[buf_size];
      vsnprintf(msg, buf_size, fmt, args);
      va_end(args);

      /* Log the error */
      pv_log_error(file, line, "malloc(%zu) failed: %s\n", size, msg);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

void *pv_calloc(const char *file, int line, size_t count, size_t size, const char *fmt, ...) {
   void *ptr = calloc(count, size);
   if (ptr == NULL) {
      /* Build up custom error string */
      va_list args;
      va_start(args, fmt);
      static int buf_size = 1024;
      char msg[buf_size];
      vsnprintf(msg, buf_size, fmt, args);
      va_end(args);

      /* Log the error */
      pv_log_error(file, line, "calloc(%zu, %zu) failed: %s\n", count, size, msg);
      exit(EXIT_FAILURE);
   }
   return ptr;
}
}
