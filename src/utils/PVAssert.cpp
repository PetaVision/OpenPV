#include "PVAssert.hpp"
#include <stdio.h>
#include <stdarg.h>

namespace PV {

void pv_assert_failed(const char *file, int line, const char *condition) {
   pv_log_error(file, line, "assert failed: %s", condition);
   print_stacktrace(stderr);
   exit(EXIT_FAILURE);
}

void pv_assert_failed_message(const char *file, int line, const char *condition, const char *fmt, ...) {
   /* Build up custom error string */
   va_list args;
   va_start(args, fmt);
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   va_end(args);

   pv_log_error(file, line, "assert failed: %s: %s", condition, msg);
   print_stacktrace(stderr);
   exit(EXIT_FAILURE);
}

}