#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <libgen.h>
#include "utils/PVLog.hpp"

namespace PV {

void vpv_log_with_prefix(FILE *stream, const char *prefix, const char *file, int line, const char *fmt, va_list args) {
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   fprintf(stream, "%s<%s:%d>: %s\n", prefix, basename((char*)file), line, msg);
}

void vpv_log_error(const char *file, int line, const char *fmt, va_list args) {
   // Flush stdout before printing to stderr. This makes the output
   // a bit cleaner if logging to the console
   fflush(stdout);
   vpv_log_with_prefix(stderr, "ERROR", file, line, fmt, args);
}

void vpv_log_debug(const char *file, int line, const char *fmt, va_list args) {
   vpv_log_with_prefix(stdout, "DEBUG", file, line, fmt, args);
}

void pv_log_error(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
}

void pv_exit_failure(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
   exit(EXIT_FAILURE);
}

void pv_log_info(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vfprintf(stdout, fmt, args);
   fprintf(stdout, "\n");
   va_end(args);
}

void pv_log_debug(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_debug(file, line, fmt, args);
   va_end(args);
}

}
