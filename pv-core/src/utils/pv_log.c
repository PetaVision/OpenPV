#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <libgen.h>
#include "pv_log.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void vpv_log_with_prefix(FILE *stream, const char *prefix, const char *file, int line, const char *fmt, va_list args) {
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   fprintf(stream, "%s<%s:%d>: %s\n", prefix, basename((char*)file), line, msg);
   fflush(stream);
}

void vpv_log_error(const char *file, int line, const char *fmt, va_list args) {
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

void pv_log_info(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vfprintf(stdout, fmt, args);
   va_end(args);
}

void pv_log_debug(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_debug(file, line, fmt, args);
   va_end(args);
}

#ifdef __cplusplus
}
#endif // __cplusplus
