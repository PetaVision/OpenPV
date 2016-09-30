#ifndef _pv_assert_h
#define _pv_assert_h

/**
 * Replacement for the assert() macro. By default, assert() is compiled out in non-release builds.
 * We
 * can stop that behavior by redefining Release compiler arguments, but this introduces compiler
 * portability
 * issues.
 *
 * pvAssert() and pvAssertMessage() add additional functionality to assert()
 *
 * 1) The filename and line number are printing
 * 2) A stack backtrace is provided
 *
 * This macro is intended to be used in C++ programs, not in C programs.
 *
 * Also, by providing a PetaVision implementation of this macro, an opportunity to handle sending
 * assert failures
 * to other MPI ranks becomes possible.
 */
#include "utils/PVLog.hpp"
#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef NDEBUG
#define pvAssert(c)
#define pvAssertMessage(c, fmt, ...)
#else

/**
 * Works just like assert(), except it is not compiled out in Release versions and provides a stack
 * trace and
 * prints out the failed condition and the filename:line number of the failure
 */
#define pvAssert(c)                                                                                \
   if (!(c)) {                                                                                     \
      PV::pv_assert_failed(__FILE__, __LINE__, #c);                                                \
   }
/**
 * Like pvAssert(). Adds an additional error message to the output.
 */
#define pvAssertMessage(c, fmt, ...)                                                               \
   if (!(c)) {                                                                                     \
      PV::pv_assert_failed_message(__FILE__, __LINE__, #c, fmt, ##__VA_ARGS__);                    \
   }
#endif

namespace PV {
void pv_assert_failed(const char *file, int line, const char *condition);
void pv_assert_failed_message(
      const char *file,
      int line,
      const char *condition,
      const char *fmt,
      ...);

/** Print a demangled stack backtrace of the caller function to FILE* out. */
static inline void print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63) {
   pvStackTrace() << "stack trace:" << std::endl;

   // storage array for stack trace address data
   void *addrlist[max_frames + 1];

   // retrieve current stack addresses
   int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void *));

   if (addrlen == 0) {
      pvStackTrace() << "  <empty, possibly corrupt>" << std::endl;
      return;
   }

   // resolve addresses into strings containing "filename(function+address)",
   // this array must be free()-ed
   char **symbollist = backtrace_symbols(addrlist, addrlen);

   // allocate string which will be filled with the demangled function name
   size_t funcnamesize = 256;
   char *funcname      = (char *)malloc(funcnamesize);

   // iterate over the returned symbol lines. skip the first, it is the
   // address of this function.
   for (int i = 1; i < addrlen; i++) {
      char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for (char *p = symbollist[i]; *p; ++p) {
         if (*p == '(')
            begin_name = p;
         else if (*p == '+')
            begin_offset = p;
         else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
         }
      }

      if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
         *begin_name++   = '\0';
         *begin_offset++ = '\0';
         *end_offset     = '\0';

         // mangled name is now in [begin_name, begin_offset) and caller
         // offset in [begin_offset, end_offset). now apply
         // __cxa_demangle():

         int status;
         char *ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
         if (status == 0) {
            funcname = ret; // use possibly realloc()-ed string
            pvStackTrace() << "  " << symbollist[i] << " : " << funcname << "+" << begin_offset
                           << std::endl;
         } else {
            // demangling failed. Output function name as a C function with
            // no arguments.
            pvStackTrace() << "  " << symbollist[i] << " : " << begin_name << "+" << begin_offset
                           << std::endl;
         }
      } else {
         // couldn't parse the line? print the whole line.
         pvStackTrace() << "  " << symbollist[i] << std::endl;
      }
   }

   free(funcname);
   free(symbollist);
}
}

#endif
