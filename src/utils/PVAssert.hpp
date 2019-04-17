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
void print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63);
} // end namespace PV

#endif
