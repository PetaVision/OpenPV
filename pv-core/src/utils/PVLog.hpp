#ifndef _pv_log_h
#define _pv_log_h

/**
 * logging macros and functions
 *
 * Provides a standardized method for logging error, debug and information messages.
 *
 * logDebug(), logError() and logInfo() all work the same as printf(). 
 *
 * logDebug() and logError() will prepend DEBUG: and ERROR: plus the file name and line number
 * of the call.
 *
 * logInfo() works just like printf. Nothing is prepended.
 *
 * logDebug() calls are compiled out in a Release build. logError() and logInfo() calls are not
 * compiled out.
 *
 * These macros provide an opportunity to handle sending debug, error and info messages to other MPI ranks
 */

namespace PV {

#ifdef NDEBUG
    // Not DEBUG build
    #ifdef PV_DEBUG_OUTPUT
       // Release build, but logDebug output was requested
       #define _PV_DEBUG_OUTPUT 1
    #else
       // Release build, no debug output
       #undef _PV_DEBUG_OUTPUT
  #endif
#else 
    // Debug build, logDebug output needed
    #define _PV_DEBUG_OUTPUT 1
#endif

#ifdef _PV_DEBUG_OUTPUT
#define logDebug(fmt, ...) pv_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define logDebug(fmt, ...)
#endif

#define logError(fmt, ...) pv_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define logInfo(fmt, ...) pv_log_info(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define exitFailure(fmt, ...) pv_exit_failure(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

void pv_log_error(const char *file, int line, const char *fmt, ...);
void pv_log_debug(const char *file, int line, const char *fmt, ...);
void pv_log_info(const char *file, int line, const char *fmt, ...);
void pv_exit_failure(const char *file, int line, const char *fmt, ...);

// Non-varargs versions, used internally, but exposed just in case somebody
// some other program wants to use them
void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

}

// Clean up preprocessor
#ifdef _PV_DEBUG_OUTPUT
#undef _PV_DEBUG_OUTPUT
#endif

#endif
