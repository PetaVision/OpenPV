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
#ifdef DEBUG
#define LOG_DEBUG_OUTPUT 1
#endif

#ifdef LOG_DEBUG_OUTPUT
#define logDebug(fmt, ...) pv_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define logDebug(fmt, ...)
#endif

#define logError(fmt, ...) pv_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define logInfo(fmt, ...) pv_log_info(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void pv_log_error(const char *file, int line, const char *fmt, ...);
void pv_log_debug(const char *file, int line, const char *fmt, ...);
void pv_log_info(const char *file, int line, const char *fmt, ...);

// Non-varargs versions, used internally, but exposed just in case somebody
// some other program wants to use them
void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

#ifdef __cplusplus
}

#endif // __cplusplus

#endif
