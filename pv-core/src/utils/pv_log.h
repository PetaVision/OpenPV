#ifndef _pv_log_h
#define _pv_log_h

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

// Non-varargs versions
void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

#ifdef __cplusplus
}

#endif // __cplusplus

#endif
