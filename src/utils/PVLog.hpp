#ifndef _pv_log_h
#define _pv_log_h

#include <iostream>
#include <libgen.h>
#include <type_traits>
#include <string>


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

//
// Logging with the C++ builder pattern.
//
// This allows one to log the appropriate file (like stdout or stderr) using:
//
// Debug() << "Some" << "stuff" << "to log" << std::endl;
// Error() << "Print an error" << std::endl;
// Info() << "Some info" << std::endl;
//
// Anything sent to Debug() is compiled out in Release versions, unless
// cmake -DPV_LOG_DEBUG:Bool=On is specified.
//
// Debug() and Error() preprend either:
//   DEBUG<Filename:line number>
//   ERROR<Filename:line number>
//
// To the output. Info() does not have anything prepended to each line.
//

#if defined(_PV_DEBUG_OUTPUT)
#define DEBUG_LOG_TEST_CONDITION        true
#else
#define DEBUG_LOG_TEST_CONDITION        false
#endif

// Because __LINE__ and __FILE__ evaluate to *this* file, not
// the file where the inline function is called. This is different
// from Clang, which puts in the __FILE__ and __LINE__ of the caller
#ifdef __GNUC__
#define pvDebug() PV::_Debug(__FILE__, __LINE__)
#define pvInfo() PV::_Info(__FILE__, __LINE__)
#define pvError() PV::_Error(__FILE__, __LINE__)
#define pvStackTrace() PV::_StackTrace(__FILE__, __LINE__)
#endif

#ifdef _PV_DEBUG_OUTPUT
#define pvLogDebug(fmt, ...) PV::pv_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define pvLogDebug(fmt, ...)
#endif

#define pvLogError(fmt, ...) PV::pv_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define pvLogInfo(fmt, ...) PV::pv_log_info(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define pvExitFailure(fmt, ...) PV::pv_exit_failure(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

namespace PV {
enum LogTypeEnum {
   _LogInfo, _LogError, _LogDebug, _LogStackTrace
};

/**
 * LogType traits class. This is the protocol for defining traits
 * for various log types, like LogInfoType, etc.
 */
template<int T>
struct LogType {
   static std::string prefix();
   // Should this log type generate output? This is used
   // for suppressing debug output in release builds
   static bool output();
   // Append prefix, file and line number? Used for suppressing
   // this information for LogInfoType
   static bool prependPrefix();
};

typedef LogType<_LogInfo> LogInfoType;
typedef LogType<_LogError> LogErrorType;
typedef LogType<_LogDebug> LogDebugType;
typedef LogType<_LogStackTrace> LogStackTraceType;

// LogType traits
template<> inline std::string LogInfoType::prefix() { return std::string("INFO"); }
template<> inline std::string LogErrorType::prefix() { return std::string("ERROR"); }
template<> inline std::string LogDebugType::prefix() { return std::string("DEBUG"); }
template<> inline std::string LogStackTraceType::prefix() { return std::string(""); }
template<> inline bool LogInfoType::output() { return true; }
template<> inline bool LogErrorType::output() { return true; }
template<> inline bool LogStackTraceType::output() { return true; }
template<> inline bool LogDebugType::output() { return DEBUG_LOG_TEST_CONDITION; }
template<> inline bool LogInfoType::prependPrefix() { return false; }
template<> inline bool LogErrorType::prependPrefix() { return true; }
template<> inline bool LogStackTraceType::prependPrefix() { return false; }
template<> inline bool LogDebugType::prependPrefix() { return true; }

// Log traits, for definining the stream to be used by the logger
template<typename C, typename LT, typename T = std::char_traits<C> >
struct LogStreamTraits {
   typedef T char_traits;
   typedef LT LogType;

   typedef std::basic_ostream<C,T>& (*StrFunc)(std::basic_ostream<C,T>&);
   static std::basic_ostream<C,T>& stream();
};

template<> inline std::ostream&  LogStreamTraits<char,    LogInfoType>::stream() { return std::cout; }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogInfoType>::stream() { return std::wcout; }
template<> inline std::ostream&  LogStreamTraits<char,    LogErrorType>::stream() { return std::cerr; }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogErrorType>::stream() { return std::wcerr; }
template<> inline std::ostream&  LogStreamTraits<char,    LogStackTraceType>::stream() { return std::cerr; }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogStackTraceType>::stream() { return std::wcerr; }
template<> inline std::ostream&  LogStreamTraits<char,    LogDebugType>::stream() { return std::cout; }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogDebugType>::stream() { return std::wcout; }

template<typename C, typename LT, typename T = std::char_traits<C> >
struct _Log
{
   typedef std::basic_ostream<C,T> basic_ostream;
   typedef LogStreamTraits<C,LT,T> LogStreamType;
   typedef LT LogType;

   _Log(const char *file = __FILE__, int line = __LINE__)
   : _stream(LogStreamType::stream())
   {
      outputPrefix(file, line);
   }

   _Log(basic_ostream& s, const char *file = __FILE__, int line = __LINE__)
   : _stream(LogStreamType::stream())
   {
      outputPrefix(file, line);
   }

   void outputPrefix(const char *file, int line) {
      // In release this is optimised away if log type is debug
      if (LogType::prependPrefix() && LogType::output()) {
         _stream << LogType::prefix() << "<" << basename((char*)file) << ":" << line << ">: ";
      }
   }

   template<typename V>
   _Log& operator<<(V const& value) {
      if (LogType::output()) {
         _stream << value;
      }
      return *this;
   }

   _Log& operator<<(typename LogStreamType::StrFunc func) {
      if (LogType::output()) {
         func(_stream);
      }
      return *this;
   }
   
private:
   std::basic_ostream<C,T>&  _stream;
};

#ifdef __GNUC__
//Macros that call these functions defined outside of namespace
typedef _Log<char,    LogDebugType>      _Debug;
typedef _Log<char,    LogInfoType>       _Info;
typedef _Log<char,    LogErrorType>      _Error;
typedef _Log<char,    LogStackTraceType> _StackTrace;

typedef _Log<wchar_t, LogDebugType>      _WDebug;
typedef _Log<wchar_t, LogInfoType>       _WInfo;
typedef _Log<wchar_t, LogErrorType>      _WError;
typedef _Log<wchar_t, LogStackTraceType> _WStackTrace;

#else
// Clang __FILE__ and __LINE__ evalaute to the caller of the function
// thus the macros aren't needed
typedef _Log<char,    LogDebugType>      Debug;
typedef _Log<char,    LogInfoType>       Info;
typedef _Log<char,    LogErrorType>      Error;
typedef _Log<char,    LogStackTraceType> StackTrace;

typedef _Log<wchar_t, LogDebugType>      WDebug;
typedef _Log<wchar_t, LogInfoType>       WInfo;
typedef _Log<wchar_t, LogErrorType>      WError;
typedef _Log<wchar_t, LogStackTraceType> WStackTrace;
#endif

/**
 * logging macros and functions
 *
 * Provides a standardized method for logging error, debug and information messages.
 *
 * pvLogDebug(), pvLogError() and pvLogInfo() take printf-style format control string and arguments.
 *
 * pvLogDebug() and pvLogError() will prepend DEBUG: and ERROR: plus the file name and line number
 * of the call, and append a newline.
 *
 * pvLogInfo() appends a newline. Nothing is prepended.
 *
 * pvLogDebug() calls are compiled out in a Release build. pvLogError() and pvLogInfo() calls are not
 * compiled out.
 *
 * pvLogDebug() and pvLogInfo() send output to stdout.  pvLogError sends output to stderr.
 *
 * These macros provide an opportunity to handle sending debug, error and info messages to other MPI ranks
 */


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
