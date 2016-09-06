#ifndef PVLOG_HPP_
#define PVLOG_HPP_

#include <iostream>
#include <libgen.h>
#include <type_traits>
#include <string>
#include <cstdarg>


#ifdef NDEBUG
    // Not DEBUG build
    #ifdef PV_DEBUG_OUTPUT
       // Release build, but logDebug output was requested
       #define _PV_DEBUG_OUTPUT 1
    #else
       // Release build, no debug output
       #undef _PV_DEBUG_OUTPUT
    #endif // PV_DEBUG_OUTPUT
#else 
    // Debug build, logDebug output needed
    #define _PV_DEBUG_OUTPUT 1
#endif // NDEBUG

#if defined(_PV_DEBUG_OUTPUT)
#define DEBUG_LOG_TEST_CONDITION        true
#else
#define DEBUG_LOG_TEST_CONDITION        false
#endif // defined(_PV_DEBUG_OUTPUT)

//
// Logging with the C++ builder pattern.
//
// After PV_Init::initialize has been called, use the following macros instead
// of writing to stdout, stderr, std::cout, std::cerr, etc.
// This way, the file will go to stdout or stderr if the -l option is not used,
// but will go to the log file if it is.
//
// pvInfo() << "Some info" << std::endl;
// pvWarn() << "Print a warning" << std::endl;
// pvError() << "Print an error" << std::endl;
// pvErrorNoExit() << "Print an error" << std::endl;
// pvDebug() << "Some" << "stuff" << "to log" << std::endl;
//
// pvErrorIf uses printf style formatting:
// pvErrorIf(condition == true, "Condition %s was true.\n", conditionName);
//
// These macros create objects that send to the stream returned one of by PV::getOutputStream() or PV::getErrorStream().
// pvInfo() sends its output to the output stream.
// pvWarn() prepends its output with "WARN " and sends to the error stream.
// pvError() prepends its output with "ERROR <file,line>: " and sends to the error stream, then exits.
// pvError() prepends its output with "ERROR <file,line>: " and sends to the error stream, without exiting.
// pvDebug() prepends its output with "DEBUG <file,line>: " and sends to the output stream.
// In release versions, pvDebug() does not print any output, unless PetaVision was built using
// cmake -DPV_LOG_DEBUG:Bool=ON.
//
// The <file,line> returned by several of these macros gives the basename of the file where the macro was invoked,
// and the line number within that file.

// Because __LINE__ and __FILE__ evaluate to *this* file, not
// the file where the inline function is called. This is different
// from Clang, which puts in the __FILE__ and __LINE__ of the caller
#ifdef __GNUC__
#define pvInfo(...) PV::_Info __VA_ARGS__(__FILE__, __LINE__)
#define pvWarn(...) PV::_Warn __VA_ARGS__(__FILE__, __LINE__)
#define pvError(...) PV::_Error __VA_ARGS__(__FILE__, __LINE__)
#define pvErrorNoExit(...) PV::_ErrorNoExit __VA_ARGS__(__FILE__, __LINE__)
#define pvDebug(...) PV::_Debug __VA_ARGS__(__FILE__, __LINE__)
#define pvStackTrace(...) PV::_StackTrace __VA_ARGS__(__FILE__, __LINE__)
#define pvErrorIf(x, ...) if(x){ pvError().printf(__VA_ARGS__); }
#endif // __GNUC__

namespace PV {
enum LogTypeEnum {
   _LogInfo, _LogWarn, _LogError, _LogErrorNoExit, _LogDebug, _LogStackTrace
};

/**
 * Returns the stream used by pvInfo and, pvDebug
 * Typically, if a log file is set, the file is opened with write access and
 * getOutputStream() returns the resulting fstream.
 * If a log file is not set, this function returns std::cout.
 */
std::ostream& getOutputStream();

/**
 * Returns the stream used by pvWarn, pvError, and pvErrorNoExit.
 * Typically, if a log file is set, this returns the same ostream as getOutputStream(),
 * and if a log file is not set, this returns std::cerr.
 */
std::ostream& getErrorStream();

/**
 * A wide-character analog of getOutputStream().
 */
std::wostream& getWOutputStream();

/**
 * A wide-character analog of getOutputStream().
 */
std::wostream& getWErrorStream();

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

   // Should this log type flush the output stream before printing its
   // message?  If output and error streams are going to a terminal,
   // error messages can get lost if output is buffered and error
   // is unbuffered.
   static bool flushOutputStream();

   // Append prefix, file and line number? Used for suppressing
   // this information for LogInfoType
   static bool prependPrefix();

   // Should this log type print the file and line information?
   static bool outputFileLine();

   // Should this log type exit once the message is printed?
   static void exit();
};

typedef LogType<_LogInfo> LogInfoType;
typedef LogType<_LogWarn> LogWarnType;
typedef LogType<_LogError> LogErrorType;
typedef LogType<_LogErrorNoExit> LogErrorNoExitType;
typedef LogType<_LogDebug> LogDebugType;
typedef LogType<_LogStackTrace> LogStackTraceType;

// LogType traits
template<> inline std::string LogInfoType::prefix() { return std::string("INFO"); }
template<> inline std::string LogWarnType::prefix() { return std::string("WARN"); }
template<> inline std::string LogErrorType::prefix() { return std::string("ERROR"); }
template<> inline std::string LogErrorNoExitType::prefix() { return std::string("ERROR"); }
template<> inline std::string LogDebugType::prefix() { return std::string("DEBUG"); }
template<> inline std::string LogStackTraceType::prefix() { return std::string(""); }

template<> inline bool LogInfoType::output() { return true; }
template<> inline bool LogWarnType::output() { return true; }
template<> inline bool LogErrorType::output() { return true; }
template<> inline bool LogErrorNoExitType::output() { return true; }
template<> inline bool LogStackTraceType::output() { return true; }
template<> inline bool LogDebugType::output() { return DEBUG_LOG_TEST_CONDITION; }

template<> inline bool LogInfoType::flushOutputStream() { return false; }
template<> inline bool LogWarnType::flushOutputStream() { return false; }
template<> inline bool LogErrorType::flushOutputStream() { return true; }
template<> inline bool LogErrorNoExitType::flushOutputStream() { return true; }
template<> inline bool LogStackTraceType::flushOutputStream() { return true; }
template<> inline bool LogDebugType::flushOutputStream() { return true; }

template<> inline bool LogInfoType::prependPrefix() { return false; }
template<> inline bool LogWarnType::prependPrefix() { return true; }
template<> inline bool LogErrorType::prependPrefix() { return true; }
template<> inline bool LogErrorNoExitType::prependPrefix() { return true; }
template<> inline bool LogStackTraceType::prependPrefix() { return false; }
template<> inline bool LogDebugType::prependPrefix() { return true; }

template<> inline bool LogInfoType::outputFileLine() { return false; }
template<> inline bool LogWarnType::outputFileLine() { return false; }
template<> inline bool LogErrorType::outputFileLine() { return true; }
template<> inline bool LogErrorNoExitType::outputFileLine() { return true; }
template<> inline bool LogStackTraceType::outputFileLine() { return false; }
template<> inline bool LogDebugType::outputFileLine() { return true; }

template<> inline void LogInfoType::exit() {}
template<> inline void LogWarnType::exit() {}
template<> inline void LogErrorType::exit() { ::exit(EXIT_FAILURE); }
template<> inline void LogErrorNoExitType::exit() {}
template<> inline void LogStackTraceType::exit() {}
template<> inline void LogDebugType::exit() {}

// Log traits, for definining the stream to be used by the logger
template<typename C, typename LT, typename T = std::char_traits<C> >
struct LogStreamTraits {
   typedef T char_traits;
   typedef LT LogType;

   typedef std::basic_ostream<C,T>& (*StrFunc)(std::basic_ostream<C,T>&);
   static std::basic_ostream<C,T>& stream();
};

template<> inline std::ostream&  LogStreamTraits<char,    LogInfoType>::stream()        { return getOutputStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogInfoType>::stream()        { return getWOutputStream(); }
template<> inline std::ostream&  LogStreamTraits<char,    LogWarnType>::stream()        { return getErrorStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogWarnType>::stream()        { return getWErrorStream(); }
template<> inline std::ostream&  LogStreamTraits<char,    LogErrorType>::stream()       { return getErrorStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogErrorNoExitType>::stream() { return getWErrorStream(); }
template<> inline std::ostream&  LogStreamTraits<char,    LogErrorNoExitType>::stream() { return getErrorStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogErrorType>::stream()       { return getWErrorStream(); }
template<> inline std::ostream&  LogStreamTraits<char,    LogDebugType>::stream()       { return getOutputStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogDebugType>::stream()       { return getWOutputStream(); }
template<> inline std::ostream&  LogStreamTraits<char,    LogStackTraceType>::stream()  { return getErrorStream(); }
template<> inline std::wostream& LogStreamTraits<wchar_t, LogStackTraceType>::stream()  { return getWErrorStream(); }

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

   ~_Log() {
      LogType::exit();
   }

   void outputPrefix(const char *file, int line) {
      // In release this is optimised away if log type is debug
      if (LogType::output()) {
         if (LogType::flushOutputStream()) {
            getOutputStream().flush();
         }
         if (LogType::prependPrefix()) {
            _stream << LogType::prefix() << " ";
         }
         if (LogType::outputFileLine()) {
            _stream << "<" << basename((char*)file) << ":" << line << ">: ";
         }
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
   
   _Log& flush() {
      if (LogType::output()) {
         _stream.flush();
      }
      return *this;
   }

   /**
    * A method intended to make it easy to convert printf(format,...) and fprintf(stderr,format,...) statements
    * to versions that use the PVLog facilities.
    * To have output recorded in the logfile, replace printf(format,...) with pvInfo().printf(format,...)
    * and fprintf(stderr,format,...) with pvXXXX().printf(format,...), with pvXXXX replaced by
    * pvWarn, pvError, or pvErrorNoExit depending on the desired behavior.
    */
   int printf(char const * fmt, ...) {
      int chars_printed;
      if (LogType::output()) {
         va_list args1, args2;
         va_start(args1, fmt);
         va_copy(args2, args1);
         char c;
         int chars_needed = vsnprintf(&c, 1, fmt, args1);
         chars_needed++;
         char output_string[chars_needed];
         chars_printed = vsnprintf(output_string, chars_needed, fmt, args2);
         _stream << output_string;
         va_end(args1);
         va_end(args2);
         return chars_needed;
      }
      else {
         chars_printed = 0;
      }
      return chars_printed;
   }

private:
   std::basic_ostream<C,T>&  _stream;
};

#ifdef __GNUC__
//Macros that call these functions defined outside of namespace
typedef _Log<char,    LogDebugType>       _Debug;
typedef _Log<char,    LogInfoType>        _Info;
typedef _Log<char,    LogWarnType>        _Warn;
typedef _Log<char,    LogErrorType>       _Error;
typedef _Log<char,    LogErrorNoExitType> _ErrorNoExit;
typedef _Log<char,    LogStackTraceType>  _StackTrace;

typedef _Log<wchar_t, LogDebugType>       _WDebug;
typedef _Log<wchar_t, LogInfoType>        _WInfo;
typedef _Log<wchar_t, LogWarnType>        _WWarn;
typedef _Log<wchar_t, LogErrorType>       _WError;
typedef _Log<char,    LogErrorNoExitType> _WErrorNoExit;
typedef _Log<wchar_t, LogStackTraceType>  _WStackTrace;

#else
// Clang __FILE__ and __LINE__ evalaute to the caller of the function
// thus the macros aren't needed
typedef _Log<char,    LogDebugType>       Debug;
typedef _Log<char,    LogInfoType>        Info;
typedef _Log<char,    LogWarnType>        Warn;
typedef _Log<char,    LogErrorType>       Error;
typedef _Log<char,    LogErrorNoExitType> ErrorNoExit;
typedef _Log<char,    LogStackTraceType>  StackTrace;

typedef _Log<wchar_t, LogDebugType>       WDebug;
typedef _Log<wchar_t, LogInfoType>        WInfo;
typedef _Log<wchar_t, LogWarnType>        WWarn;
typedef _Log<wchar_t, LogErrorType>       WError;
typedef _Log<wchar_t, LogErrorNoExitType> WErrorNoExit;
typedef _Log<wchar_t, LogStackTraceType>  WStackTrace;
#endif // __GNUC__

// setLogFile sets the file that the pvDebug, pvInfo, pvWarn, and pvError streams write to.
void setLogFile(char const * logFile, std::ios_base::openmode mode=std::ios_base::out);
void setWLogFile(char const * logFile, std::ios_base::openmode mode=std::ios_base::out);

}  // end namespace PV


// Older, deprecated versions that use C-style FILE* streams instead of C++ ostreams.

#ifdef _PV_DEBUG_OUTPUT
/**
 * pvLogDebug is deprecated.  Use pvDebug methods instead.
 */
#define pvLogDebug(fmt, ...) PV::pv_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define pvLogDebug(fmt, ...)
#endif // _PV_DEBUG_OUTPUT

/*
 * pvLogInfo is deprecated.  Use pvInfo methods instead.
 */
#define pvLogInfo(fmt, ...) PV::pv_log_info(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

/*
 * pvLogWarn is deprecated.  Use pvWarn methods instead.
 */
#define pvLogWarn(fmt, ...) PV::pv_log_warn(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

/*
 * pvLogWarn is deprecated.  Use pvWarn methods instead.
 */
#define pvLogError(fmt, ...) PV::pv_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

/*
 * pvExitFailure is deprecated.  Use pvLogError (or, preferably, pvError) instead.
 */
#define pvExitFailure(fmt, ...) PV::pv_exit_failure(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

namespace PV {
/**
 * pv_log_info is deprecated.  Use pvInfo methods instead.
 */
void pv_log_info(const char *file, int line, const char *fmt, ...);

/**
 * pv_log_warn is deprecated.  Use pvWarn methods instead.
 */
void pv_log_warn(const char *file, int line, const char *fmt, ...);

/**
 * pv_log_error is deprecated.  Use pvError methods instead.
 */
void pv_log_error(const char *file, int line, const char *fmt, ...);

/**
 * pv_log_error_noexit is deprecated.  Use pvErrorNoExit methods instead.
 */
void pv_log_error_noexit(const char *file, int line, const char *fmt, ...);

/**
 * pv_log_debug is deprecated.  Use pvDebug methods instead.
 */
void pv_log_debug(const char *file, int line, const char *fmt, ...);

/**
 * pv_exit_failure is deprecated.  Use pvError methods instead.
 */
void pv_exit_failure(const char *file, int line, const char *fmt, ...);

// Commented out June 16, 2016.  The only functions to call these functions
// are deprecated and in contained in PVLog.cpp.
// If it becomes desirable to use the capability provided by these functions,
// we should add a vprintf method to _Log instead of reactivating these declarations.

// // Non-varargs versions, used internally, but exposed just in case
// // some other program wants to use them.
// void vpv_log_error(const char *file, int line, const char *fmt, va_list args);
// void vpv_log_debug(const char *file, int line, const char *fmt, va_list args);

}  // end namespace PV

// Clean up preprocessor
#ifdef _PV_DEBUG_OUTPUT
#undef _PV_DEBUG_OUTPUT
#endif // _PV_DEBUG_OUTPUT

#endif // PVLOG_HPP_
