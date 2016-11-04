#ifndef PVLOG_HPP_
#define PVLOG_HPP_

#include <cstdarg>
#include <iostream>
#include <libgen.h>
#include <string>
#include <type_traits>

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
#define DEBUG_LOG_TEST_CONDITION true
#else
#define DEBUG_LOG_TEST_CONDITION false
#endif // defined(_PV_DEBUG_OUTPUT)

//
// Logging with the C++ builder pattern.
//
// After PV_Init::initialize has been called, use the following macros instead
// of writing to stdout, stderr, std::cout, std::cerr, etc.
// This way, the file will go to stdout or stderr if the -l option is not used,
// but will go to the log file if it is.
//
// InfoLog() << "Some info" << std::endl;
// WarnLog() << "Print a warning" << std::endl;
// Fatal() << "Print an error" << std::endl;
// ErrorLog() << "Print an error" << std::endl;
// DebugLog() << "Some" << "stuff" << "to log" << std::endl;
//
// FatalIf uses printf style formatting:
// FatalIf(condition == true, "Condition %s was true. Exiting.\n", conditionName);
//
// These macros create objets that send to the stream returned one of by
// PV::getOutputStream() or PV::getErrorStream().
// InfoLog() sends its output to the output stream.
// WarnLog() prepends its output with "WARN " and sends to the error stream.
// Fatal() prepends its output to the error stream with "ERROR <file,line>: ", then exits.
// ErrorLog() prepends its output to the error stream with "ERROR <file,line>: "
// DebugLog() prepends its output with "DEBUG <file,line>: " and sends to the output stream.
// In release versions, DebugLog() does not print any output,
// unless PetaVision was built using cmake -DPV_LOG_DEBUG:Bool=ON.
//
// The <file,line> returned by several of these macros gives the basename of the file where
// the macro was invoked, and the line number within that file.

// Because __LINE__ and __FILE__ evaluate to *this* file, not
// the file where the inline function is called. This is different
// from Clang, which puts in the __FILE__ and __LINE__ of the caller
#ifdef __GNUC__
#define InfoLog(...) PV::_Info __VA_ARGS__(__FILE__, __LINE__)
#define WarnLog(...) PV::_Warn __VA_ARGS__(__FILE__, __LINE__)
#define Fatal(...) PV::_Error __VA_ARGS__(__FILE__, __LINE__)
#define ErrorLog(...) PV::_ErrorNoExit __VA_ARGS__(__FILE__, __LINE__)
#define DebugLog(...) PV::_Debug __VA_ARGS__(__FILE__, __LINE__)
#define StackTrace(...) PV::_StackTrace __VA_ARGS__(__FILE__, __LINE__)
#define FatalIf(x, ...)                                                                            \
   if (x) {                                                                                        \
      Fatal().printf(__VA_ARGS__);                                                                 \
   }
#endif // __GNUC__

namespace PV {
enum LogTypeEnum { _LogInfo, _LogWarn, _LogError, _LogErrorNoExit, _LogDebug, _LogStackTrace };

/**
 * Returns the stream used by InfoLog and, DebugLog
 * Typically, if a log file is set, the file is opened with write access and
 * getOutputStream() returns the resulting fstream.
 * If a log file is not set, this function returns std::cout.
 */
std::ostream &getOutputStream();

/**
 * Returns the stream used by WarnLog, Fatal, and ErrorLog.
 * Typically, if a log file is set, this returns the same ostream as getOutputStream(),
 * and if a log file is not set, this returns std::cerr.
 */
std::ostream &getErrorStream();

/**
 * A wide-character analog of getOutputStream().
 */
std::wostream &getWOutputStream();

/**
 * A wide-character analog of getOutputStream().
 */
std::wostream &getWErrorStream();

/**
 * LogType traits class. This is the protocol for defining traits
 * for various log types, like LogInfoType, etc.
 */
template <int T>
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
template <>
inline std::string LogInfoType::prefix() {
   return std::string("INFO");
}
template <>
inline std::string LogWarnType::prefix() {
   return std::string("WARN");
}
template <>
inline std::string LogErrorType::prefix() {
   return std::string("ERROR");
}
template <>
inline std::string LogErrorNoExitType::prefix() {
   return std::string("ERROR");
}
template <>
inline std::string LogDebugType::prefix() {
   return std::string("DEBUG");
}
template <>
inline std::string LogStackTraceType::prefix() {
   return std::string("");
}

template <>
inline bool LogInfoType::output() {
   return true;
}
template <>
inline bool LogWarnType::output() {
   return true;
}
template <>
inline bool LogErrorType::output() {
   return true;
}
template <>
inline bool LogErrorNoExitType::output() {
   return true;
}
template <>
inline bool LogStackTraceType::output() {
   return true;
}
template <>
inline bool LogDebugType::output() {
   return DEBUG_LOG_TEST_CONDITION;
}

template <>
inline bool LogInfoType::flushOutputStream() {
   return false;
}
template <>
inline bool LogWarnType::flushOutputStream() {
   return false;
}
template <>
inline bool LogErrorType::flushOutputStream() {
   return true;
}
template <>
inline bool LogErrorNoExitType::flushOutputStream() {
   return true;
}
template <>
inline bool LogStackTraceType::flushOutputStream() {
   return true;
}
template <>
inline bool LogDebugType::flushOutputStream() {
   return true;
}

template <>
inline bool LogInfoType::prependPrefix() {
   return false;
}
template <>
inline bool LogWarnType::prependPrefix() {
   return true;
}
template <>
inline bool LogErrorType::prependPrefix() {
   return true;
}
template <>
inline bool LogErrorNoExitType::prependPrefix() {
   return true;
}
template <>
inline bool LogStackTraceType::prependPrefix() {
   return false;
}
template <>
inline bool LogDebugType::prependPrefix() {
   return true;
}

template <>
inline bool LogInfoType::outputFileLine() {
   return false;
}
template <>
inline bool LogWarnType::outputFileLine() {
   return false;
}
template <>
inline bool LogErrorType::outputFileLine() {
   return true;
}
template <>
inline bool LogErrorNoExitType::outputFileLine() {
   return true;
}
template <>
inline bool LogStackTraceType::outputFileLine() {
   return false;
}
template <>
inline bool LogDebugType::outputFileLine() {
   return true;
}

template <>
inline void LogInfoType::exit() {}
template <>
inline void LogWarnType::exit() {}
template <>
inline void LogErrorType::exit() {
   ::exit(EXIT_FAILURE);
}
template <>
inline void LogErrorNoExitType::exit() {}
template <>
inline void LogStackTraceType::exit() {}
template <>
inline void LogDebugType::exit() {}

// Log traits, for definining the stream to be used by the logger
template <typename C, typename LT, typename T = std::char_traits<C>>
struct LogStreamTraits {
   typedef T char_traits;
   typedef LT LogType;

   typedef std::basic_ostream<C, T> &(*StrFunc)(std::basic_ostream<C, T> &);
   static std::basic_ostream<C, T> &stream();
};

template <>
inline std::ostream &LogStreamTraits<char, LogInfoType>::stream() {
   return getOutputStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogInfoType>::stream() {
   return getWOutputStream();
}
template <>
inline std::ostream &LogStreamTraits<char, LogWarnType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogWarnType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, LogErrorType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogErrorNoExitType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, LogErrorNoExitType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogErrorType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, LogDebugType>::stream() {
   return getOutputStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogDebugType>::stream() {
   return getWOutputStream();
}
template <>
inline std::ostream &LogStreamTraits<char, LogStackTraceType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, LogStackTraceType>::stream() {
   return getWErrorStream();
}

template <typename C, typename LT, typename T = std::char_traits<C>>
struct Log {
   typedef std::basic_ostream<C, T> basic_ostream;
   typedef LogStreamTraits<C, LT, T> LogStreamType;
   typedef LT LogType;

   Log(const char *file = __FILE__, int line = __LINE__) : _stream(LogStreamType::stream()) {
      outputPrefix(file, line);
   }

   Log(basic_ostream &s, const char *file = __FILE__, int line = __LINE__)
         : _stream(LogStreamType::stream()) {
      outputPrefix(file, line);
   }

   ~Log() { LogType::exit(); }

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
            _stream << "<" << basename((char *)file) << ":" << line << ">: ";
         }
      }
   }

   template <typename V>
   Log &operator<<(V const &value) {
      if (LogType::output()) {
         _stream << value;
      }
      return *this;
   }

   Log &operator<<(typename LogStreamType::StrFunc func) {
      if (LogType::output()) {
         func(_stream);
      }
      return *this;
   }

   Log &flush() {
      if (LogType::output()) {
         _stream.flush();
      }
      return *this;
   }

   /**
    * A method intended to make it easy to convert printf(format,...) and fprintf(stderr,format,...)
    * statements
    * to versions that use the PVLog facilities.
    * To have output recorded in the logfile, replace printf(format,...) with
    * InfoLog().printf(format,...)
    * and fprintf(stderr,format,...) with pvXXXX().printf(format,...), with pvXXXX replaced by
    * WarnLog, Fatal, or ErrorLog depending on the desired behavior.
    */
   int printf(char const *fmt, ...) {
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
   std::basic_ostream<C, T> &_stream;
};

#ifdef __GNUC__
// Macros that call these functions defined outside of namespace
typedef Log<char, LogDebugType> _Debug;
typedef Log<char, LogInfoType> _Info;
typedef Log<char, LogWarnType> _Warn;
typedef Log<char, LogErrorType> _Error;
typedef Log<char, LogErrorNoExitType> _ErrorNoExit;
typedef Log<char, LogStackTraceType> _StackTrace;

typedef Log<wchar_t, LogDebugType> _WDebug;
typedef Log<wchar_t, LogInfoType> _WInfo;
typedef Log<wchar_t, LogWarnType> _WWarn;
typedef Log<wchar_t, LogErrorType> _WError;
typedef Log<char, LogErrorNoExitType> _WErrorNoExit;
typedef Log<wchar_t, LogStackTraceType> _WStackTrace;

#else
// Clang __FILE__ and __LINE__ evalaute to the caller of the function
// thus the macros aren't needed
typedef Log<char, LogDebugType> Debug;
typedef Log<char, LogInfoType> Info;
typedef Log<char, LogWarnType> Warn;
typedef Log<char, LogErrorType> Error;
typedef Log<char, LogErrorNoExitType> ErrorNoExit;
typedef Log<char, LogStackTraceType> StackTrace;

typedef Log<wchar_t, LogDebugType> WDebug;
typedef Log<wchar_t, LogInfoType> WInfo;
typedef Log<wchar_t, LogWarnType> WWarn;
typedef Log<wchar_t, LogErrorType> WError;
typedef Log<wchar_t, LogErrorNoExitType> WErrorNoExit;
typedef Log<wchar_t, LogStackTraceType> WStackTrace;
#endif // __GNUC__

// setLogFile sets the file that the DebugLog, InfoLog, WarnLog, and Fatal streams write to.
void setLogFile(char const *logFile, std::ios_base::openmode mode = std::ios_base::out);
void setWLogFile(char const *logFile, std::ios_base::openmode mode = std::ios_base::out);

} // end namespace PV

#endif // PVLOG_HPP_
