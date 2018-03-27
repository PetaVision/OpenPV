#ifndef PVLOG_HPP_
#define PVLOG_HPP_

#include <cstdarg>
#include <iostream>
#include <libgen.h>
#include <string>
#include <type_traits>

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
#define InfoLog(...) PV::InfoLog __VA_ARGS__(__FILE__, __LINE__)
#define WarnLog(...) PV::WarnLog __VA_ARGS__(__FILE__, __LINE__)
#define Fatal(...) PV::Fatal __VA_ARGS__(__FILE__, __LINE__)
#define ErrorLog(...) PV::ErrorLog __VA_ARGS__(__FILE__, __LINE__)
#define DebugLog(...) PV::DebugLog __VA_ARGS__(__FILE__, __LINE__)
#define StackTrace(...) PV::StackTrace __VA_ARGS__(__FILE__, __LINE__)
#define FatalIf(x, ...)                                                                            \
   if (x) {                                                                                        \
      Fatal().printf(__VA_ARGS__);                                                                 \
   }
#endif // __GNUC__

namespace PV {
enum LogTypeEnum {
   LogInfoType,
   LogWarnType,
   LogFatalType,
   LogErrorType,
   LogDebugType,
   LogStackTraceType
};

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
 * for various log types, like InfoLogType, etc.
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
   // this information for InfoLogType
   static bool prependPrefix();

   // Should this log type print the file and line information?
   static bool outputFileLine();

   // Should this log type exit once the message is printed?
   static void exit();
};

typedef LogType<LogInfoType> InfoLogType;
typedef LogType<LogWarnType> WarnLogType;
typedef LogType<LogFatalType> FatalType;
typedef LogType<LogErrorType> ErrorLogType;
typedef LogType<LogDebugType> DebugLogType;
typedef LogType<LogStackTraceType> StackTraceType;

// LogType traits
template <>
inline std::string InfoLogType::prefix() {
   return std::string("INFO");
}
template <>
inline std::string WarnLogType::prefix() {
   return std::string("WARN");
}
template <>
inline std::string FatalType::prefix() {
   return std::string("ERROR");
}
template <>
inline std::string ErrorLogType::prefix() {
   return std::string("ERROR");
}
template <>
inline std::string DebugLogType::prefix() {
   return std::string("DEBUG");
}
template <>
inline std::string StackTraceType::prefix() {
   return std::string("");
}

template <>
inline bool InfoLogType::output() {
   return true;
}
template <>
inline bool WarnLogType::output() {
   return true;
}
template <>
inline bool FatalType::output() {
   return true;
}
template <>
inline bool ErrorLogType::output() {
   return true;
}
template <>
inline bool StackTraceType::output() {
   return true;
}
template <>
inline bool DebugLogType::output() {
#if defined(NDEBUG) || defined(PV_DEBUG_OUTPUT)
   return true;
#else
   return false;
#endif // defined(NDEBUG) || defined(PV_DEBUG_OUTPUT)
}

template <>
inline bool InfoLogType::flushOutputStream() {
   return true;
}
template <>
inline bool WarnLogType::flushOutputStream() {
   return true;
}
template <>
inline bool FatalType::flushOutputStream() {
   return true;
}
template <>
inline bool ErrorLogType::flushOutputStream() {
   return true;
}
template <>
inline bool StackTraceType::flushOutputStream() {
   return true;
}
template <>
inline bool DebugLogType::flushOutputStream() {
   return true;
}

template <>
inline bool InfoLogType::prependPrefix() {
   return false;
}
template <>
inline bool WarnLogType::prependPrefix() {
   return true;
}
template <>
inline bool FatalType::prependPrefix() {
   return true;
}
template <>
inline bool ErrorLogType::prependPrefix() {
   return true;
}
template <>
inline bool StackTraceType::prependPrefix() {
   return false;
}
template <>
inline bool DebugLogType::prependPrefix() {
   return true;
}

template <>
inline bool InfoLogType::outputFileLine() {
   return false;
}
template <>
inline bool WarnLogType::outputFileLine() {
   return false;
}
template <>
inline bool FatalType::outputFileLine() {
   return true;
}
template <>
inline bool ErrorLogType::outputFileLine() {
   return true;
}
template <>
inline bool StackTraceType::outputFileLine() {
   return false;
}
template <>
inline bool DebugLogType::outputFileLine() {
   return true;
}

template <>
inline void InfoLogType::exit() {}
template <>
inline void WarnLogType::exit() {}
template <>
inline void FatalType::exit() {
   ::exit(EXIT_FAILURE);
}
template <>
inline void ErrorLogType::exit() {}
template <>
inline void StackTraceType::exit() {}
template <>
inline void DebugLogType::exit() {}

// Log traits, for definining the stream to be used by the logger
template <typename C, typename LT, typename T = std::char_traits<C>>
struct LogStreamTraits {
   typedef T char_traits;
   typedef LT LogType;

   typedef std::basic_ostream<C, T> &(*StrFunc)(std::basic_ostream<C, T> &);
   static std::basic_ostream<C, T> &stream();
};

template <>
inline std::ostream &LogStreamTraits<char, InfoLogType>::stream() {
   return getOutputStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, InfoLogType>::stream() {
   return getWOutputStream();
}
template <>
inline std::ostream &LogStreamTraits<char, WarnLogType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, WarnLogType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, FatalType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, ErrorLogType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, ErrorLogType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, FatalType>::stream() {
   return getWErrorStream();
}
template <>
inline std::ostream &LogStreamTraits<char, DebugLogType>::stream() {
   return getOutputStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, DebugLogType>::stream() {
   return getWOutputStream();
}
template <>
inline std::ostream &LogStreamTraits<char, StackTraceType>::stream() {
   return getErrorStream();
}
template <>
inline std::wostream &LogStreamTraits<wchar_t, StackTraceType>::stream() {
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

   Log(basic_ostream &s, const char *file = __FILE__, int line = __LINE__) : _stream(s) {
      outputPrefix(file, line);
   }

   ~Log() {
      if (LogType::flushOutputStream()) {
         _stream.flush();
      }
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
    * and fprintf(stderr,format,...) with XXXX().printf(format,...), with XXXX replaced by
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
typedef Log<char, DebugLogType> DebugLog;
typedef Log<char, InfoLogType> InfoLog;
typedef Log<char, WarnLogType> WarnLog;
typedef Log<char, FatalType> Fatal;
typedef Log<char, ErrorLogType> ErrorLog;
typedef Log<char, StackTraceType> StackTrace;

typedef Log<wchar_t, DebugLogType> WDebug;
typedef Log<wchar_t, InfoLogType> WInfo;
typedef Log<wchar_t, WarnLogType> WWarn;
typedef Log<wchar_t, FatalType> WFatal;
typedef Log<char, ErrorLogType> WError;
typedef Log<wchar_t, StackTraceType> WStackTrace;

#else
// Clang __FILE__ and __LINE__ evalaute to the caller of the function
// thus the macros aren't needed
typedef Log<char, DebugLogType> Debug;
typedef Log<char, InfoLogType> Info;
typedef Log<char, WarnLogType> Warn;
typedef Log<char, FatalType> Error;
typedef Log<char, ErrorLogType> ErrorNoExit;
typedef Log<char, StackTraceType> StackTrace;

typedef Log<wchar_t, DebugLogType> WDebug;
typedef Log<wchar_t, InfoLogType> WInfo;
typedef Log<wchar_t, WarnLogType> WWarn;
typedef Log<wchar_t, FatalType> WFatal;
typedef Log<wchar_t, ErrorLogType> WError;
typedef Log<wchar_t, StackTraceType> WStackTrace;
#endif // __GNUC__

// setLogFile sets the file that the DebugLog, InfoLog, WarnLog, and Fatal streams write to.
void setLogFile(std::string const &logFile, std::ios_base::openmode mode = std::ios_base::out);
void setLogFile(char const *logFile, std::ios_base::openmode mode = std::ios_base::out);
void setWLogFile(std::wstring const &logFile, std::ios_base::openmode mode = std::ios_base::out);
void setWLogFile(char const *logFile, std::ios_base::openmode mode = std::ios_base::out);

} // end namespace PV

#endif // PVLOG_HPP_
