#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <libgen.h>
#include <fstream>
#include "utils/PVLog.hpp"

namespace PV {

template <typename T>
class LogFileStream {
public:
   LogFileStream(std::basic_ostream<T>& st) : mDefaultStream(st) { setStreamDefault(); }
   virtual ~LogFileStream() {if (mCreatedWithNew) {delete mStream;} }

   std::basic_ostream<T>& getStream() { return *mStream; }
   void setStream(char const * path, std::ios_base::openmode mode=std::ios_base::out) {
      if (mCreatedWithNew) {
         delete mStream;
      }
      if (path) {
         mStream = new std::basic_ofstream<T>(path, mode);
         if (mStream->fail() || mStream->fail()) {
            std::cerr << "Unable to open logFile \"" << path << "\"." << std::endl;
            exit(EXIT_FAILURE);
         }
         mCreatedWithNew = true;
      }
      else {
         mStream = &mDefaultStream;
         mCreatedWithNew = false;
      }
   }
   void setStream(std::basic_ostream<T> * stream) {
      if (mCreatedWithNew) {
         delete mStream;
      }
      mStream = stream;
      mCreatedWithNew = false;
   }
   void setStreamDefault() {
      if (mCreatedWithNew) {
         delete mStream;
      }
      mStream = &mDefaultStream;
      mCreatedWithNew = false;
   }

private:
   std::basic_ostream<T>& mDefaultStream;
   std::basic_ostream<T> * mStream = nullptr;
   bool mCreatedWithNew = false;
};

LogFileStream<char> errorLogFileStream(std::cerr);
LogFileStream<char> outputLogFileStream(std::cout);
LogFileStream<wchar_t> errorLogFileWStream(std::wcerr);
LogFileStream<wchar_t> outputLogFileWStream(std::wcout);

std::ostream& getErrorStream() { return errorLogFileStream.getStream(); }
std::ostream& getOutputStream() { return outputLogFileStream.getStream(); }

std::wostream& getWErrorStream() { return errorLogFileWStream.getStream(); }
std::wostream& getWOutputStream() { return outputLogFileWStream.getStream(); }

void setLogFile(char const * logFile, std::ios_base::openmode mode) {
   outputLogFileStream.setStream(logFile, mode);
   if (logFile) {
      errorLogFileStream.setStream(&getOutputStream());
   }
   else {
      errorLogFileStream.setStream(logFile, mode);
   }
}

void setWLogFile(char const * logFile, std::ios_base::openmode mode) {
   errorLogFileWStream.setStream(logFile, mode);
   outputLogFileWStream.setStream(logFile, mode);
}

// TODO: make use of the LogType classes to decide whether to print prefix and file, and what the separators are.
void vpv_log_with_prefix(std::ostream& stream, const char *prefix, const char *file, int line, const char *fmt, va_list args) {
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   if (prefix) {
      stream << prefix << " ";
   }
   if (file) {
      stream << "<" << basename((char*)file) << ":" << line << ">: ";
   }
   stream << msg;
}

void vpv_log_debug(const char *file, int line, const char *fmt, va_list args) {
   std::ostream& st = getOutputStream();
   vpv_log_with_prefix(st, "DEBUG", file, line, fmt, args);
}

void vpv_log_warn(const char *file, int line, const char *fmt, va_list args) {
   // Flush stdout before printing to stderr. This makes the output
   // a bit cleaner if logging to the console
   getOutputStream().flush();
   std::ostream& st = getErrorStream();
   vpv_log_with_prefix(st, "WARN", NULL, 0, fmt, args);
}

void vpv_log_error(const char *file, int line, const char *fmt, va_list args) {
   // Flush stdout before printing to stderr. This makes the output
   // a bit cleaner if logging to the console
   getOutputStream().flush();
   std::ostream& st = getErrorStream();
   vpv_log_with_prefix(st, "ERROR", file, line, fmt, args);
}

void pv_log_debug(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_debug(file, line, fmt, args);
   va_end(args);
}

void pv_log_info(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   std::ostream& st = getOutputStream();
   vpv_log_with_prefix(st, NULL, NULL, 0, fmt, args);
   va_end(args);
}

void pv_log_warn(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_warn(file, line, fmt, args);
   va_end(args);
}

void pv_log_error(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
   exit(EXIT_FAILURE);
}

void pv_log_error_noexit(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
}

void pv_log_abort(const char *file, int line, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
   abort();
}

// pv_exit_failure was deprecated May 25, 2016.  Use pvLogError (or, preferably, pvError) instead.
void pv_exit_failure(const char *file, int line, const char *fmt, ...) {
   pvWarn(exitFailureDeprecated);
   exitFailureDeprecated << "pvExitFailure is deprecated.\n";
   exitFailureDeprecated << "     Use pvError to print to the error stream and exit.\n";
   exitFailureDeprecated << "     Use pvWarn or pvErrorNoExit to print to the error stream without exiting." << std::endl;
   va_list args;
   vpv_log_error(file, line, fmt, args);
   exit(EXIT_FAILURE);
}

}  // end namespace PV
