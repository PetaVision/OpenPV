#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <libgen.h>
#include <fstream>
#include "utils/PVLog.hpp"
#include "io/io.hpp" // expandLeadingTilde

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
         char * realPath = strdup(expandLeadingTilde(path).c_str());
         if (realPath==nullptr) {
            // Error message has to go to cerr and not pvError() because pvError uses LogFileStream.
            std::cerr << "LogFileStream::setStream failed for \"" << realPath << "\"\n";
            exit(EXIT_FAILURE);
         }
         mStream = new std::basic_ofstream<T>(realPath, mode);
         if (mStream->fail() || mStream->fail()) {
            // Error message has to go to cerr and not pvError() because pvError uses setLogStream.
            std::cerr << "Unable to open logFile \"" << path << "\"." << std::endl;
            exit(EXIT_FAILURE);
         }
         mCreatedWithNew = true;
         free(realPath);
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
   outputLogFileWStream.setStream(logFile, mode);
   if (logFile) {
      errorLogFileWStream.setStream(&getWOutputStream());
   }
   else {
      errorLogFileWStream.setStream(logFile, mode);
   }
}

// vpv_log_debug was deprecated June 16, 2016.  It was only used by functions that themselves are deprecated.
// If we decide we want va_list-based functions, add a vprintf method to _Log.
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

// vpv_log_debug was deprecated June 16, 2016.  It was only used by functions that themselves are deprecated.
// If we decide we want va_list-based functions, add a vprintf method to _Log.
void vpv_log_debug(const char *file, int line, const char *fmt, va_list args) {
   std::ostream& st = getOutputStream();
   vpv_log_with_prefix(st, "DEBUG", file, line, fmt, args);
}

// vpv_log_warn was deprecated June 16, 2016.  It was only used by functions that themselves are deprecated.
// If we decide we want va_list-based functions, add a vprintf method to _Log.
void vpv_log_warn(const char *file, int line, const char *fmt, va_list args) {
   // Flush stdout before printing to stderr. This makes the output
   // a bit cleaner if logging to the console
   getOutputStream().flush();
   std::ostream& st = getErrorStream();
   vpv_log_with_prefix(st, "WARN", NULL, 0, fmt, args);
}

// vpv_log_error was deprecated June 16, 2016.  It was only used by functions that themselves are deprecated.
// If we decide we want va_list-based functions, add a vprintf method to _Log.
void vpv_log_error(const char *file, int line, const char *fmt, va_list args) {
   // Flush stdout before printing to stderr. This makes the output
   // a bit cleaner if logging to the console
   getOutputStream().flush();
   std::ostream& st = getErrorStream();
   vpv_log_with_prefix(st, "ERROR", file, line, fmt, args);
}

// pv_log_debug was deprecated June 16, 2016.  Use pvDebug instead.
void pv_log_debug(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pvLogDebug is deprecated.\n";
   deprecationWarning << "     Use p instead" << std::endl;
   va_list args;
   va_start(args, fmt);
   vpv_log_debug(file, line, fmt, args);
   va_end(args);
}

// pv_log_info was deprecated June 16, 2016.  Use pvInfo instead.
void pv_log_info(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pvLogInfo is deprecated.\n";
   deprecationWarning << "     Use pvInfo instead" << std::endl;
   va_list args;
   va_start(args, fmt);
   std::ostream& st = getOutputStream();
   vpv_log_with_prefix(st, NULL, NULL, 0, fmt, args);
   va_end(args);
}

// pv_log_warn was deprecated June 16, 2016.  Use pvWarn instead.
void pv_log_warn(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pvLogWarn is deprecated.\n";
   deprecationWarning << "     Use pvWarn instead." << std::endl;
   va_list args;
   va_start(args, fmt);
   vpv_log_warn(file, line, fmt, args);
   va_end(args);
}

// pv_log_error was deprecated June 16, 2016.  Use pvError instead.
void pv_log_error(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pvLogError is deprecated.\n";
   deprecationWarning << "     Use pvError to print to the error stream and exit.\n";
   deprecationWarning << "     Use pvWarn or pvErrorNoExit to print to the error stream without exiting." << std::endl;
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
   exit(EXIT_FAILURE);
}

// pv_log_error_noexit was deprecated June 16, 2016.  Use pvErrorNoExit instead.
void pv_log_error_noexit(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pv_log_error_noexit is deprecated.\n";
   deprecationWarning << "     Use pvError to print to the error stream and exit.\n";
   deprecationWarning << "     Use pvWarn or pvErrorNoExit to print to the error stream without exiting." << std::endl;
   va_list args;
   va_start(args, fmt);
   vpv_log_error(file, line, fmt, args);
   va_end(args);
}

// pv_exit_failure was deprecated May 25, 2016.  Use pvError instead.
void pv_exit_failure(const char *file, int line, const char *fmt, ...) {
   pvWarn(deprecationWarning);
   deprecationWarning << "pvExitFailure is deprecated.\n";
   deprecationWarning << "     Use pvError to print to the error stream and exit.\n";
   deprecationWarning << "     Use pvWarn or pvErrorNoExit to print to the error stream without exiting." << std::endl;
   va_list args;
   vpv_log_error(file, line, fmt, args);
   exit(EXIT_FAILURE);
}

}  // end namespace PV
