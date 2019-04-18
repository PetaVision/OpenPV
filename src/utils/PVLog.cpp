#include "utils/PVLog.hpp"
#include "io/io.hpp" // expandLeadingTilde
#include <fstream>
#include <libgen.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

template <typename T>
class LogFileStream {
  public:
   LogFileStream(std::basic_ostream<T> &st) : mDefaultStream(st) { setStreamDefault(); }
   virtual ~LogFileStream() {
      if (mCreatedWithNew) {
         delete mStream;
      }
   }

   std::basic_ostream<T> &getStream() { return *mStream; }
   void setStream(char const *path, std::ios_base::openmode mode = std::ios_base::out) {
      if (mCreatedWithNew) {
         delete mStream;
      }
      if (path) {
         char *realPath = strdup(expandLeadingTilde(path).c_str());
         if (realPath == nullptr) {
            // Error message has to go to cerr and not Fatal() because Fatal uses LogFileStream.
            std::cerr << "LogFileStream::setStream failed for \"" << realPath << "\"\n";
            exit(EXIT_FAILURE);
         }
         mStream = new std::basic_ofstream<T>(realPath, mode);
         if (mStream->fail() || mStream->fail()) {
            // Error message has to go to cerr and not Fatal() because Fatal uses setLogStream.
            std::cerr << "Unable to open logFile \"" << path << "\"." << std::endl;
            exit(EXIT_FAILURE);
         }
         mCreatedWithNew = true;
         free(realPath);
      }
      else {
         mStream         = &mDefaultStream;
         mCreatedWithNew = false;
      }
   }
   void setStream(std::basic_ostream<T> *stream) {
      if (mCreatedWithNew) {
         delete mStream;
      }
      mStream         = stream;
      mCreatedWithNew = false;
   }
   void setStreamDefault() {
      if (mCreatedWithNew) {
         delete mStream;
      }
      mStream         = &mDefaultStream;
      mCreatedWithNew = false;
   }

  private:
   std::basic_ostream<T> &mDefaultStream;
   std::basic_ostream<T> *mStream = nullptr;
   bool mCreatedWithNew           = false;
};

LogFileStream<char> errorLogFileStream(std::cerr);
LogFileStream<char> outputLogFileStream(std::cout);
LogFileStream<wchar_t> errorLogFileWStream(std::wcerr);
LogFileStream<wchar_t> outputLogFileWStream(std::wcout);

std::ostream &getErrorStream() { return errorLogFileStream.getStream(); }
std::ostream &getOutputStream() { return outputLogFileStream.getStream(); }

std::wostream &getWErrorStream() { return errorLogFileWStream.getStream(); }
std::wostream &getWOutputStream() { return outputLogFileWStream.getStream(); }

void setLogFile(std::string const &logFile, std::ios_base::openmode mode) {
   if (!logFile.empty()) {
      setLogFile(logFile.c_str(), mode);
   }
}

void setLogFile(char const *logFile, std::ios_base::openmode mode) {
   outputLogFileStream.setStream(logFile, mode);
   if (logFile) {
      errorLogFileStream.setStream(&getOutputStream());
   }
   else {
      errorLogFileStream.setStream(logFile, mode);
   }
}

void setWLogFile(std::wstring const &logFile, std::ios_base::openmode mode) {
   if (!logFile.empty()) {
      setWLogFile(logFile.c_str(), mode);
   }
}

void setWLogFile(char const *logFile, std::ios_base::openmode mode) {
   outputLogFileWStream.setStream(logFile, mode);
   if (logFile) {
      errorLogFileWStream.setStream(&getWOutputStream());
   }
   else {
      errorLogFileWStream.setStream(logFile, mode);
   }
}

} // end namespace PV
