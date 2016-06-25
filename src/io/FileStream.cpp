/*
 * FileStream.cpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

extern "C" {
#include <unistd.h>
}
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include "FileStream.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "io/io.hpp"

namespace PV {

OutStream::OutStream(std::ostream& stream) {
   setOutStream(stream);
}

void OutStream::setOutStream(std::ostream& stream) {
   mOutStream = &stream;
}

int OutStream::printf(const char *fmt, ...) {
   va_list args1, args2;
   va_start(args1, fmt);
   va_copy(args2, args1);
   char c;
   int chars_needed = vsnprintf(&c, 1, fmt, args1);
   chars_needed++;
   char output_string[chars_needed];
   int chars_printed = vsnprintf(output_string, chars_needed, fmt, args2);
   pvAssert(chars_printed+1==chars_needed);
   this->outStream() << output_string;
   va_end(args1);
   va_end(args2);
   return chars_needed;
}


FileStream::FileStream(char const * path, std::ios_base::openmode mode, bool verifyWrites) {
   openFile(path, mode);
   mVerifyWrites = verifyWrites;
}

FileStream::~FileStream() {
   closeFile();
   free(mPath);
   if (mStrPtr) {
      mStrPtr->close();
      delete mStrPtr;
   }
}

void FileStream::openFile(char const * path, std::ios_base::openmode mode) {
   pvAssert(mStrPtr==nullptr);
   mPath = expandLeadingTilde(path);
   if (!mPath) { throw; }
   int attempts = 0;
   while (mStrPtr == nullptr) {
      mStrPtr = new std::fstream(mPath, mode);
      if (!mStrPtr->fail()) { break; }
      delete mStrPtr; mStrPtr = nullptr;
      attempts++;
      pvWarn() << "Failed to open \"" << mPath << "\" on attempt " << attempts << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (mStrPtr==nullptr) {
      pvError() << "FileStream::openFile failure for \"" << mPath << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts>0) {
      pvWarn() << "FileStream::openFile succeeded for \"" << mPath << "\" on attempt " << attempts+1 << "\n";
   }
   if (!mPath) { throw; }
   mMode = mode;
   std::streambuf::pos_type seekoffFailure = (std::streambuf::pos_type) -1;
   mFilePos = mStrPtr->rdbuf()->pubseekoff(0, std::ios_base::cur);
   if (mFilePos==seekoffFailure) { throw; }
   mFileLength = mStrPtr->rdbuf()->pubseekoff(0, std::ios_base::end);
   if (mFilePos==seekoffFailure) { throw; }
   std::streambuf::pos_type startingPos = mStrPtr->rdbuf()->pubseekpos(mFilePos);
   if (startingPos != mFilePos) { throw; }
   setOutStream(*mStrPtr);
}

void FileStream::closeFile() {
   pvAssert(mStrPtr!=nullptr);
   mStrPtr->close();
   if (mStrPtr->fail()) {
      pvError() << "FileStream::closeFile failure for \"" << mPath << "\"";
   }
}

} /* namespace PV */
