/*
 * FileStream.cpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

extern "C" {
#include <unistd.h>
}
#include <cstring>
#include "FileStream.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

OutStream::OutStream(std::ostream& stream) {
   setOutStream(stream);
}

void OutStream::setOutStream(std::ostream& stream) {
   mOutStream = &stream;
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
   if (!path) { throw; }
   int attempts = 0;
   while (mStrPtr == nullptr) {
      mStrPtr = new std::fstream(path, mode);
      if (!mStrPtr->fail()) { break; }
      delete mStrPtr; mStrPtr = nullptr;
      attempts++;
      pvWarn() << "Failed to open \"" << path << "\" on attempt " << attempts << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (mStrPtr==nullptr) {
      pvError() << "FileStream::openFile failure for \"" << path << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts>0) {
      pvWarn() << "FileStream::openFile succeeded for \"" << path << "\" on attempt " << attempts+1 << "\n";
   }
   mPath = strdup(path);
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
