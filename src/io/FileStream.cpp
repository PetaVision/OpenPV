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
#include <string>
#include <cinttypes>


#include "FileStream.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "io/io.hpp"

using std::string;

namespace PV {

FileStream::FileStream(char const * path, std::ios_base::openmode mode, bool verifyWrites) {
   openFile(path, mode);
   setOutStream(mFStream);
   mVerifyWrites = verifyWrites;
}

void FileStream::openFile(char const * path, std::ios_base::openmode mode) {
   string fullPath = expandLeadingTilde(path);
   int attempts = 0;
   while (!mFStream.is_open()) {
      mFStream.open(fullPath, mode);
      if (!mFStream.fail()) {
         break;
      }
      attempts++;
      pvWarn() << "Failed to open \"" << fullPath
               << "\" on attempt " << attempts << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (!mFStream.is_open()) {
      pvError() << "FileStream::openFile failure for \"" << fullPath
                << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts
                << " exceeded.\n";
   }
   else if (attempts > 0) {
      pvWarn() << "FileStream::openFile succeeded for \"" << fullPath
               << "\" on attempt " << attempts+1 << "\n";
   }
   verifyFlags("openFile");
}

void FileStream::verifyFlags(const char *caller) {
   pvErrorIf(mFStream.fail(), "%s: Logical error.\n", caller);
   pvErrorIf(mFStream.bad(), "%s: Read / Write error.\n", caller);
   pvErrorIf(writeable() && getOutPos() == -1,
         "%s: out pos == -1\n", caller);
   pvErrorIf(readable() && getInPos() == -1,
         "%s: in pos == -1\n", caller);
}

void FileStream::write(void *data, long length) {
   mFStream.write((char*)data, length);
   // TODO: Verify writes
   mFStream.flush();
   verifyFlags("write");
}

void FileStream::read(void *data, long length) {
   pvErrorIf(mFStream.eof(),
         "Attempting to read after EOF.\n");
   mFStream.read((char*)data, length);
   long numRead = mFStream.gcount();
   pvErrorIf(numRead != length,
         "Expected to read %d  bytes, read %d instead.\n"
         "Read position: %d\n",
         length, numRead, getInPos());
   verifyFlags("read");
}

void FileStream::setOutPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      mFStream.seekp(pos, std::ios_base::cur);
   } else {
      mFStream.seekp(pos);
   }
   verifyFlags("setOutPos");
}

void FileStream::setInPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      mFStream.seekg(pos, std::ios_base::cur);
   } else {
      mFStream.seekg(pos);
   }
   verifyFlags("setInPos");
}

long FileStream::getOutPos() {
   return mFStream.tellp();
}

long FileStream::getInPos() {
   return mFStream.tellg();
}

} /* namespace PV */
