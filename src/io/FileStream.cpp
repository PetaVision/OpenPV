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
   verifyFlags();
}

void FileStream::verifyFlags() {
   pvErrorIf(mFStream.fail(), "fstream: Logical error.\n");
   pvErrorIf(mFStream.bad(), "fstream: Read / Write error.\n");
}

void FileStream::write(void *data, long length) {
   mFStream.write((char*)data, length);
   // TODO: Verify writes
   mFStream.flush();
   verifyFlags();
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
   verifyFlags();
}

void FileStream::setOutPos(long pos, bool fromBeginning) {
   mFStream.seekp(pos, fromBeginning
                     ? std::ios_base::beg
                     : std::ios_base::cur);
   verifyFlags();
}

void FileStream::setInPos(long pos, bool fromBeginning) {
   mFStream.seekg(pos, fromBeginning
                     ? std::ios_base::beg
                     : std::ios_base::cur);
   verifyFlags();
}

long FileStream::getOutPos() {
   return mFStream.tellp();
}

long FileStream::getInPos() {
   return mFStream.tellg();
}

} /* namespace PV */
