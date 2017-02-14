/*
 * FileStream.cpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

extern "C" {
#include <unistd.h>
}
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "FileStream.hpp"
#include "io/io.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

using std::string;

namespace PV {

FileStream::FileStream(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   setOutStream(mFStream);
   openFile(path, mode, verifyWrites);
}

FileStream::~FileStream() {
   if (mWriteVerifier != nullptr) {
      delete mWriteVerifier;
   }
}

void FileStream::openFile(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   string fullPath = expandLeadingTilde(path);
   mFileName       = fullPath;
   int attempts    = 0;
   while (!mFStream.is_open()) {
      mFStream.open(fullPath, mode);
      if (!mFStream.fail()) {
         break;
      }
      attempts++;
      WarnLog() << "Failed to open \"" << fullPath << "\" on attempt " << attempts << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (!mFStream.is_open()) {
      Fatal() << "FileStream::openFile failure for \"" << fullPath
              << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts > 0) {
      WarnLog() << "FileStream::openFile succeeded for \"" << fullPath << "\" on attempt "
                << attempts + 1 << "\n";
   }
   verifyFlags("openFile");
   if (verifyWrites) {
      mVerifyWrites = true;
      if (binary()) {
         mWriteVerifier = new FileStream(path, std::ios_base::in | std::ios_base::binary, false);
      }
      else {
         mWriteVerifier = new FileStream(path, std::ios_base::in, false);
      }
   }
}

void FileStream::verifyFlags(const char *caller) {
   FatalIf(mFStream.fail(), "%s: Logical error.\n", caller);
   FatalIf(mFStream.bad(), "%s: Read / Write error.\n", caller);
   FatalIf(writeable() && getOutPos() == -1, "%s: out pos == -1\n", caller);
   FatalIf(readable() && getInPos() == -1, "%s: in pos == -1\n", caller);
}

void FileStream::write(void const *data, long length) {
   long startPos = getOutPos();
   mFStream.write((char *)data, length);
   mFStream.flush();

   verifyFlags("write");
   if (mVerifyWrites) {
      FatalIf(mWriteVerifier == nullptr, "Write Verifier is null.\n");
      // Set the input position to the location we wrote to
      mWriteVerifier->setInPos(startPos, true);
      std::vector<uint8_t> check(length);

      // Read from the location we wrote to and compare
      mWriteVerifier->read(check.data(), length);
      if (memcmp(check.data(), data, length) != 0) {
         Fatal() << "Verify write failed when writing " << length << " bytes to position "
                 << startPos << "\n"
                 << mFileName << "\n";
      }
   }
}

void FileStream::read(void *data, long length) {
   FatalIf(mFStream.eof(), "Attempting to read after EOF.\n");
   long startPos = getInPos();
   mFStream.read((char *)data, length);
   long numRead = mFStream.gcount();
   FatalIf(
         numRead != length,
         "Expected to read %d bytes at %d, read %d instead.\n"
         "New read position: %d\n%s\n",
         length,
         startPos,
         numRead,
         getInPos(),
         mFileName.c_str());
   verifyFlags("read");
}

void FileStream::setOutPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      mFStream.seekp(pos, std::ios_base::cur);
   }
   else {
      mFStream.seekp(pos);
   }
   verifyFlags("setOutPos");
}

void FileStream::setInPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      mFStream.seekg(pos, std::ios_base::cur);
   }
   else {
      mFStream.seekg(pos);
   }
   verifyFlags("setInPos");
}

long FileStream::getOutPos() { return mFStream.tellp(); }

long FileStream::getInPos() { return mFStream.tellg(); }

} /* namespace PV */
