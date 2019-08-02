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
#include "utils/ExpandLeadingTilde.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

using std::string;

namespace PV {

FileStream::FileStream(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   initialize(path, mode, verifyWrites);
}

void FileStream::initialize(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   PrintStream::initialize(mFStream);
   openFile(path, mode, verifyWrites);
}

FileStream::~FileStream() {}

void FileStream::openFile(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   string fullPath = expandLeadingTilde(path);
   mFileName       = fullPath;
   mMode           = mode;
   mVerifyWrites   = verifyWrites;
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
      Fatal() << "FileStream::openFile failure (" << strerror(errno) << ") for \"" << fullPath
              << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts > 0) {
      WarnLog() << "FileStream::openFile succeeded for \"" << fullPath << "\" on attempt "
                << attempts + 1 << "\n";
   }
   verifyFlags("openFile");
}

void FileStream::verifyFlags(const char *caller) {
   FatalIf(mFStream.fail(), "%s %s: Logical error.\n", mFileName.c_str(), caller);
   FatalIf(mFStream.bad(), "%s %s: Read / Write error.\n", mFileName.c_str(), caller);
   FatalIf(writeable() && getOutPos() == -1, "%s %s: out pos == -1\n", mFileName.c_str(), caller);
   FatalIf(readable() && getInPos() == -1, "%s %s: in pos == -1\n", mFileName.c_str(), caller);
}

void FileStream::write(void const *data, long length) {
   long startPos = getOutPos();
   mFStream.write((char *)data, length);
   mFStream.flush();

   std::string errmsg;
   errmsg.append("writing ").append(std::to_string(length)).append(" bytes");
   verifyFlags(errmsg.c_str());
   if (mVerifyWrites) {
      std::ios_base::openmode mode = std::ios_base::in;
      if (binary()) {
         mode |= std::ios_base::binary;
      }
      FileStream writeVerifier(mFileName.c_str(), mode, false);
      // Set the input position to the location we wrote to
      writeVerifier.setInPos(startPos, true);
      std::vector<uint8_t> check(length);

      // Read from the location we wrote to and compare
      writeVerifier.read(check.data(), length);
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
         "Expected to read %d bytes from %s at position %d; read %d instead. "
         "New read position: %d\n",
         length,
         mFileName.c_str(),
         startPos,
         numRead,
         getInPos());
   verifyFlags("read");
}

void FileStream::setOutPos(long pos, std::ios_base::seekdir seekAnchor) {
   mFStream.seekp(pos, seekAnchor);
   verifyFlags("setOutPos");
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

void FileStream::setInPos(long pos, std::ios_base::seekdir seekAnchor) {
   mFStream.seekg(pos, seekAnchor);
   verifyFlags("setInPos");
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
