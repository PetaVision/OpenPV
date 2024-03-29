/*
 * FileStream.cpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

extern "C" {
#include <unistd.h>
}
#include <cerrno>
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

FileStream::FileStream(char const *path) {
   initializePath(path);
}

FileStream::FileStream(char const *path, std::ios_base::openmode mode, bool verifyWrites) {
   initializePath(path);
   open(mode, verifyWrites);
}

FileStream::~FileStream() {}

void FileStream::initializePath(char const *path) {
   mFileName = expandLeadingTilde(path);
}

void FileStream::open() {
   int attempts  = 0;
   while (!mFStream.is_open()) {
      mFStream.open(mFileName, mMode);
      if (!mFStream.fail()) {
         break;
      }
      attempts++;
      WarnLog() << "Failed to open \"" << mFileName << "\" on attempt " << attempts << ": "
                << strerror(errno) << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (!mFStream.is_open()) {
      Fatal() << "FileStream::open failure (" << strerror(errno) << ") for \"" << mFileName
              << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts > 0) {
      WarnLog() << "FileStream::open succeeded for \"" << mFileName << "\" on attempt "
                << attempts + 1 << "\n";
   }
   verifyFlags("open");
   if (writeable()) { PrintStream::initialize(mFStream); }
}

void FileStream::open(std::ios_base::openmode mode, bool verifyWrites) {
   if (mFStream.is_open() and (mode != mMode or verifyWrites != mVerifyWrites)) {
      mFStream.close();
   }
   mMode         = mode;
   mVerifyWrites = verifyWrites;
   open();
}

void FileStream::verifyFlags(const char *caller) {
   bool failed = false;
   std::string errMsg(mFileName + " " + caller);
   if (mFStream.fail()) { errMsg.append(", failbit set"); failed = true; }
   if (mFStream.bad()) { errMsg.append(", badbit set"); failed = true; }
   if (mFStream.eof()) { errMsg.append(", eofbit set"); failed = true; }
   if (writeable() and getOutPos() == -1) { errMsg.append(", out pos == -1"); failed = true; }
   if (readable() and getInPos() == -1) { errMsg.append(", in pos == -1"); failed = true; }
   FatalIf(failed, "%s\n", errMsg.c_str());
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

void FileStream::close() {
   if (!isOpen()) { return; }
   mFStream.close();
   FatalIf(
         mFStream.fail(),
         "FileStream failed to close \"%s\": %s\n",
         mFileName.c_str(), strerror(errno));
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
