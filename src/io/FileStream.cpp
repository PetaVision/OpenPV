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
   int attempts    = 0;
   while (!mFStream.is_open()) {
      mFStream.open(fullPath, mode);
      if (!mFStream.fail()) {
         break;
      }
      attempts++;
      pvWarn() << "Failed to open \"" << fullPath << "\" on attempt " << attempts << "\n";
      if (attempts < mMaxAttempts) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (!mFStream.is_open()) {
      pvError() << "FileStream::openFile failure for \"" << fullPath
                << "\": MAX_FILESYSTEMCALL_TRIES = " << mMaxAttempts << " exceeded.\n";
   }
   else if (attempts > 0) {
      pvWarn() << "FileStream::openFile succeeded for \"" << fullPath << "\" on attempt "
               << attempts + 1 << "\n";
   }
   verifyFlags("openFile");
   updateFilePos();
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

// When restoring from checkpoint, the variables mFileReadPos and mFileWritePos
// will be modified. This method checks to see if this has happened, and seeks
// to the correct location in the files.
void FileStream::syncFilePos() {
   if (mFileReadPos != getInPos()) {
      setInPos(mFileReadPos, true);
   }
   if (mFileWritePos != getOutPos()) {
      setOutPos(mFileWritePos, true);
   }
}

void FileStream::updateFilePos() {
   mFileReadPos  = getInPos();
   mFileWritePos = getOutPos();
}

void FileStream::verifyFlags(const char *caller) {
   pvErrorIf(mFStream.fail(), "%s: Logical error.\n", caller);
   pvErrorIf(mFStream.bad(), "%s: Read / Write error.\n", caller);
   pvErrorIf(writeable() && getOutPos() == -1, "%s: out pos == -1\n", caller);
   pvErrorIf(readable() && getInPos() == -1, "%s: in pos == -1\n", caller);
}

void FileStream::write(void *data, long length) {
   syncFilePos();
   long startPos = getOutPos();
   mFStream.write((char *)data, length);
   mFStream.flush();

   verifyFlags("write");
   if (mVerifyWrites) {
      pvErrorIf(mWriteVerifier == nullptr, "Write Verifier is null.\n");
      // Set the input position to the location we wrote to
      mWriteVerifier->setInPos(startPos, true);
      uint8_t check[length];

      // Read from the location we wrote to and compare
      mWriteVerifier->read(check, length);
      if (memcmp(check, data, length) != 0) {
         pvError() << "Verify write failed when writing " << length << " bytes to position "
                   << startPos << "\n";
      }
   }
   updateFilePos();
}

void FileStream::read(void *data, long length) {
   syncFilePos();
   pvErrorIf(mFStream.eof(), "Attempting to read after EOF.\n");
   long startPos = getInPos();
   mFStream.read((char *)data, length);
   long numRead = mFStream.gcount();
   pvErrorIf(
         numRead != length,
         "Expected to read %d bytes at %d, read %d instead.\n"
         "New read position: %d\n",
         length,
         startPos,
         numRead,
         getInPos());
   verifyFlags("read");
   updateFilePos();
}

void FileStream::setOutPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos();
      mFStream.seekp(pos, std::ios_base::cur);
   }
   else {
      mFStream.seekp(pos);
   }
   verifyFlags("setOutPos");
   updateFilePos();
}

void FileStream::setInPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos(); 
      mFStream.seekg(pos, std::ios_base::cur);
   }
   else {
      mFStream.seekg(pos);
   }
   verifyFlags("setInPos");
   updateFilePos();
}

long FileStream::getOutPos() { return mFStream.tellp(); }

long FileStream::getInPos() { return mFStream.tellg(); }

} /* namespace PV */
