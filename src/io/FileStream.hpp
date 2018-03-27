/*
 * FileStream.hpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

#ifndef __FILESTREAM_HPP__
#define __FILESTREAM_HPP__

#include "PrintStream.hpp"

#include <fstream>

namespace PV {

class FileStream : public PrintStream {
  public:
   FileStream(char const *path, std::ios_base::openmode mode, bool verifyWrites = false);
   virtual ~FileStream();
   virtual void write(void const *data, long length);
   virtual void read(void *data, long length);
   virtual void setOutPos(long pos, std::ios_base::seekdir seekAnchor);
   virtual void setOutPos(long pos, bool fromBeginning);
   virtual void setInPos(long pos, std::ios_base::seekdir seekAnchor);
   virtual void setInPos(long pos, bool fromBeginning);
   bool readable() { return mMode & std::ios_base::in; }
   bool writeable() { return mMode & std::ios_base::out; }
   bool binary() { return mFStream.flags() & std::ios_base::binary; }
   bool readwrite() { return readable() && writeable(); }
   long getOutPos();
   long getInPos();
   std::string const &getFileName() const { return mFileName; }

  protected:
   FileStream() {}
   void verifyFlags(const char *caller);
   void openFile(char const *path, std::ios_base::openmode mode, bool verifyWrites);

   std::fstream mFStream;
   std::string mFileName;

  private:
   std::ios_base::openmode mMode;
   bool mVerifyWrites     = false;
   int const mMaxAttempts = 5;
};

} /* namespace PV */

#endif
