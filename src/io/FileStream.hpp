/*
 * FileStream.hpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

#ifndef FILESTREAM_HPP__
#define FILESTREAM_HPP__

#include "PrintStream.hpp"

#include <fstream>

namespace PV {

class FileStream : public PrintStream {
  public:
   FileStream(char const *path);
   FileStream(char const *path, std::ios_base::openmode mode, bool verifyWrites = false);
   virtual ~FileStream();
   void open(std::ios_base::openmode mode, bool verifyWrites);
   void open();
   void write(void const *data, long length);
   void read(void *data, long length);
   void close();
   void setOutPos(long pos, std::ios_base::seekdir seekAnchor);
   void setOutPos(long pos, bool fromBeginning);
   void setInPos(long pos, std::ios_base::seekdir seekAnchor);
   void setInPos(long pos, bool fromBeginning);
   bool readable() const { return mMode & std::ios_base::in; }
   bool writeable() const { return mMode & std::ios_base::out; }
   bool binary() const { return mMode & std::ios_base::binary; }
   bool readwrite() const { return readable() && writeable(); }
   bool isOpen() const { return mFStream.is_open(); }
   operator bool() const { return mFStream ? true : false; }
   long getOutPos();
   long getInPos();
   std::string const &getFileName() const { return mFileName; }

  protected:
   FileStream() {}
   void verifyFlags(const char *caller);

  private:
   void initializePath(char const *path);

  private:
   std::fstream mFStream;
   std::string mFileName;
   std::ios_base::openmode mMode;
   bool mVerifyWrites     = false;

   int const mMaxAttempts = 5;
};

} /* namespace PV */

#endif // FILESTREAM_HPP__
