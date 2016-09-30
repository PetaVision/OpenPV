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
      FileStream(char const * path,
                 std::ios_base::openmode mode,
                 bool verifyWrites = false);
      ~FileStream();
      bool readable()  { return mFStream.flags() & std::ios_base::in; }
      bool writeable() { return mFStream.flags() & std::ios_base::out; }
      bool binary()    { return mFStream.flags() & std::ios_base::binary; }
      bool readwrite() { return readable() && writeable(); }
      void write(void *data, long length);
      void read(void *data, long length);
      void setOutPos(long pos, bool fromBeginning);
      void setInPos(long pos, bool fromBeginning);
      long getOutPos();
      long getInPos();
   protected:
      FileStream() {}
      void verifyFlags(const char *caller);

   private:
      void openFile(char const *path, std::ios_base::openmode mode, bool verifyWrites);
      void closeFile();

   private:
      std::fstream mFStream;
      bool mVerifyWrites = false;
      int const mMaxAttempts = 5;
      FileStream *mWriteVerifier = nullptr;
};

} /* namespace PV */

#endif
