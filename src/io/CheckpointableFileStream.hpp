#ifndef __CHECKPOINTABLEFILESTREAM_HPP__
#define __CHECKPOINTABLEFILESTREAM_HPP__

#include "FileStream.hpp"
#include "Secretary.hpp"

#include <string>

using std::string;

namespace PV {

class CheckpointableFileStream
      : public FileStream,
        public SecretaryDataInterface {

   public:
      CheckpointableFileStream(char const *path,
                  std::ios_base::openmode mode,
                  string objName,
                  bool verifyWrites = false);
      virtual void write(void *data, long length);
      virtual void read(void *data, long length);
      virtual void setOutPos(long pos, bool fromBeginning);
      virtual void setInPos(long pos, bool fromBeginning);
      virtual int registerData(Secretary *secretary, const string objName);
   private:
      void syncFilePos();
      void updateFilePos();
      long mFileReadPos  = 0;
      long mFileWritePos = 0;
      string mObjName; // Used for SecretaryDataInterface
};

}

#endif
