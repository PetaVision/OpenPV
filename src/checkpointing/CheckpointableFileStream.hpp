#ifndef __CHECKPOINTABLEFILESTREAM_HPP__
#define __CHECKPOINTABLEFILESTREAM_HPP__

#include "Checkpointer.hpp"
#include "io/FileStream.hpp"

#include <string>

using std::string;

namespace PV {

class CheckpointableFileStream : public FileStream,
                                 public Observer,
                                 public CheckpointerDataInterface {

  public:
   CheckpointableFileStream(
         char const *path,
         std::ios_base::openmode mode,
         string objName,
         bool verifyWrites = false);
   virtual int respond(std::shared_ptr<BaseMessage const> message) override;
   virtual void write(void const *data, long length);
   virtual void read(void *data, long length);
   virtual void setOutPos(long pos, bool fromBeginning);
   virtual void setInPos(long pos, bool fromBeginning);
   virtual int registerData(Checkpointer *checkpointer, const string objName);

  private:
   void setDescription();
   int respondProcessCheckpointRead(ProcessCheckpointReadMessage const *message);
   void syncFilePos();
   void updateFilePos();
   long mFileReadPos  = 0;
   long mFileWritePos = 0;
   string mObjName; // Used for CheckpointerDataInterface
};
}

#endif
