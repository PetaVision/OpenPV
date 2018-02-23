#ifndef __CHECKPOINTABLEFILESTREAM_HPP__
#define __CHECKPOINTABLEFILESTREAM_HPP__

#include "CheckpointerDataInterface.hpp"
#include "io/FileStream.hpp"

#include <string>

using std::string;

namespace PV {

class CheckpointableFileStream : public FileStream, public CheckpointerDataInterface {

  public:
   /**
    * Constructor for CheckpointableFileStream. Opens a file for reading and
    * writing at the path indicated, and registers its file positions with the
    * given checkpointer. The path must be a relative path; it is
    * relative to the checkpointer's OutputPath directory.
    * If newFile is true, the file is created (clobbering the file if it
    * already exists). If newFile is false and the file does not exist, a
    * warning is issued and the file is created.
    * A CheckpointableFileStream can only be instantiated by the root process
    * of the Checkpointer's MPIBlock; all other processes generate a fatal
    * error. objName is the object name used when registering the file
    * positions with the checkpointer.
    * verifyWrites has the same meaning as in the FileStream constructor.
    */
   CheckpointableFileStream(
         string const &path,
         bool newFile,
         Checkpointer *checkpointer,
         string const &objName,
         bool verifyWrites);

   /**
    * This constructor is identical to the previous constructor, except that
    * the checkpointer's verifyWrites flag is used in place of an explicit
    * argument.
    */
   CheckpointableFileStream(
         string const &path,
         bool newFile,
         Checkpointer *checkpointer,
         string const &objName);
   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;
   virtual void write(void const *data, long length) override;
   virtual void read(void *data, long length) override;
   virtual void setOutPos(long pos, bool fromBeginning) override;
   virtual void setInPos(long pos, bool fromBeginning) override;

  private:
   void initialize(
         string const &path,
         bool newFile,
         Checkpointer *checkpointer,
         string const &objName,
         bool verifyWrites);
   void setDescription();
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   Response::Status respondProcessCheckpointRead(ProcessCheckpointReadMessage const *message);
   void syncFilePos();
   void updateFilePos();
   long mFileReadPos  = 0;
   long mFileWritePos = 0;
   string mObjName; // Used for CheckpointerDataInterface
};
}

#endif
