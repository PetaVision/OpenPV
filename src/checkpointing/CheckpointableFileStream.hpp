#ifndef CHECKPOINTABLEFILESTREAM_HPP__
#define CHECKPOINTABLEFILESTREAM_HPP__

#include "CheckpointerDataInterface.hpp"
#include "io/FileManager.hpp"
#include "io/FileStream.hpp"

#include <memory>
#include <string>

using std::string;

namespace PV {

class CheckpointableFileStream : public FileStream, public CheckpointerDataInterface {

  public:
   /**
    * Constructor for CheckpointableFileStream. Opens a file for reading and
    * writing at the path indicated, and registers its file positions with the
    * checkpointer during the RegisterData stage. The path must be a relative path; it is
    * relative to the checkpointer's OutputPath directory.
    * If newFile is true, the file is created (clobbering the file if it
    * already exists). If newFile is false and the file does not exist, a
    * warning is issued and the file is created.
    * A CheckpointableFileStream can only be instantiated by the root process
    * of the FileManager's MPIBlock; all other processes generate a fatal
    * error. objName is the object name used when registering the file
    * positions with the checkpointer.
    * verifyWrites has the same meaning as in the FileStream constructor.
    */
   CheckpointableFileStream(
         string const &path,
         bool newFile,
         std::shared_ptr<FileManager const> fileManager,
         string const &objName,
         bool verifyWrites);

   virtual int printf(const char *fmt, ...) override;
   virtual PrintStream &operator<<(std::string &s) override;
   virtual PrintStream &operator<<(char c) override;
   virtual PrintStream &operator<<(signed char c) override;
   virtual PrintStream &operator<<(unsigned char c) override;
   virtual PrintStream &operator<<(const char *c) override;
   virtual PrintStream &operator<<(const signed char *c) override;
   virtual PrintStream &operator<<(const unsigned char *c) override;
   virtual PrintStream &operator<<(short x) override;
   virtual PrintStream &operator<<(unsigned short x) override;
   virtual PrintStream &operator<<(int x) override;
   virtual PrintStream &operator<<(unsigned int x) override;
   virtual PrintStream &operator<<(long x) override;
   virtual PrintStream &operator<<(unsigned long x) override;
   virtual PrintStream &operator<<(long long x) override;
   virtual PrintStream &operator<<(unsigned long long x) override;
   virtual PrintStream &operator<<(float x) override;
   virtual PrintStream &operator<<(double x) override;
   virtual PrintStream &operator<<(long double x) override;
   virtual PrintStream &operator<<(bool x) override;
   virtual PrintStream &operator<<(void const *x) override;
   virtual PrintStream &operator<<(std::streambuf *x) override;
   virtual PrintStream &operator<<(std::ostream &(*f)(std::ostream &)) override;
   virtual PrintStream &operator<<(std::ostream &(*f)(std::ios &)) override;
   virtual PrintStream &operator<<(std::ostream &(*f)(std::ios_base &)) override;
   virtual void write(void const *data, long length) override;
   virtual void read(void *data, long length) override;
   virtual void setOutPos(long pos, std::ios_base::seekdir seekAnchor) override;
   virtual void setOutPos(long pos, bool fromBeginning) override;
   virtual void setInPos(long pos, std::ios_base::seekdir seekAnchor) override;
   virtual void setInPos(long pos, bool fromBeginning) override;

  private:
   void initialize(
         string const &path,
         bool newFile,
         std::shared_ptr<FileManager const> fileManager,
         string const &objName,
         bool verifyWrites);
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status processCheckpointRead() override;
   void syncFilePos();
   void updateFilePos();
   long mFileReadPos  = 0;
   long mFileWritePos = 0;
   string mObjName; // Used for CheckpointerDataInterface
};
}

#endif // CHECKPOINTABLEFILESTREAM_HPP_
