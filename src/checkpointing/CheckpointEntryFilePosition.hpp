#ifndef CHECKPOINTENTRYFILEPOSITION_HPP_
#define CHECKPOINTENTRYFILEPOSITION_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "io/FileStream.hpp"
#include <memory>

namespace PV {

class CheckpointEntryFilePosition : public CheckpointEntry {
  public:
   CheckpointEntryFilePosition(
         std::string const &name,
         std::shared_ptr<MPIBlock const> mpiBlock,
         std::shared_ptr<FileStream> fileStream)
         : CheckpointEntry(name, mpiBlock),
           mFileStream(fileStream) {}
   CheckpointEntryFilePosition(
         std::string const &objName,
         std::string const &dataName,
         std::shared_ptr<MPIBlock const> mpiBlock,
         std::shared_ptr<FileStream> fileStream)
         : CheckpointEntry(objName, dataName, mpiBlock),
           mFileStream(fileStream) {}
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   void readValueFromBin(
      std::string const &checkpointDirectory,
      std::string const &label,
      long *value) const;

   void writeValueToBinAndTxt(
      std::string const &checkpointDirectory,
      std::string const &label,
      long value,
      bool verifyWritesFlag) const;

  private:
   std::shared_ptr<FileStream> mFileStream = nullptr;

}; // class CheckpointEntryFilePosition

} // namespace PV

#endif // CHECKPOINTENTRYFILEPOSITION_HPP_
