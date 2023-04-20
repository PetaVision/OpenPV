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
         std::shared_ptr<FileStream> fileStream)
         : CheckpointEntry(name),
           mFileStream(fileStream) {}
   CheckpointEntryFilePosition(
         std::string const &objName,
         std::string const &dataName,
         std::shared_ptr<FileStream> fileStream)
         : CheckpointEntry(objName, dataName),
           mFileStream(fileStream) {}
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(
         std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   void readValueFromBin(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &label,
      long *value) const;

   void writeValueToBinAndTxt(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &label,
      long value,
      bool verifyWritesFlag) const;

  private:
   std::shared_ptr<FileStream> mFileStream = nullptr;

}; // class CheckpointEntryFilePosition

} // namespace PV

#endif // CHECKPOINTENTRYFILEPOSITION_HPP_
