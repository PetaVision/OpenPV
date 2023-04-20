#ifndef CHECKPOINTENTRYMPIRECVSTREAM_HPP_
#define CHECKPOINTENTRYMPIRECVSTREAM_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "io/MPIRecvStream.hpp"

namespace PV {

class CheckpointEntryMPIRecvStream : public CheckpointEntry {
  public:
   CheckpointEntryMPIRecvStream(
         std::string const &name,
         MPIRecvStream &mpiRecvStream);
   CheckpointEntryMPIRecvStream(
         std::string const &objName,
         std::string const &dataName,
         MPIRecvStream &mpiRecvStream);
   virtual void write(
         std::shared_ptr<FileManager const> fileManager,
         double simTime,
         bool verifyWritesFlag) const override;
   virtual void read(
         std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   MPIRecvStream *mMPIRecvStream = nullptr;
   // mMPIRecvStream is taken from a constructor argument; MPIRecvStream doesn't own it.

}; // class CheckpointEntryMPIRecvStream

} // namespace PV

#endif // CHECKPOINTENTRYMPIRECVSTREAM_HPP_
