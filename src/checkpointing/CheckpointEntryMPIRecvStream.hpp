#ifndef CHECKPOINTENTRYMPIRECVSTREAM_HPP_
#define CHECKPOINTENTRYMPIRECVSTREAM_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "io/MPIRecvStream.hpp"

namespace PV {

class CheckpointEntryMPIRecvStream : public CheckpointEntry {
  public:
   CheckpointEntryMPIRecvStream(
         std::string const &name,
         MPIBlock const *mpiBlock,
         MPIRecvStream &mpiRecvStream);
   CheckpointEntryMPIRecvStream(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         MPIRecvStream &mpiRecvStream);
   virtual void write(
         std::string const &checkpointDirectory,
         double simTime,
         bool verifyWritesFlag) const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   void unlinkFile(std::string const &path) const;

  private:
   MPIRecvStream *mMPIRecvStream = nullptr;
   // mMPIRecvStream is taken from a constructor argument; MPIRecvStream doesn't own it.

}; // class CheckpointEntryMPIRecvStream

} // namespace PV

#endif // CHECKPOINTENTRYMPIRECVSTREAM_HPP_
