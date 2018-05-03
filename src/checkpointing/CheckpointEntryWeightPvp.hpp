/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "components/Weights.hpp"
#include "include/pv_types.h"
#include <string>

namespace PV {

class CheckpointEntryWeightPvp : public CheckpointEntry {
  public:
   CheckpointEntryWeightPvp(
         std::string const &name,
         MPIBlock const *mpiBlock,
         Weights *weights,
         bool compressFlag)
         : CheckpointEntry(name, mpiBlock) {
      initialize(weights, compressFlag);
   }
   CheckpointEntryWeightPvp(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         Weights *weights,
         bool compressFlag)
         : CheckpointEntry(objName, dataName, mpiBlock) {
      initialize(weights, compressFlag);
   }
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  protected:
   void initialize(Weights *weights, bool compressFlag);

  private:
   Weights *mWeights = nullptr;
   bool mCompressFlag;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
