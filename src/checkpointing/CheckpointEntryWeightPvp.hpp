/*
 * CheckpointEntryWeightPvp.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYWEIGHTPVP_HPP_
#define CHECKPOINTENTRYWEIGHTPVP_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "structures/Weights.hpp"
#include <string>

namespace PV {

class CheckpointEntryWeightPvp : public CheckpointEntry {
  public:
   CheckpointEntryWeightPvp(
         std::string const &name,
         Weights *weights,
         bool compressFlag)
         : CheckpointEntry(name) {
      initialize(weights, compressFlag);
   }
   CheckpointEntryWeightPvp(
         std::string const &objName,
         std::string const &dataName,
         Weights *weights,
         bool compressFlag)
         : CheckpointEntry(objName, dataName) {
      initialize(weights, compressFlag);
   }
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(
         std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  protected:
   void initialize(Weights *weights, bool compressFlag);

  private:
   Weights *mWeights = nullptr;
   bool mCompressFlag;
};

} // end namespace PV

#endif // CHECKPOINTENTRYWEIGHTPVP_HPP_
