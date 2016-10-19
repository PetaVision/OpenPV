/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "columns/Communicator.hpp"
#include "include/pv_datatypes.h"
#include <string>

namespace PV {

class CheckpointEntryWeightPvp : public CheckpointEntry {
  public:
   CheckpointEntryWeightPvp(
         std::string const &name,
         Communicator *communicator,
         int numArbors,
         bool sharedWeights,
         PVPatch ***patchData,
         int patchDataSize,
         pvdata_t **weightData,
         int weightDataSize,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag)
         : CheckpointEntry(name, communicator) {
      initialize(numArbors, sharedWeights, patchData, patchDataSize, weightData, weightDataSize, nxp, nyp, nfp, preLoc, postLoc, compressFlag);
   }
   CheckpointEntryWeightPvp(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator,
         int numArbors,
         bool sharedWeights,
         PVPatch ***patchData,
         int patchDataSize,
         pvdata_t **weightData,
         int weightDataSize,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag)
         : CheckpointEntry(objName, dataName, communicator) {
      initialize(numArbors, sharedWeights, patchData, patchDataSize, weightData, weightDataSize, nxp, nyp, nfp, preLoc, postLoc, compressFlag);
   }
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  protected:
   void initialize(
         int numArbors,
         bool sharedWeights,
         PVPatch ***patchData,
         int patchDataSize,
         pvdata_t **weightData,
         int weightDataSize,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag);

   void calcMinMaxWeights(float *minWeight, float *maxWeight) const;

  private:
   bool mSharedWeights;
   PVPatch ***mPatchData;
   pvdata_t **mWeightData;
   int mPatchDataSize;
   int mWeightDataSize;
   int mNumArbors;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   PVLayerLoc const *mPreLoc;
   PVLayerLoc const *mPostLoc;
   bool mCompressFlag;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
