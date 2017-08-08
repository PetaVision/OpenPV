/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "components/Patch.hpp"
#include "include/pv_types.h"
#include <string>

namespace PV {

class CheckpointEntryWeightPvp : public CheckpointEntry {
  public:
   CheckpointEntryWeightPvp(
         std::string const &name,
         MPIBlock const *mpiBlock,
         int numArbors,
         bool sharedWeights,
         Patch const *const *const *patchGeometry,
         float **weightData,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag)
         : CheckpointEntry(name, mpiBlock) {
      initialize(
            numArbors,
            sharedWeights,
            patchGeometry,
            weightData,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            nxp,
            nyp,
            nfp,
            preLoc,
            postLoc,
            compressFlag);
   }
   CheckpointEntryWeightPvp(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         int numArbors,
         bool sharedWeights,
         Patch const *const *const *patchGeometry,
         float **weightData,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag)
         : CheckpointEntry(objName, dataName, mpiBlock) {
      initialize(
            numArbors,
            sharedWeights,
            patchGeometry,
            weightData,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            nxp,
            nyp,
            nfp,
            preLoc,
            postLoc,
            compressFlag);
   }
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  protected:
   void initialize(
         int numArbors,
         bool sharedWeights,
         Patch const *const *const *patchGeometry,
         float **weightData,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         int nxp,
         int nyp,
         int nfp,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         bool compressFlag);

   void calcMinMaxWeights(float *minWeight, float *maxWeight) const;

  private:
   bool mSharedWeights;
   Patch const *const *const *mPatchGeometry;
   int mNumArbors;
   float **mWeightData;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
   int mWeightDataSize;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   PVLayerLoc const *mPreLoc;
   PVLayerLoc const *mPostLoc;
   bool mCompressFlag;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
