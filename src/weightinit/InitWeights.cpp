/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"
#include "components/WeightsPair.hpp"
#include "include/PVLayerLoc.hpp"
#include "io/FileManager.hpp"
#include "io/FileStream.hpp"
#include "io/FileStreamBuilder.hpp"
#include "io/WeightsFileIO.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "structures/MPIBlock.hpp"
#include "structures/PatchGeometry.hpp"
#include "utils/PathComponents.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp" // dist2NearestCell, featureIndex, kxPos, kyPos

#include <cstdlib> // free

namespace PV {

InitWeights::InitWeights(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

InitWeights::InitWeights() {}

InitWeights::~InitWeights() {
   free(mWeightInitTypeString);
   free(mFilename);
}

void InitWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void InitWeights::setObjectType() {
   char const *initType =
         parameters()->stringValue(getName(), "weightInitType", false /*do not warn if absent*/);
   mObjectType = initType ? initType : "Initializer for";
}

int InitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_weightInitType(ioFlag);
   ioParam_initWeightsFile(ioFlag);
   ioParam_frameNumber(ioFlag);

   // obsolete parameters; issue warnings/errors if they are set.
   ioParam_useListOfArborFiles(ioFlag);
   ioParam_combineWeightFiles(ioFlag);
   return PV_SUCCESS;
}

void InitWeights::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, getName(), "weightInitType", &mWeightInitTypeString);
}

void InitWeights::ioParam_initWeightsFile(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, getName(), "initWeightsFile", &mFilename, mFilename, false /*warnIfAbsent*/);
}

void InitWeights::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "initWeightsFile"));
   if (mFilename and mFilename[0]) {
      parameters()->ioParamValue(
            ioFlag,
            getName(),
            "frameNumber",
            &mFrameNumber,
            mFrameNumber /*default*/,
            false /*warn if absent*/);
   }
}

// useListOfArborFiles and combineWeightFiles were marked obsolete July 13, 2017.
// After a reasonable fade time, ioParam_useListOfArborFiles, ioParam_combineWeightFiles,
// and handleObsoleteFlag can be removed.
// If need for these flags arises in the future, they should be added in a subclass, instead
// of complicating the base InitWeights class.
void InitWeights::ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      handleObsoleteFlag(std::string("useListOfArborFiles"));
   }
}

void InitWeights::ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      handleObsoleteFlag(std::string("useListOfArborFiles"));
   }
}

void InitWeights::handleObsoleteFlag(std::string const &flagName) {
   if (parameters()->present(getName(), flagName.c_str())) {
      if (parameters()->value(getName(), flagName.c_str())) {
         Fatal().printf(
               "%s sets the %s flag, which is obsolete.\n",
               getDescription().c_str(),
               flagName.c_str());
      }
      else {
         WarnLog().printf(
               "%s sets the %s flag to false. This flag is obsolete.\n",
               getDescription().c_str(),
               flagName.c_str());
      }
   }
}

Response::Status
InitWeights::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *weightsPair = message->mObjectTable->findObject<WeightsPair>(getName());
   pvAssert(weightsPair);
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   weightsPair->needPre();
   mWeights = weightsPair->getPreWeights();
   FatalIf(
         mWeights == nullptr,
         "%s cannot get Weights object from %s.\n",
         getDescription_c(),
         weightsPair->getDescription_c());
   return Response::SUCCESS;
}

Response::Status
InitWeights::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   FatalIf(
         mWeights == nullptr,
         "initializeState was called for %s with a null Weights object.\n",
         getDescription_c());
   if (mFilename && mFilename[0]) {
      readWeights(mFilename, mFrameNumber);
   }
   else {
      initRNGs(mWeights->weightsTypeIsShared());
      calcWeights();
   } // mFilename != null
   mWeights->setTimestamp(0.0);
   return Response::SUCCESS;
}

void InitWeights::calcWeights() {
   int numArbors  = mWeights->getNumArbors();
   int numPatches = mWeights->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataPatchIndex = 0; dataPatchIndex < numPatches; dataPatchIndex++) {
         calcWeights(dataPatchIndex, arbor);
      }
   }
}

// Override this function to calculate the weights in a single patch, given the arbor index, patch
// index and the pointer to the data
void InitWeights::calcWeights(int dataPatchIndex, int arborId) {}

int InitWeights::readWeights(
      const char *path,
      int frameNumber,
      double *timestampPtr /*default=nullptr*/) {
   double timestamp;

   // Currently, initializing weights from file assumes that the entire weights are in a single
   // file in the filesystem attached to the global root process.
   //
   // Going forward, we might want to make InitWeights be able to read from weights distributed
   // across nodes using the M-to-N directory structure.
   std::shared_ptr<MPIBlock const> globalMPIBlock = mCommunicator->getGlobalMPIBlock();
   std::string filedir = dirName(path);
   std::string filename = baseName(path);
   auto fileManager = std::make_shared<FileManager>(globalMPIBlock, filedir);
   std::shared_ptr<FileStream> fileStream = FileStreamBuilder(
         fileManager,
         filename,
         false /*isTextFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/).get();
   WeightsFileIO weightsFileIO(fileStream.get(), globalMPIBlock, mWeights);
   timestamp = weightsFileIO.readWeights(frameNumber);
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
   return PV_SUCCESS;
}

int InitWeights::dataIndexToUnitCellIndex(int dataIndex, int *kx, int *ky, int *kf) {
   PVLayerLoc const &preLoc  = mWeights->getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = mWeights->getGeometry()->getPostLoc();

   int xDataIndex, yDataIndex, fDataIndex;
   if (mWeights->weightsTypeIsShared()) {

      int nxData = mWeights->getNumDataPatchesX();
      int nyData = mWeights->getNumDataPatchesY();
      int nfData = mWeights->getNumDataPatchesF();
      pvAssert(nfData == preLoc.nf);

      xDataIndex = kxPos(dataIndex, nxData, nyData, nfData);
      yDataIndex = kyPos(dataIndex, nxData, nyData, nfData);
      fDataIndex = featureIndex(dataIndex, nxData, nyData, nfData);
   }
   else { // nonshared weights.
      // data index is extended presynaptic index; convert to restricted.
      int nxExt  = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
      int nyExt  = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
      xDataIndex = kxPos(dataIndex, nxExt, nyExt, preLoc.nf) - preLoc.halo.lt;
      yDataIndex = kyPos(dataIndex, nxExt, nyExt, preLoc.nf) - preLoc.halo.up;
      fDataIndex = featureIndex(dataIndex, nxExt, nyExt, preLoc.nf);
   }
   int xStride = (preLoc.nx > postLoc.nx) ? preLoc.nx / postLoc.nx : 1;
   pvAssert(xStride > 0);

   int yStride = (preLoc.ny > postLoc.ny) ? preLoc.ny / postLoc.ny : 1;
   pvAssert(yStride > 0);

   int xUnitCell = xDataIndex % xStride;
   if (xUnitCell < 0) {
      xUnitCell += xStride;
   }
   pvAssert(xUnitCell >= 0 and xUnitCell < xStride);

   int yUnitCell = yDataIndex % yStride;
   if (yUnitCell < 0) {
      yUnitCell += yStride;
   }
   pvAssert(yUnitCell >= 0 and yUnitCell < yStride);

   int kUnitCell = kIndex(xUnitCell, yUnitCell, fDataIndex, xStride, yStride, preLoc.nf);

   if (kx) {
      *kx = xUnitCell;
   }
   if (ky) {
      *ky = yUnitCell;
   }
   if (kf) {
      *kf = fDataIndex;
   }
   return kUnitCell;
}

int InitWeights::kernelIndexCalculations(int dataPatchIndex) {
   // kernel index stuff:
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);
   const int kxPre = kxKernelIndex;
   const int kyPre = kyKernelIndex;
   const int kfPre = kfKernelIndex;

   // get distances to nearest neighbor in post synaptic layer (meaured relative to pre-synatpic
   // cell)
   int log2ScaleDiffX = mWeights->getGeometry()->getLog2ScaleDiffX();
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre, log2ScaleDiffX, &xDistNNPreUnits, &xDistNNPostUnits);

   int log2ScaleDiffY = mWeights->getGeometry()->getLog2ScaleDiffY();
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kxPre, log2ScaleDiffY, &yDistNNPreUnits, &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre, log2ScaleDiffX);
   kyNN = nearby_neighbor(kyPre, log2ScaleDiffY);

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre, mWeights->getPatchSizeX(), log2ScaleDiffX);
   kyHead = zPatchHead(kyPre, mWeights->getPatchSizeY(), log2ScaleDiffY);

   // get distance to patch head (measured relative to pre-synaptic cell)
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale =
         xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits / xDistNNPostUnits;
   mXDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale =
         yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits / yDistNNPostUnits;
   mYDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;

   // sigma is in units of pre-synaptic layer
   mDxPost = xRelativeScale;
   mDyPost = yRelativeScale;

   return kfPre;
}

float InitWeights::calcYDelta(int jPost) { return calcDelta(jPost, mDyPost, mYDistHeadPreUnits); }

float InitWeights::calcXDelta(int iPost) { return calcDelta(iPost, mDxPost, mXDistHeadPreUnits); }

float InitWeights::calcDelta(int post, float dPost, float distHeadPreUnits) {
   return distHeadPreUnits + post * dPost;
}

} /* namespace PV */
