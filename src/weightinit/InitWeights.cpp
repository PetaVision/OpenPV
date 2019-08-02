/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"
#include "components/WeightsPair.hpp"
#include "io/WeightsFileIO.hpp"

namespace PV {

InitWeights::InitWeights(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

InitWeights::InitWeights() {}

InitWeights::~InitWeights() { free(mWeightInitTypeString); }

void InitWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void InitWeights::setObjectType() {
   char const *initType =
         parameters()->stringValue(name, "weightInitType", false /*do not warn if absent*/);
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
   parameters()->ioParamStringRequired(ioFlag, name, "weightInitType", &mWeightInitTypeString);
}

void InitWeights::ioParam_initWeightsFile(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "initWeightsFile", &mFilename, mFilename, false /*warnIfAbsent*/);
}

void InitWeights::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (mFilename and mFilename[0]) {
      parameters()->ioParamValue(
            ioFlag,
            name,
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
   if (parameters()->present(name, flagName.c_str())) {
      if (parameters()->value(name, flagName.c_str())) {
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
      initRNGs(mWeights->getSharedFlag());
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
      const char *filename,
      int frameNumber,
      double *timestampPtr /*default=nullptr*/) {
   double timestamp;
   MPIBlock const *mpiBlock = mCommunicator->getLocalMPIBlock();

   FileStream *fileStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      fileStream = new FileStream(filename, std::ios_base::in, false);
   }
   WeightsFileIO weightsFileIO(fileStream, mpiBlock, mWeights);
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
   if (mWeights->getSharedFlag()) {

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
