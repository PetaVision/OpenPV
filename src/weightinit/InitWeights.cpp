/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"
#include "components/WeightsPair.hpp"
#include "structures/PVLayerLoc.hpp"
#include "io/BroadcastPreWeightsFile.hpp"
#include "io/FileManager.hpp"
#include "io/FileStream.hpp"
#include "io/FileStreamBuilder.hpp"
#include "io/LocalPatchWeightsFile.hpp"
#include "io/SharedWeightsFile.hpp"
#include "io/WeightsFile.hpp"
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
      initRNGs(mWeights->getSharedWeightsFlag());
      calcWeights();
   } // mFilename != null
   mWeights->setTimestamp(0.0);
   return Response::SUCCESS;
}

void InitWeights::calcWeights() {
   int numArbors     = mWeights->getNumArbors();
   int numPatches    = mWeights->getNumDataPatches();
   auto mpiBlock     = getCommunicator()->getGlobalMPIBlock();
   int rowIndex      = mpiBlock->getRowIndex();
   int columnIndex   = mpiBlock->getColumnIndex();
   int mpiBatchIndex = mpiBlock->getBatchIndex();
   int mpiBatchDim   = mpiBlock->getBatchDimension();
   if (mpiBatchIndex == 0) {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         for (int dataPatchIndex = 0; dataPatchIndex < numPatches; dataPatchIndex++) {
               calcWeights(dataPatchIndex, arbor);
         }
         for (int b = 1; b < mpiBatchDim; ++b) {
            int rank = mpiBlock->calcRankFromRowColBatch(rowIndex, columnIndex, b);
            float const *values = mWeights->getData()->getData(arbor);
            int count = static_cast<int>(mWeights->getData()->getNumValuesPerArbor());
            MPI_Send(values, count, MPI_FLOAT, rank, 1234 + arbor /*tag*/, mpiBlock->getComm());
         }
      }
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         float *values = mWeights->getData()->getData(arbor);
         int count = static_cast<int>(mWeights->getData()->getNumValuesPerArbor());
         int rank  = mpiBlock->calcRankFromRowColBatch(rowIndex, columnIndex, 0);
         auto comm = mpiBlock->getComm();
         MPI_Recv(values, count, MPI_FLOAT, rank, 1234 + arbor /*tag*/, comm, MPI_STATUS_IGNORE);
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
   // Currently, initializing weights from file assumes that the entire weights are in a single
   // file in the filesystem attached to the global root process.
   //
   // Going forward, we might want to make InitWeights be able to read from weights distributed
   // across nodes using the M-to-N directory structure.
   std::shared_ptr<MPIBlock const> globalMPIBlock = mCommunicator->getGlobalMPIBlock();
   std::string filedir = dirName(path);
   std::string filename = baseName(path);
   auto fileManager = std::make_shared<FileManager>(globalMPIBlock, filedir);

   // Read header to get CompressedFlag
   std::shared_ptr<FileStream> fileStream = FileStreamBuilder(
         fileManager,
         filename,
         false /*isTextFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/).get();
   int dataTypeInfo[2];
   if (fileStream) {
      BufferUtils::WeightHeader headerFromFile;
      fileStream->read(&headerFromFile, static_cast<long>(sizeof(headerFromFile)));
      dataTypeInfo[0] = headerFromFile.baseHeader.dataSize;
      dataTypeInfo[1] = headerFromFile.baseHeader.dataType;
   }
   MPI_Bcast(dataTypeInfo, 2, MPI_INT, 0, fileManager->getMPIBlock()->getComm());
   bool compressedFlag;
   switch(dataTypeInfo[1]) {
      case BufferUtils::HeaderDataTypeEnum::BYTE:
         compressedFlag = true;
         FatalIf(
               dataTypeInfo[0] != static_cast<int>(sizeof(unsigned char)),
               "%s InitWeights file \"%s\" has inconsistent dataType (%d) and dataSize (%d)\n",
               getDescription_c(),
               filename.c_str(),
               dataTypeInfo[1],
               dataTypeInfo[0]);
         break;
      case BufferUtils::HeaderDataTypeEnum::FLOAT:
         compressedFlag = false;
         FatalIf(
               dataTypeInfo[0] != static_cast<int>(sizeof(float)),
               "%s InitWeights file \"%s\" has inconsistent dataType (%d) and dataSize (%d)\n",
               getDescription_c(),
               filename.c_str(),
               dataTypeInfo[1],
               dataTypeInfo[0]);
         break;
      default:
         Fatal().printf(
               "%s InitWeights file \"%s\" has dataType (%d) incompatible with a weights file.\n",
               getDescription_c(),
               filename.c_str(),
               dataTypeInfo[1]);
         break;
   }

   std::shared_ptr<WeightsFile> weightsFile = nullptr;
   if (mWeights->getSharedWeightsFlag()) {
      weightsFile = std::make_shared<SharedWeightsFile>(
            fileManager,
            filename,
            mWeights->getData(),
            compressedFlag,
            true /*readOnlyFlag*/,
            false /*clobberFlag*/,
            false /*verifyWrites*/);
   }
   else {
      if (mWeights->prelayerIsBroadcast()) {
         weightsFile = std::make_shared<BroadcastPreWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               mWeights->getGeometry()->getPreLoc().nf,
               mWeights->getGeometry()->getPostLoc().bcast,
               compressedFlag,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWritesFlag*/);
      }
      else {
         weightsFile = std::make_shared<LocalPatchWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               &mWeights->getGeometry()->getPreLoc(),
               &mWeights->getGeometry()->getPostLoc(),
               true /*fileExtendedFlag*/,
               compressedFlag,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWrites*/);
      }
   }
   weightsFile->setIndex(frameNumber);
   if (timestampPtr) {
      weightsFile->read(*timestampPtr);
   }
   else {
       weightsFile->read();
   }
   return PV_SUCCESS;
}

int InitWeights::dataIndexToUnitCellIndex(int dataIndex, int *kx, int *ky, int *kf) {
   PVLayerLoc const &preLoc  = mWeights->getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = mWeights->getGeometry()->getPostLoc();

   int xDataIndex, yDataIndex, fDataIndex;
   if (mWeights->getSharedWeightsFlag()) {

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
   dist2NearestCell(kyPre, log2ScaleDiffY, &yDistNNPreUnits, &yDistNNPostUnits);

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
