/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

namespace PV {

InitWeights::InitWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitWeights::InitWeights() { initialize_base(); }

InitWeights::~InitWeights() {}

int InitWeights::initialize(char const *name, HyPerCol *hc) {
   if (name == nullptr) {
      Fatal().printf("InitWeights::initialize called with a name argument of null.\n");
   }
   if (hc == nullptr) {
      Fatal().printf("InitWeights::initialize called with a HyPerCol argument of null.\n");
   }
   int status = BaseObject::initialize(name, hc);

   return status;
}

int InitWeights::setDescription() {
   description.clear();
   char const *initType =
         parent->parameters()->stringValue(name, "weightInitType", false /*do not warn if absent*/);
   if (initType == nullptr) {
      description.append("Weight initializer ");
   }
   else {
      description.append(initType);
   }
   description.append(" \"").append(name).append("\"");
   return PV_SUCCESS;
}

int InitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_initWeightsFile(ioFlag);
   ioParam_frameNumber(ioFlag);

   // obsolete parameters; issue warnings/errors if they are set.
   ioParam_useListOfArborFiles(ioFlag);
   ioParam_combineWeightFiles(ioFlag);
   return PV_SUCCESS;
}

void InitWeights::ioParam_initWeightsFile(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "initWeightsFile", &mFilename, mFilename, false /*warnIfAbsent*/);
}

void InitWeights::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (mFilename and mFilename[0]) {
      parent->parameters()->ioParamValue(
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
   if (parent->parameters()->present(name, flagName.c_str())) {
      if (parent->parameters()->value(name, flagName.c_str())) {
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

int InitWeights::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mCallingConn == nullptr) {
      mCallingConn = message->lookup<HyPerConn>(std::string(name));
   }
   if (mCallingConn == nullptr) {
      if (parent->columnId() == 0) {
         ErrorLog().printf("InitWeights error: \"%s\" is not a HyPerConn.\n", name);
      }
      return PV_FAILURE;
   }
   mPreLayer  = mCallingConn->getPre();
   mPostLayer = mCallingConn->getPost();
   if (mPreLayer == nullptr or mPostLayer == nullptr) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "InitWeights error: calling connection \"%s\" has not set pre and post layers.\n",
               mCallingConn->getName());
      }
      return PV_FAILURE;
   }
   return PV_SUCCESS;
}

/*This method does the three steps involved in initializing weights.  Subclasses usually don't need
 * to override this method.
 * Instead they should override calcWeights to do their own type of weight initialization.
 *
 * For KernelConns (i.e., sharedWeights=true), patches should be nullptr.
 *
 */
int InitWeights::initializeWeights(
      PVPatch ***patches,
      float **dataStart,
      double *timef /*default nullptr*/) {
   int numPatchesX    = mCallingConn->getNumDataPatchesX();
   int numPatchesY    = mCallingConn->getNumDataPatchesY();
   int numPatchesF    = mCallingConn->getNumDataPatchesF();
   bool sharedWeights = patches == nullptr;
   if (mFilename && mFilename[0]) {
      readWeights(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            mFilename,
            mFrameNumber,
            timef);
   }
   else {
      initRNGs(sharedWeights);
      calcWeights();
   } // filename != null
   int successFlag = zeroWeightsOutsideShrunkenPatch(patches);
   if (successFlag != PV_SUCCESS) {
      Fatal().printf(
            "Failed to zero annulus around shrunken patch for %s! Exiting...\n",
            mCallingConn->getName());
   }
   return PV_SUCCESS;
}

int InitWeights::zeroWeightsOutsideShrunkenPatch(PVPatch ***patches) {
   // hack to bypass HyPerConn's for now, because HyPerConn normalization currently needs "outside"
   // weights
   // correct solution is to implement normalization of HyPerConns from post POV
   if (patches != nullptr) {
      return PV_SUCCESS;
   }
   int numArbors = mCallingConn->numberOfAxonalArborLists();
   // initialize full sized patch dimensions
   int nxPatch       = mCallingConn->xPatchSize();
   int nyPatch       = mCallingConn->yPatchSize();
   int nkPatch       = mCallingConn->fPatchSize() * nxPatch;
   int syPatch       = mCallingConn->yPatchStride(); // stride in patch
   int offsetPatch   = 0;
   float *wData_head = nullptr;
   int delta_offset  = 0;
   for (int arborID = 0; arborID < numArbors; arborID++) {
      for (int kPre = 0; kPre < mCallingConn->getNumDataPatches(); kPre++) {
         wData_head = mCallingConn->get_wDataHead(arborID, kPre);
         if (patches != nullptr) { // mCallingConn does not use shared weights
            PVPatch *weightPatch = mCallingConn->getWeights(kPre, arborID);
            nxPatch              = weightPatch->nx;
            nyPatch              = weightPatch->ny;
            offsetPatch          = weightPatch->offset;
            float *wData         = mCallingConn->get_wData(arborID, kPre);
            delta_offset         = wData - wData_head;
         }
         else { // mCallingConn uses shared weights
            delta_offset = 0;
            nxPatch      = mCallingConn->xPatchSize();
            nyPatch      = mCallingConn->yPatchSize();
         }
         nkPatch      = mCallingConn->fPatchSize() * nxPatch;
         int dy_south = delta_offset / syPatch;
         assert(dy_south >= 0);
         assert(dy_south <= mCallingConn->yPatchSize());
         int dy_north = mCallingConn->yPatchSize() - nyPatch - dy_south;
         assert(dy_north >= 0);
         assert(dy_north <= mCallingConn->yPatchSize());
         int dx_west = (delta_offset - dy_south * syPatch) / mCallingConn->fPatchSize();
         assert(dx_west >= 0);
         assert(dx_west <= mCallingConn->xPatchSize());
         int dx_east = mCallingConn->xPatchSize() - nxPatch - dx_west;
         assert(dx_east >= 0);
         assert(dx_east <= mCallingConn->xPatchSize());
         // zero north border
         float *outside_weights = wData_head;
         for (int ky = 0; ky < dy_north; ky++) {
            for (int kPatch = 0; kPatch < syPatch; kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
         // zero south border
         outside_weights = wData_head + (dy_north + nyPatch) * syPatch;
         for (int ky = 0; ky < dy_south; ky++) {
            for (int kPatch = 0; kPatch < syPatch; kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
         // zero west border
         outside_weights = wData_head + dy_north * syPatch;
         for (int ky = 0; ky < nyPatch; ky++) {
            for (int kPatch = 0; kPatch < dx_west * mCallingConn->fPatchSize(); kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
         // zero east border
         outside_weights =
               wData_head + dy_north * syPatch + (dx_west + nxPatch) * mCallingConn->fPatchSize();
         for (int ky = 0; ky < nyPatch; ky++) {
            for (int kPatch = 0; kPatch < dx_east * mCallingConn->fPatchSize(); kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
      } // kPre
   } // arborID
   return PV_SUCCESS;
}

// Override this function to calculate weights over all patches.
// The default loops over arbors and data patches, and calls calcWeights(dataStart, dataPatchIndex,
// arborId)
void InitWeights::calcWeights() {
   int numArbors  = mCallingConn->numberOfAxonalArborLists();
   int numPatches = mCallingConn->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataPatchIndex = 0; dataPatchIndex < numPatches; dataPatchIndex++) {
         calcWeights(mCallingConn->get_wDataHead(arbor, dataPatchIndex), dataPatchIndex, arbor);
      }
   }
}

// Override this function to calculate the weights in a single patch, given the arbor index, patch
// index and the pointer to the data
void InitWeights::calcWeights(float *dataStart, int dataPatchIndex, int arborId) {}

int InitWeights::initialize_base() { return PV_SUCCESS; }

int InitWeights::readWeights(
      bool sharedWeights,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      const char *filename,
      int frameNumber,
      double *timestampPtr /*default=nullptr*/) {
   pvAssert(mCallingConn);
   double timestamp;
   MPIBlock const *mpiBlock = parent->getCommunicator()->getLocalMPIBlock();

   FileStream *fileStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      fileStream = new FileStream(filename, std::ios_base::in, false);
   }

   PVLayerLoc const *preLoc = mCallingConn->preSynapticLayer()->getLayerLoc();
   int numArbors            = mCallingConn->numberOfAxonalArborLists();
   if (sharedWeights) {
      readSharedWeights(
            fileStream,
            frameNumber,
            mpiBlock,
            preLoc,
            mCallingConn->xPatchSize(),
            mCallingConn->yPatchSize(),
            mCallingConn->fPatchSize(),
            numArbors,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF);
   }
   else {
      readNonsharedWeights(
            fileStream,
            frameNumber,
            mpiBlock,
            preLoc,
            mCallingConn->xPatchSize(),
            mCallingConn->yPatchSize(),
            mCallingConn->fPatchSize(),
            numArbors,
            dataStart,
            true /*extended*/,
            mCallingConn->postSynapticLayer()->getLayerLoc(),
            mpiBlock->getStartColumn() * preLoc->nx,
            mpiBlock->getStartRow() * preLoc->ny);
   }
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
   return PV_SUCCESS;
}

int InitWeights::kernelIndexCalculations(int dataPatchIndex) {
   // kernel index stuff:
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   mCallingConn->dataIndexToUnitCellIndex(
         dataPatchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);
   const int kxPre = kxKernelIndex;
   const int kyPre = kyKernelIndex;
   const int kfPre = kfKernelIndex;

   // get distances to nearest neighbor in post synaptic layer (meaured relative to pre-synatpic
   // cell)
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(
         kxPre,
         mPreLayer->getXScale(),
         mPostLayer->getXScale(),
         &xDistNNPreUnits,
         &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(
         kyPre,
         mPreLayer->getYScale(),
         mPostLayer->getYScale(),
         &yDistNNPreUnits,
         &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre, mPreLayer->getXScale(), mPostLayer->getXScale());
   kyNN = nearby_neighbor(kyPre, mPreLayer->getYScale(), mPostLayer->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(
         kxPre, mCallingConn->xPatchSize(), mPreLayer->getXScale(), mPostLayer->getXScale());
   kyHead = zPatchHead(
         kyPre, mCallingConn->yPatchSize(), mPreLayer->getYScale(), mPostLayer->getYScale());

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
