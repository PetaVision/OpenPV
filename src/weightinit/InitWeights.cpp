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
   ioParam_useListOfArborFiles(ioFlag);
   ioParam_combineWeightFiles(ioFlag);
   ioParam_numWeightFiles(ioFlag);
   return PV_SUCCESS;
}

void InitWeights::ioParam_initWeightsFile(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "initWeightsFile", &mFilename, mFilename, false /*warnIfAbsent*/);
}

void InitWeights::ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (mFilename != nullptr) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "useListOfArborFiles",
            &mUseListOfArborFiles,
            mUseListOfArborFiles,
            true /*warnIfAbsent*/);
   }
}

void InitWeights::ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (mFilename != nullptr) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "combineWeightFiles",
            &mCombineWeightFiles,
            mCombineWeightFiles,
            true /*warnIfAbsent*/);
   }
}

void InitWeights::ioParam_numWeightFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (mFilename != nullptr) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "combineWeightFiles"));
      if (mCombineWeightFiles) {
         parent->parameters()->ioParamValue(
               ioFlag,
               name,
               "numWeightFiles",
               &mNumWeightFiles,
               mNumWeightFiles,
               true /*warnIfAbsent*/);
      }
   }
}

int InitWeights::communicateInitInfo(CommunicateInitInfoMessage const *message) {
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
            sharedWeights, dataStart, numPatchesX, numPatchesY, numPatchesF, mFilename, timef);
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
      double *timestampPtr /*default=nullptr*/) {
   pvAssert(mCallingConn);
   double timestamp;
   int status = PV_SUCCESS;
   if (mUseListOfArborFiles) {
      this->readListOfArborFiles(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            filename,
            timestampPtr);
   }
   else if (mCombineWeightFiles) {
      this->readCombinedWeightFiles(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            filename,
            timestampPtr);
   }
   else {
      readWeightPvpFile(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            filename,
            mCallingConn->numberOfAxonalArborLists(),
            timestampPtr);
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "InitWeights::readWeights: failed to read weight file %s for connection %s.\n",
            filename,
            mCallingConn->getName());
   }
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
   return PV_SUCCESS;
}

void InitWeights::readListOfArborFiles(
      bool sharedWeights,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      const char *listOfArborsFilename,
      double *timestampPtr) {
   int arbor            = 0;
   Communicator *icComm = parent->getCommunicator();
   int numArbors        = mCallingConn->numberOfAxonalArborLists();
   double timestamp;

   std::ifstream *listOfArborsStream = nullptr;
   int rootproc                      = 0;
   int rank                          = icComm->commRank();
   if (rank == rootproc) {
      listOfArborsStream = new std::ifstream(listOfArborsFilename);
      FatalIf(
            listOfArborsStream->fail() or listOfArborsStream->bad(),
            "Unable to open list of arbor files \"%s\": %s\n",
            listOfArborsFilename,
            std::strerror(errno));
   }
   while (arbor < mCallingConn->numberOfAxonalArborLists()) {
      int arborsInFile;
      std::string arborPath;
      if (rank == rootproc) {
         FatalIf(
               listOfArborsStream->eof(),
               "File of arbor files \"%s\" ended before all %d arbors were read.\n",
               listOfArborsFilename,
               numArbors);
         std::getline(*listOfArborsStream, arborPath);
         FatalIf(
               listOfArborsStream->fail(),
               "Unable to read list of arbor files \"%s\": %s\n",
               listOfArborsFilename,
               std::strerror(errno));
         if (arborPath.empty()) {
            continue;
         }
         FileStream arborFileStream(arborPath.c_str(), std::ios_base::in, false);
         BufferUtils::WeightHeader header;
         arborFileStream.read(&header, sizeof(header));
         arborsInFile = header.baseHeader.nBands;
      } // commRank() == rootproc
      MPI_Bcast(&arborsInFile, 1, MPI_INT, rootproc, icComm->getLocalMPIBlock()->getComm());

      readWeightPvpFile(
            sharedWeights,
            &dataStart[arbor],
            numPatchesX,
            numPatchesY,
            numPatchesF,
            arborPath.c_str(),
            arborsInFile,
            &timestamp);
      arbor += arborsInFile;

   } // while
   if (rank == rootproc) {
      delete listOfArborsStream;
   }
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
}

void InitWeights::readCombinedWeightFiles(
      bool sharedWeights,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      const char *fileOfWeightFiles,
      double *timestampPtr) {
   Communicator *icComm     = parent->getCommunicator();
   int numArbors            = mCallingConn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = mCallingConn->preSynapticLayer()->getLayerLoc();
   double timestamp;
   int fileCount = 0;

   std::ifstream *listOfWeightFilesStream = nullptr;
   int rootproc                           = 0;
   int rank                               = icComm->commRank();
   if (rank == rootproc) {
      listOfWeightFilesStream = new std::ifstream(fileOfWeightFiles);
      FatalIf(
            listOfWeightFilesStream->fail() or listOfWeightFilesStream->bad(),
            "Unable to open weight files \"%s\": %s\n",
            fileOfWeightFiles,
            std::strerror(errno));
   }
   while (fileCount < mNumWeightFiles) {
      std::string weightFilePath;
      if (rank == rootproc) {
         FatalIf(
               listOfWeightFilesStream->eof(),
               "File of weight files \"%s\" ended before all %d weight files were read.\n",
               fileOfWeightFiles,
               numArbors);
         std::getline(*listOfWeightFilesStream, weightFilePath);
         FatalIf(
               listOfWeightFilesStream->fail(),
               "Unable to read list of weight files \"%s\": %s\n",
               fileOfWeightFiles,
               std::strerror(errno));
         if (weightFilePath.empty()) {
            continue;
         }
      } // commRank() == rootproc
      readWeightPvpFile(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            weightFilePath.c_str(),
            mCallingConn->numberOfAxonalArborLists(),
            &timestamp);
      fileCount++;
   } // fileCount < mNumWeightFiles
   if (rank == rootproc) {
      delete listOfWeightFilesStream;
   }
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
}

void InitWeights::readWeightPvpFile(
      bool sharedWeights,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      const char *weightPvpFile,
      int numArbors,
      double *timestampPtr) {
   double timestamp;
   MPIBlock const *mpiBlock = parent->getCommunicator()->getLocalMPIBlock();

   FileStream *fileStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      fileStream = new FileStream(weightPvpFile, std::ios_base::in, false);
   }

   PVLayerLoc const *preLoc = mCallingConn->preSynapticLayer()->getLayerLoc();
   if (sharedWeights) {
      readSharedWeights(
            fileStream,
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
