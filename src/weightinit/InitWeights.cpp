/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "columns/Communicator.hpp"
#include "include/default_params.h"
#include "io/fileio.hpp"
#include "io/io.hpp"
#include "utils/conversions.h"

namespace PV {

InitWeights::InitWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitWeights::InitWeights() { initialize_base(); }

InitWeights::~InitWeights() {
   free(name);
   name = NULL;
   delete weightParams;
   weightParams = NULL;
}

int InitWeights::initialize(char const *name, HyPerCol *hc) {
   if (name == NULL) {
      Fatal().printf("InitWeights::initialize called with a name argument of null.\n");
   }
   if (hc == NULL) {
      Fatal().printf("InitWeights::initialize called with a HyPerCol argument of null.\n");
   }
   int status  = BaseObject::initialize(name, hc);
   callingConn = NULL; // will be set during communicateInitInfo stage.

   return PV_SUCCESS;
}

int InitWeights::setDescription() {
   description.clear();
   char const *initType =
         parent->parameters()->stringValue(name, "weightInitType", false /*do not warn if absent*/);
   if (initType == nullptr) {
      description.append("weight initializer ");
   }
   else {
      description.append(initType);
   }
   description.append(" \"").append(name).append("\"");
   return PV_SUCCESS;
}

int InitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Read/write any params from the params file, typically

   if (ioFlag == PARAMS_IO_READ) {
      weightParams = createNewWeightParams();
   }
   int status = PV_SUCCESS;
   if (weightParams == NULL) {
      status = PV_FAILURE;
   }
   else {
      weightParams->ioParamsFillGroup(ioFlag);
   }
   return status;
}

int InitWeights::communicateParamsInfo() {
   // to be called during communicateInitInfo stage;
   // set any member variables that depend on other objects
   // having been initialized or communicateInitInfo'd

   int status = PV_SUCCESS;
   if (callingConn == NULL) {
      BaseConnection *baseCallingConn = parent->getConnFromName(name);
      if (baseCallingConn == NULL) {
         status = PV_FAILURE;
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "InitWeights error: \"%s\" is not a connection in the column.\n", name);
         }
      }
      else {
         callingConn = dynamic_cast<HyPerConn *>(baseCallingConn);
         if (callingConn == NULL) {
            status = PV_FAILURE;
            if (parent->columnId() == 0) {
               ErrorLog().printf("InitWeights error: \"%s\" is not a HyPerConn.\n", name);
            }
         }
      }
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status == PV_FAILURE) {
      exit(EXIT_FAILURE);
   }
   return weightParams->communicateParamsInfo();
}

/*This method does the three steps involved in initializing weights.  Subclasses usually don't need
 * to override this method.
 * Instead they should override calcWeights to do their own type of weight initialization.
 *
 * For KernelConns (i.e., sharedWeights=true), patches should be NULL.
 *
 */
int InitWeights::initializeWeights(
      PVPatch ***patches,
      float **dataStart,
      double *timef /*default NULL*/) {
   PVParams *inputParams = callingConn->getParent()->parameters();
   int numPatchesX       = callingConn->getNumDataPatchesX();
   int numPatchesY       = callingConn->getNumDataPatchesY();
   int numPatchesF       = callingConn->getNumDataPatchesF();
   bool sharedWeights    = patches == nullptr;
   if (weightParams->getFilename() != NULL && weightParams->getFilename()[0]) {
      readWeights(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            weightParams->getFilename(),
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
            callingConn->getName());
   }
   return PV_SUCCESS;
}

int InitWeights::zeroWeightsOutsideShrunkenPatch(PVPatch ***patches) {
   // hack to bypass HyPerConn's for now, because HyPerConn normalization currently needs "outside"
   // weights
   // correct solution is to implement normalization of HyPerConns from post POV
   if (patches != NULL) {
      return PV_SUCCESS;
   }
   int numArbors = callingConn->numberOfAxonalArborLists();
   // initialize full sized patch dimensions
   int nxPatch       = callingConn->xPatchSize();
   int nyPatch       = callingConn->yPatchSize();
   int nkPatch       = callingConn->fPatchSize() * nxPatch;
   int syPatch       = callingConn->yPatchStride(); // stride in patch
   int offsetPatch   = 0;
   float *wData_head = NULL;
   int delta_offset  = 0;
   for (int arborID = 0; arborID < numArbors; arborID++) {
      for (int kPre = 0; kPre < callingConn->getNumDataPatches(); kPre++) {
         wData_head = callingConn->get_wDataHead(arborID, kPre);
         if (patches != NULL) { // callingConn does not use shared weights
            PVPatch *weightPatch = callingConn->getWeights(kPre, arborID);
            nxPatch              = weightPatch->nx;
            nyPatch              = weightPatch->ny;
            offsetPatch          = weightPatch->offset;
            float *wData         = callingConn->get_wData(arborID, kPre);
            delta_offset         = wData - wData_head;
         }
         else { // callingConn uses shared weights
            delta_offset = 0;
            nxPatch      = callingConn->xPatchSize();
            nyPatch      = callingConn->yPatchSize();
         }
         nkPatch      = callingConn->fPatchSize() * nxPatch;
         int dy_south = delta_offset / syPatch;
         assert(dy_south >= 0);
         assert(dy_south <= callingConn->yPatchSize());
         int dy_north = callingConn->yPatchSize() - nyPatch - dy_south;
         assert(dy_north >= 0);
         assert(dy_north <= callingConn->yPatchSize());
         int dx_west = (delta_offset - dy_south * syPatch) / callingConn->fPatchSize();
         assert(dx_west >= 0);
         assert(dx_west <= callingConn->xPatchSize());
         int dx_east = callingConn->xPatchSize() - nxPatch - dx_west;
         assert(dx_east >= 0);
         assert(dx_east <= callingConn->xPatchSize());
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
            for (int kPatch = 0; kPatch < dx_west * callingConn->fPatchSize(); kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
         // zero east border
         outside_weights =
               wData_head + dy_north * syPatch + (dx_west + nxPatch) * callingConn->fPatchSize();
         for (int ky = 0; ky < nyPatch; ky++) {
            for (int kPatch = 0; kPatch < dx_east * callingConn->fPatchSize(); kPatch++) {
               outside_weights[kPatch] = 0;
            }
            outside_weights += syPatch;
         }
      } // kPre
   } // arborID
   return PV_SUCCESS;
}

InitWeightsParams *InitWeights::createNewWeightParams() {
   InitWeightsParams *tempPtr = new InitWeightsParams(name, parent);
   return tempPtr;
}

// Override this function to calculate weights over all patches.
// The default loops over arbors and data patches, and calls calcWeights(dataStart, dataPatchIndex,
// arborId)
int InitWeights::calcWeights() {
   int numArbors  = callingConn->numberOfAxonalArborLists();
   int numPatches = callingConn->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataPatchIndex = 0; dataPatchIndex < numPatches; dataPatchIndex++) {
         int successFlag = calcWeights(
               callingConn->get_wDataHead(arbor, dataPatchIndex), dataPatchIndex, arbor);
         if (successFlag != PV_SUCCESS) {
            Fatal().printf("Failed to create weights for %s! Exiting...\n", callingConn->getName());
         }
      }
   }
   return PV_SUCCESS;
}

// Override this function to calculate the weights in a single patch, given the arbor index, patch
// index and the pointer to the data
int InitWeights::calcWeights(float *dataStart, int dataPatchIndex, int arborId) {
   return PV_SUCCESS;
}

int InitWeights::initialize_base() {
   callingConn  = NULL;
   weightParams = NULL;
   return PV_SUCCESS;
}

int InitWeights::readWeights(
      bool sharedWeights,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      const char *filename,
      double *timestampPtr /*default=nullptr*/) {
   if (callingConn == nullptr) {
      callingConn = dynamic_cast<HyPerConn *>(parent->getConnFromName(name));
   }
   double timestamp;
   int status = PV_SUCCESS;
   if (weightParams->getUseListOfArborFiles()) {
      this->readListOfArborFiles(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            filename,
            timestampPtr);
   }
   else if (weightParams->getCombineWeightFiles()) {
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
            callingConn->numberOfAxonalArborLists(),
            timestampPtr);
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "InitWeights::readWeights: failed to read weight file %s for connection %s.\n",
            filename,
            callingConn->getName());
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
   Communicator *icComm = callingConn->getParent()->getCommunicator();
   int numArbors        = callingConn->numberOfAxonalArborLists();
   double timestamp;

   std::ifstream *listOfArborsStream = nullptr;
   int rootproc = 0;
   int rank = icComm->commRank();
   if (rank == rootproc) {
      listOfArborsStream = new std::ifstream(listOfArborsFilename);
      FatalIf(
            listOfArborsStream->fail() or listOfArborsStream->bad(),
            "Unable to open list of arbor files \"%s\": %s\n",
            listOfArborsFilename,
            std::strerror(errno));
   }
   while (arbor < callingConn->numberOfAxonalArborLists()) {
      int arborsInFile;
      std::string arborPath;
      if (rank == rootproc) {
         FatalIf(listOfArborsStream->eof(),
               "File of arbor files \"%s\" ended before all %d arbors were read.\n",
               listOfArborsFilename,
               numArbors);
         std::getline(*listOfArborsStream, arborPath);
         FatalIf(listOfArborsStream->fail(),
               "Unable to read list of arbor files \"%s\": %s\n",
               listOfArborsFilename,
               std::strerror(errno));
         if (arborPath.empty()) { continue; }
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
   Communicator *icComm     = callingConn->getParent()->getCommunicator();
   int numArbors            = callingConn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = callingConn->preSynapticLayer()->getLayerLoc();
   double timestamp;
   int max_weight_files    = 1; // arbitrary limit...
   int num_weight_files    = weightParams->getNumWeightFiles();
   int file_count          = 0;

   std::ifstream *listOfWeightFilesStream = nullptr;
   int rootproc = 0;
   int rank = icComm->commRank();
   if (rank == rootproc) {
      listOfWeightFilesStream = new std::ifstream(fileOfWeightFiles);
      FatalIf(
            listOfWeightFilesStream->fail() or listOfWeightFilesStream->bad(),
            "Unable to open weight files \"%s\": %s\n",
            fileOfWeightFiles,
            std::strerror(errno));
   }
   while (file_count < num_weight_files) {
      std::string weightFilePath;
      if (rank == rootproc) {
         FatalIf(listOfWeightFilesStream->eof(),
               "File of weight files \"%s\" ended before all %d weight files were read.\n",
               fileOfWeightFiles,
               numArbors);
         std::getline(*listOfWeightFilesStream, weightFilePath);
         FatalIf(listOfWeightFilesStream->fail(),
               "Unable to read list of weight files \"%s\": %s\n",
               fileOfWeightFiles,
               std::strerror(errno));
         if (weightFilePath.empty()) { continue; }
      } // commRank() == rootproc
      readWeightPvpFile(
            sharedWeights,
            dataStart,
            numPatchesX,
            numPatchesY,
            numPatchesF,
            weightFilePath.c_str(),
            callingConn->numberOfAxonalArborLists(),
            &timestamp);
      file_count++;
   } // file_count < num_weight_files
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
   MPIBlock const *mpiBlock = callingConn->getParent()->getCommunicator()->getLocalMPIBlock();

   FileStream *fileStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      fileStream = new FileStream(weightPvpFile, std::ios_base::in, false);
   }

   PVLayerLoc const *preLoc = callingConn->preSynapticLayer()->getLayerLoc();
   if (sharedWeights) {
      readSharedWeights(
            fileStream,
            mpiBlock,
            preLoc,
            callingConn->xPatchSize(),
            callingConn->yPatchSize(),
            callingConn->fPatchSize(),
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
            callingConn->xPatchSize(),
            callingConn->yPatchSize(),
            callingConn->fPatchSize(),
            numArbors,
            dataStart,
            true /*extended*/,
            callingConn->postSynapticLayer()->getLayerLoc(),
            mpiBlock->getStartColumn() * preLoc->nx,
            mpiBlock->getStartRow() * preLoc->ny);
   }
   if (timestampPtr != nullptr) {
      *timestampPtr = timestamp;
   }
}

} /* namespace PV */
