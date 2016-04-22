/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../columns/InterColComm.hpp"


namespace PV {

InitWeights::InitWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitWeights::InitWeights()
{
   initialize_base();
}

InitWeights::~InitWeights()
{
   free(name); name = NULL;
   delete weightParams; weightParams = NULL;
}

int InitWeights::initialize(char const * name, HyPerCol * hc) {
   if (name==NULL) {
      fprintf(stderr, "InitWeights::initialize called with a name argument of null.\n");
      exit(EXIT_FAILURE);
   }
   if (hc==NULL) {
      fprintf(stderr, "InitWeights::initialize called with a HyPerCol argument of null.\n");
      exit(EXIT_FAILURE);
   }
   int status = BaseObject::initialize(name, hc);
   callingConn = NULL; // will be set during communicateInitInfo stage.

   return PV_SUCCESS;
}

int InitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Read/write any params from the params file, typically
   // parent->ioParamValue(ioFlag, name, "param_name", &param, default_value);

   if (ioFlag==PARAMS_IO_READ) {
      weightParams = createNewWeightParams();
   }
   int status = PV_SUCCESS;
   if (weightParams==NULL) {
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
   if (callingConn==NULL) {
      BaseConnection * baseCallingConn = parent->getConnFromName(name);
      if (baseCallingConn==NULL) {
         status = PV_FAILURE;
         if (parent->columnId()==0) {
            fprintf(stderr, "InitWeights error: \"%s\" is not a connection in the column.\n", name);
         }
      }
      else {
         callingConn = dynamic_cast<HyPerConn *>(baseCallingConn);
         if (callingConn==NULL) {
            status = PV_FAILURE;
            if (parent->columnId()==0) {
               fprintf(stderr, "InitWeights error: \"%s\" is not a HyPerConn.\n", name);
            }
         }
      }
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status == PV_FAILURE) {
      exit(EXIT_FAILURE);
   }
   return weightParams->communicateParamsInfo();
}

/*This method does the three steps involved in initializing weights.  Subclasses usually don't need to override this method.
 * Instead they should override calcWeights to do their own type of weight initialization.
 *
 * For KernelConns (i.e., sharedWeights=true), patches should be NULL.
 *
 */
int InitWeights::initializeWeights(PVPatch *** patches, pvwdata_t ** dataStart,
      double * timef /*default NULL*/) {
   PVParams * inputParams = callingConn->getParent()->parameters();
   int numPatches = callingConn->getNumDataPatches();
   if (inputParams->present(callingConn->getName(), "initFromLastFlag")) {
      if (callingConn->getParent()->columnId()==0) {
         fprintf(stderr, "Connection \"%s\": initFromLastFlag is obsolete.\n", callingConn->getName());
      }
      if (inputParams->value(callingConn->getName(), "initFromLastFlag")) {
         if (callingConn->getParent()->columnId()==0) {
            fprintf(stderr, "Instead, use weightInitType=\"FileWeight\" or set HyPerCol initializeFromCheckpointDir and set initializeFromCheckpointFlag to true\n");
         }
         MPI_Barrier(callingConn->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   if (weightParams->getFilename() != NULL && weightParams->getFilename()[0]) {
      readWeights(patches, dataStart, numPatches, weightParams->getFilename(), timef);
   } else {
      initRNGs(patches==NULL);
      calcWeights();
   } // filename != null
   int successFlag = zeroWeightsOutsideShrunkenPatch(patches);
   if (successFlag != PV_SUCCESS) {
      fprintf(stderr,
            "Failed to zero annulus around shrunken patch for %s! Exiting...\n",
            callingConn->getName());
      exit(PV_FAILURE);
   }
   return PV_SUCCESS;
}

int InitWeights::zeroWeightsOutsideShrunkenPatch(PVPatch *** patches) {
    // hack to bypass HyPerConn's for now, because HyPerConn normalization currently needs "outside" weights
    // correct solution is to implement normalization of HyPerConns from post POV
    if (patches != NULL){
        return PV_SUCCESS;
    }
    int numArbors = callingConn->numberOfAxonalArborLists();
    // initialize full sized patch dimensions
    int nxPatch = callingConn->xPatchSize();
    int nyPatch  = callingConn->yPatchSize();
    int nkPatch  = callingConn->fPatchSize() * nxPatch;
    int syPatch = callingConn->yPatchStride();                   // stride in patch
    int offsetPatch = 0;
    pvwdata_t * wData_head = NULL;
    int delta_offset = 0;
    for (int arborID = 0; arborID < numArbors; arborID++) {
        for (int kPre = 0; kPre < callingConn->getNumDataPatches(); kPre++) {
            wData_head = callingConn->get_wDataHead(arborID,kPre);
            if (patches != NULL) {  // callingConn does not use shared weights
                PVPatch * weightPatch = callingConn->getWeights(kPre, arborID);
                nxPatch = weightPatch->nx;
                nyPatch = weightPatch->ny;
                offsetPatch = weightPatch->offset;
                pvwdata_t * wData = callingConn->get_wData(arborID,kPre);
                delta_offset = wData - wData_head;
            }
            else {  // callingConn uses shared weights
                delta_offset = 0;
                nxPatch = callingConn->xPatchSize();
                nyPatch = callingConn->yPatchSize();
            }
            nkPatch  = callingConn->fPatchSize() * nxPatch;
            int dy_south = delta_offset / syPatch;
            assert(dy_south >= 0); assert(dy_south <= callingConn->yPatchSize());
            int dy_north = callingConn->yPatchSize() - nyPatch - dy_south;
            assert(dy_north >= 0); assert(dy_north <= callingConn->yPatchSize());
            int dx_west = (delta_offset - dy_south * syPatch) / callingConn->fPatchSize();
            assert(dx_west >= 0); assert(dx_west <= callingConn->xPatchSize());
            int dx_east = callingConn->xPatchSize() - nxPatch - dx_west;
            assert(dx_east >= 0); assert(dx_east <= callingConn->xPatchSize());
            // zero north border
            pvwdata_t * outside_weights = wData_head;
            for (int ky = 0; ky < dy_north; ky++){
                for (int kPatch = 0; kPatch < syPatch; kPatch++){
                    outside_weights[kPatch] = 0;
                }
                outside_weights+= syPatch;
            }
            // zero south border
            outside_weights = wData_head +
                    (dy_north + nyPatch) * syPatch;
            for (int ky = 0; ky < dy_south; ky++){
                for (int kPatch = 0; kPatch < syPatch; kPatch++){
                    outside_weights[kPatch] = 0;
                }
                outside_weights+= syPatch;
            }
            // zero west border
            outside_weights = wData_head +
                    dy_north * syPatch;
            for (int ky = 0; ky < nyPatch; ky++){
                for (int kPatch = 0; kPatch < dx_west * callingConn->fPatchSize(); kPatch++){
                    outside_weights[kPatch] = 0;
                }
                outside_weights+= syPatch;
            }
            // zero east border
            outside_weights = wData_head +
                    dy_north * syPatch +
                    (dx_west + nxPatch) * callingConn->fPatchSize();
            for (int ky = 0; ky < nyPatch; ky++){
                for (int kPatch = 0; kPatch < dx_east * callingConn->fPatchSize(); kPatch++){
                    outside_weights[kPatch] = 0;
                }
                outside_weights+= syPatch;
            }
        } // kPre
    } // arborID
    return PV_SUCCESS;
}


InitWeightsParams * InitWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitWeightsParams(name, parent);
   return tempPtr;
}

// Override this function to calculate weights over all patches.
// The default loops over arbors and data patches, and calls calcWeights(dataStart, dataPatchIndex, arborId)
int InitWeights::calcWeights() {
   int numArbors = callingConn->numberOfAxonalArborLists();
   int numPatches = callingConn->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataPatchIndex = 0;
            dataPatchIndex < numPatches;
            dataPatchIndex++) {
         int successFlag = calcWeights(
               callingConn->get_wDataHead(arbor, dataPatchIndex),
               dataPatchIndex, arbor);
         if (successFlag != PV_SUCCESS) {
            fprintf(stderr,
                  "Failed to create weights for %s! Exiting...\n",
                  callingConn->getName());
            exit(PV_FAILURE);
         }
      }
   }
   return PV_SUCCESS;
}

// Override this function to calculate the weights in a single patch, given the arbor index, patch index and the pointer to the data
int InitWeights::calcWeights(pvwdata_t * dataStart, int dataPatchIndex, int arborId) {
    return PV_SUCCESS;
}

int InitWeights::initialize_base() {
   callingConn = NULL;
   weightParams = NULL;
   return PV_SUCCESS;
}

int InitWeights::readWeights(PVPatch *** patches, pvwdata_t ** dataStart, int numPatches, const char * filename, double * timeptr/*default=NULL*/) {
   if (callingConn==NULL) {
      callingConn = dynamic_cast<HyPerConn *>(parent->getConnFromName(name));
   }
   InterColComm *icComm = callingConn->getParent()->icCommunicator();
   int numArbors = callingConn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = callingConn->preSynapticLayer()->getLayerLoc();
   double timed;
   const int nxp = callingConn->xPatchSize();
   const int nyp = callingConn->yPatchSize();
   const int nfp = callingConn->fPatchSize();
   int status = PV_SUCCESS;
   if( weightParams->getUseListOfArborFiles() ) {
      status = this->readListOfArborFiles(patches, dataStart, numPatches, filename, timeptr);
   }
   else if (weightParams->getCombineWeightFiles()){
      status = this->readCombinedWeightFiles(patches, dataStart, numPatches, filename, timeptr);
   }
   else {
      status = PV::readWeights(patches, dataStart, numArbors, numPatches, nxp, nyp, nfp, filename, icComm, &timed, preLoc);
   }
   if (status != PV_SUCCESS) {
      fprintf(stderr, "PV::readWeights: problem reading weight file %s for connection %s, SHUTTING DOWN\n", filename, callingConn->getName());
      exit(EXIT_FAILURE);
   }
   if( timeptr != NULL ) *timeptr = timed;
   return PV_SUCCESS;
}

int InitWeights::readListOfArborFiles(PVPatch *** patches, pvwdata_t ** dataStart, int numPatches, const char * listOfArborsFilename, double * timef) {
   int arbor=0;
   InterColComm *icComm = callingConn->getParent()->icCommunicator();
   int numArbors = callingConn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = callingConn->preSynapticLayer()->getLayerLoc();
   double timed;
   PV_Stream * arborstream = pvp_open_read_file(listOfArborsFilename, icComm);

   int rootproc = 0;
   char arborfilename[PV_PATH_MAX];
   while( arbor < callingConn->numberOfAxonalArborLists() ) {
      if( icComm->commRank() == rootproc ) {
         char * fgetsstatus = fgets(arborfilename, PV_PATH_MAX, arborstream->fp);
         if( fgetsstatus == NULL ) {
            bool endoffile = feof(arborstream->fp)!=0;
            if( endoffile ) {
               fprintf(stderr, "File of arbor files \"%s\" reached end of file before all %d arbors were read.  Exiting.\n", listOfArborsFilename, numArbors);
               exit(EXIT_FAILURE);
            }
            else {
               int error = ferror(arborstream->fp);
               assert(error);
               fprintf(stderr, "File of arbor files: error %d while reading.  Exiting.\n", error);
               exit(error);
            }
         }
         else {
            // Remove linefeed from end of string
            arborfilename[PV_PATH_MAX-1] = '\0';
            int len = strlen(arborfilename);
            if (len > 1) {
               if (arborfilename[len-1] == '\n') {
                  arborfilename[len-1] = '\0';
               }
            }
         }
      } // commRank() == rootproc
      int filetype, datatype;
      int numParams = NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS;
      int params[NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS];
      pvp_read_header(arborfilename, icComm, &timed, &filetype, &datatype, params, &numParams);
      int thisfilearbors = params[INDEX_NBANDS];
      const int nxp = callingConn->xPatchSize();
      const int nyp = callingConn->yPatchSize();
      const int nfp = callingConn->fPatchSize();

      int status = PV::readWeights(patches ? &patches[arbor] : NULL, &dataStart[arbor], numArbors-arbor, numPatches, nxp, nyp, nfp, arborfilename, icComm, &timed, preLoc);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", arborfilename);
         exit(EXIT_FAILURE);
      }
      arbor += thisfilearbors;
   }  // while
   pvp_close_file(arborstream, icComm);
   return PV_SUCCESS;
}

int InitWeights::readCombinedWeightFiles(PVPatch *** patches, pvwdata_t ** dataStart,int numPatches, const char * fileOfWeightFiles, double * timef) {
   InterColComm *icComm = callingConn->getParent()->icCommunicator();
   int numArbors = callingConn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = callingConn->preSynapticLayer()->getLayerLoc();
   double timed;
   int rootproc = 0;
   int max_weight_files = 1;  // arbitrary limit...
   int num_weight_files = weightParams->getNumWeightFiles();
   int file_count=0;
   PV_Stream * weightstream = pvp_open_read_file(fileOfWeightFiles, icComm);
   if ((weightstream == NULL) && (icComm->commRank() == rootproc) ){
      fprintf(stderr, ""
            "Cannot open file of weight files \"%s\".  Exiting.\n", fileOfWeightFiles);
      exit(EXIT_FAILURE);
   }

   char weightsfilename[PV_PATH_MAX];
   while( file_count < num_weight_files ) {
      if( icComm->commRank() == rootproc ) {
         char * fgetsstatus = fgets(weightsfilename, PV_PATH_MAX, weightstream->fp);
         if( fgetsstatus == NULL ) {
            bool endoffile = feof(weightstream->fp)!=0;
            if( endoffile ) {
               fprintf(stderr, "File of weight files \"%s\" reached end of file before all %d weight files were read.  Exiting.\n", fileOfWeightFiles, num_weight_files);
               exit(EXIT_FAILURE);
            }
            else {
               int error = ferror(weightstream->fp);
               assert(error);
               fprintf(stderr, "File of weight files: error %d while reading.  Exiting.\n", error);
               exit(error);
            }
         }
         else {
            // Remove linefeed from end of string
            weightsfilename[PV_PATH_MAX-1] = '\0';
            int len = strlen(weightsfilename);
            if (len > 1) {
               if (weightsfilename[len-1] == '\n') {
                  weightsfilename[len-1] = '\0';
               }
            }
         }
      } // commRank() == rootproc
      int filetype, datatype;
      int numParams = NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS;
      int params[NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS];
      pvp_read_header(weightsfilename, icComm, &timed, &filetype, &datatype, params, &numParams);
      const int nxp = callingConn->xPatchSize();
      const int nyp = callingConn->yPatchSize();
      const int nfp = callingConn->fPatchSize();
      int status = PV::readWeights(patches, dataStart, numArbors, numPatches, nxp, nyp, nfp, weightsfilename, icComm, &timed, preLoc);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", weightsfilename);
         exit(EXIT_FAILURE);
      }
      file_count += 1;
   } // file_count < num_weight_files

   return PV_SUCCESS;
}

BaseObject * createInitWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitWeights(name, hc) : NULL;
}

} /* namespace PV */
