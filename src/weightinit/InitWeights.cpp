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

InitWeights::InitWeights()
{
   initialize_base();
}

InitWeights::~InitWeights()
{
}

/*This method does the three steps involved in initializing weights.  Subclasses usually don't need to override this method.
 * Instead they should override calcWeights to do their own type of weight initialization.
 *
 * This method initializes the full unshrunken patch.
 * For KernelConns, patches should be NULL.
 *
 */
int InitWeights::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
        const char * filename, HyPerConn * callingConn,
        double * timef /*default NULL*/) {
    PVParams * inputParams = callingConn->getParent()->parameters();
    int initFromLastFlag = inputParams->value(callingConn->getName(),
            "initFromLastFlag", 0.0f, false) != 0;
    InitWeightsParams *weightParams = NULL;
    int numArbors = callingConn->numberOfAxonalArborLists();
    int numPatches = callingConn->getNumDataPatches();
    if (initFromLastFlag) {
        char nametmp[PV_PATH_MAX];
        int chars_needed = snprintf(nametmp, PV_PATH_MAX, "%s/Last/%s_W.pvp",
                callingConn->getParent()->getOutputPath(),
                callingConn->getName());
        if (chars_needed >= PV_PATH_MAX) {
            fprintf(stderr,
                    "InitWeights::initializeWeights error: filename \"%s/Last/%s_W.pvp\" is too long.\n",
                    callingConn->getParent()->getOutputPath(),
                    callingConn->getName());
            abort();
        }
        readWeights(patches, dataStart, numPatches,
                nametmp, callingConn);
    } else if (filename != NULL) {
        readWeights(patches, dataStart, numPatches,
                filename, callingConn, timef);
    } else {  // calculate weights
        // modified 2/1/13 by garkenyon: no reason for each process not to calculate weights, which helps in implementing shmget
//        HyPerCol * hc = callingConn->getParent();
//        int root_proc = 0;
//        bool callCalcWeights = patches != NULL || hc->columnId() == root_proc;
//        if (callCalcWeights) {
       weightParams = createNewWeightParams(callingConn);
       initRNGs(callingConn, patches==NULL);
       for (int arbor = 0; arbor < numArbors; arbor++) {
#ifdef USE_SHMGET
          bool * shmget_owner = callingConn->getShmgetOwnerHead();
          bool shmget_flag = callingConn->getShmgetFlag();
          if (shmget_flag && !shmget_owner[arbor]) continue;
#endif
          for (int dataPatchIndex = 0;
               dataPatchIndex < numPatches;
               dataPatchIndex++) {
             int successFlag = calcWeights(
                   callingConn->get_wDataHead(arbor, dataPatchIndex),
                   dataPatchIndex, arbor, weightParams);
             if (successFlag != PV_SUCCESS) {
                fprintf(stderr,
                        "Failed to create weights for %s! Exiting...\n",
                        callingConn->getName());
                exit(PV_FAILURE);
             }
          }
       }
       delete (weightParams);
    } // filename != null
    int successFlag = zeroWeightsOutsideShrunkenPatch(patches, callingConn);
    if (successFlag != PV_SUCCESS) {
        fprintf(stderr,
                "Failed to zero annulus around shrunken patch for %s! Exiting...\n",
                callingConn->getName());
        exit(PV_FAILURE);
    }
    return PV_SUCCESS;
}


int InitWeights::zeroWeightsOutsideShrunkenPatch(PVPatch *** patches, HyPerConn * callingConn) {
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
    pvdata_t * wData_head = NULL;
    int delta_offset = 0;
    for (int arborID = 0; arborID < numArbors; arborID++) {
        for (int kPre = 0; kPre < callingConn->getNumDataPatches(); kPre++) {
            wData_head = callingConn->get_wDataHead(arborID,kPre);
            if (patches != NULL) {  // callingConn is a HyPerConn
                PVPatch * weightPatch = callingConn->getWeights(kPre, arborID);
                nxPatch = weightPatch->nx;
                nyPatch = weightPatch->ny;
                offsetPatch = weightPatch->offset;
                nkPatch  = callingConn->fPatchSize() * nxPatch;
                pvdata_t * wData = callingConn->get_wData(arborID,kPre);
                delta_offset = wData - wData_head;
            }
            else {  // callingConn is a KernelConn
                delta_offset = callingConn->getOffsetShrunken();
                nxPatch = callingConn->getNxpShrunken();
                nyPatch = callingConn->getNypShrunken();
                nkPatch  = callingConn->fPatchSize() * nxPatch;
            }
            int dy_south = delta_offset / syPatch;
            assert(dy_south >= 0); assert(dy_south <= callingConn->yPatchSize());
            int dy_north = callingConn->yPatchSize() - nyPatch - dy_south;
            assert(dy_north >= 0); assert(dy_north <= callingConn->yPatchSize());
            int dx_west = (delta_offset - dy_south * syPatch) / callingConn->fPatchSize();
            assert(dx_west >= 0); assert(dx_west <= callingConn->xPatchSize());
            int dx_east = callingConn->xPatchSize() - nxPatch - dx_west;
            assert(dx_east >= 0); assert(dx_east <= callingConn->xPatchSize());
            // zero north border
            pvdata_t * outside_weights = wData_head;
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


InitWeightsParams * InitWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitWeightsParams(callingConn);
   return tempPtr;
}

int InitWeights::calcWeights(pvdata_t * dataStart, int dataPatchIndex, int arborId,
                               InitWeightsParams *weightParams) {
    return PV_SUCCESS;
}

int InitWeights::initialize_base() {
   rnd_state = NULL;
   return PV_SUCCESS;
}

int InitWeights::readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * conn, double * timef/*default=NULL*/) {
   InterColComm *icComm = conn->getParent()->icCommunicator();
   int numArbors = conn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = conn->preSynapticLayer()->getLayerLoc();
   double timed;
   bool useListOfArborFiles = numArbors>1 &&
                              conn->getParent()->parameters()->value(conn->getName(), "useListOfArborFiles", false)!=0;
   bool combineWeightFiles = conn->getParent()->parameters()->value(conn->getName(), "combineWeightFiles", false)!=0;
#ifdef USE_SHMGET
   bool * shmget_owner = conn->getShmgetOwnerHead();
   bool shmget_flag = conn->getShmgetFlag();
#endif
   if( useListOfArborFiles ) {
      int arbor=0;
      PV_Stream * arborstream = pvp_open_read_file(filename, icComm);

      int rootproc = 0;
      char arborfilename[PV_PATH_MAX];
      while( arbor < conn->numberOfAxonalArborLists() ) {
         if( icComm->commRank() == rootproc ) {
            char * fgetsstatus = fgets(arborfilename, PV_PATH_MAX, arborstream->fp);
            if( fgetsstatus == NULL ) {
               bool endoffile = feof(arborstream->fp)!=0;
               if( endoffile ) {
                  fprintf(stderr, "File of arbor files \"%s\" reached end of file before all %d arbors were read.  Exiting.\n", filename, numArbors);
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
#ifndef USE_SHMGET
            int status = PV::readWeights(patches ? &patches[arbor] : NULL, &dataStart[arbor], numArbors-arbor, numPatches, arborfilename, icComm, &timed, preLoc);
#else
            int status = PV::readWeights(patches ? &patches[arbor] : NULL,
                    &dataStart[arbor], numArbors - arbor, numPatches,
                    arborfilename, icComm, &timed, preLoc, shmget_owner,
                    shmget_flag);
#endif
         if (status != PV_SUCCESS) {
            fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", arborfilename);
            exit(EXIT_FAILURE);
         }
         arbor += thisfilearbors;
      }  // while
   } // if useListOfArborFiles
   else if (combineWeightFiles){
      int rootproc = 0;
      int max_weight_files = 1;  // arbitrary limit...
      int num_weight_files = (int)conn->getParent()->parameters()->value(conn->getName(), "numWeightFiles", max_weight_files, true);
      int file_count=0;
      PV_Stream * weightstream = pvp_open_read_file(filename, icComm);
      if ((weightstream == NULL) && (icComm->commRank() == rootproc) ){
         fprintf(stderr, ""
               "Cannot open file of weight files \"%s\".  Exiting.\n", filename);
         exit(EXIT_FAILURE);
      }

      char weightsfilename[PV_PATH_MAX];
      while( file_count < num_weight_files ) {
         if( icComm->commRank() == rootproc ) {
            char * fgetsstatus = fgets(weightsfilename, PV_PATH_MAX, weightstream->fp);
            if( fgetsstatus == NULL ) {
               bool endoffile = feof(weightstream->fp)!=0;
               if( endoffile ) {
                  fprintf(stderr, "File of weight files \"%s\" reached end of file before all %d weight files were read.  Exiting.\n", filename, num_weight_files);
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
#ifndef USE_SHMGET
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, weightsfilename, icComm, &timed, preLoc);
#else
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, weightsfilename, icComm, &timed, preLoc, shmget_owner, shmget_flag);
#endif
         if (status != PV_SUCCESS) {
            fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", weightsfilename);
            exit(EXIT_FAILURE);
         }
         file_count += 1;
      } // file_count < num_weight_files

   } // if combineWeightFiles
   else {
#ifndef USE_SHMGET
      
      int status = PV::readWeights(patches, dataStart, numArbors, numPatches, filename, icComm, &timed, preLoc);
#else
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, filename, icComm, &timed, preLoc, shmget_owner, shmget_flag);
#endif
      if (status != PV_SUCCESS) {
         fprintf(stderr, "PV::readWeights: problem reading weight file %s, SHUTTING DOWN\n", filename);
         exit(EXIT_FAILURE);
      }
      if( timef != NULL ) *timef = (float) timed;
   }
   return PV_SUCCESS;
}

} /* namespace PV */
