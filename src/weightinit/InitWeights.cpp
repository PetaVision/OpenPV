/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Replaced by PatchProbe.
#include "../io/ConnectionProbe.hpp"
#endif // OBSOLETE
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include "../columns/InterColComm.hpp"
#include "InitGauss2DWeightsParams.hpp"


namespace PV {

InitWeights::InitWeights()
{
   initialize_base();
}

InitWeights::~InitWeights()
{
   // TODO Auto-generated destructor stub
}

/*This method does the three steps involved in initializing weights.  Subclasses shouldn't touch this method.
 * Subclasses should only generate their own versions of calcWeights to do their own type of weight initialization.
 *
 * This method first calls XXX to create an unshrunken patch.  Then it calls calcWeights to initialize
 * the weights for that unshrunken patch.  Finally it copies the weights back to the original, possibly shrunk patch.
 */
PVPatch *** InitWeights::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * callingConn, float * timef /*default NULL*/) {
//void InitWeights::initializeWeights(const char * filename, HyPerConn * callingConn, float * timef /*default NULL*/) {
   PVParams * inputParams = callingConn->getParent()->parameters();
   int initFromLastFlag = inputParams->value(callingConn->getName(), "initFromLastFlag", 0.0f, false) != 0;
   InitWeightsParams *weightParams = NULL;
   int numArbors = callingConn->numberOfAxonalArborLists();
   if( initFromLastFlag ) {
      char nametmp[PV_PATH_MAX];
      snprintf(nametmp, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", callingConn->getParent()->getOutputPath(), callingConn->getConnectionId());
      readWeights(patches, dataStart, callingConn->getNumDataPatches(), nametmp, callingConn);
   }
   else if( filename != NULL ) {
      readWeights(patches, dataStart, callingConn->getNumDataPatches(), filename, callingConn, timef);
   }
   else {
      // int patchsize = callingConn->xPatchSize() * callingConn->yPatchSize() * callingConn->fPatchSize();
      weightParams = createNewWeightParams(callingConn);

//      int nfp = weightParams->getnfPatch_tmp();
//      int nxp = weightParams->getnxPatch_tmp();
//      int nyp = weightParams->getnyPatch_tmp();
      //int patchSize = nfp*nxp*nyp;

      for( int arbor=0; arbor<numArbors; arbor++ ) {
         //for (int patchIndex = 0; patchIndex < callingConn->numDataPatches(); patchIndex++) {
         for (int dataPatchIndex = 0; dataPatchIndex < callingConn->getNumDataPatches(); dataPatchIndex++) {

            //int correctedPatchIndex = callingConn->correctPIndex(patchIndex);
            //int correctedPatchIndex = patchIndex;
            //create full sized patch:
            // PVPatch * wp_tmp = createUnShrunkenPatch(callingConn, patches[arbor][patchIndex]);
            // if(wp_tmp==NULL) continue;

            //calc weights for patch:

            //int successFlag = calcWeights(get_wDataHead[arbor]+patchIndex*patchSize, patchIndex /*correctedPatchIndex*/, arbor, weightParams);
            //int successFlag = calcWeights(callingConn->get_wDataHead(arbor, patchIndex), patchIndex /*correctedPatchIndex*/, arbor, weightParams);
            int successFlag = calcWeights(callingConn->get_wDataHead(arbor, dataPatchIndex), dataPatchIndex, arbor, weightParams);
            if (successFlag != PV_SUCCESS) {
               fprintf(stderr, "Failed to create weights for %s! Exiting...\n", callingConn->getName());
               exit(PV_FAILURE);
            }

            //copy back to unshrunk patch:
            //copyToOriginalPatch(patches[arbor][patchIndex], wp_tmp);
            //copyToOriginalPatch(patches[arbor][patchIndex], wp_tmp,
            //      callingConn->get_wDataStart(arbor), patchIndex,
            //      callingConn->fPatchSize(), callingConn->yPatchStride());
            //free(wp_tmp);
         }
      }
      delete(weightParams);
   }
   //return callingConn->weights();
}

InitWeightsParams * InitWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGauss2DWeightsParams(callingConn);
   return tempPtr;
}

int InitWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int dataPatchIndex, int arborId,
                               InitWeightsParams *weightParams) {

    InitGauss2DWeightsParams *weightParamPtr = dynamic_cast<InitGauss2DWeightsParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    gauss2DCalcWeights(dataStart, weightParamPtr);


    return PV_SUCCESS;
}

int InitWeights::initialize_base() {

   return PV_SUCCESS;
}

int InitWeights::readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * conn, float * timef/*default=NULL*/) {
   InterColComm *icComm = conn->getParent()->icCommunicator();
   int numArbors = conn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = conn->preSynapticLayer()->getLayerLoc();
   double timed;
   bool useListOfArborFiles = numArbors>1 &&
                              conn->getParent()->parameters()->value(conn->getName(), "useListOfArborFiles", false)!=0;
   if( useListOfArborFiles ) {
      int arbor=0;
      FILE * arborfp = pvp_open_read_file(filename, icComm);

      int rootproc = 0;
      char arborfilename[PV_PATH_MAX];
      while( arbor < conn->numberOfAxonalArborLists() ) {
         if( icComm->commRank() == rootproc ) {
            char * fgetsstatus = fgets(arborfilename, PV_PATH_MAX, arborfp);
            if( fgetsstatus == NULL ) {
               bool endoffile = feof(arborfp)!=0;
               if( endoffile ) {
                  fprintf(stderr, "File of arbor files \"%s\" reached end of file before all %d arbors were read.  Exiting.\n", filename, numArbors);
                  exit(EXIT_FAILURE);
               }
               else {
                  int error = ferror(arborfp);
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
         }
         int filetype, datatype;
         int numParams = NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS;
         int params[NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS];
         pvp_read_header(arborfilename, icComm, &timed, &filetype, &datatype, params, &numParams);
         int thisfilearbors = params[INDEX_NBANDS];
         int status = PV::readWeights(&patches[arbor], &dataStart[arbor], numArbors-arbor, numPatches, arborfilename, icComm, &timed, preLoc);
         if (status != PV_SUCCESS) {
            fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", arborfilename);
            exit(EXIT_FAILURE);
         }
         arbor += thisfilearbors;
      }
   }
   else {
      int status = PV::readWeights(patches, dataStart, numArbors, numPatches, filename, icComm, &timed, preLoc);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "PV::HyPerConn::readWeights: problem reading weight file %s, SHUTTING DOWN\n", filename);
         exit(EXIT_FAILURE);
      }
      if( timef != NULL ) *timef = (float) timed;
   }
   return PV_SUCCESS;
}


#ifdef OBSOLETE // Marked obsolete Feb 27, 2012.  With refactoring of wDataStart, there is no need to initialize on an unshrunken patch and copy to a shrunken patch
/*
 * Create a full sized patch for a potentially shrunken patch. This full sized patch will be used for initializing weights and will later be copied back to
 * the original.
 *
 */
PVPatch * InitWeights::createUnShrunkenPatch(HyPerConn * callingConn, PVPatch * wp) {
   // get dimensions of (potentially shrunken patch)
   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = callingConn->fPatchSize(); //wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }


   // get strides of (potentially shrunken) patch
   const int sx = callingConn->xPatchStride(); //wp->sx;
   assert(sx == nfPatch);
   //const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sf = callingConn->fPatchStride(); //wp->sf;
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   PVPatch * wp_tmp;
   wp_tmp = pvpatch_inplace_new(callingConn->xPatchSize(), callingConn->yPatchSize(), callingConn->fPatchSize());

   return wp_tmp;
}

/*
 * Copy from full sized patch back to potentially shrunken patch.
 *
 */
int InitWeights::copyToOriginalPatch(PVPatch * wp, PVPatch * wp_tmp, pvdata_t * wtop, int patchIndex, int nf_patch, int sy_patch) {
   // copy weights from full sized temporary patch to (possibly shrunken) patch
   pvdata_t * w = wp->data;
   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = nf_patch; //parentConn->fPatchSize(); //wp->nf;

   const int sy = sy_patch; //parentConn->yPatchStride(); //wp->sy; // no assert here because patch may be shrunken
   const int sy_tmp = sy; //wp_tmp->sy;


   const int nxunshrunkPatch = wp_tmp->nx;
   const int nyunshrunkPatch = wp_tmp->ny;
   const int nfunshrunkPatch = nf_patch; //parentConn->fPatchSize(); //wp_tmp->nf;
   const int unshrunkPatchSize = nxunshrunkPatch*nyunshrunkPatch*nfunshrunkPatch;
   pvdata_t * data_head1 = &wtop[unshrunkPatchSize*patchIndex]; // (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   //pvdata_t * data_head2 = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset1 = w - data_head1;
   //size_t data_offset2 = w - data_head2;
   //size_t data_offset = fabs(data_offset1) < fabs(data_offset2) ? data_offset1 : data_offset2;
   pvdata_t * w_tmp = &wp_tmp->data[data_offset1];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }

   //free(wp_tmp);

   return 1;
}
#endif // OBSOLETE

/**
 * calculate gaussian weights between oriented line segments
 */
int InitWeights::gauss2DCalcWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitGauss2DWeightsParams * weightParamPtr) {



   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float aspect=weightParamPtr->getaspect();
   float shift=weightParamPtr->getshift();
   int numFlanks=weightParamPtr->getnumFlanks();
   float sigma=weightParamPtr->getsigma();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   double r2Max=weightParamPtr->getr2Max();
   double r2Min=weightParamPtr->getr2Min();

   // pvdata_t * w_tmp = patch->data;



   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      //TODO: add additional weight factor for difference between thPre and thPost
      if(weightParamPtr->checkTheta(thPost)) continue;
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);

            if(weightParamPtr->isSameLocOrSelf(xDelta, yDelta, fPost)) continue;

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cos(thPost) + yDelta * sin(thPost);
            float yp = -xDelta * sin(thPost) + yDelta * cos(thPost);

            if(weightParamPtr->checkBowtieAngle(yp, xp)) continue;


            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            dataStart[index] = 0;
            if ((d2 <= r2Max) && (d2 >= r2Min)) {
               dataStart[index] += exp(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if ((d2 <= r2Max) && (d2 >= r2Min)) {
                  dataStart[index] += exp(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   return 0;
}



} /* namespace PV */
