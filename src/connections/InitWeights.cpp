/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#include "../io/ConnectionProbe.hpp"
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

//InitWeights::InitWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//      ChannelType channel)
//{
//   initialize_base();
//   initialize(name, hc, pre, post, channel);
//}

InitWeights::~InitWeights()
{
   // TODO Auto-generated destructor stub
}


/*This method does the three steps involved in initializing weights.  Subclasses shouldn't touch this method.
 * Subclasses should only generate their own versions of calcWeights to do their own type of weight initialization.
 *
 * This method first calls XXX to create an unshrunk patch.  Then it calls calcWeights to initialize
 * the weights for that unshrunk patch.  Finally it copies the weights back to the original, possibly shrunk patch.
 */
PVPatch ** InitWeights::initializeWeights(PVPatch ** patches, int arborId, int numPatches, const char * filename, HyPerConn * callingConn) {

   //parentConn = callingConn;
   PVParams * inputParams = callingConn->getParent()->parameters();

   int initFromLastFlag = inputParams->value(callingConn->getName(), "initFromLastFlag", 0.0f, false) != 0;

   InitWeightsParams *weightParams = NULL;

   if (initFromLastFlag) {
      char nametmp[PV_PATH_MAX];

      if(callingConn->numberOfAxonalArborLists()>1)
         snprintf(nametmp, PV_PATH_MAX-1, "%s/w%1.1d_a%1.1d_last.pvp", callingConn->getParent()->getOutputPath(), callingConn->getConnectionId(), arborId);
      else
         snprintf(nametmp, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", callingConn->getParent()->getOutputPath(), callingConn->getConnectionId());
      readWeights(patches, numPatches, nametmp, callingConn);
      //weightInitializer->initializeWeights(arbors[arborId], numPatches, nametmp, this);
   }
   else if( filename != NULL ) {
      //check status?
      readWeights(patches, numPatches, filename, callingConn);
   }
   else {
      weightParams = createNewWeightParams(callingConn);
      //allocate unshrunk patch method
      for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {

         int correctedPatchIndex = callingConn->correctPIndex(patchIndex);
         //int correctedPatchIndex = patchIndex;
         //create full sized patch:
         PVPatch * wp_tmp = createUnShrunkenPatch(callingConn, patches[patchIndex]);
         if(wp_tmp==NULL) continue;

         //calc weights for patch:
         int successFlag = calcWeights(wp_tmp, correctedPatchIndex, arborId, weightParams);
         if (successFlag != PV_SUCCESS) {
            fprintf(stderr, "Failed to create weights for %s! Exiting...\n", callingConn->getName());
            exit(PV_FAILURE);
         }

         //copy back to unshrunk patch:
         copyToOriginalPatch(patches[patchIndex], wp_tmp);

      }

      delete(weightParams);
   }

   return patches;

}

InitWeightsParams * InitWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGauss2DWeightsParams(callingConn);
   return tempPtr;
}

int InitWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                               InitWeightsParams *weightParams) {

    InitGauss2DWeightsParams *weightParamPtr = dynamic_cast<InitGauss2DWeightsParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(patch, patchIndex);

    //calculate the weights:
    gauss2DCalcWeights(patch, weightParamPtr);


    return PV_SUCCESS;
}

int InitWeights::initialize_base() {

   return PV_SUCCESS;
}

//int InitWeights::initialize(const char * name, HyPerCol * hc,
//               HyPerLayer * pre, HyPerLayer * post,
//               ChannelType channel) {
//   int status = PV_SUCCESS;
//
//   this->parent = hc;
//   this->pre = pre;
//   this->post = post;
//   this->channel = channel;
//
//   this->name = strdup(name);
//
//   return status;
//
//}
//
int InitWeights::readWeights(PVPatch ** patches, int numPatches, const char * filename, HyPerConn * callingConn)
{
   HyPerCol *parent = callingConn->getParent();
   HyPerLayer *pre = callingConn->getPre();
   double time;
   int status = PV::readWeights(patches, numPatches, filename, parent->icCommunicator(),
                                &time, pre->getLayerLoc(), true);

   if (status != PV_SUCCESS) {
      fprintf(stderr, "PV::HyPerConn::readWeights: problem reading weight file %s, SHUTTING DOWN\n", filename);
      exit(1);
   }

   return 1;
}

/*
 * Create a full sized patch for a potentially shrunken patch. This full sized patch will be used for initializing weights and will later be copied back to
 * the original.
 *
 */
PVPatch * InitWeights::createUnShrunkenPatch(HyPerConn * callingConn, PVPatch * wp) {
   // get dimensions of (potentially shrunken patch)
   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }


   // get strides of (potentially shrunken) patch
   const int sx = wp->sx;
   assert(sx == nfPatch);
   //const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sf = wp->sf;
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
int InitWeights::copyToOriginalPatch(PVPatch * wp, PVPatch * wp_tmp) {
   // copy weights from full sized temporary patch to (possibly shrunken) patch
   pvdata_t * w = wp->data;
   const int nxPatch = wp->nx;
   const int nyPatch = wp->ny;
   const int nfPatch = wp->nf;

   const int sy = wp->sy; // no assert here because patch may be shrunken
   const int sy_tmp = wp_tmp->sy;



   pvdata_t * data_head =  (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset = w - data_head;
   pvdata_t * w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }

   free(wp_tmp);

   return 1;
}

/**
 * calculate gaussian weights between oriented line segments
 */
int InitWeights::gauss2DCalcWeights(PVPatch * patch, InitGauss2DWeightsParams * weightParamPtr) {



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

   pvdata_t * w_tmp = patch->data;



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
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            if(weightParamPtr->checkBowtieAngle(yp, xp)) continue;


            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if (d2 <= r2Max) {
               w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if (d2 <= r2Max) {
                  w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   return 0;
}



} /* namespace PV */
