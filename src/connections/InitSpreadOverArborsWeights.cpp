/*
 * InitSpreadOverArborsWeights.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeights.hpp"
#include "InitSpreadOverArborsWeightsParams.hpp"

namespace PV {

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights()
{
   initialize_base();
}

InitSpreadOverArborsWeights::~InitSpreadOverArborsWeights()
{
   // TODO Auto-generated destructor stub
}

int InitSpreadOverArborsWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitSpreadOverArborsWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitSpreadOverArborsWeightsParams(callingConn);
   return tempPtr;
}

int InitSpreadOverArborsWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {
   InitSpreadOverArborsWeightsParams *weightParamPtr = dynamic_cast<InitSpreadOverArborsWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   weightParamPtr->calcOtherParams(patch, patchIndex);



   spreadOverArborsWeights(patch, arborId, weightParamPtr);
   return 1;
}

/**
 * Initializes all weights to iWeight
 *
 */
int InitSpreadOverArborsWeights::spreadOverArborsWeights(PVPatch * patch, int arborId,
      InitSpreadOverArborsWeightsParams * weightParamPtr) {



   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();

   const float iWeight = weightParamPtr->getInitWeight();
   const int nArbors = weightParamPtr->getNumArbors();


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


            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);



            float weight = 0;
            float theta;
            if(xp==0) {
               if(yp<0)
                  theta = 3.0f * PI/2.0f;
               else if(yp>0)
                  theta = PI/2.0f;
               else
                  theta = 0;
            }
            else
               theta = atan2f(yp,xp);
            if(theta<0) theta+=2.0f*PI;

            if(((arborId)*(2*PI/nArbors) <= theta)&&
                  (((arborId+1)*(2.0f*PI/nArbors) > theta)))
               weight = iWeight;

            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = weight;
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
