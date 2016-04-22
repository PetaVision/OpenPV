/*
 * initWindowed3DGaussWeights.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: kpeterson
 */

#include "InitWindowed3DGaussWeights.hpp"

namespace PV {

InitWindowed3DGaussWeights::InitWindowed3DGaussWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
}

InitWindowed3DGaussWeights::InitWindowed3DGaussWeights()
{
   initialize_base();
}

InitWindowed3DGaussWeights::~InitWindowed3DGaussWeights()
{
}

int InitWindowed3DGaussWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitWindowed3DGaussWeights::initialize(HyPerConn * conn) {
   int status = Init3DGaussWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitWindowed3DGaussWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitWindowed3DGaussWeightsParams(callingConn);
   return tempPtr;
}

int InitWindowed3DGaussWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {

   InitWindowed3DGaussWeightsParams *weightParamPtr = dynamic_cast<InitWindowed3DGaussWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);
   weightParamPtr->setTime(arborId);

   gauss3DWeights(dataStart, (Init3DGaussWeightsParams*)weightParamPtr);
   windowWeights(dataStart, weightParamPtr);

   return PV_SUCCESS;
}

/**
 * multiply the weights by a gaussian window oriented with the x-y-t axis to try
 * and get single quadrant separable receptive fields
 *
 */
int InitWindowed3DGaussWeights::windowWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitWindowed3DGaussWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float taspect=weightParamPtr->getTAspect();
   float yaspect=weightParamPtr->getYAspect();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   double r2Max=weightParamPtr->getr2Max();
   float time = (float)-weightParamPtr->getTime();
   float wShift=weightParamPtr->getWindowShift();
   float wShiftT=weightParamPtr->getWindowShiftT();

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      //TODO: add additional weight factor for difference between thPre and thPost
      if(weightParamPtr->checkThetaDiff(thPost)) continue;
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
            double d2 = xp * xp + (yaspect * (yp - wShift) * yaspect * (yp - wShift)) + (taspect * (time-wShiftT) * taspect * (time-wShiftT));

            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            if (d2 <= r2Max) {
               dataStart[index] *= expf(-d2 / (2.0f * sigma * sigma));
            }
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
