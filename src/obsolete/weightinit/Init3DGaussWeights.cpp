/*
 * Init3DGaussWeights.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "Init3DGaussWeights.hpp"
#include "Init3DGaussWeightsParams.hpp"

namespace PV {

Init3DGaussWeights::Init3DGaussWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
}

Init3DGaussWeights::Init3DGaussWeights()
{
   initialize_base();
}

Init3DGaussWeights::~Init3DGaussWeights()
{
}

int Init3DGaussWeights::initialize_base() {
   return PV_SUCCESS;
}

int Init3DGaussWeights::initialize(HyPerConn * conn) {
   int status = InitGauss2DWeights::initialize(conn);
   return status;
}

InitWeightsParams * Init3DGaussWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new Init3DGaussWeightsParams(callingConn);
   return tempPtr;
}

int Init3DGaussWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {

   Init3DGaussWeightsParams *weightParamPtr = dynamic_cast<Init3DGaussWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   weightParamPtr->calcOtherParams(patchIndex);
   weightParamPtr->setTime(arborId);

   gauss3DWeights(dataStart, weightParamPtr);

   return PV_SUCCESS;

}

/**
 * calculate temporal-spatial gaussian filter for use in optic flow detector
 */
int Init3DGaussWeights::gauss3DWeights(/* PVPatch * patch */ pvdata_t * w_tmp, Init3DGaussWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float taspect=weightParamPtr->getTAspect();
   float yaspect=weightParamPtr->getYAspect();
   float shift=weightParamPtr->getShift();
   float shiftT=weightParamPtr->getShiftT();
   int numFlanks=weightParamPtr->getNumFlanks();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   double r2Max=weightParamPtr->getr2Max();
   float time = (float)-weightParamPtr->getTime();
   float thetaXT = weightParamPtr->getThetaXT();

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

            float tp = +time * cosf(thetaXT) + yp * sinf(thetaXT);
            float ytp = -time * sinf(thetaXT) + yp * cosf(thetaXT);

            // include shift to flanks
            double d2 = xp * xp + (yaspect * (ytp - shift) * yaspect * (ytp - shift)) + (taspect * (tp-shiftT) * taspect * (tp-shiftT));

            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if (d2 <= r2Max) {
               w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (yaspect * (ytp + shift) * yaspect * (ytp + shift)) + (taspect * (tp-shiftT) * taspect * (tp-shiftT));
               if (d2 <= r2Max) {
                  w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
