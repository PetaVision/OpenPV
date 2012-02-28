/*
 * initWindowed3DGaussWeights.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: kpeterson
 */

#include "InitWindowed3DGaussWeights.hpp"

namespace PV {

InitWindowed3DGaussWeights::InitWindowed3DGaussWeights()
{
   initialize_base();
}

InitWindowed3DGaussWeights::~InitWindowed3DGaussWeights()
{
   // TODO Auto-generated destructor stub
}

InitWeightsParams * InitWindowed3DGaussWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitWindowed3DGaussWeightsParams(callingConn);
   return tempPtr;
}

int InitWindowed3DGaussWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitWindowed3DGaussWeightsParams *weightParamPtr = dynamic_cast<InitWindowed3DGaussWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);
   weightParamPtr->setTime(arborId);

   gauss3DWeights(dataStart, (Init3DGaussWeightsParams*)weightParamPtr);
   windowWeights(dataStart, weightParamPtr);

   weightParamPtr->getParentConn()->setDelay(arborId, weightParamPtr->getTime());

   return PV_SUCCESS;

}

/**
 * multiply the weights by a gaussian window oriented with the x-y-t axis to try
 * and get single quadrant separable receptive fields
 *
 */
int InitWindowed3DGaussWeights::windowWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitWindowed3DGaussWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float taspect=weightParamPtr->getTAspect();
   //taspect*=2;
   float yaspect=weightParamPtr->getYAspect();
   //yaspect*=2;
   //float shift=weightParamPtr->getShift();
   //float shiftT=weightParamPtr->getShiftT();
   //int numFlanks=weightParamPtr->getNumFlanks();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   double r2Max=weightParamPtr->getR2Max();
   float time = (float)-weightParamPtr->getTime();
   float wShift=weightParamPtr->getWindowShift();
   float wShiftT=weightParamPtr->getWindowShiftT();
   //shiftT=-5; //sqrt(shift*shift+shiftT*shiftT);
   //shift=0;
   //float thetaXT = weightParamPtr->getThetaXT();
   //float strength = weightParamPtr->getStrength();

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
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            if(weightParamPtr->checkBowtieAngle(yp, xp)) continue;

            // include shift to flanks
            double d2 = xp * xp + (yaspect * (yp - wShift) * yaspect * (yp - wShift)) + (taspect * (time-wShiftT) * taspect * (time-wShiftT));

            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            //w_tmp[index] = 0;
            if (d2 <= r2Max) {
               dataStart[index] *= expf(-d2 / (2.0f * sigma * sigma));
            }
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
