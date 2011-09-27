/*
 * Init3DGaussWeights.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "Init3DGaussWeights.hpp"
#include "Init3DGaussWeightsParams.hpp"

namespace PV {

Init3DGaussWeights::Init3DGaussWeights()
{
   initialize_base();
}

Init3DGaussWeights::~Init3DGaussWeights()
{
   // TODO Auto-generated destructor stub
}

int Init3DGaussWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * Init3DGaussWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new Init3DGaussWeightsParams(callingConn);
   return tempPtr;
}

int Init3DGaussWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   Init3DGaussWeightsParams *weightParamPtr = dynamic_cast<Init3DGaussWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);
   weightParamPtr->setTime(arborId);

   gauss3DWeights(patch, weightParamPtr);

   //PVAxonalArbor * arbor = weightParamPtr->getParentConn()->axonalArbor(patchIndex, arborId);
   //arbor->delay = weightParamPtr->getTime();

   weightParamPtr->getParentConn()->setDelay(arborId, weightParamPtr->getTime());

   return PV_SUCCESS;

}

/**
 * calculate temporal-spatial gaussian filter for use in optic flow detector
 */
int Init3DGaussWeights::gauss3DWeights(PVPatch * patch, Init3DGaussWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float taspect=weightParamPtr->getTAspect();
   float yaspect=weightParamPtr->getYAspect();
   float shift=weightParamPtr->getShift();
   float shiftT=weightParamPtr->getShiftT();
   int numFlanks=weightParamPtr->getNumFlanks();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   double r2Max=weightParamPtr->getR2Max();
   float time = (float)-weightParamPtr->getTime();
   float thetaXT = weightParamPtr->getThetaXT();
   //float strength = weightParamPtr->getStrength();

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

            float tp = +time * cosf(thetaXT) + yp * sinf(thetaXT);
            float ytp = -time * sinf(thetaXT) + yp * cosf(thetaXT);

            // include shift to flanks
            double d2 = xp * xp + (yaspect * (ytp - shift) * yaspect * (ytp - shift)) + (taspect * (tp-shiftT) * taspect * (tp-shiftT));

//            if(((xDelta>-2)&&(xDelta<2))&&((yDelta>-2)&&(yDelta<2))){
//               printf("d2  %f xDelta %f, xp %f, yDelta %f, yp %f, ytp %f, tp %f\n", d2, xDelta, xp, yDelta, yp, ytp, tp);
//               printf("xp*xp %f\n", xp*xp);
//               printf("ytp*ytp %f\n", ytp*ytp);
//               printf("yaspect*yaspect*ytp*ytp %f\n", yaspect*yaspect*ytp*ytp);
//               printf("(tp-shiftT)*(tp-shiftT) %f\n", (tp-shiftT)*(tp-shiftT));
//               printf("(taspect * (tp-shiftT) * taspect * (tp-shiftT)) %f\n", (taspect * (tp-shiftT) * taspect * (tp-shiftT)));
//               printf("-d2 / (2.0f * sigma * sigma) %f\n", -d2 / (2.0f * sigma * sigma));
//               printf("expf(-d2 / (2.0f * sigma * sigma) %f\n", expf(-d2 / (2.0f * sigma * sigma)));
//            }


            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if (d2 <= r2Max) {
               w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
//               if(((xDelta>-2)&&(xDelta<2))&&((yDelta>-2)&&(yDelta<2))){
//                  printf("index %d\n", index);
//                  printf("w_tmp[index] %f\n", w_tmp[index]);
//               }
            }
//            else {
//               if(((xDelta>-5)&&(xDelta<5))&&((yDelta>-5)&&(yDelta<5))){
//                  printf("d2 is too big %f xDelta %f, xp %f, yDelta %f, yp %f, ytp %f, tp %f\n", d2, xDelta, xp, yDelta, yp, ytp, tp);
//                  printf("xp*xp %f\n", xp*xp);
//                  printf("ytp*ytp %f\n", ytp*ytp);
//                  printf("yaspect*yaspect*ytp*ytp %f\n", yaspect*yaspect*ytp*ytp);
//                  printf("(tp-shiftT)*(tp-shiftT) %f\n", (tp-shiftT)*(tp-shiftT));
//                  printf("(taspect * (tp-shiftT) * taspect * (tp-shiftT)) %f\n", (taspect * (tp-shiftT) * taspect * (tp-shiftT)));
//               }
//            }
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
