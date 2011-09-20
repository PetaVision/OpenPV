/*
 * InitGaborWeights.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaborWeights.hpp"
#include "InitGaborWeightsParams.hpp"

namespace PV {

InitGaborWeights::InitGaborWeights()
{
   initialize_base();
}

InitGaborWeights::~InitGaborWeights()
{
   // TODO Auto-generated destructor stub
}

int InitGaborWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitGaborWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGaborWeightsParams(callingConn);
   return tempPtr;
}

int InitGaborWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitGaborWeightsParams *weightParamPtr = dynamic_cast<InitGaborWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   gaborWeights(patch, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitGaborWeights::gaborWeights(PVPatch * patch, InitGaborWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float aspect=weightParamPtr->getaspect();
   float shift=weightParamPtr->getshift();
   float lambda=weightParamPtr->getlambda();
   float phi=weightParamPtr->getphi();
   bool invert=weightParamPtr->getinvert();
   //int numFlanks=weightParamPtr->getnumFlanks();
   float sigma=weightParamPtr->getsigma();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   double r2Max=weightParamPtr->getr2Max();

   pvdata_t * w_tmp = patch->data;


   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
        for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);

            // rotate the reference frame by th ((x,y) is center of patch (0,0))
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            float factor = cos(2.0f*PI*yp/lambda + phi);
            if (fabs(yp/lambda) > 3.0f/4.0f) factor = 0.0f;  // phase < 3*PI/2 (no second positive band)

            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            float wt = factor * expf(-d2 / (2.0f*sigma*sigma));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;

            if (xDelta*xDelta + yDelta*yDelta > r2Max) {
               w_tmp[index] = 0.0f;
            }
            else {
               if (invert) wt *= -1.0f;
               if (wt < 0.0f) wt = 0.0f;       // clip negative values
               w_tmp[index] = wt;
            }


         }
      }
   }


   return 0;
}

} /* namespace PV */
