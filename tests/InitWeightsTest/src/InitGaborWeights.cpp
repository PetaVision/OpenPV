/*
 * InitGaborWeights.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaborWeights.hpp"
#include "InitGaborWeightsParams.hpp"
#include <connections/weight_conversions.hpp>

namespace PV {

InitGaborWeights::InitGaborWeights(char const * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitGaborWeights::InitGaborWeights()
{
   initialize_base();
}

InitGaborWeights::~InitGaborWeights()
{
}

int InitGaborWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitGaborWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitGaborWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitGaborWeightsParams(name, parent);
   return tempPtr;
}

int InitGaborWeights::calcWeights(pvwdata_t * dataStart, int patchIndex, int arborId) {

   InitGaborWeightsParams *weightParamPtr = dynamic_cast<InitGaborWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   weightParamPtr->calcOtherParams(patchIndex);

   gaborWeights(dataStart, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitGaborWeights::gaborWeights(pvwdata_t * dataStart, InitGaborWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float aspect=weightParamPtr->getAspect();
   float shift=weightParamPtr->getShift();
   float lambda=weightParamPtr->getlambda();
   float phi=weightParamPtr->getphi();
   bool invert=weightParamPtr->getinvert();
   //int numFlanks=weightParamPtr->getnumFlanks();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   double r2Max=weightParamPtr->getr2Max();

   float wMin = weightParamPtr->getWMin();
   float wMax = weightParamPtr->getWMax();

   //2014.6.19:Rasmussen - const should allow the compiler to optimize the
   //  if (compress) ... statement below.
   const bool compress = (sizeof(pvwdata_t) == sizeof(unsigned char));

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
               dataStart[index] = (pvwdata_t) 0;
            }
            else {
               if (invert) wt *= -1.0f;
               if (wt < 0.0f) wt = 0.0f;       // clip negative values
               if (compress) dataStart[index] = compressWeight(wt, wMin, wMax);
               else          dataStart[index] = wt;
            }
         }
      }
   }

   return 0;
}

BaseObject * createInitGaborWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitGaborWeights(name, hc) : NULL;
}

} /* namespace PV */
