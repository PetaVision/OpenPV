/*
 * InitGaborWeights.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaborWeights.hpp"
#include <connections/weight_conversions.hpp>

namespace PV {

InitGaborWeights::InitGaborWeights(char const *name, HyPerCol *hc) { initialize(name, hc); }

InitGaborWeights::InitGaborWeights() {}

InitGaborWeights::~InitGaborWeights() {}

int InitGaborWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

int InitGaborWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeights::ioParamsFillGroup(ioFlag);
   ioParam_lambda(ioFlag);
   ioParam_phi(ioFlag);
   ioParam_invert(ioFlag);
   return status;
}

void InitGaborWeights::ioParam_lambda(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "lambda", &mLambda, mLambda);
}

void InitGaborWeights::ioParam_phi(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "phi", &mPhi, mPhi);
}

void InitGaborWeights::ioParam_invert(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "invert", &mInvert, mInvert);
}

void InitGaborWeights::calcOtherParams(int patchIndex) {
   const int kfPre = kernelIndexCalculations(patchIndex);
   calculateThetas(kfPre, patchIndex);
}

void InitGaborWeights::calcWeights(int patchIndex, int arborId) {
   calcOtherParams(patchIndex);
   gaborWeights(mWeights->getDataFromDataIndex(arborId, patchIndex));
}

void InitGaborWeights::gaborWeights(float *dataStart) {
   // load necessary params:
   int nfPatch = mWeights->getPatchSizeF();
   int nyPatch = mWeights->getPatchSizeY();
   int nxPatch = mWeights->getPatchSizeX();
   int sx      = mWeights->getGeometry()->getPatchStrideX();
   int sy      = mWeights->getGeometry()->getPatchStrideY();
   int sf      = mWeights->getGeometry()->getPatchStrideF();

   float wMin = mWeights->calcMinWeight();
   float wMax = mWeights->calcMaxWeight();

   bool const compress = (sizeof(float) == sizeof(unsigned char));

   for (int fPost = 0; fPost < nfPatch; fPost++) {
      float thPost = calcThPost(fPost);
      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = calcXDelta(iPost);

            // rotate the reference frame by th ((x,y) is center of patch (0,0))
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            float factor = cosf(2.0f * PI * yp / mLambda + mPhi);
            if (fabsf(yp / mLambda) > 3.0f / 4.0f)
               factor = 0.0f; // phase < 3*PI/2 (no second positive band)

            float d2  = xp * xp + (mAspect * (yp - mFlankShift) * mAspect * (yp - mFlankShift));
            float wt  = factor * expf(-d2 / (2.0f * mSigma * mSigma));
            int index = iPost * sx + jPost * sy + fPost * sf;

            if (xDelta * xDelta + yDelta * yDelta > mRMaxSquared) {
               dataStart[index] = (float)0;
            }
            else {
               if (mInvert)
                  wt *= -1.0f;
               if (wt < 0.0f)
                  wt = 0.0f; // clip negative values
               if (compress)
                  dataStart[index] = compressWeight(wt, wMin, wMax);
               else
                  dataStart[index] = wt;
            }
         }
      }
   }
}

} /* namespace PV */
