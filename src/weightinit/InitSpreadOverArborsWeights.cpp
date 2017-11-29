/*
 * InitSpreadOverArborsWeights.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeights.hpp"

namespace PV {

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights() {}

InitSpreadOverArborsWeights::~InitSpreadOverArborsWeights() {}

int InitSpreadOverArborsWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

int InitSpreadOverArborsWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitGauss2DWeights::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitSpreadOverArborsWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "weightInit", &mWeightInit, mWeightInit);
}

void InitSpreadOverArborsWeights::calcWeights(int patchIndex, int arborId) {
   calcOtherParams(patchIndex);
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   spreadOverArborsWeights(dataStart, arborId);
}

int InitSpreadOverArborsWeights::spreadOverArborsWeights(float *dataStart, int arborId) {
   int nfPatch = mWeights->getPatchSizeF();
   int nyPatch = mWeights->getPatchSizeY();
   int nxPatch = mWeights->getPatchSizeX();

   int sx = mWeights->getGeometry()->getPatchStrideX();
   int sy = mWeights->getGeometry()->getPatchStrideY();
   int sf = mWeights->getGeometry()->getPatchStrideF();

   int const nArbors = mWeights->getNumArbors();

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch; fPost++) {
      float thPost = calcThPost(fPost);
      if (checkThetaDiff(thPost))
         continue;
      for (int jPost = 0; jPost < nyPatch; jPost++) {
         float yDelta = calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            float xDelta = calcXDelta(iPost);

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            float weight = 0;
            if (xp * xp + yp * yp < 1e-4f) {
               weight = mWeightInit / nArbors;
            }
            else {
               float theta2pi = atan2f(yp, xp) / (2 * PI);
               unsigned int xpraw, ypraw, atanraw;
               union u {
                  float f;
                  unsigned int i;
               };
               union u f2u;
               f2u.f   = xp;
               xpraw   = f2u.i;
               f2u.f   = yp;
               ypraw   = f2u.i;
               f2u.f   = theta2pi;
               atanraw = f2u.i;
               if (theta2pi < 0) {
                  theta2pi += 1;
               }
               if (theta2pi >= 1) {
                  theta2pi -= 1; // theta2pi should be in the range [0,1) but roundoff could make it
                  // exactly 1
               }
               float zone = theta2pi * nArbors;

               float intpart;
               float fracpart = modff(zone, &intpart);
               assert(intpart >= 0 && intpart < nArbors && fracpart >= 0 && fracpart < 1);
               if (intpart == arborId) {
                  weight = mWeightInit * (1 - fracpart);
               }
               else if ((int)(intpart - arborId + 1) % nArbors == 0) {
                  weight = mWeightInit * fracpart;
               }
            }

            int index        = iPost * sx + jPost * sy + fPost * sf;
            dataStart[index] = weight;
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
