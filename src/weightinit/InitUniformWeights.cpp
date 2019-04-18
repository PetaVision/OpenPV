/*
 * InitUniformWeights.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeights.hpp"

namespace PV {

InitUniformWeights::InitUniformWeights(char const *name, HyPerCol *hc) { initialize(name, hc); }

InitUniformWeights::InitUniformWeights() {}

InitUniformWeights::~InitUniformWeights() {}

int InitUniformWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitUniformWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   ioParam_connectOnlySameFeatures(ioFlag);
   return status;
}

void InitUniformWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "weightInit", &mWeightInit, mWeightInit);
}

void InitUniformWeights::ioParam_connectOnlySameFeatures(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "connectOnlySameFeatures",
         &mConnectOnlySameFeatures,
         mConnectOnlySameFeatures);
}

void InitUniformWeights::calcWeights(int patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   const int nfp    = mWeights->getPatchSizeF();
   const int kf     = patchIndex % nfp;

   uniformWeights(dataStart, mWeightInit, kf, mConnectOnlySameFeatures);
}

void InitUniformWeights::uniformWeights(
      float *dataStart,
      float weightInit,
      int kf,
      bool connectOnlySameFeatures) {
   const int nxp = mWeights->getPatchSizeX();
   const int nyp = mWeights->getPatchSizeY();
   const int nfp = mWeights->getPatchSizeF();

   const int sxp = mWeights->getGeometry()->getPatchStrideX();
   const int syp = mWeights->getGeometry()->getPatchStrideY();
   const int sfp = mWeights->getGeometry()->getPatchStrideF();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            if ((connectOnlySameFeatures) and (kf != f)) {
               dataStart[x * sxp + y * syp + f * sfp] = 0;
            }
            else {
               dataStart[x * sxp + y * syp + f * sfp] = weightInit;
            }
         }
      }
   }
}

} /* end namespace PV */
