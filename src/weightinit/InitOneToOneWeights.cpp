/*
 * InitOneToOneWeights.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeights.hpp"

namespace PV {

InitOneToOneWeights::InitOneToOneWeights(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeights::InitOneToOneWeights() { initialize_base(); }

InitOneToOneWeights::~InitOneToOneWeights() {}

int InitOneToOneWeights::initialize_base() { return PV_SUCCESS; }

int InitOneToOneWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitOneToOneWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitOneToOneWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "weightInit", &mWeightInit, mWeightInit);
}

void InitOneToOneWeights::calcWeights(float *dataStart, int patchIndex, int arborId) {
   createOneToOneConnection(dataStart, patchIndex, mWeightInit);
}

int InitOneToOneWeights::createOneToOneConnection(
      float *dataStart,
      int dataPatchIndex,
      float iWeight) {

   int k = mCallingConn->dataIndexToUnitCellIndex(dataPatchIndex);

   const int nfp = mCallingConn->fPatchSize();
   const int nxp = mCallingConn->xPatchSize();
   const int nyp = mCallingConn->yPatchSize();

   const int sxp = mCallingConn->xPatchStride();
   const int syp = mCallingConn->yPatchStride();
   const int sfp = mCallingConn->fPatchStride();

   // clear all weights in patch
   memset(dataStart, 0, nxp * nyp * nfp);
   // then set the center point of the patch for each feature
   int x = (int)(nxp / 2);
   int y = (int)(nyp / 2);
   for (int f = 0; f < nfp; f++) {
      dataStart[x * sxp + y * syp + f * sfp] = f == k ? mWeightInit : 0;
   }

   return PV_SUCCESS;
}

} /* namespace PV */
