/*
 * InitOneToOneWeightsWithDelays.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#include "InitOneToOneWeightsWithDelays.hpp"

namespace PV {

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays() { initialize_base(); }

InitOneToOneWeightsWithDelays::~InitOneToOneWeightsWithDelays() {}

int InitOneToOneWeightsWithDelays::initialize_base() { return PV_SUCCESS; }

int InitOneToOneWeightsWithDelays::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitOneToOneWeightsWithDelays::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitOneToOneWeightsWithDelays::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, getName(), "weightInit", &mWeightInit, mWeightInit);
}

void InitOneToOneWeightsWithDelays::calcWeights(float *dataStart, int patchIndex, int arborId) {
   createOneToOneConnectionWithDelays(dataStart, patchIndex, mWeightInit, arborId);
}

void InitOneToOneWeightsWithDelays::createOneToOneConnectionWithDelays(
      float *dataStart,
      int dataPatchIndex,
      float iWeight,
      int arborId) {

   const int nArbors = mCallingConn->numberOfAxonalArborLists();
   int k             = mCallingConn->dataIndexToUnitCellIndex(dataPatchIndex);

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
      dataStart[x * sxp + y * syp + f * sfp] = f == nArbors * k + arborId ? iWeight : 0;
   }
}

} /* namespace PV */
