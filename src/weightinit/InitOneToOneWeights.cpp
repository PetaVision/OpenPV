/*
 * InitOneToOneWeights.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeights.hpp"

namespace PV {

InitOneToOneWeights::InitOneToOneWeights(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InitOneToOneWeights::InitOneToOneWeights() {}

InitOneToOneWeights::~InitOneToOneWeights() {}

void InitOneToOneWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InitWeights::initialize(name, params, comm);
}

int InitOneToOneWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeights::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitOneToOneWeights::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "weightInit", &mWeightInit, mWeightInit);
}

void InitOneToOneWeights::calcWeights(long patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   createOneToOneConnection(dataStart, patchIndex, mWeightInit);
}

int InitOneToOneWeights::createOneToOneConnection(
      float *dataStart,
      long dataPatchIndex,
      float weightInit) {

   int unitCellIndex = dataIndexToUnitCellIndex(dataPatchIndex);

   int nfp = mWeights->getPatchSizeF();
   int nxp = mWeights->getPatchSizeX();
   int nyp = mWeights->getPatchSizeY();

   int sxp = mWeights->getGeometry()->getPatchStrideX();
   int syp = mWeights->getGeometry()->getPatchStrideY();
   int sfp = mWeights->getGeometry()->getPatchStrideF();

   // clear all weights in patch
   std::size_t patchSizeOverall = static_cast<std::size_t>(mWeights->getPatchSizeOverall());
   memset(dataStart, 0, patchSizeOverall * sizeof(*dataStart));
   // then set the center point of the patch for each feature
   int x = (int)(nxp / 2);
   int y = (int)(nyp / 2);
   for (int f = 0; f < nfp; f++) {
      long index = (long)x * sxp + (long)y * syp + (long)f * sfp;
      dataStart[index] = f == unitCellIndex ? weightInit : 0;
   }

   return PV_SUCCESS;
}

} /* namespace PV */
