/*
 * InitOneToOneWeights.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeights.hpp"

namespace PV {

InitOneToOneWeights::InitOneToOneWeights(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InitOneToOneWeights::InitOneToOneWeights() {}

InitOneToOneWeights::~InitOneToOneWeights() {}

void InitOneToOneWeights::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InitWeights::initialize(params, defaults, comm);
}

int InitOneToOneWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitWeights::ioParamsFillGroup(ioSwitch);
   ioParam_weightInit(ioSwitch);
   return status;
}

void InitOneToOneWeights::ioParam_weightInit(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "weightInit", &mWeightInit);
}

void InitOneToOneWeights::calcWeights(int patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   createOneToOneConnection(dataStart, patchIndex, mWeightInit);
}

int InitOneToOneWeights::createOneToOneConnection(
      float *dataStart,
      int dataPatchIndex,
      float weightInit) {

   int unitCellIndex = dataIndexToUnitCellIndex(dataPatchIndex);

   int nfp = mWeights->getPatchSizeF();
   int nxp = mWeights->getPatchSizeX();
   int nyp = mWeights->getPatchSizeY();

   int sxp = mWeights->getGeometry()->getPatchStrideX();
   int syp = mWeights->getGeometry()->getPatchStrideY();
   int sfp = mWeights->getGeometry()->getPatchStrideF();

   // clear all weights in patch
   memset(dataStart, 0, nxp * nyp * nfp);
   // then set the center point of the patch for each feature
   int x = (int)(nxp / 2);
   int y = (int)(nyp / 2);
   for (int f = 0; f < nfp; f++) {
      dataStart[x * sxp + y * syp + f * sfp] = f == unitCellIndex ? weightInit : 0;
   }

   return PV_SUCCESS;
}

} /* namespace PV */
