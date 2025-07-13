/*
 * InitUniformWeights.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeights.hpp"

namespace PV {

InitUniformWeights::InitUniformWeights(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InitUniformWeights::InitUniformWeights() {}

InitUniformWeights::~InitUniformWeights() {}

void InitUniformWeights::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InitWeights::initialize(params, defaults, comm);
}

int InitUniformWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = InitWeights::ioParamsFillGroup(ioSwitch);
   ioParam_weightInit(ioSwitch);
   ioParam_connectOnlySameFeatures(ioSwitch);
   return status;
}

void InitUniformWeights::ioParam_weightInit(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "weightInit", &mWeightInit);
}

void InitUniformWeights::ioParam_connectOnlySameFeatures(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "connectOnlySameFeatures", &mConnectOnlySameFeatures);
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
