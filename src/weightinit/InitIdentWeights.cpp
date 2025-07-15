/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

InitIdentWeights::InitIdentWeights() {}

InitIdentWeights::~InitIdentWeights() {}

void InitIdentWeights::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InitOneToOneWeights::initialize(paramsIO, comm);
}

void InitIdentWeights::ioParam_weightInit(ParamsIOSwitch ioSwitch) {
   mWeightInit = 1.0f;
   mParamsIO->handleUnnecessaryParameter("weightInit", 1.0f);
}

} /* namespace PV */
