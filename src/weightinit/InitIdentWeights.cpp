/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InitIdentWeights::InitIdentWeights() {}

InitIdentWeights::~InitIdentWeights() {}

void InitIdentWeights::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InitOneToOneWeights::initialize(params, defaults, comm);
}

void InitIdentWeights::ioParam_weightInit(ParamsIOSwitch ioSwitch) {
   mWeightInit = 1.0f;
   mParamsIO->handleUnnecessaryParameter("weightInit", 1.0f);
}

} /* namespace PV */
