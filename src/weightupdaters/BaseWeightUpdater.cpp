/*
 * BaseWeightUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "BaseWeightUpdater.hpp"

namespace PV {

BaseWeightUpdater::BaseWeightUpdater(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void BaseWeightUpdater::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseObject::initialize(params, defaults, comm);
}

void BaseWeightUpdater::setObjectType() { mObjectType = "Updater for "; }

int BaseWeightUpdater::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_plasticityFlag(ioSwitch);
   return PV_SUCCESS;
}

void BaseWeightUpdater::ioParam_plasticityFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "plasticityFlag", &mPlasticityFlag);
}

} // namespace PV
