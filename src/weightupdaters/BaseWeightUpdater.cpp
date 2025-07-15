/*
 * BaseWeightUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "BaseWeightUpdater.hpp"

namespace PV {

BaseWeightUpdater::BaseWeightUpdater(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

void BaseWeightUpdater::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
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
