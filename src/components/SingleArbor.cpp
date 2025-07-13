/*
 * SingleArbor.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "SingleArbor.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

SingleArbor::SingleArbor(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

SingleArbor::SingleArbor() {}

SingleArbor::~SingleArbor() {}

void SingleArbor::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ArborList::initialize(params, defaults, comm);
}

void SingleArbor::setObjectType() { mObjectType = "SingleArbor"; }

int SingleArbor::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return ArborList::ioParamsFillGroup(ioSwitch);
}

void SingleArbor::ioParam_numAxonalArbors(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mNumAxonalArbors = 1;
      mParamsIO->handleUnnecessaryParameter("numAxonalArbors", mNumAxonalArbors);
   }
}

} // namespace PV
