/*
 * HebbianUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "HebbianUpdater.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

HebbianUpdater::HebbianUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int HebbianUpdater::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int HebbianUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseWeightUpdater::ioParamsFillGroup(ioFlag);
   return PV_SUCCESS;
}

int HebbianUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return BaseWeightUpdater::communicateInitInfo(message);
}

} // namespace PV
