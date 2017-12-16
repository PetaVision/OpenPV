/*
 * BaseWeightUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "BaseWeightUpdater.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

BaseWeightUpdater::BaseWeightUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int BaseWeightUpdater::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int BaseWeightUpdater::setDescription() {
   description.clear();
   description.append("Weight Updater").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int BaseWeightUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_plasticityFlag(ioFlag);
   return PV_SUCCESS;
}

void BaseWeightUpdater::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "plasticityFlag", &mPlasticityFlag, mPlasticityFlag /*default value*/);
}

} // namespace PV
