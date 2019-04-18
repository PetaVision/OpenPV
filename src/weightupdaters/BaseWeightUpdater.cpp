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

void BaseWeightUpdater::setObjectType() { mObjectType = "Updater for "; }

int BaseWeightUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_plasticityFlag(ioFlag);
   return PV_SUCCESS;
}

void BaseWeightUpdater::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "plasticityFlag", &mPlasticityFlag, mPlasticityFlag /*default value*/);
}

} // namespace PV
