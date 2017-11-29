/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: Pete Schultz
 */

#include "NormalizeBase.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

NormalizeBase::NormalizeBase(char const *name, HyPerCol *hc) { initialize(name, hc); }

int NormalizeBase::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int NormalizeBase::setDescription() {
   description.clear();
   description.append("Weight normalizer \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int NormalizeBase::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   ioParam_normalizeArborsIndividually(ioFlag);
   ioParam_normalizeOnInitialize(ioFlag);
   ioParam_normalizeOnWeightUpdate(ioFlag);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_strength(enum ParamsIOFlag ioFlag) {
   //
}

void NormalizeBase::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   //
}

void NormalizeBase::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {
   //
}

void NormalizeBase::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {
   //
}

void NormalizeBase::normalizeWeightsIfNeeded() {
   bool needUpdate = false;
   double simTime  = parent->simulationTime();
   if (mNormalizeOnInitialize and simTime == parent->getStartTime()) {
      needUpdate = true;
   }
   else if (mNormalizeOnWeightUpdate and weightsHaveUpdated()) {
      needUpdate = true;
   }
   if (needUpdate) {
      normalizeWeights();
   }
}

} // namespace PV
