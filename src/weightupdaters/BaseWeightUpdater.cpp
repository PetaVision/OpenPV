/*
 * BaseWeightUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "BaseWeightUpdater.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"
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

int BaseWeightUpdater::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   pvAssert(mConnectionData == nullptr);
   mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   pvAssert(mConnectionData != nullptr);

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }
   return PV_SUCCESS;
}

} // namespace PV
