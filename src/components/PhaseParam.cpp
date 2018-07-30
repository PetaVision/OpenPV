/*
 * PhaseParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "PhaseParam.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PhaseParam::PhaseParam(char const *name, HyPerCol *hc) { initialize(name, hc); }

PhaseParam::~PhaseParam() {}

int PhaseParam::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void PhaseParam::setObjectType() { mObjectType = "PhaseParam"; }

void PhaseParam::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerSetMaxPhaseMessage const>(msgptr);
      return respondLayerSetMaxPhase(castMessage);
   };
   mMessageActionMap.emplace("LayerSetMaxPhase", action);
}

int PhaseParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_phase(ioFlag);
   return PV_SUCCESS;
}

void PhaseParam::ioParam_phase(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "phase", &mPhase, mPhase);
   if (mPhase < 0) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: phase must be >= 0 (given value was %d).\n", getDescription_c(), mPhase);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status
PhaseParam::respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message) {
   return setMaxPhase(message->mMaxPhase);
}

Response::Status PhaseParam::setMaxPhase(int *maxPhase) {
   if (*maxPhase < mPhase) {
      *maxPhase = mPhase;
   }
   return Response::SUCCESS;
}

} // namespace PV
