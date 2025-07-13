/*
 * PhaseParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "PhaseParam.hpp"

namespace PV {

PhaseParam::PhaseParam(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PhaseParam::~PhaseParam() {}

void PhaseParam::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseObject::initialize(params, defaults, comm);
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

int PhaseParam::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_phase(ioSwitch);
   return PV_SUCCESS;
}

void PhaseParam::ioParam_phase(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "phase", &mPhase);
   if (mPhase < 0) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: phase must be >= 0 (given value was %d).\n", getDescription_c(), mPhase);
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
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
