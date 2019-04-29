/*
 * DependentPhaseParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "DependentPhaseParam.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentPhaseParam::DependentPhaseParam(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentPhaseParam::~DependentPhaseParam() {}

void DependentPhaseParam::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void DependentPhaseParam::setObjectType() { mObjectType = "DependentPhaseParam"; }

int DependentPhaseParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return PhaseParam::ioParamsFillGroup(ioFlag);
}

void DependentPhaseParam::ioParam_phase(enum ParamsIOFlag ioFlag) {
   parameters()->handleUnnecessaryStringParameter(name, "phase");
}

Response::Status DependentPhaseParam::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *originalLayerNameParam =
         message->mObjectTable->findObject<OriginalLayerNameParam>(getName());
   pvAssert(originalLayerNameParam);

   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalLayerNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   auto linkedObjectName    = std::string(originalLayerNameParam->getLinkedObjectName());
   auto *originalPhaseParam = message->mObjectTable->findObject<PhaseParam>(linkedObjectName);
   FatalIf(
         originalPhaseParam == nullptr,
         "%s linked object \"%s\" does not have a PhaseParam component.\n",
         getDescription_c(),
         linkedObjectName.c_str());

   if (!originalPhaseParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original layer \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               linkedObjectName.c_str());
      }
      return Response::POSTPONE;
   }
   mPhase = originalPhaseParam->getPhase();
   parameters()->handleUnnecessaryParameter(name, "phase", mPhase);

   auto status = PhaseParam::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
