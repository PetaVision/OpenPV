/*
 * DependentPhaseParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "DependentPhaseParam.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentPhaseParam::DependentPhaseParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

DependentPhaseParam::~DependentPhaseParam() {}

void DependentPhaseParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void DependentPhaseParam::setObjectType() { mObjectType = "DependentPhaseParam"; }

int DependentPhaseParam::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return PhaseParam::ioParamsFillGroup(ioSwitch);
}

void DependentPhaseParam::ioParam_phase(ParamsIOSwitch ioSwitch) {
   mParamsIO->handleUnnecessaryParameter("phase");
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

   std::string const &linkedObjectName   = originalLayerNameParam->getLinkedObjectName();
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
   mParamsIO->handleUnnecessaryParameter("phase", mPhase);

   auto status = PhaseParam::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
