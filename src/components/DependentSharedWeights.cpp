/*
 * DependentSharedWeights.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentSharedWeights.hpp"
#include "components/OriginalConnNameParam.hpp"

namespace PV {

DependentSharedWeights::DependentSharedWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

DependentSharedWeights::DependentSharedWeights() {}

DependentSharedWeights::~DependentSharedWeights() {}

void DependentSharedWeights::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   SharedWeights::initialize(paramsIO, comm);
}

void DependentSharedWeights::setObjectType() { mObjectType = "DependentSharedWeights"; }

int DependentSharedWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return SharedWeights::ioParamsFillGroup(ioSwitch);
}

void DependentSharedWeights::ioParam_sharedWeights(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
}

Response::Status DependentSharedWeights::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s could not find an OriginalConnNameParam component.\n",
         getDescription_c());

   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalConnNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();
   auto *originalSharedWeights  = objectTable->findObject<SharedWeights>(originalConnName);

   if (!originalSharedWeights->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName.c_str());
      }
      return Response::POSTPONE;
   }
   mSharedWeightsFlag = originalSharedWeights->getSharedWeightsFlag();
   mParamsIO->handleUnnecessaryParameter("sharedWeights", mSharedWeightsFlag);

   auto status = SharedWeights::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
