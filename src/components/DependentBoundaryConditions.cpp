/*
 * DependentBoundaryConditions.cpp
 *
 *  Created on: Jul 30, 2018
 *      Author: Pete Schultz
 */

#include "DependentBoundaryConditions.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentBoundaryConditions::DependentBoundaryConditions(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

DependentBoundaryConditions::~DependentBoundaryConditions() {}

void DependentBoundaryConditions::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void DependentBoundaryConditions::setObjectType() { mObjectType = "DependentBoundaryConditions"; }

int DependentBoundaryConditions::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return BoundaryConditions::ioParamsFillGroup(ioSwitch);
}

void DependentBoundaryConditions::ioParam_mirrorBCflag(ParamsIOSwitch ioSwitch) {
   mParamsIO->handleUnnecessaryParameter("mirrorBCflag");
}

void DependentBoundaryConditions::ioParam_valueBC(ParamsIOSwitch ioSwitch) {
   mParamsIO->handleUnnecessaryParameter("valueBC");
}

Response::Status DependentBoundaryConditions::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable            = message->mObjectTable;
   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerNameParam.\n",
         getDescription_c());

   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalLayerNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   std::string const &originalLayerName = originalLayerNameParam->getLinkedObjectName();
   auto *originalBoundaryConditions =
         objectTable->findObject<BoundaryConditions>(originalLayerName);
   FatalIf(
         originalBoundaryConditions == nullptr,
         "%s original connection \"%s\" does not have a BoundaryConditions component.\n",
         getDescription_c(),
         originalLayerName.c_str());

   if (!originalBoundaryConditions->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original layer \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalLayerName.c_str());
      }
      return Response::POSTPONE;
   }

   mMirrorBCflag = originalBoundaryConditions->getMirrorBCflag();
   mParamsIO->handleUnnecessaryParameter("mirrorBCflag", mMirrorBCflag);

   mValueBC = originalBoundaryConditions->getValueBC();
   mParamsIO->handleUnnecessaryParameter("valueBC", mValueBC);

   auto status = BoundaryConditions::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
