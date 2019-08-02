/*
 * DependentBoundaryConditions.cpp
 *
 *  Created on: Jul 30, 2018
 *      Author: Pete Schultz
 */

#include "DependentBoundaryConditions.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentBoundaryConditions::DependentBoundaryConditions(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentBoundaryConditions::~DependentBoundaryConditions() {}

void DependentBoundaryConditions::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void DependentBoundaryConditions::setObjectType() { mObjectType = "DependentBoundaryConditions"; }

int DependentBoundaryConditions::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return BoundaryConditions::ioParamsFillGroup(ioFlag);
}

void DependentBoundaryConditions::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   parameters()->handleUnnecessaryStringParameter(name, "mirrorBCflag");
}

void DependentBoundaryConditions::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   parameters()->handleUnnecessaryStringParameter(name, "valueBC");
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

   char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();
   auto *originalBoundaryConditions =
         objectTable->findObject<BoundaryConditions>(originalLayerName);
   FatalIf(
         originalBoundaryConditions == nullptr,
         "%s original connection \"%s\" does not have a BoundaryConditions component.\n",
         getDescription_c(),
         originalLayerName);

   if (!originalBoundaryConditions->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original layer \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalLayerName);
      }
      return Response::POSTPONE;
   }

   mMirrorBCflag = originalBoundaryConditions->getMirrorBCflag();
   parameters()->handleUnnecessaryParameter(name, "mirrorBCflag", mMirrorBCflag);

   mValueBC = originalBoundaryConditions->getValueBC();
   parameters()->handleUnnecessaryParameter(name, "valueBC", mValueBC);

   auto status = BoundaryConditions::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
