/*
 * DependentBoundaryConditions.cpp
 *
 *  Created on: Jul 30, 2018
 *      Author: Pete Schultz
 */

#include "DependentBoundaryConditions.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentBoundaryConditions::DependentBoundaryConditions(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

DependentBoundaryConditions::~DependentBoundaryConditions() {}

int DependentBoundaryConditions::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void DependentBoundaryConditions::setObjectType() { mObjectType = "DependentBoundaryConditions"; }

int DependentBoundaryConditions::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return BoundaryConditions::ioParamsFillGroup(ioFlag);
}

void DependentBoundaryConditions::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryStringParameter(name, "mirrorBCflag");
}

void DependentBoundaryConditions::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryStringParameter(name, "valueBC");
}

Response::Status DependentBoundaryConditions::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *originalLayerNameParam = mapLookupByType<OriginalLayerNameParam>(message->mHierarchy);
   pvAssert(originalLayerNameParam);

   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalLayerNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   ComponentBasedObject *originalObject = nullptr;
   try {
      originalObject = originalLayerNameParam->findLinkedObject(message->mHierarchy);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalObject);

   auto *originalBoundaryConditions = originalObject->getComponentByType<BoundaryConditions>();
   FatalIf(
         originalBoundaryConditions == nullptr,
         "%s original connection \"%s\" does not have a BoundaryConditions component.\n",
         getDescription_c(),
         originalObject->getName());

   if (!originalBoundaryConditions->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original layer \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalObject->getName());
      }
      return Response::POSTPONE;
   }

   mMirrorBCflag = originalBoundaryConditions->getMirrorBCflag();
   parent->parameters()->handleUnnecessaryParameter(name, "mirrorBCflag", mMirrorBCflag);

   mValueBC = originalBoundaryConditions->getValueBC();
   parent->parameters()->handleUnnecessaryParameter(name, "valueBC", mValueBC);

   auto status = BoundaryConditions::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
