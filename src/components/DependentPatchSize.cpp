/*
 * DependentPatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentPatchSize.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/BaseConnection.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

DependentPatchSize::DependentPatchSize(char const *name, HyPerCol *hc) { initialize(name, hc); }

DependentPatchSize::DependentPatchSize() {}

DependentPatchSize::~DependentPatchSize() {}

int DependentPatchSize::initialize(char const *name, HyPerCol *hc) {
   return PatchSize::initialize(name, hc);
}

void DependentPatchSize::setObjectType() { mObjectType = "DependentPatchSize"; }

int DependentPatchSize::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return PatchSize::ioParamsFillGroup(ioFlag);
}

void DependentPatchSize::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void DependentPatchSize::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void DependentPatchSize::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nfp");
   }
   // During the communication phase, nfp will be copied from originalConn
}

Response::Status
DependentPatchSize::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;

   char const *originalConnName = getOriginalConnName(hierarchy);
   pvAssert(originalConnName);

   auto *originalPatchSize = getOriginalPatchSize(hierarchy, originalConnName);
   pvAssert(originalPatchSize);

   if (!originalPatchSize->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName);
      }
      return Response::POSTPONE;
   }

   setPatchSize(originalPatchSize);
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", mPatchSizeX);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", mPatchSizeY);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", mPatchSizeF);

   auto status = PatchSize::communicateInitInfo(message);
   return status;
}

void DependentPatchSize::setPatchSize(PatchSize *originalPatchSize) {
   mPatchSizeX = originalPatchSize->getPatchSizeX();
   mPatchSizeY = originalPatchSize->getPatchSizeY();
   mPatchSizeF = originalPatchSize->getPatchSizeF();
}

char const *
DependentPatchSize::getOriginalConnName(std::map<std::string, Observer *> const hierarchy) const {
   OriginalConnNameParam *originalConnNameParam =
         mapLookupByType<OriginalConnNameParam>(hierarchy, getDescription());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires an OriginalConnNameParam component.\n",
         getDescription_c());
   char const *originalConnName = originalConnNameParam->getOriginalConnName();
   return originalConnName;
}

PatchSize *DependentPatchSize::getOriginalPatchSize(
      std::map<std::string, Observer *> const hierarchy,
      char const *originalConnName) const {
   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   pvAssert(objectMapComponent);
   BaseConnection *originalConn =
         objectMapComponent->lookup<BaseConnection>(std::string(originalConnName));
   if (originalConn == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" does not correspond to a BaseConnection in the "
               "column.\n",
               getDescription_c(),
               originalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }

   auto *originalPatchSize = originalConn->getComponentByType<PatchSize>();
   FatalIf(
         originalPatchSize == nullptr,
         "%s original connection \"%s\" does not have an PatchSize.\n",
         getDescription_c(),
         originalConnName);
   return originalPatchSize;
}

} // namespace PV
