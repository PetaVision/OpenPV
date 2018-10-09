/*
 * DependentPatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentPatchSize.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

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
      parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void DependentPatchSize::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void DependentPatchSize::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "nfp");
   }
   // During the communication phase, nfp will be copied from originalConn
}

Response::Status
DependentPatchSize::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *originalConnNameParam = message->mHierarchy->lookupByType<OriginalConnNameParam>();
   pvAssert(originalConnNameParam);

   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalConnNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   ComponentBasedObject *originalConn = nullptr;
   try {
      originalConn = originalConnNameParam->findLinkedObject(message->mHierarchy);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalConn);

   auto *originalPatchSize = originalConn->getComponentByType<PatchSize>();
   FatalIf(
         originalPatchSize == nullptr,
         "%s original connection \"%s\" does not have an PatchSize.\n",
         getDescription_c(),
         originalConn->getName());

   if (!originalPatchSize->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return Response::POSTPONE;
   }

   setPatchSize(originalPatchSize);
   parameters()->handleUnnecessaryParameter(name, "nxp", mPatchSizeX);
   parameters()->handleUnnecessaryParameter(name, "nyp", mPatchSizeY);
   parameters()->handleUnnecessaryParameter(name, "nfp", mPatchSizeF);

   auto status = PatchSize::communicateInitInfo(message);
   return status;
}

void DependentPatchSize::setPatchSize(PatchSize *originalPatchSize) {
   mPatchSizeX = originalPatchSize->getPatchSizeX();
   mPatchSizeY = originalPatchSize->getPatchSizeY();
   mPatchSizeF = originalPatchSize->getPatchSizeF();
}

} // namespace PV
