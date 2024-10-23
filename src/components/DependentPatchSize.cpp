/*
 * DependentPatchSize.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentPatchSize.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

DependentPatchSize::DependentPatchSize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentPatchSize::DependentPatchSize() {}

DependentPatchSize::~DependentPatchSize() {}

void DependentPatchSize::initialize(char const *name, PVParams *params, Communicator const *comm) {
   PatchSize::initialize(name, params, comm);
}

void DependentPatchSize::setObjectType() { mObjectType = "DependentPatchSize"; }

int DependentPatchSize::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return PatchSize::ioParamsFillGroup(ioFlag);
}

void DependentPatchSize::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(getName(), "nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void DependentPatchSize::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(getName(), "nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void DependentPatchSize::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(getName(), "nfp");
   }
   // During the communication phase, nfp will be copied from originalConn
}

Response::Status
DependentPatchSize::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s could not find an OriginalConnNameParam.\n",
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
   char const *originalConnName = originalConnNameParam->getLinkedObjectName();
   mOriginalPatchSize      = objectTable->findObject<PatchSize>(originalConnName);
   FatalIf(
         mOriginalPatchSize == nullptr,
         "%s original connection \"%s\" does not have a PatchSize.\n",
         getDescription_c(),
         originalConnName);

   if (!mOriginalPatchSize->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName);
      }
      return Response::POSTPONE;
   }

   auto status = PatchSize::communicateInitInfo(message);
   return status;
}

void DependentPatchSize::setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) {
   mPatchSizeX = mOriginalPatchSize->getPatchSizeX();
   parameters()->handleUnnecessaryParameter(getName(), "nxp", mNxp);
}

void DependentPatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) { 
   mPatchSizeY = mOriginalPatchSize->getPatchSizeY();
   parameters()->handleUnnecessaryParameter(getName(), "nyp", mNyp);
}

void DependentPatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   mPatchSizeF = mOriginalPatchSize->getPatchSizeF();
   parameters()->handleUnnecessaryParameter(getName(), "nfp", mNfp);
}

} // namespace PV
