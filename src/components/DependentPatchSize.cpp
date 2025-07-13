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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

DependentPatchSize::DependentPatchSize() {}

DependentPatchSize::~DependentPatchSize() {}

void DependentPatchSize::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   PatchSize::initialize(params, defaults, comm);
}

void DependentPatchSize::setObjectType() { mObjectType = "DependentPatchSize"; }

int DependentPatchSize::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return PatchSize::ioParamsFillGroup(ioSwitch);
}

void DependentPatchSize::ioParam_nxp(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void DependentPatchSize::ioParam_nyp(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void DependentPatchSize::ioParam_nfp(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("nfp");
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
   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();
   mOriginalPatchSize      = objectTable->findObject<PatchSize>(originalConnName);
   FatalIf(
         mOriginalPatchSize == nullptr,
         "%s original connection \"%s\" does not have a PatchSize.\n",
         getDescription_c(),
         originalConnName.c_str());

   if (!mOriginalPatchSize->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName.c_str());
      }
      return Response::POSTPONE;
   }

   auto status = PatchSize::communicateInitInfo(message);
   return status;
}

void DependentPatchSize::setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) {
   mPatchSizeX = mOriginalPatchSize->getPatchSizeX();
   mParamsIO->handleUnnecessaryParameter("nxp", mNxp);
}

void DependentPatchSize::setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) { 
   mPatchSizeY = mOriginalPatchSize->getPatchSizeY();
   mParamsIO->handleUnnecessaryParameter("nyp", mNyp);
}

void DependentPatchSize::setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) {
   mPatchSizeF = mOriginalPatchSize->getPatchSizeF();
   mParamsIO->handleUnnecessaryParameter("nfp", mNfp);
}

} // namespace PV
