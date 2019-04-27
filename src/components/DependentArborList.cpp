/*
 * DependentArborList.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentArborList.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

DependentArborList::DependentArborList(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentArborList::DependentArborList() {}

DependentArborList::~DependentArborList() {}

void DependentArborList::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ArborList::initialize(name, params, comm);
}

void DependentArborList::setObjectType() { mObjectType = "DependentArborList"; }

int DependentArborList::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return ArborList::ioParamsFillGroup(ioFlag);
}

void DependentArborList::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "numAxonalArbors");
   }
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

Response::Status
DependentArborList::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s does not have an OriginalConnNameParam.\n",
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
   auto *originalArborList      = objectTable->findObject<ArborList>(originalConnName);
   FatalIf(
         originalArborList == nullptr,
         "%s original connection \"%s\" does not have an ArborList.\n",
         getDescription_c(),
         originalConnName);

   if (!originalArborList->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName);
      }
      return Response::POSTPONE;
   }

   mNumAxonalArbors = originalArborList->getNumAxonalArbors();
   parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", mNumAxonalArbors);

   auto status = ArborList::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
