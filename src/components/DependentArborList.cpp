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

DependentArborList::DependentArborList(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

DependentArborList::DependentArborList() {}

DependentArborList::~DependentArborList() {}

void DependentArborList::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ArborList::initialize(paramsIO, comm);
}

void DependentArborList::setObjectType() { mObjectType = "DependentArborList"; }

int DependentArborList::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return ArborList::ioParamsFillGroup(ioSwitch);
}

void DependentArborList::ioParam_numAxonalArbors(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("numAxonalArbors");
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

   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();
   auto *originalArborList      = objectTable->findObject<ArborList>(originalConnName);
   FatalIf(
         originalArborList == nullptr,
         "%s original connection \"%s\" does not have an ArborList.\n",
         getDescription_c(),
         originalConnName.c_str());

   if (!originalArborList->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName.c_str());
      }
      return Response::POSTPONE;
   }

   mNumAxonalArbors = originalArborList->getNumAxonalArbors();
   mParamsIO->handleUnnecessaryParameter("numAxonalArbors", mNumAxonalArbors);

   auto status = ArborList::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
