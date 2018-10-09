/*
 * DependentArborList.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentArborList.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

DependentArborList::DependentArborList(char const *name, HyPerCol *hc) { initialize(name, hc); }

DependentArborList::DependentArborList() {}

DependentArborList::~DependentArborList() {}

int DependentArborList::initialize(char const *name, HyPerCol *hc) {
   return ArborList::initialize(name, hc);
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

   auto *originalArborList = originalConn->getComponentByType<ArborList>();
   FatalIf(
         originalArborList == nullptr,
         "%s original connection \"%s\" does not have an ArborList.\n",
         getDescription_c(),
         originalConn->getName());

   if (!originalArborList->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
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
