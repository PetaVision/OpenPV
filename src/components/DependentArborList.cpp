/*
 * DependentArborList.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentArborList.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/BaseConnection.hpp"
#include "utils/MapLookupByType.hpp"

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
      parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors");
   }
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

Response::Status
DependentArborList::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;

   char const *originalConnName = getOriginalConnName(hierarchy);
   pvAssert(originalConnName);

   auto *originalArborList = getOriginalArborList(hierarchy, originalConnName);
   pvAssert(originalArborList);

   if (!originalArborList->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConnName);
      }
      return Response::POSTPONE;
   }
   mNumAxonalArbors = originalArborList->getNumAxonalArbors();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", mNumAxonalArbors);

   auto status = ArborList::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

char const *
DependentArborList::getOriginalConnName(std::map<std::string, Observer *> const hierarchy) const {
   OriginalConnNameParam *originalConnNameParam =
         mapLookupByType<OriginalConnNameParam>(hierarchy, getDescription());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires an OriginalConnNameParam component.\n",
         getDescription_c());
   char const *originalConnName = originalConnNameParam->getOriginalConnName();
   return originalConnName;
}

ArborList *DependentArborList::getOriginalArborList(
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

   auto *originalArborList = originalConn->getComponentByType<ArborList>();
   FatalIf(
         originalArborList == nullptr,
         "%s original connection \"%s\" does not have an ArborList.\n",
         getDescription_c(),
         originalConnName);
   return originalArborList;
}

} // namespace PV
