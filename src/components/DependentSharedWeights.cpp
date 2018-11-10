/*
 * DependentSharedWeights.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "DependentSharedWeights.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

DependentSharedWeights::DependentSharedWeights(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

DependentSharedWeights::DependentSharedWeights() {}

DependentSharedWeights::~DependentSharedWeights() {}

void DependentSharedWeights::initialize(char const *name, PVParams *params, Communicator *comm) {
   SharedWeights::initialize(name, params, comm);
}

void DependentSharedWeights::setObjectType() { mObjectType = "DependentSharedWeights"; }

int DependentSharedWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}

void DependentSharedWeights::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
}

Response::Status DependentSharedWeights::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *originalConnNameParam = message->mHierarchy->lookupByType<OriginalConnNameParam>();
   pvAssert(originalConnNameParam);

   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
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

   auto *originalSharedWeights = originalConn->getComponentByType<SharedWeights>();
   FatalIf(
         originalSharedWeights == nullptr,
         "%s original connection \"%s\" does not have an SharedWeights.\n",
         getDescription_c(),
         originalConn->getName());

   if (!originalSharedWeights->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return Response::POSTPONE;
   }
   mSharedWeights = originalSharedWeights->getSharedWeights();
   parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);

   auto status = SharedWeights::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

} // namespace PV
