/*
 * NormalizeGroup.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: pschultz
 */

#include "normalizers/NormalizeGroup.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

NormalizeGroup::NormalizeGroup(char const *name, HyPerCol *hc) { initialize(name, hc); }

NormalizeGroup::NormalizeGroup() {}

NormalizeGroup::~NormalizeGroup() { free(mNormalizeGroupName); }

int NormalizeGroup::initialize(char const *name, HyPerCol *hc) {
   int status = NormalizeBase::initialize(name, hc);
   return status;
}

int NormalizeGroup::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_normalizeGroupName(ioFlag);
   return status;
}

// The NormalizeBase parameters are overridden to do nothing in NormalizeGroup.
void NormalizeGroup::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {}

void NormalizeGroup::ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "normalizeGroupName", &mNormalizeGroupName);
}

Response::Status
NormalizeGroup::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = NormalizeBase::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(message->mHierarchy, getDescription());
   pvAssert(objectMapComponent);
   HyPerConn *groupHeadConn =
         objectMapComponent->lookup<HyPerConn>(std::string(mNormalizeGroupName));
   mGroupHead = groupHeadConn->getComponentByType<NormalizeBase>();

   if (mGroupHead == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
               getDescription_c(),
               mNormalizeGroupName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   auto hierarchy           = message->mHierarchy;
   WeightsPair *weightsPair = mapLookupByType<WeightsPair>(hierarchy, getDescription());
   Weights *preWeights      = weightsPair->getPreWeights();
   pvAssert(preWeights); // NormalizeBase::communicateInitInfo should have called needPre.
   mGroupHead->addWeightsToList(preWeights);
   return Response::SUCCESS;
}

int NormalizeGroup::normalizeWeights() { return PV_SUCCESS; }

} /* namespace PV */
