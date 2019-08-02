/*
 * NormalizeGroup.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: pschultz
 */

#include "normalizers/NormalizeGroup.hpp"
#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

NormalizeGroup::NormalizeGroup(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

NormalizeGroup::NormalizeGroup() {}

NormalizeGroup::~NormalizeGroup() { free(mNormalizeGroupName); }

void NormalizeGroup::initialize(char const *name, PVParams *params, Communicator const *comm) {
   NormalizeBase::initialize(name, params, comm);
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
   parameters()->ioParamStringRequired(ioFlag, name, "normalizeGroupName", &mNormalizeGroupName);
}

Response::Status
NormalizeGroup::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = NormalizeBase::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   auto *objectTable        = message->mObjectTable;
   HyPerConn *groupHeadConn = objectTable->findObject<HyPerConn>(mNormalizeGroupName);
   mGroupHead               = objectTable->findObject<NormalizeBase>(mNormalizeGroupName);
   if (mGroupHead == nullptr) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
               getDescription_c(),
               mNormalizeGroupName);
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   WeightsPair *weightsPair = objectTable->findObject<WeightsPair>(getName());
   pvAssert(weightsPair); // NormalizeBase::communicateInitInfo should have checked for this.
   Weights *preWeights = weightsPair->getPreWeights();
   pvAssert(preWeights); // NormalizeBase::communicateInitInfo should have called needPre.
   mGroupHead->addWeightsToList(preWeights);
   return Response::SUCCESS;
}

int NormalizeGroup::normalizeWeights() { return PV_SUCCESS; }

} /* namespace PV */
