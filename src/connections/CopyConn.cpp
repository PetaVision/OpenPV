/*
 * CopyConn.cpp
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#include "CopyConn.hpp"
#include "components/CopyWeightsPair.hpp"
#include "components/DependentArborList.hpp"
#include "components/DependentPatchSize.hpp"
#include "components/DependentSharedWeights.hpp"
#include "weightupdaters/CopyUpdater.hpp"

namespace PV {

CopyConn::CopyConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

CopyConn::CopyConn() {}

CopyConn::~CopyConn() {}

void CopyConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

void CopyConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

ArborList *CopyConn::createArborList() {
   return new DependentArborList(name, parameters(), mCommunicator);
}

PatchSize *CopyConn::createPatchSize() {
   return new DependentPatchSize(name, parameters(), mCommunicator);
}

SharedWeights *CopyConn::createSharedWeights() {
   return new DependentSharedWeights(name, parameters(), mCommunicator);
}

WeightsPairInterface *CopyConn::createWeightsPair() {
   return new CopyWeightsPair(name, parameters(), mCommunicator);
}

InitWeights *CopyConn::createWeightInitializer() { return nullptr; }

BaseWeightUpdater *CopyConn::createWeightUpdater() {
   return new CopyUpdater(name, parameters(), mCommunicator);
}

OriginalConnNameParam *CopyConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parameters(), mCommunicator);
}

Response::Status CopyConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto *copyWeightsPair = getComponentByType<CopyWeightsPair>();
   pvAssert(copyWeightsPair);
   if (!copyWeightsPair->getOriginalWeightsPair()->getInitialValuesSetFlag()) {
      return Response::POSTPONE;
   }
   copyWeightsPair->copy();
   return Response::SUCCESS;
}

} // namespace PV
