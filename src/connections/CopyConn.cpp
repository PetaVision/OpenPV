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

CopyConn::CopyConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

CopyConn::CopyConn() {}

CopyConn::~CopyConn() {}

void CopyConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

void CopyConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

ArborList *CopyConn::createArborList() {
   return new DependentArborList(mParamsIO, mCommunicator);
}

PatchSize *CopyConn::createPatchSize() {
   return new DependentPatchSize(mParamsIO, mCommunicator);
}

SharedWeights *CopyConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO, mCommunicator);
}

WeightsPairInterface *CopyConn::createWeightsPair() {
   return new CopyWeightsPair(mParamsIO, mCommunicator);
}

InitWeights *CopyConn::createWeightInitializer() { return nullptr; }

BaseWeightUpdater *CopyConn::createWeightUpdater() {
   return new CopyUpdater(mParamsIO, mCommunicator);
}

OriginalConnNameParam *CopyConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO, mCommunicator);
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
