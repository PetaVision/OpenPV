/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#include "TransposePoolingConn.hpp"
#include "components/DependentSharedWeights.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "components/TransposePatchSize.hpp"
#include "delivery/TransposePoolingDelivery.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

TransposePoolingConn::TransposePoolingConn() {}

TransposePoolingConn::~TransposePoolingConn() {}

void TransposePoolingConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   PoolingConn::initialize(paramsIO, comm);
}

void TransposePoolingConn::fillComponentTable() {
   PoolingConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *TransposePoolingConn::createDeliveryObject() {
   return new TransposePoolingDelivery(mParamsIO, mCommunicator);
}

PatchSize *TransposePoolingConn::createPatchSize() {
   return new TransposePatchSize(mParamsIO, mCommunicator);
}

SharedWeights *TransposePoolingConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO, mCommunicator);
}

OriginalConnNameParam *TransposePoolingConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO, mCommunicator);
}

} // namespace PV
