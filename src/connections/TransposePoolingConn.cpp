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

TransposePoolingConn::TransposePoolingConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

TransposePoolingConn::TransposePoolingConn() {}

TransposePoolingConn::~TransposePoolingConn() {}

void TransposePoolingConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   PoolingConn::initialize(params, defaults, comm);
}

void TransposePoolingConn::fillComponentTable() {
   PoolingConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *TransposePoolingConn::createDeliveryObject() {
   return new TransposePoolingDelivery(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

PatchSize *TransposePoolingConn::createPatchSize() {
   return new TransposePatchSize(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

SharedWeights *TransposePoolingConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

OriginalConnNameParam *TransposePoolingConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
