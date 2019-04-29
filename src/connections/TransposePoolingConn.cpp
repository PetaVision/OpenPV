/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#include "TransposePoolingConn.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "components/TransposePatchSize.hpp"
#include "delivery/TransposePoolingDelivery.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

TransposePoolingConn::TransposePoolingConn() {}

TransposePoolingConn::~TransposePoolingConn() {}

void TransposePoolingConn::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   PoolingConn::initialize(name, params, comm);
}

void TransposePoolingConn::fillComponentTable() {
   PoolingConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *TransposePoolingConn::createDeliveryObject() {
   return new TransposePoolingDelivery(name, parameters(), mCommunicator);
}

PatchSize *TransposePoolingConn::createPatchSize() {
   return new TransposePatchSize(name, parameters(), mCommunicator);
}

OriginalConnNameParam *TransposePoolingConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parameters(), mCommunicator);
}

} // namespace PV
