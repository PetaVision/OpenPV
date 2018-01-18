/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#include "TransposePoolingConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "components/TransposePatchSize.hpp"
#include "delivery/TransposePoolingDelivery.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposePoolingConn::TransposePoolingConn() {}

TransposePoolingConn::~TransposePoolingConn() {}

int TransposePoolingConn::initialize(char const *name, HyPerCol *hc) {
   int status = PoolingConn::initialize(name, hc);
   return status;
}

void TransposePoolingConn::defineComponents() {
   PoolingConn::defineComponents();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addObserver(mOriginalConnNameParam);
   }
}

BaseDelivery *TransposePoolingConn::createDeliveryObject() {
   return new TransposePoolingDelivery(name, parent);
}

PatchSize *TransposePoolingConn::createPatchSize() { return new TransposePatchSize(name, parent); }

OriginalConnNameParam *TransposePoolingConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parent);
}

} // namespace PV
