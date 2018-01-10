/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#include "TransposePoolingConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "delivery/TransposePoolingDelivery.hpp"

namespace PV {

TransposePoolingConn::TransposePoolingConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposePoolingConn::TransposePoolingConn() {}

TransposePoolingConn::~TransposePoolingConn() {}

int TransposePoolingConn::initialize(char const *name, HyPerCol *hc) {
   int status = TransposeConn::initialize(name, hc);
   return status;
}

BaseDelivery *TransposePoolingConn::createDeliveryObject() {
   return new TransposePoolingDelivery(name, parent);
}

SharedWeights *TransposePoolingConn::createSharedWeights() { return nullptr; }

WeightsPairInterface *TransposePoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(name, parent);
}

} // namespace PV
