/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/CloneWeightsPair.hpp"

namespace PV {

CloneConn::CloneConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

CloneConn::CloneConn() {}

CloneConn::~CloneConn() {}

int CloneConn::initialize(char const *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

WeightsPair *CloneConn::createWeightsPair() { return new CloneWeightsPair(name, parent); }

InitWeights *CloneConn::createWeightInitializer() { return nullptr; }

NormalizeBase *CloneConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *CloneConn::createWeightUpdater() { return nullptr; }

int CloneConn::initializeState() { return PV_SUCCESS; }

} // namespace PV
