/* TransposeConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "TransposeConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/TransposeWeightsPair.hpp"

namespace PV {

TransposeConn::TransposeConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposeConn::TransposeConn() {}

TransposeConn::~TransposeConn() {}

int TransposeConn::initialize(char const *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

WeightsPair *TransposeConn::createWeightsPair() { return new TransposeWeightsPair(name, parent); }

InitWeights *TransposeConn::createWeightInitializer() { return nullptr; }

NormalizeBase *TransposeConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *TransposeConn::createWeightUpdater() { return nullptr; }

int TransposeConn::initializeState() { return PV_SUCCESS; }

} // namespace PV
