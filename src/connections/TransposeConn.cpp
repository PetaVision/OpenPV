/* TransposeConn.cpp
 *
 * Created on: May 16, 2011
 *     Author: peteschultz
 */

#include "TransposeConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/DependentArborList.hpp"
#include "components/DependentSharedWeights.hpp"
#include "components/TransposePatchSize.hpp"
#include "components/TransposeWeightsPair.hpp"

namespace PV {

TransposeConn::TransposeConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposeConn::TransposeConn() {}

TransposeConn::~TransposeConn() {}

int TransposeConn::initialize(char const *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

void TransposeConn::defineComponents() {
   HyPerConn::defineComponents();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addObserver(mOriginalConnNameParam);
   }
}

ArborList *TransposeConn::createArborList() { return new DependentArborList(name, parent); }

PatchSize *TransposeConn::createPatchSize() { return new TransposePatchSize(name, parent); }

SharedWeights *TransposeConn::createSharedWeights() {
   return new DependentSharedWeights(name, parent);
}

WeightsPairInterface *TransposeConn::createWeightsPair() {
   return new TransposeWeightsPair(name, parent);
}

InitWeights *TransposeConn::createWeightInitializer() { return nullptr; }

NormalizeBase *TransposeConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *TransposeConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *TransposeConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parent);
}

Response::Status TransposeConn::initializeState() { return Response::NO_ACTION; }

} // namespace PV
