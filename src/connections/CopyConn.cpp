/*
 * CopyConn.cpp
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#include "CopyConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/CopyWeightsPair.hpp"
#include "components/DependentArborList.hpp"
#include "components/DependentPatchSize.hpp"
#include "components/DependentSharedWeights.hpp"
#include "utils/MapLookupByType.hpp"
#include "weightupdaters/CopyUpdater.hpp"

namespace PV {

CopyConn::CopyConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

CopyConn::CopyConn() {}

CopyConn::~CopyConn() {}

int CopyConn::initialize(char const *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

void CopyConn::defineComponents() {
   HyPerConn::defineComponents();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addObserver(mOriginalConnNameParam);
   }
}

ArborList *CopyConn::createArborList() { return new DependentArborList(name, parent); }

PatchSize *CopyConn::createPatchSize() { return new DependentPatchSize(name, parent); }

SharedWeights *CopyConn::createSharedWeights() { return new DependentSharedWeights(name, parent); }

WeightsPairInterface *CopyConn::createWeightsPair() { return new CopyWeightsPair(name, parent); }

InitWeights *CopyConn::createWeightInitializer() { return nullptr; }

BaseWeightUpdater *CopyConn::createWeightUpdater() { return new CopyUpdater(name, parent); }

OriginalConnNameParam *CopyConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parent);
}

Response::Status CopyConn::initializeState() {
   auto *copyWeightsPair = dynamic_cast<CopyWeightsPair *>(mWeightsPair);
   pvAssert(copyWeightsPair);
   if (!copyWeightsPair->getOriginalWeightsPair()->getInitialValuesSetFlag()) {
      return Response::POSTPONE;
   }
   copyWeightsPair->copy();
   return Response::SUCCESS;
}

} // namespace PV
