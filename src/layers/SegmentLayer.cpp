#include "SegmentLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentBuffer.hpp"

namespace PV {

SegmentLayer::SegmentLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

SegmentLayer::SegmentLayer() {}

SegmentLayer::~SegmentLayer() {}

void SegmentLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void SegmentLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *SegmentLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

LayerInputBuffer *SegmentLayer::createLayerInput() { return nullptr; }

ActivityComponent *SegmentLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentBuffer>(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
