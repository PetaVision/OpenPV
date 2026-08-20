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
   BaseLayer::initialize(name, params, comm);
}

void SegmentLayer::fillComponentTable() {
   BaseLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *SegmentLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(getName(), parameters(), mCommunicator);
}

ActivityComponent *SegmentLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentBuffer>(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
