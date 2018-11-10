#include "SegmentLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentBuffer.hpp"

namespace PV {

SegmentLayer::SegmentLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

SegmentLayer::SegmentLayer() {}

SegmentLayer::~SegmentLayer() {}

int SegmentLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void SegmentLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *SegmentLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

LayerInputBuffer *SegmentLayer::createLayerInput() { return nullptr; }

ActivityComponent *SegmentLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentBuffer>(getName(), parent);
}

} /* namespace PV */
