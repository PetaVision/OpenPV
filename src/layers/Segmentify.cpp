#include "Segmentify.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentifyBuffer.hpp"

namespace PV {

Segmentify::Segmentify(const char *name, HyPerCol *hc) { initialize(name, hc); }

Segmentify::Segmentify() {
   // initialize() gets called by subclass's initialize method
}

Segmentify::~Segmentify() {}

int Segmentify::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void Segmentify::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *Segmentify::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

LayerInputBuffer *Segmentify::createLayerInput() { return nullptr; }

ActivityComponent *Segmentify::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentifyBuffer>(getName(), parent);
}

} /* namespace PV */
