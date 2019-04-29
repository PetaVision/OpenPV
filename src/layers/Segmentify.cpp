#include "Segmentify.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentifyBuffer.hpp"

namespace PV {

Segmentify::Segmentify(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

Segmentify::Segmentify() {
   // initialize() gets called by subclass's initialize method
}

Segmentify::~Segmentify() {}

void Segmentify::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void Segmentify::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *Segmentify::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

LayerInputBuffer *Segmentify::createLayerInput() { return nullptr; }

ActivityComponent *Segmentify::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentifyBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
