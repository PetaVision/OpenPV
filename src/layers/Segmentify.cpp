#include "Segmentify.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentifyBuffer.hpp"

namespace PV {

Segmentify::Segmentify(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

Segmentify::Segmentify() {
   // initialize() gets called by subclass's initialize method
}

Segmentify::~Segmentify() {}

void Segmentify::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

void Segmentify::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *Segmentify::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

LayerInputBuffer *Segmentify::createLayerInput() { return nullptr; }

ActivityComponent *Segmentify::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentifyBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
