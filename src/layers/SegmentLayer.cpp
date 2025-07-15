#include "SegmentLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentBuffer.hpp"

namespace PV {

SegmentLayer::SegmentLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

SegmentLayer::SegmentLayer() {}

SegmentLayer::~SegmentLayer() {}

void SegmentLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

void SegmentLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *SegmentLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

LayerInputBuffer *SegmentLayer::createLayerInput() { return nullptr; }

ActivityComponent *SegmentLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentBuffer>(mParamsIO, mCommunicator);
}

} /* namespace PV */
