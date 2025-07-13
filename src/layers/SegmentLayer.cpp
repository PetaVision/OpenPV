#include "SegmentLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "components/SegmentBuffer.hpp"

namespace PV {

SegmentLayer::SegmentLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

SegmentLayer::SegmentLayer() {}

SegmentLayer::~SegmentLayer() {}

void SegmentLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

void SegmentLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *SegmentLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LayerInputBuffer *SegmentLayer::createLayerInput() { return nullptr; }

ActivityComponent *SegmentLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SegmentBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
