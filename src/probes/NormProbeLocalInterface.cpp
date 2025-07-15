#include "NormProbeLocalInterface.hpp"

namespace PV {

NormProbeLocalInterface::NormProbeLocalInterface(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

NormProbeLocalInterface::~NormProbeLocalInterface() {}

void NormProbeLocalInterface::checkMaskLayerDimensions() const {
   // checkMaskLayerDimensions() should only be called after TargetLayer has been set,
   // and only if there is a MaskLayer.
   pvAssert(mTargetLayer != nullptr);
   pvAssert(mMaskLayer != nullptr);
   PVLayerLoc const *loc     = mTargetLayer->getLayerLoc();
   PVLayerLoc const *maskLoc = mMaskLayer->getLayerLoc();
   pvAssert(loc != nullptr and loc != nullptr);
   FatalIf(
         maskLoc->nxGlobal != loc->nxGlobal or maskLoc->nyGlobal != loc->nyGlobal,
         "Probe %s: maskLayerName \"%s\" does not have the same x and y dimensions.\n"
         "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
         getName_c(),
         mMaskLayerName.c_str(),
         maskLoc->nxGlobal,
         maskLoc->nyGlobal,
         loc->nxGlobal,
         loc->nyGlobal);
   pvAssert(maskLoc->nx == loc->nx and maskLoc->ny == loc->ny);
   FatalIf(
         maskLoc->nf != 1 and maskLoc->nf != loc->nf,
         "Probe %s: maskLayerName \"%s\" must either have the same number of features as "
         "target layer \"%s\", or one feature.\n",
         "    mask has %d features versus target layer's %d features.\n",
         getName_c(),
         mMaskLayerName.c_str(),
         mTargetLayer->getName(),
         maskLoc->nf,
         loc->nf);
}

void NormProbeLocalInterface::clearStoredValues() { mStoredValues.clear(); }

Response::Status NormProbeLocalInterface::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status;
   if (mMaskLayer == nullptr and !mMaskLayerName.empty()) {
      mMaskLayer = message->mObjectTable->findObject<HyPerLayer>(mMaskLayerName);
      FatalIf(
            mMaskLayer == nullptr,
            "Probe %s maskLayerName \"%s\" is not a layer in the column.\n",
            getName_c(),
            mMaskLayerName.c_str());
      status = Response::SUCCESS;
   }
   else {
      status = Response::NO_ACTION;
   }
   return status;
}

float const *NormProbeLocalInterface::findDataBuffer(HyPerLayer *layer) const {
   auto *layerData = layer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         layerData == nullptr,
         "Probe %s findDataBuffer() failed: \"%s\" does not have an activity buffer.\n",
         getName_c(),
         mTargetLayer->getName());
   return layerData->getLayerData();
}

void NormProbeLocalInterface::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   ProbeComponent::initialize(paramsIO);
}

void NormProbeLocalInterface::initializeState(HyPerLayer *targetLayer) {
   mTargetLayer  = targetLayer;
   mTargetBuffer = findDataBuffer(targetLayer);
   if (mMaskLayer) {
      checkMaskLayerDimensions();
      mMaskBuffer = findDataBuffer(mMaskLayer);
   }
}

void NormProbeLocalInterface::ioParam_maskLayerName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "maskLayerName", &mMaskLayerName);
}

void NormProbeLocalInterface::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_maskLayerName(ioSwitch);
}

void NormProbeLocalInterface::storeValues(double simTime) {
   ProbeData<double> newValues(simTime, getLayerLoc()->nbatch);
   calculateNorms(simTime, newValues);
   mStoredValues.store(newValues);
}

} // namespace PV
