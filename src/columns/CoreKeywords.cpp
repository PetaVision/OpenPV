/*
 * CoreKeywords.cpp
 *
 *  Created on: Mar 6, 2019
 *      Author: pschultz
 */

#include "cMakeHeader.h"

#include "CoreKeywords.hpp"

#include "columns/Factory.hpp"

#include "layers/ANNErrorLayer.hpp"
#include "layers/ANNLayer.hpp"
#include "layers/ANNSquaredLayer.hpp"
#include "layers/BackgroundLayer.hpp"
#include "layers/BinningLayer.hpp"
#include "layers/CloneVLayer.hpp"
#include "layers/ConstantLayer.hpp"
#include "layers/DependentFirmThresholdCostLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/FilenameParsingLayer.hpp"
#include "layers/FirmThresholdCostLayer.hpp"
#include "layers/GapLayer.hpp"
#include "layers/GaussianNoiseLayer.hpp"
#include "layers/HyPerLCALayer.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/ISTALayer.hpp"
#include "layers/ImageLayer.hpp"
#include "layers/InputRegionLayer.hpp"
#include "layers/LIF.hpp"
#include "layers/LIFGap.hpp"
#include "layers/LinearTransformLayer.hpp"
#include "layers/LeakyIntegrator.hpp"
#include "layers/MaskLayer.hpp"
#include "layers/MomentumLCALayer.hpp"
#include "layers/PoolingIndexLayer.hpp"
#include "layers/PtwiseProductLayer.hpp"
#include "layers/PtwiseQuotientLayer.hpp"
#include "layers/PvpLayer.hpp"
#include "layers/PvpListLayer.hpp"
#include "layers/RescaleLayer.hpp"
#include "layers/Retina.hpp"
#include "layers/SigmoidLayer.hpp"

#include "connections/CloneConn.hpp"
#include "connections/CopyConn.hpp"
#include "connections/FeedbackConn.hpp"
#include "connections/GapConn.hpp"
#include "connections/HyPerConn.hpp"
#include "connections/IdentConn.hpp"
#include "connections/MomentumConn.hpp"
#include "connections/PlasticCloneConn.hpp"
#include "connections/PoolingConn.hpp"
#include "connections/RescaleConn.hpp"
#include "connections/TransposeConn.hpp"
#include "connections/TransposePoolingConn.hpp"
#include "connections/WTAConn.hpp"

#include "probes/AdaptiveTimeScaleProbe.hpp"
#include "probes/ColumnEnergyProbe.hpp"
#include "probes/FirmThresholdCostFnLCAProbe.hpp"
#include "probes/FirmThresholdCostFnProbe.hpp"
#include "probes/KneeTimeScaleProbe.hpp"
#include "probes/L0NormLCAProbe.hpp"
#include "probes/L0NormProbe.hpp"
#include "probes/L1NormLCAProbe.hpp"
#include "probes/L1NormProbe.hpp"
#include "probes/L2NormProbe.hpp"
#include "probes/LogTimeScaleProbe.hpp"
#include "probes/QuotientProbe.hpp"
#include "probes/RequireAllZeroActivityProbe.hpp"
#include "probes/StatsProbe.hpp"

#include "initv/ConstantV.hpp"
#include "initv/DiscreteUniformRandomV.hpp"
#include "initv/GaussianRandomV.hpp"
#include "initv/InitVFromFile.hpp"
#include "initv/UniformRandomV.hpp"
#include "initv/ZeroV.hpp"

#include "delivery/IdentDelivery.hpp"
#include "delivery/PostsynapticPerspectiveConvolveDelivery.hpp"
#include "delivery/PostsynapticPerspectiveStochasticDelivery.hpp"
#include "delivery/PresynapticPerspectiveConvolveDelivery.hpp"
#include "delivery/PresynapticPerspectiveStochasticDelivery.hpp"
#include "delivery/RescaleDelivery.hpp"
#include "delivery/WTADelivery.hpp"

#ifdef PV_USE_CUDA
#include "delivery/PostsynapticPerspectiveGPUDelivery.hpp"
#include "delivery/PresynapticPerspectiveGPUDelivery.hpp"
#endif // PV_USE_CUDA

#include "weightinit/InitCocircWeights.hpp"
#include "weightinit/InitDiscreteUniformRandomWeights.hpp"
#include "weightinit/InitGauss2DWeights.hpp"
#include "weightinit/InitGaussianRandomWeights.hpp"
#include "weightinit/InitIdentWeights.hpp"
#include "weightinit/InitOneToOneWeights.hpp"
#include "weightinit/InitOneToOneWeightsWithDelays.hpp"
#include "weightinit/InitSmartWeights.hpp"
#include "weightinit/InitSpreadOverArborsWeights.hpp"
#include "weightinit/InitUniformRandomWeights.hpp"
#include "weightinit/InitUniformWeights.hpp"
#include "weightinit/InitWeights.hpp"

#include "weightupdaters/HebbianUpdater.hpp"

#include "normalizers/NormalizeContrastZeroMean.hpp"
#include "normalizers/NormalizeGroup.hpp"
#include "normalizers/NormalizeL2.hpp"
#include "normalizers/NormalizeMax.hpp"
#include "normalizers/NormalizeNone.hpp"
#include "normalizers/NormalizeSum.hpp"

// Deprecated Oct 7, 2021.
// These probes currently are not being used, and maintaining them would
// complicate refactoring.
// #include "deprecated/KernelProbe.hpp"
// #include "deprecated/L2ConnProbe.hpp"
// #include "deprecated/PointLIFProbe.hpp"
// #include "deprecated/PointProbe.hpp"

namespace PV {

void registerCoreKeywords() {

   auto factory = Factory::instance();

   // Layers
   factory->registerKeyword("ANNErrorLayer", Factory::create<ANNErrorLayer>);
   factory->registerKeyword("ANNLayer", Factory::create<ANNLayer>);
   factory->registerKeyword("ANNSquaredLayer", Factory::create<ANNSquaredLayer>);
   factory->registerKeyword("BackgroundLayer", Factory::create<BackgroundLayer>);
   factory->registerKeyword("BinningLayer", Factory::create<BinningLayer>);
   factory->registerKeyword("CloneVLayer", Factory::create<CloneVLayer>);
   factory->registerKeyword("ConstantLayer", Factory::create<ConstantLayer>);
   factory->registerKeyword(
         "DependentFirmThresholdCostLayer", Factory::create<DependentFirmThresholdCostLayer>);
   factory->registerKeyword("DropoutLayer", Factory::create<DropoutLayer>);
   factory->registerKeyword("FilenameParsingLayer", Factory::create<FilenameParsingLayer>);
   factory->registerKeyword("FirmThresholdCostLayer", Factory::create<FirmThresholdCostLayer>);
   factory->registerKeyword("GapLayer", Factory::create<GapLayer>);
   factory->registerKeyword("GaussianNoiseLayer", Factory::create<GaussianNoiseLayer>);
   factory->registerKeyword("HyPerLayer", Factory::create<HyPerLayer>);
   factory->registerKeyword("HyPerLCALayer", Factory::create<HyPerLCALayer>);
   factory->registerKeyword("ISTALayer", Factory::create<ISTALayer>);
   factory->registerKeyword("ImageLayer", Factory::create<ImageLayer>);
   factory->registerKeyword("InputRegionLayer", Factory::create<InputRegionLayer>);
   factory->registerKeyword("LIF", Factory::create<LIF>);
   factory->registerKeyword("LIFGap", Factory::create<LIFGap>);
   factory->registerKeyword("LeakyIntegrator", Factory::create<LeakyIntegrator>);
   factory->registerKeyword("MaskLayer", Factory::create<MaskLayer>);
   factory->registerKeyword("MomentumLCALayer", Factory::create<MomentumLCALayer>);
   factory->registerKeyword("PoolingIndexLayer", Factory::create<PoolingIndexLayer>);
   factory->registerKeyword("PvpLayer", Factory::create<PvpLayer>);
   factory->registerKeyword("PvpListLayer", Factory::create<PvpListLayer>);
   factory->registerKeyword("PtwiseProductLayer", Factory::create<PtwiseProductLayer>);
   factory->registerKeyword("PtwiseQuotientLayer", Factory::create<PtwiseQuotientLayer>);
   factory->registerKeyword("RescaleLayer", Factory::create<RescaleLayer>);
   factory->registerKeyword("Retina", Factory::create<Retina>);
   factory->registerKeyword("RotateLayer", Factory::create<LinearTransformLayer>);
   factory->registerKeyword("ScaleXLayer", Factory::create<LinearTransformLayer>);
   factory->registerKeyword("ScaleYLayer", Factory::create<LinearTransformLayer>);
   factory->registerKeyword("SigmoidLayer", Factory::create<SigmoidLayer>);

   // Connections
   factory->registerKeyword("HyPerConn", Factory::create<HyPerConn>);
   factory->registerKeyword("CloneConn", Factory::create<CloneConn>);
   factory->registerKeyword("CopyConn", Factory::create<CopyConn>);
   factory->registerKeyword("FeedbackConn", Factory::create<FeedbackConn>);
   factory->registerKeyword("GapConn", Factory::create<GapConn>);
   factory->registerKeyword("IdentConn", Factory::create<IdentConn>);
   factory->registerKeyword("MomentumConn", Factory::create<MomentumConn>);
   factory->registerKeyword("PlasticCloneConn", Factory::create<PlasticCloneConn>);
   factory->registerKeyword("PoolingConn", Factory::create<PoolingConn>);
   factory->registerKeyword("RescaleConn", Factory::create<RescaleConn>);
   factory->registerKeyword("TransposeConn", Factory::create<TransposeConn>);
   factory->registerKeyword("TransposePoolingConn", Factory::create<TransposePoolingConn>);
   factory->registerKeyword("WTAConn", Factory::create<WTAConn>);

   // Probes
   factory->registerKeyword("AdaptiveTimeScaleProbe", Factory::create<AdaptiveTimeScaleProbe>);
   factory->registerKeyword("KneeTimeScaleProbe", Factory::create<KneeTimeScaleProbe>);
   factory->registerKeyword("LogTimeScaleProbe", Factory::create<LogTimeScaleProbe>);
   factory->registerKeyword("ColumnEnergyProbe", Factory::create<ColumnEnergyProbe>);
   factory->registerKeyword(
         "FirmThresholdCostFnLCAProbe", Factory::create<FirmThresholdCostFnLCAProbe>);
   factory->registerKeyword("FirmThresholdCostFnProbe", Factory::create<FirmThresholdCostFnProbe>);
   factory->registerKeyword("L0NormLCAProbe", Factory::create<L0NormLCAProbe>);
   factory->registerKeyword("L0NormProbe", Factory::create<L0NormProbe>);
   factory->registerKeyword("L1NormLCAProbe", Factory::create<L1NormLCAProbe>);
   factory->registerKeyword("L1NormProbe", Factory::create<L1NormProbe>);
   factory->registerKeyword("L2NormProbe", Factory::create<L2NormProbe>);
   factory->registerKeyword("QuotientProbe", Factory::create<QuotientProbe>);
   factory->registerKeyword(
         "RequireAllZeroActivityProbe", Factory::create<RequireAllZeroActivityProbe>);
   factory->registerKeyword("StatsProbe", Factory::create<StatsProbe>);

   // InitV objects
   factory->registerKeyword("ConstantV", Factory::create<ConstantV>);
   factory->registerKeyword("DiscreteUniformRandomV", Factory::create<DiscreteUniformRandomV>);
   factory->registerKeyword("GaussianRandomV", Factory::create<GaussianRandomV>);
   factory->registerKeyword("InitVFromFile", Factory::create<InitVFromFile>);
   factory->registerKeyword("UniformRandomV", Factory::create<UniformRandomV>);
   factory->registerKeyword("ZeroV", Factory::create<ZeroV>);

   // Delivery objects
   factory->registerKeyword("IdentDelivery", Factory::create<IdentDelivery>);
   factory->registerKeyword(
         "PostsynapticPerspectiveConvolveDelivery",
         Factory::create<PostsynapticPerspectiveConvolveDelivery>);
   factory->registerKeyword(
         "PostsynapticPerspectiveStochasticDelivery",
         Factory::create<PostsynapticPerspectiveStochasticDelivery>);
   factory->registerKeyword(
         "PresynapticPerspectiveConvolveDelivery",
         Factory::create<PresynapticPerspectiveConvolveDelivery>);
   factory->registerKeyword(
         "PresynapticPerspectiveStochasticDelivery",
         Factory::create<PresynapticPerspectiveStochasticDelivery>);
   factory->registerKeyword("RescaleDelivery", Factory::create<RescaleDelivery>);
   factory->registerKeyword("WTADelivery", Factory::create<WTADelivery>);
#ifdef PV_USE_CUDA
   factory->registerKeyword(
         "PostsynapticPerspectiveGPUDelivery", Factory::create<PostsynapticPerspectiveGPUDelivery>);
   factory->registerKeyword(
         "PresynapticPerspectiveGPUDelivery", Factory::create<PresynapticPerspectiveGPUDelivery>);
#endif // PV_USE_CUDA

   // InitWeights objects
   factory->registerKeyword("Gauss2DWeight", Factory::create<InitGauss2DWeights>);
   factory->registerKeyword("CoCircWeight", Factory::create<InitCocircWeights>);
   factory->registerKeyword(
         "DiscreteUniformRandomWeight", Factory::create<InitDiscreteUniformRandomWeights>);
   factory->registerKeyword("UniformWeight", Factory::create<InitUniformWeights>);
   factory->registerKeyword("SmartWeight", Factory::create<InitSmartWeights>);
   factory->registerKeyword("UniformRandomWeight", Factory::create<InitUniformRandomWeights>);
   factory->registerKeyword("GaussianRandomWeight", Factory::create<InitGaussianRandomWeights>);
   factory->registerKeyword("IdentWeight", Factory::create<InitIdentWeights>);
   factory->registerKeyword("OneToOneWeights", Factory::create<InitOneToOneWeights>);
   factory->registerKeyword(
         "OneToOneWeightsWithDelays", Factory::create<InitOneToOneWeightsWithDelays>);
   factory->registerKeyword("SpreadOverArborsWeight", Factory::create<InitSpreadOverArborsWeights>);
   factory->registerKeyword("FileWeight", Factory::create<InitWeights>);

   // Weight updater objects
   factory->registerKeyword("HebbianUpdater", Factory::create<HebbianUpdater>);

   factory->registerKeyword(
         "normalizeContrastZeroMean", Factory::create<NormalizeContrastZeroMean>);
   factory->registerKeyword("normalizeL2", Factory::create<NormalizeL2>);
   factory->registerKeyword("normalizeMax", Factory::create<NormalizeMax>);
   factory->registerKeyword("none", Factory::create<NormalizeNone>);
   factory->registerKeyword("normalizeSum", Factory::create<NormalizeSum>);
   factory->registerKeyword("normalizeGroup", Factory::create<NormalizeGroup>);

   // Deprecated Oct 7, 2021.
   // These probes currently are not being used, and maintaining them would
   // complicate refactoring.
   //   factory->registerKeyword("KernelProbe", Factory::create<KernelProbe>);
   //   factory->registerKeyword("L2ConnProbe", Factory::create<L2ConnProbe>);
   //   factory->registerKeyword("PointLIFProbe", Factory::create<PointLIFProbe>);
   //   factory->registerKeyword("PointProbe", Factory::create<PointProbe>);
}

} // end namespace PV
