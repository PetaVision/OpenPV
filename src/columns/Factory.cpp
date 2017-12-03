/*
 * Factory.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "Factory.hpp"
#include "include/pv_common.h"

#include "columns/HyPerCol.hpp"

#include "layers/ANNErrorLayer.hpp"
#include "layers/ANNLayer.hpp"
#include "layers/ANNSquaredLayer.hpp"
#include "layers/ANNWhitenedLayer.hpp"
#include "layers/BackgroundLayer.hpp"
#include "layers/BinningLayer.hpp"
#include "layers/CloneVLayer.hpp"
#include "layers/ConstantLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/FilenameParsingGroundTruthLayer.hpp"
#include "layers/GapLayer.hpp"
#include "layers/HyPerLCALayer.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/ISTALayer.hpp"
#include "layers/ImageFromMemoryBuffer.hpp"
#include "layers/ImageLayer.hpp"
#include "layers/InputRegionLayer.hpp"
#include "layers/KmeansLayer.hpp"
// #include "layers/LCALIFLayer.hpp"
#include "layers/LIF.hpp"
// #include "layers/LIFGap.hpp"
#include "layers/LabelErrorLayer.hpp"
#include "layers/LeakyIntegrator.hpp"
#include "layers/MaskLayer.hpp"
#include "layers/MomentumLCALayer.hpp"
#include "layers/PoolingIndexLayer.hpp"
#include "layers/PtwiseProductLayer.hpp"
#include "layers/PtwiseQuotientLayer.hpp"
#include "layers/PvpLayer.hpp"
#include "layers/RescaleLayer.hpp"
#include "layers/Retina.hpp"
#include "layers/RunningAverageLayer.hpp"
#include "layers/ShuffleLayer.hpp"
#include "layers/SigmoidLayer.hpp"
#include "layers/WTALayer.hpp"

#include "connections/CloneConn.hpp"
// #include "connections/CopyConn.hpp"
// #include "connections/FeedbackConn.hpp"
// #include "connections/GapConn.hpp"
#include "connections/HyPerConn.hpp"
#include "connections/IdentConn.hpp"
// #include "connections/ImprintConn.hpp"
// #include "connections/MomentumConn.hpp"
// #include "connections/PlasticCloneConn.hpp"
// #include "connections/PoolingConn.hpp"
#include "connections/RescaleConn.hpp"
// #include "connections/TransposeConn.hpp"
// #include "connections/TransposePoolingConn.hpp"

#include "probes/AdaptiveTimeScaleProbe.hpp"
#include "probes/ColumnEnergyProbe.hpp"
#include "probes/FirmThresholdCostFnLCAProbe.hpp"
#include "probes/FirmThresholdCostFnProbe.hpp"
// #include "probes/KernelProbe.hpp"
#include "probes/KneeTimeScaleProbe.hpp"
#include "probes/L0NormLCAProbe.hpp"
#include "probes/L0NormProbe.hpp"
#include "probes/L1NormLCAProbe.hpp"
#include "probes/L1NormProbe.hpp"
#include "probes/L2NormProbe.hpp"
#include "probes/LogTimeScaleProbe.hpp"
#include "probes/PointLIFProbe.hpp"
#include "probes/PointProbe.hpp"
#include "probes/QuotientColProbe.hpp"
#include "probes/RequireAllZeroActivityProbe.hpp"
#include "probes/StatsProbe.hpp"

#include "initv/ConstantV.hpp"
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

// #include "weightinit/InitCocircWeights.hpp"
#include "weightinit/InitGauss2DWeights.hpp"
// #include "weightinit/InitGaussianRandomWeights.hpp"
// #include "weightinit/InitIdentWeights.hpp"
// #include "weightinit/InitOneToOneWeights.hpp"
// #include "weightinit/InitOneToOneWeightsWithDelays.hpp"
// #include "weightinit/InitSmartWeights.hpp"
#include "weightinit/InitSpreadOverArborsWeights.hpp"
// #include "weightinit/InitUniformRandomWeights.hpp"
#include "weightinit/InitUniformWeights.hpp"
#include "weightinit/InitWeights.hpp"

#include "normalizers/NormalizeContrastZeroMean.hpp"
// #include "normalizers/NormalizeGroup.hpp"
#include "normalizers/NormalizeL2.hpp"
#include "normalizers/NormalizeMax.hpp"
#include "normalizers/NormalizeSum.hpp"

namespace PV {

Factory::Factory() { registerCoreKeywords(); }

int Factory::registerCoreKeywords() {
   keywordHandlerList = std::vector<KeywordHandler *>();

   registerKeyword("ANNErrorLayer", Factory::create<ANNErrorLayer>);
   registerKeyword("ANNLayer", Factory::create<ANNLayer>);
   registerKeyword("ANNSquaredLayer", Factory::create<ANNSquaredLayer>);
   registerKeyword("ANNWhitenedLayer", Factory::create<ANNWhitenedLayer>);
   registerKeyword("BackgroundLayer", Factory::create<BackgroundLayer>);
   registerKeyword("BinningLayer", Factory::create<BinningLayer>);
   registerKeyword("CloneVLayer", Factory::create<CloneVLayer>);
   registerKeyword("ConstantLayer", Factory::create<ConstantLayer>);
   registerKeyword("DropoutLayer", Factory::create<DropoutLayer>);
   registerKeyword(
         "FilenameParsingGroundTruthLayer", Factory::create<FilenameParsingGroundTruthLayer>);
   registerKeyword("GapLayer", Factory::create<GapLayer>);
   registerKeyword("HyPerLayer", Factory::create<HyPerLayer>);
   registerKeyword("HyPerLCALayer", Factory::create<HyPerLCALayer>);
   registerKeyword("ISTALayer", Factory::create<ISTALayer>);

   registerKeyword("ImageLayer", Factory::create<ImageLayer>);
   registerKeyword("InputRegionLayer", Factory::create<InputRegionLayer>);
   registerKeyword("PvpLayer", Factory::create<PvpLayer>);
   registerKeyword("ImageFromMemoryBuffer", Factory::create<ImageFromMemoryBuffer>);
   registerKeyword("KmeansLayer", Factory::create<KmeansLayer>);
   // registerKeyword("LCALIFLayer", Factory::create<LCALIFLayer>);
   registerKeyword("LIF", Factory::create<LIF>);
   // registerKeyword("LIFGap", Factory::create<LIFGap>);
   registerKeyword("LabelErrorLayer", Factory::create<LabelErrorLayer>);
   registerKeyword("LeakyIntegrator", Factory::create<LeakyIntegrator>);
   registerKeyword("MaskLayer", Factory::create<MaskLayer>);
   registerKeyword("MomentumLCALayer", Factory::create<MomentumLCALayer>);
   registerKeyword("PoolingIndexLayer", Factory::create<PoolingIndexLayer>);
   registerKeyword("PtwiseProductLayer", Factory::create<PtwiseProductLayer>);
   registerKeyword("PtwiseQuotientLayer", Factory::create<PtwiseQuotientLayer>);
   registerKeyword("RescaleLayer", Factory::create<RescaleLayer>);
   registerKeyword("RunningAverageLayer", Factory::create<RunningAverageLayer>);
   registerKeyword("Retina", Factory::create<Retina>);
   registerKeyword("ShuffleLayer", Factory::create<ShuffleLayer>);
   registerKeyword("SigmoidLayer", Factory::create<SigmoidLayer>);
   registerKeyword("WTALayer", Factory::create<WTALayer>);

   registerKeyword("HyPerConn", Factory::create<HyPerConn>);
   registerKeyword("CloneConn", Factory::create<CloneConn>);
   // registerKeyword("ComponentsConn", Factory::create<ComponentsConn>);
   // registerKeyword("CopyConn", Factory::create<CopyConn>);
   // registerKeyword("FeedbackConn", Factory::create<FeedbackConn>);
   // registerKeyword("GapConn", Factory::create<GapConn>);
   registerKeyword("IdentConn", Factory::create<IdentConn>);
   // registerKeyword("ImprintConn", Factory::create<ImprintConn>);
   // registerKeyword("MomentumConn", Factory::create<MomentumConn>);
   // registerKeyword("PlasticCloneConn", Factory::create<PlasticCloneConn>);
   // registerKeyword("PoolingConn", Factory::create<PoolingConn>);
   registerKeyword("RescaleConn", Factory::create<RescaleConn>);
   // registerKeyword("TransposeConn", Factory::create<TransposeConn>);
   // registerKeyword("TransposePoolingConn", Factory::create<TransposePoolingConn>);
   registerKeyword("AdaptiveTimeScaleProbe", Factory::create<AdaptiveTimeScaleProbe>);
   registerKeyword("KneeTimeScaleProbe", Factory::create<KneeTimeScaleProbe>);
   registerKeyword("LogTimeScaleProbe", Factory::create<LogTimeScaleProbe>);
   registerKeyword("ColumnEnergyProbe", Factory::create<ColumnEnergyProbe>);
   registerKeyword("FirmThresholdCostFnLCAProbe", Factory::create<FirmThresholdCostFnLCAProbe>);
   registerKeyword("FirmThresholdCostFnProbe", Factory::create<FirmThresholdCostFnProbe>);
   // registerKeyword("KernelProbe", Factory::create<KernelProbe>);
   registerKeyword("L0NormLCAProbe", Factory::create<L0NormLCAProbe>);
   registerKeyword("L0NormProbe", Factory::create<L0NormProbe>);
   registerKeyword("L1NormLCAProbe", Factory::create<L1NormLCAProbe>);
   registerKeyword("L1NormProbe", Factory::create<L1NormProbe>);
   registerKeyword("L2NormProbe", Factory::create<L2NormProbe>);
   registerKeyword("PointLIFProbe", Factory::create<PointLIFProbe>);
   registerKeyword("PointProbe", Factory::create<PointProbe>);
   registerKeyword("QuotientColProbe", Factory::create<QuotientColProbe>);
   registerKeyword("RequireAllZeroActivityProbe", Factory::create<RequireAllZeroActivityProbe>);
   registerKeyword("StatsProbe", Factory::create<StatsProbe>);

   registerKeyword("ConstantV", Factory::create<ConstantV>);
   registerKeyword("GaussianRandomV", Factory::create<GaussianRandomV>);
   registerKeyword("InitVFromFile", Factory::create<InitVFromFile>);
   registerKeyword("UniformRandomV", Factory::create<UniformRandomV>);
   registerKeyword("ZeroV", Factory::create<ZeroV>);

   registerKeyword("IdentDelivery", Factory::create<IdentDelivery>);
   registerKeyword(
         "PostsynapticPerspectiveConvolveDelivery",
         Factory::create<PostsynapticPerspectiveConvolveDelivery>);
   registerKeyword(
         "PostsynapticPerspectiveStochasticDelivery",
         Factory::create<PostsynapticPerspectiveStochasticDelivery>);
   registerKeyword(
         "PresynapticPerspectiveConvolveDelivery",
         Factory::create<PresynapticPerspectiveConvolveDelivery>);
   registerKeyword(
         "PresynapticPerspectiveStochasticDelivery",
         Factory::create<PresynapticPerspectiveStochasticDelivery>);
   registerKeyword("RescaleDelivery", Factory::create<RescaleDelivery>);

   registerKeyword("Gauss2DWeight", Factory::create<InitGauss2DWeights>);
   // registerKeyword("CoCircWeight", Factory::create<InitCocircWeights>);
   registerKeyword("UniformWeight", Factory::create<InitUniformWeights>);
   // registerKeyword("SmartWeight", Factory::create<InitSmartWeights>);
   // registerKeyword("UniformRandomWeight", Factory::create<InitUniformRandomWeights>);
   // registerKeyword("GaussianRandomWeight", Factory::create<InitGaussianRandomWeights>);
   // registerKeyword("IdentWeight", Factory::create<InitIdentWeights>);
   // registerKeyword("OneToOneWeights", Factory::create<InitOneToOneWeights>);
   // registerKeyword("OneToOneWeightsWithDelays",
   //   Factory::create<InitOneToOneWeightsWithDelays>);
   registerKeyword("SpreadOverArborsWeight", Factory::create<InitSpreadOverArborsWeights>);
   registerKeyword("FileWeight", Factory::create<InitWeights>);

   registerKeyword("normalizeContrastZeroMean", Factory::create<NormalizeContrastZeroMean>);
   registerKeyword("normalizeL2", Factory::create<NormalizeL2>);
   registerKeyword("normalizeMax", Factory::create<NormalizeMax>);
   registerKeyword("normalizeSum", Factory::create<NormalizeSum>);
   // registerKeyword("normalizeGroup", Factory::create<NormalizeGroup>);

   return PV_SUCCESS;
}

int Factory::copyKeywordHandlerList(std::vector<KeywordHandler *> const &orig) {
   for (auto &kh : orig) {
      registerKeyword(kh->getKeyword(), kh->getCreator());
   }
   return PV_SUCCESS;
}

int Factory::registerKeyword(char const *keyword, ObjectCreateFn creator) {
   KeywordHandler const *keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler != nullptr) {
      return PV_FAILURE;
   }
   KeywordHandler *newKeyword = new KeywordHandler(keyword, creator);
   keywordHandlerList.push_back(newKeyword);
   return PV_SUCCESS;
}

BaseObject *Factory::createByKeyword(char const *keyword, char const *name, HyPerCol *hc) const {
   if (keyword == nullptr) {
      return nullptr;
   }
   KeywordHandler const *keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler == nullptr) {
      auto errorString = std::string(keyword).append(" \"").append(name).append("\": ");
      errorString.append("keyword \"").append(keyword).append("\" is unrecognized.");
      throw std::invalid_argument(errorString);
   }
   return keywordHandler ? keywordHandler->create(name, hc) : nullptr;
}

KeywordHandler const *Factory::getKeywordHandler(char const *keyword) const {
   pvAssert(keyword != nullptr);
   for (auto &typeCreator : keywordHandlerList) {
      if (!strcmp(typeCreator->getKeyword(), keyword)) {
         return typeCreator;
      }
   }
   return nullptr;
}

int Factory::clearKeywordHandlerList() {
   for (auto &kh : keywordHandlerList) {
      delete kh;
   }
   keywordHandlerList.clear();
   return PV_SUCCESS;
}

Factory::~Factory() { clearKeywordHandlerList(); }

} /* namespace PV */
