/*
 * Factory.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "Factory.hpp"
#include <include/pv_common.h>

#include <columns/HyPerCol.hpp>

#include <layers/ANNErrorLayer.hpp>
#include <layers/ANNLayer.hpp>
#include <layers/ANNSquaredLayer.hpp>
#include <layers/ANNWhitenedLayer.hpp>
#include <layers/BackgroundLayer.hpp>
#include <layers/BinningLayer.hpp>
#include <layers/CloneVLayer.hpp>
#include <layers/ConstantLayer.hpp>
#include <layers/FilenameParsingGroundTruthLayer.hpp>
#include <layers/GapLayer.hpp>
#include <layers/HyPerLayer.hpp>
#include <layers/HyPerLCALayer.hpp>
#include <layers/ISTALayer.hpp>
#include <layers/Image.hpp>
#include <layers/ImagePvp.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include <layers/KmeansLayer.hpp>
#include <layers/LCALIFLayer.hpp>
#include <layers/LIF.hpp>
#include <layers/LIFGap.hpp>
#include <layers/LabelErrorLayer.hpp>
#include <layers/LabelLayer.hpp>
#include <layers/LeakyIntegrator.hpp>
#include <layers/MaskLayer.hpp>
#include <layers/MomentumLCALayer.hpp>
#include <layers/Movie.hpp>
#include <layers/MoviePvp.hpp>
#include <layers/Patterns.hpp>
#include <layers/PoolingIndexLayer.hpp>
#include <layers/PtwiseLinearTransferLayer.hpp>
#include <layers/PtwiseProductLayer.hpp>
#include <layers/PtwiseQuotientLayer.hpp>
#include <layers/RescaleLayer.hpp>
#include <layers/RunningAverageLayer.hpp>
#include <layers/Retina.hpp>
#include <layers/ShuffleLayer.hpp>
#include <layers/SigmoidLayer.hpp>
#include <layers/WTALayer.hpp>

#include <connections/HyPerConn.hpp>
#include <connections/CloneConn.hpp>
#include <connections/CloneKernelConn.hpp>
#include <connections/CopyConn.hpp>
#include <connections/FeedbackConn.hpp>
#include <connections/GapConn.hpp>
#include <connections/IdentConn.hpp>
#include <connections/ImprintConn.hpp>
#include <connections/KernelConn.hpp>
#include <connections/MomentumConn.hpp>
#include <connections/PlasticCloneConn.hpp>
#include <connections/PoolingConn.hpp>
#include <connections/RescaleConn.hpp>
#include <connections/TransposeConn.hpp>
#include <connections/TransposePoolingConn.hpp>

#include <io/ColumnEnergyProbe.hpp>
#include <io/QuotientColProbe.hpp>
#include <io/FirmThresholdCostFnLCAProbe.hpp>
#include <io/FirmThresholdCostFnProbe.hpp>
#include <io/L0NormLCAProbe.hpp>
#include <io/L0NormProbe.hpp>
#include <io/L1NormLCAProbe.hpp>
#include <io/L1NormProbe.hpp>
#include <io/L2NormProbe.hpp>
#include <io/PointLIFProbe.hpp>
#include <io/PointProbe.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>
#include <io/StatsProbe.hpp>
#include <io/KernelProbe.hpp>

#include <weightinit/InitWeights.hpp>
#include <weightinit/InitCocircWeights.hpp>
#include <weightinit/InitGauss2DWeights.hpp>
#include <weightinit/InitGaussianRandomWeights.hpp>
#include <weightinit/InitIdentWeights.hpp>
#include <weightinit/InitMaxPoolingWeights.hpp>
#include <weightinit/InitOneToOneWeights.hpp>
#include <weightinit/InitOneToOneWeightsWithDelays.hpp>
#include <weightinit/InitSmartWeights.hpp>
#include <weightinit/InitSpreadOverArborsWeights.hpp>
#include <weightinit/InitUniformRandomWeights.hpp>
#include <weightinit/InitUniformWeights.hpp>

#include <normalizers/NormalizeContrastZeroMean.hpp>
#include <normalizers/NormalizeGroup.hpp>
#include <normalizers/NormalizeL2.hpp>
#include <normalizers/NormalizeMax.hpp>
#include <normalizers/NormalizeSum.hpp>

namespace PV {

Factory::Factory() {
   registerCoreKeywords();
}

Factory::Factory(Factory const& orig) {
   copyKeywordHandlerList(orig.keywordHandlerList);
}

Factory& Factory::operator=(Factory const& orig) {
   clearKeywordHandlerList();
   copyKeywordHandlerList(orig.keywordHandlerList);
   return *this;
}


int Factory::registerCoreKeywords() {
   keywordHandlerList = std::vector<KeywordHandler*>();

   registerKeyword("ANNErrorLayer", Factory::standardCreate<ANNErrorLayer>);
   registerKeyword("ANNLayer", Factory::standardCreate<ANNLayer>);
   registerKeyword("ANNSquaredLayer", Factory::standardCreate<ANNSquaredLayer>);
   registerKeyword("ANNWhitenedLayer", Factory::standardCreate<ANNWhitenedLayer>);
   registerKeyword("BackgroundLayer", Factory::standardCreate<BackgroundLayer>);
   registerKeyword("BinningLayer", Factory::standardCreate<BinningLayer>);
   registerKeyword("CloneVLayer", Factory::standardCreate<CloneVLayer>);
   registerKeyword("ConstantLayer", Factory::standardCreate<ConstantLayer>);
   registerKeyword("FilenameParsingGroundTruthLayer", Factory::standardCreate<FilenameParsingGroundTruthLayer>);
   registerKeyword("GapLayer", Factory::standardCreate<GapLayer>);
   registerKeyword("HyPerLayer", Factory::standardCreate<HyPerLayer>);
   registerKeyword("HyPerLCALayer", Factory::standardCreate<HyPerLCALayer>);
   registerKeyword("ISTALayer", Factory::standardCreate<ISTALayer>);
   registerKeyword("Image", Factory::standardCreate<Image>);
   registerKeyword("ImagePvp", Factory::standardCreate<ImagePvp>);
   registerKeyword("ImageFromMemoryBuffer", Factory::standardCreate<ImageFromMemoryBuffer>);
   registerKeyword("KmeansLayer", Factory::standardCreate<KmeansLayer>);
   registerKeyword("LCALIFLayer", Factory::standardCreate<LCALIFLayer>);
   registerKeyword("LIF", Factory::standardCreate<LIF>);
   registerKeyword("LIFGap", Factory::standardCreate<LIFGap>);
   registerKeyword("LabelErrorLayer", Factory::standardCreate<LabelErrorLayer>);
   registerKeyword("LabelLayer", Factory::standardCreate<LabelLayer>);
   registerKeyword("LeakyIntegrator", Factory::standardCreate<LeakyIntegrator>);
   registerKeyword("MaskLayer", Factory::standardCreate<MaskLayer>);
   registerKeyword("MomentumLCALayer", Factory::standardCreate<MomentumLCALayer>);
   registerKeyword("Movie", Factory::standardCreate<Movie>);
   registerKeyword("MoviePvp", Factory::standardCreate<MoviePvp>);
   registerKeyword("Patterns", Factory::standardCreate<Patterns>);
   registerKeyword("PoolingIndexLayer", Factory::standardCreate<PoolingIndexLayer>);
   registerKeyword("PtwiseLinearTransferLayer", Factory::standardCreate<PtwiseLinearTransferLayer>);
   registerKeyword("PtwiseProductLayer", Factory::standardCreate<PtwiseProductLayer>);
   registerKeyword("PtwiseQuotientLayer", Factory::standardCreate<PtwiseQuotientLayer>);
   registerKeyword("RescaleLayer", Factory::standardCreate<RescaleLayer>);
   registerKeyword("RunningAverageLayer", Factory::standardCreate<RunningAverageLayer>);
   registerKeyword("Retina", Factory::standardCreate<Retina>);
   registerKeyword("ShuffleLayer", Factory::standardCreate<ShuffleLayer>);
   registerKeyword("SigmoidLayer", Factory::standardCreate<SigmoidLayer>);
   registerKeyword("WTALayer", Factory::standardCreate<WTALayer>);

   registerKeyword("HyPerConn", Factory::standardCreate<HyPerConn>);
   registerKeyword("CloneConn", Factory::standardCreate<CloneConn>);
   registerKeyword("CloneKernelConn", Factory::standardCreate<CloneKernelConn>);
   registerKeyword("CopyConn", Factory::standardCreate<CopyConn>);
   registerKeyword("FeedbackConn", Factory::standardCreate<FeedbackConn>);
   registerKeyword("GapConn", Factory::standardCreate<GapConn>);
   registerKeyword("IdentConn", Factory::standardCreate<IdentConn>);
   registerKeyword("ImprintConn", Factory::standardCreate<ImprintConn>);
   registerKeyword("KernelConn", Factory::standardCreate<KernelConn>);
   registerKeyword("MomentumConn", Factory::standardCreate<MomentumConn>);
   registerKeyword("PlasticCloneConn", Factory::standardCreate<PlasticCloneConn>);
   registerKeyword("PoolingConn", Factory::standardCreate<PoolingConn>);
   registerKeyword("RescaleConn", Factory::standardCreate<RescaleConn>);
   registerKeyword("TransposeConn", Factory::standardCreate<TransposeConn>);
   registerKeyword("TransposePoolingConn", Factory::standardCreate<TransposePoolingConn>);

   registerKeyword("ColumnEnergyProbe", Factory::standardCreate<ColumnEnergyProbe>);
   registerKeyword("FirmThresholdCostFnLCAProbe", Factory::standardCreate<FirmThresholdCostFnLCAProbe>);
   registerKeyword("FirmThresholdCostFnProbe", Factory::standardCreate<FirmThresholdCostFnProbe>);
   registerKeyword("KernelProbe", Factory::standardCreate<KernelProbe>);
   registerKeyword("L0NormLCAProbe", Factory::standardCreate<L0NormLCAProbe>);
   registerKeyword("L0NormProbe", Factory::standardCreate<L0NormProbe>);
   registerKeyword("L1NormLCAProbe", Factory::standardCreate<L1NormLCAProbe>);
   registerKeyword("L1NormProbe", Factory::standardCreate<L1NormProbe>);
   registerKeyword("L2NormProbe", Factory::standardCreate<L2NormProbe>);
   registerKeyword("PointLIFProbe", Factory::standardCreate<PointLIFProbe>);
   registerKeyword("PointProbe", Factory::standardCreate<PointProbe>);
   registerKeyword("QuotientColProbe", Factory::standardCreate<QuotientColProbe>);
   registerKeyword("RequireAllZeroActivityProbe", Factory::standardCreate<RequireAllZeroActivityProbe>);
   registerKeyword("StatsProbe", Factory::standardCreate<StatsProbe>);

   registerKeyword("Gauss2DWeight", Factory::standardCreate<InitGauss2DWeights>);
   registerKeyword("CoCircWeight", Factory::standardCreate<InitCocircWeights>);
   registerKeyword("UniformWeight", Factory::standardCreate<InitUniformWeights>);
   registerKeyword("SmartWeight", Factory::standardCreate<InitSmartWeights>);
   registerKeyword("UniformRandomWeight", Factory::standardCreate<InitUniformRandomWeights>);
   registerKeyword("GaussianRandomWeight", Factory::standardCreate<InitGaussianRandomWeights>);
   registerKeyword("IdentWeight", Factory::standardCreate<InitIdentWeights>);
   registerKeyword("OneToOneWeights", Factory::standardCreate<InitOneToOneWeights>);
   registerKeyword("OneToOneWeightsWithDelays", Factory::standardCreate<InitOneToOneWeightsWithDelays>);
   registerKeyword("SpreadOverArborsWeight", Factory::standardCreate<InitSpreadOverArborsWeights>);
   registerKeyword("MaxPoolingWeight", Factory::standardCreate<InitMaxPoolingWeights>);
   registerKeyword("FileWeight", Factory::standardCreate<InitWeights>);

   registerKeyword("normalizeContrastZeroMean", Factory::standardCreate<NormalizeContrastZeroMean>);
   registerKeyword("normalizeL2", Factory::standardCreate<NormalizeL2>);
   registerKeyword("normalizeMax", Factory::standardCreate<NormalizeMax>);
   registerKeyword("normalizeSum", Factory::standardCreate<NormalizeSum>);
   registerKeyword("normalizeGroup", Factory::standardCreate<NormalizeGroup>);

   return PV_SUCCESS;
}

int Factory::copyKeywordHandlerList(std::vector<KeywordHandler*> const& orig) {
   for (auto& kh : orig) {
      registerKeyword(kh->getKeyword(), kh->getCreator());
   }
   return PV_SUCCESS;
}

int Factory::registerKeyword(char const * keyword, ObjectCreateFn creator) {
   KeywordHandler const * keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler != NULL) {
      return PV_FAILURE;
   }
   KeywordHandler * newKeyword = new KeywordHandler(keyword, creator);
   keywordHandlerList.push_back(newKeyword);
   return PV_SUCCESS;
}

BaseObject * Factory::create(char const * keyword, char const * name, HyPerCol * hc) const {
   KeywordHandler const * keywordHandler = getKeywordHandler(keyword);
   return keywordHandler ? keywordHandler->create(name, hc) : NULL;
}

KeywordHandler const * Factory::getKeywordHandler(char const * keyword) const {
   for (auto& typeCreator : keywordHandlerList) {
      if (!strcmp(typeCreator->getKeyword(), keyword)) {
         return typeCreator;
      }
   }
   return NULL;
}

int Factory::clearKeywordHandlerList() {
   for (auto& kh : keywordHandlerList) {
      delete kh;
   }
   keywordHandlerList.clear();
   return PV_SUCCESS;
}


Factory::~Factory() {
   clearKeywordHandlerList();
}

} /* namespace PV */
