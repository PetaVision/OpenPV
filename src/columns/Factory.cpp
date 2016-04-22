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
#include <layers/ANNNormalizedErrorLayer.hpp>
#include <layers/ANNSquaredLayer.hpp>
#include <layers/ANNWhitenedLayer.hpp>
#include <layers/BackgroundLayer.hpp>
#include <layers/BinningLayer.hpp>
#include <layers/CloneVLayer.hpp>
#include <layers/ConstantLayer.hpp>
#include <layers/FilenameParsingGroundTruthLayer.hpp>
#include <layers/GapLayer.hpp>
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

#include <normalizers/NormalizeBase.hpp>
#include <normalizers/NormalizeContrastZeroMean.hpp>
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
}


int Factory::registerCoreKeywords() {
   keywordHandlerList = std::vector<KeywordHandler*>();

   registerKeyword("ANNErrorLayer", createANNErrorLayer);
   registerKeyword("ANNLayer", createANNLayer);
   registerKeyword("ANNNormalizedErrorLayer", createANNNormalizedErrorLayer);
   registerKeyword("ANNSquaredLayer", createANNSquaredLayer);
   registerKeyword("ANNWhitenedLayer", createANNWhitenedLayer);
   registerKeyword("BackgroundLayer", createBackgroundLayer);
   registerKeyword("BinningLayer", createBinningLayer);
   registerKeyword("CloneVLayer", createCloneVLayer);
   registerKeyword("ConstantLayer", createConstantLayer);
   registerKeyword("FilenameParsingGroundTruthLayer", createFilenameParsingGroundTruthLayer);
   registerKeyword("GapLayer", createGapLayer);
   registerKeyword("HyPerLCALayer", createHyPerLCALayer);
   registerKeyword("ISTALayer", createISTALayer);
   registerKeyword("Image", createImage);
   registerKeyword("ImagePvp", createImagePvp);
   registerKeyword("ImageFromMemoryBuffer", createImageFromMemoryBuffer);
   registerKeyword("KmeansLayer", createKmeansLayer);
   registerKeyword("LCALIFLayer", createLCALIFLayer);
   registerKeyword("LIF", createLIF);
   registerKeyword("LIFGap", createLIFGap);
   registerKeyword("LabelErrorLayer", createLabelErrorLayer);
   registerKeyword("LabelLayer", createLabelLayer);
   registerKeyword("LeakyIntegrator", createLeakyIntegrator);
   registerKeyword("MaskLayer", createMaskLayer);
   registerKeyword("MomentumLCALayer", createMomentumLCALayer);
   registerKeyword("Movie", createMovie);
   registerKeyword("MoviePvp", createMoviePvp);
   registerKeyword("Patterns", createPatterns);
   registerKeyword("PoolingIndexLayer", createPoolingIndexLayer);
   registerKeyword("PtwiseLinearTransferLayer", createPtwiseLinearTransferLayer);
   registerKeyword("PtwiseProductLayer", createPtwiseProductLayer);
   registerKeyword("RescaleLayer", createRescaleLayer);
   registerKeyword("RunningAverageLayer", createRunningAverageLayer);
   registerKeyword("Retina", createRetina);
   registerKeyword("ShuffleLayer", createShuffleLayer);
   registerKeyword("SigmoidLayer", createSigmoidLayer);
   registerKeyword("WTALayer", createWTALayer);

   registerKeyword("HyPerConn", createHyPerConn);
   registerKeyword("CloneConn", createCloneConn);
   registerKeyword("CloneKernelConn", createCloneKernelConn);
   registerKeyword("CopyConn", createCopyConn);
   registerKeyword("FeedbackConn", createFeedbackConn);
   registerKeyword("GapConn", createGapConn);
   registerKeyword("IdentConn", createIdentConn);
   registerKeyword("ImprintConn", createImprintConn);
   registerKeyword("KernelConn", createKernelConn);
   registerKeyword("MomentumConn", createMomentumConn);
   registerKeyword("PlasticCloneConn", createPlasticCloneConn);
   registerKeyword("PoolingConn", createPoolingConn);
   registerKeyword("RescaleConn", createRescaleConn);
   registerKeyword("TransposeConn", createTransposeConn);
   registerKeyword("TransposePoolingConn", createTransposePoolingConn);

   registerKeyword("ColumnEnergyProbe", createColumnEnergyProbe);
   registerKeyword("FirmThresholdCostFnLCAProbe", createFirmThresholdCostFnLCAProbe);
   registerKeyword("FirmThresholdCostFnProbe", createFirmThresholdCostFnProbe);
   registerKeyword("KernelProbe", createKernelProbe);
   registerKeyword("L0NormLCAProbe", createL0NormLCAProbe);
   registerKeyword("L0NormProbe", createL0NormProbe);
   registerKeyword("L1NormLCAProbe", createL1NormLCAProbe);
   registerKeyword("L1NormProbe", createL1NormProbe);
   registerKeyword("L2NormProbe", createL2NormProbe);
   registerKeyword("PointLIFProbe", createPointLIFProbe);
   registerKeyword("PointProbe", createPointProbe);
   registerKeyword("QuotientColProbe", createQuotientColProbe);
   registerKeyword("RequireAllZeroActivityProbe", createRequireAllZeroActivityProbe);
   registerKeyword("StatsProbe", createStatsProbe);

   registerKeyword("Gauss2DWeight", createInitGauss2DWeights);
   registerKeyword("CoCircWeight", createInitCocircWeights);
   registerKeyword("UniformWeight", createInitUniformWeights);
   registerKeyword("SmartWeight", createInitSmartWeights);
   registerKeyword("UniformRandomWeight", createInitUniformRandomWeights);
   registerKeyword("GaussianRandomWeight", createInitGaussianRandomWeights);
   registerKeyword("IdentWeight", createInitIdentWeights);
   registerKeyword("OneToOneWeights", createInitOneToOneWeights);
   registerKeyword("OneToOneWeightsWithDelays", createInitOneToOneWeightsWithDelays);
   registerKeyword("SpreadOverArborsWeight", createInitSpreadOverArborsWeights);
   registerKeyword("MaxPoolingWeight", createInitMaxPoolingWeights);
   registerKeyword("FileWeight", createInitWeights);

   registerKeyword("normalizeContrastZeroMean", createNormalizeContrastZeroMean);
   registerKeyword("normalizeL2", createNormalizeL2);
   registerKeyword("normalizeMax", createNormalizeMax);
   registerKeyword("normalizeSum", createNormalizeSum);
   registerKeyword("normalizeGroup", Factory::createNull);
   registerKeyword("none", Factory::createNull);
   registerKeyword("", Factory::createNull);

   return PV_SUCCESS;
}

int Factory::copyKeywordHandlerList(std::vector<KeywordHandler*> const& orig) {
   for (std::vector<KeywordHandler*>::const_iterator iter = orig.begin(); iter < orig.end(); iter++) {
      KeywordHandler* kh = *iter;
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
   for (std::vector<KeywordHandler*>::const_iterator iter = keywordHandlerList.begin(); iter < keywordHandlerList.end(); iter++) {
      KeywordHandler * typeCreator = *iter;
      if (!strcmp(typeCreator->getKeyword(), keyword)) {
         return typeCreator;
      }
   }
   return NULL;
}

int Factory::clearKeywordHandlerList() {
   for (std::vector<KeywordHandler*>::const_iterator iter = keywordHandlerList.begin(); iter < keywordHandlerList.end(); iter++) {
      delete *iter;
   }
   keywordHandlerList.clear();
   return PV_SUCCESS;
}


Factory::~Factory() {
   clearKeywordHandlerList();
}

} /* namespace PV */
