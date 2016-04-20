/*
 * CoreParamGroupHandler.cpp
 *
 *  Created on: Jan 5, 2015
 *      Author: pschultz
 */

// Note: ParamGroupHandler and functions that depend on it were deprecated
// on March 24, 2016.  Instead, creating layers, connections, etc. should
// be handled using the PV_Init::registerKeyword, PV_Init::create, and
// PV_Init::build methods.

#include "CoreParamGroupHandler.hpp"
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include "../layers/ANNErrorLayer.hpp"
#include "../layers/ANNLayer.hpp"
#include "../layers/ANNNormalizedErrorLayer.hpp"
#include "../layers/ANNSquaredLayer.hpp"
#include "../layers/ANNWhitenedLayer.hpp"
#include "../layers/BackgroundLayer.hpp"
#include "../layers/BinningLayer.hpp"
#include "../layers/CloneVLayer.hpp"
#include "../layers/ConstantLayer.hpp"
#include "../layers/GapLayer.hpp"
#include "../layers/HyPerLCALayer.hpp"
#include "../layers/MomentumLCALayer.hpp"
#include "../layers/ISTALayer.hpp"
#include "../layers/ImagePvp.hpp"
#include "../layers/ImageFromMemoryBuffer.hpp"
#include "../layers/KmeansLayer.hpp"
#include "../layers/LCALIFLayer.hpp"
#include "../layers/LIF.hpp"
#include "../layers/LIFGap.hpp"
#include "../layers/LabelErrorLayer.hpp"
#include "../layers/LeakyIntegrator.hpp"
#include "../layers/MaskLayer.hpp"
#include "../layers/MoviePvp.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/PoolingIndexLayer.hpp"
#include "../layers/PtwiseLinearTransferLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/RescaleLayer.hpp"
#include "../layers/RunningAverageLayer.hpp"
#include "../layers/Retina.hpp"
#include "../layers/Segmentify.hpp"
#include "../layers/SegmentLayer.hpp"
#include "../layers/ShuffleLayer.hpp"
#include "../layers/SigmoidLayer.hpp"
#include "../layers/WTALayer.hpp"
#ifdef PV_USE_GDAL
#   include "../layers/FilenameParsingGroundTruthLayer.hpp"
#   include "../layers/Image.hpp"
#   include "../layers/LabelLayer.hpp"
#   include "../layers/Movie.hpp"
#endif // PV_USE_GDAL

#include "../connections/HyPerConn.hpp"
#include "../connections/CloneConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/CopyConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/ImprintConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/MomentumConn.hpp"
#include "../connections/PlasticCloneConn.hpp"

#include "../connections/PoolingConn.hpp"
#include "../connections/RescaleConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "../connections/TransposePoolingConn.hpp"

#include "ColumnEnergyProbe.hpp"
#include "QuotientColProbe.hpp"
#include "FirmThresholdCostFnLCAProbe.hpp"
#include "FirmThresholdCostFnProbe.hpp"
#include "L0NormLCAProbe.hpp"
#include "L0NormProbe.hpp"
#include "L1NormLCAProbe.hpp"
#include "L1NormProbe.hpp"
#include "L2NormProbe.hpp"
#include "PointLIFProbe.hpp"
#include "PointProbe.hpp"
#include "RequireAllZeroActivityProbe.hpp"
#include "StatsProbe.hpp"
#include "KernelProbe.hpp"
#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitGauss2DWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"
#include "../weightinit/InitOneToOneWeights.hpp"
#include "../weightinit/InitOneToOneWeightsWithDelays.hpp"
#include "../weightinit/InitIdentWeights.hpp"
#include "../weightinit/InitUniformWeights.hpp"
#include "../weightinit/InitSpreadOverArborsWeights.hpp"
#include "../weightinit/InitMaxPoolingWeights.hpp"
#include "../normalizers/NormalizeBase.hpp"
#include "../normalizers/NormalizeSum.hpp"
#include "../normalizers/NormalizeL2.hpp"
#include "../normalizers/NormalizeMax.hpp"
#include "../normalizers/NormalizeContrastZeroMean.hpp"

namespace PV {

CoreParamGroupHandler::CoreParamGroupHandler() {
}

CoreParamGroupHandler::~CoreParamGroupHandler() {
}

ParamGroupType CoreParamGroupHandler::getGroupType(char const * keyword) {
   struct keyword_grouptype_entry  {char const * kw; ParamGroupType type;};
   struct keyword_grouptype_entry keywordtable[] = {
         // HyPerCol
         {"HyPerCol", HyPerColGroupType},

         // Layers
         {"HyPerLayer", LayerGroupType},
         {"ANNErrorLayer", LayerGroupType},
         {"ANNLayer", LayerGroupType},
         {"ANNNormalizedErrorLayer", LayerGroupType},
         {"ANNSquaredLayer", LayerGroupType},
         {"ANNTriggerUpdateOnNewImageLayer", LayerGroupType},
         {"ANNWhitenedLayer", LayerGroupType},
         {"BackgroundLayer", LayerGroupType},
         {"BinningLayer", LayerGroupType},
         {"CloneVLayer", LayerGroupType},
         {"ConstantLayer", LayerGroupType},
         {"GapLayer", LayerGroupType},
         {"HyPerLCALayer", LayerGroupType},
         {"MomentumLCALayer", LayerGroupType},
	 {"ISTALayer", LayerGroupType},
         {"ImagePvp", LayerGroupType},
         {"ImageFromMemoryBuffer", LayerGroupType},
         {"KmeansLayer", LayerGroupType},
         {"LCALIFLayer", LayerGroupType},
         {"LIF", LayerGroupType},
         {"LIFGap", LayerGroupType},
         {"LabelErrorLayer", LayerGroupType},
         {"LeakyIntegrator", LayerGroupType},
         {"MaskLayer", LayerGroupType},
         {"MaxPooling", LayerGroupType},
         {"MoviePvp", LayerGroupType},
         {"Patterns", LayerGroupType},
         {"PoolingIndexLayer", LayerGroupType},
         {"PtwiseLinearTransferLayer", LayerGroupType},
         {"PtwiseProductLayer", LayerGroupType},
         {"RescaleLayer", LayerGroupType},
         {"RunningAverageLayer", LayerGroupType},
         {"Retina", LayerGroupType},
         {"Segmentify", LayerGroupType},
         {"SegmentLayer", LayerGroupType},
         {"ShuffleLayer", LayerGroupType},
         {"SigmoidLayer", LayerGroupType},
//         {"TrainingLayer", LayerGroupType}, //Marked obsolete June 17, 2015
         {"WTALayer", LayerGroupType},
#ifdef PV_USE_GDAL
         {"FilenameParsingGroundTruthLayer", LayerGroupType},
         {"Image", LayerGroupType},
         {"LabelLayer", LayerGroupType},
         {"Movie", LayerGroupType},
#endif // PV_USE_GDAL

         // Connections
         {"HyPerConn", ConnectionGroupType},
         {"CloneConn", ConnectionGroupType},
         {"CloneKernelConn", ConnectionGroupType},
         {"CopyConn", ConnectionGroupType},
         {"FeedbackConn", ConnectionGroupType},
         {"GapConn", ConnectionGroupType},
         {"IdentConn", ConnectionGroupType},
         {"ImprintConn", ConnectionGroupType},
         {"KernelConn", ConnectionGroupType},
         {"MomentumConn", ConnectionGroupType},
         {"PlasticCloneConn", ConnectionGroupType},
         {"PoolingConn", ConnectionGroupType},
         {"RescaleConn", ConnectionGroupType},
         {"TransposeConn", ConnectionGroupType},
         {"TransposePoolingConn", ConnectionGroupType},

         // Probes

         // ColProbes
         {"ColumnEnergyProbe", ColProbeGroupType},
         {"QuotientColProbe", ColProbeGroupType},

         // // Layer probes
         {"LayerProbe", ProbeGroupType},
         {"FirmThresholdCostFnLCAProbe", ProbeGroupType},
         {"FirmThresholdCostFnProbe", ProbeGroupType},
         {"L0NormLCAProbe", ProbeGroupType},
         {"L0NormProbe", ProbeGroupType},
         {"L1NormLCAProbe", ProbeGroupType},
         {"L1NormProbe", ProbeGroupType},
         {"L2NormProbe", ProbeGroupType},
         {"PointLIFProbe", ProbeGroupType},
         {"PointProbe", ProbeGroupType},
         {"RequireAllZeroActivityProbe", ProbeGroupType},
         {"StatsProbe", ProbeGroupType},

         // // Connection probes
         {"KernelProbe", ProbeGroupType},

         // Weight initializers
         {"Gauss2DWeight", WeightInitializerGroupType},
         {"CoCircWeight", WeightInitializerGroupType},
         {"UniformWeight", WeightInitializerGroupType},
         {"SmartWeight", WeightInitializerGroupType},
         {"UniformRandomWeight", WeightInitializerGroupType},
         {"GaussianRandomWeight", WeightInitializerGroupType},
         {"IdentWeight", WeightInitializerGroupType},
         {"OneToOneWeights", WeightInitializerGroupType},
         {"OneToOneWeightsWithDelays", WeightInitializerGroupType},
         {"SpreadOverArborsWeight", WeightInitializerGroupType},
         {"MaxPoolingWeight", WeightInitializerGroupType},
         {"FileWeight", WeightInitializerGroupType},

         {"normalizeSum", WeightNormalizerGroupType},
         {"normalizeL2", WeightNormalizerGroupType},
         {"normalizeMax", WeightNormalizerGroupType},
         {"normalizeContrastZeroMean", WeightNormalizerGroupType},
         {"normalizeGroup", WeightNormalizerGroupType},
         {"none", WeightNormalizerGroupType},

         {NULL, UnrecognizedGroupType}
   };
   ParamGroupType result = UnrecognizedGroupType;
   if (keyword==NULL) { return result; }
   for (int k=0; keywordtable[k].kw != NULL; k++) {
      if (!strcmp(keywordtable[k].kw, keyword)) {
         result = keywordtable[k].type;
         break;
      }
   }
   return result;
}

HyPerCol * CoreParamGroupHandler::createHyPerCol(char const * keyword, char const * name, HyPerCol * hc) {
   HyPerCol * addedHyPerCol = NULL;
   if ( keyword && !strcmp(keyword, "HyPerCol")) {
      addedHyPerCol = hc;
      if (dynamic_cast<HyPerCol *>(hc)==NULL) {
         if (hc->columnId()==0) {
            fprintf(stderr, "createHyPerCol error: unable to add %s\n", keyword);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   return addedHyPerCol;
}

HyPerLayer * CoreParamGroupHandler::createLayer(char const * keyword, char const * name, HyPerCol * hc) {
   HyPerLayer * addedLayer = NULL;
   if (keyword==NULL) {
      addedLayer = NULL;
   }
   else if( !strcmp(keyword, "HyPerLayer") ) {
      fprintf(stderr, "Group \"%s\": abstract class HyPerLayer cannot be instantiated.\n", name);
      addedLayer = NULL;
   }
   else if( !strcmp(keyword, "ANNErrorLayer") ) {
      addedLayer = new ANNErrorLayer(name, hc);
   }
   else if( !strcmp(keyword, "ANNLayer") ) {
      addedLayer = new ANNLayer(name, hc);
   }
   else if( !strcmp(keyword, "ANNNormalizedErrorLayer") ) {
      addedLayer = new ANNNormalizedErrorLayer(name, hc);
   }
   else if( !strcmp(keyword, "ANNSquaredLayer") ) {
      addedLayer = new ANNSquaredLayer(name, hc);
   }
      if( !strcmp(keyword, "ANNTriggerUpdateOnNewImageLayer") ) {
         // ANNTriggerUpdateOnNewImageLayer is obsolete as of April 23, 2014.  Leaving it in the code for a while for a useful error message.
         // Use ANNLayer with triggerFlag set to true and triggerLayerName for the triggering layer
         if (hc->columnId()==0) {
            fprintf(stderr, "Error: ANNTriggerUpdateOnNewImageLayer is obsolete.\n");
            fprintf(stderr, "    Use ANNLayer with parameter triggerFlag set to true\n");
            fprintf(stderr, "    and triggerLayerName set to the triggering layer.\n");
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   else if( !strcmp(keyword, "ANNWhitenedLayer") ) {
      addedLayer = new ANNWhitenedLayer(name, hc);
   }
   else if( !strcmp(keyword, "BackgroundLayer") ) {
      addedLayer = new BackgroundLayer(name, hc);
   }
   else if( !strcmp(keyword, "BinningLayer") ) {
      addedLayer = new BinningLayer(name, hc);
   }
   else if( !strcmp(keyword, "CloneVLayer") ) {
      addedLayer = new CloneVLayer(name, hc);
   }
   else if( !strcmp(keyword, "ConstantLayer") ) {
      addedLayer = new ConstantLayer(name, hc);
   }
   else if( !strcmp(keyword, "GapLayer") ) {
      addedLayer = new GapLayer(name, hc);
   }
   else if( !strcmp(keyword, "HyPerLCALayer") ) {
      addedLayer = new HyPerLCALayer(name, hc);
   }
   else if( !strcmp(keyword, "MomentumLCALayer") ) {
      addedLayer = new MomentumLCALayer(name, hc);
   }
   else if( !strcmp(keyword, "ISTALayer") ) {
     addedLayer = new ISTALayer(name, hc);
   }
   else if( !strcmp(keyword, "ImagePvp") ) {
      addedLayer = new ImagePvp(name, hc);
   }
   else if( !strcmp(keyword, "ImageFromMemoryBuffer") ) {
      addedLayer = new ImageFromMemoryBuffer(name, hc);
   }
   else if( !strcmp(keyword, "KmeansLayer") ) {
      addedLayer = new KmeansLayer(name, hc);
   }
   else if( !strcmp(keyword, "LCALIFLayer") ) {
      addedLayer = new LCALIFLayer(name, hc);
   }
   else if( !strcmp(keyword, "LIF") ) {
      addedLayer = new LIF(name, hc);
   }
   else if( !strcmp(keyword, "LIFGap") ) {
      addedLayer = new LIFGap(name, hc);
   }
   else if( !strcmp(keyword, "LabelErrorLayer") ) {
      addedLayer = new LabelErrorLayer(name, hc);
   }
   else if( !strcmp(keyword, "LeakyIntegrator") ) {
      addedLayer = new LeakyIntegrator(name, hc);
   }
   else if( !strcmp(keyword, "MaskLayer") ) {
      addedLayer = new MaskLayer(name, hc);
   }
   else if( !strcmp(keyword, "MoviePvp") ) {
      addedLayer = new MoviePvp(name, hc);
   }
   else if( !strcmp(keyword, "Patterns") ) {
      addedLayer = new Patterns(name, hc);
   }
   else if( !strcmp(keyword, "PoolingIndexLayer") ) {
      addedLayer = new PoolingIndexLayer(name, hc);
   }
   else if( !strcmp(keyword, "PtwiseLinearTransferLayer") ) {
      addedLayer = new PtwiseLinearTransferLayer(name, hc);
   }
   else if (!strcmp(keyword, "PtwiseProductLayer") ) {
      addedLayer = new PtwiseProductLayer(name, hc);
   }
   else if( !strcmp(keyword, "RescaleLayer") ) {
      addedLayer = new RescaleLayer(name, hc);
   }
   else if( !strcmp(keyword, "RunningAverageLayer") ) {
      addedLayer = new RunningAverageLayer(name, hc);
   }
   else if( !strcmp(keyword, "Retina") ) {
      addedLayer = new Retina(name, hc);
   }
   else if( !strcmp(keyword, "Segmentify") ) {
      addedLayer = new Segmentify(name, hc);
   }
   else if( !strcmp(keyword, "SegmentLayer") ) {
      addedLayer = new SegmentLayer(name, hc);
   }
   else if( !strcmp(keyword, "ShuffleLayer") ) {
      addedLayer = new ShuffleLayer(name, hc);
   }
   else if( !strcmp(keyword, "SigmoidLayer") ) {
      addedLayer = new SigmoidLayer(name, hc);
   }
   else if( !strcmp(keyword, "WTALayer") ) {
      addedLayer = new WTALayer(name, hc);
   }
#ifdef PV_USE_GDAL
   else if( !strcmp(keyword, "FilenameParsingGroundTruthLayer") ) {
      addedLayer = new FilenameParsingGroundTruthLayer(name, hc);
   }
  else if( !strcmp(keyword, "Image") ) {
     addedLayer = new Image(name, hc);
  }
  else if( !strcmp(keyword, "LabelLayer") ) {
     addedLayer = new LabelLayer(name, hc);
  }
  else if( !strcmp(keyword, "Movie") ) {
     addedLayer = new Movie(name, hc);
  }
#endif // PV_USE_GDAL

   if (addedLayer==NULL && getGroupType(keyword)==LayerGroupType) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createLayer error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return addedLayer;
}

BaseConnection * CoreParamGroupHandler::createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   BaseConnection * addedConnection = NULL;

   if (keyword==NULL) {
      addedConnection = NULL;
   }
   else if( !strcmp(keyword, "HyPerConn") ) {
      addedConnection = new HyPerConn(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "CloneConn") ) {
      addedConnection = new CloneConn(name, hc);
   }
   else if( !strcmp(keyword, "CloneKernelConn") ) {
      // Deprecated as of June 6, 2014.  Use CloneConn with sharedWeight = true
      addedConnection = new CloneKernelConn(name, hc);
   }
   else if( !strcmp(keyword, "CopyConn") ) {
      addedConnection = new CopyConn(name, hc, weightNormalizer);
   }
   else if( !strcmp(keyword, "FeedbackConn") ) {
      addedConnection = new FeedbackConn(name, hc);
   }
   else if( !strcmp(keyword, "GapConn") ) {
      addedConnection = new GapConn(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "IdentConn") ) {
      addedConnection = new IdentConn(name, hc);
   }
   else if( !strcmp(keyword, "ImprintConn") ) {
      addedConnection = new ImprintConn(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "KernelConn") ) {
      // Deprecated as of June 5, 2014.  Use HyPerConn with sharedWeight = true
      addedConnection = new KernelConn(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "MomentumConn") ) {
      addedConnection = new MomentumConn(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "PlasticCloneConn") ) {
      addedConnection = new PlasticCloneConn(name, hc);
   }
   else if( !strcmp(keyword, "PoolingConn") ) {
      addedConnection = new PoolingConn(name, hc);
   }
   else if( !strcmp(keyword, "RescaleConn") ) {
      addedConnection = new RescaleConn(name, hc);
   }
   else if( !strcmp(keyword, "TransposeConn") ) {
      addedConnection = new TransposeConn(name, hc);
   }
   else if( !strcmp(keyword, "TransposePoolingConn") ) {
      addedConnection = new TransposePoolingConn(name, hc);
   }

   if (addedConnection==NULL &&getGroupType(keyword)==ConnectionGroupType) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createConnection error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return addedConnection;
}

ColProbe * CoreParamGroupHandler::createColProbe(char const * keyword, char const * name, HyPerCol * hc) {
   ColProbe * addedColProbe = NULL;

   if (keyword==NULL) {
      addedColProbe = NULL;
   }
   else if (!strcmp(keyword, "ColumnEnergyProbe")) {
      addedColProbe = new ColumnEnergyProbe(name, hc);
   }
   else if (!strcmp(keyword, "QuotientColProbe")) {
      addedColProbe = new QuotientColProbe(name, hc);
   }

   if (addedColProbe==NULL && getGroupType(keyword)==ColProbeGroupType) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createColProbe error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return addedColProbe;
}

BaseProbe * CoreParamGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   BaseProbe * addedProbe = NULL;

   // Layer probe keywords
   if (keyword==NULL) {
      addedProbe = NULL;
   }
   else if( !strcmp(keyword, "LayerProbe") ) {
      fprintf(stderr, "LayerProbe \"%s\": Abstract class LayerProbe cannot be instantiated.\n", name);
      addedProbe = NULL;
   }
   else if( !strcmp(keyword, "FirmThresholdCostFnLCAProbe") ) {
      addedProbe = new FirmThresholdCostFnLCAProbe(name, hc);
   }
   else if( !strcmp(keyword, "FirmThresholdCostFnProbe") ) {
      addedProbe = new FirmThresholdCostFnProbe(name, hc);
   }
   else if( !strcmp(keyword, "L0NormLCAProbe") ) {
      addedProbe = new L0NormLCAProbe(name, hc);
   }
   else if( !strcmp(keyword, "L0NormProbe") ) {
      addedProbe = new L0NormProbe(name, hc);
   }
   else if( !strcmp(keyword, "L1NormLCAProbe") ) {
      addedProbe = new L1NormLCAProbe(name, hc);
   }
   else if( !strcmp(keyword, "L1NormProbe") ) {
      addedProbe = new L1NormProbe(name, hc);
   }
   else if( !strcmp(keyword, "L2NormProbe") ) {
      addedProbe = new L2NormProbe(name, hc);
   }
   else if( !strcmp(keyword, "PointLIFProbe") ) {
      addedProbe = new PointLIFProbe(name, hc);
   }
   else if( !strcmp(keyword, "PointProbe") ) {
      addedProbe = new PointProbe(name, hc);
   }
   else if( !strcmp(keyword, "RequireAllZeroActivityProbe") ) {
      addedProbe = new RequireAllZeroActivityProbe(name, hc);
   }
   else if( !strcmp(keyword, "StatsProbe") ) {
      addedProbe = new StatsProbe(name, hc);
   }

   // Connection probe keywords
   else if( !strcmp(keyword, "KernelProbe") ) {
      addedProbe = new KernelProbe(name, hc);
   }

   if (addedProbe==NULL && getGroupType(keyword)==ProbeGroupType) {
         if (hc->columnId()==0) {
            fprintf(stderr, "createProbe error: unable to add %s\n", keyword);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
   }

   return addedProbe;
}

InitWeights * CoreParamGroupHandler::createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc) {
   InitWeights * weightInitializer = NULL;

   if (keyword==NULL) {
      weightInitializer = NULL;
   }
   else if (!strcmp(keyword, "Gauss2DWeight")) {
      weightInitializer = new InitGauss2DWeights(name, hc);
   }
   else if (!strcmp(keyword, "CoCircWeight")) {
      weightInitializer = new InitCocircWeights(name, hc);
   }
   else if (!strcmp(keyword, "UniformWeight")) {
      weightInitializer = new InitUniformWeights(name, hc);
   }
   else if (!strcmp(keyword, "SmartWeight")) {
      weightInitializer = new InitSmartWeights(name, hc);
   }
   else if (!strcmp(keyword, "UniformRandomWeight")) {
      weightInitializer = new InitUniformRandomWeights(name, hc);
   }
   else if (!strcmp(keyword, "GaussianRandomWeight")) {
      weightInitializer = new InitGaussianRandomWeights(name, hc);
   }
   else if (!strcmp(keyword, "IdentWeight")) {
      weightInitializer = new InitIdentWeights(name, hc);
   }
   else if (!strcmp(keyword, "OneToOneWeights")) {
      weightInitializer = new InitOneToOneWeights(name, hc);
   }
   else if (!strcmp(keyword, "OneToOneWeightsWithDelays")) {
      weightInitializer = new InitOneToOneWeightsWithDelays(name, hc);
   }
   else if (!strcmp(keyword, "SpreadOverArborsWeight")) {
      weightInitializer = new InitSpreadOverArborsWeights(name, hc);
   }
   else if (!strcmp(keyword, "MaxPoolingWeight")) {
      weightInitializer = new InitMaxPoolingWeights(name, hc);
   }
   else if (!strcmp(keyword, "FileWeight")) {
      weightInitializer = new InitWeights(name, hc);
   }

   if (weightInitializer==NULL && getGroupType(keyword)==WeightInitializerGroupType) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createWeightInitializer error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return weightInitializer;
}

NormalizeBase * CoreParamGroupHandler::createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc) {
   NormalizeBase * weightNormalizer = NULL;
   bool newNormalizer = false;

   if (keyword==NULL) {
      weightNormalizer = NULL;
   }
   else if (!strcmp(keyword, "normalizeSum")) {
      newNormalizer = true;
      weightNormalizer = new NormalizeSum(name, hc);
   }
   else if (!strcmp(keyword, "normalizeL2")) {
      newNormalizer = true;
      weightNormalizer = new NormalizeL2(name, hc);
   }
   else if (!strcmp(keyword, "normalizeMax")) {
      newNormalizer = true;
      weightNormalizer = new NormalizeMax(name, hc);
   }
   else if (!strcmp(keyword, "normalizeContrastZeroMean")) {
      newNormalizer = true;
      weightNormalizer = new NormalizeContrastZeroMean(name, hc);
   }
   else if (!strcmp(keyword, "normalizeGroup")) {
      newNormalizer = false;
      weightNormalizer = NULL;
   }
   else if (!strcmp(keyword, "") || !strcmp(keyword, "none")) {
      newNormalizer = false;
      weightNormalizer = NULL;
   }

   if (weightNormalizer==NULL && newNormalizer) {
      assert(getGroupType(keyword)==WeightNormalizerGroupType);
      if (hc->columnId()==0) {
         fprintf(stderr, "createWeightInitializer error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return weightNormalizer;
}

} /* namespace PV */
