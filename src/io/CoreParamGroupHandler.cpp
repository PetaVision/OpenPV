/*
 * CoreParamGroupHandler.cpp
 *
 *  Created on: Jan 5, 2015
 *      Author: pschultz
 */

#include "CoreParamGroupHandler.hpp"
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include "../layers/ANNErrorLayer.hpp"
#include "../layers/ANNLayer.hpp"
#include "../layers/ANNNormalizedErrorLayer.hpp"
#include "../layers/ANNSquaredLayer.hpp"
#include "../layers/ANNWhitenedLayer.hpp"
#include "../layers/BIDSCloneLayer.hpp"
#include "../layers/BIDSLayer.hpp"
#include "../layers/BIDSMovieCloneMap.hpp"
#include "../layers/BIDSSensorLayer.hpp"
#include "../layers/BinningLayer.hpp"
#include "../layers/CloneVLayer.hpp"
#include "../layers/ConstantLayer.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/FilenameParsingGroundTruthLayer.hpp"
#include "../layers/GapLayer.hpp"
#include "../layers/GenerativeLayer.hpp"
#include "../layers/HyPerLCALayer.hpp"
#include "../layers/Image.hpp"
#include "../layers/ImageFromMemoryBuffer.hpp"
#include "../layers/IncrementLayer.hpp"
#include "../layers/KmeansLayer.hpp"
#include "../layers/LCALIFLayer.hpp"
#include "../layers/LIF.hpp"
#include "../layers/LIFGap.hpp"
#include "../layers/LabelErrorLayer.hpp"
#include "../layers/LabelLayer.hpp"
#include "../layers/LeakyIntegrator.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#include "../layers/MLPErrorLayer.hpp"
#include "../layers/MLPForwardLayer.hpp"
#include "../layers/MLPOutputLayer.hpp"
#include "../layers/MLPSigmoidLayer.hpp"
#include "../layers/MatchingPursuitLayer.hpp"
#include "../layers/MatchingPursuitResidual.hpp"
#include "../layers/Movie.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/RescaleLayer.hpp"
#include "../layers/Retina.hpp"
#include "../layers/ShuffleLayer.hpp"
#include "../layers/SigmoidLayer.hpp"
#include "../layers/TextStream.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/WTALayer.hpp"
#ifdef PV_USE_SNDFILE
#include "../layers/NewCochlear.h"
#include "../layers/SoundStream.hpp"
#endif // PV_USE_SNDFILE
#include "../connections/HyPerConn.hpp"
#include "../connections/BIDSConn.hpp"
#include "../connections/CloneConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/CopyConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/LCALIFLateralConn.hpp"
#include "../connections/OjaSTDPConn.hpp"
#include "../connections/PlasticCloneConn.hpp"
#include "../connections/PoolingConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "../connections/PoolingConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "L2NormProbe.hpp"
#include "LayerFunctionProbe.hpp"
#include "LogLatWTAProbe.hpp"
#include "PointLIFProbe.hpp"
#include "PointProbe.hpp"
#include "RequireAllZeroActivityProbe.hpp"
#include "SparsityLayerProbe.hpp"
#include "StatsProbe.hpp"
#include "TextStreamProbe.hpp"
#include "KernelProbe.hpp"

namespace PV {

CoreParamGroupHandler::CoreParamGroupHandler() {
}

CoreParamGroupHandler::~CoreParamGroupHandler() {
}

void * CoreParamGroupHandler::createObject(char const * keyword, char const * name, HyPerCol * hc) {
   const char * allowedkeywordarray[] = {
         // HyPerCol
         "HyPerCol",

         // Layers
         "HyPerLayer",
         "ANNErrorLayer",
         "ANNLayer",
         "ANNNormalizedErrorLayer",
         "ANNSquaredLayer",
         "ANNTriggerUpdateOnNewImageLayer", // Leaving ANNTriggerUpdateOnNewImageLayer in for now, to provide a meaningful error message if someone tries to use it (ANNTriggerUpdateOnNewImageLayer was marked obsolete Apr 23, 2014)
         "ANNWhitenedLayer",
         "BIDSCloneLayer",
         "BIDSLayer",
         "BIDSMovieCloneMap",
         "BIDSSensorLayer",
         "BinningLayer",
         "CloneVLayer",
         "ConstantLayer",
         "CreateMovies",
         "FilenameParsingGroundTruthLayer",
         "GapLayer",
         "GenerativeLayer",
         "HyPerLCALayer",
         "Image",
         "ImageFromMemoryBuffer",
         "IncrementLayer",
         "KmeansLayer",
         "LCALIFLayer",
         "LIF",
         "LIFGap",
         "LabelErrorLayer",
         "LabelLayer",
         "LeakyIntegrator",
         "LogLatWTAGenLayer",
         "MLPErrorLayer",
         "MLPForwardLayer",
         "MLPOutputLayer",
         "MLPSigmoidLayer",
         "MatchingPursuitLayer",
         "MatchingPursuitResidual",
         "MaxPooling", // Obsolete; have the connection's pvpatchAccumulateType set to "maxpooling" (case insensitive).
         "Movie",
         "Patterns",
         "PoolingANNLayer",
         "PtwiseProductLayer",
         "RescaleLayer",
         "Retina",
         "ShuffleLayer",
         "SigmoidLayer",
         "TextStream",
         "TrainingLayer",
         "WTALayer",
#ifdef PV_USE_SNDFILE
         "NewCochlearLayer",
         "SoundStream",
#endif
#ifdef OBSOLETE // Marked obsolete Dec 29, 2014.  Removing several long-unused layers.
         "ANNDivInhLayer",
         "ANNLabelLayer",
         "ANNWeightedErrorLayer",
         "AccumulateLayer",
         "CliqueLayer",
#endif // OBSOLETE

         // Connections
         "HyPerConn",
         "BIDSConn",
         "CloneConn",
         "CloneKernelConn", //Deprecated as of June 6, 2014, in favor of CloneConn with sharedWeights = true
         "CopyConn",
         "FeedbackConn",
         "GapConn",
         "IdentConn",
         "KernelConn", // Deprecated as of June 5, 2014, in favor of HyPerConn with sharedWeights = true
         "LCALIFLateralConn",
         "OjaSTDPConn",
         "PlasticCloneConn",
         "PoolingConn",
         "TransposeConn",
#ifdef OBSOLETE // Marked obsolete Nov 25, 2014.  Use HyPerConn instead of GenerativeConn and PoolingConn instead of PoolingGenConn
         "GenerativeConn",
         "PoolingGenConn",
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Oct 20, 2014.  Normalizers are being generalized to allow for group normalization
         "NoSelfKernelConn",
         "SiblingConn",
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Nov 25, 2014.  No longer used.
         "ReciprocalConn",
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Dec 2, 2014.  Use sharedWeights=false instead of windowing.
         "WindowConn",
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Dec 29, 2014.  Removing several long-unused connections.
         "CliqueConn",
         "InhibSTDPConn",
         "LCALIFLateralKernelConn",
         "MapReduceKernelConn",
         "OjaKernelConn",
         "STDP3Conn",
         "STDPConn",
#endif // OBSOLETE

         // Column Probes
         "ColProbe",
         "GenColProbe",
         // Layer Probes
         "LayerProbe",
         "L2NormProbe",
         "LayerFunctionProbe",
         "LogLatWTAProbe",
         "PointLIFProbe",
         "PointProbe",
         "RequireAllZeroActivityProbe",
         "SparsityLayerProbe",
         "StatsProbe",
         "TextStreamProbe",
#ifdef OBSOLETE // Marked obsolete Dec 29, 2014.  Removing several long-unused probes.
         "PointLCALIFProbe",
         "SparsityTermProbe",
#endif // OBSOLETE

         // Connection Probes
         "KernelProbe",
#ifdef OBSOLETE // Marked obsolete Nov 25, 2014.  No longer used.
         "ReciprocalEnergyProbe",
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Dec 29, 2014.  Removing several long-unused probes.
         "ConnStatsProbe",
         "LCALIFLateralProbe",
         "OjaConnProbe",
         "OjaKernelSpikeRateProbe",
         "PatchProbe",
#endif // OBSOLETE
         NULL
   };
   char const * allowedkeyword;
   for (int k=0; (allowedkeyword = allowedkeywordarray[k])!=NULL; k++) {
      if (!strcmp(keyword, allowedkeyword)) { break; }
   }
   if (allowedkeyword==NULL) {
      return NULL; // Unrecognized keyword, but since there might be another ParamGroupHandler object that does recognize it, it's not an error.
   }

   void * addedObject = NULL;

   // Column keyword
   if ( !strcmp(keyword, "HyPerCol")) {
      addedObject = (void *) hc;
   }

   // Layer keywords
   if( !strcmp(keyword, "HyPerLayer") ) {
      fprintf(stderr, "Group \"%s\": abstract class HyPerLayer cannot be instantiated.\n", name);
      addedObject = NULL;
   }
   if( !strcmp(keyword, "ANNErrorLayer") ) {
      addedObject = (void *) new ANNErrorLayer(name, hc);
   }
   if( !strcmp(keyword, "ANNLayer") ) {
      addedObject = (void *) new ANNLayer(name, hc);
   }
   if( !strcmp(keyword, "ANNNormalizedErrorLayer") ) {
      addedObject = (void *) new ANNNormalizedErrorLayer(name, hc);
   }
   if( !strcmp(keyword, "ANNSquaredLayer") ) {
      addedObject = (void *) new ANNSquaredLayer(name, hc);
   }
      if( !strcmp(keyword, "ANNTriggerUpdateOnNewImageLayer") ) {
         // ANNTriggerUpdateOnNewImageLayer is obsolete as of April 23, 2014.  Leaving it in the code for a while for a useful error message.
         // Use ANNLayer with triggerFlag set to true and triggerLayerName for the triggering layer
         if (hc->columnId()==0) {
            fprintf(stderr, "Error: ANNTriggerUpdateOnNewImageLayer is obsolete.\n");
            fprintf(stderr, "    Use ANNLayer with parameter triggerFlag set to true\n");
            fprintf(stderr, "    and triggerLayerName set to the triggering layer.\n");
         }
   #ifdef PV_USE_MPI
         MPI_Barrier(hc->icCommunicator()->communicator());
   #endif
         exit(EXIT_FAILURE);
      }
   if( !strcmp(keyword, "ANNWhitenedLayer") ) {
      addedObject = (void *) new ANNWhitenedLayer(name, hc);
   }
   if( !strcmp(keyword, "BIDSCloneLayer") ) {
      addedObject = (void *) new BIDSCloneLayer(name, hc);
   }
   if( !strcmp(keyword, "BIDSLayer") ) {
      addedObject = (void *) new BIDSLayer(name, hc);
   }
   if( !strcmp(keyword, "BIDSMovieCloneMap") ) {
      addedObject = (void *) new BIDSMovieCloneMap(name, hc);
   }
   if( !strcmp(keyword, "BIDSSensorLayer") ) {
      addedObject = (void *) new BIDSSensorLayer(name, hc);
   }
   if( !strcmp(keyword, "BinningLayer") ) {
      addedObject = (void *) new BinningLayer(name, hc);
   }
   if( !strcmp(keyword, "CloneVLayer") ) {
      addedObject = (void *) new CloneVLayer(name, hc);
   }
   if( !strcmp(keyword, "ConstantLayer") ) {
      addedObject = (void *) new ConstantLayer(name, hc);
   }
   if( !strcmp(keyword, "CreateMovies") ) {
      addedObject = (void *) new CreateMovies(name, hc);
   }
   if( !strcmp(keyword, "FilenameParsingGroundTruthLayer") ) {
      addedObject = (void *) new FilenameParsingGroundTruthLayer(name, hc);
   }
   if( !strcmp(keyword, "GapLayer") ) {
      addedObject = (void *) new GapLayer(name, hc);
   }
   if( !strcmp(keyword, "GenerativeLayer") ) {
      addedObject = (void *) new GenerativeLayer(name, hc);
   }
   if( !strcmp(keyword, "HyPerLCALayer") ) {
      addedObject = (void *) new HyPerLCALayer(name, hc);
   }
   if( !strcmp(keyword, "Image") ) {
      addedObject = (void *) new Image(name, hc);
   }
   if( !strcmp(keyword, "ImageFromMemoryBuffer") ) {
      addedObject = (void *) new ImageFromMemoryBuffer(name, hc);
   }
   if( !strcmp(keyword, "IncrementLayer") ) {
      addedObject = (void *) new IncrementLayer(name, hc);
   }
   if( !strcmp(keyword, "KmeansLayer") ) {
      addedObject = (void *) new KmeansLayer(name, hc);
   }
   if( !strcmp(keyword, "LCALIFLayer") ) {
      addedObject = (void *) new LCALIFLayer(name, hc);
   }
   if( !strcmp(keyword, "LIF") ) {
      addedObject = (void *) new LIF(name, hc);
   }
   if( !strcmp(keyword, "LIFGap") ) {
      addedObject = (void *) new LIFGap(name, hc);
   }
   if( !strcmp(keyword, "LabelErrorLayer") ) {
      addedObject = (void *) new LabelErrorLayer(name, hc);
   }
   if( !strcmp(keyword, "LabelLayer") ) {
      addedObject = (void *) new LabelLayer(name, hc);
   }
   if( !strcmp(keyword, "LeakyIntegrator") ) {
      addedObject = (void *) new LeakyIntegrator(name, hc);
   }
   if( !strcmp(keyword, "LogLatWTAGenLayer") ) {
      addedObject = (void *) new LogLatWTAGenLayer(name, hc);
   }
   if( !strcmp(keyword, "MLPErrorLayer") ) {
      addedObject = (void *) new MLPErrorLayer(name, hc);
   }
   if( !strcmp(keyword, "MLPForwardLayer") ) {
      addedObject = (void *) new MLPForwardLayer(name, hc);
   }
   if( !strcmp(keyword, "MLPOutputLayer") ) {
      addedObject = (void *) new MLPOutputLayer(name, hc);
   }
   if( !strcmp(keyword, "MLPSigmoidLayer") ) {
      addedObject = (void *) new MLPSigmoidLayer(name, hc);
   }
   if( !strcmp(keyword, "MatchingPursuitLayer") ) {
      addedObject = (void *) new MatchingPursuitLayer(name, hc);
   }
   if( !strcmp(keyword, "MatchingPursuitResidual") ) {
      addedObject = (void *) new MatchingPursuitResidual(name, hc);
   }
   if( !strcmp(keyword, "MaxPooling") ) {
      // MaxPooling was marked obsolete Oct 30, 2014
      if (hc->columnId()==0) {
         fprintf(stderr, "Params group \"%s\": MaxPooling is obsolete.  Use a different layer type and set the connections going to \"%s\" to use pvpatchAccumulateType = \"maxpooling\".\n", name, name);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if( !strcmp(keyword, "Movie") ) {
      addedObject = (void *) new Movie(name, hc);
   }
   if( !strcmp(keyword, "Patterns") ) {
      addedObject = (void *) new Patterns(name, hc);
   }
   if( !strcmp(keyword, "PoolingANNLayer") ) {
      addedObject = (void *) new PoolingANNLayer(name, hc);
   }
   if( !strcmp(keyword, "PtwiseProductLayer") ) {
      addedObject = (void *) new PtwiseProductLayer(name, hc);
   }
   if( !strcmp(keyword, "RescaleLayer") ) {
      addedObject = (void *) new RescaleLayer(name, hc);
   }
   if( !strcmp(keyword, "Retina") ) {
      addedObject = (void *) new Retina(name, hc);
   }
   if( !strcmp(keyword, "ShuffleLayer") ) {
      addedObject = (void *) new ShuffleLayer(name, hc);
   }
   if( !strcmp(keyword, "SigmoidLayer") ) {
      addedObject = (void *) new SigmoidLayer(name, hc);
   }
   if( !strcmp(keyword, "TextStream") ) {
      addedObject = (void *) new TextStream(name, hc);
   }
   if( !strcmp(keyword, "TrainingLayer") ) {
      addedObject = (void *) new TrainingLayer(name, hc);
   }
   if( !strcmp(keyword, "WTALayer") ) {
      addedObject = (void *) new WTALayer(name, hc);
   }
#ifdef PV_USE_SNDFILE
   if( !strcmp(keyword, "NewCochlearLayer") ) {
      addedObject = (void *) new NewCochlearLayer(name, hc);
   }
   if( !strcmp(keyword, "SoundStream") ) {
      addedObject = (void *) new SoundStream(name, hc);
   }
#endif

   // Connection keywords
   if( !strcmp(keyword, "HyPerConn") ) {
      addedObject = (void *) new HyPerConn(name, hc);
   }
   if( !strcmp(keyword, "BIDSConn") ) {
      addedObject = (void *) new BIDSConn(name, hc);
   }
   if( !strcmp(keyword, "CloneConn") ) {
      addedObject = (void *) new CloneConn(name, hc);
   }
   if( !strcmp(keyword, "CloneKernelConn") ) {
      // Deprecated as of June 6, 2014.  Use CloneConn with sharedWeight = true
      addedObject = (void *) new CloneKernelConn(name, hc);
   }
   if( !strcmp(keyword, "CopyConn") ) {
      addedObject = (void *) new CopyConn(name, hc);
   }
   if( !strcmp(keyword, "FeedbackConn") ) {
      addedObject = (void *) new FeedbackConn(name, hc);
   }
   if( !strcmp(keyword, "GapConn") ) {
      addedObject = (void *) new GapConn(name, hc);
   }
   if( !strcmp(keyword, "IdentConn") ) {
      addedObject = (void *) new IdentConn(name, hc);
   }
   if( !strcmp(keyword, "KernelConn") ) {
      // Deprecated as of June 5, 2014.  Use HyPerConn with sharedWeight = true
      addedObject = (void *) new KernelConn(name, hc);
   }
   if( !strcmp(keyword, "LCALIFLateralConn") ) {
      addedObject = (void *) new LCALIFLateralConn(name, hc);
   }
   if( !strcmp(keyword, "OjaSTDPConn") ) {
      addedObject = (void *) new OjaSTDPConn(name, hc);
   }
   if( !strcmp(keyword, "PlasticCloneConn") ) {
      addedObject = (void *) new PlasticCloneConn(name, hc);
   }
   if( !strcmp(keyword, "PoolingConn") ) {
      addedObject = (void *) new PoolingConn(name, hc);
   }
   if( !strcmp(keyword, "TransposeConn") ) {
      addedObject = (void *) new TransposeConn(name, hc);
   }

   // Column Probe keywords
   if( !strcmp(keyword, "ColProbe") ) {
      addedObject = (void *) new ColProbe(name, hc);
   }
   if( !strcmp(keyword, "GenColProbe") ) {
      addedObject = (void *) new GenColProbe(name, hc);
   }

   // Layer probe keywords
   if( !strcmp(keyword, "LayerProbe") ) {
      fprintf(stderr, "LayerProbe \"%s\": Abstract class LayerProbe cannot be instantiated.\n", name);
      addedObject = NULL;
   }
   if( !strcmp(keyword, "L2NormProbe") ) {
      addedObject = (void *) new L2NormProbe(name, hc);
   }
   if( !strcmp(keyword, "LayerFunctionProbe") ) {
      addedObject = (void *) new LayerFunctionProbe(name, hc);
   }
   if( !strcmp(keyword, "LogLatWTAProbe") ) {
      addedObject = (void *) new LogLatWTAProbe(name, hc);
   }
   if( !strcmp(keyword, "PointLIFProbe") ) {
      addedObject = (void *) new PointLIFProbe(name, hc);
   }
   if( !strcmp(keyword, "PointProbe") ) {
      addedObject = (void *) new PointProbe(name, hc);
   }
   if( !strcmp(keyword, "RequireAllZeroActivityProbe") ) {
      addedObject = (void *) new RequireAllZeroActivityProbe(name, hc);
   }
   if( !strcmp(keyword, "SparsityLayerProbe") ) {
      addedObject = (void *) new SparsityLayerProbe(name, hc);
   }
   if( !strcmp(keyword, "StatsProbe") ) {
      addedObject = (void *) new StatsProbe(name, hc);
   }
   if( !strcmp(keyword, "TextStreamProbe") ) {
      addedObject = (void *) new TextStreamProbe(name, hc);
   }

   // Connection probe keywords
   if( !strcmp(keyword, "KernelProbe") ) {
      addedObject = (void *) new KernelProbe(name, hc);
   }

   if (addedObject==NULL) {
      fprintf(stderr, "Unable to add ");
      exit(EXIT_FAILURE);
   }

   return addedObject;
}

} /* namespace PV */
