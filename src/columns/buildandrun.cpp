/*
 * buildandrun.cpp
 *
 * buildandrun(argc, argv)  builds the layers, connections, and
 * (to a limited extent) probes from the params file and then calls the hypercol's run method.
 *
 * build(argc, argv) builds the hypercol but does not run it.  That way additional objects
 * can be created and added by hand if they are not yet supported by build().
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#include "buildandrun.hpp"

using namespace PV;

int buildandrun(int argc, char * argv[], int (*custominit)(HyPerCol *, int, char **), int (*customexit)(HyPerCol *, int, char **), void * (*customgroups)(const char *, const char *, HyPerCol *)) {

   //Parse param file
   char * param_file = NULL;
   pv_getopt_str(argc, argv, "-p", &param_file);
   InterColComm * icComm = new InterColComm(&argc, &argv);
   PVParams * params = new PVParams(param_file, 2*(INITIAL_LAYER_ARRAY_SIZE+INITIAL_CONNECTION_ARRAY_SIZE), icComm);
   free(param_file);

   int numSweepValues = params->getSweepSize();

   int status = PV_SUCCESS;
   if (numSweepValues) {
      for (int k=0; k<numSweepValues; k++) {
         if (icComm->commRank()==0) {
            printf("Parameter sweep: starting run %d of %d\n", k+1, numSweepValues);
         }
         params->setSweepValues(k);
         status = buildandrun1paramset(argc, argv, custominit, customexit, customgroups, params) == PV_SUCCESS ? status : PV_FAILURE;
      }
   }
   else {
      status = buildandrun1paramset(argc, argv, custominit, customexit, customgroups, params);
   }

   delete params;
   delete icComm;
   return status;
}

int buildandrun1paramset(int argc, char * argv[],
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         void * (*customgroups)(const char *, const char *, HyPerCol *),
                         PVParams * params) {
   HyPerCol * hc = build(argc, argv, customgroups, params);
   if( hc == NULL ) return PV_FAILURE;  // build() prints error message

   int status = PV_SUCCESS;
   if( custominit != NULL ) {
      status = (*custominit)(hc, argc, argv);
      if(status != PV_SUCCESS) {
         fprintf(stderr, "custominit function failed with return value %d\n", status);
      }
   }

   if( status==PV_SUCCESS && hc->getInitialStep() < hc->getFinalStep() ) {
      status = hc->run();
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "HyPerCol::run() returned with error code %d\n", status);
      }
   }
   if( status==PV_SUCCESS && customexit != NULL ) {
      status = (*customexit)(hc, argc, argv);
      if( status != PV_SUCCESS) {
         fprintf(stderr, "customexit function failed with return value %d\n", status);
      }
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
   return status;
}

HyPerCol * build(int argc, char * argv[], void * (*customgroups)(const char *, const char *, HyPerCol *), PVParams * params) {
   HyPerCol * hc = new HyPerCol("column", argc, argv, params);
   if( hc == NULL ) {
      fprintf(stderr, "Unable to create HyPerCol\n");
      return NULL;
   }
   HyPerCol * addedHyPerCol;
   HyPerConn * addedHyPerConn;
   HyPerLayer * addedHyPerLayer;
   ColProbe * addedColProbe;
   LayerProbe * addedLayerProbe;
   BaseConnectionProbe * addedBaseConnectionProbe;

   const char * allowedkeywordarray[] = { // indentation indicates derived class hierarchy
           "_Start_HyPerCols_",
             "HyPerCol",
           "_Stop_HyPerCols_",
           "_Start_HyPerLayers_",
             "HyPerLayer",
             "ANNLayer",
               "AccumulateLayer",
               "ANNSquaredLayer",
               "ANNWhitenedLayer",
               "ANNDivInhLayer",
               "CliqueLayer",
               "GenerativeLayer",
                 "LogLatWTAGenLayer",
                 "PursuitLayer",
               "IncrementLayer",
               "LeakyIntegrator",
               "MatchingPursuitResidual",
               "PoolingANNLayer",
               "PtwiseProductLayer",
               "TrainingLayer",
               "MaxPooling",
               "HyPerLCALayer",
               "ANNErrorLayer",
	         "ANNNormalizedErrorLayer",
               "MLPErrorLayer",
               "MLPForwardLayer",
               "MLPOutputLayer",
               "LabelErrorLayer",
               "ANNLabelLayer",
// Leaving ANNTriggerUpdateOnNewImageLayer in for now, to provide a meaningful error message if someone tries to use it
               "ANNTriggerUpdateOnNewImageLayer",
               "ConstantLayer",
             "CloneVLayer",
               "BIDSCloneLayer",
               "GapLayer",
               "RescaleLayer",
               "SigmoidLayer",
               "MLPSigmoidLayer",
             "BinningLayer",
             "LabelLayer",
             "TextStream",
#ifdef PV_USE_SNDFILE
             "SoundStream",
             "NewCochlearLayer",
#endif
             "Image",
               "CreateMovies",
               "Movie",
               "Patterns",
             "LIF",
                "LIFGap",
                "BIDSLayer",
                "LCALIFLayer",
             "MatchingPursuitLayer",
             "Retina",
             "ShuffleLayer",
             "BIDSMovieCloneMap",
             "BIDSSensorLayer",
           "_Stop_HyPerLayers_",
           "_Start_HyPerConns_",
             "HyPerConn",
               "BIDSConn",
               "CloneConn",
               "PlasticCloneConn",
               "KernelConn", // Deprecated
                 "MapReduceKernelConn",
                 "CliqueConn",
                 "CloneKernelConn", //Deprecated
                 "IdentConn",
                 "ImprintConn",
                 "GapConn",
                 "GenerativeConn",
                   "PoolingGenConn",
#ifdef OBSOLETE  // Marked obsolete Sept 16, 2013.  Learning rule for LCA is the same in KernelConn, so no need to subclass
                 "LCAConn",
#endif // OBSOLETE
                 "LCALIFLateralKernelConn",
                 "NoSelfKernelConn",
                   "SiblingConn",
                 "OjaKernelConn",
                 "ReciprocalConn",
                 "TransposeConn",
                   "FeedbackConn",
               "LCALIFLateralConn",
               "OjaSTDPConn",
               "InhibSTDPConn",
               "STDPConn",
               "STDP3Conn",
           "_Stop_HyPerConns_",
           "_Start_ColProbes_",
             "ColProbe",
               "GenColProbe",
           "_Stop_ColProbes_",
           "_Start_LayerProbes_",
             "LayerProbe",
               "PointProbe",
               "TextStreamProbe",
	       "LCAProbe",
                  "PointLIFProbe",
                    "PointLCALIFProbe",
               "StatsProbe",
               "SparsityLayerProbe",
               "LayerFunctionProbe",
                 "L2NormProbe",
                 "SparsityTermProbe",
                 "LogLatWTAProbe",
               "RequireAllZeroActivityProbe",
           "_Stop_LayerProbes_",
           "_Start_BaseConnectionProbes_",
             "KernelProbe",
             "OjaConnProbe",
             "OjaKernelSpikeRateProbe",
             "LCALIFLateralProbe",
             "PatchProbe",
             "ReciprocalEnergyProbe",
           "_Stop_BaseConnectionProbes_",
           "_Start_ConnectionProbes_",
           "_Stop_ConnectionProbes_",
           "_End_allowedkeywordarray" // Don't delete this; it provides a for-loop test that doesn't require you to keep track of the total number of keywords.
   };
   int first_hypercol_index = -1;
   int last_hypercol_index = -1;
   int first_hyperlayer_index = -1;
   int last_hyperlayer_index = -1;
   int first_hyperconn_index = -1;
   int last_hyperconn_index = -1;
   int first_colprobe_index = -1;
   int last_colprobe_index = -1;
   int first_baseconnectionprobe_index = -1;
   int last_baseconnectionprobe_index = -1;
   int first_connectionprobe_index = -1;
   int last_connectionprobe_index = -1;
   int first_layerprobe_index = -1;
   int last_layerprobe_index = -1;

   int j;
   for( j=0; strcmp(allowedkeywordarray[j],"_End_allowedkeywordarray"); j++ ) {
       const char * kw = allowedkeywordarray[j];
       if( !strcmp(kw,"_Start_HyPerCols_") ) { first_hypercol_index = j; continue;}
       if( !strcmp(kw,"_Stop_HyPerCols_") )  { last_hypercol_index = j; continue;}
       if( !strcmp(kw,"_Start_HyPerLayers_") ) { first_hyperlayer_index = j; continue;}
       if( !strcmp(kw,"_Stop_HyPerLayers_") ) { last_hyperlayer_index = j; continue;}
       if( !strcmp(kw,"_Start_HyPerConns_") ) { first_hyperconn_index = j; continue;}
       if( !strcmp(kw,"_Stop_HyPerConns_") ) { last_hyperconn_index = j; continue;}
       if( !strcmp(kw,"_Start_ColProbes_") ) { first_colprobe_index = j; continue;}
       if( !strcmp(kw,"_Stop_ColProbes_") ) { last_colprobe_index = j; continue;}
       if( !strcmp(kw,"_Start_BaseConnectionProbes_") ) { first_baseconnectionprobe_index = j; continue;}
       if( !strcmp(kw,"_Stop_BaseConnectionProbes_") ) { last_baseconnectionprobe_index = j; continue;}
       if( !strcmp(kw,"_Start_ConnectionProbes_") ) { first_connectionprobe_index = j; continue;}
       if( !strcmp(kw,"_Stop_ConnectionProbes_") ) { last_connectionprobe_index = j; continue;}
       if( !strcmp(kw,"_Start_LayerProbes_") ) { first_layerprobe_index = j; continue;}
       if( !strcmp(kw,"_Stop_LayerProbes_") ) { last_layerprobe_index = j; continue;}
   }

   int numclasskeywords = j;

   assert( first_hypercol_index >= 0 && first_hypercol_index < numclasskeywords );
   assert( last_hypercol_index >= 0 && last_hypercol_index < numclasskeywords );
   assert( first_hyperconn_index >= 0 && first_hyperconn_index < numclasskeywords );
   assert( last_hyperconn_index >= 0 && last_hyperconn_index < numclasskeywords );
   assert( first_hyperlayer_index >= 0 && first_hyperlayer_index < numclasskeywords );
   assert( last_hyperlayer_index >= 0 && last_hyperlayer_index < numclasskeywords );
   assert( first_colprobe_index >= 0 && first_colprobe_index < numclasskeywords );
   assert( last_colprobe_index >= 0 && last_colprobe_index < numclasskeywords );
   assert( first_connectionprobe_index >= 0 && first_connectionprobe_index < numclasskeywords );
   assert( last_connectionprobe_index >= 0 && last_connectionprobe_index < numclasskeywords );
   assert( first_layerprobe_index >= 0 && first_layerprobe_index < numclasskeywords );
   assert( last_layerprobe_index >= 0 && last_layerprobe_index < numclasskeywords );

   PVParams * hcparams = hc->parameters();
   int numGroups = hcparams->numberOfGroups();

   // Make sure first group defines a column
   if( strcmp(hcparams->groupKeywordFromIndex(0), "HyPerCol") ) {
      fprintf(stderr, "First group of params file did not define a HyPerCol.\n");
      delete hc;
      return NULL;
   }

   for( int k=0; k<numGroups; k++ ) {
      const char * kw = hcparams->groupKeywordFromIndex(k);
      const char * name = hcparams->groupNameFromIndex(k);
      bool didAddObject = false;

      int matchedkeyword = -1;
      for( j=0; j<numclasskeywords; j++ ) {
         if( !strcmp( kw, allowedkeywordarray[j]) ) {
            matchedkeyword = j;
            break;
         }
      }
      if( matchedkeyword < 0 && customgroups != NULL ) {
         void * addedCustomObject = customgroups(kw, name, hc);
         didAddObject = addedCustomObject != NULL;
      }
      else if( j > first_hypercol_index && j < last_hypercol_index ) {
         addedHyPerCol = addHyPerColToColumn(kw, name, hc);
         didAddObject = addedHyPerCol != NULL;
      }
      else if( j > first_hyperconn_index && j < last_hyperconn_index ) {
         addedHyPerConn = addConnToColumn(kw, name, hc);
         didAddObject = addedHyPerConn != NULL;
      }
      else if( j > first_hyperlayer_index && j < last_hyperlayer_index ) {
         addedHyPerLayer = addLayerToColumn(kw, name, hc);
         didAddObject = addedHyPerLayer != NULL;
      }
      else if( j > first_colprobe_index && j < last_colprobe_index ) {
         addedColProbe = addColProbeToColumn(kw, name, hc);
         didAddObject = addedColProbe != NULL;
      }
      else if( j > first_baseconnectionprobe_index && j < last_baseconnectionprobe_index ) {
         addedBaseConnectionProbe = addBaseConnectionProbeToColumn(kw, name, hc);
         didAddObject = addedBaseConnectionProbe != NULL;
      }
     else if( j > first_layerprobe_index && j < last_layerprobe_index ) {
         addedLayerProbe = addLayerProbeToColumn(kw, name, hc);
         didAddObject = addedLayerProbe != NULL;
      }
      else {
         fprintf(stderr,"%s \"%s\" in params: Keyword %s is unrecognized.\n", kw, name, kw);
         exit(EXIT_FAILURE);
      }

      if( !didAddObject) {
         fprintf(stderr, "Parameter group \"%s\": %s could not be created in rank %d process.\n", name, kw, hc->columnId());
         exit(EXIT_FAILURE);
      }
   }

   if( hc->numberOfLayers() == 0 ) {
      fprintf(stderr, "HyPerCol \"%s\" does not have any layers.\n", hc->getName());
      delete hc;
      return NULL;
   }
   // Allow a column with no connections, because layers can be connected through mechanisms other than HyPerConns.
   return hc;
}

HyPerCol * addHyPerColToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   return hc;
}

HyPerLayer * addLayerToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   bool keywordMatched;
   HyPerLayer * addedLayer;
   assert( hc != NULL );
   if( !strcmp(classkeyword, "HyPerLayer") ) {
      keywordMatched = true;
      fprintf(stderr, "Group \"%s\": abstract class HyPerLayer cannot be instantiated.\n", name);
      addedLayer = NULL;
   }
   int status = PV_SUCCESS;
   if( !strcmp(classkeyword, "ANNLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNLayer(name, hc);
   }
   if( !strcmp(classkeyword, "AccumulateLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new AccumulateLayer(name, hc);
   }
   if( !strcmp(classkeyword, "BIDSMovieCloneMap") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSMovieCloneMap(name, hc);
   }
   if( !strcmp(classkeyword, "BIDSSensorLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSSensorLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNSquaredLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNSquaredLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNWhitenedLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNWhitenedLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNDivInhLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNDivInh(name, hc);
   }
   if( !strcmp(classkeyword, "CliqueLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new CliqueLayer(name, hc);
   }
   if( !strcmp(classkeyword, "GenerativeLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new GenerativeLayer(name, hc);
   }
   if( !strcmp(classkeyword, "LogLatWTAGenLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LogLatWTAGenLayer(name, hc);
   }
   if (!strcmp(classkeyword, "PursuitLayer")) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PursuitLayer(name, hc);
   }
   if( !strcmp(classkeyword, "IncrementLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new IncrementLayer(name, hc);
   }
   if( !strcmp(classkeyword, "LeakyIntegrator") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LeakyIntegrator(name, hc);
   }
   if (!strcmp(classkeyword, "MatchingPursuitResidual")) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MatchingPursuitResidual(name, hc);
   }
   if( !strcmp(classkeyword, "PoolingANNLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PoolingANNLayer(name, hc);
   }
   if( !strcmp(classkeyword, "PtwiseProductLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PtwiseProductLayer(name, hc);
   }
   if( !strcmp(classkeyword, "TrainingLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new TrainingLayer(name, hc);
   }
   if( !strcmp(classkeyword, "MaxPooling") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MaxPooling(name, hc);
   }
   if( !strcmp(classkeyword, "CloneVLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new CloneVLayer(name, hc);
   }
   if( !strcmp(classkeyword, "BinningLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BinningLayer(name, hc);
   }
   if( !strcmp(classkeyword, "TextStream") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new TextStream(name, hc);
   }
#ifdef PV_USE_SNDFILE
   if( !strcmp(classkeyword, "SoundStream") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new SoundStream(name, hc);
   }
    if( !strcmp(classkeyword, "NewCochlearLayer") ) {
        keywordMatched = true;
        addedLayer = (HyPerLayer *) new NewCochlearLayer(name, hc);
    }
#endif
   if( !strcmp(classkeyword, "Image") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new Image(name, hc);
   }
   if( !strcmp(classkeyword, "CreateMovies") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new CreateMovies(name, hc);
   }
   if( !strcmp(classkeyword, "Movie") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new Movie(name, hc);
   }
   if ( !strcmp(classkeyword, "LabelLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LabelLayer(name,hc);
   }
   if( !strcmp(classkeyword, "Patterns") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new Patterns(name, hc);
   }
   if( !strcmp(classkeyword, "LIF") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LIF(name, hc);
   }
   if( !strcmp(classkeyword, "LIFGap") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LIFGap(name, hc);
   }
   if( !strcmp(classkeyword, "LCALIFLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LCALIFLayer(name, hc);
   }
   if( !strcmp(classkeyword, "GapLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new GapLayer(name, hc);
   }
   if( !strcmp(classkeyword, "HyPerLCALayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new HyPerLCALayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNErrorLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNErrorLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNNormalizedErrorLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNNormalizedErrorLayer(name, hc);
   }
   if( !strcmp(classkeyword, "MLPErrorLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MLPErrorLayer(name, hc);
   }
   if( !strcmp(classkeyword, "MLPForwardLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MLPForwardLayer(name, hc);
   }
   if( !strcmp(classkeyword, "MLPOutputLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MLPOutputLayer(name, hc);
   }
   if( !strcmp(classkeyword, "LabelErrorLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LabelErrorLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ANNLabelLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNLabelLayer(name, hc);
   }
// Use ANNLayer with triggerFlag set to true and triggerLayerName for the triggering layer
   if( !strcmp(classkeyword, "ANNTriggerUpdateOnNewImageLayer") ) {
      keywordMatched = false; // true;
      // addedLayer = (HyPerLayer *) new ANNTriggerUpdateOnNewImageLayer(name, hc);
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
   if( !strcmp(classkeyword, "ConstantLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ConstantLayer(name, hc);
   }
   if( !strcmp(classkeyword, "SigmoidLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new SigmoidLayer(name, hc);
   }
   if( !strcmp(classkeyword, "MLPSigmoidLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MLPSigmoidLayer(name, hc);
   }
   if( !strcmp(classkeyword, "RescaleLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new RescaleLayer(name, hc);
   }
   if (!strcmp(classkeyword, "MatchingPursuitLayer")) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new MatchingPursuitLayer(name, hc);
   }
   if( !strcmp(classkeyword, "Retina") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new Retina(name, hc);
   }
   if( !strcmp(classkeyword, "BIDSLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSLayer(name, hc);
   }
   if( !strcmp(classkeyword, "ShuffleLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ShuffleLayer(name, hc);
   }
   if( !strcmp(classkeyword, "BIDSCloneLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSCloneLayer(name, hc);
   }
   status = checknewobject((void *) addedLayer, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, name);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   return addedLayer;
}

/*
 * This method is getting changed radically - again.
 * The constructors for HyPerConn take only the name and the HyPerCol as
 * arguments; everything else, including weightInitializer, is read from
 * hc->parameters().
 */
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   HyPerConn * addedConn = NULL;
   assert( hc != NULL );

   bool keywordMatched = false;
   int status = PV_SUCCESS;
   if( !strcmp(classkeyword, "HyPerConn") ) {
      keywordMatched = true;
      addedConn = new HyPerConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "BIDSConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn*) new BIDSConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "CloneConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn*) new CloneConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "PlasticCloneConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn*) new PlasticCloneConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "KernelConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn * ) new KernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "ImprintConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn *) new ImprintConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "CliqueConn") ) {
      keywordMatched = true;
      addedConn = new CliqueConn(name, hc);
   }
   if( !keywordMatched && !strcmp( classkeyword, "CloneKernelConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn *) new CloneKernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "GapConn") ) {
      keywordMatched = true;
      addedConn = new GapConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "GenerativeConn") ) {
      keywordMatched = true;
      addedConn = new GenerativeConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "PoolingGenConn") ) {
      keywordMatched = true;
      addedConn = new PoolingGenConn(name, hc);
   }
   if( !keywordMatched && !strcmp( classkeyword, "IdentConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn * ) new IdentConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "LCALIFLateralKernelConn") ) {
      keywordMatched = true;
      addedConn = new LCALIFLateralKernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "MapReduceKernelConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn * ) new MapReduceKernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "NoSelfKernelConn") ) {
      keywordMatched = true;
      addedConn = new NoSelfKernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "SiblingConn") ) {
      keywordMatched = true;
      addedConn = new SiblingConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "OjaKernelConn") ) {
      keywordMatched = true;
      addedConn = new OjaKernelConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "ReciprocalConn") ) {
      keywordMatched = true;
      addedConn = new ReciprocalConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "TransposeConn") ) {
      keywordMatched = true;
      addedConn = new TransposeConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "FeedbackConn") ) {
      keywordMatched = true;
      addedConn = (HyPerConn *) new FeedbackConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "LCALIFLateralConn")) {
     keywordMatched = true;
     addedConn = (HyPerConn * ) new LCALIFLateralConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "OjaSTDPConn")) {
        keywordMatched = true;
        addedConn = (HyPerConn * ) new OjaSTDPConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "InhibSTDPConn")) {
      keywordMatched = true;
      addedConn = (HyPerConn * ) new InhibSTDPConn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "STDP3Conn")) {
      keywordMatched = true;
      addedConn = (HyPerConn * ) new STDP3Conn(name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "STDPConn")) {
     keywordMatched = true;
     addedConn = (HyPerConn * ) new STDPConn(name, hc);
   }
   status = checknewobject((void *) addedConn, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.

   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, name);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   // delete weightInitializer; // The connection takes ownership of the InitWeights object
   // weightInitializer = NULL;
   return addedConn;
}

const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterStringName, bool warnIfAbsent) {
   bool shouldGetValue = warnIfAbsent ? true : params->stringPresent(groupName, parameterStringName);
   const char * str;
   str = shouldGetValue ? params->stringValue(groupName, parameterStringName) : NULL;
   return str;
}

// make a method in HyPerCol?
HyPerLayer * getLayerFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbsent) {
   PVParams * params = hc->parameters();
   const char * layerName = getStringValueFromParameterGroup(groupName, params, parameterStringName, warnIfAbsent);
   if( !layerName ) return NULL;
   HyPerLayer * l = hc->getLayerFromName(layerName);
   if( l == NULL && warnIfAbsent )  {
      fprintf(stderr, "Group \"%s\": could not find layer \"%s\"\n", groupName, layerName);
   }
   return l;
}

HyPerConn * getConnFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbsent) {
   PVParams * params = hc->parameters();
   const char * connName = getStringValueFromParameterGroup(groupName, params, parameterStringName, warnIfAbsent);
   if( !connName ) return NULL; // error message was printed by getStringValueFromParameterGroup
   HyPerConn * c = hc->getConnFromName(connName);
   if( c == NULL && warnIfAbsent)  {
      fprintf(stderr, "Group \"%s\": could not find connection \"%s\"\n", groupName, connName);
   }
   return c;
}

ColProbe * getColProbeFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName) {
   ColProbe * colprobe = NULL;
   const char * colprobename = hc->parameters()->stringValue(groupName, parameterStringName);
   if (colprobename != NULL && colprobename[0] != '\0') {
      colprobe = hc->getColProbeFromName(colprobename); // getColProbeFromParameterGroup(name, hc, "parentGenColProbe");
   }
   return colprobe;
}

/*
ColProbe * getColProbeFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName) {
   ColProbe * p = NULL;
   PVParams * params = hc->parameters();
   const char * colProbeName = getStringValueFromParameterGroup(groupName, params, parameterStringName, false);
   if( !colProbeName ) return NULL;  // error message was printed by getStringValueFromParameterGroup
   int n = hc->numberOfProbes();
   for( int i=0; i<n; i++ ) {
      ColProbe * curColProbe = hc->getColProbe(i);
      const char * curName = curColProbe->getColProbeName();
      assert(curName);
      if( !strcmp(curName,colProbeName) ) {
         p = curColProbe;
      }
   }
   return p;
}
*/

ColProbe * addColProbeToColumn(const char * classkeyword, const char * probeName, HyPerCol * hc) {
   ColProbe * addedProbe = NULL;
   bool keywordMatched = false;
   const char * fileName = getStringValueFromParameterGroup(probeName, hc->parameters(), "probeOutputFile", false);
   if( !strcmp(classkeyword, "ColProbe") ) {
      keywordMatched = true;
      addedProbe = new ColProbe(probeName, hc);
      insertColProbe(addedProbe, hc);
   }
   if( !strcmp(classkeyword, "GenColProbe") ) {
      keywordMatched = true;
      addedProbe = (ColProbe *) new GenColProbe(probeName, hc);
      insertColProbe(addedProbe, hc);
   }
   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, probeName);
      addedProbe = NULL;
   }
   return addedProbe;
}

void insertColProbe(ColProbe * colProbe, HyPerCol * hc) {
   const char * classkeyword = hc->parameters()->groupKeywordFromName(colProbe->getColProbeName());
   if( colProbe != NULL ) {
      hc->insertProbe(colProbe);
      printf("Added %s \"%s\" to column.\n", classkeyword, colProbe->getColProbeName());
   }
   else {
      fprintf(stderr, "Unable to add %s \"%s\" to column.\n", classkeyword, colProbe->getColProbeName());
   }
}

BaseConnectionProbe * addBaseConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   BaseConnectionProbe * addedProbe = NULL;
   // PVParams * params = hc->parameters();
   bool keywordMatched = false;
   int status = PV_SUCCESS;
   if( !strcmp(classkeyword, "ReciprocalEnergyProbe") ) {
      keywordMatched = true;
      addedProbe = new ReciprocalEnergyProbe(name, hc);
   }
   if( !strcmp(classkeyword, "KernelProbe") ) {
      keywordMatched = true;
      addedProbe = new KernelProbe(name, hc);
   }
   if( !strcmp(classkeyword, "LCALIFLateralProbe") ) {
      keywordMatched = true;
      addedProbe = new LCALIFLateralProbe(name, hc);
   }
   if( !strcmp(classkeyword, "OjaConnProbe") ) {
      keywordMatched = true;
      addedProbe = new LCALIFLateralProbe(name, hc);
   }
   if( !strcmp(classkeyword, "OjaKernelSpikeRateProbe") ) {
      keywordMatched = true;
      addedProbe = new OjaKernelSpikeRateProbe(name, hc);
   }
   if( !strcmp(classkeyword, "PatchProbe") ) {
      keywordMatched = true;
      addedProbe = new PatchProbe(name, hc);
   }
   status = checknewobject((void *) addedProbe, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   assert(keywordMatched);
   assert( !(status == PV_SUCCESS && !addedProbe) );
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   return addedProbe;
}

LayerProbe * addLayerProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   int status;
   bool errorFound = false;

   LayerProbe * addedProbe;
   // char * probename;
   char * message = NULL;
   const char * filename;
   PVParams * params = hc->parameters();
   GenColProbe * parentcolprobe;
   if( !strcmp(classkeyword, "LayerProbe") ) {
      fprintf(stderr, "LayerProbe \"%s\": Abstract class LayerProbe cannot be instantiated.\n", name);
      addedProbe = NULL;
   }
   int xLoc, yLoc, fLoc;
   if( !strcmp(classkeyword, "PointProbe") ) {
      addedProbe = new PointProbe(name, hc);
   }
   if( !strcmp(classkeyword, "TextStreamProbe") ) {
      addedProbe =  new TextStreamProbe(name, hc);
   }
   if( !strcmp(classkeyword, "PointLIFProbe") ) {
      addedProbe = (LayerProbe *) new PointLIFProbe(name, hc);
   }
   if( !strcmp(classkeyword, "PointLCALIFProbe") ) {
      addedProbe = (LayerProbe *) new PointLIFProbe(name, hc);
   }
   if( !strcmp(classkeyword, "StatsProbe") ) {
      addedProbe = (LayerProbe *) new StatsProbe(name, hc);
   }
   if( !strcmp(classkeyword, "SparsityLayerProbe") ) {
      addedProbe = (LayerProbe *) new SparsityLayerProbe(name, hc);
   }
   if( !strcmp(classkeyword, "L2NormProbe") ) {
      addedProbe = (LayerProbe *) new L2NormProbe(name, hc);
   }
   if( !strcmp(classkeyword, "SparsityTermProbe") ) {
      addedProbe = (LayerProbe *) new SparsityTermProbe(name, hc);
   }
   if( !strcmp(classkeyword, "LogLatWTAProbe") ) {
      addedProbe = new LogLatWTAProbe(name, hc);
   }
   if( !strcmp(classkeyword, "RequireAllZeroActivityProbe") ) {
      addedProbe = new RequireAllZeroActivityProbe(name, hc);
   }
   status = checknewobject((void *) addedProbe, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   assert( !(status == PV_SUCCESS && !addedProbe) );
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   return addedProbe;
}

#define LAYERPROBEMSGLENGTH 31
int getLayerFunctionProbeParameters(const char * name, const char * keyword, HyPerCol * hc, HyPerLayer ** targetLayerPtr, char ** messagePtr, const char ** filenamePtr) {
   // If messagePtr is null, no memory is allocated for the message.
   // If messagePtr is non-null, memory is allocated and the calling routine is responsible for freeing it.
   PVParams * params = hc->parameters();
   int rank = hc->icCommunicator()->commRank();
   const char * filename;
   *targetLayerPtr = getLayerFromParameterGroup(name, hc, "targetLayer");
   if( *targetLayerPtr==NULL && rank==0 ) {
      fprintf(stderr, "Group \"%s\": Class %s must define targetLayer\n", name, keyword);
      return PV_FAILURE;
   }
   if( messagePtr ) {
      char * message = NULL;
      const char * msgFromParams = getStringValueFromParameterGroup(name, params, "message", false);
      if( msgFromParams ) {
         message = strdup(msgFromParams);
      }
      else {
         size_t messagelen = strlen(name);
         assert(LAYERPROBEMSGLENGTH>0);
         messagelen = messagelen < LAYERPROBEMSGLENGTH ? messagelen : LAYERPROBEMSGLENGTH;
         message = (char *) malloc(LAYERPROBEMSGLENGTH+1);
         if( ! message ) {
            fprintf(stderr, "Group \"%s\": Rank %d process unable to allocate memory for message\n", name, rank);
            return PV_FAILURE;
         }
         memcpy(message, name, messagelen);
         for( size_t c=messagelen; c<LAYERPROBEMSGLENGTH; c++ ) {
            message[c] = ' ';
         }
         message[LAYERPROBEMSGLENGTH] = '\0';
         if( rank == 0 ) {
            printf("Group \"%s\" will use \"%s\" for the message\n", name, message);
         }
      }
      *messagePtr = message;
   }
   filename = getStringValueFromParameterGroup(name, params, "probeOutputFile", false);
   *filenamePtr = filename;
   filename = NULL;
   return PV_SUCCESS;
}

int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (hc==NULL) {
      fprintf(stderr, "checknewobject error: HyPerCol argument must be set.\n");
      exit(EXIT_FAILURE);
   }
   int rank = hc->icCommunicator()->commRank();
   if( object == NULL ) {
      fprintf(stderr, "Rank %d process: Group \"%s\" unable to add object of class %s\n", rank, name, kw);
      status = PV_FAILURE;
   }
   else {
      if( rank==0 ) printf("Added %s \"%s\"\n", kw, name);
   }
   return status;
}
