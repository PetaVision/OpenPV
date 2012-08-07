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

int buildandrun(int argc, char * argv[], int (*customadd)(HyPerCol *, int, char **), int (*customexit)(HyPerCol *, int, char **), void * (*customgroups)(const char *, const char *, HyPerCol *)) {
   HyPerCol * hc = build(argc, argv, customgroups);
   if( hc == NULL ) return PV_FAILURE;  // build() prints error message

   int status = PV_SUCCESS;
   if( customadd != NULL ) {
      status = (*customadd)(hc, argc, argv);
      if(status != PV_SUCCESS) {
         fprintf(stderr, "customadd function failed with return value %d\n", status);
         exit(status);
      }
   }

   if( hc->numberOfTimeSteps() > 0 ) {
      status = hc->run();
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "HyPerCol::run() returned with error code %d\n", status);
      }
   }
   if( customexit != NULL ) {
      status = (*customexit)(hc, argc, argv);
      if( status != PV_SUCCESS) {
         fprintf(stderr, "customexit function failed with return value %d\n", status);
         exit(status);
      }
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
   return status;
}

HyPerCol * build(int argc, char * argv[], void * (*customgroups)(const char *, const char *, HyPerCol *)) {
   HyPerCol * hc = new HyPerCol("column", argc, argv);
   if( hc == NULL ) {
      fprintf(stderr, "Unable to create HyPerCol\n");
      return NULL;
   }
   PVParams * params = hc->parameters();
   HyPerCol * addedHyPerCol;
   HyPerConn * addedHyPerConn;
   HyPerLayer * addedHyPerLayer;
   ColProbe * addedColProbe;
   LayerProbe * addedLayerProbe;
   BaseConnectionProbe * addedBaseConnectionProbe;
   // ConnectionProbe * addedConnectionProbe;

   const char * allowedkeywordarray[] = { // indentation indicates derived class hierarchy
           "_Start_HyPerCols_",
             "HyPerCol",
           "_Stop_HyPerCols_",
           "_Start_HyPerLayers_",
             "HyPerLayer",
             "ANNLayer",
               "BIDSSourceLayer",
               "ANNSquaredLayer",
               "ANNDivInhLayer",
               "CliqueLayer",
               "GenerativeLayer",
                 "LogLatWTAGenLayer",
                 "PursuitLayer",
               "IncrementLayer",
               "PoolingANNLayer",
               "PtwiseProductLayer",
               "TrainingLayer",
             "GapLayer",
             "HMaxSimple",
             "Image",
               "CreateMovies",
               "ImageCreator",
               "Movie",
               "Patterns",
             "LIF",
                "LIFGap",
                "BIDSLayer",
             "Retina",
             "SigmoidLayer",
             "BIDSCloneLayer",
           "_Stop_HyPerLayers_",
           "_Start_HyPerConns_",
             "HyPerConn",
               "KernelConn",
                 "CloneKernelConn",
                 "NoSelfKernelConn",
                 "IdentConn",
                 "GenerativeConn",
                   "PoolingGenConn",
                 "ReciprocalConn",
                 "CliqueConn",
                 "CliqueApplyConn",
                 "SiblingConn",
                 "GapConn",
                 "TransposeConn",
                   "FeedbackConn",
               "PoolConn",
               "RuleConn",
               "STDPConn",
               "SubunitConn",
           "_Stop_HyPerConns_",
           "_Start_ColProbes_",
             "ColProbe",
               "GenColProbe",
           "_Stop_ColProbes_",
           "_Start_LayerProbes_",
             "LayerProbe",
               "PointProbe",
                  "PointLIFProbe",
               "StatsProbe",
               "LayerFunctionProbe",
                 "L2NormProbe",
                 "SparsityTermProbe",
                 "LogLatWTAProbe",
           "_Stop_LayerProbes_",
           "_Start_BaseConnectionProbes_",
             "KernelProbe",
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
   assert( first_hypercol_index >= 0 );
   assert( last_hypercol_index >= 0 );
   assert( first_hyperconn_index >= 0 );
   assert( last_hyperconn_index >= 0 );
   assert( first_hyperlayer_index >= 0 );
   assert( last_hyperlayer_index >= 0 );
   assert( first_colprobe_index >= 0 );
   assert( last_colprobe_index >= 0 );
   assert( first_connectionprobe_index >= 0 );
   assert( last_connectionprobe_index >= 0 );
   assert( first_layerprobe_index >= 0 );
   assert( last_layerprobe_index > 0 );

   int numclasskeywords = j;

   int numGroups = params->numberOfGroups();

   // Make sure first group defines a column
   if( strcmp(params->groupKeywordFromIndex(0), "HyPerCol") ) {
      fprintf(stderr, "First group of params file did not define a HyPerCol.\n");
      delete hc;
      return NULL;
   }

   for( int k=0; k<numGroups; k++ ) {
      const char * kw = params->groupKeywordFromIndex(k);
      const char * name = params->groupNameFromIndex(k);
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
      if( j > first_hypercol_index && j < last_hypercol_index ) {
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
      }

      if( !didAddObject && hc->icCommunicator()->commRank()==0 ) {
         fprintf(stderr, "Parameter group \"%s\": %s could not be created.\n", name, kw);
      }
   }

   if( hc->numberOfLayers() == 0 ) {
      fprintf(stderr, "HyPerCol \"%s\" does not have any layers.\n", hc->getName());
      delete hc;
      return NULL;
   }
   if( hc->numberOfConnections() == 0 ) {
      fprintf(stderr, "HyPerCol \"%s\" does not have any connections.\n", hc->getName());
      delete hc;
      return NULL;
   }
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
      status = checknewobject((void *) addedLayer, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   }
   if( !strcmp(classkeyword, "BIDSSourceLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSSourceLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   }
   if( !strcmp(classkeyword, "ANNSquaredLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNSquaredLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   }
   if( !strcmp(classkeyword, "ANNDivInhLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ANNDivInh(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   }
   if( !strcmp(classkeyword, "CliqueLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new CliqueLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "GenerativeLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new GenerativeLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "LogLatWTAGenLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LogLatWTAGenLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if (!strcmp(classkeyword, "PursuitLayer")) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PursuitLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "IncrementLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new IncrementLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "PoolingANNLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PoolingANNLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "PtwiseProductLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new PtwiseProductLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "TrainingLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addTrainingLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "HMaxSimple") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new HMaxSimple(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "Image") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addImage(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "CreateMovies") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new CreateMovies(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "ImageCreator") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new ImageCreator(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "Movie") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addMovie(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "Patterns") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addPatterns(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "LIF") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LIF(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "LIFGap") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new LIFGap(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "GapLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addGapLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "SigmoidLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addSigmoidLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "Retina") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new Retina(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "BIDSLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) new BIDSLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "BIDSCloneLayer") ) {
      keywordMatched = true;
      addedLayer = (HyPerLayer *) addBIDSCloneLayer(name, hc);
      status = checknewobject((void *) addedLayer, classkeyword, name, hc);
   }
   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, name);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   return addedLayer;
}

TrainingLayer * addTrainingLayer(const char * name, HyPerCol * hc) {
   TrainingLayer * addedLayer;
   const char * traininglabelspath = hc->parameters()->stringValue(name, "trainingLabelsPath");
   if( traininglabelspath ) {
      addedLayer = new TrainingLayer(name, hc, traininglabelspath);
   }
   else {
      fprintf(stderr, "Group \"%s\": Parameter group for class TrainingLayer must set string parameter trainingLabelsPath\n", name);
      addedLayer = NULL;
   }
   return addedLayer;
}

GapLayer * addGapLayer(const char * name, HyPerCol * hc) {
   HyPerLayer * originalLayer = getLayerFromParameterGroup(name, hc, "originalLayerName");
   if( originalLayer == NULL ) {
      fprintf(stderr, "Group \"%s\": Parameter group for class GapLayer must set string parameter originalLayerName\n", name);
      return NULL;
   }
   LIFGap * originalLIFLayer = dynamic_cast<LIFGap *>(originalLayer);
   GapLayer * addedLayer;
   if (originalLIFLayer) {
      addedLayer = new GapLayer(name, hc, originalLIFLayer);
   }
   else {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must a LIFGap layer\n", name, originalLayer->getName());
      addedLayer = NULL;
   }
   return addedLayer;
}

Image * addImage( const char * name, HyPerCol * hc) {
   Image * addedLayer;
   const char * imagelabelspath = hc->parameters()->stringValue(name, "imagePath");
   if (imagelabelspath) {
      addedLayer = new Image(name, hc, imagelabelspath);
   }
   else {
      fprintf(stderr, "Group \"%s\": Parameter group for class Image must set string parameter imagePath\n", name);
      addedLayer = NULL;
   }
   return addedLayer;
}

Movie * addMovie(const char * name, HyPerCol * hc) {
   Movie * addedLayer;
   const char * imagelabelspath = hc->parameters()->stringValue(name, "imageListPath");
   if (imagelabelspath) {
      addedLayer = new Movie(name, hc, imagelabelspath);
   }
   else {
      fprintf(stderr, "Group \"%s\": Parameter group for class Movie must set string parameter imageListPath\n", name);
      addedLayer = NULL;
   }
   return addedLayer;
}

Patterns * addPatterns(const char * name, HyPerCol *hc) {
   const char * allowedPatternTypes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "BARS",
         "RECTANGLES",
         "SINEWAVE",
         "COSWAVE",
         "IMPULSE",
         "SINEV",
         "COSV",
         "_End_allowedPatternTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   const char * patternTypeStr = hc->parameters()->stringValue(name, "patternType");
   if( ! patternTypeStr ) {
      fprintf(stderr, "Group \"%s\": Parameter group for class Patterns must set string parameter patternType\n", name);
      return NULL;
   }
   PatternType patternType;
   int patternTypeMatch = false;
   for( int i=0; strcmp(allowedPatternTypes[i],"_End_allowedPatternTypes"); i++ ) {
      const char * thispatterntype = allowedPatternTypes[i];
      if( !strcmp(patternTypeStr, thispatterntype) ) {
         patternType = (PatternType) i;
         patternTypeMatch = true;
         break;
      }
   }
   if( patternTypeMatch ) {
      return new Patterns(name, hc, patternType);
   }
   else {
      fprintf(stderr, "Group \"%s\": Pattern type \"%s\" not recognized.\n", name, patternTypeStr);
      return NULL;
   }
}

SigmoidLayer * addSigmoidLayer(const char * name, HyPerCol * hc) {
   HyPerLayer * originalLayer = getLayerFromParameterGroup(name, hc, "originalLayerName");
   if( originalLayer == NULL ) {
      fprintf(stderr, "Group \"%s\": Parameter group for class SigmoidLayer must set string parameter originalLayerName\n", name);
      return NULL;
   }
   LIF * originalLIFLayer = dynamic_cast<LIF *>(originalLayer);
   SigmoidLayer * addedLayer;
   if (originalLIFLayer) {
      addedLayer = new SigmoidLayer(name, hc, originalLIFLayer);
   }
   else {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must be a LIF layer\n", name, originalLayer->getName());
      addedLayer = NULL;
   }
   return addedLayer;
}

BIDSCloneLayer * addBIDSCloneLayer(const char * name, HyPerCol * hc) {
   HyPerLayer * originalLayer = getLayerFromParameterGroup(name, hc, "originalLayerName");
   if( originalLayer == NULL ) {
      fprintf(stderr, "Group \"%s\": Parameter group for class BIDSCloneLayer must set string parameter originalLayerName\n", name);
      return NULL;
   }
   LIF * originalLIFLayer = dynamic_cast<LIF *>(originalLayer);
   BIDSCloneLayer * addedLayer;
   if (originalLIFLayer) {
      addedLayer = new BIDSCloneLayer(name, hc, originalLIFLayer);
   }
   else {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must be a LIF layer\n", name, originalLayer->getName());
      addedLayer = NULL;
   }
   return addedLayer;
}


/*
 * This method parses the weightInitType parameter and creates an
 * appropriate InitWeight object for the chosen weight initialization.
 *
 */
InitWeights *createInitWeightsObject(const char * name, HyPerCol * hc) {

   // Get weightInitType.  The HyPerConn subclass may have a natural weightInitType so don't issue a warning yet if weightInitType is missing.
   // The warning is issued in getDefaultInitWeightsMethod().
   PVParams * params = hc->parameters();
   const char * weightInitTypeStr = params->stringValue(name, "weightInitType",false);
   InitWeights *weightInitializer;

   if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "CoCircWeight"))) {
      weightInitializer = new InitCocircWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "UniformWeight"))) {
      weightInitializer = new InitUniformWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "SmartWeight"))) {
      weightInitializer = new InitSmartWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "DistributedWeight"))) {
      weightInitializer = new InitDistributedWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "BIDSWeight"))) {
      weightInitializer = new InitBIDSWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "UniformRandomWeight"))) {
      weightInitializer = new InitUniformRandomWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "GaussianRandomWeight"))) {
      weightInitializer = new InitGaussianRandomWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "GaborWeight"))) {
      weightInitializer = new InitGaborWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "PoolWeight"))) {
      weightInitializer = new InitPoolWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "RuleWeight"))) {
      weightInitializer = new InitRuleWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "SubUnitWeight"))) {
      weightInitializer = new InitSubUnitWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "IdentWeight"))) {
      weightInitializer = new InitIdentWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "OneToOneWeights"))) {
      weightInitializer = new InitOneToOneWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "Gauss2DWeight"))) {
      weightInitializer = new InitWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "SpreadOverArborsWeight"))) {
      weightInitializer = new InitSpreadOverArborsWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "Gauss3DWeight"))) {
      weightInitializer = new Init3DGaussWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "Windowed3DGaussWeights"))) {
      weightInitializer = new InitWindowed3DGaussWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "MTWeight"))) {
      weightInitializer = new InitMTWeights();
   }
   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "FileWeight"))) {
      if( params->stringPresent(name, "initWeightsFile") == 0 ) {
#ifdef PV_USE_MPI
         fprintf(stderr, "Error (process %d): connection \"%s\": weightInitType \"FileWeight\" requires parameter \"initWeightsFile\".  Exiting.\n", hc->icCommunicator()->commRank(), name);
#else
         fprintf(stderr, "Error: connection \"%s\": weightInitType \"FileWeight\" requires parameter \"initWeightsFile\".  Exiting.\n", name);
#endif // PV_USE_MPI
         exit(EXIT_FAILURE);
      }
      weightInitializer = new InitWeights();
   }
   else {
      weightInitializer = NULL;
   }

   return weightInitializer;
}

InitWeights * getDefaultInitWeightsMethod(const char * keyword) {
   InitWeights * weightInitializer;
   if( !strcmp(keyword, "IdentConn") ) {
      weightInitializer = NULL; // new InitIdentWeights(); will be called in IdentConn::initialize
   }
   else if( !strcmp(keyword, "CloneKernelConn") ) {
      weightInitializer = NULL; // new InitCloneKernelWeights(); will be called in CloneKernelConn::initialize
   }
   else if( !strcmp(keyword, "TransposeConn") ) {
      weightInitializer = NULL; // weights are initialized by transposing originalConn's initial weights
   }
   else if( !strcmp(keyword, "FeedbackConn") ) {
      weightInitializer = NULL; // inherits from TransposeConn
   }
   else {
      weightInitializer = new InitWeights();
      fprintf(stderr, "weightInitType not set or unrecognized.  Using default method.\n");
   }
   return weightInitializer;
}

/*
 * This method is getting changed radically to use the new InitWeights class.  This class and any that extend it
 * will implement any weight initialization methods that are necessary.  Depending on keywords set in the params file
 * a different subtype of InitWeights will be created and passed to the connection class.  A few classes will be made
 * obsolete because all of their code was weight initialization, all of which will be moved to the new InitWeights
 * class.
 *
 */
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   HyPerConn * addedConn = NULL;
   assert( hc != NULL );
   const char * fileName;
   HyPerLayer * preLayer, * postLayer;
   HyPerConn * auxConn;
   PVParams * params = hc->parameters();
   InitWeights *weightInitializer;

#ifdef OBSOLETE // Marked obsolete Aug 7, 2012.  The channel type is now read by HyPerConn::initialize calling HyPerConn::readChannelCode (so that specialized conns like GapConn can override)
   ChannelType channelType;
   if (strcmp(classkeyword, "GapConn")) {
      int channelNo = (int) params->value(name, "channelCode", -1);
      if( decodeChannel( channelNo, &channelType ) != PV_SUCCESS) {
         fprintf(stderr, "Group \"%s\": Parameter group for class %s must set parameter channelCode.\n", name, classkeyword);
         return NULL;
      }

   }
#endif // OBSOLETE

   weightInitializer = createInitWeightsObject(name, hc);
   if( weightInitializer == NULL ) {
      weightInitializer = getDefaultInitWeightsMethod(classkeyword);
   }

   bool keywordMatched = false;
   int status = PV_SUCCESS;
   if( !strcmp(classkeyword, "HyPerConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {

         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);

         addedConn = new HyPerConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "KernelConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);

         addedConn = (HyPerConn * ) new KernelConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp( classkeyword, "CloneKernelConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      auxConn = getConnFromParameterGroup(name, hc, "originalConnName");
      if( auxConn && preLayer && postLayer ) {
         addedConn = (HyPerConn *) new CloneKernelConn(name, hc, preLayer, postLayer, dynamic_cast<KernelConn *>(auxConn)  );
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "NoSelfKernelConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new NoSelfKernelConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "ReciprocalConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new ReciprocalConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "CliqueConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new CliqueConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "SiblingConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      HyPerConn * temp_conn = getConnFromParameterGroup(name, hc, "siblingConnName");
      SiblingConn * sibling_conn;
      if (temp_conn != NULL){
         sibling_conn = dynamic_cast<SiblingConn *>(temp_conn);
      }
      else{
         sibling_conn = NULL;
      }
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new SiblingConn(name, hc, preLayer, postLayer, fileName, weightInitializer, sibling_conn);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp( classkeyword, "IdentConn") ) {
      // Filename is ignored
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         addedConn = (HyPerConn * ) new IdentConn(name, hc, preLayer, postLayer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "GenerativeConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new GenerativeConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "PoolingGenConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = (HyPerConn *) addPoolingGenConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "TransposeConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      auxConn = getConnFromParameterGroup(name, hc, "originalConnName");
      if( auxConn && preLayer && postLayer ) {
         addedConn = (HyPerConn *) new TransposeConn(name, hc, preLayer, postLayer, dynamic_cast<KernelConn *>(auxConn) );
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "FeedbackConn") ) {
      keywordMatched = true;
      auxConn = getConnFromParameterGroup(name, hc, "originalConnName");
      if( auxConn ) {
         addedConn = (HyPerConn *) new FeedbackConn(name, hc, dynamic_cast<KernelConn *>(auxConn) );
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "STDPConn")) {
     keywordMatched = true;
     getPreAndPostLayers(name, hc, &preLayer, &postLayer);
     bool stdpFlag = params->value(name, "stdpFlag", (float) true, true);
     if( preLayer && postLayer ) {
       fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
       addedConn = (HyPerConn * ) new STDPConn(name, hc, preLayer, postLayer, fileName, stdpFlag, weightInitializer);
     }
     status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched && !strcmp(classkeyword, "GapConn") ) {
      keywordMatched = true;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedConn = new GapConn(name, hc, preLayer, postLayer, fileName, weightInitializer);
      }
      status = checknewobject((void *) addedConn, classkeyword, name, hc);
   }
   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, name);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   delete weightInitializer;
   weightInitializer = NULL;
   return addedConn;
}

PoolingGenConn * addPoolingGenConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, const char * filename, InitWeights *weightInit) {
   PoolingGenConn * addedConn;
   HyPerLayer * secondaryPreLayer = getLayerFromParameterGroup(name, hc, "secondaryPreLayerName");
   HyPerLayer * secondaryPostLayer = getLayerFromParameterGroup(name, hc, "secondaryPostLayerName");
   if( secondaryPreLayer && secondaryPostLayer ) {
       addedConn = new PoolingGenConn(name, hc, pre, post, secondaryPreLayer, secondaryPostLayer, filename, weightInit);
   }
   else {
       addedConn = NULL;
   }
   return addedConn;
}

const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterStringName, bool warnIfAbsent) {
   bool shouldGetValue = warnIfAbsent ? true : params->stringPresent(groupName, parameterStringName);
   const char * str;
   str = shouldGetValue ? params->stringValue(groupName, parameterStringName) : NULL;
   return str;
}

int getPreAndPostLayers(const char * name, HyPerCol * hc, HyPerLayer ** preLayerPtr, HyPerLayer **postLayerPtr) {
   const char * separator = " to ";
   *preLayerPtr = getLayerFromParameterGroup(name, hc, "preLayerName", false);
   *postLayerPtr = getLayerFromParameterGroup(name, hc, "postLayerName", false);
   if( *preLayerPtr == NULL && *postLayerPtr == NULL ) {
      // Check to see if the string " to " appears exactly once in name
      // If so, use part preceding " to " as pre-layer, and part after " to " as post.
      const char * locto = strstr(name, separator);
      if( locto != NULL ) {
         const char * nextto = strstr(locto+1, separator);
         if( nextto == NULL ) {
            char * layerNames = (char *) malloc(strlen(name) + 1);
            assert(layerNames);
            strcpy(layerNames, name);
            char * preLayerName = layerNames;
            size_t preLen = locto - name;
            preLayerName[preLen] = '\0';
            char * postLayerName = layerNames + preLen + strlen(separator);
            *preLayerPtr = hc->getLayerFromName(preLayerName);
            if( *preLayerPtr == NULL ) {
               fprintf(stderr, "Group \"%s\" preLayerName: No layer named \"%s\".\n", name, preLayerName);
            }
            *postLayerPtr = hc->getLayerFromName(postLayerName);
            if( *postLayerPtr == NULL ) {
               fprintf(stderr, "Group \"%s\" postLayerName: No layer named \"%s\".\n", name, postLayerName);
            }
            free(layerNames);
         }
      }
      else {
         if( *preLayerPtr == NULL ) {
            fprintf(stderr, "Parameter string \"preLayerName\" missing from group \"%s\"\n",name);
         }
         if( *postLayerPtr == NULL ) {
            fprintf(stderr, "Parameter string \"postLayerName\" missing from group \"%s\"\n",name);
         }
      }
   }
   return *preLayerPtr != NULL && *postLayerPtr != NULL ? PV_SUCCESS : PV_FAILURE;
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

ColProbe * addColProbeToColumn(const char * classkeyword, const char * probeName, HyPerCol * hc) {
   ColProbe * addedProbe = NULL;
   bool keywordMatched = false;
   const char * fileName = getStringValueFromParameterGroup(probeName, hc->parameters(), "probeOutputFile", false);
   if( !strcmp(classkeyword, "ColProbe") ) {
      keywordMatched = true;
      addedProbe = new ColProbe(probeName, fileName, hc);
      insertColProbe(addedProbe, hc, classkeyword);
   }
   if( !strcmp(classkeyword, "GenColProbe") ) {
      keywordMatched = true;
      addedProbe = (ColProbe *) new GenColProbe(probeName, fileName, hc);
      insertColProbe(addedProbe, hc, classkeyword);
   }
   if( !keywordMatched ) {
      fprintf(stderr, "Class keyword \"%s\" of group \"%s\" not recognized\n", classkeyword, probeName);
      addedProbe = NULL;
   }
   return addedProbe;
}

void insertColProbe(ColProbe * colProbe, HyPerCol * hc, const char * classkeyword) {
   if( colProbe != NULL ) {
      hc->insertProbe(colProbe);
      printf("Added %s \"%s\" to column.\n", classkeyword, colProbe->getColProbeName());
   }
   else {
      fprintf(stderr, "Unable to add %s \"%s\" to column.\n", classkeyword, colProbe->getColProbeName());
   }
}

BaseConnectionProbe * addBaseConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc) {
   BaseConnectionProbe * addedProbe;
   PVParams * params = hc->parameters();
   HyPerConn * targetConn = NULL;
   bool keywordMatched = false;
   int status = PV_SUCCESS;
   if( !strcmp(classkeyword, "KernelProbe") ) {
      keywordMatched = true;
      int kernelIndex = params->value(name, "kernelIndex", 0);
      int arborId = params->value(name, "arborId", 0);
      targetConn = hc->getConnFromName(params->stringValue(name, "targetConnection"));
      if( targetConn ) {
         const char * filename = params->stringValue(name, "probeOutputFile");
         addedProbe = new KernelProbe(name, filename, targetConn, kernelIndex, arborId);
         status = checknewobject((void *) addedProbe, classkeyword, name, hc);
      }
      else {
         fprintf(stderr, "Error: connection probe \"%s\" requires parameter \"targetConnection\".\n", name);
         status = PV_FAILURE;
      }
   }
   if( !strcmp(classkeyword, "PatchProbe") ) {
      keywordMatched = true;
      int arborID = params->value(name, "arborID");
      int indexmethod = params->present(name, "kPre");
      int coordmethod = params->present(name, "kxPre") && params->present(name,"kyPre") && params->present(name,"kfPre");
      if( indexmethod && coordmethod ) {
         fprintf(stderr, "PatchProbe \"%s\": Ambiguous definition with both kPre and (kxPre,kyPre,kfPre) defined\n", name);
         return NULL;
      }
      if( !indexmethod && !coordmethod ) {
         fprintf(stderr, "PatchProbe \"%s\": Exactly one of kPre and (kxPre,kyPre,kfPre) must be defined\n", name);
         return NULL;
      }
      const char * filename = params->stringValue(name, "probeOutputFile");
      targetConn = hc->getConnFromName(params->stringValue(name, "targetConnection"));
      if( targetConn == NULL ) {
         fprintf(stderr, "Error: connection probe \"%s\" requires parameter \"targetConnection\".\n", name);
         return NULL;
      }
      if( indexmethod ) {
         int kPre = params->value(name, "kPre");
         addedProbe = new PatchProbe(name, filename, targetConn, kPre, arborID);
      }
      else {
         assert(coordmethod);
         int kxPre = params->value(name, "kxPre");
         int kyPre = params->value(name, "kyPre");
         int kfPre = params->value(name, "kfPre");
         addedProbe = new PatchProbe(name, filename, targetConn, kxPre, kyPre, kfPre, arborID);
      }
      status = checknewobject((void *) addedProbe, classkeyword, name, hc);
   }
   if( !strcmp(classkeyword, "ReciprocalEnergyProbe") ) {
      keywordMatched = true;
      const char * filename = params->stringValue(name, "probeOutputFile");
      targetConn = hc->getConnFromName(params->stringValue(name, "targetConnection"));
      if( targetConn == NULL ) {
         fprintf(stderr, "Error: connection probe \"%s\" requires parameter \"targetConnection\".\n", name);
         status = PV_FAILURE;
      }
      if( status == PV_SUCCESS ) {
         addedProbe = new ReciprocalEnergyProbe(name, filename, targetConn);
         status = checknewobject((void *) addedProbe, classkeyword, name, hc);
      }
      if( status == PV_SUCCESS ) {
         ColProbe * colProbeFromParams = getColProbeFromParameterGroup(name, hc, "parentGenColProbe");
         if( colProbeFromParams != NULL ) {
            GenColProbe * parentcolprobe = dynamic_cast<GenColProbe *>(colProbeFromParams);
            if( parentcolprobe == NULL) {
               fprintf(stderr, "ReciprocalEnergyProbe \"%s\": parentGenColProbe \"%s\" is not a GenColProbe\n", name, parentcolprobe->getColProbeName());
               status = PV_FAILURE;
            }
            else {
               parentcolprobe->addConnTerm((ConnFunctionProbe *) addedProbe, targetConn, 1.0f);
               // ReciprocalEnergyProbe uses reciprocalFidelityCoeff for the energy
               // so I think a separate probe parameter for the coefficient won't be necessary.
            }
         }
      }
   }
   assert(keywordMatched);
   assert( !(status == PV_SUCCESS && targetConn == NULL) );
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
   HyPerLayer * targetlayer;
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
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         xLoc = params->value(name, "xLoc", -1);
         yLoc = params->value(name, "yLoc", -1);
         fLoc = params->value(name, "fLoc", -1);
         if( xLoc <= -1 || yLoc <= -1 || fLoc <= -1) {
            fprintf(stderr, "Group \"%s\": Class %s requires xLoc, yLoc, and fLoc be set\n", name, classkeyword);
            errorFound = true;
         }
      }
      if( !errorFound ) {
         if( filename ) {
            addedProbe = (LayerProbe *) new PointProbe(filename, targetlayer, xLoc, yLoc, fLoc, message);
         }
         else {
            addedProbe = (LayerProbe *) new PointProbe(targetlayer, xLoc, yLoc, fLoc, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   if( !strcmp(classkeyword, "PointLIFProbe") ) {
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         xLoc = params->value(name, "xLoc", -1);
         yLoc = params->value(name, "yLoc", -1);
         fLoc = params->value(name, "fLoc", -1);
         if( xLoc <= -1 || yLoc <= -1 || fLoc <= -1) {
            fprintf(stderr, "Group \"%s\": Class %s requires xLoc, yLoc, and fLoc be set\n", name, classkeyword);
            errorFound = true;
         }
      }
      if( !errorFound ) {
         if( filename ) {
            addedProbe = (LayerProbe *) new PointLIFProbe(filename, targetlayer, xLoc, yLoc, fLoc, message);
         }
         else {
            addedProbe = (LayerProbe *) new PointLIFProbe(targetlayer, xLoc, yLoc, fLoc, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   if( !strcmp(classkeyword, "StatsProbe") ) {
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         PVBufType buf_type = BufV;
         if (targetlayer->getSpikingFlag()) {
            buf_type = BufActivity;
         }
         if( filename ) {
            addedProbe = (LayerProbe *) new StatsProbe(filename, targetlayer, message);
         }
         else {
            addedProbe = (LayerProbe *) new StatsProbe(targetlayer, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   if( !strcmp(classkeyword, "L2NormProbe") ) {
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         if( filename ) {
            addedProbe = (LayerProbe *) new L2NormProbe(filename, targetlayer, message);
         }
         else {
            addedProbe = (LayerProbe *) new L2NormProbe(targetlayer, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe \n", name);
             errorFound = true;
         }
      }
      if( !errorFound ) {
         parentcolprobe = (GenColProbe *) getColProbeFromParameterGroup(name, hc, "parentGenColProbe");
         if( parentcolprobe )
         {
            pvdata_t coeff = params->value(name, "coeff", 1);
            parentcolprobe->addLayerTerm((LayerFunctionProbe *) addedProbe, targetlayer, coeff);
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   if( !strcmp(classkeyword, "SparsityTermProbe") ) {
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         if( filename ) {
            addedProbe = (LayerProbe *) new SparsityTermProbe(filename, targetlayer, message);
         }
         else {
            addedProbe = (LayerProbe *) new SparsityTermProbe(targetlayer, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      if( !errorFound ) {
         parentcolprobe = (GenColProbe *) getColProbeFromParameterGroup(name, hc, "parentGenColProbe");
         if( parentcolprobe )
         {
            pvdata_t coeff = params->value(name, "coeff", 1);
            parentcolprobe->addLayerTerm((LayerFunctionProbe *) addedProbe, targetlayer, coeff);
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   if( !strcmp(classkeyword, "LogLatWTAProbe") ) {
      status = getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         if( filename ) {
            addedProbe = (LayerProbe *) new LogLatWTAProbe(filename, targetlayer, message);
         }
         else {
            addedProbe = (LayerProbe *) new LogLatWTAProbe(targetlayer, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      if( !errorFound ) {
         parentcolprobe = (GenColProbe *) getColProbeFromParameterGroup(name, hc, "parentGenColProbe");
         if( parentcolprobe )
         {
            pvdata_t coeff = params->value(name, "coeff", 1);
            parentcolprobe->addLayerTerm((LayerFunctionProbe *) addedProbe, targetlayer, coeff);
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   assert(targetlayer);
   checknewobject((void *) addedProbe, classkeyword, name, hc);
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
         message = (char *) malloc(LAYERPROBEMSGLENGTH);
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

#ifdef OBSOLETE // Marked obsolete Aug 7, 2012.  This function is now the static method HyPerConn::readChannelCode
int decodeChannel(int channel, ChannelType * channelType) {
   int status = PV_SUCCESS;
   switch( channel ) {
   case CHANNEL_EXC:
      *channelType = CHANNEL_EXC;
      break;
   case CHANNEL_INH:
      *channelType = CHANNEL_INH;
      break;
   case CHANNEL_INHB:
      *channelType = CHANNEL_INHB;
      break;
   case CHANNEL_GAP:
      *channelType = CHANNEL_GAP;
      break;
   default:
      status = PV_FAILURE;
      break;
   }
   return status;
}
#endif // OBSOLETE

int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc) {
   int rank;
   if( hc != NULL ) {
      rank = hc->icCommunicator()->commRank();
   }
   else {
      fprintf(stderr, "Warning: checknewobject() now takes a 4th argument of type <HyPerCol *>.\n");
      rank = 0;
   }
   if( object == NULL ) {
      fprintf(stderr, "Rank %d process: Group \"%s\" unable to add object of class %s\n", rank, name, kw);
      return PV_FAILURE;
   }
   else {
      if( rank==0 ) printf("Added %s \"%s\"\n", kw, name);
      return PV_SUCCESS;
   }
}
