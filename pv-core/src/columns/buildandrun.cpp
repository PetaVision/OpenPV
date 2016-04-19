/*
 * buildandrun.cpp
 *
 * buildandrun()  builds the layers, connections, and
 * (to a limited extent) probes from the params file and then calls the hypercol's run method.
 * It deletes the PV_Init and HyPerCol objects that it creates.
 * Often, the main() function consists only of a call to buildandrun.
 *
 * outputParams(argc, argv, path, factory) builds the
 * layers, connections, etc. and then calls the hypercol's processParams
 * method, which fills in default parameters, ignores unnecessary parameters
 * and sends the parameters to the file specified in the path argument.
 * Relative paths are relative to the params file outputParams deletes the
 * PV_Init and HyPerCol objects that it creates; it is written to be a
 * stand-alone function to create a cleaned-up params file.
 *
 * build() builds the hypercol but does not run it.  That way additional objects
 * can be created and added by hand if they are not yet supported by build().
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#include "buildandrun.hpp"
#include <columns/BaseObject.hpp>

using namespace PV;

// The buildandrun, rebuildandrun, and buildandrun1paramset functions below,
// which use the Factory class, are the preferred versions of these
// functions.  Older versions follow, and are kept for backwards
// compatibility, but are deprecated as of March 24, 2016.
int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **)) {
   PV_Init initObj(&argc, &argv, false/*value of allowUnrecognizedArguments*/);
   int status = buildandrun(&initObj, custominit, customexit);
   return status;
}

int buildandrun(PV_Init * initObj,
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **)) {
   initObj->initialize();
   if(initObj->isExtraProc()){
      return 0;
   }
   PVParams * params = initObj->getParams();
   if (params==NULL) {
      if (initObj->getWorldRank()==0) {
         char const * progName = initObj->getArguments()->getProgramName();
         if (progName==NULL) { progName = "PetaVision"; }
         fprintf(stderr, "%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(initObj->getComm()->communicator());
      exit(EXIT_FAILURE);
   }

   int numParamSweepValues = initObj->getParams()->getParameterSweepSize();

   int status = PV_SUCCESS;
   if (numParamSweepValues) {
      for (int k=0; k<numParamSweepValues; k++) {
         if (initObj->getWorldRank()==0) {
            printf("Parameter sweep: starting run %d of %d\n", k+1, numParamSweepValues);
         }
         status = buildandrun1paramset(initObj, custominit, customexit, k) == PV_SUCCESS ? status : PV_FAILURE;
      }
   }
   else{
      if(initObj->getComm()->numCommBatches() > 1){
         initObj->getParams()->setBatchSweepValues();
      }
      status = buildandrun1paramset(initObj, custominit, customexit) == PV_SUCCESS ? status : PV_FAILURE;
   }

   return status;
}

// A synonym for the form of buildandrun() that takes a PV_Init object.
// It is older than that form, and has been kept for backwards compatibility.
int rebuildandrun(PV_Init * initObj,
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **)) {
   return buildandrun(initObj, custominit, customexit);
}

int buildandrun1paramset(PV_Init * initObj,
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         int sweepindex) {
   if (sweepindex>=0) { initObj->getParams()->setParameterSweepValues(sweepindex); }
   HyPerCol * hc = initObj->build();
   if( hc == NULL ) return PV_FAILURE;  // build() prints error message

   int status = PV_SUCCESS;
   int argc = 0;
   char ** argv = NULL;
   if (custominit || customexit) {
      argc = initObj->getArguments()->getNumArgs();
      argv = initObj->getArguments()->getArgsCopy();
   }
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
   if (custominit || customexit) {
      initObj->getArguments()->freeArgs(argc, argv);
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
   return status;
}

HyPerCol * build(PV_Init* initObj) {
   return initObj ? initObj->build() : NULL;
}

// This version of buildandrun was deprecated March 24, 2016 in favor of the Factory version.
int buildandrun(int argc, char * argv[],
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **),
      void * (*customgroups)(const char *, const char *, HyPerCol *)) {
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj->initialize();
   initObj->printBuildAndRunDeprecationWarning("buildandrun", "argc, argv, custominit, customexit, customgroups");
   int status = rebuildandrun(initObj, custominit, customexit, customgroups);
   delete initObj;
   return status;
}

// This version of rebuildandrun was deprecated March 24, 2016 in favor of the Factory version.
int rebuildandrun(PV_Init* initObj,
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **),
      void * (*customgroups)(const char *, const char *, HyPerCol *)) {

   initObj->initialize();
   initObj->printBuildAndRunDeprecationWarning("rebuildandrun", "initObj, custominit, customexit, customgroups");
   if(initObj->isExtraProc()){
      return 0;
   }
   PVParams * params = initObj->getParams();
   if (params==NULL) {
      if (initObj->getWorldRank()==0) {
         char const * progName = initObj->getArguments()->getProgramName();
         if (progName==NULL) { progName = "PetaVision"; }
         fprintf(stderr, "%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(initObj->getComm()->communicator());
      exit(EXIT_FAILURE);
   }

   int numParamSweepValues = initObj->getParams()->getParameterSweepSize();

   int status = PV_SUCCESS;
   if (numParamSweepValues) {
      for (int k=0; k<numParamSweepValues; k++) {
         if (initObj->getWorldRank()==0) {
            printf("Parameter sweep: starting run %d of %d\n", k+1, numParamSweepValues);
         }
         initObj->getParams()->setParameterSweepValues(k);
         status = buildandrun1paramset(initObj, custominit, customexit, customgroups) == PV_SUCCESS ? status : PV_FAILURE;
      }
   }
   else{
      if(initObj->getComm()->numCommBatches() > 1){
         initObj->getParams()->setBatchSweepValues();
      }
      status = buildandrun1paramset(initObj, custominit, customexit, customgroups) == PV_SUCCESS ? status : PV_FAILURE;
   }

   return status;
}

// This version of buildandrun was deprecated March 24, 2016 in favor of the Factory version.
int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {

   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj->initialize();
   initObj->printBuildAndRunDeprecationWarning("buildandrun", "argc, argv, custominit, customexit, groupHandlerList, numGroupHandlers");
   int status = rebuildandrun(initObj, custominit, customexit, groupHandlerList, numGroupHandlers);
   delete initObj;
   return status;
}

// This version of rebuildandrun was deprecated March 24, 2016 in favor of the Factory version.
int rebuildandrun(PV_Init* initObj,
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {

   initObj->initialize();
   initObj->printBuildAndRunDeprecationWarning("rebuildandrun", "initObj, custominit, customexit, groupHandlerList, numGroupHandlers");
   if(initObj->isExtraProc()){
      return 0;
   }
   PVParams * params = initObj->getParams();
   if (params==NULL) {
      if (initObj->getWorldRank()==0) {
         char const * progName = initObj->getArguments()->getProgramName();
         if (progName==NULL) { progName = "PetaVision"; }
         fprintf(stderr, "%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(initObj->getComm()->communicator());
      exit(EXIT_FAILURE);
   }

   int numParamSweepValues = initObj->getParams()->getParameterSweepSize();

   int status = PV_SUCCESS;
   if (numParamSweepValues) {
      for (int k=0; k<numParamSweepValues; k++) {
         if (initObj->getWorldRank()==0) {
            printf("Parameter sweep: starting run %d of %d\n", k+1, numParamSweepValues);
         }
         initObj->getParams()->setParameterSweepValues(k);
         status = buildandrun1paramset(initObj, custominit, customexit, groupHandlerList, numGroupHandlers) == PV_SUCCESS ? status : PV_FAILURE;
      }
   }
   else{
      if(initObj->getComm()->numCommBatches() > 1){
         initObj->getParams()->setBatchSweepValues();
      }
      status = buildandrun1paramset(initObj, custominit, customexit, groupHandlerList, numGroupHandlers) == PV_SUCCESS ? status : PV_FAILURE;
   }

   //delete params;
   //delete icComm;
   return status;
}

// This version of buildandrun1paramset was deprecated March 24, 2016 in favor of the Factory version.
int buildandrun1paramset(PV_Init* initObj,
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         void * (*customgroups)(const char *, const char *, HyPerCol *)) {
   initObj->printBuildAndRunDeprecationWarning("rebuildandrun1paramset", "initObj, custominit, customexit, customgroups");
   HyPerCol * hc = build(initObj, customgroups);
   if( hc == NULL ) return PV_FAILURE;  // build() prints error message

   int status = PV_SUCCESS;
   int argc = 0;
   char ** argv = NULL;
   if (custominit || customexit) {
      argc = initObj->getArguments()->getNumArgs();
      argv = initObj->getArguments()->getArgsCopy();
   }
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
   if( status==PV_SUCCESS && customexit != NULL && !initObj->getArguments()->getDryRunFlag()) {
      status = (*customexit)(hc, argc, argv);
      if( status != PV_SUCCESS) {
         fprintf(stderr, "customexit function failed with return value %d\n", status);
      }
   }
   if (custominit || customexit) {
      initObj->getArguments()->freeArgs(argc, argv);
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
   return status;
}

// This version of build was deprecated March 24, 2016 in favor of the Factory version.
HyPerCol * build(PV_Init* initObj,
                 void * (*customgroups)(const char *, const char *, HyPerCol *)) {
   initObj->printBuildAndRunDeprecationWarning("build", "initObj, customgroups");
   HyPerCol * hc = new HyPerCol("column", initObj);
   if( hc == NULL ) {
      fprintf(stderr, "Unable to create HyPerCol\n");
      return NULL;
   }
   PVParams * hcparams = hc->parameters();
   int numGroups = hcparams->numberOfGroups();

   // Make sure first group defines a column
   if( strcmp(hcparams->groupKeywordFromIndex(0), "HyPerCol") ) {
      fprintf(stderr, "First group of params file did not define a HyPerCol.\n");
      delete hc;
      return NULL;
   }

   CoreParamGroupHandler * handler = new CoreParamGroupHandler();
   for( int k=0; k<numGroups; k++ ) {
      const char * kw = hcparams->groupKeywordFromIndex(k);
      const char * name = hcparams->groupNameFromIndex(k);
      ParamGroupType groupType = handler->getGroupType(kw);
      void * addedObject = NULL;
      InitWeights * weightInitializer = NULL;
      NormalizeBase * weightNormalizer = NULL;
      switch(groupType) {
      case UnrecognizedGroupType:
         if (customgroups != NULL) {
            addedObject = customgroups(kw, name, hc); // Handler for custom groups
         }
         break;
      case HyPerColGroupType:
         addedObject = handler->createHyPerCol(kw, name, hc);
         break;
      case LayerGroupType:
         addedObject = handler->createLayer(kw, name, hc);
         break;
      case ConnectionGroupType:
         addedObject = createConnection(handler, NULL, 0, kw, name, hc);
         break;
      case ProbeGroupType:
         addedObject = handler->createProbe(kw, name, hc);
         break;
      case ColProbeGroupType: // TODO: make ColProbe a subclass of BaseProbe and eliminate this group type.
         addedObject = handler->createColProbe(kw, name, hc);
         break;
      default:
         assert(0); // All possibilities for groupType are handled above
      }
      if(addedObject == NULL) {
         if (hc->globalRank()==0) {
            fprintf(stderr, "Parameter group \"%s\": %s could not be created.\n", name, kw);
         }
         MPI_Barrier(hc->icCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
   delete handler;

   if( hc->numberOfLayers() == 0 ) {
      fprintf(stderr, "HyPerCol \"%s\" does not have any layers.\n", hc->getName());
      delete hc;
      return NULL;
   }
   return hc;
}


int buildandrun1paramset(PV_Init * initObj,
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {
   HyPerCol * hc = build(initObj, groupHandlerList, numGroupHandlers);
   if( hc == NULL ) return PV_FAILURE;  // build() prints error message

   int status = PV_SUCCESS;
   int argc = 0;
   char ** argv = NULL;
   if (custominit || customexit) {
      argc = initObj->getArguments()->getNumArgs();
      argv = initObj->getArguments()->getArgsCopy();
   }
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
   if( status==PV_SUCCESS && customexit != NULL && !initObj->getArguments()->getDryRunFlag() ) {
      status = (*customexit)(hc, argc, argv);
      if( status != PV_SUCCESS) {
         fprintf(stderr, "customexit function failed with return value %d\n", status);
      }
   }
   if (custominit || customexit) {
      initObj->getArguments()->freeArgs(argc, argv);
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
   return status;
}

// Deprecated April 14, 2016.
int outputParams(int argc, char * argv[], char const * path, ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {
   printf("\nWarning: outputParams is deprecated.  Instead use the -n option on the command line or the dryRunFlag in PV_Arguments.\n\n");
   PV::PV_Init * initObj = new PV::PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj->initialize();
   if (initObj->isExtraProc()) { return EXIT_SUCCESS; }
   PV::HyPerCol * hc = build(initObj, groupHandlerList, numGroupHandlers);

   int status = hc->processParams(path);

   delete hc;
   delete initObj;
   return status;
}

HyPerCol * build(PV_Init* initObj, ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {
   HyPerCol * hc = new HyPerCol("column", initObj);
   if( hc == NULL ) {
      fprintf(stderr, "Unable to create HyPerCol\n");
      return NULL;
   }
   PVParams * hcparams = hc->parameters();
   int numGroups = hcparams->numberOfGroups();

   // Make sure first group defines a column
   if( strcmp(hcparams->groupKeywordFromIndex(0), "HyPerCol") ) {
      fprintf(stderr, "First group of params file did not define a HyPerCol.\n");
      delete hc;
      return NULL;
   }

   CoreParamGroupHandler * coreHandler = new CoreParamGroupHandler(); // The ParamGroupHandler for keywords known by trunk
   for( int k=0; k<numGroups; k++ ) {
      const char * kw = hcparams->groupKeywordFromIndex(k);
      const char * name = hcparams->groupNameFromIndex(k);
      void * addedObject = NULL;
      ParamGroupType groupType = UnrecognizedGroupType;
      ParamGroupHandler * handler = getGroupHandlerFromList(kw, coreHandler, groupHandlerList, numGroupHandlers, &groupType);
      if (handler == NULL) {
         if (hc->globalRank()==0) {
            fprintf(stderr, "Error: parameter group \"%s\": %s is not recognized by any known ParamGroupHandler.\n", name, kw);
         }
         MPI_Barrier(hc->icCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
      switch (groupType) {
      case UnrecognizedGroupType:
         assert(0); // UnrecognizedGroupType should only happen if handler==NULL, and vice versa.
         break;
      case HyPerColGroupType:
         addedObject = handler->createHyPerCol(kw, name, hc);
         break;
      case LayerGroupType:
         addedObject = handler->createLayer(kw, name, hc);
         break;
      case ConnectionGroupType:
         addedObject = createConnection(coreHandler, groupHandlerList, numGroupHandlers, kw, name, hc);
         break;
      case ProbeGroupType:
         addedObject = handler->createProbe(kw, name, hc);
         break;
      case ColProbeGroupType: // TODO: make ColProbe a subclass of BaseProbe and eliminate this group type.
         addedObject = handler->createColProbe(kw, name, hc);
         break;
      case WeightInitializerGroupType:
         addedObject = handler->createWeightInitializer(kw, name, hc);
         break;
      case WeightNormalizerGroupType:
         addedObject = handler->createWeightNormalizer(kw, name, hc);
         break;
      default:
          assert(0); // All possibilities for groupType are handled above
      }
      if (addedObject==NULL) {
         if (hc->globalRank()==0) {
            fprintf(stderr, "Error creating %s \"%s\".\n", kw, name);
         }
         MPI_Barrier(hc->icCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
   delete coreHandler;

   if( hc->numberOfLayers() == 0 ) {
      fprintf(stderr, "HyPerCol \"%s\" does not have any layers.\n", hc->getName());
      delete hc;
      return NULL;
   }
   return hc;
}

ParamGroupHandler * getGroupHandlerFromList(char const * keyword, CoreParamGroupHandler * coreHandler, ParamGroupHandler ** groupHandlerList, int numGroupHandlers, ParamGroupType * foundGroupType) {
   ParamGroupType groupType = coreHandler->getGroupType(keyword);
   if (groupType != UnrecognizedGroupType) {
      if (foundGroupType) { *foundGroupType = groupType; }
      return coreHandler;
   }
   else {
      for (int h=0; h<numGroupHandlers; h++) {
         groupType = groupHandlerList[h]->getGroupType(keyword);
         if (groupType != UnrecognizedGroupType) {
            if (foundGroupType) { *foundGroupType = groupType; }
            return groupHandlerList[h];
         }
      }
   }
   if (foundGroupType) { *foundGroupType = UnrecognizedGroupType; }
   return NULL;
}

BaseConnection * createConnection(CoreParamGroupHandler * coreGroupHandler, ParamGroupHandler ** customHandlerList, int numGroupHandlers, char const * keyword, char const * groupname, HyPerCol * hc) {
   // The basic logic is:
   //     get the weight initializer by reading the weightInitType parameter, and create an InitWeights object
   //     get the weight normalizer by reading the normalizeMethod parameter, and create a NormalizeBase object
   //     get the connection type from the keyword and create a BaseConnection object, passing the InitWeights and NormalizeBase objects to the connection's constructor.
   //
   // The complications are: the weightInitType, normalizeMethod, and connection type could in principle be handled by three different ParamGroupHandlers, so we have to call getGroupHandlerFromList for each one;
   // and it is allowed for some subclasses not to have a weightInitType and/or a normalizeMethod parameter, so we have to allow for them to be null.
   ParamGroupType groupType;
   InitWeights * weightInitializer = NULL;
   char const * weightInitStr = hc->parameters()->stringValue(groupname, "weightInitType", false/*warnIfAbsent*/);
   if (weightInitStr!=NULL) {
      ParamGroupHandler * weightInitHandler = getGroupHandlerFromList(weightInitStr, coreGroupHandler, customHandlerList, numGroupHandlers, &groupType);
      if (weightInitHandler==NULL || groupType != WeightInitializerGroupType) {
         fprintf(stderr, "Connection %s error: weightInitType \"%s\" is not recognized.\n", keyword, weightInitStr);
         exit(EXIT_FAILURE);
      }
      weightInitializer = weightInitHandler->createWeightInitializer(weightInitStr, groupname, hc);
   }
   NormalizeBase * weightNormalizer = NULL;
   char const * weightNormalizeStr = hc->parameters()->stringValue(groupname, "normalizeMethod", false/*warnIfAbsent*/);
   if (weightNormalizeStr!=NULL) {
      ParamGroupHandler * normalizeHandler = getGroupHandlerFromList(weightNormalizeStr, coreGroupHandler, customHandlerList, numGroupHandlers, &groupType);
      if (normalizeHandler==NULL || groupType != WeightNormalizerGroupType) {
         fprintf(stderr, "Connection %s error: normalizeMethod \"%s\" is not recognized.\n", keyword, weightNormalizeStr);
         exit(EXIT_FAILURE);
      }
      weightNormalizer = normalizeHandler->createWeightNormalizer(weightNormalizeStr, groupname, hc);
   }
   ParamGroupHandler * connectionHandler = getGroupHandlerFromList(keyword, coreGroupHandler, customHandlerList, numGroupHandlers, &groupType);
   if (connectionHandler==NULL || groupType != ConnectionGroupType) {
      fprintf(stderr, "Connection %s error: connection type \"%s\" is not recognized.\n", keyword, weightNormalizeStr);
      exit(EXIT_FAILURE);
   }
   BaseConnection * baseConn = connectionHandler->createConnection(keyword, groupname, hc, weightInitializer, weightNormalizer);
   return baseConn;
}

int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (hc==NULL) {
      fprintf(stderr, "checknewobject error: HyPerCol argument must be set.\n");
      exit(EXIT_FAILURE);
   }
   int rank = hc->globalRank();
   if( object == NULL ) {
      fprintf(stderr, "Global Rank %d process: Group \"%s\" unable to add object of class %s\n", rank, name, kw);
      status = PV_FAILURE;
   }
   else {
      if( rank==0 ) printf("Added %s \"%s\"\n", kw, name);
   }
   return status;
}
