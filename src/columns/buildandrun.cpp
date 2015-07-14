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

// TODO: Lots of duplicated code between the two flavors of buildandrun, buildandrun1paramset, and build
int buildandrun(int argc, char * argv[], int (*custominit)(HyPerCol *, int, char **), int (*customexit)(HyPerCol *, int, char **), void * (*customgroups)(const char *, const char *, HyPerCol *)) {

   //Parse param file
   char * param_file = NULL;
   pv_getopt_str(argc, argv, "-p", &param_file, NULL);
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
         if (hc->icCommunicator()->commRank()==0) {
            fprintf(stderr, "Parameter group \"%s\": %s could not be created.\n", name, kw);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
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

int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers) {
   //Parse param file
   char * param_file = NULL;
   pv_getopt_str(argc, argv, "-p", &param_file, NULL);
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
         status = buildandrun1paramset(argc, argv, custominit, customexit, groupHandlerList, numGroupHandlers, params) == PV_SUCCESS ? status : PV_FAILURE;
      }
   }
   else {
      status = buildandrun1paramset(argc, argv, custominit, customexit, groupHandlerList, numGroupHandlers, params);
   }

   delete params;
   delete icComm;
   return status;
}

int buildandrun1paramset(int argc, char * argv[],
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         ParamGroupHandler ** groupHandlerList, int numGroupHandlers,
                         PVParams * params) {
   HyPerCol * hc = build(argc, argv, groupHandlerList, numGroupHandlers, params);
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

HyPerCol * build(int argc, char * argv[], ParamGroupHandler ** groupHandlerList, int numGroupHandlers, PVParams * params) {
   HyPerCol * hc = new HyPerCol("column", argc, argv, params);
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
         if (hc->icCommunicator()->commRank()==0) {
            fprintf(stderr, "Error: parameter group \"%s\": %s is not recognized by any known ParamGroupHandler.\n", name, kw);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
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
         if (hc->columnId()==0) {
            fprintf(stderr, "Error creating %s \"%s\".\n", kw, name);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
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
