/*
 * buildandrun.cpp
 *
 * buildandrun()  builds the layers, connections, and
 * (to a limited extent) probes from the params file and then calls the
 * hypercol's run method.
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

using namespace PV;

// The buildandrun, rebuildandrun, and buildandrun1paramset functions below
// automate creating the HyPerCol, filling it with layers, connections, etc.,
// and running it.  To add custom groups, instantiate a PV_Init object
// and call PV_Init::registerKeyword with the create function (in most cases,
// the static function template PV::Factory::create<CustomClass>.
int buildandrun(
      int argc,
      char *argv[],
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **)) {
   PV_Init initObj(&argc, &argv, false /*value of allowUnrecognizedArguments*/);
   int status = buildandrun(&initObj, custominit, customexit);
   return status;
}

int buildandrun(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **)) {
   if (initObj->isExtraProc()) {
      return 0;
   }
   PVParams *params = initObj->getParams();
   if (params == NULL) {
      if (initObj->getWorldRank() == 0) {
         char const *progName = initObj->getProgramName();
         if (progName == NULL) {
            progName = "PetaVision";
         }
         ErrorLog().printf("%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(initObj->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   int numParamSweepValues = initObj->getParams()->getParameterSweepSize();

   int status = PV_SUCCESS;
   if (numParamSweepValues) {
      for (int k = 0; k < numParamSweepValues; k++) {
         if (initObj->getWorldRank() == 0) {
            InfoLog().printf(
                  "Parameter sweep: starting run %d of %d\n", k + 1, numParamSweepValues);
         }
         status = buildandrun1paramset(initObj, custominit, customexit, k) == PV_SUCCESS
                        ? status
                        : PV_FAILURE;
      }
   }
   else {
      status = buildandrun1paramset(initObj, custominit, customexit) == PV_SUCCESS ? status
                                                                                   : PV_FAILURE;
   }

   return status;
}

// A synonym for the form of buildandrun() that takes a PV_Init object.
// It is older than that form, and has been kept for backwards compatibility.
int rebuildandrun(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **)) {
   return buildandrun(initObj, custominit, customexit);
}

int buildandrun1paramset(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **),
      int (*customexit)(HyPerCol *, int, char **),
      int sweepindex) {
   if (sweepindex >= 0) {
      initObj->getParams()->setParameterSweepValues(sweepindex);
   }
   HyPerCol *hc = new HyPerCol(initObj);

   int status  = PV_SUCCESS;
   int argc    = 0;
   char **argv = NULL;
   if (custominit || customexit) {
      argc = initObj->getNumArgs();
      argv = initObj->getArgsCopy();
   }
   if (custominit != NULL) {
      status = (*custominit)(hc, argc, argv);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("custominit function failed with return value %d\n", status);
      }
   }

   if (status == PV_SUCCESS && hc->getInitialStep() < hc->getFinalStep()) {
      status = hc->run();
      if (status != PV_SUCCESS) {
         ErrorLog().printf("HyPerCol::run() returned with error code %d\n", status);
      }
   }
   if (status == PV_SUCCESS && customexit != NULL) {
      status = (*customexit)(hc, argc, argv);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("customexit function failed with return value %d\n", status);
      }
   }
   if (custominit || customexit) {
      initObj->freeArgs(argc, argv);
   }
   delete hc; /* HyPerCol's destructor takes care of deleting layers and
                 connections */
   return status;
}

HyPerCol *build(PV_Init *initObj) { return initObj ? new HyPerCol(initObj) : nullptr; }
