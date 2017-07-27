/*
 * buildandrun.hpp
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#ifndef BUILDANDRUN_HPP_
#define BUILDANDRUN_HPP_

#include <iostream>
#include <string>
#include <time.h>

#include <include/pv_common.h>

#include <columns/HyPerCol.hpp>

#include <columns/PV_Init.hpp>
#include <normalizers/NormalizeBase.hpp>
#include <weightinit/InitWeights.hpp>

using namespace PV;

/**
 * The most basic function for creating and running a column from command line
 * settings.
 * If the command line arguments are complete and the params file contains only
 * groups in the core PetaVision functionality, it may be sufficient to execute
 * the following program:
 *
 * #include <columns/buildandrun.hpp>
 *
 * int main(int argc, char * argv[]) {
 *    int status = buildandrun(argc, argv);
 *    return status;
 * }
 *
 * At its heart, buildandrun() creates a HyPerCol and adds layers, connections,
 * and probes; and then calls HyPerCol::run() and then deletes the HyPerCol.
 * It returns the return value of the run() method (zero for success, nonzero
 * for failure).
 *
 * The function pointers custominit and customexit provide
 * additional flexibility.
 *
 * The custominit() hook is executed after the HyPerCol is built with all
 * layers, connections, and probes specified in the params file, but
 * before HyPerCol::run() is called.
 *
 * The customexit() hook is executed after HyPerCol::run() is called but before
 * the HyPerCol is deleted.
 *
 * More technically, the buildandrun function creates a PV_Init object, then
 * passes it to rebuildandrun(), and returns the result of the rebuildandrun()
 * call.
 * Note that the PetaVision objects and environment are no longer available
 * after buildandrun() returns.  If more flexibility is required, see the
 * flavor of buildandrun that takes a PV_Init object as an argument,
 * the buildandrun1paramset, or the PV_Init::build() method.
 */
int buildandrun(
      int argc,
      char *argv[],
      int (*custominit)(HyPerCol *, int, char **) = NULL,
      int (*customexit)(HyPerCol *, int, char **) = NULL);

/**
 * This form of buildandrun takes a PV_Init object instead of argc and argv.
 * It can be used when some command line options are meant to be hardwired
 * create the PV_Init object, then call PV_Init set-methods, then call
 * buildandrun), or when there are params group types not in the core
 * functionality
 * (create the PV_Init object, then call PV_Init::registerKeyword for
 * each custom group type, and then call buildandrun).
 *
 * If the params file has a ParameterSweep, it calls buildandrun1paramset
 * in a loop, once for each element of the ParameterSweep.
 *
 * Otherwise, it calls buildandrun1paramset once.
 *
 * It returns success (return value zero) if all the calls to
 * buildandrun1paramset
 * succeed, and failure if any of the calls to buildandrun1paramset fail.
 */
int buildandrun(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **) = NULL,
      int (*customexit)(HyPerCol *, int, char **) = NULL);

/**
 * A synonym for the PV_Init flavor of buildandrun.  It was originally written
 * as a way to allow a second column to be built and run under the same
 * PV_Init environment, but when it is used, it is usually the first
 * function from the buildandrun suite to be called.
 */
int rebuildandrun(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **) = NULL,
      int (*customexit)(HyPerCol *, int, char **) = NULL);

/**
 * A buildandrun function for running a particular element of a ParameterSweep.
 * The sweepindex specifies which element to run, with sweepindex=0 identifying
 * the first element of the sweep.  If sweepindex is negative (the default),
 * a sweep element is not selected, so the params file either does not have
 * a ParameterSweep, or the sweep element was selected prior to the call
 * (using the initObj->getParams()->setParameterSweepValues() method).
 */
int buildandrun1paramset(
      PV_Init *initObj,
      int (*custominit)(HyPerCol *, int, char **) = NULL,
      int (*customexit)(HyPerCol *, int, char **) = NULL,
      int sweepindex = -1);

/**
 * A convenience function for PV_Init::build() method, included for backwards
 * compatibility.
 * It creates a HyPerCol object, layers, connections, and probes based on the
 * params set in the PV_Init object.
 */
HyPerCol *build(PV_Init *initObj);

/**
 * Parses the params file specified by the input arguments,
 * discards unused parameter settings, fills in missing parameter settings with
 * defaults,
 * and sends the params file with standardized formatting to the path specified
 * in the
 * arguments.  It creates and deletes the PV_Init object, so that it is best
 * used
 * as a stand-alone method for generating a standardized params file.
 */
int outputParams(int argc, char *argv[], char const *path);

/**
 * Parses the params file specified by the PV_Init object,
 * discards unused parameter settings, fills in missing parameter settings with
 * defaults,
 * and sends the params file with standardized formatting to the path specified
 * in the
 * arguments.  The PV_Init object is not modified or deleted during the call.
 */
int outputParams(PV_Init *initObj, char const *path);

#endif /* BUILDANDRUN_HPP_ */
