/*
 * main.cpp
 *
 * Minimal interface to PetaVision
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "StochasticReleaseTestProbe.hpp"

int main(int argc, char * argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("StochasticReleaseTestProbe", createStochasticReleaseTestProbe);
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
