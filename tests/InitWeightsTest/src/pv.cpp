/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "HyPerConnDebugInitWeights.hpp"
#include "InitGaborWeights.hpp"
#include "InitWeightTestProbe.hpp"
#include "KernelConnDebugInitWeights.hpp"

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("HyPerConnDebugInitWeights", createHyPerConnDebugInitWeights);
   pv_initObj.registerKeyword("GaborWeight", createInitGaborWeights);
   pv_initObj.registerKeyword("InitWeightTestProbe", createInitWeightTestProbe);
   pv_initObj.registerKeyword("KernelConnDebugInitWeights", createKernelConnDebugInitWeights);
   int status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
