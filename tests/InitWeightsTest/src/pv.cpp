/*
 * pv.cpp
 *
 */

#include "HyPerConnDebugInitWeights.hpp"
#include "InitGaborWeights.hpp"
#include "InitWeightTestProbe.hpp"
#include "KernelConnDebugInitWeights.hpp"
#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword(
         "HyPerConnDebugInitWeights", Factory::create<HyPerConnDebugInitWeights>);
   pv_initObj.registerKeyword("GaborWeight", Factory::create<InitGaborWeights>);
   pv_initObj.registerKeyword("InitWeightTestProbe", Factory::create<InitWeightTestProbe>);
   pv_initObj.registerKeyword(
         "KernelConnDebugInitWeights", Factory::create<KernelConnDebugInitWeights>);
   int status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
