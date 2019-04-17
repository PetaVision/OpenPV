/*
 * pv.cpp
 *
 */

// using ShrunkenPatchTestLayer
// activity/V are initialized to the global x/y/f position
// using uniform weights with total output strength of 1,
// all post synaptic cells should receive a total weighted input
// equal to their global position
// ShrunkenPatchProbe checks whether he above suppositions are satisfied

#include "ShrunkenPatchTestLayer.hpp"
#include "ShrunkenPatchTestProbe.hpp"
#include "columns/buildandrun.hpp"
#include <utils/PVLog.hpp>

int main(int argc, char *argv[]) {

   int status;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   initObj.registerKeyword("ShrunkenPatchTestLayer", Factory::create<ShrunkenPatchTestLayer>);
   initObj.registerKeyword("ShrunkenPatchTestProbe", Factory::create<ShrunkenPatchTestProbe>);
   if (initObj.getParams() == NULL) {
      initObj.setParams("input/ShrunkenPatchTest.params");
   }
   status = rebuildandrun(&initObj) == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
