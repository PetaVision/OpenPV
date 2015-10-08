/*
 * pv.cpp
 *
 */

// using ShrunkenPatchTestLayer
// activity/V are initialized to the global x/y/f position
// using uniform weights with total output strength of 1,
// all post synaptic cells should receive a total weighted input
// equal to thier global position
// ShrunkenPatchProbe checks whether he above suppositions are satisfied

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "CustomGroupHandler.hpp"
#include <assert.h>

int main(int argc, char * argv[]) {

   int status;
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   PV_Arguments * arguments = initObj->getArguments();
   if (arguments->getParamsFile() == NULL) {
      arguments->setParamsFile("input/ShrunkenPatchTest.params");
   }
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler;
   status = rebuildandrun(initObj, NULL, NULL, &customGroupHandler, 1)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   delete customGroupHandler;
   delete initObj;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
