/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "RescaleLayerTestProbe.hpp"
#include "RescaleLayerTestGroupHandler.hpp"

int main(int argc, char * argv[]) {

   int status;
   RescaleLayerTestGroupHandler * groupHandlerList[1];
   groupHandlerList[0] = new RescaleLayerTestGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, (ParamGroupHandler **) groupHandlerList, 1);
   delete groupHandlerList[0];
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
