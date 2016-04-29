/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "TriggerTestLayer.hpp"
#include "TriggerTestConn.hpp"
#include "TriggerTestLayerProbe.hpp"

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int status;
   status = pv_initObj.registerKeyword("TriggerTestLayer", createTriggerTestLayer);
   assert(status==PV_SUCCESS);
   status = pv_initObj.registerKeyword("TriggerTestConn", createTriggerTestConn);
   assert(status==PV_SUCCESS);
   status = pv_initObj.registerKeyword("TriggerTestLayerProbe", createTriggerTestLayerProbe);
   assert(status==PV_SUCCESS);
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
