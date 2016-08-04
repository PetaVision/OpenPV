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
   status = pv_initObj.registerKeyword("TriggerTestLayer", Factory::create<TriggerTestLayer>);
   pvErrorIf(!(status==PV_SUCCESS), "Test failed.\n");
   status = pv_initObj.registerKeyword("TriggerTestConn", Factory::create<TriggerTestConn>);
   pvErrorIf(!(status==PV_SUCCESS), "Test failed.\n");
   status = pv_initObj.registerKeyword("TriggerTestLayerProbe", Factory::create<TriggerTestLayerProbe>);
   pvErrorIf(!(status==PV_SUCCESS), "Test failed.\n");
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
