/*
 * main.cpp
 *
 */

#include "TriggerTestConn.hpp"
#include "TriggerTestLayer.hpp"
#include "TriggerTestLayerProbe.hpp"
#include <columns/Factory.hpp>
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   int status;
   status = pv_initObj.registerKeyword("TriggerTestLayer", Factory::create<TriggerTestLayer>);
   FatalIf(status != PV_SUCCESS, "Unable to register params keyword \"TriggerTestLayer\".\n");
   status = pv_initObj.registerKeyword("TriggerTestConn", Factory::create<TriggerTestConn>);
   FatalIf(status != PV_SUCCESS, "Unable to register params keyword \"TriggerTestConn\".\n");
   status = pv_initObj.registerKeyword(
         "TriggerTestLayerProbe", Factory::create<TriggerTestLayerProbe>);
   FatalIf(status != PV_SUCCESS, "Unable to register params keyword \"TriggerTestLayerProbe\".\n");
   FatalIf(!pv_initObj.getParams(), "%s was called without having set a params file\n", argv[0]);
   std::string customDefaultsPath("input/DefaultParams.txt");
   status = pv_initObj.registerDefaults(customDefaultsPath);
   FatalIf(status != PV_SUCCESS, "Error parsing \"%s\"\n", customDefaultsPath.c_str());
   status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
