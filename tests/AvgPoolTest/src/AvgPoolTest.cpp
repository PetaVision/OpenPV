/*
 * ReceiveFromPostTest
 *
 *
 */

#include "AvgPoolTestInputLayer.hpp"
#include "AvgPoolTestLayer.hpp"
#include "GateAvgPoolTestLayer.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

#define MAIN_USES_CUSTOMGROUPS

int main(int argc, char *argv[]) {

#ifdef MAIN_USES_CUSTOMGROUPS
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AvgPoolTestLayer", Factory::create<AvgPoolTestLayer>);
   pv_initObj.registerKeyword("AvgPoolTestInputLayer", Factory::create<AvgPoolTestInputLayer>);
   pv_initObj.registerKeyword("GateAvgPoolTestLayer", Factory::create<GateAvgPoolTestLayer>);
   int status = buildandrun(&pv_initObj);
#else // MAIN_USES_CUSTOMGROUPS
   int status = buildandrun(argc, argv, NULL, NULL);
#endif // MAIN_USES_CUSTOMGROUPS

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
