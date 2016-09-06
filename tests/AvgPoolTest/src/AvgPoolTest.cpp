/*
 * ReceiveFromPostTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "AvgPoolTestLayer.hpp"
#include "AvgPoolTestInputLayer.hpp"
#include "GatePoolTestLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

int main(int argc, char * argv[]) {

#ifdef MAIN_USES_CUSTOMGROUPS
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AvgPoolTestLayer", Factory::create<AvgPoolTestLayer>);
   pv_initObj.registerKeyword("AvgPoolTestInputLayer", Factory::create<AvgPoolTestInputLayer>);
   pv_initObj.registerKeyword("GatePoolTestLayer", Factory::create<GatePoolTestLayer>);
   int status = buildandrun(&pv_initObj);
#else // MAIN_USES_CUSTOMGROUPS
   int status = buildandrun(argc, argv, NULL, NULL);
#endif // MAIN_USES_CUSTOMGROUPS

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
