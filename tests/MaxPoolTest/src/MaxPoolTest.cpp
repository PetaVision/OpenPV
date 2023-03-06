/*
 * ReceiveFromPostTest
 *
 *
 */

#include "GateMaxPoolTestLayer.hpp"
#include "MaxPoolTestLayer.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

#define MAIN_USES_CUSTOMGROUPS

int main(int argc, char *argv[]) {

#ifdef MAIN_USES_CUSTOMGROUPS
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("MaxPoolTestLayer", Factory::create<MaxPoolTestLayer>);
   pv_initObj.registerKeyword("GateMaxPoolTestLayer", Factory::create<GateMaxPoolTestLayer>);
   int status = buildandrun(&pv_initObj, NULL, NULL);
#else // MAIN_USES_CUSTOMGROUPS
   int status = buildandrun(argc, argv, NULL, NULL);
#endif // MAIN_USES_CUSTOMGROUPS

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
