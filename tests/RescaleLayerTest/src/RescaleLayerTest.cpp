/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "RescaleLayerTestProbe.hpp"

int main(int argc, char * argv[]) {

   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("RescaleLayerTestProbe", Factory::standardCreate<RescaleLayerTestProbe>);
   int status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
