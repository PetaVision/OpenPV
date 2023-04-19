/*
 * pv.cpp
 *
 */

#include "ArborTestForOnesProbe.hpp"
#include "ArborTestProbe.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ArborTestProbe", Factory::create<ArborTestProbe>);
   pv_initObj.registerKeyword("ArborTestForOnesProbe", Factory::create<ArborTestForOnesProbe>);
   int status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
