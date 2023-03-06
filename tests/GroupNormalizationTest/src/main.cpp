/*
 * pv.cpp
 *
 */

#include "AllConstantValueProbe.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

int main(int argc, char *argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AllConstantValueProbe", Factory::create<AllConstantValueProbe>);
   status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
