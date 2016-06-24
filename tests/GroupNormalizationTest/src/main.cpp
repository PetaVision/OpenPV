/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "AllConstantValueProbe.hpp"

int main(int argc, char * argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AllConstantValueProbe", createAllConstantValueProbe);
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
