/*
 * UpdateFromCloneTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "TestConnProbe.hpp"
#include "MomentumTestConnProbe.hpp"

int main(int argc, char * argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   status = pv_initObj.registerKeyword("TestConnProbe", Factory::standardCreate<TestConnProbe>);
   status = pv_initObj.registerKeyword("MomentumTestConnProbe", Factory::standardCreate<MomentumTestConnProbe>);
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
