/*
 * ReceiveFromPostTest
 *
 *
 */

#include "ReceiveFromPostProbe.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

int main(int argc, char *argv[]) {

   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ReceiveFromPostProbe", Factory::create<ReceiveFromPostProbe>);
   int status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
