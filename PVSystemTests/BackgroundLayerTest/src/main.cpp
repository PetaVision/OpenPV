/*
 * ReceiveFromPostTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "ComparisonLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

int main(int argc, char * argv[]) {

   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ComparisonLayer", createComparisonLayer);
   int status = buildandrun(&pv_initObj);

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
