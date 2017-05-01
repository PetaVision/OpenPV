/*
 * main function for BatchMethodTest
 */

#include "FixedImageSequenceByFile.hpp"
#include "FixedImageSequenceByList.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword(
         "FixedImageSequenceByFile", Factory::create<FixedImageSequenceByFile>);
   pv_initObj.registerKeyword(
         "FixedImageSequenceByList", Factory::create<FixedImageSequenceByList>);
   int status = buildandrun(&pv_initObj, NULL, NULL);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
