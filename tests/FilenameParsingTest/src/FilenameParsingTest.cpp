/*
 * pv.cpp
 *
 */

#include "FilenameParsingProbe.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("FilenameParsingProbe", Factory::create<FilenameParsingProbe>);
   int status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
