/*
 * ImageOffsetTest
 *
 *
 */


#include <columns/buildandrun.hpp>

#include <columns/PV_Init.hpp>
#include "ImageOffsetTestLayer.hpp"
#include "ImagePvpOffsetTestLayer.hpp"

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ImageOffsetTestLayer", PV::Factory::create<ImageOffsetTestLayer>);
   pv_initObj.registerKeyword("ImagePvpOffsetTestLayer", PV::Factory::create<ImagePvpOffsetTestLayer>);
   int status = buildandrun(&pv_initObj, NULL, NULL);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
