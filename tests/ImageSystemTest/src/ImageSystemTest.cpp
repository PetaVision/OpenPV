/*
 * ImageSystemTest
 *
 * The idea of this test is to somehow make sure that negative offsets, mirrorBC and imageBC
 * flags have not been broken.
 *
 */


#include <columns/buildandrun.hpp>
#include "ImageTestLayer.hpp"
#include "ImagePvpTestLayer.hpp"
#include "MovieTestLayer.hpp"
#include "MoviePvpTestLayer.hpp"

int main(int argc, char * argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ImageTestLayer", Factory::create<ImageTestLayer>);
   pv_initObj.registerKeyword("ImagePvpTestLayer", Factory::create<ImagePvpTestLayer>);
   pv_initObj.registerKeyword("MovieTestLayer", Factory::create<MovieTestLayer>);
   pv_initObj.registerKeyword("MoviePvpTestLayer", Factory::create<MoviePvpTestLayer>);
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
