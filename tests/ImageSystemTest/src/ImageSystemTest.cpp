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

#define MAIN_USES_CUSTOM_GROUPS

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOM_GROUPS
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ImageTestLayer", createImageTestLayer);
   pv_initObj.registerKeyword("ImagePvpTestLayer", createImagePvpTestLayer);
   pv_initObj.registerKeyword("MovieTestLayer", createMovieTestLayer);
   pv_initObj.registerKeyword("MoviePvpTestLayer", createMoviePvpTestLayer);
   status = buildandrun(&pv_initObj);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOM_GROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
