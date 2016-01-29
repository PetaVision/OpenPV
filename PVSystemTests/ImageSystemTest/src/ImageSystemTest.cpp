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
//#include "BatchImageTestLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
#ifdef PV_USE_GDAL
   if (strcmp(keyword, "ImageTestLayer") == 0){
      addedGroup = new ImageTestLayer(name, hc);
   }
   if (strcmp(keyword, "MovieTestLayer") == 0){
      addedGroup = new MovieTestLayer(name, hc);
   }
#endif // PV_USE_GDAL
   if (strcmp(keyword, "ImagePvpTestLayer") == 0){
      addedGroup = new ImagePvpTestLayer(name, hc);
   }
   if (strcmp(keyword, "MoviePvpTestLayer") == 0){
      addedGroup = new MoviePvpTestLayer(name, hc);
   }
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
