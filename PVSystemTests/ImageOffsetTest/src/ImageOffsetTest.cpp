/*
 * ImageOffsetTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "ImageOffsetTestLayer.hpp"

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
   if (strcmp(keyword, "ImageOffsetTestLayer") == 0){
      addedGroup = new ImageOffsetTestLayer(name, hc);
   }
#endif // PV_USE_GDAL
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
