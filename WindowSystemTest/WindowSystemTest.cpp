/*
 * ImageSystemTest
 *
 * The idea of this test is to somehow make sure that negative offsets, mirrorBC and imageBC
 * flags have not been broken.
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "WindowTestLayer.hpp"
#include "WindowLCALayer.hpp"
#include "WindowProbe.hpp"

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
   if (strcmp(keyword, "WindowLCALayer") == 0){
      HyPerLayer * addedLayer = (HyPerLayer *) new WindowLCALayer(name, hc);
      int status = checknewobject((void *) addedLayer, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
      assert(status == PV_SUCCESS);
      addedGroup = (void *) addedLayer;
   }
   if (strcmp(keyword, "WindowTestLayer") == 0){
      HyPerLayer * addedLayer = (HyPerLayer *) new WindowTestLayer(name, hc);
      int status = checknewobject((void *) addedLayer, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
      assert(status == PV_SUCCESS);
      addedGroup = (void *) addedLayer;
   }
   if (strcmp(keyword, "WindowProbe") == 0){
      addedGroup = new WindowProbe(hc->getLayerFromName(hc->parameters()->stringValue(name, "targetLayer")),
            hc->parameters()->stringValue(name, "message"));
   }
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
