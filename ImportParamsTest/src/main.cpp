/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "ImportParamsLayer.hpp"
#include "ImportParamsConn.hpp"

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
   if (strcmp(keyword, "ImportParamsLayer") == 0){
      HyPerLayer * addedLayer = (HyPerLayer *) new ImportParamsLayer(name, hc);
      int status = checknewobject((void *) addedLayer, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
      assert(status == PV_SUCCESS);
      addedGroup = (void *) addedLayer;
   }

   char * preLayerName = NULL;
   char * postLayerName = NULL;
   if (strcmp(keyword, "ImportParamsConn") == 0){
      HyPerConn::getPreAndPostLayerNames(name, hc->parameters(), &preLayerName, &postLayerName);
      HyPerConn * addedConn = (HyPerConn *) new ImportParamsConn(name, hc, preLayerName, postLayerName);
      int status = checknewobject((void *) addedConn, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
      assert(status == PV_SUCCESS);
      addedGroup = (void *) addedConn;
   }
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
