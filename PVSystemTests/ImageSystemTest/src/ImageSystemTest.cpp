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
#include <io/ParamGroupHandler.hpp>
// CustomGroupHandler is for adding objects not supported by CoreParamGroupHandler().
class CustomGroupHandler: public ParamGroupHandler {
public:
   CustomGroupHandler() {}

   virtual ~CustomGroupHandler() {}

   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      //
      // This routine should compare keyword to the list of keywords handled by CustomGroupHandler and return one of
      // LayerGroupType, ConnectionGroupType, ProbeGroupType, ColProbeGroupType, WeightInitializerGroupType, or WeightNormalizerGroupType
      // according to the keyword, or UnrecognizedGroupType if this ParamGroupHandler object does not know the keyword.
      //
      if (keyword == NULL) {
         return result;
      }
      else if (!strcmp(keyword, "ImageTestLayer")) {
         return LayerGroupType;
      }
      else if (!strcmp(keyword, "ImagePvpTestLayer")) {
         return LayerGroupType;
      }
      else if (!strcmp(keyword, "MovieTestLayer")) {
         return LayerGroupType;
      }
      else if (!strcmp(keyword, "MoviePvpTestLayer")) {
         return LayerGroupType;
      }
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom objects
   // corresponding to that group type.
   virtual HyPerLayer * createLayer (char const *keyword, char const *name, HyPerCol *hc) {
      HyPerLayer * addedLayer = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedLayer;
      }
      else if (!strcmp(keyword, "ImageTestLayer")) {
         addedLayer = new ImageTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else if (!strcmp(keyword, "ImagePvpTestLayer")) {
         addedLayer = new ImagePvpTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else if (!strcmp(keyword, "MovieTestLayer")) {
         addedLayer = new MovieTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else if (!strcmp(keyword, "MoviePvpTestLayer")) {
         addedLayer = new MoviePvpTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error creating %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
         addedLayer = NULL;
      }
      if (errorFound) {
         assert(addedLayer == NULL);
         fprintf(stderr, "CustomGroupHandler error: could not create %s \"%s\"\n", keyword, name);
      }
      return addedLayer;
   }

}; /* class CustomGroupHandler */
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
