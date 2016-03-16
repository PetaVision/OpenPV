/*
 * ImageOffsetTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "ImageOffsetTestLayer.hpp"

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
      // LayerGroupType, ConnectionGroupType, ProbeGroupType, ColProbeGroupType, WeightInitializerGroupType, or
      // WeightNormalizerGroupType
      // according to the keyword, or UnrecognizedGroupType if this ParamGroupHandler object does not know the keyword.
      //
      if (keyword==NULL) {
         return result;
      }
      else if (!strcmp(keyword, "ImageOffsetTestLayer")) {
         result = LayerGroupType;
      }
      else {
         result = UnrecognizedGroupType;
      }
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom
   // objects
   // corresponding to that group type.
   virtual HyPerLayer * createLayer (char const *keyword, char const *name, HyPerCol *hc) {
      HyPerLayer * addedLayer = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedLayer;
      }
      else if (!strcmp(keyword, "ImageOffsetTestLayer")) {
         addedLayer = new ImageOffsetTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error creating %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }

      if (errorFound) {
         assert(addedLayer==NULL);
         fprintf(stderr, "CustomGroupHandler error: could not create %s \"%s\"\n", keyword, name);
      }
      return addedLayer;
   }
}; /* class CustomGroupHandler */
#endif // MAIN_USES_CUSTOMGROUPS

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
#else
   status = buildandrun(argc, argv, NULL/*customGroupHandlerList*/, 0/*numGroupHandlers*/);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
