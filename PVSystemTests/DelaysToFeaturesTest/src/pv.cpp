/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include "DelayTestProbe.hpp"

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
      if (keyword==NULL) {
         return result;
      }
      else if (!strcmp(keyword, "DelayTestProbe")) {
         result = ProbeGroupType;
      }
      else {
         result = UnrecognizedGroupType;
      }
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom objects
   // corresponding to that group type.
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
      }
      else if (!strcmp(keyword, "DelayTestProbe")) {
         addedProbe = new DelayTestProbe(name, hc);
         if (addedProbe==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler unable to create probe %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }
      if (errorFound) {
         assert(addedProbe==NULL);
         fprintf(stderr, "CustomGroupHandler error: creating %s \"%s\" failed.\n", keyword, name);
      }
      return addedProbe;
   }
}; /* class CustomGroupHandler */

int main(int argc, char * argv[]) {
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   int status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
