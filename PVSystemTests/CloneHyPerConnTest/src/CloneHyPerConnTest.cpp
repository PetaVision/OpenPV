/*
 * Main file for CloneHyPerConnTest
 * To run, use arguments -p input/CloneHyPerConnTest.params
 *
 */


#include <columns/buildandrun.hpp>
#include "CloneHyPerConnTestProbe.hpp"

#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
// CustomGroupHandler is for adding objects not supported by CoreParamGroupHandler().
class CustomGroupHandler: public ParamGroupHandler {
public:
   CustomGroupHandler() {}

   virtual ~CustomGroupHandler() {}

   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      if (keyword==NULL) {
         return result;
      }
      else if (!strcmp(keyword, "CloneHyPerConnTestProbe")) {
         return ProbeGroupType;
      }
      //
      // This routine should compare keyword to the list of keywords handled by CustomGroupHandler and return one of
      // LayerGroupType, ConnectionGroupType, ProbeGroupType, ColProbeGroupType, WeightInitializerGroupType, or WeightNormalizerGroupType
      // according to the keyword, or UnrecognizedGroupType if this ParamGroupHandler object does not know the keyword.
      //
      return result;
   }
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedProbe;
      }
      else if (!strcmp(keyword, "CloneHyPerConnTestProbe")) {
         addedProbe = new CloneHyPerConnTestProbe(name, hc);
         if (addedProbe==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error: unable to create probe %s \"%s\".\n", keyword, name);
      }
      
      if (errorFound) {
         assert(addedProbe==NULL);
         fprintf(stderr, "CustomGroupHandler error creating probe %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }
      return addedProbe;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom objects
   // corresponding to that group type.

}; /* class CustomGroupHandler */
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUP
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUP
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
