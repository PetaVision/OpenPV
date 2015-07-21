/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

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
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom objects
   // corresponding to that group type.

}; /* class CustomGroupHandler */
#endif // MAIN_USES_CUSTOMGROUPS

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
