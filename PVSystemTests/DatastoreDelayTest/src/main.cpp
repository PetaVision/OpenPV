/*
 * DatastoreDelayTest.cpp
 *
 */

// using DatastoreDelayLayer, an input layer is filled with
// random data with the property that summing across four
// adjacent rows gives zeroes.
//
// On each timestep the data is rotated by one column.
// The input goes through four connections, with delays 0,1,2,3,
// each on the excitatory channel.
//
// The output layer should therefore be all zeros.

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "DatastoreDelayTestLayer.hpp"
#include "DatastoreDelayTestProbe.hpp"
#include <assert.h>

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
      else if (!strcmp(keyword, "DatastoreDelayTestLayer")) {
         result = LayerGroupType;
      }
      else if (!strcmp(keyword, "DatastoreDelayTestProbe")) {
         result = ProbeGroupType;
      }
      else {
         result = UnrecognizedGroupType;
      }
      //
      // This routine should compare keyword to the list of keywords handled by CustomGroupHandler and return one of
      // LayerGroupType, ConnectionGroupType, ProbeGroupType, ColProbeGroupType, WeightInitializerGroupType, or WeightNormalizerGroupType
      // according to the keyword, or UnrecognizedGroupType if this ParamGroupHandler object does not know the keyword.
      //
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom objects
   // corresponding to that group type.
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc) {
      HyPerLayer * addedLayer = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedLayer;
      }
      else if (!strcmp(keyword, "DatastoreDelayTestLayer")) {
         addedLayer = new DatastoreDelayTestLayer(name, hc);
         if (addedLayer==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error: unable to create layer %s \"%s\".\n", keyword, name);
      }
      
      if (errorFound) {
         assert(addedLayer==NULL);
         fprintf(stderr, "CustomGroupHandler error creating layer %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }
      return addedLayer;
   }
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedProbe;
      }
      else if (!strcmp(keyword, "DatastoreDelayTestProbe")) {
         addedProbe = new DatastoreDelayTestProbe(name, hc);
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
}; /* class CustomGroupHandler */
#endif // MAIN_USES_CUSTOMGROUP

int main(int argc, char * argv[]) {

    int status;
#ifdef MAIN_USES_CUSTOMGROUP // TODO: rewrite using subclass of ParamGroupHandler
    PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
    PV_Arguments * arguments = initObj->getArguments();
    if (arguments->getParamsFile()==NULL) {
        arguments->setParamsFile("input/DatastoreDelayTest.params");
    }
    ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
    status = rebuildandrun(initObj, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
#else
    status = buildandrun(argc, argv, NULL/*groupHandlerList*/, 0/*numGroupHandlers*/);
#endif // MAIN_USES_ADDCUSTOM
    delete initObj;
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
