/*
 * pv.cpp
 *
 */

// using MPITestLayer
// activity/V are initialized to the global x/y/f position
// using uniform weights with total output strength of 1,
// all post synaptic cells should receive a total weighted input
// equal to thier global position
// MPITestProbe checks whether he above suppositions are satisfied

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "MPITestProbe.hpp"
#include "MPITestLayer.hpp"
#include <assert.h>

// use compiler directive in case MPITestLayer gets moved to PetaVision trunk
#define MAIN_USES_CUSTOMGROUP // TODO: rewrite using subclass of ParamGroupHandler

#ifdef MAIN_USES_CUSTOMGROUP
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
      else if (!strcmp(keyword, "MPITestLayer")) {
         result = LayerGroupType;
      }
      else if (!strcmp(keyword, "MPITestProbe")) {
         result = ProbeGroupType;
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
      else if (!strcmp(keyword, "MPITestLayer")) {
         addedLayer = new MPITestLayer(name, hc);
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
   virtual BaseProbe * createProbe (char const *keyword, char const *name, HyPerCol *hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedProbe;
      }
      else if (!strcmp(keyword, "MPITestProbe")) {
         addedProbe = new MPITestProbe(name, hc);
         if (addedProbe==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error creating %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }

      if (errorFound) {
         assert(addedProbe==NULL);
         fprintf(stderr, "CustomGroupHandler error: could not create %s \"%s\"\n", keyword, name);
      }
      return addedProbe;
   }
}; /* class CustomGroupHandler */
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   PV_Arguments * arguments = initObj->getArguments();
   if (arguments->getParamsFile()==NULL) {
      arguments->setParamsFile("input/MPI_test.params");
   }
#ifdef MAIN_USES_CUSTOMGROUP
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = rebuildandrun(initObj, NULL, NULL, &customGroupHandler, 1/*numGroupHandler*/)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
#else
   status = rebuildandrun(initObj, NULL, NULL, NULL/*groupHandlerList*/, 0/*numGroupHandler*/);
#endif // MAIN_USES_ADDCUSTOM
   delete initObj;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
