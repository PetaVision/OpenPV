/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "AverageRateConn.hpp"
#include "LIFTestProbe.hpp"
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
      else if (!strcmp(keyword, "AverageRateConn")) {
         result = ConnectionGroupType;
      }
      else if (!strcmp(keyword, "LIFTestProbe")) {
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
   virtual BaseConnection * createConnection (char const *keyword, char const *name, HyPerCol *hc) {
      BaseConnection * addedConn = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedConn;
      }
      else if (!strcmp(keyword, "AverageRateConn")) {
         addedConn = new AverageRateConn(name, hc);
         if (addedConn==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupHandler error creating %s \"%s\": %s is not a recognized keyword.\n", keyword, name, keyword);
      }

      if (errorFound) {
         assert(addedConn==NULL);
         fprintf(stderr, "CustomGroupHandler error: could not create %s \"%s\"\n", keyword, name);
      }
      return addedConn;
   }
   virtual BaseProbe * createProbe (char const *keyword, char const *name, HyPerCol *hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedProbe;
      }
      else if (!strcmp(keyword, "LIFTestProbe")) {
         addedProbe = new LIFTestProbe(name, hc);
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

int customexit(HyPerCol * hc, int argc, char * argv[]);
// customexit is called after each entry in the parameter sweep (or once at the end if there are no parameter sweeps) and before the HyPerCol is deleted.


int main(int argc, char * argv[]) {

   int status;
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, &customexit, &customGroupHandler, 1/*numGroupHandlers*/);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * spikecount = hc->getLayerFromName("LIFGapTestSpikeCounter");
   int status = spikecount != NULL ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS) {
      if (hc->icCommunicator()->commRank()==0) {
         fprintf(stderr, "Error:  No layer named \"LIFGapTestSpikeCounter\"");
      }
      status = PV_FAILURE;
   }
   return status;
}
