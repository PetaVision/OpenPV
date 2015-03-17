/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TestAllZerosProbe.hpp"

class CustomGroupHandler : public PV::ParamGroupHandler {
public:
   CustomGroupHandler() {}
   virtual ~CustomGroupHandler() {}

   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      if (keyword == NULL) { return result; }
      else if (!strcmp(keyword, "TestAllZerosProbe")) { result = ProbeGroupType; }
      return result;
   }

   // Uncomment and define the methods below for any custom keywords that need to be handled.
   // virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }
   // virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) { return NULL; }

   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      BaseProbe * addedProbe = NULL;
      if (keyword == NULL || getGroupType(keyword) != ProbeGroupType) { return addedProbe; }
      else if (!strcmp(keyword, "TestAllZerosProbe")) {
         addedProbe = new TestAllZerosProbe(name, hc);
      }
      if (addedProbe==NULL) {
         fprintf(stderr, "Rank %d process unable to add %s \"%s\"\n", hc->columnId(), keyword, name);
         exit(EXIT_FAILURE);
      }
      return addedProbe;
    }

   // virtual InitWeights * createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }
   // virtual NormalizeBase * createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }
};

int main(int argc, char * argv[]) {
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   int status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
