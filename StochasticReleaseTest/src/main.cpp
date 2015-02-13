/*
 * main.cpp
 *
 * Minimal interface to PetaVision
 */


#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include "StochasticReleaseTestProbe.hpp"

namespace PV {

class StochasticReleaseTestGroupHandler: public ParamGroupHandler {
public:
   StochasticReleaseTestGroupHandler() {}

   virtual ~StochasticReleaseTestGroupHandler() {}

   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      if (!strcmp(keyword, "StochasticReleaseTestProbe")) {
         result = ProbeGroupType;
      }
      else {
         result = UnrecognizedGroupType;
      }
      return result;
   }

   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      int status;
      BaseProbe * addedObject = NULL;
      if( !strcmp(keyword, "StochasticReleaseTestProbe") ) {
         addedObject = new StochasticReleaseTestProbe(name, hc);
         if( !addedObject ) {
            fprintf(stderr, "Group \"%s\": Unable to create %s\n", name, keyword);
         }
      }
      return addedObject;
   }
}; /* class StochasticReleaseTestGroupHandler */

} /* namespace PV */


int main(int argc, char * argv[]) {

   int status;
   StochasticReleaseTestGroupHandler * customGroupHandler = new StochasticReleaseTestGroupHandler;
   status = buildandrun(argc, argv, NULL, NULL, (PV::ParamGroupHandler **) &customGroupHandler, 1);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
