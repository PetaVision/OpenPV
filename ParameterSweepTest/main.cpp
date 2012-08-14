/*
 * pv.cpp
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "ParameterSweepTestProbe.hpp"

void * customgroups(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, NULL, &customgroups);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   int status;
   HyPerLayer * targetlayer;
   const char * filename;
   char * message;
   if( !strcmp( keyword, "ParameterSweepTestProbe") ) {
      status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer, &message, &filename);
      if(status != PV_SUCCESS) {
         fprintf(stderr, "Skipping params group \"%s\"\n", name);
         return NULL;
      }
      assert(targetlayer);
      ParameterSweepTestProbe * new_probe = new ParameterSweepTestProbe(filename, targetlayer, message);
      checknewobject((void *) new_probe, keyword, name, hc);
      return (void *) new_probe;
   }
   fprintf(stderr, "Group \"%s\": Keyword \"%s\" unrecognized.  Skipping group.\n", name, keyword);
   // TODO smarter error handling
   return NULL;
}
