/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
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
      ParameterSweepTestProbe * new_probe = new ParameterSweepTestProbe(name, hc);
      checknewobject((void *) new_probe, keyword, name, hc);
      return (void *) new_probe;
   }
   fprintf(stderr, "Group \"%s\": Keyword \"%s\" unrecognized.  Skipping group.\n", name, keyword);
   // TODO smarter error handling
   return NULL;
}
