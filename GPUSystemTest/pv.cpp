/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "GPUTestProbe.hpp"
#include "GPUTestForOnesProbe.hpp"
#include "GPUTestForTwosProbe.hpp"
#include "GPUTestForNinesProbe.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   int status;
   LayerProbe * addedProbe = NULL;
   HyPerLayer * targetlayer;
   char * message = NULL;
   const char * filename;
   if( !strcmp( keyword, "ArborTestProbe") ) {
      addedProbe = new GPUTestProbe(groupname, hc);
   }
   if( !strcmp( keyword, "ArborTestForOnesProbe") ) {
      addedProbe = new GPUTestForOnesProbe(groupname, hc);
   }
   if( !strcmp( keyword, "ArborTestForTwosProbe") ) {
      addedProbe = new GPUTestForTwosProbe(groupname, hc);
   }
   if( !strcmp( keyword, "ArborTestForNinesProbe") ) {
      addedProbe = new GPUTestForTwosProbe(groupname, hc);
   }
   if (addedProbe) {
      status = checknewobject((void *) addedProbe, keyword, groupname, hc);
   }
   else {
      status = PV_FAILURE;
      if (hc->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: snrecognized params keyword \"%s\"\n", keyword, groupname, keyword);
      }
   }
   if( status != PV_SUCCESS ) {
      exit(EXIT_FAILURE);
   }
   assert(addedProbe);
   return addedProbe;
}
