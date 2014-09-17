/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "ArborTestProbe.hpp"
#include "ArborTestForOnesProbe.hpp"

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
      addedProbe = new ArborTestProbe(groupname, hc);
   }
   if( !strcmp( keyword, "ArborTestForOnesProbe") ) {
      addedProbe = new ArborTestForOnesProbe(groupname, hc);
   }
   checknewobject(addedProbe, keyword, groupname, hc);
   if(!addedProbe)
   {
      fprintf(stderr, "Unrecognized params keyword \"%s\"\n", keyword);
      exit(EXIT_FAILURE);
   }
   return addedProbe;
}
