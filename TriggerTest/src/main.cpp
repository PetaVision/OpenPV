/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TriggerTestLayer.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   HyPerLayer * addedLayer= NULL;

   if ( !strcmp(keyword, "TriggerTestLayer") ) {
      addedLayer = new TriggerTestLayer(groupname, hc);
   }
   if (!addedLayer) {
      fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
      exit(EXIT_SUCCESS);
   }
   checknewobject((void *) addedLayer, keyword, groupname, hc);
   return addedLayer;
}
