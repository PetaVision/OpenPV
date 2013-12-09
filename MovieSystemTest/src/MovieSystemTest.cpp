/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TestAllZerosProbe.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   void * addedProbe = NULL;
   HyPerLayer * target_layer = NULL;
   char * message = NULL;
   const char * filename = NULL;

   if ( !strcmp(keyword, "TestAllZerosProbe") ) {
      int status = getLayerFunctionProbeParameters(groupname, keyword, hc, &target_layer, &message, &filename);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
         return addedProbe;
      }
      if (filename) {
         addedProbe = new TestAllZerosProbe(filename, target_layer, message);
      }
      else {
         addedProbe = new TestAllZerosProbe(target_layer, message);
      }
   }
   free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   if (!addedProbe) {
      fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
      exit(EXIT_SUCCESS);
   }
   assert(target_layer);
   checknewobject((void *) addedProbe, keyword, groupname, hc);
   return addedProbe;
}
