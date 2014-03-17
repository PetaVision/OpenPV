/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TriggerTestLayer.hpp"
#include "TriggerTestConn.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   void* addedGroup= NULL;

   if ( !strcmp(keyword, "TriggerTestLayer") ) {
      addedGroup = new TriggerTestLayer(groupname, hc);
   }
   if ( !strcmp(keyword, "TriggerTestConn") ) {
      char * preLayerName = NULL;
      char * postLayerName = NULL;
      HyPerConn::getPreAndPostLayerNames(groupname, hc->parameters(), &preLayerName, &postLayerName);
      addedGroup = new TriggerTestConn(groupname, hc, preLayerName, postLayerName);
   }
   if (!addedGroup) {
      fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
      exit(EXIT_SUCCESS);
   }
   checknewobject((void *) addedGroup, keyword, groupname, hc);
   return addedGroup;
}
