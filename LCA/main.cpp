/*
 * pv.cpp
 *
 */


#include <src/columns/buildandrun.hpp>
#include "LCALayer.hpp"
#include "LCAProbe.hpp"

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "LCALayer")) {
      addedGroup = (void *) new LCALayer(name, hc, MAX_CHANNELS);
   }
   if( !strcmp(keyword, "LCAProbe") ) {
      LCAProbe * addedProbe = NULL;
      HyPerLayer * targetlayer = NULL;
      char * message = NULL;
      const char * filename = NULL;
      int status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer, &message, &filename);
      int errorFound = status!=PV_SUCCESS;
      int xLoc, yLoc, fLoc;
      PVParams * params = targetlayer->getParent()->parameters();
      if( !errorFound ) {
         xLoc = params->value(name, "xLoc", -1);
         yLoc = params->value(name, "yLoc", -1);
         fLoc = params->value(name, "fLoc", -1);
         if( xLoc <= -1 || yLoc <= -1 || fLoc <= -1) {
            fprintf(stderr, "Group \"%s\": Class %s requires xLoc, yLoc, and fLoc be set\n", name, keyword);
            errorFound = true;
         }
      }
      if( !errorFound ) {
         if( filename ) {
            addedProbe =new LCAProbe(filename, targetlayer, xLoc, yLoc, fLoc, message);
         }
         else {
            addedProbe = new LCAProbe(targetlayer, xLoc, yLoc, fLoc, message);
         }
         if( !addedProbe ) {
             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
             errorFound = true;
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
      addedGroup = (void *) addedProbe;
   }
   checknewobject(addedGroup, keyword, name, hc);
   return addedGroup;
}
