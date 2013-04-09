/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "HyperConnDebugInitWeights.hpp"
#include "KernelConnDebugInitWeights.hpp"
#include "InitWeightTestProbe.hpp"

// customgroups is for adding objects not understood by build().
void * customgroups(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {
   return buildandrun(argc, argv, NULL, NULL, &customgroups)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   bool errorFound = false;
   void * newobject = NULL;
   if (!strcmp( keyword, "HyperConnDebugInitWeights") ) {
      HyPerLayer * pre = NULL;
      HyPerLayer * post = NULL;
      getPreAndPostLayers(name, hc, &pre, &post);
      const char * copiedConnName = hc->parameters()->stringValue(name, "copiedConn");
      HyPerConn * copiedConn = hc->getConnFromName(copiedConnName);
      newobject = new HyperConnDebugInitWeights(name, hc, pre, post, copiedConn);
      if( !newobject ) {
          fprintf(stderr, "Group \"%s\": Unable to create HyperConnDebugInitWeights\n", name);
          errorFound = true;
      }
   }
   else if (!strcmp(keyword, "KernelConnDebugInitWeights") ) {
      HyPerLayer * pre = NULL;
      HyPerLayer * post = NULL;
      getPreAndPostLayers(name, hc, &pre, &post);
      const char * copiedConnName = hc->parameters()->stringValue(name, "copiedConn");
      HyPerConn * copiedConn = hc->getConnFromName(copiedConnName);
      newobject = new KernelConnDebugInitWeights(name, hc, pre, post, copiedConn);
      if( !newobject ) {
          fprintf(stderr, "Group \"%s\": Unable to create KernelConnDebugInitWeights\n", name);
          errorFound = true;
      }
   }
   else if (!strcmp(keyword, "InitWeightTestProbe")) {
      HyPerLayer * targetlayer;
      char * message = NULL;
      const char * filename;
      int status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer,
            &message, &filename);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "Skipping params group \"%s\"\n", name);
         return NULL;
      }
      if( filename ) {
         newobject =  new InitWeightTestProbe(filename, targetlayer, message);
      }
      else {
         newobject =  new InitWeightTestProbe(targetlayer, message);
      }
      free(message); message=NULL;
      if( !newobject ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
      }
      assert(targetlayer);
      checknewobject(newobject, keyword, name, hc);
   }
   return newobject;
}
