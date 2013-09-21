/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
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
   char * pre_layer_name = NULL;
   char * post_layer_name = NULL;
   PVParams * params = hc->parameters();
   if (!strcmp( keyword, "HyperConnDebugInitWeights") ) {
      HyPerConn::getPreAndPostLayerNames(name, params, &pre_layer_name, &post_layer_name);
      const char * copiedConnName = params->stringValue(name, "copiedConn");
      HyPerConn * copiedConn = hc->getConnFromName(copiedConnName);
      newobject = new HyperConnDebugInitWeights(name, hc, pre_layer_name, post_layer_name, copiedConn);
      if( !newobject ) {
          fprintf(stderr, "Group \"%s\": Unable to create HyperConnDebugInitWeights\n", name);
          errorFound = true;
      }
   }
   else if (!strcmp(keyword, "KernelConnDebugInitWeights") ) {
      HyPerConn::getPreAndPostLayerNames(name, params, &pre_layer_name, &post_layer_name);
      const char * copiedConnName = params->stringValue(name, "copiedConn");
      HyPerConn * copiedConn = hc->getConnFromName(copiedConnName);
      newobject = new KernelConnDebugInitWeights(name, hc, pre_layer_name, post_layer_name, copiedConn);
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
   free(pre_layer_name);
   free(post_layer_name);
   return newobject;
}
