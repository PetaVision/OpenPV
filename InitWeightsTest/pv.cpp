/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "HyperConnDebugInitWeights.hpp"
#include "KernelConnDebugInitWeights.hpp"
#include "InitWeightTestProbe.hpp"

int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not understood by build().

int main(int argc, char * argv[]) {
   return buildandrun(argc, argv, &addcustom)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int addcustom(HyPerCol * hc, int argc, char * argv[]) {

   int numOfConns=hc->numberOfConnections();
   int rank=hc->icCommunicator()->commRank();
   for(int i=0; i<numOfConns;i++){
      HyPerConn *tempConnPtr = hc->getConnection(i);
      const char * weightInitTypeStr = hc->parameters()->stringValue(tempConnPtr->getName(), "weightInitType");
      ParameterGroup *grp =hc->parameters()->group(tempConnPtr->getName());

      if( rank==0 ) {
         printf("connection %s\n", tempConnPtr->getName());
         printf("weightInitType %s\n", weightInitTypeStr);
         printf("keyword %s\n", grp->getGroupKeyword());
         char newName[100];
         strncpy(newName, tempConnPtr->getName(), 94);
         strncat(newName, "_copy", 5);
         printf("new conn name %s\n", newName);
      }

      HyPerLayer * preLayer = tempConnPtr->getPre();
      HyPerLayer * postLayer = tempConnPtr->getPost();
      // PVParams * params = hc->parameters();
      // ChannelType channelType = CHANNEL_INH;
      //int channelNo = (int) params->value(tempConnPtr->getName(), "channelCode", -1);
      HyPerConn * newConn;
      if( !strcmp(grp->getGroupKeyword(), "HyPerConn") ) {
         newConn = new HyperConnDebugInitWeights(tempConnPtr->getName(), hc, preLayer, postLayer, tempConnPtr);
      }
      else if( !strcmp(grp->getGroupKeyword(), "KernelConn") ) {
         newConn = new KernelConnDebugInitWeights(tempConnPtr->getName(), hc, preLayer, postLayer, tempConnPtr);
      }

   }

   PVParams * params = hc->parameters();
   int status;
   int numGroups = params->numberOfGroups();
   for (int n = 0; n < numGroups; n++) {
      const char * kw = params->groupKeywordFromIndex(n);
      const char * name = params->groupNameFromIndex(n);
      HyPerLayer * targetlayer;
      char * message = NULL;
      const char * filename;
      InitWeightTestProbe * addedProbe;
      if (!strcmp(kw, "InitWeightTestProbe")) {
         status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer,
               &message, &filename);
         if (status != PV_SUCCESS) {
            fprintf(stderr, "Skipping params group \"%s\"\n", name);
            continue;
         }
         if( filename ) {
            addedProbe =  new InitWeightTestProbe(filename, targetlayer, message);
         }
         else {
            addedProbe =  new InitWeightTestProbe(targetlayer, message);
         }
         free(message); message=NULL;
         if( !addedProbe ) {
            fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         }
         assert(targetlayer);
         checknewobject((void *) addedProbe, kw, name, hc);
      }
   }

   return PV_SUCCESS;
}
