/*
 * pv.cpp
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "AverageRateConn.hpp"
#include "LIFTestProbe.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);
// customexit is called after each entry in the parameter sweep (or once at the end if there are no parameter sweeps) and before the HyPerCol is deleted.
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * spikecount = hc->getLayerFromName("LIFGapTest Spike Counter");
   int status = spikecount != NULL ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS) {
      if (hc->icCommunicator()->commRank()==0) {
         fprintf(stderr, "Error:  No layer named \"LIFGapTest Spike Counter\"");
      }
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {

   }
   return 0;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "AverageRateConn") ) {
      HyPerLayer * pre, * post;
      AverageRateConn * g = NULL;
      getPreAndPostLayers(name, hc, &pre, &post);
      if( pre && post ) {
         g = new AverageRateConn(name, hc, pre, post);
      }
      if (checknewobject((void *) g, keyword, name, hc) == PV_SUCCESS) {
         addedGroup = (void *) g;
      }
   }
   if (!strcmp(keyword, "LIFTestProbe") ) {
      LIFTestProbe * p = NULL;
      HyPerLayer * target_layer;
      char * message;
      const char * filename;
      int status = getLayerFunctionProbeParameters(name, keyword, hc, &target_layer, &message, &filename);
      if( status == PV_SUCCESS ) {
         PVBufType buf_type = BufV;
         if( filename ) {
            p = new LIFTestProbe(filename, target_layer, buf_type, message);
         }
         else {
            p = new LIFTestProbe(target_layer, buf_type, message);
         }
         if (checknewobject((void *) p, keyword, name, hc) == PV_SUCCESS) {
            addedGroup = (void *) p;
         }
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
   }
   return addedGroup;
}
