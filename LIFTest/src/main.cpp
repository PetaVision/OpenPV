/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
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
   HyPerLayer * spikecount = hc->getLayerFromName("LIFGapTestSpikeCounter");
   int status = spikecount != NULL ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS) {
      if (hc->icCommunicator()->commRank()==0) {
         fprintf(stderr, "Error:  No layer named \"LIFGapTestSpikeCounter\"");
      }
      status = PV_FAILURE;
   }
   return status;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "AverageRateConn") ) {
      addedGroup = (void *) new AverageRateConn(name, hc);
   }
   if (!strcmp(keyword, "LIFTestProbe") ) {
      addedGroup = (void *) new LIFTestProbe(name, hc);
   }
   checknewobject((void *) addedGroup, keyword, name, hc);
   return addedGroup;
}
