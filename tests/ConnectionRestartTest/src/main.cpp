/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char ** argv);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char ** argv) {
   BaseConnection * baseConn;
   baseConn = hc->getConnFromName("initializeFromInitWeights");
   HyPerConn * initializeFromInitWeightsConn = dynamic_cast<HyPerConn *>(baseConn);
   // There must be a connection named initializeFromInitWeights. It should have a single weight with value 1
   assert(initializeFromInitWeightsConn);
   assert(initializeFromInitWeightsConn->xPatchSize()==1);
   assert(initializeFromInitWeightsConn->yPatchSize()==1);
   assert(initializeFromInitWeightsConn->fPatchSize()==1);
   assert(initializeFromInitWeightsConn->numberOfAxonalArborLists()==1);
   assert(initializeFromInitWeightsConn->get_wData(0,0)[0] == 1.0f);

   // There must be a connection named initializeFromCheckpoint.  It should have a single weight with value 2
   baseConn = hc->getConnFromName("initializeFromCheckpoint");
   HyPerConn * initializeFromCheckpointConn = dynamic_cast<HyPerConn *>(baseConn);
   assert(initializeFromCheckpointConn);
   assert(initializeFromCheckpointConn->xPatchSize()==1);
   assert(initializeFromCheckpointConn->yPatchSize()==1);
   assert(initializeFromCheckpointConn->fPatchSize()==1);
   assert(initializeFromCheckpointConn->numberOfAxonalArborLists()==1);
   assert(initializeFromCheckpointConn->get_wData(0,0)[0] == 2.0f);
   return PV_SUCCESS;
}
