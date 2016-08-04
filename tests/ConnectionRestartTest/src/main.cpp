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
   pvErrorIf(!(initializeFromInitWeightsConn), "Test failed.\n");
   pvErrorIf(!(initializeFromInitWeightsConn->xPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromInitWeightsConn->yPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromInitWeightsConn->fPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromInitWeightsConn->numberOfAxonalArborLists()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromInitWeightsConn->get_wData(0,0)[0] == 1.0f), "Test failed.\n");

   // There must be a connection named initializeFromCheckpoint.  It should have a single weight with value 2
   baseConn = hc->getConnFromName("initializeFromCheckpoint");
   HyPerConn * initializeFromCheckpointConn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(initializeFromCheckpointConn), "Test failed.\n");
   pvErrorIf(!(initializeFromCheckpointConn->xPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromCheckpointConn->yPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromCheckpointConn->fPatchSize()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromCheckpointConn->numberOfAxonalArborLists()==1), "Test failed.\n");
   pvErrorIf(!(initializeFromCheckpointConn->get_wData(0,0)[0] == 2.0f), "Test failed.\n");
   return PV_SUCCESS;
}
