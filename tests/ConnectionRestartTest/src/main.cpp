/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>

int customexit(HyPerCol *hc, int argc, char **argv);

int main(int argc, char *argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char **argv) {
   Observer *baseObject;
   baseObject                               = hc->getObjectFromName("initializeFromInitWeights");
   HyPerConn *initializeFromInitWeightsConn = dynamic_cast<HyPerConn *>(baseObject);
   // There must be a connection named initializeFromInitWeights. It should have a single weight
   // with value 1
   FatalIf(!initializeFromInitWeightsConn, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->xPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->yPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->fPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->numberOfAxonalArborLists() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->get_wData(0, 0)[0] != 1.0f, "Test failed.\n");

   // There must be a connection named initializeFromCheckpoint.  It should have a single weight
   // with value 2
   baseObject                              = hc->getObjectFromName("initializeFromCheckpoint");
   HyPerConn *initializeFromCheckpointConn = dynamic_cast<HyPerConn *>(baseObject);
   FatalIf(!initializeFromCheckpointConn, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->xPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->yPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->fPatchSize() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->numberOfAxonalArborLists() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->get_wData(0, 0)[0] != 2.0f, "Test failed.\n");
   return PV_SUCCESS;
}
