/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>

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
   FatalIf(initializeFromInitWeightsConn->getPatchSizeX() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->getPatchSizeY() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->getPatchSizeF() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->getNumAxonalArbors() != 1, "Test failed.\n");
   FatalIf(initializeFromInitWeightsConn->getWeightsData(0, 0)[0] != 1.0f, "Test failed.\n");

   // There must be a connection named initializeFromCheckpoint.  It should have a single weight
   // with value 2
   baseObject                              = hc->getObjectFromName("initializeFromCheckpoint");
   HyPerConn *initializeFromCheckpointConn = dynamic_cast<HyPerConn *>(baseObject);
   FatalIf(!initializeFromCheckpointConn, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->getPatchSizeX() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->getPatchSizeY() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->getPatchSizeF() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->getNumAxonalArbors() != 1, "Test failed.\n");
   FatalIf(initializeFromCheckpointConn->getWeightsData(0, 0)[0] != 2.0f, "Test failed.\n");
   return PV_SUCCESS;
}
