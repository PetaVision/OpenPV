#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

int ImportParamsConn::initialize_base() { return PV_SUCCESS; }

void ImportParamsConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);

   // Test grabbed array value
   int size;
   const float *delayVals = params->arrayValues(name, "delay", &size);
   auto *preLayerName     = getComponentByType<ConnectionData>()->getPreLayerName();
   if (strcmp(name, "origConn") == 0) {
      FatalIf(size != 2, "Test failed.\n");
      FatalIf(delayVals[0] != 0, "Test failed.\n");
      FatalIf(delayVals[1] != 1, "Test failed.\n");
      FatalIf(strcmp(preLayerName, "orig") != 0, "Test failed.\n");
   }
   else {
      FatalIf(size != 3, "Test failed.\n");
      FatalIf(delayVals[0] != 3, "Test failed.\n");
      FatalIf(delayVals[1] != 4, "Test failed.\n");
      FatalIf(delayVals[2] != 5, "Test failed.\n");
      FatalIf(strcmp(preLayerName, "copy") != 0, "Test failed.\n");
   }
}

Response::Status
ImportParamsConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return HyPerConn::communicateInitInfo(message);
}

Response::Status ImportParamsConn::allocateDataStructures() {
   return HyPerConn::allocateDataStructures();
}

} /* namespace PV */
