#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int ImportParamsConn::initialize_base() { return PV_SUCCESS; }

int ImportParamsConn::initialize(const char *name, HyPerCol *hc) {
   HyPerConn::initialize(name, hc);

   PVParams *params = parent->parameters();
   // Test grabbed array value
   int size;
   const float *delayVals = params->arrayValues(name, "delay", &size);
   if (strcmp(name, "origConn") == 0) {
      FatalIf(size != 2, "Test failed.\n");
      FatalIf(delayVals[0] != 0, "Test failed.\n");
      FatalIf(delayVals[1] != 1, "Test failed.\n");
      FatalIf(strcmp(getPreLayerName(), "orig") != 0, "Test failed.\n");
   }
   else {
      FatalIf(size != 3, "Test failed.\n");
      FatalIf(delayVals[0] != 3, "Test failed.\n");
      FatalIf(delayVals[1] != 4, "Test failed.\n");
      FatalIf(delayVals[2] != 5, "Test failed.\n");
      FatalIf(strcmp(getPreLayerName(), "copy") != 0, "Test failed.\n");
   }

   return PV_SUCCESS;
}

Response::Status
ImportParamsConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return HyPerConn::communicateInitInfo(message);
}

Response::Status ImportParamsConn::allocateDataStructures() {
   return HyPerConn::allocateDataStructures();
}

} /* namespace PV */
