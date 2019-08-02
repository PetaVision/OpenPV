#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void ImportParamsLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   ANNLayer::initialize(name, params, comm);

   if (strcmp(name, "orig") == 0) {
      // Test grabbed value
      FatalIf(!(params->value(name, "nxScale") == 1), "Test failed.\n");
      // Test grabbed filename
      FatalIf(
            !(strcmp(params->stringValue(name, "Vfilename"), "input/a0.pvp") == 0),
            "Test failed.\n");
   }
   else {
      // Test overwritten value
      FatalIf(!(params->value(name, "nxScale") == 2), "Test failed.\n");
      // Test overwritten filename
      FatalIf(
            !(strcmp(params->stringValue(name, "Vfilename"), "input/a1.pvp") == 0),
            "Test failed.\n");
   }
}

Response::Status
ImportParamsLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return ANNLayer::communicateInitInfo(message);
}

Response::Status ImportParamsLayer::allocateDataStructures() {
   return ANNLayer::allocateDataStructures();
}

} /* namespace PV */
