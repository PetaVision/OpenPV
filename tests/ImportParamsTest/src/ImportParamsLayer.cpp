#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int ImportParamsLayer::initialize_base() { return PV_SUCCESS; }

int ImportParamsLayer::initialize(const char *name, HyPerCol *hc) {
   ANNLayer::initialize(name, hc);

   PVParams *params = parent->parameters();
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

   return PV_SUCCESS;
}

Response::Status
ImportParamsLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return ANNLayer::communicateInitInfo(message);
}

Response::Status ImportParamsLayer::allocateDataStructures() {
   return ANNLayer::allocateDataStructures();
}

} /* namespace PV */
