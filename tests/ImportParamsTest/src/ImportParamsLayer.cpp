#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

void ImportParamsLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ANNLayer::initialize(paramsIO, comm);

   std::string const &name = paramsIO->getName();
   if (name == "orig") {
      // Test grabbed value
      double nxScale =  paramsIO->readValue<double>("nxScale");
      FatalIf(nxScale != 1.0, "Test failed.\n");
      // Test grabbed filename
      std::string filenameFromParams = paramsIO->readValue<std::string>("Vfilename");
      FatalIf(filenameFromParams != "input/a0.pvp", "Test failed.\n");
   }
   else {
      // Test overwritten value
      double nxScale =  paramsIO->readValue<double>("nxScale");
      FatalIf(nxScale != 2.0, "Test failed.\n");
      // Test overwritten filename
      std::string filenameFromParams = paramsIO->readValue<std::string>("Vfilename");
      FatalIf(filenameFromParams != "input/a1.pvp", "Test failed.\n");
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
