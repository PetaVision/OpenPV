#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void ImportParamsLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ANNLayer::initialize(params, defaults, comm);

   std::string const &name = params->getName();
   if (name == "orig") {
      // Test grabbed value
      double const *nxScale =  params->read<double>("nxScale");
      FatalIf(*nxScale != 1.0, "Test failed.\n");
      // Test grabbed filename
      std::string const *filenameFromParams = params->read<std::string>("Vfilename");
      FatalIf(*filenameFromParams != "input/a0.pvp", "Test failed.\n");
   }
   else {
      // Test overwritten value
      double const *nxScale =  params->read<double>("nxScale");
      FatalIf(*nxScale != 2.0, "Test failed.\n");
      // Test overwritten filename
      std::string const *filenameFromParams = params->read<std::string>("Vfilename");
      FatalIf(*filenameFromParams != "input/a1.pvp", "Test failed.\n");
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
