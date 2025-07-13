#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize_base();
   initialize(params, defaults, comm);
}

int ImportParamsConn::initialize_base() { return PV_SUCCESS; }

void ImportParamsConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);

   // Test grabbed array value
   std::vector<double> const *delayVals = params->read<std::vector<double>>("delay");
   std::string const &preLayerName      = getComponentByType<ConnectionData>()->getPreLayerName();

   std::string const &name = params->getName();
   if (name == "origConn") {
      FatalIf(delayVals->size() != 2, "Test failed.\n");
      FatalIf(delayVals->at(0) != 0, "Test failed.\n");
      FatalIf(delayVals->at(1) != 1, "Test failed.\n");
      FatalIf(preLayerName != "orig", "Test failed.\n");
   }
   else {
      FatalIf(delayVals->size() != 3, "Test failed.\n");
      FatalIf(delayVals->at(0) != 3, "Test failed.\n");
      FatalIf(delayVals->at(1) != 4, "Test failed.\n");
      FatalIf(delayVals->at(2) != 5, "Test failed.\n");
      FatalIf(preLayerName != "copy", "Test failed.\n");
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
