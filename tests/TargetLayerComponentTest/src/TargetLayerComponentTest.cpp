#include <columns/HyPerCol.hpp>
#include <columns/Messages.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <observerpattern/ObserverTable.hpp>
#include <probes/TargetLayerComponent.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>
#include <string>

using PV::HyPerCol;
using PV::HyPerLayer;
using PV::PV_Init;
using PV::TargetLayerComponent;

TargetLayerComponent initTargetLayerObject(
      HyPerCol &hypercol,
      std::string const &probeName,
      std::string const &layerName);
int run(PV::PV_Init &pv_init);

int main(int argc, char **argv) {
   PV_Init pv_init(&argc, &argv, false);
   int status = run(pv_init);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

TargetLayerComponent initTargetLayerObject(
      HyPerCol &hypercol,
      std::string const &probeName,
      std::string const &layerName) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("TargetLayerComponent \"").append(probeName).append("\" = {\n");
   paramsString.append("   targetLayer = \"").append(layerName).append("\";\n");
   paramsString.append("};\n");

   MPI_Comm mpiComm = hypercol.getCommunicator()->globalCommunicator();
   PV::PVParams params(paramsString.data(), paramsString.size(), 1UL, mpiComm);

   TargetLayerComponent targetLayerObject(probeName.c_str(), &params);
   targetLayerObject.ioParamsFillGroup(PV::PARAMS_IO_READ);

   PV::ObserverTable objectTable = hypercol.getAllObjectsFlat();
   auto communicateMessage       = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hypercol.getDeltaTime(),
         hypercol.getNxGlobal(),
         hypercol.getNyGlobal(),
         hypercol.getNBatchGlobal(),
         hypercol.getNumThreads());
   targetLayerObject.communicateInitInfo(communicateMessage);

   return targetLayerObject;
}

int run(PV::PV_Init &pv_init) {
   PV::HyPerCol hypercol(&pv_init);
   hypercol.allocateColumn();

   std::string probeName("Probe");
   std::string layerName("TestLayer");
   TargetLayerComponent targetLayerObj = initTargetLayerObject(hypercol, probeName, layerName);

   char const *nameFromTargetLayerObject = targetLayerObj.getTargetLayerName();
   FatalIf(
         layerName != nameFromTargetLayerObject,
         "TargetLayerComponent::getTargetLayerName() returned %s instead of %s\n",
         nameFromTargetLayerObject,
         layerName.c_str());

   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hypercol.getObjectFromName(layerName));
   pvAssert(layer != nullptr);

   HyPerLayer *layerFromTargetLayerObj = targetLayerObj.getTargetLayer();
   FatalIf(
         layerFromTargetLayerObj != layer,
         "TargetLayerComponent::getTargetLayer() failed (return value %p instead of %p)\n",
         layerFromTargetLayerObj,
         layer);
   return PV_SUCCESS;
}
