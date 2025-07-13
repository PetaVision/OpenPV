#include <columns/HyPerCol.hpp>
#include <columns/Messages.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <params/PVParams.hpp>
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

int run(PV::PV_Init &pv_init);

int main(int argc, char **argv) {
   PV_Init pv_init(&argc, &argv, false);
   int status = run(pv_init);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(PV::PV_Init &pv_init) {
   PV::HyPerCol hypercol(&pv_init);
   hypercol.allocateColumn();

   std::string probeName("Probe");
   std::string layerName("TestLayer");

   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("TargetLayerComponent \"").append(probeName).append("\" = {\n");
   paramsString.append("   targetLayer = \"").append(layerName).append("\";\n");
   paramsString.append("};\n");

   MPI_Comm mpiComm = hypercol.getCommunicator()->globalCommunicator();
   PV::PVParams params(paramsString.data(), paramsString.size(), mpiComm);

   auto paramsIO = params.makeParamsIO(probeName);
   TargetLayerComponent targetLayerObj(paramsIO->getParams(), paramsIO->getDefaults());
   targetLayerObj.ioParamsFillGroup(PV::ParamsIOSwitch::Read);

   PV::ObserverTable objectTable = hypercol.getAllObjectsFlat();
   auto communicateMessage       = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hypercol.getDeltaTime(),
         hypercol.getNxGlobal(),
         hypercol.getNyGlobal(),
         hypercol.getNBatchGlobal(),
         hypercol.getNumThreads());
   targetLayerObj.communicateInitInfo(communicateMessage);

   std::string const &nameFromTargetLayerObject = targetLayerObj.getTargetLayerName();
   FatalIf(
         layerName != nameFromTargetLayerObject,
         "TargetLayerComponent::getTargetLayerName() returned %s instead of %s\n",
         nameFromTargetLayerObject.c_str(),
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
