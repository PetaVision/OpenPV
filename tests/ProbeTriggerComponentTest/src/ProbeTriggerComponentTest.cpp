#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <layers/HyPerLayer.hpp>
#include <observerpattern/ObserverTable.hpp>
#include <probes/ProbeTriggerComponent.hpp>

using PV::HyPerCol;
using PV::HyPerLayer;
using PV::ProbeTriggerComponent;
using PV::PV_Init;

ProbeTriggerComponent initTriggerObject(
      HyPerCol &hypercol,
      std::string const &probeName,
      std::string const &layerName,
      int offset);
int run(PV::PV_Init &pv_init, int offset);

int main(int argc, char **argv) {
   int status = PV_SUCCESS;

   PV_Init pv_init(&argc, &argv, false);
   for (int offset = 0; offset <= 3; offset++) {
      if (run(pv_init, offset) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

ProbeTriggerComponent initTriggerObject(
      HyPerCol &hypercol,
      std::string const &probeName,
      std::string const &layerName,
      int offset) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("ProbeTriggerComponent \"").append(probeName).append("\" = {\n");
   paramsString.append("   triggerLayerName = \"").append(layerName).append("\";\n");
   paramsString.append("   triggerOffset = ").append(std::to_string(offset)).append(";\n");
   paramsString.append("};\n");

   MPI_Comm mpiComm = hypercol.getCommunicator()->globalCommunicator();
   PV::PVParams params(paramsString.data(), paramsString.size(), 1UL, mpiComm);

   ProbeTriggerComponent triggerObject(probeName.c_str(), &params);
   triggerObject.ioParamsFillGroup(PV::PARAMS_IO_READ);

   PV::ObserverTable objectTable = hypercol.getAllObjectsFlat();
   auto communicateMessage       = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hypercol.getDeltaTime(),
         hypercol.getNxGlobal(),
         hypercol.getNyGlobal(),
         hypercol.getNBatchGlobal(),
         hypercol.getNumThreads());
   triggerObject.communicateInitInfo(communicateMessage);

   return triggerObject;
}

int run(PV::PV_Init &pv_init, int offset) {
   int status = PV_SUCCESS;
   PV::HyPerCol hypercol(&pv_init);
   hypercol.allocateColumn();

   std::string probeName("Probe");
   std::string layerName("TestLayer");
   ProbeTriggerComponent triggerObject = initTriggerObject(hypercol, probeName, layerName, offset);

   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hypercol.getObjectFromName(layerName));
   pvAssert(layer != nullptr);

   for (int t = 1; t <= 100; ++t) {
      double simTime          = static_cast<double>(t);
      double deltaT           = hypercol.getDeltaTime();
      bool pending            = false;
      bool acted              = false;
      auto updateStateMessage = std::make_shared<PV::LayerUpdateStateMessage>(
            0 /*phase*/,
#ifdef PV_USE_CUDA
            false /* recvOnGpuFlag*/,
            false /* updateOnGpuFlag*/,
#endif // PV_USE_CUDA
            simTime,
            deltaT,
            &pending,
            &acted);
      layer->respond(updateStateMessage);
      bool updateNeeded = triggerObject.needUpdate(simTime, deltaT);
      bool correct      = (t + offset) % 10 == 1;
      if (updateNeeded != correct) {
         ErrorLog().printf(
               "Offset %d, t = %3d, needUpdate() should return %s but returned %s\n",
               offset,
               t,
               correct ? "true" : "false",
               updateNeeded ? "true" : "false");
         status = PV_FAILURE;
      }
   }

   return status;
}
