/*
 * BroadcastPreLayerTest program
 */

#include <columns/buildandrun.hpp>
#include <columns/Communicator.hpp>
#include <columns/HyPerCol.hpp>
#include <components/ActivityComponent.hpp>
#include <include/PVLayerLoc.hpp>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <io/LayerFile.hpp>
#include <layers/HyPerLayer.hpp>
#include <structures/Buffer.hpp>
#include <utils/PVLog.hpp>

#include <memory>
#include <vector>

int checkOutput(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, &checkOutput);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkOutput(HyPerCol *hc, int argc, char *argv[]) {
   auto *outputObject = hc->getObjectFromName("Output");
   FatalIf(outputObject == nullptr, "No object named \"Output\"\n");
   HyPerLayer *outputLayer = dynamic_cast<HyPerLayer *>(outputObject);
   FatalIf(outputLayer == nullptr, "Object \"Output\" is not a HyPerLayer\n");
   auto *outputActivity = outputLayer->getComponentByType<ActivityComponent>();
   FatalIf(outputActivity == nullptr, "Object \"Output\" does not have an ActivityComponent\n");
   auto *outputActivityBuffer = outputActivity->getComponentByType<ActivityBuffer>();
   FatalIf(outputActivity == nullptr, "Object \"Output\" does not have an ActivityBuffer\n");
   float const *outputData = outputActivityBuffer->getBufferData();

   int status = PV_SUCCESS;

   PVLayerLoc const *loc = outputLayer->getLayerLoc();
   FatalIf(
         loc->nbatchGlobal != 1,
         "This test currently expects batch size of 1 (instead it is %d)\n", loc->nbatchGlobal);
   FatalIf(
         loc->halo.lt != 0 or loc->halo.rt != 0 or loc->halo.dn != 0 or loc->halo.up != 0,
         "output layer should have margins 0 but instead has (lt=%d, rt=%d, dn=%d, up=%d)\n",
         loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   int dataSize = loc->nx * loc->ny * loc->nf;

   Communicator *communicator = hc->getCommunicator();
   auto fileManager = std::make_shared<FileManager>(communicator->getGlobalMPIBlock(), ".");
   LayerFile correctValuesFile(
         fileManager,
         "input/correct.pvp",
         *loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/);
   std::vector<float> correctData(dataSize);
   correctValuesFile.setDataLocation(correctData.data(), 0);
   correctValuesFile.read();

   bool isCorrect = std::equal(outputData, &outputData[dataSize], correctData.begin());
   FatalIf(!isCorrect, "Output data does not match correct.pvp\n");

   return status;
}
