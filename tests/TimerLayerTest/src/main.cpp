/*
 * main.cpp
 */

#include <cassert>
#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <components/ActivityBuffer.hpp>
#include <components/ActivityComponent.hpp>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>

int checkEqual(int observed, int correct, char const *valuesDescription);
int checkOutput(HyPerCol *hc, int argc, char *argv[]);
ActivityBuffer const *getBuffer(HyPerCol *hc, char const *layerName);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, checkOutput);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkEqual(int observed, int correct, char const *valuesDescription) {
   if (observed != correct) {
      ErrorLog().printf(
            "Output and CheckOutput layers have different %s (%d versus %d)\n",
            valuesDescription, observed, correct);
      return PV_FAILURE;
   }
   return PV_SUCCESS;
}

int checkOutput(HyPerCol *hc, int argc, char *argv[]) {
   int status = PV_SUCCESS;

   std::string const &paramsFilename = hc->getPV_InitObj()->getStringArgument("ParamsFile");

   ActivityBuffer const *correctData = getBuffer(hc, "CheckOutput");
   ActivityBuffer const *observedData = getBuffer(hc, "Output");

   PVLayerLoc const *correctLoc = correctData->getLayerLoc();
   PVLayerLoc const *observedLoc = observedData->getLayerLoc();
   if (checkEqual(observedLoc->nbatch, correctLoc->nbatch, "batch sizes") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->nx, correctLoc->nx, "nx") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->ny, correctLoc->ny, "ny") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->nf, correctLoc->nf, "nf") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->halo.lt, correctLoc->halo.lt, "halo.lt") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->halo.rt, correctLoc->halo.rt, "halo.rt") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->halo.dn, correctLoc->halo.dn, "halo.dn") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->halo.up, correctLoc->halo.up, "halo.up") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (checkEqual(observedLoc->bcast, correctLoc->bcast, "broadcast flags") != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   FatalIf(
         status != PV_SUCCESS, "Output and CheckOutput have different dimensions. Test failed.\n");

   int N = correctData->getBufferSizeAcrossBatch();
   assert(observedData->getBufferSizeAcrossBatch() == N);
   float const *observed = observedData->getBufferData();
   float const *correct = correctData->getBufferData();
   for (int n = 0; n < N; ++n) {
      if (observed[n] != correct[n]) {
         ErrorLog().printf(
               "Output and CheckOutput data buffers differ at index %d (%f versus %f)\n",
               n, static_cast<double>(observed[n]), static_cast<double>(correct[n]));
         status = PV_FAILURE;
      }
   }
   return status;
}

ActivityBuffer const *getBuffer(HyPerCol *hc, char const *layerName) {
   auto *layer = hc->getTable()->findObject<HyPerLayer>(layerName);
   FatalIf(layer == nullptr, "Unable to find layer \"%s\"\n", layerName);
   auto *activityComponent = layer->getComponentByType<PV::ActivityComponent>();
   FatalIf(activityComponent == nullptr, "Layer \"%s\" has no ActivityComponent\n", layerName);
   auto *activityBuffer = activityComponent->getComponentByType<PV::ActivityBuffer>();
   FatalIf(activityBuffer == nullptr, "Layer \"%s\" has no ActivityBuffer\n", layerName);
   return activityBuffer;
}
