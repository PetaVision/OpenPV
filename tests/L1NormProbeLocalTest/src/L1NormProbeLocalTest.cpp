#include "probes/L1NormProbeLocal.hpp"
#include "cMakeHeader.h"
#include "columns/HyPerCol.hpp"
#include "columns/Messages.hpp"
#include "columns/PV_Init.hpp"
#include "components/BasePublisherComponent.hpp"
#include "include/PVLayerLoc.hpp"
#include "include/pv_common.h"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Observer.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace PV;

int checkStoredValues(
      ProbeDataBuffer<double> const &storedValues,
      std::vector<std::vector<double>> const &correctValues);
double computeCorrectValue(int batchIndex, HyPerLayer *targetLayer);
double computeCorrectValue(int batchIndex, HyPerLayer *targetLayer, HyPerLayer *maskLayer);
HyPerLayer *findLayer(std::string const &name, HyPerCol &hc);
std::shared_ptr<LayerUpdateStateMessage>
makeUpdateMessage(double simTime, double deltaTime, bool *isPending, bool *hasActed);
L1NormProbeLocal makeProbeLocal(char const *name, HyPerCol &hc, HyPerLayer *targetLayer);
int run(PV_Init *pv_init_obj);
int runNoMask(PV_Init *pv_init_obj);
int runWithMask(PV_Init *pv_init_obj);
int runWithSingleFeatureMask(PV_Init *pv_init_obj);
void updateLayers(
      double simTime,
      HyPerLayer *targetLayer,
      HyPerLayer *maskLayer,
      L1NormProbeLocal &probeLocal,
      std::vector<double> &correctValues);

int main(int argc, char **argv) {
   PV_Init *pv_init_obj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);

   int status = run(pv_init_obj);

   if (status == PV_SUCCESS) {
      InfoLog().printf("Test passed.\n");
   }
   delete pv_init_obj;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkStoredValues(
      ProbeDataBuffer<double> const &storedValues,
      std::vector<std::vector<double>> const &correctValues) {
   int status        = PV_SUCCESS;
   int numTimestamps = static_cast<int>(storedValues.size());
   for (int t = 1; t <= numTimestamps; ++t) {
      ProbeData<double> const &storedValuesAtTime = storedValues.getData(t - 1);

      int nbatch = static_cast<int>(storedValuesAtTime.size());
      for (int b = 0; b < nbatch; ++b) {
         double observed = storedValuesAtTime.getValue(b);
         double correct  = correctValues.at(t - 1).at(b);
         if (observed != correct) {
            ErrorLog().printf(
                  "t=%d, b=%d, observed = %f, correct = %f, discrepancy %g\n",
                  t,
                  b,
                  observed,
                  correct,
                  observed - correct);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

L1NormProbeLocal makeProbeLocal(char const *name, HyPerCol &hc, HyPerLayer *targetLayer) {
   PVParams probeParams(
         "input/L1NormProbeLocalTest.params",
         static_cast<size_t>(3),
         hc.getCommunicator()->communicator());

   L1NormProbeLocal probeLocal(name, &probeParams);
   probeLocal.ioParamsFillGroup(PARAMS_IO_READ);

   ObserverTable objectTable = hc.getAllObjectsFlat();
   auto communicateMessage   = std::make_shared<CommunicateInitInfoMessage>(
         &objectTable,
         hc.getDeltaTime(),
         hc.getNxGlobal(),
         hc.getNyGlobal(),
         hc.getNBatchGlobal(),
         hc.getNumThreads());
   probeLocal.communicateInitInfo(communicateMessage);

   probeLocal.initializeState(targetLayer);
   return probeLocal;
}

std::shared_ptr<LayerUpdateStateMessage>
makeUpdateMessage(double simTime, double deltaTime, bool *isPending, bool *hasActed) {
   int phase = 0;
#ifdef PV_USE_CUDA
   bool recvGpuFlag   = false;
   bool updateGpuFlag = false;
   auto updateMessage = std::make_shared<LayerUpdateStateMessage>(
         phase, recvGpuFlag, updateGpuFlag, simTime, deltaTime, isPending, hasActed);
#else
   auto updateMessage = std::make_shared<LayerUpdateStateMessage>(
         phase, simTime, deltaTime, isPending, hasActed);
#endif // PV_USE_CUDA
   return updateMessage;
}

double computeCorrectValue(int batchIndex, HyPerLayer *targetLayer) {
   auto *targetLayerPublisher = targetLayer->getComponentByType<BasePublisherComponent>();

   int numExtended         = targetLayer->getNumExtended();
   float const *targetData = &targetLayerPublisher->getLayerData()[batchIndex * numExtended];

   PVLayerLoc const *targetLayerLoc = targetLayer->getLayerLoc();
   int nx                           = targetLayerLoc->nx;
   int ny                           = targetLayerLoc->ny;
   int nf                           = targetLayerLoc->nf;
   int lt                           = targetLayerLoc->halo.lt;
   int rt                           = targetLayerLoc->halo.rt;
   int dn                           = targetLayerLoc->halo.dn;
   int up                           = targetLayerLoc->halo.up;

   double sum     = 0.0;
   int numNeurons = targetLayer->getNumNeurons();
   for (int k = 0; k < numNeurons; ++k) {
      int kExt = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      sum += std::fabs((double)targetData[kExt]);
   }
   return sum;
}

double computeCorrectValue(int batchIndex, HyPerLayer *targetLayer, HyPerLayer *maskLayer) {
   if (maskLayer == nullptr) {
      return computeCorrectValue(batchIndex, targetLayer);
   }
   auto *targetLayerPublisher = targetLayer->getComponentByType<BasePublisherComponent>();

   int targetNumExtended   = targetLayer->getNumExtended();
   float const *targetData = &targetLayerPublisher->getLayerData()[batchIndex * targetNumExtended];

   PVLayerLoc const *targetLayerLoc = targetLayer->getLayerLoc();
   int targetNx                     = targetLayerLoc->nx;
   int targetNy                     = targetLayerLoc->ny;
   int targetNf                     = targetLayerLoc->nf;
   int targetLt                     = targetLayerLoc->halo.lt;
   int targetRt                     = targetLayerLoc->halo.rt;
   int targetDn                     = targetLayerLoc->halo.dn;
   int targetUp                     = targetLayerLoc->halo.up;

   auto *maskLayerPublisher = maskLayer->getComponentByType<BasePublisherComponent>();

   int maskNumExtended   = maskLayer->getNumExtended();
   float const *maskData = &maskLayerPublisher->getLayerData()[batchIndex * maskNumExtended];

   PVLayerLoc const *maskLayerLoc = maskLayer->getLayerLoc();
   FatalIf(
         maskLayerLoc->nx != targetNx or maskLayerLoc->ny != targetNy,
         "targetLayer \"%s\" and maskLayer \"%s\" have different (nx,ny)\n"
         "targetLayer has(%d,%d) versus maskLayer (%d,%d)\n",
         targetLayer->getName(),
         maskLayer->getName(),
         targetNx,
         targetNy,
         maskLayerLoc->nx,
         maskLayerLoc->ny);
   int maskNx = maskLayerLoc->nx;
   int maskNy = maskLayerLoc->ny;
   int maskNf = maskLayerLoc->nf;
   int maskLt = maskLayerLoc->halo.lt;
   int maskRt = maskLayerLoc->halo.rt;
   int maskDn = maskLayerLoc->halo.dn;
   int maskUp = maskLayerLoc->halo.up;

   int maskNumNeurons = maskLayer->getNumNeurons();

   double sum = 0.0;
   if (maskNf == targetNf) {
      pvAssert(targetLayer->getNumNeurons() == maskNumNeurons);
      for (int k = 0; k < maskNumNeurons; ++k) {
         int kExtMask = kIndexExtended(k, maskNx, maskNy, maskNf, maskLt, maskRt, maskDn, maskUp);
         if (maskData[kExtMask]) {
            int kExtTarget = kIndexExtended(
                  k, targetNx, targetNy, targetNf, targetLt, targetRt, targetDn, targetUp);
            sum += maskData[kExtMask] ? std::fabs((double)targetData[kExtTarget]) : 0.0;
         }
      }
   }
   else if (targetNf > 1 and maskNf == 1) {
      pvAssert(targetLayer->getNumNeurons() == targetNf * maskNumNeurons);
      for (int k = 0; k < maskNumNeurons; ++k) {
         int kExtMask = kIndexExtended(k, maskNx, maskNy, 1, maskLt, maskRt, maskDn, maskUp);
         if (maskData[kExtMask]) {
            for (int f = 0; f < targetNf; ++f) {
               int kTarget    = k * targetNf + f;
               int kExtTarget = kIndexExtended(
                     kTarget, targetNx, targetNy, targetNf, targetLt, targetRt, targetDn, targetUp);
               sum += maskData[kExtMask] ? std::fabs((double)targetData[kExtTarget]) : 0.0;
            }
         }
      }
   }
   else {
      Fatal().printf(
            "targetLayer \"%s\" and maskLayer \"%s\" have incompatible numbers of features (%d "
            "versus %d)\n",
            targetLayer->getName(),
            maskLayer->getName(),
            targetNf,
            maskNf);
   }
   return sum;
}

HyPerLayer *findLayer(std::string const &name, HyPerCol &hc) {
   Observer *object  = hc.getObjectFromName(name);
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(object);
   FatalIf(layer == nullptr, "Unable to find layer named \"%s\".\n", name.c_str());
   return layer;
}

int run(PV_Init *pv_init_obj) {
   FatalIf(
         pv_init_obj->getParams() != nullptr,
         "L1NormProbeLocalTest should be run without a params file.\n");
   pv_init_obj->setParams("input/ProbeTestLayers.params");

   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      status = runNoMask(pv_init_obj);
   }
   if (status == PV_SUCCESS) {
      status = runWithMask(pv_init_obj) == PV_SUCCESS ? status : PV_FAILURE;
      status = runWithSingleFeatureMask(pv_init_obj) == PV_SUCCESS ? status : PV_FAILURE;
   }
   return status;
}

int runNoMask(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;
   HyPerCol hc(pv_init_obj);
   hc.allocateColumn();

   HyPerLayer *targetLayer     = findLayer(std::string("TargetLayer"), hc);
   L1NormProbeLocal probeLocal = makeProbeLocal("ProbeWithNoMask", hc, targetLayer);
   std::vector<std::vector<double>> correctValues(3);

   for (int t = 1; t <= 3; ++t) {
      double simTime                           = static_cast<double>(t);
      std::vector<double> &correctValuesAtTime = correctValues.at(t - 1);
      updateLayers(simTime, targetLayer, nullptr /*maskLayer*/, probeLocal, correctValuesAtTime);
   }

   status = checkStoredValues(probeLocal.getStoredValues(), correctValues);
   return status;
}

int runWithMask(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;
   HyPerCol hc(pv_init_obj);
   hc.allocateColumn();

   HyPerLayer *targetLayer     = findLayer(std::string("TargetLayer"), hc);
   HyPerLayer *maskLayer       = findLayer(std::string("Layer3Features"), hc);
   L1NormProbeLocal probeLocal = makeProbeLocal("ProbeWith3FeatureMask", hc, targetLayer);
   std::vector<std::vector<double>> correctValues(3);

   for (int t = 1; t <= 3; ++t) {
      double simTime                           = static_cast<double>(t);
      std::vector<double> &correctValuesAtTime = correctValues.at(t - 1);
      updateLayers(simTime, targetLayer, maskLayer, probeLocal, correctValuesAtTime);
   }

   status = checkStoredValues(probeLocal.getStoredValues(), correctValues);
   return status;
}

int runWithSingleFeatureMask(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;
   HyPerCol hc(pv_init_obj);
   hc.allocateColumn();

   HyPerLayer *targetLayer     = findLayer(std::string("TargetLayer"), hc);
   HyPerLayer *maskLayer       = findLayer(std::string("Layer1Feature"), hc);
   L1NormProbeLocal probeLocal = makeProbeLocal("ProbeWith1FeatureMask", hc, targetLayer);
   std::vector<std::vector<double>> correctValues(3);

   for (int t = 1; t <= 3; ++t) {
      double simTime                           = static_cast<double>(t);
      std::vector<double> &correctValuesAtTime = correctValues.at(t - 1);
      updateLayers(simTime, targetLayer, maskLayer, probeLocal, correctValuesAtTime);
   }

   status = checkStoredValues(probeLocal.getStoredValues(), correctValues);
   return status;
}

void updateLayers(
      double simTime,
      HyPerLayer *targetLayer,
      HyPerLayer *maskLayer,
      L1NormProbeLocal &probeLocal,
      std::vector<double> &correctValues) {
   int phase        = 0;
   double deltaTime = 1.0;

   bool isPending         = false;
   bool hasActed          = false;
   auto clearFlagsMessage = std::make_shared<LayerClearProgressFlagsMessage>();
   auto updateMessage     = makeUpdateMessage(simTime, deltaTime, &isPending, &hasActed);
   auto publishMessage    = std::make_shared<LayerPublishMessage>(phase, simTime);
   auto outputMessage     = std::make_shared<LayerOutputStateMessage>(phase, simTime, deltaTime);

   // Simulate the column advancing by one timestep, by sending targetLayer and maskLayer the
   // messages that the HyPerCol would send.
   *(updateMessage->mSomeLayerIsPending) = false;
   *(updateMessage->mSomeLayerHasActed)  = false;
   targetLayer->respond(clearFlagsMessage);
   targetLayer->respond(updateMessage);
   targetLayer->respond(publishMessage);
   targetLayer->respond(outputMessage);
   if (maskLayer) {
      *(updateMessage->mSomeLayerIsPending) = false;
      *(updateMessage->mSomeLayerHasActed)  = false;
      maskLayer->respond(clearFlagsMessage);
      maskLayer->respond(updateMessage);
      maskLayer->respond(publishMessage);
      maskLayer->respond(outputMessage);
   }

   probeLocal.storeValues(simTime);

   int nbatch = targetLayer->getLayerLoc()->nbatch;
   correctValues.resize(nbatch);
   for (int b = 0; b < nbatch; ++b) {
      correctValues.at(b) = computeCorrectValue(b, targetLayer, maskLayer);
   }
}
