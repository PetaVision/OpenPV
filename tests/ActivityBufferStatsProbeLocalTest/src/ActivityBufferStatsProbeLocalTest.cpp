#include <arch/mpi/mpi.h>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <columns/Random.hpp>
#include <columns/RandomSeed.hpp>
#include <components/ActivityBuffer.hpp>
#include <components/ActivityComponent.hpp>
#include <components/BasePublisherComponent.hpp>
#include <include/PVLayerLoc.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <observerpattern/Observer.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/ProbeDataBuffer.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/conversions.hpp>

#include "CheckValue.hpp"

#include <cfloat>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using PV::ActivityBufferStatsProbeLocal;
using PV::ActivityComponent;
using PV::checkValue;
using PV::HyPerLayer;
using PV::LayerStats;
using PV::ProbeData;
using PV::ProbeDataBuffer;
using PV::StatsBufferType;

ProbeData<LayerStats> calcCorrectValues(
      double timestamp,
      std::vector<float> const &dataBuffer,
      PVLayerLoc const *loc,
      float nnzThreshold);
int compareStatsBatch(
      std::string const &baseDescription,
      ProbeData<LayerStats> const &observedBatch,
      ProbeData<LayerStats> const &correctBatch);
float *findDataBufferA(ActivityComponent *activityComponent);
void generateRandomValues(std::vector<float> &dataBuffer, float threshold, unsigned int seed);
std::vector<float> initDataBuffer(PVLayerLoc const *loc);
void setLayerData(HyPerLayer *layer, std::vector<float> const &values);
int testStoredValues(HyPerLayer *layer, float nnzThreshold, unsigned int seed);

int main(int argc, char **argv) {
   int status = PV_SUCCESS;

   auto *pv_init  = new PV::PV_Init(&argc, &argv, false);
   auto *hypercol = new PV::HyPerCol(pv_init);
   hypercol->allocateColumn();
   PV::Observer *testLayerObject = hypercol->getObjectFromName(std::string("TestLayer"));
   PV::HyPerLayer *testLayer     = dynamic_cast<PV::HyPerLayer *>(testLayerObject);

   if (testStoredValues(testLayer, 0.0f, 12345678U) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testStoredValues(testLayer, 0.1f, 34567890U) != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   delete hypercol;
   delete pv_init;

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

ProbeData<LayerStats> calcCorrectValues(
      double timestamp,
      std::vector<float> const &dataBuffer,
      PVLayerLoc const *loc,
      float nnzThreshold) {
   ProbeData<LayerStats> result(timestamp, loc->nbatch);

   pvAssert(static_cast<int>(dataBuffer.size()) % loc->nbatch == 0);
   int elementSize = static_cast<int>(dataBuffer.size()) / loc->nbatch;
   PVHalo halo     = loc->halo;

   for (int b = 0; b < loc->nbatch; ++b) {
      LayerStats &elementStats = result.getValue(b);
      float const *data        = &dataBuffer.at(b * elementSize);
      double correctSum        = 0.0f;
      double correctSumSquared = 0.0f;
      float correctMin         = FLT_MAX;
      float correctMax         = -FLT_MAX;
      int correctNumNeurons    = loc->nx * loc->ny * loc->nf;
      int correctNumNonzero    = 0;
      for (int k = 0; k < correctNumNeurons; ++k) {
         int kExt =
               PV::kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo.lt, halo.rt, halo.dn, halo.up);
         pvAssert(kExt >= 0 and kExt < elementSize);
         float value     = data[kExt];
         double valueDbl = static_cast<double>(value);
         correctSum += valueDbl;
         correctSumSquared += valueDbl * valueDbl;
         correctMin = value < correctMin ? value : correctMin;
         correctMax = value > correctMax ? value : correctMax;
         correctNumNonzero += (std::abs(value) > nnzThreshold);
      }
      elementStats.mSum        = correctSum;
      elementStats.mSumSquared = correctSumSquared;
      elementStats.mMin        = correctMin;
      elementStats.mMax        = correctMax;
      elementStats.mNumNeurons = correctNumNeurons;
      elementStats.mNumNonzero = correctNumNonzero; // number outside of threshold
   }

   return result;
}

int compareStatsBatch(
      std::string const &baseDescription,
      ProbeData<LayerStats> const &observedBatch,
      ProbeData<LayerStats> const &correctBatch) {
   int status            = PV_SUCCESS;
   int observedBatchSize = static_cast<int>(observedBatch.size());
   int correctBatchSize  = static_cast<int>(correctBatch.size());
   try {
      checkValue(baseDescription, std::string("size"), observedBatchSize, correctBatchSize, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      return PV_FAILURE;
   }

   try {
      auto observed = observedBatch.getTimestamp();
      auto correct  = correctBatch.getTimestamp();
      checkValue(baseDescription, std::string("timestamp"), observed, correct, 0.0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }

   for (int b = 0; b < observedBatchSize; ++b) {
      auto const &observedStats = observedBatch.getValue(b);
      auto const &correctStats  = correctBatch.getValue(b);
      std::string messageHead   = baseDescription + " batch element " + std::to_string(b);
      try {
         double observed = observedStats.mSum;
         double correct  = correctStats.mSum;
         checkValue(messageHead, std::string("sum"), observed, correct, 0.0);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }

      try {
         double observed = observedStats.mSumSquared;
         double correct  = correctStats.mSumSquared;
         checkValue(messageHead, std::string("sum of squares"), observed, correct, 1.0e-13);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }

      try {
         float observed = observedStats.mMin;
         float correct  = correctStats.mMin;
         checkValue(messageHead, std::string("min"), observed, correct, 0.0f);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }

      try {
         float observed = observedStats.mMax;
         float correct  = correctStats.mMax;
         checkValue(messageHead, std::string("max"), observed, correct, 0.0f);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }

      try {
         int observed = observedStats.mNumNeurons;
         int correct  = correctStats.mNumNeurons;
         checkValue(messageHead, std::string("NumNeurons"), observed, correct, 0);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }

      try {
         int observed = observedStats.mNumNonzero;
         int correct  = correctStats.mNumNonzero;
         checkValue(messageHead, std::string("NumNonzero"), observed, correct, 0);
      } catch (std::exception const &e) {
         ErrorLog() << e.what();
         status = PV_FAILURE;
      }
   }
   return status;
}

float *findDataBufferA(ActivityComponent *activityComponent) {
   auto layerDataA = activityComponent->getComponentByType<PV::ActivityBuffer>();
   FatalIf(
         layerDataA == nullptr,
         "Target layer \"%s\" does not have an activity buffer.\n",
         activityComponent->getName());
   return layerDataA->getReadWritePointer();
}

void generateRandomValues(std::vector<float> &dataBuffer, float threshold, unsigned int seed) {
   PV::RandomSeed::instance()->initialize(seed);
   int bufferSize = static_cast<int>(dataBuffer.size());
   auto rng       = PV::Random(bufferSize);
   rng.uniformRandom(&dataBuffer.at(0), 0, bufferSize, -1.0f, 1.0f);
   if (threshold > 0.0f) {
      for (auto &x : dataBuffer) {
         if (std::abs(x) < threshold) {
            x = 0.0f;
         }
      }
   }
}

std::vector<float> initDataBuffer(PVLayerLoc const *loc) {
   std::vector<float> dataBuffer;
   int nxExt = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt = loc->ny + loc->halo.dn + loc->halo.up;
   int area  = nxExt * nyExt * loc->nf * loc->nbatch;
   dataBuffer.resize(area);

   return dataBuffer;
}

void setLayerData(HyPerLayer *layer, std::vector<float> const &values) {
   auto *activityComponent = layer->getComponentByType<PV::ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "Target layer \"%s\" does not have an activity component.\n",
         activityComponent->getName());
   float *dataBuffer;
   int dataBufferSize = static_cast<int>(values.size());
   pvAssert(dataBufferSize == layer->getNumExtendedAllBatches());
   dataBuffer = findDataBufferA(activityComponent);
   for (int k = 0; k < dataBufferSize; ++k) {
      dataBuffer[k] = values[k];
   }
}

int testStoredValues(HyPerLayer *layer, float nnzThreshold, unsigned int seed) {
   int status = PV_SUCCESS;

   std::string nnzThresholdString(std::to_string(nnzThreshold));

   std::string description("testStoredValues(");
   description.append("nnzThreshold=").append(nnzThresholdString).append(")");

   std::string paramsString("debugParsing = false;\n");
   paramsString.append("StatsProbe \"probe\" = {\n");
   paramsString.append("   nnzThreshold = ").append(nnzThresholdString).append(";\n");
   paramsString.append("};\n");
   PV::PVParams probeParams(paramsString.c_str(), paramsString.size(), 1UL, MPI_COMM_WORLD);

   ActivityBufferStatsProbeLocal activityBufferProbeLocal("probe", &probeParams);
   activityBufferProbeLocal.ioParamsFillGroup(PV::PARAMS_IO_READ);
   activityBufferProbeLocal.initializeState(layer);
   FatalIf(
         activityBufferProbeLocal.getBufferType() != StatsBufferType::A,
         "ActivityBufferStatsProbe failed to set buffer type to activity.\n");

   PVLayerLoc const *loc         = layer->getLayerLoc();
   std::vector<float> dataBuffer = initDataBuffer(loc);

   int const storeSize   = 3;
   auto const bufferSize = static_cast<unsigned int>(dataBuffer.size());
   float threshold       = nnzThreshold ? 0.0f : 0.05f;
   ProbeDataBuffer<LayerStats> correctStatsStore;
   for (int N = 1; N <= storeSize; ++N) {
      unsigned int seedN = seed + static_cast<unsigned int>(N - 1) * bufferSize;
      generateRandomValues(dataBuffer, threshold, seedN);
      double timestamp = static_cast<double>(N);
      setLayerData(layer, dataBuffer);
      auto *publisherComponent = layer->getComponentByType<PV::BasePublisherComponent>();
      auto *publisher          = publisherComponent->getPublisher();
      publisher->publish(timestamp);
      activityBufferProbeLocal.storeValues(timestamp);
      correctStatsStore.store(calcCorrectValues(timestamp, dataBuffer, loc, nnzThreshold));
   }
   auto const &observedStatsStore = activityBufferProbeLocal.getStoredValues();

   int observedSize = static_cast<int>(observedStatsStore.size());
   int correctSize  = static_cast<int>(correctStatsStore.size());
   try {
      checkValue(description, "store length", observedSize, correctSize, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }

   if (status == PV_SUCCESS) {
      for (int N = 0; N < observedSize; ++N) {
         auto const &observedStatsBatch = observedStatsStore.getData(N);
         auto const &correctStatsBatch  = correctStatsStore.getData(N);
         std::string storeElementDesc(description);
         storeElementDesc.append(", N=").append(std::to_string(N));
         int statusN = compareStatsBatch(storeElementDesc, observedStatsBatch, correctStatsBatch);
         if (statusN != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
      activityBufferProbeLocal.clearStoredValues();
      auto const &clearedStore = activityBufferProbeLocal.getStoredValues();
      observedSize             = static_cast<int>(clearedStore.size());
      if (observedSize != 0) {
         ErrorLog().printf(
               "%s clearStoredValues() failed to clear the stored values\n", description.c_str());
         status = PV_FAILURE;
      }
      observedSize = static_cast<int>(activityBufferProbeLocal.getStoredValues().size());
      if (observedSize != 0) {
         ErrorLog().printf(
               "%s size() returns %d instead of 0 after clearStoredValues()\n",
               description.c_str(),
               observedSize);
         status = PV_FAILURE;
      }
   }

   return status;
}
