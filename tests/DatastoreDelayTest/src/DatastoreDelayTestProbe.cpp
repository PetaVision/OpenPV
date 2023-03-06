/*
 * DatastoreDelayTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "DatastoreDelayTestProbe.hpp"
#include "include/pv_common.h"
#include "layers/HyPerLayer.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "probes/VMembraneBufferStatsProbeLocal.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVLog.hpp"
#include <algorithm>

namespace PV {

DatastoreDelayTestProbe::DatastoreDelayTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void DatastoreDelayTestProbe::checkStats() {
   if (mCommunicator->commRank() != 0) {
      return;
   }
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);

   int numDelayLevels    = mInputPublisher->getNumDelayLevels();
   double simTime        = stats.getTimestamp();
   int status            = PV_SUCCESS;
   int nbatch            = static_cast<int>(stats.size());
   HyPerLayer *l         = getTargetLayer();
   double correctValue   = std::min(std::max(simTime - 1.0, 0.0), (double)numDelayLevels);
   auto ioMPIBlock       = getCommunicator()->getIOMPIBlock();
   int globalBatchOffset = nbatch * (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());
   for (int b = 0; b < nbatch; b++) {
      LayerStats const &statsElem = stats.getValue(b);
      int globalBatchIndex        = b + globalBatchOffset;
      if (statsElem.mMax > (float)correctValue) {
         ErrorLog().printf(
               "%s: time %f: batch element %d has maximum %f but "
               "all neurons should have value %d\n",
               l->getDescription_c(),
               simTime,
               globalBatchIndex,
               (double)statsElem.mMax,
               (int)correctValue);
         status = PV_FAILURE;
      }
      if (statsElem.mMin < (float)correctValue) {
         ErrorLog().printf(
               "%s: time %f: batch element %d has minimum %f but "
               "all neurons should have value get above %d\n",
               l->getDescription_c(),
               simTime,
               globalBatchIndex,
               (double)statsElem.mMin,
               (int)correctValue);
         status = PV_FAILURE;
      }
   }

#ifdef KOCHAB
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
#else
   if (status != PV_SUCCESS) {
      ErrorLog().printf("%s: t = %f. Test failed.\n", getDescription_c(), simTime);
   }
#endif // KOCHAB
}

void DatastoreDelayTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<VMembraneBufferStatsProbeLocal>(name, params);
}

void DatastoreDelayTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

Response::Status DatastoreDelayTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = StatsProbeImmediate::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mInputPublisher = message->mObjectTable->findObject<BasePublisherComponent>("input");
   FatalIf(
         mInputPublisher == nullptr,
         "%s could not find an object named \"input\" with a publisher component.\n",
         getDescription_c());

   return Response::SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {}

} // namespace PV
