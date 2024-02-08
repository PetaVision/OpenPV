/*
 * StochasticReleaseTestProbe.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#include "StochasticReleaseTestProbe.hpp"
#include "StochasticReleaseTestProbeOutputter.hpp"
#include "arch/mpi/mpi.h"
#include "components/ArborList.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/ConnectionData.hpp"
#include "components/PatchSize.hpp"
#include "components/Weights.hpp"
#include "components/WeightsPair.hpp"
#include "include/PVLayerLoc.hpp"
#include "include/pv_common.h"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Observer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ActivityBufferStatsProbeLocal.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace PV {

StochasticReleaseTestProbe::StochasticReleaseTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

StochasticReleaseTestProbe::StochasticReleaseTestProbe() {}

void StochasticReleaseTestProbe::checkStats() {
   bool failed                        = false;
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   double simTime                     = stats.getTimestamp();
   if (simTime > 0.0) {
      int pValuesStatus = computePValues();
      if (pValuesStatus) {
         failed = true;
      }
      if (mCommunicator->commRank() == 0) {
         std::sort(m_pValues.begin(), m_pValues.end());
         size_t N = m_pValues.size();
         for (std::size_t k = 0; k < N; k++) {
            double hbCorr = m_pValues.at(k) * (double)(N - k);
            if (hbCorr < 0.05) {
               ErrorLog().printf(
                     "%s: p-value %zu out of %zu (ordered by size) with Holm-Bonferroni correction "
                     "= %f\n",
                     getTargetLayer()->getDescription_c(),
                     k,
                     N,
                     hbCorr);
               failed = true;
            }
         }
      }
   }
   FatalIf(
         failed,
         ": %s failed in StochasticReleaseTestProbe::checkStats at time %f.\n",
         getTargetLayer()->getName(),
         simTime);
}

void StochasticReleaseTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void StochasticReleaseTestProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<StochasticReleaseTestProbeOutputter>(name, params, comm);
}

Response::Status StochasticReleaseTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mConn == nullptr) {
      auto status = StatsProbeImmediate::communicateInitInfo(message);
      if (!Response::completed(status)) {
         return status;
      }
      FatalIf(!getTargetLayer(), ": %s did not set target layer.\n", getDescription_c());
      FatalIf(
            getTargetLayer()->getLayerLoc()->nbatch != 1,
            "%s can only be used with nbatch = 1.\n",
            getDescription_c());
      FatalIf(
            mConn != nullptr,
            ": %s, communicateInitInfo called with connection already set.\n",
            getDescription_c());
      for (auto &obj : *message->mObjectTable) {
         ComponentBasedObject *hyperconn = dynamic_cast<ComponentBasedObject *>(obj);
         if (hyperconn == nullptr) {
            continue;
         }
         auto *connectionData = hyperconn->getComponentByType<ConnectionData>();
         if (connectionData == nullptr) {
            continue;
         }
         if (!strcmp(connectionData->getPostLayerName(), getTargetLayer()->getName())) {
            FatalIf(
                  mConn != nullptr,
                  ": %s cannot have more than one connnection going to target %s.\n",
                  getDescription_c(),
                  getTargetLayer()->getName());
            mConn = hyperconn;
         }
      }
      FatalIf(
            mConn == nullptr,
            ": %s requires a connection going to target %s.\n",
            getDescription_c(),
            getTargetLayer()->getName());
   }
   if (!mConn->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   auto *arborList = mConn->getComponentByType<ArborList>();
   FatalIf(
         arborList->getNumAxonalArbors() != 1,
         ": %s connection %s has %d arbors; only one is allowed.\n",
         getDescription_c(),
         mConn->getName(),
         arborList->getNumAxonalArbors());
   auto *patchSize = mConn->getComponentByType<PatchSize>();
   FatalIf(
         patchSize->getPatchSizeX() != 1,
         ": %s connection %s has nxp=%d, nxp=1 is required.\n",
         getDescription_c(),
         mConn->getName(),
         patchSize->getPatchSizeX());
   FatalIf(
         patchSize->getPatchSizeY() != 1,
         ": %s connection %s has nyp=%d, nyp=1 is required.\n",
         getDescription_c(),
         mConn->getName(),
         patchSize->getPatchSizeY());
   auto *weightsPair = mConn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair->getPreWeights()->getNumDataPatches() != patchSize->getPatchSizeF(),
         ": %s connection %s must have number of data patches (%d) and nfp equal (%d).\n",
         getDescription_c(),
         mConn->getName(),
         weightsPair->getPreWeights()->getNumDataPatches(),
         patchSize->getPatchSizeF());
   return Response::SUCCESS;
}

int StochasticReleaseTestProbe::computePValues() {
   int status                        = PV_SUCCESS;
   HyPerLayer *layer                 = getTargetLayer();
   BasePublisherComponent *publisher = layer->getComponentByType<BasePublisherComponent>();
   int nf                            = publisher->getLayerLoc()->nf;
   auto *preWeights                  = mConn->getComponentByType<WeightsPair>()->getPreWeights();
   auto *preLayer                    = mConn->getComponentByType<ConnectionData>()->getPre();
   for (int f = 0; f < nf; f++) {
      float wgt = preWeights->getData(0)[f * (nf + 1)]; // weights should be one-to-one weights

      auto *prePublisher       = preLayer->getComponentByType<BasePublisherComponent>();
      const float *preactPtr   = prePublisher->getLayerData();
      const PVLayerLoc *preLoc = prePublisher->getLayerLoc();
      const int numPreNeurons  = preLayer->getNumNeurons();
      bool found               = false;
      float preact             = 0.0f;
      for (int n = f; n < numPreNeurons; n += nf) {
         int nExt = kIndexExtended(
               n,
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               preLoc->halo.lt,
               preLoc->halo.rt,
               preLoc->halo.dn,
               preLoc->halo.up);
         float a = preactPtr[nExt];
         if (a != 0.0f) {
            if (found) {
               FatalIf(preact != a, "Input activity layer has more than one nonzero value.\n");
            }
            else {
               found  = true;
               preact = a;
            }
         }
      }

      const PVLayerLoc *loc = publisher->getLayerLoc();
      const float *activity = publisher->getLayerData();
      int nnzf              = 0;
      const int numNeurons  = layer->getNumNeurons();
      for (int n = f; n < numNeurons; n += nf) {
         int nExt = kIndexExtended(
               n,
               loc->nx,
               loc->ny,
               loc->nf,
               loc->halo.lt,
               loc->halo.rt,
               loc->halo.dn,
               loc->halo.up);
         FatalIf(
               activity[nExt] != 0 and activity[nExt] != wgt,
               "An activity value was neither zero nor weight equal to %f\n",
               (double)wgt);
         nnzf += (activity[nExt] != 0) ? 1 : 0;
      }

      MPI_Allreduce(MPI_IN_PLACE, &nnzf, 1, MPI_INT, MPI_SUM, mCommunicator->communicator());
      const int neuronsPerFeature = layer->getNumGlobalNeurons() / nf;
      if (preact <= 0.0f) {
         if (nnzf != 0) {
            ErrorLog().printf("nnzf is %d for f = %d; expected 0\n", nnzf, f);
            status = PV_FAILURE;
         }
         continue;
      }
      if (preact >= 1.0f) {
         if (nnzf != neuronsPerFeature) {
            ErrorLog().printf("nnzf is %d for f = %d; expected 0\n", nnzf, f);
            status = PV_FAILURE;
         }
         continue;
      }

      double mean   = preact * neuronsPerFeature;
      double stddev = std::sqrt(static_cast<float>(neuronsPerFeature) * preact * (1.0f - preact));
      pvAssert(stddev > 0.0);
      double numdevs = (nnzf - mean) / stddev;
      double pval    = std::erfc(std::fabs(numdevs) / std::sqrt(2.0));
      m_pValues.push_back(pval);
      auto *outputter = dynamic_cast<StochasticReleaseTestProbeOutputter *>(mProbeOutputter.get());
      pvAssert(outputter != nullptr);
      outputter->printNumNonzeroData(f, nnzf, mean, stddev, numdevs, pval);
   }
   return status;
}

void StochasticReleaseTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

StochasticReleaseTestProbe::~StochasticReleaseTestProbe() {}

} /* namespace PV */
