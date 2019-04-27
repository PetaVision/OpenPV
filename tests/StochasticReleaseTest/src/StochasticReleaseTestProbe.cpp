/*
 * StochasticReleaseTestProbe.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#include "StochasticReleaseTestProbe.hpp"
#include "components/ArborList.hpp"
#include "components/ConnectionData.hpp"
#include "components/PatchSize.hpp"
#include "components/WeightsPair.hpp"
#include "layers/HyPerLayer.hpp"
#include <algorithm>
#include <cmath>

namespace PV {

StochasticReleaseTestProbe::StochasticReleaseTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

StochasticReleaseTestProbe::StochasticReleaseTestProbe() { initialize_base(); }

int StochasticReleaseTestProbe::initialize_base() { return PV_SUCCESS; }

void StochasticReleaseTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

void StochasticReleaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

Response::Status StochasticReleaseTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (conn == nullptr) {
      auto status = StatsProbe::communicateInitInfo(message);
      if (!Response::completed(status)) {
         return status;
      }
      FatalIf(!getTargetLayer(), ": %s did not set target layer.\n", getDescription_c());
      FatalIf(
            getTargetLayer()->getLayerLoc()->nbatch != 1,
            "%s can only be used with nbatch = 1.\n",
            getDescription_c());
      FatalIf(
            conn != nullptr,
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
                  conn != nullptr,
                  ": %s cannot have more than one connnection going to target %s.\n",
                  getDescription_c(),
                  getTargetLayer()->getName());
            conn = hyperconn;
         }
      }
      FatalIf(
            conn == nullptr,
            ": %s requires a connection going to target %s.\n",
            getDescription_c(),
            getTargetLayer()->getName());
   }
   if (!conn->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   auto *arborList = conn->getComponentByType<ArborList>();
   FatalIf(
         arborList->getNumAxonalArbors() != 1,
         ": %s connection %s has %d arbors; only one is allowed.\n",
         getDescription_c(),
         conn->getName(),
         arborList->getNumAxonalArbors());
   auto *patchSize = conn->getComponentByType<PatchSize>();
   FatalIf(
         patchSize->getPatchSizeX() != 1,
         ": %s connection %s has nxp=%d, nxp=1 is required.\n",
         getDescription_c(),
         conn->getName(),
         patchSize->getPatchSizeX());
   FatalIf(
         patchSize->getPatchSizeY() != 1,
         ": %s connection %s has nyp=%d, nyp=1 is required.\n",
         getDescription_c(),
         conn->getName(),
         patchSize->getPatchSizeY());
   auto *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair->getPreWeights()->getNumDataPatches() != patchSize->getPatchSizeF(),
         ": %s connection %s must have number of data patches (%d) and nfp equal (%d).\n",
         getDescription_c(),
         conn->getName(),
         weightsPair->getPreWeights()->getNumDataPatches(),
         patchSize->getPatchSizeF());
   return Response::SUCCESS;
}

bool compar(double const &a, double const &b) {
   // routine for sorting p-values.
   // If the theoretical variance is zero and the observed mean is correct, the p-value will be
   // not-a-number.
   // If the theoretical variance is zero and the observed mean is incorrect, the p-value will be
   // plus or minus infinity
   // Sort so that all the nan's are at the end; they won't be included in the Holm-Bonferroni test.
   if (std::isnan(a))
      return false;
   if (std::isnan(b))
      return true;
   return a < b;
}

Response::Status StochasticReleaseTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   FatalIf(
         status != Response::SUCCESS,
         ": %s failed in StatsProbe::outputState at time %f.\n",
         getDescription_c(),
         simTime);
   bool failed = false;
   if (simTime > 0.0) {
      computePValues();
      if (mCommunicator->commRank() == 0) {
         size_t N = pvalues.size();
         std::sort(pvalues.begin(), pvalues.end(), compar);
         while (N > 0 && std::isnan(pvalues.at(N - 1))) {
            N--;
         }
         pvalues.resize(N);
         for (std::size_t k = 0; k < N; k++) {
            double hbCorr = pvalues.at(k) * (double)(N - k);
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
         ": %s failed in StochasticReleaseTestProbe::outputState at time %f.\n",
         getTargetLayer()->getName(),
         simTime);
   return Response::SUCCESS;
}

void StochasticReleaseTestProbe::computePValues() {
   HyPerLayer *layer                 = getTargetLayer();
   BasePublisherComponent *publisher = layer->getComponentByType<BasePublisherComponent>();
   int nf                            = publisher->getLayerLoc()->nf;
   auto oldsize                      = pvalues.size();
   pvalues.resize(oldsize + nf);
   auto *preWeights = conn->getComponentByType<WeightsPair>()->getPreWeights();
   auto *preLayer   = conn->getComponentByType<ConnectionData>()->getPre();
   for (int f = 0; f < nf; f++) {
      int nf = publisher->getLayerLoc()->nf;
      FatalIf(!(f >= 0 && f < nf), "Test failed.\n");
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
               FatalIf(!(preact == a), "Test failed.\n");
            }
            else {
               found  = true;
               preact = a;
            }
         }
      }
      if (preact < 0.0f)
         preact = 0.0f;
      if (preact > 1.0f)
         preact = 1.0f;

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
         FatalIf(!(activity[nExt] == 0 || activity[nExt] == wgt), "Test failed.\n");
         if (activity[nExt] != 0)
            nnzf++;
      }
      MPI_Allreduce(MPI_IN_PLACE, &nnzf, 1, MPI_INT, MPI_SUM, mCommunicator->communicator());
      const int neuronsPerFeature = layer->getNumGlobalNeurons() / nf;
      double mean                 = preact * neuronsPerFeature;
      double stddev               = sqrt(neuronsPerFeature * preact * (1 - preact));
      double numdevs              = (nnzf - mean) / stddev;
      double pval                 = std::erfc(std::fabs(numdevs) / std::sqrt(2));
      pvalues.at(oldsize + f)     = pval;
      if (!mOutputStreams.empty()) {
         pvAssert(mOutputStreams.size() == (std::size_t)1);
         output(0).printf(
               "    Feature %d, nnz=%5d, expectation=%7.1f, std.dev.=%5.1f, discrepancy of %f "
               "deviations, p-value %f\n",
               f,
               nnzf,
               mean,
               stddev,
               numdevs,
               pval);
      }
   }
}

StochasticReleaseTestProbe::~StochasticReleaseTestProbe() {}

} /* namespace PV */
