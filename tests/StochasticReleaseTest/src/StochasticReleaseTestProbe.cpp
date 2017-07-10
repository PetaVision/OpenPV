/*
 * StochasticReleaseTestProbe.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#include "StochasticReleaseTestProbe.hpp"
#include <algorithm>
#include <cmath>

namespace PV {

StochasticReleaseTestProbe::StochasticReleaseTestProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

StochasticReleaseTestProbe::StochasticReleaseTestProbe() { initialize_base(); }

int StochasticReleaseTestProbe::initialize_base() { return PV_SUCCESS; }

int StochasticReleaseTestProbe::initialize(const char *name, HyPerCol *hc) {
   int status = StatsProbe::initialize(name, hc);
   pvAssert(parent->getInitialStep() == 0L);
   return status;
}

void StochasticReleaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int StochasticReleaseTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = StatsProbe::communicateInitInfo(message);
   FatalIf(!getTargetLayer(), ": %s did not set target layer.\n", getDescription_c());
   FatalIf(
         getTargetLayer()->getLayerLoc()->nbatch != 1,
         "%s can only be used with nbatch = 1.\n",
         getDescription_c());
   FatalIf(
         conn != nullptr,
         ": %s, communicateInitInfo called with connection already set.\n",
         getDescription_c());
   for (auto &obj : message->mHierarchy) {
      HyPerConn *hyperconn = dynamic_cast<HyPerConn *>(obj.second);
      if (hyperconn == nullptr) {
         continue;
      }
      if (!strcmp(hyperconn->getPostLayerName(), getTargetLayer()->getName())) {
         FatalIf(
               conn != nullptr,
               ": %s cannot have more than one connnection going to target %s.\n",
               getDescription_c(),
               getTargetLayer()->getName());
         conn = hyperconn;
      }
   }
   FatalIf(
         !(conn != nullptr),
         ": %s requires a connection going to target %s.\n",
         getDescription_c(),
         getTargetLayer()->getName());
   return status;
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

int StochasticReleaseTestProbe::outputState(double timed) {
   FatalIf(
         !(conn->numberOfAxonalArborLists() == 1),
         ": %s connection %s has %d arbors; only one is allowed.\n",
         getDescription_c(),
         conn->getName(),
         conn->numberOfAxonalArborLists());
   FatalIf(
         !(conn->xPatchSize() == 1),
         ": %s connection %s has nxp=%d, nxp=1 is required.\n",
         getDescription_c(),
         conn->getName(),
         conn->xPatchSize());
   FatalIf(
         !(conn->yPatchSize() == 1),
         ": %s connection %s has nyp=%d, nyp=1 is required.\n",
         getDescription_c(),
         conn->getName(),
         conn->yPatchSize());
   FatalIf(
         !(conn->getNumDataPatches() == conn->fPatchSize()),
         ": %s connection %s must have number of data patches (%d) and nfp equal (%d).\n",
         getDescription_c(),
         conn->getName(),
         conn->getNumDataPatches(),
         conn->fPatchSize());
   int status = StatsProbe::outputState(timed);
   FatalIf(
         !(status == PV_SUCCESS),
         ": %s failed in StatsProbe::outputState at time %f.\n",
         getDescription_c(),
         timed);
   if (timed > 0.0) {
      computePValues();
      if (parent->getCommunicator()->commRank() == 0
          && timed + 0.5 * parent->getDeltaTime() >= parent->getStopTime()) {
         // This is the last timestep
         // sort the p-values and apply Holm-Bonferroni method since there is one for each timestep
         // and each feature.
         size_t N = pvalues.size();
         std::sort(pvalues.begin(), pvalues.end(), compar);
         while (N > 0 && std::isnan(pvalues.at(N - 1))) {
            N--;
         }
         pvalues.resize(N);
         for (size_t k = 0; k < N; k++) {
            double hbCorr = pvalues.at(k) * (double)(N - k);
            if (hbCorr < 0.05) {
               ErrorLog().printf(
                     "%s: p-value %zu out of %zu (ordered by size) with Holm-Bonferroni correction "
                     "= %f\n",
                     getTargetLayer()->getDescription_c(),
                     k,
                     N,
                     hbCorr);
               status = PV_FAILURE;
            }
         }
      }
   }
   FatalIf(
         status != PV_SUCCESS,
         ": %s failed in StochasticReleaseTestProbe::outputState at time %f.\n",
         timed);
   return status;
}

void StochasticReleaseTestProbe::computePValues() {
   int nf       = getTargetLayer()->getLayerLoc()->nf;
   auto oldsize = pvalues.size();
   pvalues.resize(oldsize + nf);
   for (int f = 0; f < nf; f++) {
      int nf = getTargetLayer()->getLayerLoc()->nf;
      FatalIf(!(f >= 0 && f < nf), "Test failed.\n");
      float wgt = conn->get_wDataStart(0)[f * (nf + 1)]; // weights should be one-to-one weights

      HyPerLayer *pre          = conn->preSynapticLayer();
      const float *preactPtr   = pre->getLayerData();
      const PVLayerLoc *preLoc = pre->getLayerLoc();
      const int numPreNeurons  = pre->getNumNeurons();
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

      const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
      const float *activity = getTargetLayer()->getLayerData();
      int nnzf              = 0;
      const int numNeurons  = getTargetLayer()->getNumNeurons();
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
      HyPerLayer *l = getTargetLayer();
      MPI_Allreduce(
            MPI_IN_PLACE, &nnzf, 1, MPI_INT, MPI_SUM, parent->getCommunicator()->communicator());
      if (!mOutputStreams.empty()) {
         pvAssert(mOutputStreams.size() == (std::size_t)1);
         const int neuronsPerFeature = l->getNumGlobalNeurons() / nf;
         double mean                 = preact * neuronsPerFeature;
         double stddev               = sqrt(neuronsPerFeature * preact * (1 - preact));
         double numdevs              = (nnzf - mean) / stddev;
         double pval                 = std::erfc(std::fabs(numdevs) / std::sqrt(2));
         pvalues.at(oldsize + f)     = pval;
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
