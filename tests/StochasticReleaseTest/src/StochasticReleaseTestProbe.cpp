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

StochasticReleaseTestProbe::StochasticReleaseTestProbe(const char * name, HyPerCol * hc) {
   initialize_base();
   initStochasticReleaseTestProbe(name, hc);
}

StochasticReleaseTestProbe::StochasticReleaseTestProbe() {
   initialize_base();
}

int StochasticReleaseTestProbe::initialize_base() {
   return PV_SUCCESS;
}

int StochasticReleaseTestProbe::initStochasticReleaseTestProbe(const char * name, HyPerCol * hc) {
   pvAssert(hc->getInitialStep()==0L);
   int status = initStatsProbe(name, hc);
   return status;
}

void StochasticReleaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int StochasticReleaseTestProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   pvErrorIf(!(getTargetLayer()), ": %s did not set target layer.\n", getDescription_c());
   pvErrorIf(conn!=nullptr, ": %s, communicateInitInfo called with connection already set.\n", getDescription_c());
   int numconns = getParent()->numberOfConnections();
   for (int c=0; c<numconns; c++) {
      BaseConnection * baseConn = getParent()->getConnection(c);
      if (!strcmp(baseConn->getPostLayerName(),getTargetLayer()->getName())) {
         pvErrorIf(conn!=nullptr, ": %s cannot have more than one connnection going to target %s.\n", getDescription_c(), getTargetLayer()->getName());
         conn = dynamic_cast<HyPerConn *>(baseConn);
      }
   }
   pvErrorIf(!(conn!=nullptr), ": %s requires a connection going to target %s.\n", getDescription_c(), getTargetLayer()->getName());
   return status;
}

bool compar(double const& a, double const& b) {
   // routine for sorting p-values.
   // If the theoretical variance is zero and the observed mean is correct, the p-value will be not-a-number.
   // If the theoretical variance is zero and the observed mean is incorrect, the p-value will be plus or minus infinity
   // Sort so that all the nan's are at the end; they won't be included in the Holm-Bonferroni test.
   if (std::isnan(a)) return false;
   if (std::isnan(b)) return true;
   return a < b;
}

int StochasticReleaseTestProbe::outputState(double timed) {
   pvErrorIf(!(conn->numberOfAxonalArborLists()==1), ": %s connection %s has %d arbors; only one is allowed.\n", getDescription_c(), conn->getName(), conn->numberOfAxonalArborLists());
   pvErrorIf(!(conn->xPatchSize()==1), ": %s connection %s has nxp=%d, nxp=1 is required.\n", getDescription_c(), conn->getName(), conn->xPatchSize());
   pvErrorIf(!(conn->yPatchSize()==1), ": %s connection %s has nyp=%d, nyp=1 is required.\n", getDescription_c(), conn->getName(), conn->yPatchSize());
   pvErrorIf(!(conn->getNumDataPatches()==conn->fPatchSize()), ": %s connection %s must have number of data patches (%d) and nfp equal (%d).\n",
         getDescription_c(), conn->getName(), conn->getNumDataPatches(), conn->fPatchSize());
   int status = StatsProbe::outputState(timed);
   pvErrorIf(!(status==PV_SUCCESS), ": %s failed in StatsProbe::outputState at time %f.\n", getDescription_c(), timed);
   if (timed>0.0) {
      computePValues();
      if (getParent()->getCommunicator()->commRank()==0 && timed+0.5*getParent()->getDeltaTime() >= getParent()->getStopTime()) {
         // This is the last timestep
         // sort the p-values and apply Holm-Bonferroni method since there is one for each timestep and each feature.
         size_t N = pvalues.size();
         std::sort(pvalues.begin(), pvalues.end(), compar);
         while(N>0 && std::isnan(pvalues.at(N-1))) {
            N--;
         }
         pvalues.resize(N);
         for (size_t k=0; k<N; k++) {
            double hbCorr = pvalues.at(k)*(double)(N-k);
            if (hbCorr<0.05) {
               pvErrorNoExit().printf("%s: p-value %zu out of %zu (ordered by size) with Holm-Bonferroni correction = %f\n", getTargetLayer()->getDescription_c(), k, N, hbCorr);
               status = PV_FAILURE;
            }
         }
      }

   }
   pvErrorIf(status!=PV_SUCCESS, ": %s failed in StochasticReleaseTestProbe::outputState at time %f.\n", timed);
   return status;
}

void StochasticReleaseTestProbe::computePValues() {
   int nf = getTargetLayer()->getLayerLoc()->nf;
   auto oldsize = pvalues.size();
   pvalues.resize(oldsize+nf);
   for (int f=0; f < nf; f++) {
      int nf = getTargetLayer()->getLayerLoc()->nf;
      pvErrorIf(!(f >= 0 && f < nf), "Test failed.\n");
      pvwdata_t wgt = conn->get_wDataStart(0)[f*(nf+1)]; // weights should be one-to-one weights

      HyPerLayer * pre = conn->preSynapticLayer();
      const pvdata_t * preactPtr = pre->getLayerData();
      const PVLayerLoc * preLoc = pre->getLayerLoc();
      const int numPreNeurons = pre->getNumNeurons();
      bool found=false;
      pvdata_t preact = 0.0f;
      for (int n=f; n<numPreNeurons; n+=nf) {
         int nExt = kIndexExtended(n, preLoc->nx, preLoc->ny, preLoc->nf, preLoc->halo.lt, preLoc->halo.rt, preLoc->halo.dn, preLoc->halo.up);
         pvdata_t a = preactPtr[nExt];
         if (a!=0.0f) {
            if (found) {
               pvErrorIf(!(preact==a), "Test failed.\n");
            }
            else {
               found = true;
               preact = a;
            }
         }
      }
      if (preact < 0.0f) preact = 0.0f;
      if (preact > 1.0f) preact = 1.0f;

      const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
      const pvdata_t * activity = getTargetLayer()->getLayerData();
      int nnzf = 0;
      const int numNeurons = getTargetLayer()->getNumNeurons();
      for (int n=f; n<numNeurons; n+=nf) {
         int nExt = kIndexExtended(n, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         pvErrorIf(!(activity[nExt]==0 || activity[nExt]==wgt), "Test failed.\n");
         if (activity[nExt]!=0) nnzf++;
      }
      HyPerLayer * l = getTargetLayer();
      MPI_Allreduce(MPI_IN_PLACE, &nnzf, 1, MPI_INT, MPI_SUM, getParent()->getCommunicator()->communicator());
      if (getParent()->getCommunicator()->commRank()==0) {
         const int neuronsPerFeature = l->getNumGlobalNeurons()/nf;
         double mean = preact * neuronsPerFeature;
         double stddev = sqrt(neuronsPerFeature*preact*(1-preact));
         double numdevs = (nnzf-mean)/stddev;
         double pval = std::erfc(std::fabs(numdevs)/std::sqrt(2));
         pvalues.at(oldsize+f) = pval;
         outputStream->printf("    Feature %d, nnz=%5d, expectation=%7.1f, std.dev.=%5.1f, discrepancy of %f deviations, p-value %f\n",
               f, nnzf, mean, stddev, numdevs, pval);
      }
   }
}

StochasticReleaseTestProbe::~StochasticReleaseTestProbe() {
}

} /* namespace PV */
