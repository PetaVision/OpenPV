/*
 * StochasticReleaseTestProbe.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#include "StochasticReleaseTestProbe.hpp"

namespace PV {

StochasticReleaseTestProbe::StochasticReleaseTestProbe(const char * name, HyPerCol * hc) {
   initialize_base();
   initStochasticReleaseTestProbe(name, hc);
}

StochasticReleaseTestProbe::StochasticReleaseTestProbe() {
   initialize_base();
}

int StochasticReleaseTestProbe::initialize_base() {
   conn = NULL;
   pvalues = NULL;

   return PV_SUCCESS;
}

int StochasticReleaseTestProbe::initStochasticReleaseTestProbe(const char * name, HyPerCol * hc) {
   int status = initStatsProbe(name, hc);
   return status;
}

void StochasticReleaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int StochasticReleaseTestProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   assert(getTargetLayer());
   long int num_steps = getParent()->getFinalStep() - getParent()->getInitialStep();
   pvalues = (double *) calloc(num_steps*getTargetLayer()->getLayerLoc()->nf, sizeof(double));
   if (pvalues == NULL) {
      fprintf(stderr, "StochasticReleaseTestProbe error: unable to allocate memory for pvalues: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   return status;
}

int compar(const void * a, const void * b) {
   // routine for sorting p-values.
   // If the theoretical variance is zero and the observed mean is correct, the p-value will be not-a-number.
   // If the theoretical variance is zero and the observed mean is incorrect, the p-value will by plus or minus infinity
   // Sort so that all the nan's are at the end; they won't be included in the Holm-Bonferroni test.
   double aval=*(double *) a;
   double bval=*(double *) b;
   if (isnan(aval)) return 1;
   if (isnan(bval)) return -1;
   return aval < bval ? -1 : aval > bval ? 1 : 0;
}

int StochasticReleaseTestProbe::outputState(double timed) {
   // Set conn.  Can't do that in initStochasticReleaseTestProbe because we need to search for a conn with the given post, and connections' postLayerName is not necessarily set.
   if (conn==NULL) {
      HyPerCol * hc = getTargetLayer()->getParent();
      int numconns = hc->numberOfConnections();
      for (int c=0; c<numconns; c++) {
         if (!strcmp(hc->getConnection(c)->getPostLayerName(),getTargetLayer()->getName())) {
            assert(conn==NULL); // Only one connection can go to this layer for this probe to work
            BaseConnection * baseConn = hc->getConnection(c);
            conn = dynamic_cast<HyPerConn *>(baseConn);
         }
      }
      assert(conn!=NULL);
   }
   assert(conn->numberOfAxonalArborLists()==1);
   assert(conn->xPatchSize()==1);
   assert(conn->yPatchSize()==1);
   assert(conn->getNumDataPatches()==conn->fPatchSize());
   int status = StatsProbe::outputState(timed);
   assert(status==PV_SUCCESS);
   HyPerLayer * l = getTargetLayer();
   HyPerCol * hc = l->getParent();
   int nf = l->getLayerLoc()->nf;
   if (timed>0.0) {
      for (int f=0; f < nf; f++) {
         if (computePValues(hc->getCurrentStep(), f)!=PV_SUCCESS) status = PV_FAILURE;
      }
      assert(status == PV_SUCCESS);
      if (hc->columnId()==0 && hc->simulationTime()+hc->getDeltaTime()/2>=hc->getStopTime()) {
         // This is the last timestep
         // sort the p-values and apply Holm-Bonferroni method since there is one for each timestep and each feature.
         long int num_steps = hc->getFinalStep() - hc->getInitialStep();
         long int N = num_steps * nf;
         qsort(pvalues, (size_t) N, sizeof(*pvalues), compar);
         while(N>0 && isnan(pvalues[N-1])) {
            N--;
         }
         for (long int k=0; k<N; k++) {
            if (pvalues[k]*(N-k)<0.05) {
               fprintf(stderr, "layer \"%s\" FAILED: p-value %ld out of %ld (ordered by size) with Holm-Bonferroni correction = %f\n", getTargetLayer()->getName(), k, N, pvalues[k]*(N-k));
               status = PV_FAILURE;
            }
         }
      }

   }
   assert(status==PV_SUCCESS);
   return status;
}

int StochasticReleaseTestProbe::computePValues(long int step, int f) {
   int status = PV_SUCCESS;
   assert(step >=0 && step < INT_MAX);
   int nf = getTargetLayer()->getLayerLoc()->nf;
   assert(f >= 0 && f < nf);
   int idx = (step-1)*nf + f;
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
            assert(preact==a);
         }
         else {
            found = true;
            preact = a;
         }
      }
   }
   preact *= getParent()->getDeltaTime();
   if (preact < 0.0f) preact = 0.0f;
   if (preact > 1.0f) preact = 1.0f;

   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   const pvdata_t * activity = getTargetLayer()->getLayerData();
   int nnzf = 0;
   const int numNeurons = getTargetLayer()->getNumNeurons();
   for (int n=f; n<numNeurons; n+=nf) {
      int nExt = kIndexExtended(n, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      assert(activity[nExt]==0 || activity[nExt]==wgt);
      if (activity[nExt]!=0) nnzf++;
   }
   HyPerLayer * l = getTargetLayer();
   HyPerCol * hc = l->getParent();
   MPI_Allreduce(MPI_IN_PLACE, &nnzf, 1, MPI_INT, MPI_SUM, hc->icCommunicator()->communicator());
   if (hc->columnId()==0) {
      const int neuronsPerFeature = l->getNumGlobalNeurons()/nf;
      double mean = preact * neuronsPerFeature;
      double stddev = sqrt(neuronsPerFeature*preact*(1-preact));
      double numdevs = (nnzf-mean)/stddev;
      pvalues[idx] = erfc(fabs(numdevs)/sqrt(2));
      fprintf(outputstream->fp, "    Feature %d, nnz=%5d, expectation=%7.1f, std.dev.=%5.1f, discrepancy of %f deviations, p-value %f\n",
              f, nnzf, mean, stddev, numdevs, pvalues[idx]);
   }
   assert(status==PV_SUCCESS);
   return status;
}

StochasticReleaseTestProbe::~StochasticReleaseTestProbe() {
   free(pvalues);
}

BaseObject * createStochasticReleaseTestProbe(char const * name, HyPerCol * hc) {
   return hc ? new StochasticReleaseTestProbe(name, hc) : NULL;
}

} /* namespace PV */
