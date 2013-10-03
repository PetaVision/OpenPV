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
   for (int k=0; k<9; k++) {
      bins[k] = 0;
   }
   sumbins = 0;
   binprobs[0] = 0.00023262907903552502; // erfc(3.5/sqrt(2))/2
   binprobs[1] = 0.005977036246740614; // (erfc(2.5/sqrt(2))-erfc(3.5/sqrt(2)))/2
   binprobs[2] = 0.06059753594308195; // (erfc(1.5/sqrt(2))-erfc(2.5/sqrt(2)))/2
   binprobs[3] = 0.24173033745712885; // (erfc(0.5/sqrt(2))-erfc(1.5/sqrt(2)))/2
   binprobs[4] = 0.3829249225480262; // erf(0.5/sqrt(2)
   binprobs[5] = 0.24173033745712885; // (erfc(0.5/sqrt(2))-erfc(1.5/sqrt(2)))/2
   binprobs[6] = 0.06059753594308195; // (erfc(1.5/sqrt(2))-erfc(2.5/sqrt(2)))/2
   binprobs[7] = 0.005977036246740614; // (erfc(2.5/sqrt(2))-erfc(3.5/sqrt(2)))/2
   binprobs[8] = 0.00023262907903552502; // erfc(3.5/sqrt(2))/2
   return PV_SUCCESS;
}

int StochasticReleaseTestProbe::initStochasticReleaseTestProbe(const char * name, HyPerCol * hc) {
   const char * classkeyword = hc->parameters()->groupKeywordFromName(name);
   HyPerLayer * targetlayer = NULL;
   char * message = NULL;
   const char * filename;
   getLayerFunctionProbeParameters(name, classkeyword, hc, &targetlayer, &message, &filename);
   int status = initStatsProbe(filename, targetlayer, BufActivity, message);
   free(message); message = NULL; // getLayerFunctionProbeParameters uses strdup; initStatsProbe copies message, so we're done with it.

   return PV_SUCCESS;
}

int StochasticReleaseTestProbe::outputState(double timed) {
   // Set conn.  Can't do that in initStochasticReleaseTestProbe because we need to search for a conn with the given post, and connections' postLayerName is not necessarily set.
   if (conn==NULL) {
      HyPerCol * hc = getTargetLayer()->getParent();
      PVParams * params = hc->parameters();
      int numconns = hc->numberOfConnections();
      for (int c=0; c<numconns; c++) {
         if (!strcmp(hc->getConnection(c)->getPostLayerName(),getTargetLayer()->getName())) {
            assert(conn==NULL); // Only one connection can go to this layer for this probe to work
            conn = hc->getConnection(c);
         }
      }
      assert(conn!=NULL);
   }
   int status = StatsProbe::outputState(timed);
   assert(status==PV_SUCCESS);

   assert(conn->numberOfAxonalArborLists()==1);
   assert(conn->getNumDataPatches()==1);
   assert(conn->xPatchSize()==1);
   assert(conn->yPatchSize()==1);
   assert(conn->fPatchSize()==1);
   pvdata_t wgt = *conn->get_wDataStart(0);
   HyPerLayer * pre = conn->preSynapticLayer();
   const pvdata_t * preactPtr = pre->getLayerData();
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const int numPreNeurons = pre->getNumNeurons();
   bool found=false;
   pvdata_t preact = 0.0f;
   for (int n=0; n<numPreNeurons; n++) {
      int nExt = kIndexExtended(n, preLoc->nx, preLoc->ny, preLoc->nf, preLoc->nb);
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
   if (preact < 0.0f) preact = 0.0f;
   if (preact > 1.0f) preact = 1.0f;

   const int numNeurons = getTargetLayer()->getNumNeurons();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   const pvdata_t * activity = getTargetLayer()->getLayerData();
   for (int n=0; n<numNeurons; n++) {
      int nExt = kIndexExtended(n, loc->nx, loc->ny, loc->nf, loc->nb);
      assert(activity[nExt]==0 || activity[nExt]==wgt);
   }
   const int numGlobalNeurons = getTargetLayer()->getNumGlobalNeurons();
   double mean = preact * numGlobalNeurons;
   double stddev = sqrt(numGlobalNeurons*preact*(1-preact));
   double numStdDevs = stddev==0.0 && mean==nnz ? 0.0 : (nnz-mean)/stddev;
   HyPerCol * hc = getTargetLayer()->getParent();
   if (timed>0.0 && hc->columnId()==0) {
      fprintf(outputstream->fp, "    t=%f, number of standard deviations = %f\n", timed, numStdDevs);
      int bin = numStdDevs < -3.5 ? 0 :
                numStdDevs < -2.5 ? 1 :
                numStdDevs < -1.5 ? 2 :
                numStdDevs < -0.5 ? 3 :
                numStdDevs <= 0.5 ? 4 :
                numStdDevs <= 1.5 ? 5 :
                numStdDevs <= 2.5 ? 6 :
                numStdDevs <= 3.5 ? 7 : 8;
      bins[bin]++;
      sumbins++;
      if (hc->simulationTime()+hc->getDeltaTime()>=hc->getStopTime()) {
         fprintf(outputstream->fp, "    Histogram:  ");
         for (int k=0; k<9; k++) {
            fprintf(outputstream->fp, " %7d", bins[k]);
         }
         fprintf(outputstream->fp, "\n");

         int minallowed[9];
         int maxallowed[9];

         if (stddev==0) {
            for (int k=0; k<9; k++) {
               minallowed[k] = (k==4 ? sumbins : 0);
               maxallowed[k] = (k==4 ? sumbins : 0);
               assert(bins[k]==(k==4 ? sumbins : 0));
            }
         }
         else {
            assert(preact<1.0f && preact>0.0f);
            for (int k=0; k<9; k++) {
               // find first m for which prob(bins[k]<m) >= 0.005
               double p = binprobs[k];
               double outcomeprob = pow(1-p,sumbins);
               double cumulativeprob = outcomeprob;
               double m=0;
               printf("m=%10.4f, outcomeprob=%.20f, cumulativeprob=%.20f\n", m, outcomeprob, cumulativeprob);
               while(cumulativeprob < 0.005 && m <= sumbins) {
                  m++;
                  outcomeprob *= (sumbins+1-m)/m*p/(1-p);
                  cumulativeprob += outcomeprob;
                  printf("m=%10.4f, outcomeprob=%.20f, cumulativeprob=%.20f\n", m, outcomeprob, cumulativeprob);
               }
               minallowed[k] = m;
               if (bins[k]<minallowed[k]) status = PV_FAILURE;

               // find first m for which prob(bins[k]<m) < 0.995
               while(cumulativeprob <= 0.995 && sumbins) {
                  m++;
                  outcomeprob *= (sumbins+1-m)/m*p/(1-p);
                  cumulativeprob += outcomeprob;
                  printf("m=%10.4f, outcomeprob=%.20f, cumulativeprob=%.20f\n", m, outcomeprob, cumulativeprob);
               }
               maxallowed[k] = m;
               if (bins[k]>maxallowed[k]) status = PV_FAILURE;
            }
            fprintf(outputstream->fp, "    Min allowed:");
            for (int k=0; k<9; k++) {
               fprintf(outputstream->fp, " %7d", minallowed[k]);
            }
            fprintf(outputstream->fp, "\n");
            fprintf(outputstream->fp, "    Max allowed:");
            for (int k=0; k<9; k++) {
               fprintf(outputstream->fp, " %7d", maxallowed[k]);
            }
            fprintf(outputstream->fp, "\n");
            assert(status==PV_SUCCESS);
         }
      }
   }
   return status;
}

StochasticReleaseTestProbe::~StochasticReleaseTestProbe() {
}

} /* namespace PV */
