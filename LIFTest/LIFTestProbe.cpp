/*
 * LIFTestProbe.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#include "LIFTestProbe.hpp"

#define LIFTESTPROBE_BINS 5

namespace PV {
LIFTestProbe::LIFTestProbe(const char * filename, HyPerLayer * layer, const char * msg) : StatsProbe(filename, layer, msg) {
   initialize_base();
   initLIFTestProbe(filename, layer, BufActivity, msg);
}

LIFTestProbe::LIFTestProbe(HyPerLayer * layer, const char * msg) : StatsProbe(layer, msg) {
   initialize_base();
   initLIFTestProbe(NULL, layer, BufActivity, msg);

}

LIFTestProbe::LIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg) : StatsProbe(filename, layer, type, msg) {
   initialize_base();
   initLIFTestProbe(filename, layer, type, msg);

}

LIFTestProbe::LIFTestProbe(HyPerLayer * layer, PVBufType type, const char * msg) : StatsProbe(layer, type, msg) {
   initialize_base();
   initLIFTestProbe(NULL, layer, type, msg);

}

LIFTestProbe::LIFTestProbe() : StatsProbe() {
   initialize_base();
}

int LIFTestProbe::initialize_base() {
   radii = NULL;
   rates = NULL;
   targetrates = NULL;
   stddevs = NULL;
   return PV_SUCCESS;
}

int LIFTestProbe::initLIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg) {
   int status = PV_SUCCESS;
   radii = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   rates = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   targetrates = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   stddevs = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   counts = (int *) calloc(LIFTESTPROBE_BINS, sizeof(int));
   if (radii == NULL || rates == NULL || targetrates == NULL) {
      fprintf(stderr, "LIFTestProbe::initLIFTestProbe error in probe \"%s\": unable to allocate memory for radii and rates.\n", msg);
      abort();
   }
   // Bin the LIFGap layer's activity into bins based on pixel position.  The pixels are assigned x- and y-coordinates in -31.5 to 31.5
   // and the distance r  to the origin of each pixel is calculated.  Bin 0 is 0 <= r < 10, bin 1 is 10 <= r < 15, and subsequent
   // bins are annuli of thickness 5.  The 0<=r<5 and 5<=r<10 are lumped together because the rates
   // The simulation is run for 2 seconds (8000 timesteps with dt=0.25).  The average rate over each bin is calculated and compared with
   // the values in the r[] array.  It needs to be within 2.5 standard deviations (the s[] array) of the correct value.
   // The hard-coded values in r[] and s[] were determined empirically.
   double r[] = {2.0993117,1.5412729,0.9843403,0.4933890,0.1832700}; // Expected rates of each bin
   double s[] = {0.1035321,0.0717574,0.0494596,0.0285325,0.0134950}; // Standard deviations of each bin
   int c[] = {316,400,548,712,852}; // Number of pixels that fall into each bin
   for (int k=0; k<LIFTESTPROBE_BINS; k++) {
      radii[k] = k*5;
      targetrates[k] = r[k];
      stddevs[k] = s[k];
      counts[k] = c[k];
   }
   if (layer->getParent()->icCommunicator()->commRank()==0) {
      fprintf(fp, "%s Correct: ", msg);
      for (int k=0; k<LIFTESTPROBE_BINS; k++) {
         fprintf(fp, " %f", targetrates[k]);
      }
      fprintf(fp, "\n");
   }
   return status;
}

LIFTestProbe::~LIFTestProbe() {
   free(radii);
   free(rates);
   free(targetrates);
   free(stddevs);
   free(counts);
}

int LIFTestProbe::outputState(float timef) {
   int status = PV_SUCCESS;

   HyPerLayer * l = getTargetLayer();
   const PVLayerLoc * loc = l->getLayerLoc();
   int n = l->getNumNeurons();
   double xctr = 0.5*(loc->nx-1);
   double yctr = 0.5*(loc->ny-1);
   for (int j=0; j<LIFTESTPROBE_BINS; j++) {
      rates[j] = 0;
   }
   for (int k=0; k<n; k++) {
      int x = kxPos(k, loc->nx, loc->ny, loc->nf);
      int y = kyPos(k, loc->nx, loc->ny, loc->nf);
      double r = sqrt((x-xctr)*(x-xctr) + (y-yctr)*(y-yctr));
      int bin_number = (int) floor(r/5.0);
      bin_number -= bin_number > 0 ? 1 : 0;
      if (bin_number < LIFTESTPROBE_BINS) {
         rates[bin_number] += l->getV()[k];
      }
   }
   int root_proc = 0;
   InterColComm * icComm = l->getParent()->icCommunicator();
   double avgrates[LIFTESTPROBE_BINS];
#ifdef PV_USE_MPI
   MPI_Reduce(rates, avgrates, LIFTESTPROBE_BINS, MPI_DOUBLE, MPI_SUM, root_proc, icComm->communicator());
   // TODO: get working under MPI
#endif // PV_USE_MPI
   if (icComm->commRank()==root_proc) {
      fprintf(fp, "%s t=%f:", msg, timef);
      for (int j=0; j<LIFTESTPROBE_BINS; j++) {
         rates[j] = avgrates[j];
         rates[j] /= counts[j]*timef/1000.0;
         fprintf(fp, " %f", rates[j]);
      }
      fprintf(fp, "\n");
      if (timef >= 2000) {
         for (int j=0; j<LIFTESTPROBE_BINS; j++) {
            assert(fabs((rates[j]-targetrates[j])/stddevs[j])<2.5);
         }
      }
   }

   return status;
}

} /* namespace PV */
