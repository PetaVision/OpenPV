/*
 * LIFTestProbe.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#include "LIFTestProbe.hpp"

#define LIFTESTPROBE_DEFAULTENDINGTIME 2000
#define LIFTESTPROBE_DEFAULTTOLERANCE  3.0
#define LIFTESTPROBE_BINS 5

namespace PV {
LIFTestProbe::LIFTestProbe(const char * filename, HyPerLayer * layer, const char * msg, const char * probename) : StatsProbe(filename, layer, msg) {
   initialize_base();
   initLIFTestProbe(filename, layer, BufActivity, msg, probename);
}

LIFTestProbe::LIFTestProbe(HyPerLayer * layer, const char * msg, const char * probename) : StatsProbe(layer, msg) {
   initialize_base();
   initLIFTestProbe(NULL, layer, BufActivity, msg, probename);

}

LIFTestProbe::LIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg, const char * probename) : StatsProbe(filename, layer, type, msg) {
   initialize_base();
   initLIFTestProbe(filename, layer, type, msg, probename);

}

LIFTestProbe::LIFTestProbe(HyPerLayer * layer, PVBufType type, const char * msg, const char * probename) : StatsProbe(layer, type, msg) {
   initialize_base();
   initLIFTestProbe(NULL, layer, type, msg, probename);

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

int LIFTestProbe::initLIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg, const char * probename) {

   int status = PV_SUCCESS;

   PVParams * params = layer->getParent()->parameters();
   endingTime = params->value(probename, "endingTime", LIFTESTPROBE_DEFAULTENDINGTIME);
   tolerance = params->value(probename, "tolerance", LIFTESTPROBE_DEFAULTTOLERANCE);

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
   // bins are annuli of thickness 5.  The 0<=r<5 and 5<=r<10 are lumped together because the stimuli in these two annuli are very similar.
   // The simulation is run for 2 seconds (8000 timesteps with dt=0.25).  The average rate over each bin is calculated and compared with
   // the values in the r[] array.  It needs to be within 2.5 standard deviations (the s[] array) of the correct value.
   // The hard-coded values in r[] and s[] were determined empirically.
   double r[] = {2.0964560,1.5421708,0.9808872,0.4943285,0.1828809}; // Expected rates of each bin
   double s[] = {0.1460102,0.0969807,0.0718997,0.0432749,0.0217653}; // Standard deviations of each bin at on t=1000.
   // Note: s[] was determined by running the test 100 times for t=100 seconds (1e5 ms) and then multiplying by 10 since sigma should vary as 1/sqrt(t).
   int c[] = {316,400,548,712,852}; // Number of pixels that fall into each bin // TODO calculate on the fly based on layer size and bin boundaries

   // Bins are r<10, 10<=r<15, 15<=r<20, 20<=r<25, and 25<=r<30.
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
   double xctr = 0.5*(loc->nxGlobal-1) - loc->kx0;
   double yctr = 0.5*(loc->nyGlobal-1) - loc->ky0;
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
#ifdef PV_USE_MPI
#endif // PV_USE_MPI
   if (icComm->commRank()==root_proc) {
      MPI_Reduce(MPI_IN_PLACE, rates, LIFTESTPROBE_BINS, MPI_DOUBLE, MPI_SUM, root_proc, icComm->communicator());
      fprintf(fp, "%s t=%f:", msg, timef);
      for (int j=0; j<LIFTESTPROBE_BINS; j++) {
         rates[j] /= counts[j]*timef/1000.0;
         fprintf(fp, " %f", rates[j]);
      }
      fprintf(fp, "\n");
      if (timef >= endingTime) {
         double stdfactor = sqrt(timef/1000.0); // Since the values of std are based on t=1000.
         for (int j=0; j<LIFTESTPROBE_BINS; j++) {
            double scaledstdev = stddevs[j]/stdfactor;
            double observed = (rates[j]-targetrates[j])/scaledstdev;
            if(fabs(observed)>tolerance) {
               fprintf(stderr, "Bin number %d failed at time %f: %f standard deviations off, with tolerance %f.\n", j, timef, observed, tolerance);
               abort();
            }
         }
      }
   }
   else {
      MPI_Reduce(rates, rates, LIFTESTPROBE_BINS, MPI_DOUBLE, MPI_SUM, root_proc, icComm->communicator());
      // Not using Allreduce, so the value of rates does not get updated in non-root processes.
   }

   return status;
}

} /* namespace PV */
