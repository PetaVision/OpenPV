/*
 * LIFTestProbe.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#include "LIFTestProbe.hpp"

#define LIFTESTPROBE_DEFAULTENDINGTIME 2000.0f
#define LIFTESTPROBE_DEFAULTTOLERANCE  3.0f
#define LIFTESTPROBE_BINS 5

namespace PV {
LIFTestProbe::LIFTestProbe(const char * probeName, HyPerCol * hc) : StatsProbe() {
   initialize_base();
   initLIFTestProbe(probeName, hc);
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

int LIFTestProbe::initLIFTestProbe(const char * probeName, HyPerCol * hc) {

   int status = initStatsProbe(probeName, hc);


   radii = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   rates = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   targetrates = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   stddevs = (double *) calloc(LIFTESTPROBE_BINS, sizeof(double));
   counts = (int *) calloc(LIFTESTPROBE_BINS, sizeof(int));
   if (radii == NULL || rates == NULL || targetrates == NULL) {
      fprintf(stderr, "LIFTestProbe::initLIFTestProbe error in probe \"%s\": unable to allocate memory for radii and rates.\n", getMessage());
      abort();
   }
   // Bin the LIFGap layer's activity into bins based on pixel position.  The pixels are assigned x- and y-coordinates in -31.5 to 31.5
   // and the distance r  to the origin of each pixel is calculated.  Bin 0 is 0 <= r < 10, bin 1 is 10 <= r < 15, and subsequent
   // bins are annuli of thickness 5.  The 0<=r<5 and 5<=r<10 are lumped together because the stimuli in these two annuli are very similar.
   // The simulation is run for 2 seconds (8000 timesteps with dt=0.25).  The average rate over each bin is calculated and compared with
   // the values in the r[] array.  It needs to be within 2.5 standard deviations (the s[] array) of the correct value.
   // The hard-coded values in r[] and s[] were determined empirically.
   double r[] = {25.058765822784814,24.429162500000004,23.701505474452546,22.788644662921353,21.571396713615037}; // Expected rates of each bin
   double s[] = {0.10532785056608626,0.09163171768337709,0.08387269359631463,0.05129454286195273,0.05482686550202272}; // Standard deviations of each bin at t=1000.
   // Note: s[] was determined by running the test 100 times for t=2000ms.
   int c[] = {316,400,548,712,852}; // Number of pixels that fall into each bin // TODO calculate on the fly based on layer size and bin boundaries

   // Bins are r<10, 10<=r<15, 15<=r<20, 20<=r<25, and 25<=r<30.
   for (int k=0; k<LIFTESTPROBE_BINS; k++) {
      radii[k] = k*5;
      targetrates[k] = r[k];
      stddevs[k] = s[k];
      counts[k] = c[k];
   }
   return status;
}

int LIFTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_endingTime(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

void LIFTestProbe::ioParam_endingTime(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "endingTime", &endingTime, LIFTESTPROBE_DEFAULTENDINGTIME);
}

void LIFTestProbe::ioParam_tolerance(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "tolerance", &tolerance, LIFTESTPROBE_DEFAULTTOLERANCE);
}

LIFTestProbe::~LIFTestProbe() {
   free(radii);
   free(rates);
   free(targetrates);
   free(stddevs);
   free(counts);
}

int LIFTestProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   return status;
}

int LIFTestProbe::allocateDataStructures() {
   int status = StatsProbe::allocateDataStructures();
   if (status == PV_SUCCESS && getParent()->columnId()==0) {
      assert(outputstream);
      fprintf(outputstream->fp, "%s Correct: ", getMessage());
      for (int k=0; k<LIFTESTPROBE_BINS; k++) {
         fprintf(outputstream->fp, " %f", targetrates[k]);
      }
      fprintf(outputstream->fp, "\n");
   }
   return status;
}

int LIFTestProbe::outputState(double timed) {
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
   if (icComm->commRank()==root_proc) {
      MPI_Reduce(MPI_IN_PLACE, rates, LIFTESTPROBE_BINS, MPI_DOUBLE, MPI_SUM, root_proc, icComm->communicator());
      fprintf(outputstream->fp, "%s t=%f:", getMessage(), timed);
      for (int j=0; j<LIFTESTPROBE_BINS; j++) {
         rates[j] /= counts[j]*timed/1000.0;
         fprintf(outputstream->fp, " %f", rates[j]);
      }
      fprintf(outputstream->fp, "\n");
      if (timed >= endingTime) {
         double stdfactor = sqrt(timed/2000.0); // Since the values of std are based on t=2000.
         for (int j=0; j<LIFTESTPROBE_BINS; j++) {
            double scaledstdev = stddevs[j]/stdfactor;
            double observed = (rates[j]-targetrates[j])/scaledstdev;
            if(fabs(observed)>tolerance) {
               fprintf(stderr, "Bin number %d failed at time %f: %f standard deviations off, with tolerance %f.\n", j, timed, observed, tolerance);
               status = PV_FAILURE;
            }
         }
      }
   }
   else {
      MPI_Reduce(rates, rates, LIFTESTPROBE_BINS, MPI_DOUBLE, MPI_SUM, root_proc, icComm->communicator());
      // Not using Allreduce, so the value of rates does not get updated in non-root processes.
   }
   if (status!=PV_SUCCESS) abort();
   return status;
}

BaseObject * createLIFTestProbe(char const * name, HyPerCol * hc) {
   return hc ? new LIFTestProbe(name, hc) : NULL;
}

} /* namespace PV */
