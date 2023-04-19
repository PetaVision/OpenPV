/*
 * LIFTestProbe.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#include "LIFTestProbe.hpp"
#include "arch/mpi/mpi.h"
#include "include/PVLayerLoc.h"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <cmath>
#include <memory>
#include <string>

namespace PV {
LIFTestProbe::LIFTestProbe(const char *name, PVParams *params, Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

LIFTestProbe::LIFTestProbe() : StatsProbeImmediate() {}

void LIFTestProbe::checkStats() {
   bool failed = false;

   HyPerLayer *l         = getTargetLayer();
   const PVLayerLoc *loc = l->getLayerLoc();
   int n                 = l->getNumNeurons();
   float xctr            = 0.5f * (float)(loc->nxGlobal - 1) - (float)loc->kx0;
   float yctr            = 0.5f * (float)(loc->nyGlobal - 1) - (float)loc->ky0;
   for (int j = 0; j < mNumBins; j++) {
      mRates[j] = 0.0;
   }
   for (int k = 0; k < n; k++) {
      int x          = kxPos(k, loc->nx, loc->ny, loc->nf);
      int y          = kyPos(k, loc->nx, loc->ny, loc->nf);
      float r        = std::sqrt((x - xctr) * (x - xctr) + (y - yctr) * (y - yctr));
      int bin_number = static_cast<int>(std::floor(r / 5.0f));
      bin_number -= bin_number > 0 ? 1 : 0;
      if (bin_number < mNumBins) {
         mRates[bin_number] += (double)l->getV()[k];
      }
   }
   int root_proc              = 0;
   Communicator const *icComm = mCommunicator;
   if (icComm->commRank() == root_proc) {
      MPI_Reduce(
            MPI_IN_PLACE,
            mRates.data(),
            mNumBins,
            MPI_DOUBLE,
            MPI_SUM,
            root_proc,
            icComm->communicator());

      auto const &storedValues           = mProbeAggregator->getStoredValues();
      auto numTimestamps                 = storedValues.size();
      int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
      ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);

      double simTime = stats.getTimestamp();
      std::string ratesString(" t=");
      ratesString.append(std::to_string(simTime));
      for (int j = 0; j < mNumBins; j++) {
         mRates[j] /= (double)mCounts[j] * simTime / 1000.0;
         ratesString.append(" ").append(std::to_string(mRates[j]));
      }
      mProbeOutputter->writeString(ratesString, 0);

      if (simTime >= mEndingTime) {
         double stdfactor = std::sqrt(simTime / 2000.0); // The values of std are based on t=2000.
         for (int j = 0; j < mNumBins; j++) {
            double scaledstdev = mStdDevs[j] / stdfactor;
            double observed    = (mRates[j] - mTargetRates[j]) / scaledstdev;
            if (std::fabs(observed) > mTolerance) {
               ErrorLog().printf(
                     "Bin number %d failed at time %f: %f standard deviations off, with tolerance "
                     "%f.\n",
                     j,
                     simTime,
                     observed,
                     mTolerance);
               failed = true;
            }
         }
      }
   }
   else {
      MPI_Reduce(
            mRates.data(),
            mRates.data(),
            mNumBins,
            MPI_DOUBLE,
            MPI_SUM,
            root_proc,
            icComm->communicator());
      // Not using Allreduce, so the value of Rates does not get updated in non-root processes.
   }
   FatalIf(failed, "%s failed.\n", getDescription_c());
}

void LIFTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {

   StatsProbeImmediate::initialize(name, params, comm);

   mRadii.resize(mNumBins);
   mRates.resize(mNumBins);
   mTargetRates.resize(mNumBins);
   mStdDevs.resize(mNumBins);
   mCounts.resize(mNumBins);

   // Bin the LIFGap layer's activity into bins based on pixel position. The pixels are assigned
   // x- and y-coordinates in -31.5 to 31.5 and the distance r  to the origin of each pixel is
   // calculated. Bin 0 is 0 <= r < 10, bin 1 is 10 <= r < 15, and subsequent bins are annuli of
   // thickness 5. The 0<=r<5 and 5<=r<10 are lumped together because the stimuli in these two
   // annuli are very similar. The simulation is run for 2 seconds (8000 timesteps with dt=0.25).
   // The average rate over each bin is calculated and compared with the values in the r[] array.
   // It needs to be within 2.5 standard deviations (the s vector) of the correct value.
   // The hard-coded values in r and s were determined empirically.
   std::vector<double> r{
         25.058765822784814,
         24.429162500000004,
         23.701505474452546,
         22.788644662921353,
         21.571396713615037}; // Expected rates of each bin
   std::vector<double> s{
         0.10532785056608626,
         0.09163171768337709,
         0.08387269359631463,
         0.05129454286195273,
         0.05482686550202272}; // Standard deviations of each bin at t=1000.
   // Note: the vector s was determined by running the test 100 times for t=2000ms.
   std::vector<int> c{316, 400, 548, 712, 852};
   // The vector c is the number of pixels that fall into each bin
   // TODO calculate on the fly based on layer size and bin boundaries

   // Bins are r<10, 10<=r<15, 15<=r<20, 20<=r<25, and 25<=r<30.
   for (int k = 0; k < mNumBins; k++) {
      mRadii[k]       = k * 5;
      mTargetRates[k] = r[k];
      mStdDevs[k]     = s[k];
      mCounts[k]      = c[k];
   }
}

int LIFTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_endingTime(ioFlag);
   ioParam_tolerance(ioFlag);
   return status;
}

void LIFTestProbe::ioParam_endingTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "endingTime", &mEndingTime, mEndingTime);
}

void LIFTestProbe::ioParam_tolerance(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "tolerance", &mTolerance, mTolerance);
}

LIFTestProbe::~LIFTestProbe() {}

Response::Status
LIFTestProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = StatsProbeImmediate::communicateInitInfo(message);
   FatalIf(
         getTargetLayer()->getLayerLoc()->nbatch != 1,
         "%s requires nbatch = 1.\n",
         getDescription_c());
   return status;
}

Response::Status
LIFTestProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = StatsProbeImmediate::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   std::string header(" Correct: ");
   for (int k = 0; k < mNumBins; ++k) {
      header.append(" ").append(std::to_string(mTargetRates[k]));
   }
   mProbeOutputter->writeString(header, 0);
   return Response::SUCCESS;
}

} /* namespace PV */
