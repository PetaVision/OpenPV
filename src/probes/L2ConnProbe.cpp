/*
 * L2ConnProbe.cpp
 *
 *  Created on: July 24th, 2015
 *      Author: Kendall Stewart
 */

#include "L2ConnProbe.hpp"

namespace PV {

L2ConnProbe::L2ConnProbe() {}

L2ConnProbe::L2ConnProbe(const char *probename, PVParams *params, Communicator *comm)
      : KernelProbe(probename, params, comm) {}

L2ConnProbe::~L2ConnProbe() {}

Response::Status L2ConnProbe::outputState(double simTime, double deltaTime) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   pvAssert(getTargetConn() != nullptr);
   Communicator *icComm = mCommunicator;
   const int rank       = icComm->commRank();
   int nxp              = getPatchSize()->getPatchSizeX();
   int nyp              = getPatchSize()->getPatchSizeY();
   int nfp              = getPatchSize()->getPatchSizeF();
   int patchSize        = nxp * nyp * nfp;

   int arborID = getArbor();
   int numKern = getWeights()->getNumDataPatches();

   if (numKern != getWeights()->getGeometry()->getPreLoc().nf) {
      Fatal().printf(
            "L2ConnProbe %s: L2ConnProbe only works for 1-to-many or 1-to-1 weights.\n", name);
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
   for (int kernelIndex = 0; kernelIndex < numKern; ++kernelIndex) {
      const float *wdata = getWeightData() + patchSize * kernelIndex;

      float sumsq = 0;

      for (int f = 0; f < nfp; f++) {

         for (int y = 0; y < nyp; y++) {

            for (int x = 0; x < nxp; x++) {

               int k   = kIndex(x, y, f, nxp, nyp, nfp);
               float w = (float)wdata[k];
               sumsq += w * w;
            }
         }
      }
      output(0) << "t=" << simTime << ", f=" << kernelIndex << ", squaredL2=" << sumsq << "\n";
   }

   return Response::SUCCESS;
}

} // end namespace PV
