/*
 * L2ConnProbe.cpp
 *
 *  Created on: July 24th, 2015
 *      Author: Kendall Stewart
 */

#include "L2ConnProbe.hpp"

namespace PV {

L2ConnProbe::L2ConnProbe() { initialize_base(); }

L2ConnProbe::L2ConnProbe(const char *probename, HyPerCol *hc) : KernelProbe(probename, hc) {}

L2ConnProbe::~L2ConnProbe() {}

int L2ConnProbe::initialize_base() { return PV_SUCCESS; }

int L2ConnProbe::outputState(double timed) {
   Communicator *icComm = parent->getCommunicator();
   const int rank       = icComm->commRank();
   if (rank != 0)
      return PV_SUCCESS;
   assert(getTargetConn() != NULL);
   int nxp       = getTargetHyPerConn()->xPatchSize();
   int nyp       = getTargetHyPerConn()->yPatchSize();
   int nfp       = getTargetHyPerConn()->fPatchSize();
   int patchSize = nxp * nyp * nfp;

   int arborID = getArbor();
   int numKern = getTargetHyPerConn()->getNumDataPatches();

   if (numKern != getTargetHyPerConn()->preSynapticLayer()->getLayerLoc()->nf) {
      pvError().printf(
            "L2ConnProbe %s: L2ConnProbe only works for 1-to-many or 1-to-1 weights.\n", name);
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
   for (int kernelIndex = 0; kernelIndex < numKern; ++kernelIndex) {
      const pvwdata_t *wdata =
            getTargetHyPerConn()->get_wDataStart(arborID) + patchSize * kernelIndex;

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
      output() << "t=" << timed << ", f=" << kernelIndex << ", squaredL2=" << sumsq << "\n";
   }

   return PV_SUCCESS;
}
};
