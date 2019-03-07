#include "SumPoolTestInputBuffer.hpp"

namespace PV {

SumPoolTestInputBuffer::SumPoolTestInputBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

// Makes a layer such that the restricted space is the index, but with spinning order be [x, y, f]
// as opposed to [f, x, y]
void SumPoolTestInputBuffer::updateBufferCPU(double simTime, double deltaTime) {
   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nxGlobal          = loc->nxGlobal;
   int nyGlobal          = loc->nyGlobal;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;

   for (int b = 0; b < loc->nbatch; b++) {
      float *A = mBufferData.data() + b * getBufferSize();
      // looping over ext
      for (int iY = 0; iY < ny + loc->halo.up + loc->halo.dn; iY++) {
         for (int iX = 0; iX < nx + loc->halo.lt + loc->halo.rt; iX++) {
            // Calculate x and y global extended
            int xGlobalExt = iX + loc->kx0;
            int yGlobalExt = iY + loc->ky0;
            // Calculate x and y in restricted space
            int xGlobalRes = xGlobalExt - loc->halo.lt;
            int yGlobalRes = yGlobalExt - loc->halo.up;
            // Calculate base value
            // xGlobal and yGlobalRes can be negative
            int baseActivityVal = yGlobalRes * nxGlobal + xGlobalRes;

            for (int iFeature = 0; iFeature < nf; iFeature++) {
               int ext_idx = kIndex(
                     iX,
                     iY,
                     iFeature,
                     nx + loc->halo.lt + loc->halo.rt,
                     ny + loc->halo.dn + loc->halo.up,
                     nf);
               // Feature gives an offset, since it spins slowest
               int activityVal = baseActivityVal + iFeature * nxGlobal * nyGlobal;
               A[ext_idx]      = activityVal;
            }
         }
      }
   }
}

} /* namespace PV */
