#include "MovieTestBuffer.hpp"
#include <components/BatchIndexer.hpp>

namespace PV {

MovieTestBuffer::MovieTestBuffer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void MovieTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   ImageActivityBuffer::updateBufferCPU(simTime, deltaTime);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   int numNeurons        = nx * ny * nf;
   for (int b = 0; b < nbatch; b++) {
      float const *dataBatch = getBufferData(b);
      int frameIdx;
      if (mBatchMethod == BatchIndexer::BYFILE) {
         frameIdx = (simTime - 1) * nbatch + b;
      }
      else if (mBatchMethod == BatchIndexer::BYLIST) {
         frameIdx = b * 2 + (simTime - 1);
      }

      for (int nkRes = 0; nkRes < numNeurons; nkRes++) {
         // Calculate extended index
         int nkExt = kIndexExtended(
               nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         // checkVal is the value from batch index 0
         float checkVal = dataBatch[nkExt] * 255;

         int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
         int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0;
         int kf       = featureIndex(nkRes, nx, ny, nf);

         float expectedVal =
               kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf) + 10 * frameIdx;
         if (std::fabs(checkVal - expectedVal) >= 1e-4f) {
            Fatal() << name << " time: " << simTime << " batch: " << b
                    << " Expected: " << expectedVal << " Actual: " << checkVal << "\n";
         }
      }
   }
}

} // end namespace PV
