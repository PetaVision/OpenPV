#include "MoviePvpTestBuffer.hpp"
#include <structures/BatchIndexer.hpp>

namespace PV {

MoviePvpTestBuffer::MoviePvpTestBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void MoviePvpTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PvpActivityBuffer::updateBufferCPU(simTime, deltaTime);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   int nbatchGlobal      = loc->nbatchGlobal;
   int commBatch         = mCommunicator->commBatch();
   int numBatchPerProc   = mCommunicator->numCommBatches();
   long numNeurons       = (long)nx * (long)ny * (long)nf;

   for (int b = 0; b < nbatch; b++) {
      int timestep = static_cast<int>(std::nearbyint(simTime));
      float *dataBatch = mBufferData.data() + b * getBufferSize();
      int frameIdx     = 0;
      if (mBatchMethod == BYFILE || mBatchMethod == BYSPECIFIED) {
         frameIdx = (timestep - 1) * nbatchGlobal + commBatch * numBatchPerProc + b;
      }
      else if (mBatchMethod == BYLIST) {
         frameIdx = b * 2 + (timestep - 1);
      }
      for (long nkRes = 0; nkRes < numNeurons; nkRes++) {
         // Calculate extended index
         long nkExt = kIndexExtended(
               nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         // checkVal is the value from batch index 0
         float checkVal = dataBatch[nkExt];

         int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
         int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0;
         int kf       = featureIndex(nkRes, nx, ny, nf);

         long globalIndex = kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf);
         float expectedVal = (float)(globalIndex + frameIdx * 192);
         if (std::fabs(checkVal - expectedVal) >= 1e-5f) {
            ErrorLog() << "ImageFileIO " << getName() << " test Expected: " << expectedVal
                       << " Actual: " << checkVal << "\n";
         }
      }
   }
}

} // end namespace PV
