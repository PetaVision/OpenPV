#include "MoviePvpTestBuffer.hpp"
#include <components/BatchIndexer.hpp>

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
   int numNeurons        = nx * ny * nf;

   for (int b = 0; b < nbatch; b++) {
      float *dataBatch = mBufferData.data() + b * getBufferSize();
      int frameIdx     = 0;
      if (mBatchMethod == BatchIndexer::BYFILE || mBatchMethod == BatchIndexer::BYSPECIFIED) {
         frameIdx = (simTime - 1) * nbatchGlobal + commBatch * numBatchPerProc + b;
      }
      else if (mBatchMethod == BatchIndexer::BYLIST) {
         frameIdx = b * 2 + (simTime - 1);
      }
      for (int nkRes = 0; nkRes < numNeurons; nkRes++) {
         // Calculate extended index
         int nkExt = kIndexExtended(
               nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         // checkVal is the value from batch index 0
         float checkVal = dataBatch[nkExt];

         int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
         int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0;
         int kf       = featureIndex(nkRes, nx, ny, nf);

         float expectedVal =
               kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf) + frameIdx * 192;
         if (std::fabs(checkVal - expectedVal) >= 1e-5f) {
            ErrorLog() << "ImageFileIO " << name << " test Expected: " << expectedVal
                       << " Actual: " << checkVal << "\n";
         }
      }
   }
}

} // end namespace PV
