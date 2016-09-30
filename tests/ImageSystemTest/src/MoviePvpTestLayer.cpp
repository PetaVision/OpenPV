#include "MoviePvpTestLayer.hpp"
#include <components/BatchIndexer.hpp>

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

int MoviePvpTestLayer::updateState(double time, double dt) {
   PvpLayer::updateState(time, dt);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   int nbatchGlobal      = loc->nbatchGlobal;
   int commBatch         = parent->commBatch();
   int numBatchPerProc   = parent->numCommBatches();

   for (int b = 0; b < nbatch; b++) {
      pvdata_t *dataBatch = getActivity() + b * getNumExtended();
      int frameIdx        = 0;
      if (mBatchMethod == BatchIndexer::BYFILE || mBatchMethod == BatchIndexer::BYSPECIFIED) {
         frameIdx = (time - 1) * nbatchGlobal + commBatch * numBatchPerProc + b;
      } else if (mBatchMethod == BatchIndexer::BYLIST) {
         frameIdx = b * 2 + (time - 1);
      }
      for (int nkRes = 0; nkRes < getNumNeurons(); nkRes++) {
         // Calculate extended index
         int nkExt = kIndexExtended(
               nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         // checkVal is the value from batch index 0
         pvdata_t checkVal = dataBatch[nkExt];

         int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
         int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0;
         int kf       = featureIndex(nkRes, nx, ny, nf);

         pvdata_t expectedVal =
               kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf) + frameIdx * 192;
         if (fabs(checkVal - expectedVal) >= 1e-5) {
            pvErrorNoExit() << "ImageFileIO " << name << " test Expected: " << expectedVal
                            << " Actual: " << checkVal << "\n";
         }
      }
   }
   return PV_SUCCESS;
}

} // end namespace PV
