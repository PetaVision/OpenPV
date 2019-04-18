#include "ImagePvpTestLayer.hpp"

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

Response::Status ImagePvpTestLayer::registerData(Checkpointer *checkpointer) {
   auto status = PvpLayer::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   if (getMPIBlock()->getRank() == 0) {
      mNumFrames = countInputImages();
   }
   MPI_Bcast(&mNumFrames, 1, MPI_INT, 0, getMPIBlock()->getComm());
   return Response::SUCCESS;
}

Response::Status ImagePvpTestLayer::updateState(double time, double dt) {
   PvpLayer::updateState(time, dt);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   for (int b = 0; b < nbatch; b++) {
      int frameIdx     = (getStartIndex(b) + b) % mNumFrames;
      float *dataBatch = getActivity() + b * getNumExtended();
      for (int nkRes = 0; nkRes < getNumNeurons(); nkRes++) {
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
            Fatal() << "ImageFileIO " << name << " test Expected: " << expectedVal
                    << " Actual: " << checkVal << "\n";
         }
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
