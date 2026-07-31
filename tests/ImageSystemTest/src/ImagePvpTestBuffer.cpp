#include "ImagePvpTestBuffer.hpp"

namespace PV {

ImagePvpTestBuffer::ImagePvpTestBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status
ImagePvpTestBuffer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = PvpActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      mNumFrames = countInputImages();
   }
   MPI_Bcast(&mNumFrames, 1, MPI_INT, 0, getCommunicator()->ioCommunicator());
   return Response::SUCCESS;
}

void ImagePvpTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PvpActivityBuffer::updateBufferCPU(simTime, deltaTime);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   long const numNeurons = (long)nx * (long)ny * (long)nf;
   for (int b = 0; b < nbatch; b++) {
      int frameIdx           = (mStartFrameIndex[b] + b) % mNumFrames;
      float const *dataBatch = getBufferData(b);
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
            Fatal() << "ImageFileIO " << getName() << " test Expected: " << expectedVal
                    << " Actual: " << checkVal << "\n";
         }
      }
   }
}

} // end namespace PV
