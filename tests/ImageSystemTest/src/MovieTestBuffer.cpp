#include "MovieTestBuffer.hpp"
#include <components/BatchIndexer.hpp>

namespace PV {

MovieTestBuffer::MovieTestBuffer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void MovieTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   FatalIf(
         mBatchMethod == BatchIndexer::RANDOM,
         "%s has BatchMethod = random. This test does not check that case.\n");
   ImageActivityBuffer::updateBufferCPU(simTime, deltaTime);
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int nbatch            = loc->nbatch;
   int numNeurons        = nx * ny * nf;
   int batchIndexerData[2 * nbatch];
   if (getMPIBlock()->getRank() == 0) {
      for (int b = 0; b < nbatch; b++) {
         batchIndexerData[b]          = mBatchIndexer->getStartIndex(b);
         batchIndexerData[b + nbatch] = mBatchIndexer->getSkipAmount(b);
      }
   }
   MPI_Bcast(batchIndexerData, 2 * nbatch, MPI_INT, 0, getMPIBlock()->getComm());
   for (int b = 0; b < nbatch; b++) {
      float const *dataBatch = getBufferData(b);
      int startIndex         = batchIndexerData[b];
      int skipAmount         = batchIndexerData[b + nbatch];
      int t                  = (int)std::nearbyint(simTime);
      int frameIdx           = (t - 1) * skipAmount + startIndex;
      if (mBatchMethod == BatchIndexer::BYFILE) {
         FatalIf(
               skipAmount != nbatch,
               "%s has BatchMethod = byFile, but SkipAmounts[%d] is %d instead of nbatch=%d\n",
               getDescription_c(),
               b,
               skipAmount,
               nbatch);
         FatalIf(
               startIndex != b,
               "%s has BatchMethod = byFile, but StartIndices[%d] is %d instead of b=%d\n",
               getDescription_c(),
               b,
               startIndex,
               b);
      }
      if (mBatchMethod == BatchIndexer::BYLIST) {
         FatalIf(
               skipAmount != 1,
               "%s has BatchMethod = byList, but SkipAmounts[%d] is %d instead of 1\n",
               getDescription_c(),
               b,
               skipAmount);
         FatalIf(
               startIndex != 2 * b,
               "%s has BatchMethod = byList, but StartIndices[%d] is %d instead of 2*b=%d\n",
               getDescription_c(),
               b,
               startIndex,
               2 * b);
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
