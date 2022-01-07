#include "LayerBatchGatherScatter.hpp"

#include "utils/BufferUtilsMPI.hpp" // gather, scatter

namespace PV {

LayerBatchGatherScatter::LayerBatchGatherScatter(
      std::shared_ptr<MPIBlock const> mpiBlock,
      PVLayerLoc const &layerLoc,
      int rootProcessRank,
      bool localExtended,
      bool rootExtended) {

   mMPIBlock = mpiBlock;
   mLayerLoc = layerLoc;
   mRootProcessRank = rootProcessRank;
   if (!localExtended) {
      mLayerLoc.halo.lt = 0;
      mLayerLoc.halo.rt = 0;
      mLayerLoc.halo.dn = 0;
      mLayerLoc.halo.up = 0;
   }
   mRootExtended = rootExtended;
}

void LayerBatchGatherScatter::gather(
      int mpiBatchIndex, float *rootDataLocation, float const *localDataLocation) {
   int nxExt = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
   int nyExt = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;
   int nf = mLayerLoc.nf;
   if (mMPIBlock->getRank() == mRootProcessRank) {
      Buffer<float> localBuffer(nxExt, nyExt, nf);
      if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
         localBuffer.set(localDataLocation, nxExt, nyExt, nf);
      }
      auto gatheredBuffer = BufferUtils::gather(
            mMPIBlock, localBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
      // gatheredBuffer is extended. If mRootExtended is false, need to crop
      if (!mRootExtended) {
         int rootWidth  = mLayerLoc.nx * mMPIBlock->getNumColumns();
         int rootHeight = mLayerLoc.ny * mMPIBlock->getNumRows();
         gatheredBuffer = gatheredBuffer.extract(
            mLayerLoc.halo.lt, mLayerLoc.halo.up, rootWidth, rootHeight);
      }
      int numElements = gatheredBuffer.getTotalElements();
      for (int k = 0; k < gatheredBuffer.getTotalElements(); ++k) {
         rootDataLocation[k] = gatheredBuffer.at(k);
      }
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      int nxExt = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      int nyExt = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;
      int nf = mLayerLoc.nf;
      Buffer<float> localBuffer(localDataLocation, nxExt, nyExt, nf);
      BufferUtils::gather(
            mMPIBlock, localBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
   }
}

void LayerBatchGatherScatter::scatter(
      int mpiBatchIndex, float const *rootDataLocation, float *localDataLocation) {

   if (mMPIBlock->getRank() == mRootProcessRank) {
      int nfRoot = mLayerLoc.nf;
      int nxRoot = mLayerLoc.nx * mMPIBlock->getNumColumns();
      int nyRoot = mLayerLoc.ny * mMPIBlock->getNumRows();
      if (mRootExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.dn + mLayerLoc.halo.up;
      }
      Buffer<float> rootBuffer(rootDataLocation, nxRoot, nyRoot, nfRoot);
      BufferUtils::scatter(
            mMPIBlock, rootBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
      if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
         copyToDataLocation(localDataLocation, rootBuffer);
      }
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      int nf = mLayerLoc.nf;
      int nx = mLayerLoc.nx;
      int ny = mLayerLoc.ny;
      if (mRootExtended) {
         nx += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         ny += mLayerLoc.halo.dn + mLayerLoc.halo.up;
      }
      Buffer<float> localDataBuffer(nx, ny, nf);
      int batchIndex = mMPIBlock->getBatchIndex();
      BufferUtils::scatter(
            mMPIBlock, localDataBuffer, mLayerLoc.nx, mLayerLoc.ny, batchIndex, mRootProcessRank);
      copyToDataLocation(localDataLocation, localDataBuffer);
   }
}

// localDataBuffer is the extended iff RootExtended is true
// dataLocation is always extended
void LayerBatchGatherScatter::copyToDataLocation(
      float *dataLocation, Buffer<float> const &localDataBuffer) {
   int dataLocationWidth  = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
   int dataLocationHeight = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;
   int dataLocationStartX = mRootExtended ? 0 : mLayerLoc.halo.lt;
   int dataLocationStartY = mRootExtended ? 0 : mLayerLoc.halo.up;

   int bufferWidth  = mRootExtended ? dataLocationWidth : mLayerLoc.nx;
   int bufferHeight = mRootExtended ? dataLocationHeight : mLayerLoc.ny;

   int numFeatures = mLayerLoc.nf;

   pvAssert(localDataBuffer.getWidth() == bufferWidth);
   pvAssert(localDataBuffer.getHeight() == bufferHeight);
   pvAssert(localDataBuffer.getFeatures() == numFeatures);
   // Probably better to use Buffer::at(int,int,int) and not need bufferWidth etc.

   for (int y = 0; y < bufferHeight; ++y) {
      for (int x = 0; x < bufferWidth; ++x) {
         for (int f = 0; f < numFeatures; ++f) {
            int dataLocationOffset = kIndex(
                  x + dataLocationStartX, y + dataLocationStartY, f,
                  dataLocationWidth, dataLocationHeight, numFeatures);
            int bufferOffset = kIndex(x, y, f, bufferWidth, bufferHeight, numFeatures);
            dataLocation[dataLocationOffset] = localDataBuffer.at(bufferOffset);
         }
      }
   }
}

} // namespace PV
