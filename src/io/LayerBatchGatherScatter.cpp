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
      for (int k = 0; k < numElements; ++k) {
         rootDataLocation[k] = gatheredBuffer.at(k);
      }
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      Buffer<float> localBuffer(localDataLocation, nxExt, nyExt, nf);
      BufferUtils::gather(
            mMPIBlock, localBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
   }
}

void LayerBatchGatherScatter::scatter(
      int mpiBatchIndex, float const *rootDataLocation, float *localDataLocation) {

   Buffer<float> dataBuffer;
   if (mMPIBlock->getRank() == mRootProcessRank) {
      int nxRoot = mLayerLoc.nx * mMPIBlock->getNumColumns();
      int nyRoot = mLayerLoc.ny * mMPIBlock->getNumRows();
      if (mRootExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.dn + mLayerLoc.halo.up;
      }
      dataBuffer = Buffer<float>(rootDataLocation, nxRoot, nyRoot, mLayerLoc.nf);
      if (!mRootExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.dn + mLayerLoc.halo.up;
         dataBuffer.grow(nxRoot, nyRoot, Buffer<float>::CENTER);
      }
      BufferUtils::scatter(
            mMPIBlock, dataBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      int nf = mLayerLoc.nf;
      int nx = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      int ny = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;
      dataBuffer = Buffer<float>(nx, ny, nf);
      BufferUtils::scatter(
            mMPIBlock, dataBuffer, mLayerLoc.nx, mLayerLoc.ny, mpiBatchIndex, mRootProcessRank);
   }
   if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      copyToDataLocation(localDataLocation, dataBuffer);
   }
}

// localDataBuffer is the extended iff RootExtended is true
// dataLocation is always extended
void LayerBatchGatherScatter::copyToDataLocation(
      float *dataLocation, Buffer<float> const &localDataBuffer) {
   int dataLocationWidth  = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
   int dataLocationHeight = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;

   int numFeatures = mLayerLoc.nf;

   pvAssert(localDataBuffer.getWidth() == dataLocationWidth);
   pvAssert(localDataBuffer.getHeight() == dataLocationHeight);
   pvAssert(localDataBuffer.getFeatures() == numFeatures);

   int const nk = dataLocationWidth * dataLocationHeight * numFeatures;
   for (int k = 0; k < nk; ++k) {
      dataLocation[k] = localDataBuffer.at(k);
   }
}

} // namespace PV
