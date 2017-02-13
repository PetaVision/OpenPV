/*
 * CheckpointEntryPvp.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryPvp class.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cstring>
#include <vector>

namespace PV {

// TODO: many commonalities between CheckpointEntryPvp and CheckpointEntryDataStore.
// Refactor to eliminate code duplication

template <typename T>
void CheckpointEntryPvp<T>::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   int const numFrames = getMPIBlock()->getBatchDimension() * mLayerLoc->nbatch;
   int const nxBlock   = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock   = mLayerLoc->ny * getMPIBlock()->getNumRows();

   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      fileStream       = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      BufferUtils::ActivityHeader header =
            BufferUtils::buildActivityHeader<T>(nxBlock, nyBlock, mLayerLoc->nf, numFrames);
      BufferUtils::writeActivityHeader(*fileStream, header);
   }
   PVHalo const &halo   = mLayerLoc->halo;
   int const nxExtLocal = mLayerLoc->nx + (mExtended ? halo.lt + halo.rt : 0);
   int const nyExtLocal = mLayerLoc->ny + (mExtended ? halo.dn + halo.up : 0);
   int const nf         = mLayerLoc->nf;

   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % mLayerLoc->nbatch;
      int const mpiBatchIndex   = frame / mLayerLoc->nbatch; // Integer division

      T *localData = calcBatchElementStart(localBatchIndex);
      Buffer<T> pvpBuffer{localData, nxExtLocal, nyExtLocal, nf};
      pvpBuffer.crop(mLayerLoc->nx, mLayerLoc->ny, Buffer<T>::CENTER);

      // All ranks with BatchIndex==mpiBatchIndex must call gather; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from gather() immediately.
      Buffer<T> globalPvpBuffer = BufferUtils::gather(
            getMPIBlock(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny, mpiBatchIndex, 0);

      if (getMPIBlock()->getRank() == 0) {
         pvAssert(fileStream);
         pvAssert(globalPvpBuffer.getWidth() == nxBlock);
         pvAssert(globalPvpBuffer.getHeight() == nyBlock);
         BufferUtils::writeFrame(*fileStream, &globalPvpBuffer, simTime);
      }
   }
   delete fileStream;
}

template <typename T>
void CheckpointEntryPvp<T>::read(std::string const &checkpointDirectory, double *simTimePtr) const {
   int const numFrames = getMPIBlock()->getBatchDimension() * mLayerLoc->nbatch;
   int const nxBlock   = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock   = mLayerLoc->ny * getMPIBlock()->getNumRows();

   PVHalo const &halo    = mLayerLoc->halo;
   int const nxExtLocal  = mLayerLoc->nx + (mExtended ? halo.lt + halo.rt : 0);
   int const nyExtLocal  = mLayerLoc->ny + (mExtended ? halo.dn + halo.up : 0);
   int const nxExtGlobal = nxBlock + (mExtended ? halo.lt + halo.rt : 0);
   int const nyExtGlobal = nyBlock + (mExtended ? halo.dn + halo.up : 0);

   std::string path = generatePath(checkpointDirectory, "pvp");
   Buffer<T> pvpBuffer;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % mLayerLoc->nbatch;
      int const mpiBatchIndex   = frame / mLayerLoc->nbatch; // Integer division

      if (getMPIBlock()->getRank() == 0) {
         *simTimePtr = BufferUtils::readActivityFromPvp(path.c_str(), &pvpBuffer, frame);
         pvpBuffer.grow(nxExtGlobal, nyExtGlobal, Buffer<float>::CENTER);
      }
      else if (mpiBatchIndex == getMPIBlock()->getBatchIndex()) {
         pvpBuffer.resize(nxExtLocal, nyExtLocal, mLayerLoc->nf);
      }
      // All ranks with BatchIndex==mpiBatchIndex must call scatter; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from scatter() immediately.
      BufferUtils::scatter(
            getMPIBlock(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny, mpiBatchIndex, 0);
      if (mpiBatchIndex == getMPIBlock()->getBatchIndex()) {
         std::vector<T> bufferData = pvpBuffer.asVector();
         T *localData              = calcBatchElementStart(localBatchIndex);
         std::memcpy(
               localData, bufferData.data(), (std::size_t)pvpBuffer.getTotalElements() * sizeof(T));
      }
   }
   MPI_Bcast(simTimePtr, 1, MPI_DOUBLE, 0, getMPIBlock()->getComm());
}

template <typename T>
T *CheckpointEntryPvp<T>::calcBatchElementStart(int batchElement) const {
   int nx = mLayerLoc->nx;
   int ny = mLayerLoc->ny;
   if (mExtended) {
      nx += mLayerLoc->halo.lt + mLayerLoc->halo.rt;
      ny += mLayerLoc->halo.dn + mLayerLoc->halo.up;
   }
   return &mDataPointer[batchElement * nx * ny * mLayerLoc->nf];
}

template <typename T>
void CheckpointEntryPvp<T>::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}
} // end namespace PV
