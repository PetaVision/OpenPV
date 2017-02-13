/*
 * CheckpointEntryPvpBuffer.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryPvpBuffer class.
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

// TODO: many commonalities between CheckpointEntryPvpBuffer and CheckpointEntryDataStore.
// Refactor to eliminate code duplication

template <typename T>
CheckpointEntryPvpBuffer<T>::CheckpointEntryPvpBuffer(
      std::string const &name,
      MPIBlock const *mpiBlock,
      T *dataPtr,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(name, mpiBlock) {
   initialize(dataPtr, layerLoc, extended);
}

template <typename T>
CheckpointEntryPvpBuffer<T>::CheckpointEntryPvpBuffer(
      std::string const &objName,
      std::string const &dataName,
      MPIBlock const *mpiBlock,
      T *dataPtr,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(objName, dataName, mpiBlock) {
   initialize(dataPtr, layerLoc, extended);
}

template <typename T>
void CheckpointEntryPvpBuffer<T>::initialize(T *dataPtr, PVLayerLoc const *layerLoc, bool extended) {
   mDataPointer = dataPtr;
   mLayerLoc    = layerLoc;
   if (extended) {
      mXMargins = layerLoc->halo.lt + layerLoc->halo.rt;
      mYMargins = layerLoc->halo.dn + layerLoc->halo.up;
   }
   else {
      mXMargins = 0;
      mYMargins = 0;
   }
}

template <typename T>
void CheckpointEntryPvpBuffer<T>::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   int const numFrames = getNumFrames();
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
   int const nxExtLocal = mLayerLoc->nx + mXMargins;
   int const nyExtLocal = mLayerLoc->ny + mYMargins;
   int const nf         = mLayerLoc->nf;

   for (int frame = 0; frame < numFrames; frame++) {
      T const *localData = calcBatchElementStart(frame);

      Buffer<T> pvpBuffer{localData, nxExtLocal, nyExtLocal, nf};
      pvpBuffer.crop(mLayerLoc->nx, mLayerLoc->ny, Buffer<T>::CENTER);

      // All ranks with BatchIndex==mpiBatchIndex must call gather; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from gather() immediately.
      int const mpiBatchIndex   = calcMPIBatchIndex(frame);
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
void CheckpointEntryPvpBuffer<T>::read(std::string const &checkpointDirectory, double *simTimePtr) const {
   int const numFrames = getNumFrames();
   int const nxBlock   = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock   = mLayerLoc->ny * getMPIBlock()->getNumRows();

   int const nxExtLocal  = mLayerLoc->nx + mXMargins;
   int const nyExtLocal  = mLayerLoc->ny + mYMargins;
   int const nxExtGlobal = nxBlock + mXMargins;
   int const nyExtGlobal = nyBlock + mYMargins;

   std::string path;
   if (getMPIBlock()->getRank() == 0) {
      path = generatePath(checkpointDirectory, "pvp");
      FileStream fileStream(path.c_str(), std::ios_base::in, false);
      struct BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(fileStream);
      FatalIf(
            header.nBands != numFrames,
            "CheckpointEntryDataStore::read error reading \"%s\": delays*batchwidth in file is %d, "
            "but delays*batchwidth in layer is %d\n",
            path.c_str(),
            header.nBands,
            numFrames);
   }
   Buffer<T> pvpBuffer;
   std::vector<double> frameTimestamps;
   frameTimestamps.resize(numFrames);
   for (int frame = 0; frame < numFrames; frame++) {
      int const mpiBatchIndex = calcMPIBatchIndex(frame);
      if (getMPIBlock()->getRank() == 0) {
         frameTimestamps.at(frame) =
               BufferUtils::readActivityFromPvp(path.c_str(), &pvpBuffer, frame);
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
         T *localData              = calcBatchElementStart(frame);
         std::memcpy(
               localData, bufferData.data(), (std::size_t)pvpBuffer.getTotalElements() * sizeof(T));
      }
   }
   MPI_Bcast(
         frameTimestamps.data(), getNumFrames(), MPI_DOUBLE, 0 /*root*/, getMPIBlock()->getComm());
   *simTimePtr = frameTimestamps[0];
}

template <typename T>
int CheckpointEntryPvpBuffer<T>::getNumFrames() const {
   return getMPIBlock()->getBatchDimension() * mLayerLoc->nbatch;
}

template <typename T>
T *CheckpointEntryPvpBuffer<T>::calcBatchElementStart(int frame) const {
   int const localBatchIndex = frame % mLayerLoc->nbatch;
   int const nx              = mLayerLoc->nx + mXMargins;
   int const ny              = mLayerLoc->ny + mYMargins;
   return &mDataPointer[localBatchIndex * nx * ny * mLayerLoc->nf];
}

template <typename T>
int CheckpointEntryPvpBuffer<T>::calcMPIBatchIndex(int frame) const {
   return frame / mLayerLoc->nbatch; // Integer division
}

template <typename T>
void CheckpointEntryPvpBuffer<T>::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}
} // end namespace PV
