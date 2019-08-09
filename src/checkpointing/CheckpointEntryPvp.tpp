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

template <typename T>
CheckpointEntryPvp<T>::CheckpointEntryPvp(
      std::string const &name,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(name, mpiBlock) {
   initialize(layerLoc, extended);
}

template <typename T>
CheckpointEntryPvp<T>::CheckpointEntryPvp(
      std::string const &objName,
      std::string const &dataName,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(objName, dataName, mpiBlock) {
   initialize(layerLoc, extended);
}

template <typename T>
void CheckpointEntryPvp<T>::initialize(PVLayerLoc const *layerLoc, bool extended) {
   mLayerLoc = layerLoc;
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
void CheckpointEntryPvp<T>::write(
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
void CheckpointEntryPvp<T>::read(std::string const &checkpointDirectory, double *simTimePtr) const {
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
      if (numFrames < header.nBands) {
         WarnLog().printf(
               "CheckpointEntryPvp reading \"%s\" expects %d bands, but the file has %d bands. "
               "Extra bands will be ignored.\n",
               path.c_str(),
               numFrames,
               header.nBands);
      }
      else if (numFrames > header.nBands) {
         WarnLog().printf(
               "CheckpointEntryPvp reading \"%s\" expects %d bands, but the file only has "
               "%d bands. Bands will be repeated cyclically.\n",
               path.c_str(),
               numFrames,
               header.nBands);
      }
   }
   Buffer<T> pvpBuffer;
   std::vector<double> frameTimestamps;
   frameTimestamps.resize(numFrames);
   for (int frame = 0; frame < numFrames; frame++) {
      int const mpiBatchIndex = calcMPIBatchIndex(frame);
      if (getMPIBlock()->getRank() == 0) {
         frameTimestamps.at(frame) =
               BufferUtils::readActivityFromPvp(path.c_str(), &pvpBuffer, frame, nullptr);
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
   applyTimestamps(frameTimestamps);
   *simTimePtr = frameTimestamps[0];
}

template <typename T>
void CheckpointEntryPvp<T>::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}
} // end namespace PV
