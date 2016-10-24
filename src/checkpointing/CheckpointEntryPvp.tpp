/*
 * CheckpointEntry.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntry class hierarchy.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "structures/Buffer.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cstring>
#include <vector>

namespace PV {

template <typename T>
void CheckpointEntryPvp<T>::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   FileStream *fileStream = nullptr;
   if (getCommunicator()->commRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      fileStream = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      BufferUtils::ActivityHeader header = BufferUtils::buildActivityHeader<T>(
         mLayerLoc->nxGlobal, mLayerLoc->nyGlobal, mLayerLoc->nf, mLayerLoc->nbatch);
      BufferUtils::writeActivityHeader(*fileStream, header);
   }
   for (int b = 0; b < mLayerLoc->nbatch; b++) {
      T *batchElementStart = calcBatchElementStart(b);
      int nxLocal = mLayerLoc->nx;
      int nyLocal = mLayerLoc->ny;
      int nf      = mLayerLoc->nf;
      if (mExtended) {
         nxLocal += mLayerLoc->halo.lt + mLayerLoc->halo.rt;
         nyLocal += mLayerLoc->halo.dn + mLayerLoc->halo.up;
      }
      Buffer<T> pvpBuffer{batchElementStart, nxLocal, nyLocal, nf};
      pvpBuffer.crop(mLayerLoc->nx, mLayerLoc->ny, Buffer<T>::CENTER);
      Buffer<T> pvpBufferGlobal = BufferUtils::gather(getCommunicator(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny);
      if (fileStream) {
         fileStream->write(&simTime, sizeof(simTime));
         fileStream->write(pvpBufferGlobal.asVector().data(), sizeof(T)*pvpBufferGlobal.getTotalElements());
      }
   }
   delete fileStream;
}

template <typename T>
void CheckpointEntryPvp<T>::read(std::string const &checkpointDirectory, double *simTimePtr) const {
   std::string path = generatePath(checkpointDirectory, "pvp");
   MPI_Datatype *exchangeDatatypes = mExtended ? getCommunicator()->newDatatypes(mLayerLoc) : nullptr;
   for (int b = 0; b < mLayerLoc->nbatch; b++) {
      Buffer<T> pvpBuffer;
      if (getCommunicator()->commRank()==0) {
         *simTimePtr = BufferUtils::readFromPvp(path.c_str(), &pvpBuffer, b);
      }
      else {
         pvpBuffer.resize(mLayerLoc->nx, mLayerLoc->ny, mLayerLoc->nf);
      }
      BufferUtils::scatter(getCommunicator(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny);
      if (mExtended) {
         int const nxExt = mLayerLoc->nx + mLayerLoc->halo.lt + mLayerLoc->halo.rt;
         int const nyExt = mLayerLoc->ny + mLayerLoc->halo.dn + mLayerLoc->halo.up;
         pvpBuffer.grow(nxExt, nyExt, Buffer<T>::CENTER);
      }
      std::vector<T> bufferData = pvpBuffer.asVector();
      T *batchElementStart      = calcBatchElementStart(b);
      std::memcpy(batchElementStart, bufferData.data(), sizeof(T)*pvpBuffer.getTotalElements());
      if (mExtended) {
         std::vector<MPI_Request> req{};
         getCommunicator()->exchange(batchElementStart, exchangeDatatypes, mLayerLoc, req);
         getCommunicator()->wait(req);
      }
   }
   MPI_Bcast(simTimePtr, 1, MPI_DOUBLE, 0, getCommunicator()->communicator());
   getCommunicator()->freeDatatypes(exchangeDatatypes);
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
}  // end namespace PV
