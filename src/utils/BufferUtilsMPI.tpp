#include <vector>
#include "arch/mpi/mpi.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"

namespace PV {
namespace BufferUtils {

template <typename T>
void scatter(
      std::shared_ptr<MPIBlock const> mpiBlock,
      Buffer<T> &buffer,
      unsigned int localWidth,
      unsigned int localHeight,
      int mpiBatchIndex,
      int sourceProcess) {
   long dataSize = (long)sizeof(T);
   if (mpiBlock->getRank() == sourceProcess) {
      // This assumes buffer's dimensions are nxGlobal x nyGlobal
      int xMargins       = buffer.getWidth() - (localWidth * mpiBlock->getNumColumns());
      int yMargins       = buffer.getHeight() - (localHeight * mpiBlock->getNumRows());
      int localWidthExt  = localWidth + xMargins;
      int localHeightExt = localHeight + yMargins;
      long numElements   = (long)localWidthExt * (long)localHeightExt * (long)buffer.getFeatures();
      int numRows        = mpiBlock->getNumRows();
      int numColumns     = mpiBlock->getNumColumns();

      // Loop through each rank.
      // Uses Buffer::crop and MPI_Send to give each process the correct slice of input data.
      long numBytes   = numElements * dataSize;
      int numBytesInt = static_cast<int>(numBytes);
      FatalIf(
            static_cast<long>(numBytesInt) != numBytes,
            "BufferUtils::scatter() needs to send/recv %ld bytes over MPI, which is too large.\n",
            numBytes);
      for (int sendRow = mpiBlock->getNumRows() - 1; sendRow >= 0; --sendRow) {
         for (int sendColumn = numColumns - 1; sendColumn >= 0; --sendColumn) {
            int sendRank = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, mpiBatchIndex);

            int sliceRank         = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, 0);
            unsigned int cropLeft = localWidth * columnFromRank(sliceRank, numRows, numColumns);
            unsigned int cropTop  = localHeight * rowFromRank(sliceRank, numRows, numColumns);

            Buffer<T> croppedBuffer =
                  buffer.extract(cropLeft, cropTop, localWidthExt, localHeightExt);
            pvAssert(numElements == croppedBuffer.getTotalElements());

            if (sendRank != sourceProcess) {
               // If this isn't for root, ship it off to the appropriate process.
               MPI_Send(
                     croppedBuffer.asVector().data(),
                     numBytesInt,
                     MPI_BYTE,
                     sendRank,
                     31,
                     mpiBlock->getComm());
            }
            else {
               // Root process is in this batch element; this is our slice.
               buffer.set(
                     croppedBuffer.asVector(),
                     localWidthExt,
                     localHeightExt,
                     buffer.getFeatures());
            }
         }
      }
   }
   else if (mpiBlock->getBatchIndex() == mpiBatchIndex) {
      pvAssert(mpiBlock->getRank() != sourceProcess);
      long numBytes   = buffer.getTotalElements() * dataSize;
      int numBytesInt = static_cast<int>(numBytes);
      FatalIf(
            static_cast<long>(numBytesInt) != numBytes,
            "BufferUtils::scatter() needs to send/recv %ld bytes over MPI, which is too large.\n",
            numBytes);
      MPI_Recv(
            buffer.asVector().data(),
            numBytesInt,
            MPI_BYTE,
            sourceProcess,
            31,
            mpiBlock->getComm(),
            MPI_STATUS_IGNORE);
   }
}

template <typename T>
Buffer<T> gather(
      std::shared_ptr<MPIBlock const> mpiBlock,
      Buffer<T> const &buffer,
      unsigned int localWidth,
      unsigned int localHeight,
      int mpiBatchIndex,
      int destProcess) {
   // Here, we assume that buffer is the size of local,
   // not global, nx and ny. If we have margins, then
   // buffer.getWidth != localWidth. Same for Y.
   int xMargins  = buffer.getWidth() - localWidth;
   int yMargins  = buffer.getHeight() - localHeight;
   long dataSize = (long)sizeof(T);

   if (mpiBlock->getRank() == destProcess) {
      int const numRows    = mpiBlock->getNumRows();
      int const numColumns = mpiBlock->getNumColumns();
      int globalWidth      = localWidth * numColumns + xMargins;
      int globalHeight     = localHeight * numRows + yMargins;
      long numElements     = buffer.getTotalElements();

      Buffer<T> globalBuffer(globalWidth, globalHeight, buffer.getFeatures());

      // Receive each slice of our full buffer from each MPI process
      long numBytes = static_cast<std::size_t>(numElements) * dataSize;
      int numBytesInt      = static_cast<int>(numBytes);
      FatalIf(
            static_cast<long>(numBytesInt) != numBytes,
            "Buffer:gather() needs to send/recv %ld bytes over MPI, which is too large.\n",
            numBytes);
      std::vector<T> tempMem(numElements);
      for (int recvRow = numRows - 1; recvRow >= 0; --recvRow) {
         for (int recvColumn = numColumns - 1; recvColumn >= 0; --recvColumn) {
            int recvRank = mpiBlock->calcRankFromRowColBatch(recvRow, recvColumn, mpiBatchIndex);
            Buffer<T> smallBuffer;
            if (recvRank != destProcess) {
               // This is nearly identical to the non-root receive in scatter
               MPI_Recv(
                     tempMem.data(),
                     numBytesInt,
                     MPI_BYTE,
                     recvRank,
                     32,
                     mpiBlock->getComm(),
                     MPI_STATUS_IGNORE);
               smallBuffer.set(
                     tempMem, buffer.getWidth(), buffer.getHeight(), buffer.getFeatures());
            }
            else {
               smallBuffer = buffer;
            }
            int sliceRank       = mpiBlock->calcRankFromRowColBatch(recvRow, recvColumn, 0);
            unsigned int sliceX = localWidth * columnFromRank(sliceRank, numRows, numColumns);
            unsigned int sliceY = localHeight * rowFromRank(sliceRank, numRows, numColumns);

            // crop out the border regions of small buffer, unless the rank sits on the edge
            // of the MPI quilt
            int topMargin  = yMargins / 2; // integer division, although usu. margins are even
            int leftMargin = xMargins / 2;
            if (recvRow > 0) {
               sliceY += topMargin;
               smallBuffer.crop(
                     smallBuffer.getWidth(), smallBuffer.getHeight() - topMargin, Buffer<T>::SOUTH);
            }
            if (recvRow < numRows - 1) {
               smallBuffer.crop(
                     smallBuffer.getWidth(),
                     smallBuffer.getHeight() - (yMargins - topMargin),
                     Buffer<T>::NORTH);
            }
            if (recvColumn > 0) {
               sliceX += leftMargin;
               smallBuffer.crop(
                     smallBuffer.getWidth() - leftMargin, smallBuffer.getHeight(), Buffer<T>::EAST);
            }
            if (recvColumn < numColumns - 1) {
               smallBuffer.crop(
                     smallBuffer.getWidth() - (xMargins - leftMargin),
                     smallBuffer.getHeight(),
                     Buffer<T>::WEST);
            }

            globalBuffer.insert(smallBuffer, sliceX, sliceY);
         }
      }
      return globalBuffer;
   }
   else if (mpiBlock->getBatchIndex() == mpiBatchIndex) {
      pvAssert(mpiBlock->getRank() != destProcess);
      // Send our chunk of the global buffer to root for reassembly
      long numBytes   = buffer.getTotalElements() * dataSize;
      int numBytesInt = static_cast<int>(numBytes);
      FatalIf(
            static_cast<long>(numBytesInt) != numBytes,
            "Buffer:gather() needs to send/recv %ld bytes over MPI, which is too large.\n",
            numBytes);
      MPI_Send(
            buffer.asVector().data(),
            numBytesInt,
            MPI_BYTE,
            destProcess,
            32,
            mpiBlock->getComm());
   }
   return buffer;
}

template <typename T>
SparseList<T> gatherSparse(
      std::shared_ptr<MPIBlock const> mpiBlock,
      SparseList<T> list,
      int mpiBatchIndex,
      int destProcess) {
   unsigned int entrySize = (unsigned int)sizeof(typename SparseList<T>::Entry);
   if (mpiBlock->getRank() == destProcess) {
      SparseList<T> globalList;
      for (int recvRow = mpiBlock->getNumRows() - 1; recvRow >= 0; --recvRow) {
         for (int recvColumn = mpiBlock->getNumColumns() - 1; recvColumn >= 0; --recvColumn) {
            int recvRank = mpiBlock->calcRankFromRowColBatch(recvRow, recvColumn, mpiBatchIndex);
            SparseList<T> listChunk;
            if (recvRank != destProcess) {
               unsigned int numToRecv = 0U;
               MPI_Recv(
                     &numToRecv,
                     1,
                     MPI_UNSIGNED,
                     recvRank,
                     33,
                     mpiBlock->getComm(),
                     MPI_STATUS_IGNORE);
               if (numToRecv > 0) {
                  struct SparseList<T>::Entry *recvBuffer =
                        (struct SparseList<T>::Entry *)calloc(numToRecv, entrySize);
                  FatalIf(
                        recvBuffer == nullptr,
                        "Could not allocate a receive buffer of %u bytes.\n",
                        numToRecv * entrySize);
                  MPI_Recv(
                        recvBuffer,
                        numToRecv * entrySize,
                        MPI_BYTE,
                        recvRank,
                        34,
                        mpiBlock->getComm(),
                        MPI_STATUS_IGNORE);
                  for (uint32_t i = 0; i < numToRecv; ++i) {
                     listChunk.addEntry(recvBuffer[i]);
                  }
                  free(recvBuffer);
               }
            }
            else {
               listChunk = list;
            }
            globalList.merge(listChunk);
         }
      }
      return globalList;
   }
   else if (mpiBlock->getBatchIndex() == mpiBatchIndex) {
      vector<struct SparseList<T>::Entry> toSend = list.getContents();
      unsigned int numToSend                     = toSend.size();
      MPI_Send(&numToSend, 1, MPI_UNSIGNED, destProcess, 33, mpiBlock->getComm());
      if (numToSend > 0) {
         MPI_Send(
               toSend.data(),
               numToSend * entrySize,
               MPI_BYTE,
               destProcess,
               34,
               mpiBlock->getComm());
      }
   }
   return list;
}

} // end namespace BufferUtils

} // end namespace PV
