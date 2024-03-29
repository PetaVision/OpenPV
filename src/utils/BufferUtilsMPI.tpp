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
   size_t dataSize = sizeof(T);
   if (mpiBlock->getRank() == sourceProcess) {
      // This assumes buffer's dimensions are nxGlobal x nyGlobal
      int xMargins      = buffer.getWidth() - (localWidth * mpiBlock->getNumColumns());
      int yMargins      = buffer.getHeight() - (localHeight * mpiBlock->getNumRows());
      int numElements   = (localWidth + xMargins) * (localHeight + yMargins) * buffer.getFeatures();
      int const numRows = mpiBlock->getNumRows();
      int const numColumns = mpiBlock->getNumColumns();

      // Loop through each rank.
      // Uses Buffer::crop and MPI_Send to give each process
      // the correct slice of input data.
      for (int sendRow = mpiBlock->getNumRows() - 1; sendRow >= 0; --sendRow) {
         for (int sendColumn = numColumns - 1; sendColumn >= 0; --sendColumn) {
            int sendRank = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, mpiBatchIndex);

            int sliceRank         = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, 0);
            unsigned int cropLeft = localWidth * columnFromRank(sliceRank, numRows, numColumns);
            unsigned int cropTop  = localHeight * rowFromRank(sliceRank, numRows, numColumns);

            Buffer<T> croppedBuffer =
                  buffer.extract(cropLeft, cropTop, localWidth + xMargins, localHeight + yMargins);
            pvAssert(numElements == croppedBuffer.getTotalElements());

            if (sendRank != sourceProcess) {
               // If this isn't for root, ship it off to the appropriate process.
               MPI_Send(
                     croppedBuffer.asVector().data(),
                     numElements * dataSize,
                     MPI_BYTE,
                     sendRank,
                     31,
                     mpiBlock->getComm());
            }
            else {
               // Root process is in this batch element; this is our slice.
               buffer.set(
                     croppedBuffer.asVector(),
                     localWidth + xMargins,
                     localHeight + yMargins,
                     buffer.getFeatures());
            }
         }
      }
   }
   else if (mpiBlock->getBatchIndex() == mpiBatchIndex) {
      pvAssert(mpiBlock->getRank() != sourceProcess);
      MPI_Recv(
            buffer.asVector().data(),
            buffer.getTotalElements() * dataSize,
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
   int xMargins    = buffer.getWidth() - localWidth;
   int yMargins    = buffer.getHeight() - localHeight;
   size_t dataSize = sizeof(T);

   if (mpiBlock->getRank() == destProcess) {
      int const numRows    = mpiBlock->getNumRows();
      int const numColumns = mpiBlock->getNumColumns();
      int globalWidth      = localWidth * numColumns + xMargins;
      int globalHeight     = localHeight * numRows + yMargins;
      int numElements      = buffer.getTotalElements();

      Buffer<T> globalBuffer(globalWidth, globalHeight, buffer.getFeatures());

      // Receive each slice of our full buffer from each MPI process
      T *tempMem = (T *)calloc(numElements, dataSize);
      FatalIf(
            tempMem == nullptr,
            "Could not allocate a receive buffer of %d bytes.\n",
            numElements * dataSize);
      for (int recvRow = numRows - 1; recvRow >= 0; --recvRow) {
         for (int recvColumn = numColumns - 1; recvColumn >= 0; --recvColumn) {
            int recvRank = mpiBlock->calcRankFromRowColBatch(recvRow, recvColumn, mpiBatchIndex);
            Buffer<T> smallBuffer;
            if (recvRank != destProcess) {
               // This is nearly identical to the non-root receive in scatter
               MPI_Recv(
                     tempMem,
                     numElements * dataSize,
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
      free(tempMem);
      return globalBuffer;
   }
   else if (mpiBlock->getBatchIndex() == mpiBatchIndex) {
      pvAssert(mpiBlock->getRank() != destProcess);
      // Send our chunk of the global buffer to root for reassembly
      MPI_Send(
            buffer.asVector().data(),
            buffer.getTotalElements() * dataSize,
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
   size_t entrySize = sizeof(typename SparseList<T>::Entry);
   if (mpiBlock->getRank() == destProcess) {
      SparseList<T> globalList;
      for (int recvRow = mpiBlock->getNumRows() - 1; recvRow >= 0; --recvRow) {
         for (int recvColumn = mpiBlock->getNumColumns() - 1; recvColumn >= 0; --recvColumn) {
            int recvRank = mpiBlock->calcRankFromRowColBatch(recvRow, recvColumn, mpiBatchIndex);
            SparseList<T> listChunk;
            if (recvRank != destProcess) {
               uint32_t numToRecv = 0;
               MPI_Recv(
                     &numToRecv, 1, MPI_INT, recvRank, 33, mpiBlock->getComm(), MPI_STATUS_IGNORE);
               if (numToRecv > 0) {
                  struct SparseList<T>::Entry *recvBuffer =
                        (struct SparseList<T>::Entry *)calloc(numToRecv, entrySize);
                  FatalIf(
                        recvBuffer == nullptr,
                        "Could not allocate a receive buffer of %d bytes.\n",
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
      uint32_t numToSend                         = toSend.size();
      MPI_Send(&numToSend, 1, MPI_INT, destProcess, 33, mpiBlock->getComm());
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
