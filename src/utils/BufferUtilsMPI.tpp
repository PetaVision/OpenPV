#include "PVAssert.hpp"
#include "PVLog.hpp"
#include "arch/mpi/mpi.h"
#include "conversions.hpp"

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
      for (int sendColumn = numColumns - 1; sendColumn >= 0; --sendColumn) {
         for (int sendRow = mpiBlock->getNumRows() - 1; sendRow >= 0; --sendRow) {
            int sendRank = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, mpiBatchIndex);
            // Copy the input data to a temporary buffer.
            // This gets cropped to the layer size below.
            Buffer<T> croppedBuffer = buffer;
            int sliceRank           = mpiBlock->calcRankFromRowColBatch(sendRow, sendColumn, 0);
            unsigned int cropLeft   = localWidth * columnFromRank(sliceRank, numRows, numColumns);
            unsigned int cropTop    = localHeight * rowFromRank(sliceRank, numRows, numColumns);

            // Crop the input data to the size of one process.
            croppedBuffer.translate(-cropLeft, -cropTop);
            croppedBuffer.crop(localWidth + xMargins, localHeight + yMargins, Buffer<T>::NORTHWEST);

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
               // Root process is in this batch element; keep a slice for ourselves
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
      // Create a temporary array to receive from MPI, move the values into
      // a vector, and then set our Buffer's contents to that vector.
      // This set of conversions could be greatly reduced by giving Buffer
      // a constructor that accepts raw memory.
      T *tempMem = (T *)calloc(buffer.getTotalElements(), dataSize);
      FatalIf(
            tempMem == nullptr,
            "Could not allocate a receive buffer of %d bytes.\n",
            buffer.getTotalElements() * dataSize);
      MPI_Recv(
            tempMem,
            buffer.getTotalElements() * dataSize,
            MPI_BYTE,
            sourceProcess,
            31,
            mpiBlock->getComm(),
            MPI_STATUS_IGNORE);
      buffer.set(tempMem, buffer.getWidth(), buffer.getHeight(), buffer.getFeatures());
      free(tempMem);
   }
}

template <typename T>
Buffer<T> gather(
      std::shared_ptr<MPIBlock const> mpiBlock,
      Buffer<T> buffer,
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
      for (int recvColumn = numColumns - 1; recvColumn >= 0; --recvColumn) {
         for (int recvRow = numRows - 1; recvRow >= 0; --recvRow) {
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

            // Place our chunk into the global buffer
            for (int y = 0; y < buffer.getHeight(); ++y) {
               for (int x = 0; x < buffer.getWidth(); ++x) {
                  for (int f = 0; f < buffer.getFeatures(); ++f) {
                     globalBuffer.set(sliceX + x, sliceY + y, f, smallBuffer.at(x, y, f));
                  }
               }
            }
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
      for (int recvColumn = mpiBlock->getNumColumns() - 1; recvColumn >= 0; --recvColumn) {
         for (int recvRow = mpiBlock->getNumRows() - 1; recvRow >= 0; --recvRow) {
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
            listChunk.appendToList(globalList);
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
