#include "BufferSlicer.hpp"
#include "arch/mpi/mpi.h"

namespace PV {

BufferSlicer::BufferSlicer(Communicator &comm) : mComm(comm) {}

void BufferSlicer::scatter(Buffer &buffer,
                           unsigned int sliceStrideX,
                           unsigned int sliceStrideY) {
   // This assumes buffer's dimensions are nxGlobal x nyGlobal
   int xMargins    = buffer.getWidth()
                   - (sliceStrideX * mComm.numCommColumns());
   int yMargins    = buffer.getHeight()
                   - (sliceStrideY * mComm.numCommRows());
   int numElements = (sliceStrideX + xMargins)
                   * (sliceStrideY + yMargins)
                   * buffer.getFeatures();

   // Defining this outside of the loop lets it contain the correct
   // data for the root process at the end
   Buffer croppedBuffer;
   if (mComm.commRank() == 0) {
      // Loop through each rank, ending on the root process.
      // Uses Buffer::crop and MPI_Send to give each process
      // the correct slice of input data.
      for (int sendRank = mComm.commSize()-1; sendRank >= 0; --sendRank) {
         // Copy the input data to a temporary buffer.
         // This gets cropped to the layer size below.
         croppedBuffer = buffer;
         unsigned int cropLeft = sliceStrideX
                               * columnFromRank(
                                     sendRank,
                                     mComm.numCommRows(),
                                     mComm.numCommColumns());
         unsigned int cropTop  = sliceStrideY
                               * rowFromRank(
                                     sendRank,
                                     mComm.numCommRows(),
                                     mComm.numCommColumns());

         // Crop the input data to the size of one process.
         croppedBuffer.translate(-cropLeft, -cropTop);
         croppedBuffer.crop(sliceStrideX + xMargins,
                            sliceStrideY + yMargins,
                            Buffer::NORTHWEST);

         assert(numElements == croppedBuffer.getTotalElements());

         if (sendRank != 0) {
            // If this isn't for root, ship it off to the appropriate process.
            MPI_Send(croppedBuffer.asVector().data(),
                     numElements,
                     MPI_FLOAT,
                     mComm.commRank(),
                     31,
                     mComm.communicator());
         }
         else { 
            // This is root, keep a slice for ourselves
            buffer.set(croppedBuffer.asVector(),
                       sliceStrideX + xMargin,
                       sliceStrideY + yMargin,
                       buffer.getFeatures());
         }
      }
   }
   else {
      // Create a temporary array to receive from MPI, move the values into
      // a vector, and then set our Buffer's contents to that vector.
      // This set of conversions could be greatly reduced by giving Buffer
      // a constructor that accepts raw memory.
      float *tempMem = (float*)calloc(numElements, sizeof(float));
      pvErrorIf(tempMem == nullptr,
                "Could not allocate a receive buffer of %d bytes.\n",
                numElements * sizeof(float));
      MPI_Recv(tempMem,
               numElements,
               MPI_FLOAT,
               0,
               31,
               mComm.communicator(),
               MPI_STATUS_IGNORE);
      std::vector<float> tempVector(numElements);
      for (int i = 0; i < numElements; ++i) {
         tempVector.at(i) = tempMem[i];
      }
      free(tempMem);
      buffer.set(tempVector,
                 sliceStrideX + xMargin,
                 sliceStrideY + xMargin,
                 buffer.getFeatures());
   }
}

void BufferSlicer::gather(Buffer &buffer,
                          unsigned int sliceStrideX,
                          unsigned int sliceStrideY) {
   // Here, we assume that buffer is the size of local,
   // not global, nx and ny. If we have margins, then
   // buffer.getWidth != sliceStrideX. Same for Y.
   int xMargin = buffer.getWidth()  - sliceStrideX;
   int yMargin = buffer.getHeight() - sliceStrideY;

   if (mComm.commRank() == 0) {
      int globalWidth   = sliceStrideX * mComm.numCommColumns() + xMargin;
      int globalHeight  = sliceStrideY * mComm.numCommRows()    + yMargin;
      int numElements   = buffer.getTotalElements();

      Buffer globalBuffer(globalWidth, globalHeight, buffer.getFeatures());

      // Receive each slice of our full buffer from each MPI process
      float *tempMem = (float*)calloc(numElements, sizeof(float));
      pvErrorIf(tempMem == nullptr,
            "Could not allocate a receive buffer of %d bytes.\n",
            numElements * sizeof(float));
      for (int recvRank = mComm.commSize() - 1; recvRank >= 0; --recvRank) {
         Buffer smallBuffer;
         if (recvRank != 0) {
            // This is nearly identical to the non-root receive in scatter
            MPI_Recv(tempMem,
                     numElements,
                     MPI_FLOAT,
                     recvRank,
                     171 + recvRank, // Unique tag for each rank
                     mComm.communicator(),
                     MPI_STATUS_IGNORE);
            std::vector<float> tempVector(numElements);
            for (int i = 0; i < numElements; ++i) {
               tempVector.at(i) = tempMem[i];
            }
            smallBuffer.set(tempVector,
                            buffer.getWidth(),
                            buffer.getHeight(),
                            buffer.getFeatures()); 
         }
         else {
            smallBuffer = buffer;
         }
         unsigned int sliceX = sliceStrideX
                             * columnFromRank(
                                   recvRank,
                                   mComm.numCommRows(),
                                   mComm.numCommColumns());
         unsigned int sliceY = sliceStrideY
                             * rowFromRank(
                                   recvRank,
                                   mComm.numCommRows(),
                                   mComm.numCommColumns());

         // Place our chunk into the global buffer
         for (int y = 0; y < buffer.getHeight(); ++y) {
            for (int x = 0; x < buffer.getWidth(); ++x) {
               for(int f = 0; f < buffer.getFeatures(); ++f) {
                  globalBuffer.set(sliceX + x,
                                   sliceY + y,
                                   f,
                                   smallBuffer.at(x, y, f));
               }
            }
         }
      }
      free(tempMem);
      buffer = globalBuffer;
   }
   else {
      // Send our chunk of the global buffer to root for reassembly
      MPI_Send(buffer.asVector().data(),
               buffer.getTotalElements(),
               MPI_FLOAT,
               mComm.commRank(),
               171 + mComm.commRank(),
               mComm.communicator());
   }
}










}
