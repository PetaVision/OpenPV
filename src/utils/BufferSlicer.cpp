#include "BufferSlicer.hpp"
#include "arch/mpi/mpi.h"

namespace PV {

BufferSlicer::BufferSlicer(Communicator &comm) : mComm(comm) {}

void BufferSlicer::scatter(Buffer &buffer,
                           unsigned int sliceStrideX,
                           unsigned int sliceStrideY) {

   // This assumes buffer's dimensions are nxGlobal x nyGlobal
   int xMargins = buffer.getWidth()  - (sliceStrideX * mComm.numCommColumns());
   int yMargins = buffer.getHeight() - (sliceStrideY * mComm.numCommRows());
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
      float *tempBuffer = (float*)calloc(numElements, sizeof(float));
      MPI_Recv(tempBuffer,
               numElements,
               MPI_FLOAT,
               0,
               31,
               mComm.communicator(),
               MPI_STATUS_IGNORE);
      std::vector<float> bufferData(numElements);
      for (int i = 0; i < numElements; ++i) {
         bufferData.at(i) = tempBuffer[i];
      }
      free(tempBuffer);
      buffer.set(bufferData,
                 sliceStrideX + xMargin,
                 sliceStrideY + xMargin,
                 buffer.getFeatures());
   }
}

void BufferSlicer::gather(Buffer &buffer) {
}










}
