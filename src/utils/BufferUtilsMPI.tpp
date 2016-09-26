#include "arch/mpi/mpi.h"
#include "conversions.h"
#include "PVLog.hpp"

namespace PV {
   namespace BufferUtils {
      
      template <typename T>
      void scatter(Communicator *comm,
                   Buffer<T> &buffer,
                   unsigned int localWidth,
                   unsigned int localHeight) {
         size_t dataSize = sizeof(T);
         if (comm->commRank() == 0) {
            // This assumes buffer's dimensions are nxGlobal x nyGlobal
            int xMargins    = buffer.getWidth()
                            - (localWidth * comm->numCommColumns());
            int yMargins    = buffer.getHeight()
                            - (localHeight * comm->numCommRows());
            int numElements = (localWidth + xMargins)
                            * (localHeight + yMargins)
                            * buffer.getFeatures();

            // Loop through each rank, ending on the root process.
            // Uses Buffer::crop and MPI_Send to give each process
            // the correct slice of input data.
            for (int sendRank = comm->commSize()-1; sendRank >= 0; --sendRank) {
               // Copy the input data to a temporary buffer.
               // This gets cropped to the layer size below.
               Buffer<T> croppedBuffer = buffer;
               unsigned int cropLeft = localWidth
                                     * columnFromRank(
                                           sendRank,
                                           comm->numCommRows(),
                                           comm->numCommColumns());
               unsigned int cropTop  = localHeight
                                     * rowFromRank(
                                           sendRank,
                                           comm->numCommRows(),
                                           comm->numCommColumns());

               // Crop the input data to the size of one process.
               croppedBuffer.translate(-cropLeft, -cropTop);
               croppedBuffer.crop(localWidth + xMargins,
                                  localHeight + yMargins,
                                  Buffer<T>::NORTHWEST);

               assert(numElements == croppedBuffer.getTotalElements());

               if (sendRank != 0) {
                  // If this isn't for root, ship it off to the appropriate process.
                  MPI_Send(croppedBuffer.asVector().data(),
                           numElements * dataSize,
                           MPI_BYTE,
                           sendRank,
                           31,
                           comm->communicator());
               }
               else { 
                  // This is root, keep a slice for ourselves
                  buffer.set(croppedBuffer.asVector(),
                             localWidth + xMargins,
                             localHeight + yMargins,
                             buffer.getFeatures());
               }
            }
         }
         else {
            // Create a temporary array to receive from MPI, move the values into
            // a vector, and then set our Buffer's contents to that vector.
            // This set of conversions could be greatly reduced by giving Buffer
            // a constructor that accepts raw memory.
            T *tempMem = (T*)calloc(buffer.getTotalElements(), dataSize);
            pvErrorIf(tempMem == nullptr,
                      "Could not allocate a receive buffer of %d bytes.\n",
                      buffer.getTotalElements() * dataSize);
            MPI_Recv(tempMem,
                     buffer.getTotalElements() * dataSize,
                     MPI_BYTE,
                     0,
                     31,
                     comm->communicator(),
                     MPI_STATUS_IGNORE);
           buffer.set(tempMem,
                       buffer.getWidth(),
                       buffer.getHeight(),
                       buffer.getFeatures());
            free(tempMem);
         }
      }

      template <typename T>
      Buffer<T> gather(Communicator *comm,
                       Buffer<T> buffer,
                       unsigned int localWidth,
                       unsigned int localHeight) {
         // Here, we assume that buffer is the size of local,
         // not global, nx and ny. If we have margins, then
         // buffer.getWidth != localWidth. Same for Y.
         int xMargins = buffer.getWidth()  - localWidth;
         int yMargins = buffer.getHeight() - localHeight;
         size_t dataSize = sizeof(T);

         if (comm->commRank() == 0) {
            int globalWidth   = localWidth * comm->numCommColumns() + xMargins;
            int globalHeight  = localHeight * comm->numCommRows()    + yMargins;
            int numElements   = buffer.getTotalElements();

            Buffer<T> globalBuffer(globalWidth, globalHeight, buffer.getFeatures());

            // Receive each slice of our full buffer from each MPI process
            T *tempMem = (T*)calloc(numElements, dataSize);
            pvErrorIf(tempMem == nullptr,
                  "Could not allocate a receive buffer of %d bytes.\n",
                  numElements * dataSize);
            for (int recvRank = comm->commSize() - 1; recvRank >= 0; --recvRank) {
               Buffer<T> smallBuffer;
               if (recvRank != 0) {
                  // This is nearly identical to the non-root receive in scatter
                  MPI_Recv(tempMem,
                           numElements * dataSize,
                           MPI_BYTE,
                           recvRank,
                           171 + recvRank, // Unique tag for each rank
                           comm->communicator(),
                           MPI_STATUS_IGNORE);
                 smallBuffer.set(tempMem,
                                 buffer.getWidth(),
                                 buffer.getHeight(),
                                 buffer.getFeatures()); 
               }
               else {
                  smallBuffer = buffer;
               }
               unsigned int sliceX = localWidth
                                   * columnFromRank(
                                         recvRank,
                                         comm->numCommRows(),
                                         comm->numCommColumns());
               unsigned int sliceY = localHeight
                                   * rowFromRank(
                                         recvRank,
                                         comm->numCommRows(),
                                         comm->numCommColumns());

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
            return globalBuffer;
         }
         else {
            // Send our chunk of the global buffer to root for reassembly
            MPI_Send(buffer.asVector().data(),
                     buffer.getTotalElements() * dataSize,
                     MPI_BYTE,
                     0,
                     171 + comm->commRank(),
                     comm->communicator());
         }
         return buffer;
      }

      template <typename T>
      SparseList<T> gatherSparse(Communicator *comm,
                                 SparseList<T> list) {
         size_t entrySize = sizeof(SparseList<T>::Entry);
         if (comm->commRank() == 0) {
            SparseList<T> globalList;
            for (int recvRank = comm->commSize() - 1; recvRank >= 0; --recvRank) {
               SparseList<T> listChunk;
               if (recvRank != 0) {
                  uint32_t numToRecv = 0;
                  MPI_Recv(&numToRecv,
                           1,
                           MPI_INT,
                           recvRank,
                           171 + recvRank * 2, // Odd tag for numToRecv
                           comm->communicator(),
                           MPI_STATUS_IGNORE);
                 if (numToRecv > 0) {
                     struct SparseList<T>::Entry *recvBuffer =
                        (struct SparseList<T>::Entry*)calloc(numToRecv, entrySize);
                     pvErrorIf(recvBuffer == nullptr, 
                        "Could not allocate a receive buffer of %d bytes.\n",
                        numToRecv * entrySize);
                     MPI_Recv(recvBuffer,
                           numToRecv * entrySize,
                           MPI_BYTE,
                           recvRank,
                           172 + recvRank * 2, // Even tag for data
                           comm->communicator(),
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
            return globalList;
         }
         else {
            vector<struct SparseList<T>::Entry> toSend = list.getContents();
            uint32_t numToSend = toSend.size();
            MPI_Send(&numToSend,
                     1,
                     MPI_INT,
                     0,
                     171 + comm->commRank() * 2,
                     comm->communicator());
            if (numToSend > 0) {
               MPI_Send(toSend.data(),
                        numToSend * entrySize,
                        MPI_BYTE,
                        0,
                        172 + comm->commRank() * 2,
                        comm->communicator());
            }
         }
         return list;
      }
   }
}  // end namespace PV
