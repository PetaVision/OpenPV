#ifndef __BUFFERUTILSMPI_HPP_
#define __BUFFERUTILSMPI_HPP_

#include "columns/Communicator.hpp"
#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"

namespace PV
{
   namespace BufferUtils {
      // Every MPI process should call this with a buffer with
      // dimensions globalNx x globalNy, with the root process
      // buffer having the actual contents to be scattered.
      // Afterwards, buffer will contain the slice of the data 
      // that has been assigned to the current MPI process,
      template <typename T>
      void scatter(Communicator *comm,
                   Buffer<T> &buffer,
                   unsigned int localWidth,  // These values should be the
                   unsigned int localHeight); // layer's local nx and ny

      // This performs the reverse of scatter. Every MPI process
      // should call gather with the slice of data they have, and
      // afterwards the root process will return  a filled buffer
      // with dimensions globalNx x globalNy. The return value
      // is the input on non-root processess.
      template <typename T>
      Buffer<T> gather(Communicator *comm,
                       Buffer<T> buffer,
                       unsigned int localWidth,
                       unsigned int localHeight);

      template <typename T>
      SparseList<T> gatherSparse(Communicator *comm,
                                 SparseList<T> list);

   } // end namespace BufferUtils

}  // end namespace PV

#include "BufferUtilsMPI.tpp" // template implementations file

#endif
