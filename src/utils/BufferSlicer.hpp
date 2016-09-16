#ifndef __BUFFERSLICER_HPP_
#define __BUFFERSLICER_HPP_

#include "columns/Communicator.hpp"
#include "Buffer.hpp"

namespace PV
{

template <class T>
class BufferSlicer {
   public:
      BufferSlicer(Communicator *comm);

      // Every MPI process should call this with a buffer with
      // dimensions globalNx x globalNy, with the root process
      // buffer having the actual contents to be scattered.
      // Afterwards, buffer will contain the slice of the data 
      // that has been assigned to the current MPI process,
      void scatter(Buffer<T> &buffer,
                   unsigned int localWidth,  // These values should be the
                   unsigned int localHeight); // layer's local nx and ny

      // This performs the reverse of scatter. Every MPI process
      // should call gather with the slice of data they have, and
      // afterwards the root process will have a filled buffer with
      // dimensions globalNx x globalNy. Buffer will be unchanged
      // on non-root processess.
      Buffer<T> gather(Buffer<T> buffer,
                  unsigned int localWidth,
                  unsigned int localHeight);

   private:
      Communicator *mComm;
};
}
#endif