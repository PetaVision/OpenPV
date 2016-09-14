#ifndef __BUFFERSLICER_HPP_
#define __BUFFERSLICER_HPP_

#include "columns/Communicator.hpp"
#include "Buffer.hpp"

namespace PV
{

class BufferSlicer {
   public:
      BufferSlicer(Communicator &comm);

      void scatter(Buffer &buffer,
                   unsigned int sliceStrideX,  // These values should be the
                   unsigned int sliceStrideY); // layer's local restricted nx and ny
      void gather( Buffer &buffer,
                   unsigned int sliceStrideX,
                   unsigned int sliceStrideY);

   private:
      Communicator& mComm;
};
}
#endif
