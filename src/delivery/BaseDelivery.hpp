/*
 * BaseDelivery.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef BASEDELIVERY_HPP_
#define BASEDELIVERY_HPP_

#include "components/ConnectionData.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/PublisherComponent.hpp"
#include "delivery/LayerInputDelivery.hpp"

namespace PV {

/**
 * BaseDelivery provides the common interface for delivery classes. It extends LayerInputDelivery
 * by providing pre- and post-synaptic layers, and provides functions for handling
 * threaded delivery to avoid collisions when using OpenMP.
 */
class BaseDelivery : public LayerInputDelivery {
  public:
   BaseDelivery(char const *name, PVParams *params, Communicator *comm);

   virtual ~BaseDelivery() {}

   PublisherComponent *getPreData() const { return mPreData; }
   LayerInputBuffer *getPostGSyn() const { return mPostGSyn; }

  protected:
   BaseDelivery() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_OPENMP_THREADS
   /**
    * If called, allocates one buffer per openmp thread, where each buffer is the
    * size of the restricted postsynaptic layer (not including batching).
    * That is, the size of each openmp thread's buffer is mPostGSyn->getBufferSize().
    */
   void allocateThreadGSyn();

   /**
    * Sets all ThreadGSyn buffers to 0.
    */
   void clearThreadGSyn();
#endif // PV_USE_OPENMP_THREADS

   /**
    * Accumulates the buffers in ThreadGSyn into the given buffer of size
    * mPostGSyn->getBufferSize().
    * That is, if the value of buffer[k] is G0 on entry, its value is
    *      G0 + sum_n ThreadGSyn[n][k].
    * If not using OpenMP, or if the ThreadGSyn vector is empty, the routine has no effect.
    */
   void accumulateThreadGSyn(float *baseGSynBuffer);

   /**
    * If not using OpenMP or if there is only one OpenMP thread, returns the
    * input argument baseGSynBuffer.
    * If using more than one OpenMP thread, returns the pointer to the
    * mThreadGSyn element corresponding to the current OpenMP thread.
    * This way, threads can work in parallel on the GSyn delivery without worrying about
    * collisions. After each thread has done its work, the accumulateThreadGSyn
    * function member should be called, with the same argument baseGSynBuffer, to accumulate
    * the mThreadGSyn elements into that buffer.
    *
    */
   float *setWorkingGSynBuffer(float *baseGSynBuffer);
   // Note: this gets called in a loop over neuron index. If this function is not inlined by the
   // compiler, it should probably be unrolled.

  protected:
   ConnectionData *mConnectionData = nullptr;
   PublisherComponent *mPreData    = nullptr;
   LayerInputBuffer *mPostGSyn     = nullptr;
// Rather than the layers, should we store the buffers and the PVLayerLoc data?

#ifdef PV_USE_OPENMP_THREADS
   // Accumulate buffer, used by some subclasses if numThreads > 1 to avoid
   // parallelization collisions.
   std::vector<std::vector<float>> mThreadGSyn;
   int mNumThreads = 1;
#endif // PV_USE_OPENMP_THREADS
};

} // namespace PV

#endif // BASEDELIVERY_HPP_
