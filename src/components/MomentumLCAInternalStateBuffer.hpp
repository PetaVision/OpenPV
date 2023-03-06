/*
 * MomentumLCAInternalStateBuffer.hpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#ifndef MOMENTUMLCAINTERNALSTATEBUFFER_HPP_
#define MOMENTUMLCAINTERNALSTATEBUFFER_HPP_

#include "components/HyPerLCAInternalStateBuffer.hpp"

namespace PV {

class MomentumLCAInternalStateBuffer : public HyPerLCAInternalStateBuffer {
  protected:
   /**
    * List of parameters needed from the MomentumLCAInternalStateBuffer class
    * @name MomentumLCALayer Parameters
    * @{
    */

   virtual void ioParam_LCAMomentumRate(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   MomentumLCAInternalStateBuffer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~MomentumLCAInternalStateBuffer();

  protected:
   MomentumLCAInternalStateBuffer();
   int initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;
#endif

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
#ifdef PV_USE_CUDA
   virtual void updateBufferGPU(double simTime, double deltaTime) override;

   virtual void runKernel();
#endif // PV_USE_CUDA

   // Data members
  protected:
   float mLCAMomentumRate       = 0.0f;
   RestrictedBuffer *mPrevDrive = nullptr;
}; // class MomentumLCAInternalStateBuffer

} /* namespace PV */
#endif /* MOMENTUMLCAINTERNALSTATEBUFFER_HPP_ */
