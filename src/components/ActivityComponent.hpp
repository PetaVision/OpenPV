/*
 * ActivityComponent.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef ACTIVITYCOMPONENT_HPP_
#define ACTIVITYCOMPONENT_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "components/ActivityBuffer.hpp"
#include "utils/Timer.hpp"

#ifdef PV_USE_CUDA
#include <arch/cuda/CudaTimer.hpp>
#endif // PV_USE_CUDA

namespace PV {

/**
 * The base class for the activity component of HyPerLayer.
 */
class ActivityComponent : public ComponentBasedObject {
  protected:
   /**
    * List of parameters needed from the ActivityComponent class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief updateGpu: When compiled using CUDA or OpenCL GPU acceleration, this flag tells whether
    * this layer's updateState method should use the GPU.
    * If PetaVision was compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   ActivityComponent(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ActivityComponent();

   Response::Status updateState(double simTime, double deltaTime);

   bool getUpdateGpu() { return mUpdateGpu; }

#ifdef PV_USE_CUDA
   void useCuda();
   void copyFromCuda();
   void copyToCuda();
#endif // PV_USE_CUDA

   PVLayerLoc const *getLayerLoc() const { return mActivity->getLayerLoc(); }
   int getNumNeurons() const { return getLayerLoc()->nx * getLayerLoc()->ny * getLayerLoc()->nf; }
   int getNumNeuronsAcrossBatch() const { return getNumNeurons() * getLayerLoc()->nbatch; }
   int getNumExtended() const { return mActivity->getBufferSize(); }
   int getNumExtendedAcrossBatch() const { return getNumExtended() * getLayerLoc()->nbatch; }

   float const *getActivity() const { return mActivity->getBufferData(); }

  protected:
   ActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void fillComponentTable() override;

   virtual ActivityBuffer *createActivity();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
   virtual Response::Status copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

   virtual Response::Status updateActivity(double simTime, double deltaTime);

  protected:
   bool mUpdateGpu = false;

   ActivityBuffer *mActivity = nullptr;

   Timer *mUpdateTimer = nullptr;
#ifdef PV_USE_CUDA
   PVCuda::CudaTimer *mUpdateCudaTimer = nullptr;
#endif // PV_USE_CUA
};

} // namespace PV

#endif // ACTIVITYCOMPONENT_HPP_
