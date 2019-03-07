/*
 * BaseObject.hpp
 *
 *  This is the base class for layers, connections, probes, components
 *  of those objects, and anything else that the Factory object needs to know about.
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef BASEOBJECT_HPP_
#define BASEOBJECT_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "columns/Messages.hpp"
#include "columns/ParamsInterface.hpp"
#include "include/pv_common.h"
#include "observerpattern/Observer.hpp"
#include "utils/PVAlloc.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <memory>

#ifdef PV_USE_CUDA
#include "arch/cuda/CudaDevice.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * The base class for layers, connections, probes, and components of those
 * objects. Provides common interfaces for CommunicateInitInfo, AllocateDataStructures,
 * SetInitialValues messages, and a few others.
 */
class BaseObject : public ParamsInterface {
  public:
   virtual ~BaseObject();

   /**
    * Get-method for mInitInfoCommunicatedFlag, which is false on initialization
    * and
    * then becomes true once setInitInfoCommunicatedFlag() is called.
    */
   bool getInitInfoCommunicatedFlag() const { return mInitInfoCommunicatedFlag; }

   /**
    * Get-method for mDataStructuresAllocatedFlag, which is false on
    * initialization and
    * then becomes true once setDataStructuresAllocatedFlag() is called.
    */
   bool getDataStructuresAllocatedFlag() const { return mDataStructuresAllocatedFlag; }

   /**
    * Get-method for mInitialValuesSetFlag, which is false on initialization and
    * then becomes true once setInitialValuesSetFlag() is called.
    */
   bool getInitialValuesSetFlag() const { return mInitialValuesSetFlag; }

#ifdef PV_USE_CUDA
   /**
    * Returns true if the object requires the GPU; false otherwise.
    * HyPerCol will not initialize the GPU unless one of the objects in its
    * hierarchy returns true
    */
   bool isUsingGPU() const { return mUsingGPUFlag; }
#endif // PV_USE_CUDA

  protected:
   BaseObject();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   void setCommunicator(Communicator const *comm);
   virtual void initMessageActionMap() override;

   Response::Status
   respondCommunicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
#ifdef PV_USE_CUDA
   Response::Status respondSetCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);
#endif // PV_USE_CUDA
   Response::Status
   respondAllocateDataStructures(std::shared_ptr<AllocateDataStructuresMessage const> message);
   Response::Status respondInitializeState(std::shared_ptr<InitializeStateMessage const> message);
   Response::Status
   respondCopyInitialStateToGPU(std::shared_ptr<CopyInitialStateToGPUMessage const> message);
   Response::Status respondCleanup(std::shared_ptr<CleanupMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
      return Response::SUCCESS;
   }
#ifdef PV_USE_CUDA
   virtual Response::Status setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);
#endif // PV_USE_CUDA
   virtual Response::Status allocateDataStructures() { return Response::NO_ACTION; }
   virtual Response::Status initializeState(std::shared_ptr<InitializeStateMessage const> message) {
      return Response::NO_ACTION;
   }
   virtual Response::Status copyInitialStateToGPU() { return Response::SUCCESS; }
   virtual Response::Status cleanup() { return Response::NO_ACTION; }

   /**
    * This method sets mInitInfoCommunicatedFlag to true.
    */
   void setInitInfoCommunicatedFlag() { mInitInfoCommunicatedFlag = true; }

   /**
    * This method sets mDataStructuresAllocatedFlag to true.
    */
   void setDataStructuresAllocatedFlag() { mDataStructuresAllocatedFlag = true; }

   /**
    * This method sets the flag returned by getInitialValuesSetFlag to true.
    */
   void setInitialValuesSetFlag() { mInitialValuesSetFlag = true; }

   /**
    * Creates a new BaseObject with the given keyword,
    * and the same name, params, and communicator as the current object.
    * The keyword must have been registered previously with the Factory singleton.
    */
   BaseObject *createSubobject(char const *keyword);

   // Data members
  protected:
   Communicator const *mCommunicator = nullptr;
   bool mInitInfoCommunicatedFlag    = false;
   bool mDataStructuresAllocatedFlag = false;
   bool mInitialValuesSetFlag        = false;
#ifdef PV_USE_CUDA
   bool mUsingGPUFlag              = false;
   PVCuda::CudaDevice *mCudaDevice = nullptr;
#endif // PV_USE_CUDA
}; // class BaseObject

} // namespace PV

#endif /* BASEOBJECT_HPP_ */
