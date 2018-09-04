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

class HyPerCol;

/**
 * The base class for layers, connections, probes, and components of those
 * objects. Provides common interfaces for CommunicateInitInfo, AllocateDataStructures,
 * SetInitialValues messages, and a few others.
 */
class BaseObject : public ParamsInterface {
  public:
   // No getParent method because we are refactoring away from having objects
   // having access to their containing HyPerCol.

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
   int initialize(char const *name, HyPerCol *hc);
   void setParent(HyPerCol *hc);
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

   // Data members
  protected:
   // TODO: eliminate HyPerCol argument to constructor in favor of PVParams argument
   HyPerCol *parent                  = nullptr;
   bool mInitInfoCommunicatedFlag    = false;
   bool mDataStructuresAllocatedFlag = false;
   bool mInitialValuesSetFlag        = false;
#ifdef PV_USE_CUDA
   bool mUsingGPUFlag              = false;
   PVCuda::CudaDevice *mCudaDevice = nullptr;
#endif // PV_USE_CUDA

  private:
   int initialize_base();
}; // class BaseObject

} // namespace PV

#endif /* BASEOBJECT_HPP_ */
