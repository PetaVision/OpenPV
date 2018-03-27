/*
 * BaseObject.hpp
 *
 *  This is the base class for HyPerCol, layers, connections, probes, and
 *  anything else that the Factory object needs to know about.
 *
 *  All objects in the BaseObject hierarchy should have an associated
 *  instantiating function, with the prototype
 *  BaseObject * createNameOfClass(char const * name, HyPerCol * initData);
 *
 *  Each class's instantiating function should create an object of that class,
 *  with the arguments specifying the object's name and any necessary
 *  initializing data (for most classes, this is the parent HyPerCol.
 *  For HyPerCol, it is the PVInit object).  This way, the class can be
 *  registered with the Factory object by calling
 *  Factory::registerKeyword() with a pointer to the class's instantiating
 *  method.
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef BASEOBJECT_HPP_
#define BASEOBJECT_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointerDataInterface.hpp"
#include "columns/Messages.hpp"
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

class BaseObject : public CheckpointerDataInterface {
  public:
   inline char const *getName() const { return name; }
   // No getParent method because we are refactoring away from having objects
   // having access to their containing HyPerCol.

   inline std::string const &getObjectType() const { return mObjectType; }

   /**
    * Look up the keyword of the params group with the same name as the object.
    */
   char const *lookupKeyword() const;

   /**
    * A method that reads the parameters for the group whose name matches the name of the object.
    * It, along with writeParams(), is a wrapper around ioParams, so that readParams and
    * writeParams automatically run through the same parameters in the same order.
    */
   void readParams() { ioParams(PARAMS_IO_READ, false, false); }

   /**
    * A method that writes the parameters for the group whose name matches the name of the object.
    * It, along with readParams(), is a wrapper around ioParams, so that readParams and writeParams
    * automatically run through the same parameters in the same order.
    */
   void writeParams() { ioParams(PARAMS_IO_WRITE, true, true); }

   /**
    * Method for reading or writing the params from group in the parent HyPerCol's parameters.
    * The group from params is selected using the name of the connection.
    *
    * If ioFlag is set to write, the printHeader and printFooter flags control whether
    * a header and footer for the parameter group is produces. These flags are set to true
    * for layers, connections, and probes; and set to false for weight initializers and
    * normalizers. If ioFlag is set to read, the printHeader and printFooter flags are ignored.
    *
    * Note that ioParams is not virtual.  To add parameters in a derived class, override
    * ioParamFillGroup.
    */
   void ioParams(enum ParamsIOFlag ioFlag, bool printHeader, bool printFooter);

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;
   // TODO: should return enum with values corresponding to PV_SUCCESS, PV_FAILURE, PV_POSTPONE

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
   int setName(char const *name);
   int setParent(HyPerCol *hc);
   virtual void setObjectType();
   void setDescription();

   /**
    * The virtual method for reading parameters from the parent HyPerCol's parameters, and writing
    * to the output params file.
    *
    * Derived classes with additional parameters typically override ioParamsFillGroup to call the
    * base class's ioParamsFillGroup
    * method and then call ioParam_[parametername] for each of their parameters.
    * The ioParam_[parametername] methods should call the parent HyPerCol's ioParamValue() and
    * related methods,
    * to ensure that all parameters that get read also get written to the outputParams-generated
    * file.
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

   Response::Status
   respondCommunicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
#ifdef PV_USE_CUDA
   Response::Status respondSetCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);
#endif // PV_USE_CUDA
   Response::Status respondAllocateData(std::shared_ptr<AllocateDataMessage const> message);
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
   virtual Response::Status initializeState() { return Response::NO_ACTION; }
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override {
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
   char *name = nullptr;
   std::string mObjectType;
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
