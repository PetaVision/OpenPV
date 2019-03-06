/*
 * BasePublisherComponent.hpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#ifndef PUBLISHERCOMPONENT_HPP_
#define PUBLISHERCOMPONENT_HPP_

#include "columns/BaseObject.hpp"

#include "columns/Publisher.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerUpdateController.hpp"
#include "utils/Timer.hpp"

#define MAX_F_DELAY 1001 // 21 // can have 0:MAX_F_DELAY-1 buffers of delay

namespace PV {

/**
 * A component to hold a HyPerLayer's activity ring buffer and to publish the
 * layer's activity to be used by delivery objects, probes, etc.
 * During initialization, indicate how many timesteps in the past will be needed
 * by calling the increaseDelayLevels(int) method, with the needed number of
 * timesteps. The activity is retrieved by calling the getLayerData(int) method,
 * where the argument is the number of timesteps in the past, and the default
 * is zero.
 */
class BasePublisherComponent : public BaseObject {
  public:
   BasePublisherComponent(char const *name, PVParams *params, Communicator *comm);
   virtual ~BasePublisherComponent();

   /**
    * Call this routine to increase the number of levels in the data store ring buffer.
    * Calls to this routine after the data store has been initialized will have no effect.
    * The routine returns the new value of numDelayLevels
    */
   int increaseDelayLevels(int neededDelay);

   virtual void publish(Communicator *comm, double simTime);

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   int waitOnPublish(Communicator *comm);

   void updateAllActiveIndices();
   void updateActiveIndices();

   /**
    * Returns true if the MPI exchange for the specified delay has finished;
    * false if it is still in process.
    */
   bool isExchangeFinished(int delay = 0);

   /**
    * Returns the activity data for the layer.  This data is in the
    * extended space (with margins).
    */
   float const *getLayerData(int delay = 0) const;

   PVLayerLoc const *getLayerLoc() const { return mActivity->getLayerLoc(); }

   bool getSparseLayer() const { return mSparseLayer; }

   Publisher *getPublisher() { return mPublisher; }

   int getNumDelayLevels() const { return mNumDelayLevels; }

   int getNumExtended() const { return mActivity->getBufferSize(); }
   int getNumExtendedAcrossBatch() const { return mActivity->getBufferSizeAcrossBatch(); }

// get-methods for CudaBuffers
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *getCudaDatastore() { return mCudaDatastore; }

   PVCuda::CudaBuffer *getCudaActiveIndices() { return mCudaActiveIndices; }

   PVCuda::CudaBuffer *getCudaNumActive() { return mCudaNumActive; }

   bool getUpdatedCudaDatastoreFlag() const { return mUpdatedCudaDatastore; }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnDatastore() { return mCudnnDatastore; }
#endif // PV_USE_CUDNN

   void setAllocCudaDatastore() { mAllocCudaDatastore = true; }

   void setAllocCudaActiveIndices() { mAllocCudaActiveIndices = true; }

   void setUpdatedCudaDatastoreFlag(bool in) { mUpdatedCudaDatastore = in; }
#endif // PV_USE_CUDA

  protected:
   BasePublisherComponent();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void initMessageActionMap() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead() override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   Response::Status
   respondLayerAdvanceDataStore(std::shared_ptr<LayerAdvanceDataStoreMessage const> message);

   virtual void advanceDataStore();

   Response::Status respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message);

   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);

#ifdef PV_USE_CUDA
   virtual void allocateCudaBuffers();
#endif // PV_USE_CUDA

  protected:
   bool mSparseLayer = false; // If true, Publisher uses sparse representation.
   // BasePublisherComponent does not provide any mechanism for setting this flag,
   // but subclasses can.

   ActivityBuffer *mActivity                = nullptr;
   BoundaryConditions *mBoundaryConditions  = nullptr;
   LayerUpdateController *mUpdateController = nullptr;
   Publisher *mPublisher                    = nullptr;
   int mNumDelayLevels                      = 1;
   // The number of delay levels. Objects that need layer data with a delay should call
   // the increaseDelayLevels() method.

   Timer *mPublishTimer = nullptr;

#ifdef PV_USE_CUDA
   // OpenCL buffers and their corresponding flags
   bool mAllocCudaDatastore     = false;
   bool mAllocCudaActiveIndices = false;

   PVCuda::CudaBuffer *mCudaDatastore     = nullptr;
   PVCuda::CudaBuffer *mCudaNumActive     = nullptr;
   PVCuda::CudaBuffer *mCudaActiveIndices = nullptr;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *mCudnnDatastore = nullptr;
#endif // PV_USE_CUDNN

   bool mUpdatedCudaDatastore = true;
#endif // PV_USE_CUDA

}; // class BasePublisherComponent

} // namespace PV

#endif // PUBLISHERCOMPONENT_HPP_
