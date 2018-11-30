/**
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include "checkpointing/CheckpointableFileStream.hpp"
#include "columns/Communicator.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/Publisher.hpp"
#include "columns/Random.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/LayerUpdateController.hpp"
#include "components/PhaseParam.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "io/fileio.hpp"
#include "utils/Timer.hpp"

#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_CUDA
#include <arch/cuda/CudaBuffer.hpp>
#include <arch/cuda/CudaKernel.hpp>
#include <arch/cuda/CudaTimer.hpp>
#endif // PV_USE_CUDA

#include <vector>

// default constants
#define HYPERLAYER_FEEDBACK_DELAY 1
#define HYPERLAYER_FEEDFORWARD_DELAY 0

namespace PV {

class BaseConnection;

class HyPerLayer : public ComponentBasedObject {

  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   // The dataType param was marked obsolete Mar 29, 2018.
   /** @brief dataType: no longer used. */
   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeStep: Specifies how often to output a pvp file for this layer
    * @details Defaults to every timestep. -1 specifies not to write at all.
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWriteTime: Specifies the first timestep to start outputing pvp files
    */
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief sparseLayer: Specifies if the layer should be considered sparse for optimization and
    * output
    */
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag);
   /** @} */

  private:
   int initialize_base();

  protected:
   // only subclasses can be constructed directly
   HyPerLayer();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual void initMessageActionMap() override;
   virtual void createComponentTable(char const *description) override;
   virtual LayerGeometry *createLayerGeometry();
   virtual PhaseParam *createPhaseParam();
   virtual BoundaryConditions *createBoundaryConditions();
   virtual LayerUpdateController *createLayerUpdateController();
   virtual LayerInputBuffer *createLayerInput();
   virtual ActivityComponent *createActivityComponent();

   void addPublisher();

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readDelaysFromCheckpoint(Checkpointer *checkpointer);
#ifdef PV_USE_CUDA
   virtual Response::Status setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);
   virtual Response::Status copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

   void updateNBands(int numCalls);

   virtual Response::Status processCheckpointRead() override;

  public:
   HyPerLayer(const char *name, PVParams *params, Communicator *comm);
   virtual double getTimeScale(int batchIdx) { return -1.0; };

  protected:
   /**
    * The function that calls all ioParam functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void freeActivityCube();

  public:
   virtual ~HyPerLayer();

   void synchronizeMarginWidth(HyPerLayer *layer);

   Response::Status respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message);
   Response::Status respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);
   Response::Status
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);
   Response::Status
   respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);
   Response::Status respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message);
#ifdef PV_USE_CUDA
   Response::Status respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);
#endif // PV_USE_CUDA
   Response::Status
   respondLayerAdvanceDataStore(std::shared_ptr<LayerAdvanceDataStoreMessage const> message);
   Response::Status respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message);
   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);
   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   virtual int publish(Communicator *comm, double simTime);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   virtual int waitOnPublish(Communicator *comm);

   virtual void updateAllActiveIndices();
   void updateActiveIndices();
   int resetBuffer(float *buf, int numItems);

   virtual Response::Status outputState(double timestamp, double deltaTime);
   virtual int writeActivity(double timed);
   virtual int writeActivitySparse(double timed);

   /**
    * Returns true if the MPI exchange for the specified delay has finished;
    * false if it is still in process.
    */
   bool isExchangeFinished(int delay = 0);

   // Public access functions:

   int getNumNeurons() const { return mLayerGeometry->getNumNeurons(); }
   int getNumExtended() const { return mLayerGeometry->getNumExtended(); }
   int getNumNeuronsAllBatches() const { return mLayerGeometry->getNumNeuronsAllBatches(); }
   int getNumExtendedAllBatches() const { return mLayerGeometry->getNumExtendedAllBatches(); }

   int getNumGlobalNeurons() {
      const PVLayerLoc *loc = getLayerLoc();
      return loc->nxGlobal * loc->nyGlobal * loc->nf;
   }
   int getNumGlobalExtended() {
      const PVLayerLoc *loc = getLayerLoc();
      return (loc->nxGlobal + loc->halo.lt + loc->halo.rt)
             * (loc->nyGlobal + loc->halo.dn + loc->halo.up) * loc->nf;
   }
   int getNumDelayLevels() { return numDelayLevels; }

   int increaseDelayLevels(int neededDelay);

   float const *getV() const {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getBufferData();
   }
   float *getV() {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getReadWritePointer();
   }
   int getNumChannels() { return mLayerInput->getNumChannels(); }

   // Eventually, anything that calls one of getXScale, getYScale, or getLayerLoc should retrieve
   // the LayerGeometry component, and these get-methods can be removed from HyPerLayer.
   int getXScale() const { return mLayerGeometry->getXScale(); }
   int getYScale() const { return mLayerGeometry->getYScale(); }
   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }

   bool getSparseFlag() { return this->sparseLayer; }

   int getPhase() { return mPhaseParam->getPhase(); }

   // implementation of LayerDataInterface interface
   //
   const float *getLayerData(int delay = 0);

   Publisher *getPublisher() { return publisher; }

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message);

   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * This routine initializes the InternalStateBuffer and ActivityBuffer components.
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   int openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   /**
    * Deprecated. A virtual function called after the LayerUpdateController updates the state.
    * Provided because of the large number of system tests written before the layer refactoring
    * that worked by writing a layer subclass and overriding HyPerLayer::updateState().
    */
   virtual Response::Status checkUpdateState(double simTime, double deltaTime);

   Publisher *publisher = nullptr;

   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   BoundaryConditions *mBoundaryConditions = nullptr;

   LayerUpdateController *mLayerUpdateController = nullptr;

   LayerInputBuffer *mLayerInput = nullptr;

   ActivityComponent *mActivityComponent = nullptr;

   int numDelayLevels; // The number of timesteps in the datastore ring buffer to store older
   // timesteps for connections with delays

   double initialWriteTime                      = 0.0; // time of next output
   double writeTime                             = 0.0; // time of next output
   double writeStep                             = 0.0; // output time interval
   CheckpointableFileStream *mOutputStateStream = nullptr; // activity generated by outputState

   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
   int writeActivityCalls; // Number of calls to writeActivity (written to nbands in the header of
   // the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the
   // header of the a%d.pvp file)

   unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves
   // {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use
   // by this layer

   std::vector<BaseConnection *> recvConns;

   bool mHasUpdated = false;

// GPU variables
#ifdef PV_USE_CUDA
  public:
   PVCuda::CudaBuffer *getDeviceDatastore() { return d_Datastore; }

   PVCuda::CudaBuffer *getDeviceActiveIndices() { return d_ActiveIndices; }

   PVCuda::CudaBuffer *getDeviceNumActive() { return d_numActive; }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnDatastore() { return cudnn_Datastore; }
#endif // PV_USE_CUDNN

   void setAllocDeviceDatastore() { allocDeviceDatastore = true; }

   void setAllocDeviceActiveIndices() { allocDeviceActiveIndices = true; }

   bool getUpdatedDeviceDatastoreFlag() { return updatedDeviceDatastore; }

   void setUpdatedDeviceDatastoreFlag(bool in) { updatedDeviceDatastore = in; }

  protected:
   virtual int allocateDeviceBuffers();
   // OpenCL buffers and their corresponding flags
   //

   bool allocDeviceDatastore;
   bool allocDeviceActiveIndices;
   bool updatedDeviceDatastore;

   PVCuda::CudaBuffer *d_Datastore;
   PVCuda::CudaBuffer *d_numActive;
   PVCuda::CudaBuffer *d_ActiveIndices;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_Datastore;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

  protected:
   Timer *publish_timer;
   Timer *io_timer;
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
