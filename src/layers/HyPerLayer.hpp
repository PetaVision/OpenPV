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
#include "components/ActivityComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/LayerOutputComponent.hpp"
#include "components/LayerUpdateController.hpp"
#include "components/PhaseParam.hpp"
#include "components/PublisherComponent.hpp"
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
   virtual PublisherComponent *createPublisher();
   virtual LayerOutputComponent *createLayerOutput();

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
   void publish(Communicator *comm, double simTime) { mPublisher->publish(comm, simTime); }
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   int waitOnPublish(Communicator *comm) { return mPublisher->waitOnPublish(comm); }

   void updateAllActiveIndices() { mPublisher->updateAllActiveIndices(); }
   void updateActiveIndices() { mPublisher->updateActiveIndices(); }

   /**
    * Returns true if the MPI exchange for the specified delay has finished;
    * false if it is still in process.
    */
   bool isExchangeFinished(int delay = 0) { return mPublisher->isExchangeFinished(delay); }

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
   int getNumDelayLevels() { return mPublisher->getNumDelayLevels(); }

   int increaseDelayLevels(int neededDelay) { return mPublisher->increaseDelayLevels(neededDelay); }

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

   bool getSparseFlag() { return mPublisher->getSparseLayer(); }

   int getPhase() { return mPhaseParam->getPhase(); }

   // implementation of LayerDataInterface interface
   //
   float const *getLayerData(int delay = 0) const { return mPublisher->getLayerData(delay); }

   Publisher *getPublisher() { return mPublisher->getPublisher(); }

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * This routine initializes the ActivityComponent component.
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Deprecated. A virtual function called after the LayerUpdateController updates the state.
    * Provided because before the layer refactoring, a large number of system tests
    * worked by writing a layer subclass and overriding HyPerLayer::updateState().
    * Instead, use a probe or override the relevant component to do the check.
    */
   virtual Response::Status checkUpdateState(double simTime, double deltaTime);

   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   BoundaryConditions *mBoundaryConditions = nullptr;

   LayerUpdateController *mLayerUpdateController = nullptr;

   LayerInputBuffer *mLayerInput = nullptr;

   ActivityComponent *mActivityComponent = nullptr;

   PublisherComponent *mPublisher = nullptr;

   LayerOutputComponent *mLayerOutput = nullptr;

   // timesteps for connections with delays

   std::vector<BaseConnection *> recvConns;

// GPU variables
#ifdef PV_USE_CUDA
  public:
   PVCuda::CudaBuffer *getCudaDatastore() { return mPublisher->getCudaDatastore(); }

   PVCuda::CudaBuffer *getCudaActiveIndices() { return mPublisher->getCudaActiveIndices(); }

   PVCuda::CudaBuffer *getCudaNumActive() { return mPublisher->getCudaNumActive(); }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnDatastore() { return mPublisher->getCudnnDatastore(); }
#endif // PV_USE_CUDNN

   void setAllocCudaDatastore() { mPublisher->setAllocCudaDatastore(); }

   void setAllocCudaActiveIndices() { mPublisher->setAllocCudaActiveIndices(); }

   bool getUpdatedCudaDatastoreFlag() { return mPublisher->getUpdatedCudaDatastoreFlag(); }

   void setUpdatedCudaDatastoreFlag(bool in) { mPublisher->setUpdatedCudaDatastoreFlag(in); }
#endif // PV_USE_CUDA
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
