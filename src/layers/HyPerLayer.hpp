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

#include "bindings/InteractionMessages.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/LayerOutputComponent.hpp"
#include "components/LayerUpdateController.hpp"
#include "components/PhaseParam.hpp"

namespace PV {

/**
 * The top of the layer hierarchy. HyPerLayer has several components:
 *
 * a LayerGeometry component that defines the dimensions of the layer.
 *
 * a LayerUpdateController component that determines whether the layer
 * acts on a given timestep.
 *
 * a LayerInputBuffer component that receives synaptic input from a connection.
 *
 * an ActivityComponent that uses the contents of the LayerInputBuffer to
 * maintain the ActivityBuffer within the ActivityComponent.
 *
 * a PublisherComponent which manages a ring buffer of delays and makes the
 * activity available to other objects in the HyPerCol hierarchy.
 *
 * a PhaseParam component which gives each layer a phase, creating a partial ordering
 * of layer updates within a timestep.
 *
 * a LayerOutputComponent to output the state of the layer.
 *
 * Subclasses may have additional components, or may skip some of these components
 * (e.g. InputLayer does not use LayerInputBuffer).
 */
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

  public:
   HyPerLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~HyPerLayer();

   void synchronizeMarginWidth(HyPerLayer *layer);

   // Public access functions.
   // As much as possible, anything that needs one of these quantities should instead retrieve
   // the appropriate component and use access functions of the component.
   int getNumNeurons() const { return mLayerGeometry->getNumNeurons(); }
   int getNumExtended() const { return mLayerGeometry->getNumExtended(); }
   int getNumNeuronsAllBatches() const { return mLayerGeometry->getNumNeuronsAllBatches(); }
   int getNumExtendedAllBatches() const { return mLayerGeometry->getNumExtendedAllBatches(); }

   int getNumGlobalNeurons() const {
      return getLayerLoc()->nxGlobal * getLayerLoc()->nyGlobal * getLayerLoc()->nf;
   }
   int getNumGlobalExtended() const {
      const PVLayerLoc *loc = getLayerLoc();
      return (loc->nxGlobal + loc->halo.lt + loc->halo.rt)
             * (loc->nyGlobal + loc->halo.dn + loc->halo.up) * loc->nf;
   }

   float const *getV() const {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getBufferData();
   }
   float *getV() {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getReadWritePointer();
   }

   // Eventually, anything that calls one of getLayerLoc should retrieve
   // the LayerGeometry component, and getLayerLoc() can be removed from HyPerLayer.
   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }

  protected:
   HyPerLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void initMessageActionMap() override;
   virtual void fillComponentTable() override;
   virtual LayerGeometry *createLayerGeometry();
   virtual PhaseParam *createPhaseParam();
   virtual BoundaryConditions *createBoundaryConditions();
   virtual LayerUpdateController *createLayerUpdateController();
   virtual LayerInputBuffer *createLayerInput();
   virtual ActivityComponent *createActivityComponent();
   virtual BasePublisherComponent *createPublisher();
   virtual LayerOutputComponent *createLayerOutput();

   /**
    * The function that calls all ioParam_[parameter name] functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   Response::Status respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
#endif // PV_USE_CUDA

   virtual Response::Status allocateDataStructures() override;

   Response::Status respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message);

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * This routine initializes the ActivityComponent component.
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   Response::Status
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);
#ifdef PV_USE_CUDA
   Response::Status respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);
   virtual Response::Status copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

   Response::Status
   respondLayerAdvanceDataStore(std::shared_ptr<LayerAdvanceDataStoreMessage const> message);
   Response::Status respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message);
   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status
   respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);
   Response::Status respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message);

   /**
    * Deprecated. A virtual function called after the LayerUpdateController updates the state.
    * Provided because before the layer refactoring, a large number of system tests
    * worked by writing a layer subclass and overriding HyPerLayer::updateState().
    * Instead, use a probe or override the relevant component to do the check.
    */
   virtual Response::Status checkUpdateState(double simTime, double deltaTime);

   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);

   Response::Status
   respondLayerGetActivity(std::shared_ptr<LayerGetActivityMessage const> message);
   Response::Status
   respondLayerSetInternalState(std::shared_ptr<LayerSetInternalStateMessage const> message);
   Response::Status
   respondLayerGetInternalState(std::shared_ptr<LayerGetInternalStateMessage const> message);
   Response::Status
   respondLayerGetShape(std::shared_ptr<LayerGetShapeMessage const> message);




   // Data members
  protected:
   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   BoundaryConditions *mBoundaryConditions = nullptr;

   LayerUpdateController *mLayerUpdateController = nullptr;

   LayerInputBuffer *mLayerInput = nullptr;

   ActivityComponent *mActivityComponent = nullptr;

   BasePublisherComponent *mPublisher = nullptr;

   LayerOutputComponent *mLayerOutput = nullptr;
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
