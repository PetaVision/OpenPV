/**
 * BaseLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 */

#ifndef BASELAYER_HPP_
#define BASELAYER_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerOutputComponent.hpp"
#include "components/LayerUpdateController.hpp"
#include "components/PhaseParam.hpp"

namespace PV {

/**
 * The top of the layer hierarchy. BaseLayer cannot be instantiated directly;
 * derived classes should implement a constructor with the signature
 * DerivedLayer(char const *name, PVParams *params, Communicator const *comm)
 *
 * BaseLayer has several components:
 *
 * a LayerGeometry component that defines the dimensions of the layer.
 *
 * a LayerUpdateController component that determines whether the layer acts on a given timestep.
 *
 * an ActivityComponent that updates the ActivityBuffer and any other buffers that the activity
 * depends on (it might do so by calling the components' updateState function members).
 *
 * a PublisherComponent that manages a ring buffer of delays and makes the activity available to
 * other objects in the HyPerCol hierarchy.
 *
 * a BoundaryConditions component that controls how to fill the values in the extended region
 * beyond the restricted region.
 *
 * a PhaseParam component that gives each layer a phase, creating a partial ordering of layer
 * updates within a timestep.
 *
 * a LayerOutputComponent to output the state of the layer.
 *
 * Derived classes may have additional components
 */
class BaseLayer : public ComponentBasedObject {
  public:
   virtual ~BaseLayer();

   void synchronizeMarginWidth(BaseLayer *layer);

   // Public access functions.
   // As much as possible, anything that needs one of these quantities should instead retrieve
   // the appropriate component and use access functions of the component.
   long getNumNeurons() const { return mLayerGeometry->getNumNeurons(); }
   long getNumExtended() const { return mLayerGeometry->getNumExtended(); }
   long getNumNeuronsAllBatches() const { return mLayerGeometry->getNumNeuronsAllBatches(); }
   long getNumExtendedAllBatches() const { return mLayerGeometry->getNumExtendedAllBatches(); }

   long getNumGlobalNeurons() const {
      PVLayerLoc const *loc = getLayerLoc();
      return (long)loc->nxGlobal * (long)loc->nyGlobal * (long)loc->nf;
   }
   long getNumGlobalExtended() const {
      PVLayerLoc const *loc = getLayerLoc();
      int nxGlobalExt       = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
      int nyGlobalExt       = loc->nyGlobal + loc->halo.dn + loc->halo.up;
      return (long)nxGlobalExt * (long)nyGlobalExt * (long)loc->nf;
   }

   // Eventually, anything that calls one of getLayerLoc should retrieve
   // the LayerGeometry component, and getLayerLoc() can be removed from BaseLayer.
   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }

  protected:
   BaseLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void initMessageActionMap() override;
   virtual void fillComponentTable() override;
   virtual LayerGeometry *createLayerGeometry();
   virtual PhaseParam *createPhaseParam();
   virtual LayerUpdateController *createLayerUpdateController();
   virtual ActivityComponent *createActivityComponent();
   virtual BasePublisherComponent *createPublisher();
   virtual BoundaryConditions *createBoundaryConditions();
   virtual LayerOutputComponent *createLayerOutput();

   /**
    * The function that calls all ioParam_[parameter name] functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   Response::Status respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
#endif // PV_USE_CUDA

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

   // Called by respondLayerCopyFromGpu() if the layer's phase matches the phase in the message
   virtual Response::Status
   layerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);

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
    * worked by writing a layer subclass and overriding BaseLayer::updateState().
    * Instead, use a probe or override the relevant component to do the check.
    */
   virtual Response::Status checkUpdateState(double simTime, double deltaTime);

   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);

   // Data members
  protected:
   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   LayerUpdateController *mLayerUpdateController = nullptr;

   ActivityComponent *mActivityComponent = nullptr;

   BasePublisherComponent *mPublisher = nullptr;
   
   BoundaryConditions *mBoundaryConditions = nullptr;

   LayerOutputComponent *mLayerOutput = nullptr;
};

} // namespace PV

#endif /* BASELAYER_HPP_ */
