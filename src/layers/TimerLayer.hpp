/**
 * TimerLayer.hpp
 *
 *  A layer that has no input, output, or data, but does have an update controller
 *  that provides a periodic trigger. It can be used as a trigger layer for another
 *  layer without itself requiring resources.
 */

#ifndef TIMERLAYER_HPP_
#define TIMERLAYER_HPP_

#include "columns/Communicator.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/LayerOutputComponent.hpp"
#include "components/LayerUpdateController.hpp"
#include "components/PhaseParam.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"

namespace PV {

/**
 * A layer class that has no data, input, or output but has an update controller
 * with a timerPeriod parameter
 */
class TimerLayer : public HyPerLayer {
  public:
   TimerLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~TimerLayer();

  protected:
   TimerLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void fillComponentTable() override;
   virtual void initMessageActionMap() override;
   virtual LayerGeometry *createLayerGeometry() override { return nullptr; }
   virtual PhaseParam *createPhaseParam() override { return nullptr; }
   virtual BoundaryConditions *createBoundaryConditions() override { return nullptr; }
   virtual LayerUpdateController *createLayerUpdateController() override;
   virtual LayerInputBuffer *createLayerInput() override { return nullptr; }
   virtual ActivityComponent *createActivityComponent() override { return nullptr; }
   virtual BasePublisherComponent *createPublisher() override { return nullptr; }
   virtual LayerOutputComponent *createLayerOutput() override { return nullptr; }

   /**
    * This routine initializes the ActivityComponent component.
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);
};

} // namespace PV

#endif /* TIMERLAYER_HPP_ */
