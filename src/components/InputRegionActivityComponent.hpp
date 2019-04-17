/*
 * InputRegionActivityComponent.hpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#ifndef INPUTREGIONACTIVITYCOMPONENT_HPP_
#define INPUTREGIONACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"

namespace PV {

/**
 * InputRegionActivityComponent is the ActivityComponent for InputRegionLayer.
 * It uses the InputRegionActivityBuffer class for its ActivityBuffer.
 */
class InputRegionActivityComponent : public ActivityComponent {
  protected:
   /**
    * @brief updateGpu: InputRegionActivityComponent always sets this flag to false.
    */
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag) override;

  public:
   InputRegionActivityComponent(const char *name, PVParams *params, Communicator const *comm);
   virtual ~InputRegionActivityComponent();

  protected:
   InputRegionActivityComponent();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual ActivityBuffer *createActivity() override;

   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

}; // class InputRegionActivityComponent

} // namespace PV

#endif /* INPUTREGIONACTIVITYCOMPONENT_HPP_ */
