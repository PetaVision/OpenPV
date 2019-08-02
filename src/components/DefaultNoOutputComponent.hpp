/*
 * DefaultNoOutputComponent.hpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#ifndef DEFAULTNOOUTPUTCOMPONENT_HPP_
#define DEFAULTNOOUTPUTCOMPONENT_HPP_

#include "components/LayerOutputComponent.hpp"

namespace PV {

/**
 * A subclass of LayerOutputComponent where the default writeStep is -1 (never write output)
 * as opposed to 0 (write every timestep). The writeStep parameter is still read.
 */
class DefaultNoOutputComponent : public LayerOutputComponent {
  public:
   DefaultNoOutputComponent(char const *name, PVParams *params, Communicator const *comm);
   virtual ~DefaultNoOutputComponent();

  protected:
   DefaultNoOutputComponent();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;
}; // class DefaultNoOutputComponent

} // namespace PV

#endif // DEFAULTNOOUTPUTCOMPONENT_HPP_
