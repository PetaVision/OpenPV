/*
 * SpikingIntegrator.hpp
 *
 *  Created on: Sep 12, 2018
 *      Author: twatkins
 */

#ifndef SPIKINGINTEGRATOR_HPP_
#define SPIKINGINTEGRATOR_HPP_

#include "ANNLayer.hpp"
#include <cmath>

namespace PV {

class SpikingIntegrator : public HyPerLayer {
   // Member functions
  public:
   SpikingIntegrator(const char *name, PVParams *params, Communicator const *comm);
   virtual ~SpikingIntegrator();

  protected:
   SpikingIntegrator();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual ActivityComponent *createActivityComponent();

  private:
  protected:
}; // class SpikingIntegrator

} /* namespace PV */
#endif /* SPIKINGINTEGRATOR_HPP_ */
