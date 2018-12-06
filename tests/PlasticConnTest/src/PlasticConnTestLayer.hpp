/*
 * PlasticConnTestLayer.hpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#ifndef PLASTICCONNTESTLAYER_HPP_
#define PLASTICCONNTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class PlasticConnTestLayer : public PV::HyPerLayer {
  public:
   PlasticConnTestLayer(const char *name, PVParams *params, Communicator *comm);

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual ActivityComponent *createActivityComponent();
}; // end class PlasticConnTestLayer

} // end namespace PV
#endif /* PLASTICCONNTESTLAYER_HPP_ */
