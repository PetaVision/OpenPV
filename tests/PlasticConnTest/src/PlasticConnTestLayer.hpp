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

/**
 * A class used by the layers of PlasticConnTest. The layer's activity is
 * set by a PlasticConnTestLayer object: each pixel's activity is the x-coordinate
 * in global restricted space. Although the layer has a LayerInputBuffer, the input
 * is not used when setting the activity.
 */
class PlasticConnTestLayer : public PV::HyPerLayer {
  public:
   PlasticConnTestLayer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
}; // end class PlasticConnTestLayer

} // end namespace PV
#endif /* PLASTICCONNTESTLAYER_HPP_ */
