/*
 * PlasticConnTestLayer.hpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#ifndef PLASTICCONNTESTLAYER_HPP_
#define PLASTICCONNTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class PlasticConnTestLayer : public PV::ANNLayer {
  public:
   PlasticConnTestLayer(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double timef, double dt) override;
   virtual int publish(Communicator *comm, double timef) override;

  protected:
   int copyAtoV();
   int setActivitytoGlobalPos();
   int initialize(const char *name, HyPerCol *hc);
}; // end class PlasticConnTestLayer

} // end namespace PV
#endif /* PLASTICCONNTESTLAYER_HPP_ */
