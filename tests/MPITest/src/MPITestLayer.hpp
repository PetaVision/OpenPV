/*
 * MPITestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef MPITESTLAYER_HPP_
#define MPITESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MPITestLayer : public PV::ANNLayer {
  public:
   MPITestLayer(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double time, double dt) override;
   virtual int publish(Communicator *comm, double timed) override;
   int setVtoGlobalPos();
   int setActivitytoGlobalPos();

  private:
   int initialize(const char *name, HyPerCol *hc);
};

} /* namespace PV */
#endif /* MPITESTLAYER_HPP_ */
