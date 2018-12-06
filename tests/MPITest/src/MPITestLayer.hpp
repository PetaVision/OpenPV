/*
 * MPITestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef MPITESTLAYER_HPP_
#define MPITESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class MPITestLayer : public PV::HyPerLayer {
  public:
   MPITestLayer(const char *name, PVParams *params, Communicator *comm);

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual ActivityComponent *createActivityComponent();
};

} /* namespace PV */
#endif /* MPITESTLAYER_HPP_ */
