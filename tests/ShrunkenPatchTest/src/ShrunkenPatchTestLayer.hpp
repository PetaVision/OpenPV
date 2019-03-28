/*
 * ShrunkenPatchTestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef SHRUNKENPATCHTESTLAYER_HPP_
#define SHRUNKENPATCHTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class ShrunkenPatchTestLayer : public PV::HyPerLayer {
  public:
   ShrunkenPatchTestLayer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
};

} /* namespace PV */
#endif /* SHRUNKENPATCHTESTLAYER_HPP_ */
