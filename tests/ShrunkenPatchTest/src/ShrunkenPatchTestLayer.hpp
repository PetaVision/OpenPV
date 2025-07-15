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
   ShrunkenPatchTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
};

} /* namespace PV */
#endif /* SHRUNKENPATCHTESTLAYER_HPP_ */
