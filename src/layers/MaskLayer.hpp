/*
 * MaskLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MASKLAYER_HPP_
#define MASKLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class MaskLayer : public HyPerLayer {
  public:
   MaskLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~MaskLayer();

  protected:
   MaskLayer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV
#endif /* MASKLAYER_HPP_ */
