/*
 * MomentumLCALayer.hpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#ifndef MOMENTUMLCALAYER_HPP__
#define MOMENTUMLCALAYER_HPP__

#include "HyPerLCALayer.hpp"

namespace PV {

class MomentumLCALayer : public HyPerLCALayer {
  public:
   MomentumLCALayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~MomentumLCALayer();

  protected:
   MomentumLCALayer() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
