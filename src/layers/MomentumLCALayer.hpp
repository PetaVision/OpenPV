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
   MomentumLCALayer(const char *name, HyPerCol *hc);
   virtual ~MomentumLCALayer();

  protected:
   MomentumLCALayer() {}

   int initialize(const char *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
