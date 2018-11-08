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
   MomentumLCALayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~MomentumLCALayer();

  protected:
   MomentumLCALayer() {}
   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
