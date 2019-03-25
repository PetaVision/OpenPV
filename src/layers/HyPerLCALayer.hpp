/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCALAYER_HPP__
#define HYPERLCALAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class HyPerLCALayer : public HyPerLayer {
  public:
   HyPerLCALayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~HyPerLCALayer();

  protected:
   HyPerLCALayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
