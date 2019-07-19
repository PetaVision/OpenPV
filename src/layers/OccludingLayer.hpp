/*
 * OccludingLayer.hpp
 *
 *  Created on: Jul 19, 2019
 *      Author: Jacob Springer
 */

#ifndef OCCLUDINGLAYER_HPP__
#define OCCLUDINGLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class OccludingLayer : public HyPerLayer {
  public:
   OccludingLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~OccludingLayer();

  protected:
   OccludingLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
