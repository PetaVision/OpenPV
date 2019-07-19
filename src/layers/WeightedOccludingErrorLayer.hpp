/*
 * WeightedOccludingErrorLayer.hpp
 *
 *  Created on: Jul 19, 2019 
 *      Author: Jacob Springer
 */

#ifndef WEIGHTEDOCCLUDINGERRORLAYER_HPP__
#define WEIGHTEDOCCLUDINGERRORLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class WeightedOccludingErrorLayer : public HyPerLayer {
  public:
   WeightedOccludingErrorLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~WeightedOccludingErrorLayer();

  protected:
   WeightedOccludingErrorLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
