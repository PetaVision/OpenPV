/*
 * ISTALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef ISTALAYER_HPP__
#define ISTALAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class ISTALayer : public HyPerLayer {
  public:
   ISTALayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ISTALayer();

  protected:
   ISTALayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif /* ISTALAYER_HPP_ */
