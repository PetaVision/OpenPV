/*
 * LIF.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 *
 */

#ifndef LIF_HPP_
#define LIF_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class LIF : public HyPerLayer {
  public:
   LIF(const char *name, HyPerCol *hc);
   virtual ~LIF();

  protected:
   LIF();

   int initialize(const char *name, HyPerCol *hc);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;
};

} // namespace PV

#endif /* LIF_HPP_ */
