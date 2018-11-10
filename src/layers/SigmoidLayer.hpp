/*
 * SigmoidLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef SIGMOIDLAYER_HPP_
#define SIGMOIDLAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

// SigmoidLayer can be used to implement Sigmoid junctions between spiking neurons
class SigmoidLayer : public CloneVLayer {
  public:
   SigmoidLayer(const char *name, HyPerCol *hc);
   virtual ~SigmoidLayer();

  protected:
   SigmoidLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual ActivityComponent *createActivityComponent() override;
}; // class SigmoidLayer

} // namespace PV

#endif /* SIGMOIDLAYER_HPP_ */
