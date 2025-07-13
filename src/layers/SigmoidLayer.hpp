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
   SigmoidLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~SigmoidLayer();

  protected:
   SigmoidLayer();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
}; // class SigmoidLayer

} // namespace PV

#endif /* SIGMOIDLAYER_HPP_ */
