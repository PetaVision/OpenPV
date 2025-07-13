/*
 * ANNLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP__
#define ANNLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * Subclass that applies a thresholding transfer function
 */
class ANNLayer : public HyPerLayer {
  public:
   ANNLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~ANNLayer();

  protected:
   ANNLayer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // ANNLAYER_HPP_
