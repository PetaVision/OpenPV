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
   ANNLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~ANNLayer();

  protected:
   ANNLayer() {}

   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // ANNLAYER_HPP_
