/*
 * FirmThresholdCostLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef FIRMTHRESHOLDCOSTLAYER_HPP__
#define FIRMTHRESHOLDCOSTLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * Subclass that applies a thresholding transfer function
 */
class FirmThresholdCostLayer : public HyPerLayer {
  public:
   FirmThresholdCostLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FirmThresholdCostLayer();

  protected:
   FirmThresholdCostLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // FIRMTHRESHOLDCOSTLAYER_HPP_
