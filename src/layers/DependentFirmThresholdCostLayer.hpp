/*
 * DependentFirmThresholdCostLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef DEPENDENTFIRMTHRESHOLDCOSTLAYER_HPP__
#define DEPENDENTFIRMTHRESHOLDCOSTLAYER_HPP__

#include "FirmThresholdCostLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

/**
 * Subclass that applies a thresholding transfer function
 */
class DependentFirmThresholdCostLayer : public FirmThresholdCostLayer {
  public:
   DependentFirmThresholdCostLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~DependentFirmThresholdCostLayer();

  protected:
   DependentFirmThresholdCostLayer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void fillComponentTable() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
};

} // end namespace PV

#endif // DEPENDENTFIRMTHRESHOLDCOSTLAYER_HPP_
