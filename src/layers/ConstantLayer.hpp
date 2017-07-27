/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#ifndef CONSTANTLAYER_HPP_
#define CONSTANTLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class ConstantLayer : public PV::HyPerLayer {
  public:
   ConstantLayer(const char *name, HyPerCol *hc);
   virtual ~ConstantLayer();
   virtual bool needUpdate(double time, double dt) override;

  protected:
   ConstantLayer();
   int initialize(const char *name, HyPerCol *hc);

   /**
    * List of parameters needed from the ConstantLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief ConstantLayer does not use triggerLayerName.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   /** @} */ // End list of ConstantLayer parameters

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  private:
   int initialize_base();
}; // class ConstantLayer

} /* namespace PV */
#endif
