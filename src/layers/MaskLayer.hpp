/*
 * MaskLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MASKLAYER_HPP_
#define MASKLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class MaskLayer : public PV::ANNLayer {
  public:
   MaskLayer(const char *name, HyPerCol *hc);
   MaskLayer();
   virtual ~MaskLayer();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   virtual int updateState(double time, double dt) override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_maskMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_featureIdxs(enum ParamsIOFlag ioFlag);
   char *maskMethod;
   char *maskLayerName;
   int *features;
   int numSpecifiedFeatures;
   HyPerLayer *maskLayer;

  private:
   int initialize_base();

}; // class MaskLayer

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
