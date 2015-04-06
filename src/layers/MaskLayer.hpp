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

class MaskLayer: public PV::ANNLayer {
public:
   MaskLayer(const char * name, HyPerCol * hc);
   MaskLayer();
   virtual ~MaskLayer();
   virtual int communicateInitInfo();
protected:
   virtual int updateState(double time, double dt);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();
   char* maskLayerName;
   HyPerLayer* maskLayer;

};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
