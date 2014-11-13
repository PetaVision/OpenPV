/*
 * HeliTileGTLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef HELITILEGTLAYER_HPP_
#define HELITILEGTLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include "HeliTileMovie.hpp"

namespace PV {

class HeliTileGTLayer: public PV::ANNLayer {
public:
   HeliTileGTLayer(const char * name, HyPerCol * hc);
   virtual ~HeliTileGTLayer();
   virtual int communicateInitInfo();
protected:
   HeliTileGTLayer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InputTileLayer(enum ParamsIOFlag ioFlag);

   virtual int updateState(double time, double dt);
private:
   int initialize_base();
   char * inputTileLayer;
   HeliTileMovie* inputLayer;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
