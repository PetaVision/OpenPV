/*
 * RescaleLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef RESCALELAYER_HPP_
#define RESCALELAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class RescaleLayer: public HyPerLayer {
public:
   RescaleLayer(const char * name, HyPerCol * hc, HyPerLayer * clone);
   virtual ~RescaleLayer();
   virtual int updateState(double timef, double dt);
   HyPerLayer * sourceLayer;
   virtual int setActivity();
protected:
   RescaleLayer();
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * clone);
   int setParams(PVParams * params);
   void readSetMax(PVParams * params);
   void readSetMin(PVParams * params);
private:
   int initialize_base();
   float setMax;
   float setMin;
};

}

#endif /* CLONELAYER_HPP_ */
