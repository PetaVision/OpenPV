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
   void readTargetMax(PVParams * params);
   void readTargetMin(PVParams * params);
   void readTargetMean(PVParams * params);
   void readTargetStd(PVParams * params);
   void readRescaleMethod(PVParams * params);
private:
   int initialize_base();
   float targetMax;
   float targetMin;
   float targetMean;
   float targetStd;
   char * rescaleMethod; //can be either maxmin or meanstd
};

}

#endif /* CLONELAYER_HPP_ */
