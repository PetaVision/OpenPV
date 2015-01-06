/*
 * RescaleLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef RESCALELAYER_HPP_
#define RESCALELAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class RescaleLayer: public CloneVLayer {
public:
   RescaleLayer(const char * name, HyPerCol * hc);
   virtual ~RescaleLayer();
   virtual int communicateInitInfo();
   //virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();

   float getTargetMax() { return targetMax; }
   float getTargetMin() { return targetMin; }
   float getTargetMean() { return targetMean; }
   float getTargetStd() { return targetMean; }
   char const * getRescaleMethod() { return rescaleMethod; }
protected:
   RescaleLayer();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   void ioParam_targetMax(enum ParamsIOFlag ioFlag);
   void ioParam_targetMin(enum ParamsIOFlag ioFlag);
   void ioParam_targetMean(enum ParamsIOFlag ioFlag);
   void ioParam_targetStd(enum ParamsIOFlag ioFlag);
   void ioParam_rescaleMethod(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();
   //bool checkIfUpdateNeeded();

   // Handled by CloneVLayer
   // char * originalLayerName;
   // HyPerLayer * originalLayer;

protected:
   float targetMax;
   float targetMin;
   float targetMean;
   float targetStd;
   char * rescaleMethod; //can be either maxmin or meanstd
};

}

#endif /* CLONELAYER_HPP_ */
