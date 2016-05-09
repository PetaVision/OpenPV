/*
 * RescaleLayer.cpp
 * Rescale layer is a cloneVLayer, grabs activity from orig layer and rescales it
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
   virtual int allocateV();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();

   float getTargetMax() { return targetMax; }
   float getTargetMin() { return targetMin; }
   float getTargetMean() { return targetMean; }
   float getTargetStd() { return targetStd; }
   float getL2PatchSize() { return patchSize; }
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
   void ioParam_patchSize(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();

   // Handled by CloneVLayer
   // char * originalLayerName;
   // HyPerLayer * originalLayer;

protected:
   float targetMax;
   float targetMin;
   float targetMean;
   float targetStd;
   char * rescaleMethod; //can be either maxmin or meanstd
   int patchSize;
}; // class RescaleLayer

BaseObject * createRescaleLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* CLONELAYER_HPP_ */
