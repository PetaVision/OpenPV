/*
 * WTALayer.hpp
 * Author: slundquist
 */

#ifndef WTALAYER_HPP_ 
#define WTALAYER_HPP_ 
#include "ANNLayer.hpp"

namespace PV{

class WTALayer : public PV::HyPerLayer{
public:
   WTALayer(const char * name, HyPerCol * hc);
   virtual ~WTALayer();
   virtual int updateState(double timef, double dt);
   virtual int communicateInitInfo();
   virtual bool activityIsSpiking() { return false; }
protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_binMaxMin(enum ParamsIOFlag ioFlag);
   int allocateV();
   int initializeV();
   virtual int initializeActivity();
private:
   int initialize_base();
   float binMax;
   float binMin;

protected:
   char * originalLayerName;
   HyPerLayer * originalLayer;

}; // class WTALayer

BaseObject * createWTALayer(char const * name, HyPerCol * hc);

}  // namespace PV
#endif 
