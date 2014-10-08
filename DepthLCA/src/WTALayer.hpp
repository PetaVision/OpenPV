/*
 * WTALayer.hpp
 * Author: slundquist
 */

#ifndef WTALAYER_HPP_ 
#define WTALAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class WTALayer : public PV::HyPerLayer{
public:
   WTALayer(const char * name, HyPerCol * hc);
   virtual ~WTALayer();
   virtual int updateState(double timef, double dt);
   virtual int communicateInitInfo();
protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   int allocateV();
   int initializeV();
   virtual int initializeActivity();
private:
   int initialize_base();

protected:
   char * originalLayerName;
   HyPerLayer * originalLayer;

};

}
#endif 
