/*
 * ConstGTLayer.hpp
 * Author: slundquist
 */

#ifndef CONSTGTLAYER_HPP_ 
#define CONSTGTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class ConstGTLayer : public PV::ANNLayer{
public:
   ConstGTLayer(const char * name, HyPerCol * hc);
   virtual ~ConstGTLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_GTVal(enum ParamsIOFlag ioFlag);
private:
   int gtVal;
};

}
#endif 
