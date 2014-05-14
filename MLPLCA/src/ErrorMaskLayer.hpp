/*
 * ErrorMaskLayer.hpp
 * Author: slundquist
 */

#ifndef ERRORMASKLAYER_HPP_ 
#define ERRORMASKLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class ErrorMaskLayer : public PV::ANNLayer{
public:
   ErrorMaskLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_ErrThresh(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();
   float errThresh;
};

}
#endif 
