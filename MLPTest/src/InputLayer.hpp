/*
 * InputLayer.hpp
 * Author: slundquist
 */

#ifndef INPUTLAYER_HPP_ 
#define INPUTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class InputLayer : public PV::ANNLayer{
public:
   InputLayer(const char * name, HyPerCol * hc);
   virtual ~InputLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_inFilename(enum ParamsIOFlag ioFlag);
private:
   std::string inputString;
   char* inFilename;
   int numExamples;
   int iterator;
};

}
#endif 
