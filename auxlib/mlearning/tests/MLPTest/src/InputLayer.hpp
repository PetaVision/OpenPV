/*
 * InputLayer.hpp
 * Author: slundquist
 */

#ifndef INPUTLAYER_HPP_ 
#define INPUTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PVMLearning{

class InputLayer : public PV::ANNLayer{
public:
   InputLayer(const char * name, PV::HyPerCol * hc);
   virtual ~InputLayer();
   virtual int initialize(const char * name, PV::HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_inFilename(enum ParamsIOFlag ioFlag);
   void ioParam_constantValue(enum ParamsIOFlag ioFlag);
private:
   std::string inputString;
   char* inFilename;
   int numExamples;
   int iterator;
   bool constantValue;
   bool firstRun;
   int iVal;
}; // end class InputLayer

PV::BaseObject * createInputLayer(char const * name, PV::HyPerCol * hc);

}  // end namespace PVMLearning
#endif 
