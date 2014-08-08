/*
 * GTLayer.hpp
 * Author: slundquist
 */

#ifndef GTLAYER_HPP_ 
#define GTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class GTLayer : public PV::ANNLayer{
public:
   GTLayer(const char * name, HyPerCol * hc);
   virtual ~GTLayer();
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
