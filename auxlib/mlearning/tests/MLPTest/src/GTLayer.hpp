/*
 * GTLayer.hpp
 * Author: slundquist
 */

#ifndef GTLAYER_HPP_ 
#define GTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PVMLearning{

class GTLayer : public PV::ANNLayer{
public:
   GTLayer(const char * name, PV::HyPerCol * hc);
   virtual ~GTLayer();
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
}; // end class GTLayer

PV::BaseObject * createGTLayer(char const * name, PV::HyPerCol * hc);

}  // end namespace PVMLearning

#endif // GTLAYER_HPP_
