/*
 * CIFARGTLayer.hpp
 * Author: slundquist
 */

#ifndef CIFARGTLAYER_HPP_ 
#define CIFARGTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class CIFARGTLayer : public PV::ANNLayer{
public:
   CIFARGTLayer(const char * name, HyPerCol * hc);
   virtual ~CIFARGTLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_inFilename(enum ParamsIOFlag ioFlag);
   void ioParam_StartFrame(enum ParamsIOFlag ioFlag);
private:
   std::string inputString;
   char* inFilename;
   std::ifstream inputfile;
   long startFrame; //Zero indexed
};

}
#endif 
