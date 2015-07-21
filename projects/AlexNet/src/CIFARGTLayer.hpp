/*
 * CIFARGTLayer.hpp
 * Author: slundquist
 */

#ifndef CIFARGTLAYER_HPP_ 
#define CIFARGTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>
#include <layers/Image.hpp>

namespace PV{

class CIFARGTLayer : public PV::ANNLayer{
public:
   CIFARGTLayer(const char * name, HyPerCol * hc);
   virtual ~CIFARGTLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   virtual int communicateInitInfo();

protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   //virtual void ioParam_inFilename(enum ParamsIOFlag ioFlag);
   //virtual void ioParam_StartFrame(enum ParamsIOFlag ioFlag);
   virtual void ioParam_NegativeGt(enum ParamsIOFlag ioFlag);
   //virtual void ioParam_constantValue(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ImageLayerName(enum ParamsIOFlag ioFlag);
private:
   std::string inputString;
   char* imageLayerName;
   Image* imageLayer;
   char* inFilename;
   //std::ifstream inputfile;
   //long startFrame; //Zero indexed
   bool negativeGt;
   //bool firstRun;
   //bool constantValue;
   int iVal;
};

}
#endif 
