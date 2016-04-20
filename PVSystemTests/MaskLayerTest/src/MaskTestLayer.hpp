#ifndef MAXPOOLTESTLAYER_HPP_ 
#define MAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskTestLayer: public PV::ANNLayer{
public:
	MaskTestLayer(const char* name, HyPerCol * hc);
   ~MaskTestLayer();

protected:
   virtual int updateState(double timef, double dt);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maskMethod(enum ParamsIOFlag ioFlag);

private:
   char* maskMethod;
};

BaseObject * createMaskTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
