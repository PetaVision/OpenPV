#ifndef SEGMENTTESTLAYER_HPP_ 
#define SEGMENTTESTLAYER_HPP_

#include <layers/SegmentLayer.hpp>

namespace PV {

class SegmentTestLayer: public PV::SegmentLayer{
public:
	SegmentTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
};

BaseObject * createSegmentTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
