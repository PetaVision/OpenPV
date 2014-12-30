#ifndef IMAGEOFFSETTESTLAYER_HPP_ 
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/Image.hpp>

namespace PV {

class ImageOffsetTestLayer: public PV::Image{
public:
	ImageOffsetTestLayer(const char* name, HyPerCol * hc);
   virtual double getDeltaUpdateTime();

protected:
   int updateState(double timef, double dt);

private:
};

} /* namespace PV */
#endif
