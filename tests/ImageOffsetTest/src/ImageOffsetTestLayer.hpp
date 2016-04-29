#ifndef IMAGEOFFSETTESTLAYER_HPP_ 
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/Image.hpp>
#include <cMakeHeader.h>

namespace PV {

class ImageOffsetTestLayer: public PV::Image{
public:
   ImageOffsetTestLayer(const char* name, HyPerCol * hc);
#ifdef PV_USE_GDAL
   virtual double getDeltaUpdateTime();

protected:
   int updateState(double timef, double dt);
#endif // PV_USE_GDAL

private:
};

BaseObject * createImageOffsetTestLayer(const char * name, HyPerCol * hc);

} /* namespace PV */
#endif
