/*
 * ImageTestLayer.hpp
 * Author: slundquist
 */

#ifndef IMAGETESTLAYER_HPP_
#define IMAGETESTLAYER_HPP_
#include <layers/Image.hpp>

namespace PV{

class ImageTestLayer : public PV::Image{
public:
   ImageTestLayer(const char * name, HyPerCol * hc);
#ifdef PV_USE_GDAL
   virtual int updateStateWrapper(double time, double dt);
   virtual int updateState(double time, double dt);
#endif // PV_USE_GDAL
};

BaseObject * createImageTestLayer(char const * name, HyPerCol * hc);

}
#endif /* IMAGETESTLAYER_HPP */
