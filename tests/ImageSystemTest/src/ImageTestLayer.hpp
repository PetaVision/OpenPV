/*
 * ImageTestLayer.hpp
 * Author: slundquist
 */

#pragma once
#include <layers/ImageLayer.hpp>

namespace PV{

class ImageTestLayer : public PV::ImageLayer {
public:
   ImageTestLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double time, double dt);
};


}
