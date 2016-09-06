/*
 * MovieTestLayer.hpp
 * Author: slundquist
 */

#pragma once
#include <layers/ImageLayer.hpp>

namespace PV{

class MovieTestLayer : public PV::ImageLayer {
public:
   MovieTestLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double time, double dt);
};


}
