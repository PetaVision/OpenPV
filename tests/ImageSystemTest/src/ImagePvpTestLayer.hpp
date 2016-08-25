/*
 * ImagePvpTestLayer.hpp
 * Author: slundquist
 */

#pragma once
#include <layers/PvpLayer.hpp>

namespace PV{

class ImagePvpTestLayer : public PV::PvpLayer{
public:
   ImagePvpTestLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double time, double dt);
};


}
