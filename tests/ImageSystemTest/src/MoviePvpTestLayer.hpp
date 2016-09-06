/*
 * MoviePvpTestLayer.hpp
 * Author: slundquist
 */

#pragma once
#include <layers/PvpLayer.hpp>

namespace PV{

class MoviePvpTestLayer : public PV::PvpLayer{
public:
   MoviePvpTestLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double time, double dt);
};


}  // end namespace PV
