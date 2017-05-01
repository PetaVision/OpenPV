/*
 * MoviePvpTestLayer.hpp
 * Author: slundquist
 */

#ifndef MOVIEPVPTESTLAYER_HPP_
#define MOVIEPVPTESTLAYER_HPP_
#include <layers/PvpLayer.hpp>

namespace PV {

class MoviePvpTestLayer : public PV::PvpLayer {
  public:
   MoviePvpTestLayer(const char *name, HyPerCol *hc);
   virtual int updateState(double time, double dt);
};

} // end namespace PV

#endif // MOVIEPVPTESTLAYER_HPP_
