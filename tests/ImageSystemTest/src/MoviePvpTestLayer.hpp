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
   virtual Response::Status updateState(double time, double dt) override;
};

} // end namespace PV

#endif // MOVIEPVPTESTLAYER_HPP_
