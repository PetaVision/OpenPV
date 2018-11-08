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
   MoviePvpTestLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~MoviePvpTestLayer();
   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // MOVIEPVPTESTLAYER_HPP_
