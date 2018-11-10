/*
 * MovieTestLayer.hpp
 * Author: slundquist
 */

#ifndef MOVIETESTLAYER_HPP_
#define MOVIETESTLAYER_HPP_
#include <layers/ImageLayer.hpp>

namespace PV {

class MovieTestLayer : public PV::ImageLayer {
  public:
   MovieTestLayer(const char *name, HyPerCol *hc);
   virtual ~MovieTestLayer();
   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // MOVIETESTLAYER_HPP_
