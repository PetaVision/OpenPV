/*
 * ImageTestLayer.hpp
 * Author: slundquist
 */

#ifndef IMAGETESTLAYER_HPP_
#define IMAGETESTLAYER_HPP_
#include <layers/ImageLayer.hpp>

namespace PV {

class ImageTestLayer : public PV::ImageLayer {
  public:
   ImageTestLayer(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;
};
}

#endif // IMAGETESTLAYER_HPP_
