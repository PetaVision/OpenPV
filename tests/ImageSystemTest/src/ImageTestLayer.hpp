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
   ImageTestLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual Response::Status checkUpdateState(double time, double dt) override;
};
}

#endif // IMAGETESTLAYER_HPP_
