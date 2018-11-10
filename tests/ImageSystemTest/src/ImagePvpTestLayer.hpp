/*
 * ImagePvpTestLayer.hpp
 * Author: slundquist
 */

#ifndef IMAGEPVPTESTLAYER_HPP_
#define IMAGEPVPTESTLAYER_HPP_
#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpTestLayer : public PvpLayer {
  public:
   ImagePvpTestLayer(char const *name, HyPerCol *hc);
   virtual ~ImagePvpTestLayer();

  protected:
   ImagePvpTestLayer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // IMAGEPVPTESTLAYER_HPP_
