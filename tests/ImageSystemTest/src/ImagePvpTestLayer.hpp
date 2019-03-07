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
   ImagePvpTestLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~ImagePvpTestLayer();

  protected:
   ImagePvpTestLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // IMAGEPVPTESTLAYER_HPP_
