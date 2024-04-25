#ifndef SCALEXLAYER_HPP_
#define SCALEXLAYER_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {

class ScaleXLayer : public HyPerLayer {
  public:
   ScaleXLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ScaleXLayer() {}

  protected:
   ScaleXLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // namespace PV

#endif // SCALEXLAYER_HPP_
