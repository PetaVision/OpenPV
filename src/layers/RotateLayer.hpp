#ifndef ROTATELAYER_HPP_
#define ROTATELAYER_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {

class RotateLayer : public HyPerLayer {
  public:
   RotateLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~RotateLayer() {}

  protected:
   RotateLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // namespace PV

#endif // ROTATELAYER_HPP_
