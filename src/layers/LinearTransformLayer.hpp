#ifndef LINEARTRANSFORMLAYER_HPP_
#define LINEARTRANSFORMLAYER_HPP_

#include "components/ActivityComponent.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

/**
 * A layer class to handle a few different types of linear transformation.
 * Params group keywords RotateLayer, ScaleXLayer, and ScaleYLayer are handled by this class.
 * The createActivityComponent() function member reads the group keyword from params and
 * selects the appropriate activity buffer type.
 */
class LinearTransformLayer : public HyPerLayer {
  public:
   LinearTransformLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~LinearTransformLayer() {}

  protected:
   LinearTransformLayer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // namespace PV

#endif // LINEARTRANSFORMLAYER_HPP_
