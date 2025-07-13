#ifndef GAUSSIANNOISELAYER_HPP__
#define GAUSSIANNOISELAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class GaussianNoiseLayer : public HyPerLayer {

  public:
   GaussianNoiseLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~GaussianNoiseLayer();

  protected:
   GaussianNoiseLayer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // GAUSSIANNOISELAYER_HPP__
