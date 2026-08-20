#ifndef GAUSSIANNOISELAYER_HPP__
#define GAUSSIANNOISELAYER_HPP__

#include "BaseLayer.hpp"

namespace PV {

class GaussianNoiseLayer : public BaseLayer {

  public:
   GaussianNoiseLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~GaussianNoiseLayer();

  protected:
   GaussianNoiseLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // GAUSSIANNOISELAYER_HPP__
