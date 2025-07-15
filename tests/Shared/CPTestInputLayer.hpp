// CPTestInputLayer
// HyPerLayer subclass that applies a thresholding transfer function.

#ifndef CPTESTINPUTLAYER_HPP__
#define CPTESTINPUTLAYER_HPP__

#include <layers/HyPerLayer.hpp>

namespace PV {

class CPTestInputLayer : public HyPerLayer {
  public:
   CPTestInputLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~CPTestInputLayer();

  protected:
   CPTestInputLayer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
