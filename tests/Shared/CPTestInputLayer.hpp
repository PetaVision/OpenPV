// CPTestInputLayer
// HyPerLayer subclass that applies a thresholding transfer function.

#ifndef CPTESTINPUTLAYER_HPP__
#define CPTESTINPUTLAYER_HPP__

#include <layers/HyPerLayer.hpp>

namespace PV {

class CPTestInputLayer : public HyPerLayer {
  public:
   CPTestInputLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~CPTestInputLayer();

  protected:
   CPTestInputLayer() {}

   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
