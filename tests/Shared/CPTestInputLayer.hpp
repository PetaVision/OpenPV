// CPTestInputLayer
// HyPerLayer subclass that applies a thresholding transfer function.

#ifndef CPTESTINPUTLAYER_HPP__
#define CPTESTINPUTLAYER_HPP__

#include <layers/HyPerLayer.hpp>

namespace PV {

class CPTestInputLayer : public HyPerLayer {
  public:
   CPTestInputLayer(const char *name, HyPerCol *hc);
   virtual ~CPTestInputLayer();

  protected:
   CPTestInputLayer() {}

   int initialize(const char *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
