#ifndef MAXPOOLTESTLAYER_HPP_
#define MAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskTestLayer : public PV::ANNLayer {
  public:
   MaskTestLayer(const char *name, HyPerCol *hc);
   ~MaskTestLayer();

  protected:
   virtual Response::Status updateState(double timef, double dt) override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_maskMethod(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base() {
      maskMethod = nullptr;
      return PV_SUCCESS;
   }

  private:
   char *maskMethod;
};

} /* namespace PV */
#endif
