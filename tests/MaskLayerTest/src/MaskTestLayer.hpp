#ifndef MASKTESTLAYER_HPP_
#define MASKTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskTestLayer : public PV::ANNLayer {
  public:
   MaskTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   ~MaskTestLayer();

  protected:
   virtual Response::Status checkUpdateState(double timef, double dt) override;
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void ioParam_maskMethod(ParamsIOSwitch ioSwitch);

  private:
   std::string mMaskMethod;
};

} /* namespace PV */
#endif // MASKTESTLAYER_HPP_
