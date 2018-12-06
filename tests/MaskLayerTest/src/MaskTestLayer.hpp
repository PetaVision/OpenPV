#ifndef MAXPOOLTESTLAYER_HPP_
#define MAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskTestLayer : public PV::ANNLayer {
  public:
   MaskTestLayer(const char *name, PVParams *params, Communicator *comm);
   ~MaskTestLayer();

  protected:
   virtual Response::Status checkUpdateState(double timef, double dt) override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_maskMethod(enum ParamsIOFlag ioFlag);

  private:
   char *maskMethod = nullptr;
};

} /* namespace PV */
#endif
