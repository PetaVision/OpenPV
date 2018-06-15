/*
 * WTALayer.hpp
 * Author: slundquist
 */

#ifndef WTALAYER_HPP_
#define WTALAYER_HPP_

#include "ANNLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class WTALayer : public PV::HyPerLayer {
  public:
   WTALayer(const char *name, HyPerCol *hc);
   virtual ~WTALayer();
   virtual Response::Status updateState(double timef, double dt) override;
   virtual bool activityIsSpiking() override { return false; }

  protected:
   virtual void setObserverTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_binMaxMin(enum ParamsIOFlag ioFlag);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer();
   virtual void allocateV() override;
   virtual void initializeV() override;
   virtual void initializeActivity() override;

  private:
   int initialize_base();
   float binMax;
   float binMin;

  protected:
   HyPerLayer *mOriginalLayer = nullptr;

}; // class WTALayer

} // namespace PV
#endif
