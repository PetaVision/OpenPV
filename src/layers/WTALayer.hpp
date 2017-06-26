/*
 * WTALayer.hpp
 * Author: slundquist
 */

#ifndef WTALAYER_HPP_
#define WTALAYER_HPP_
#include "ANNLayer.hpp"

namespace PV {

class WTALayer : public PV::HyPerLayer {
  public:
   WTALayer(const char *name, HyPerCol *hc);
   virtual ~WTALayer();
   virtual int updateState(double timef, double dt) override;
   virtual int communicateInitInfo(CommunicateInitInfoMessage const *message) override;
   virtual bool activityIsSpiking() override { return false; }

  protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_binMaxMin(enum ParamsIOFlag ioFlag);
   int allocateV() override;
   int initializeV() override;
   virtual int initializeActivity() override;

  private:
   int initialize_base();
   float binMax;
   float binMin;

  protected:
   char *originalLayerName;
   HyPerLayer *originalLayer;

}; // class WTALayer

} // namespace PV
#endif
