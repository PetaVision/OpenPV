/*
 * WTALayer.hpp
 * Author: slundquist
 */

// WTALayer was deprecated on Aug 15, 2018, in favor of WTAConn.

#ifndef WTALAYER_HPP_
#define WTALAYER_HPP_
#include "layers/HyPerLayer.hpp"

namespace PV {

class WTALayer : public HyPerLayer {
  public:
   WTALayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~WTALayer();
   virtual Response::Status updateState(double timef, double dt) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   void ioParam_originalLayerName(ParamsIOSwitch ioSwitch);
   void ioParam_binMaxMin(ParamsIOSwitch ioSwitch);

   virtual LayerInputBuffer *createLayerInput() override;
   virtual InternalStateBuffer *createInternalState() override;

   virtual void initializeActivity() override;

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
