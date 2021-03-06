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
   WTALayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~WTALayer();
   virtual Response::Status updateState(double timef, double dt) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_binMaxMin(enum ParamsIOFlag ioFlag);

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
