#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "BaseLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class SegmentLayer : public BaseLayer {
  public:
   SegmentLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~SegmentLayer();

  protected:
   SegmentLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual ActivityComponent *createActivityComponent() override;

}; // class SegmentLayer

} /* namespace PV */
#endif
