#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class SegmentLayer : public HyPerLayer {
  public:
   SegmentLayer(const char *name, HyPerCol *hc);
   virtual ~SegmentLayer();

  protected:
   SegmentLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual void createComponentTable(char const *description) override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;

}; // class SegmentLayer

} /* namespace PV */
#endif
