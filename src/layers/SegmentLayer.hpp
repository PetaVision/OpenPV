#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class SegmentLayer : public HyPerLayer {
  public:
   SegmentLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~SegmentLayer();

  protected:
   SegmentLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void createComponentTable(char const *description) override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;

}; // class SegmentLayer

} /* namespace PV */
#endif
