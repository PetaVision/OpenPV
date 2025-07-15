#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class SegmentLayer : public HyPerLayer {
  public:
   SegmentLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~SegmentLayer();

  protected:
   SegmentLayer();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;

}; // class SegmentLayer

} /* namespace PV */
#endif
