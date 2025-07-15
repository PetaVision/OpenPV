#ifndef SEGMENTIFY_HPP_
#define SEGMENTIFY_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class Segmentify : public HyPerLayer {
  public:
   Segmentify(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~Segmentify();

  protected:
   Segmentify();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;

}; // class Segmentify

} /* namespace PV */
#endif
