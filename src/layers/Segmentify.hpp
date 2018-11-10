#ifndef SEGMENTIFY_HPP_
#define SEGMENTIFY_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class Segmentify : public HyPerLayer {
  public:
   Segmentify(const char *name, HyPerCol *hc);
   virtual ~Segmentify();

  protected:
   Segmentify();
   int initialize(const char *name, HyPerCol *hc);
   virtual void createComponentTable(char const *description) override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;

}; // class Segmentify

} /* namespace PV */
#endif
