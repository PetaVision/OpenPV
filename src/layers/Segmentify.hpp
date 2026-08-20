#ifndef SEGMENTIFY_HPP_
#define SEGMENTIFY_HPP_

#include "BaseLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class Segmentify : public BaseLayer {
  public:
   Segmentify(const char *name, PVParams *params, Communicator const *comm);
   virtual ~Segmentify();

  protected:
   Segmentify();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual ActivityComponent *createActivityComponent() override;

}; // class Segmentify

} /* namespace PV */
#endif
