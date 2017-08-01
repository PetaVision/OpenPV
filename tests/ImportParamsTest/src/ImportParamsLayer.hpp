#ifndef IMPORTPARAMSLAYER_HPP_
#define IMPORTPARAMSLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ImportParamsLayer : public PV::ANNLayer {
  public:
   ImportParamsLayer(const char *name, HyPerCol *hc);
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;

  private:
   int initialize(const char *name, HyPerCol *hc);
   int initialize_base();
};

} /* namespace PV */
#endif
