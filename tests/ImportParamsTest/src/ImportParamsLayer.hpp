#ifndef IMPORTPARAMSLAYER_HPP_
#define IMPORTPARAMSLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ImportParamsLayer : public PV::ANNLayer {
  public:
   ImportParamsLayer(const char *name, PVParams *params, Communicator *comm);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  private:
   void initialize(const char *name, PVParams *params, Communicator *comm);
};

} /* namespace PV */
#endif
