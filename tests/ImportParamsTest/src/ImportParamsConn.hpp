#ifndef IMPORTPARAMSCONN_HPP_
#define IMPORTPARAMSCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class ImportParamsConn : public PV::HyPerConn {
  public:
   ImportParamsConn(const char *name, HyPerCol *hc);

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  private:
   int initialize(const char *name, HyPerCol *hc);
   int initialize_base();
};

} /* namespace PV */
#endif
