#ifndef IMPORTPARAMSCONN_HPP_
#define IMPORTPARAMSCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class ImportParamsConn : public PV::HyPerConn {
  public:
   ImportParamsConn(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  private:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int initialize_base();
};

} /* namespace PV */
#endif
