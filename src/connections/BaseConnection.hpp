/*
 * BaseConnection.hpp
 *
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 */

#ifndef BASECONNECTION_HPP_
#define BASECONNECTION_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "components/ConnectionData.hpp"
#include "delivery/BaseDelivery.hpp"
#include "utils/Timer.hpp"

namespace PV {

class BaseConnection : public ComponentBasedObject {
  public:
   BaseConnection(char const *name, PVParams *params, Communicator const *comm);

   virtual ~BaseConnection();

   // Jul 10, 2018: get-methods have been moved into the corresponding component classes.
   // For example, the old BaseConnection::getPre() has been moved into the ConnectionData class.
   // To get the presynaptic layer from a connection named "conn", get the PatchSize component using
   // "ConnectionData *connectionData = conn->getComponentByType<ConnectionData>()" and then call
   // "connectionData->getPre()"

  protected:
   BaseConnection();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void initMessageActionMap() override;

   virtual void fillComponentTable() override;

   virtual ConnectionData *createConnectionData();
   virtual BaseDelivery *createDeliveryObject();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status
   respondConnectionWriteParams(std::shared_ptr<ConnectionWriteParamsMessage const> message);

   Response::Status
   respondConnectionFinalizeUpdate(std::shared_ptr<ConnectionFinalizeUpdateMessage const> message);

   Response::Status respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
#endif // PV_USE_CUDA

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual Response::Status copyInitialStateToGPU() override;

  private:
   Timer *mIOTimer = nullptr;

}; // class BaseConnection

} // namespace PV

#endif // BASECONNECTION_HPP_
