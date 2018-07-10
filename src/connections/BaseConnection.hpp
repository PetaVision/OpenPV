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

class HyPerCol;

class BaseConnection : public ComponentBasedObject {
  public:
   BaseConnection(char const *name, HyPerCol *hc);

   virtual ~BaseConnection();

   template <typename S>
   void addComponent(S *observer);

   // Jul 10, 2018: get-methods have been moved into the corresponding component classes.
   // For example, the old HyPerConn::getPatchSizeX() has been moved into the PatchSize class.
   // To get the PatchSizeX value from a HyPerConn conn , get the PatchSize component using
   // "PatchSize *patchsize = conn->getComponentByType<PatchSize>()" and then call
   // "patchSize->getPatchSizeX()"
   //   HyPerLayer *getPre() const { return mConnectionData->getPre(); }
   //   HyPerLayer *getPost() const { return mConnectionData->getPost(); }
   //   char const *getPreLayerName() const { return mConnectionData->getPreLayerName(); }
   //   char const *getPostLayerName() const { return mConnectionData->getPostLayerName(); }
   //
   //   ChannelType getChannelCode() const { return mDeliveryObject->getChannelCode(); }
   //   bool getReceiveGpu() const { return mDeliveryObject->getReceiveGpu(); }

  protected:
   BaseConnection();

   int initialize(char const *name, HyPerCol *hc);

   virtual void initMessageActionMap() override;

   virtual void setObserverTable() override;

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

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

  private:
   ConnectionData *mConnectionData = nullptr;
   BaseDelivery *mDeliveryObject   = nullptr;

   Timer *mIOTimer = nullptr;

}; // class BaseConnection

template <typename S>
void BaseConnection::addComponent(S *observer) {
   auto addedObject = dynamic_cast<BaseObject *>(observer);
   FatalIf(
         addedObject == nullptr,
         "%s is not a BaseObject-derived class, and therefore cannot be a component of %s.\n",
         getDescription_c(),
         observer->getDescription_c());
   addUniqueComponent(observer->getDescription(), observer);
}

} // namespace PV

#endif // BASECONNECTION_HPP_
