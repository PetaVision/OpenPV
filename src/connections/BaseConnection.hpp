/*
 * BaseConnection.hpp
 *
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 */

#ifndef BASECONNECTION_HPP_
#define BASECONNECTION_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "delivery/BaseDelivery.hpp"
#include "observerpattern/Subject.hpp"
#include "utils/MapLookupByType.hpp"
#include "utils/Timer.hpp"

namespace PV {

class HyPerCol;

class BaseConnection : public BaseObject, public Subject {
  public:
   BaseConnection(char const *name, HyPerCol *hc);

   virtual ~BaseConnection();

   virtual void addObserver(Observer *observer) override;

   template <typename S>
   S *getComponentByType();

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   /**
    * The function that calls the DeliveryObject's deliver method
    */
   int deliver() {
      mDeliveryObject->deliver();
      return PV_SUCCESS;
   }

   void deliverUnitInput(float *recvBuffer) { mDeliveryObject->deliverUnitInput(recvBuffer); }

   bool isAllInputReady() { return mDeliveryObject->isAllInputReady(); }

   HyPerLayer *getPre() const { return mConnectionData->getPre(); }
   HyPerLayer *getPost() const { return mConnectionData->getPost(); }
   char const *getPreLayerName() const { return mConnectionData->getPreLayerName(); }
   char const *getPostLayerName() const { return mConnectionData->getPostLayerName(); }

   ChannelType getChannelCode() const { return mDeliveryObject->getChannelCode(); }
   bool getReceiveGpu() const { return mDeliveryObject->getReceiveGpu(); }

  protected:
   BaseConnection();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents();

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

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual void deleteComponents();

  protected:
   ObserverTable mComponentTable;

  private:
   ConnectionData *mConnectionData = nullptr;
   BaseDelivery *mDeliveryObject   = nullptr;

   Timer *mIOTimer = nullptr;

}; // class BaseConnection

template <typename S>
S *BaseConnection::getComponentByType() {
   return mapLookupByType<S>(mComponentTable.getObjectMap(), getDescription());
}

} // namespace PV

#endif // BASECONNECTION_HPP_
