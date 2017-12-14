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
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HyPerCol;

class BaseConnection : public BaseObject, Subject {
  public:
   BaseConnection(char const *name, HyPerCol *hc);

   virtual ~BaseConnection();

   virtual void addObserver(Observer *observer) override;

   template <typename S>
   S *getComponentByType();

   virtual int respond(std::shared_ptr<BaseMessage const> message) override;

   /**
    * A pure virtual function for modifying the post-synaptic layer's GSyn buffer based on the
    * connection and the presynaptic activity
    */
   int deliver() {
      mDeliveryObject->deliver();
      return PV_SUCCESS;
   }

   void deliverUnitInput(float *recvBuffer) { mDeliveryObject->deliverUnitInput(recvBuffer); }

   HyPerLayer *getPre() const { return mConnectionData->getPre(); }
   HyPerLayer *getPost() const { return mConnectionData->getPost(); }
   char const *getPreLayerName() const { return mConnectionData->getPreLayerName(); }
   char const *getPostLayerName() const { return mConnectionData->getPostLayerName(); }
   int getNumAxonalArbors() const { return mConnectionData->getNumAxonalArbors(); }

   ChannelType getChannelCode() const { return mDeliveryObject->getChannelCode(); }
   int getDelay(int arbor) const { return mConnectionData->getDelay(arbor); }
   bool getReceiveGpu() const { return mDeliveryObject->getReceiveGpu(); }
   bool getPlasticityFlag() const { return mWeightUpdater->getPlasticityFlag(); }

  protected:
   BaseConnection();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents();

   virtual ConnectionData *createConnectionData();
   virtual BaseDelivery *createDeliveryObject();
   virtual BaseWeightUpdater *createWeightUpdater();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int respondConnectionWriteParams(std::shared_ptr<ConnectionWriteParamsMessage const> message);

   int respondConnectionUpdate(std::shared_ptr<ConnectionUpdateMessage const> message);

   int
   respondConnectionFinalizeUpdate(std::shared_ptr<ConnectionFinalizeUpdateMessage const> message);

   int respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int allocateDataStructures() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual void deleteComponents();

  protected:
   ObserverTable mComponentTable;

  private:
   ConnectionData *mConnectionData   = nullptr;
   BaseDelivery *mDeliveryObject     = nullptr;
   BaseWeightUpdater *mWeightUpdater = nullptr;

}; // class BaseConnection

template <typename S>
S *BaseConnection::getComponentByType() {
   return mapLookupByType<S>(mComponentTable.getObjectMap(), getDescription());
}

} // namespace PV

#endif // BASECONNECTION_HPP_
