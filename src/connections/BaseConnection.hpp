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
#include "weightupdaters/BaseWeightUpdater.hpp"
//#include "normalizers/NormalizeBase.hpp"
#include "observerpattern/Subject.hpp"

namespace PV {

class HyPerCol;

class BaseConnection : public BaseObject, Subject {
  protected:
   /**
    * List of parameters needed from the BaseConnection class
    * @name BaseConnection Parameters
    * @{
    */

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   /** @} */ // End of BaseConnection Parameters

  public:
   BaseConnection(char const *name, HyPerCol *hc);

   virtual ~BaseConnection();

   virtual void addObserver(Observer *observer) override;

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
   int getNumAxonalArbors() const { return mConnectionData->getNumAxonalArbors(); }

   ChannelType getChannelCode() const { return mDeliveryObject->getChannelCode(); }
   int getDelay(int arbor) const { return mDeliveryObject->getDelay(arbor); }
   bool getConvertRateToSpikeCount() const { return mDeliveryObject->getConvertRateToSpikeCount(); }
   bool getReceiveGpu() const { return mDeliveryObject->getReceiveGpu(); }

  protected:
   BaseConnection();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents();

   virtual ConnectionData *createConnectionData();
   // virtual NormalizeBase *createWeightNormalizer();
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

   virtual void deleteComponents();

  protected:
   // If this flag is set and HyPerCol sets initializeFromCheckpointDir, load initial state from
   // the initializeFromCheckpointDir directory.
   bool initializeFromCheckpointFlag = true;

   char *mNormalizeMethod = nullptr;

   ObserverTable mComponentTable;

  private:
   ConnectionData *mConnectionData = nullptr;
   // NormalizeBase *mWeightNormalizer = nullptr;
   BaseDelivery *mDeliveryObject     = nullptr;
   BaseWeightUpdater *mWeightUpdater = nullptr;

}; // class BaseConnection

} // namespace PV

#endif // BASECONNECTION_HPP_
