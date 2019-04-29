/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "columns/Factory.hpp"
#include "components/StrengthParam.hpp"
#include "delivery/HyPerDeliveryCreator.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

HyPerConn::HyPerConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerConn::HyPerConn() {}

HyPerConn::~HyPerConn() { delete mUpdateTimer; }

void HyPerConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseConnection::initialize(name, params, comm);
}

void HyPerConn::initMessageActionMap() {
   BaseConnection::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionUpdateMessage const>(msgptr);
      return respondConnectionUpdate(castMessage);
   };
   mMessageActionMap.emplace("ConnectionUpdate", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionNormalizeMessage const>(msgptr);
      return respondConnectionNormalize(castMessage);
   };
   mMessageActionMap.emplace("ConnectionNormalize", action);
}

void HyPerConn::fillComponentTable() {
   BaseConnection::fillComponentTable();
   auto *arborList = createArborList();
   if (arborList) {
      addUniqueComponent(arborList);
   }
   auto *patchSize = createPatchSize();
   if (patchSize) {
      addUniqueComponent(patchSize);
   }
   auto *sharedWeights = createSharedWeights();
   if (sharedWeights) {
      addUniqueComponent(sharedWeights);
   }
   auto *weightsPair = createWeightsPair();
   if (weightsPair) {
      addUniqueComponent(weightsPair);
   }
   auto *weightInitializer = createWeightInitializer();
   if (weightInitializer) {
      addUniqueComponent(weightInitializer);
   }
   auto *weightNormalizer = createWeightNormalizer();
   if (weightNormalizer) {
      addUniqueComponent(weightNormalizer);
   }
   auto *weightUpdater = createWeightUpdater();
   if (weightUpdater) {
      addUniqueComponent(weightUpdater);
   }
}

BaseDelivery *HyPerConn::createDeliveryObject() {
   auto *deliveryCreator = new HyPerDeliveryCreator(name, parameters(), mCommunicator);
   addUniqueComponent(deliveryCreator);
   return deliveryCreator->create();
}

ArborList *HyPerConn::createArborList() { return new ArborList(name, parameters(), mCommunicator); }

PatchSize *HyPerConn::createPatchSize() { return new PatchSize(name, parameters(), mCommunicator); }

SharedWeights *HyPerConn::createSharedWeights() {
   return new SharedWeights(name, parameters(), mCommunicator);
}

WeightsPairInterface *HyPerConn::createWeightsPair() {
   return new WeightsPair(name, parameters(), mCommunicator);
}

InitWeights *HyPerConn::createWeightInitializer() {
   char *weightInitTypeString = nullptr;
   parameters()->ioParamString(
         PARAMS_IO_READ,
         name,
         "weightInitType",
         &weightInitTypeString,
         nullptr,
         true /*warnIfAbsent*/);
   // Note: The weightInitType string param gets read both here and by the
   // InitWeights::ioParam_weightInitType() method. It is read here because we need
   // to know the weight init type in order to instantiate the correct class. It is read in
   // InitWeights to store the value, in order to print it into the generated params file.
   // We don't write weightInitType in a HyPerConn method because we'd like to keep
   // all the WeightInitializer params together in the generated file, and
   // BaseConnection::ioParamsFillGroup() calls the components' ioParams() methods
   // in a loop, without knowing which component is which.

   FatalIf(
         weightInitTypeString == nullptr or weightInitTypeString[0] == '\0',
         "%s must set weightInitType.\n",
         getDescription_c());
   BaseObject *baseObject  = Factory::instance()->createByKeyword(weightInitTypeString, this);
   auto *weightInitializer = dynamic_cast<InitWeights *>(baseObject);
   FatalIf(
         weightInitializer == nullptr,
         "%s unable to create weightInitializer: %s is not an InitWeights keyword.\n",
         getDescription_c(),
         weightInitTypeString);

   free(weightInitTypeString);

   return weightInitializer;
}

NormalizeBase *HyPerConn::createWeightNormalizer() {
   NormalizeBase *normalizer = nullptr;
   char *normalizeMethod     = nullptr;
   parameters()->ioParamString(
         PARAMS_IO_READ, name, "normalizeMethod", &normalizeMethod, nullptr, true /*warnIfAbsent*/);
   // Note: The normalizeMethod string param gets read both here and by the
   // NormalizeBase::ioParam_weightInitType() method. It is read here because we need
   // to know the normalization method in order to instantiate the correct class. It is read in
   // NormalizeBase to store the value, in order to print it into the generated params file.
   // We don't write normalizeMethod in a HyPerConn method because we'd like to keep
   // all the WeightNormalizer params together in the generated file, and
   // BaseConnection::ioParamsFillGroup() calls the components' ioParams() methods
   // in a loop, without knowing which component is which.

   if (normalizeMethod == nullptr) {
      if (mCommunicator->globalCommRank() == 0) {
         Fatal().printf(
               "%s: specifying a normalizeMethod string is required.\n", getDescription_c());
      }
   }
   if (!strcmp(normalizeMethod, "")) {
      free(normalizeMethod);
      normalizeMethod = strdup("none");
   }
   if (strcmp(normalizeMethod, "none")) {
      auto strengthParam = new StrengthParam(name, parameters(), mCommunicator);
      addUniqueComponent(strengthParam);
   }
   BaseObject *baseObj = Factory::instance()->createByKeyword(normalizeMethod, this);
   normalizer          = dynamic_cast<NormalizeBase *>(baseObj);
   if (normalizer == nullptr) {
      pvAssert(baseObj);
      if (mCommunicator->commRank() == 0) {
         Fatal() << getDescription_c() << ": normalizeMethod \"" << normalizeMethod
                 << "\" is not a recognized normalization method." << std::endl;
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   free(normalizeMethod);
   return normalizer;
}

BaseWeightUpdater *HyPerConn::createWeightUpdater() {
   return new HebbianUpdater(name, parameters(), mCommunicator);
}

Response::Status
HyPerConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseConnection::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto updater = getComponentByType<BaseWeightUpdater>();
   if (updater and updater->getPlasticityFlag()) {
      auto delivery = getComponentByType<BaseDelivery>();
      if (delivery and delivery->getChannelCode() == CHANNEL_GAP) {
         WarnLog() << getDescription() << " is a gap connection but has plasticity set to true.\n"
                   << "The gap strength is calculated only once, during initialization, "
                   << "and will not change if the connection weights are updated.\n";
         // Perhaps GapConn should be resurrected to handle this warning and only this warning?
      }
   }
   return Response::SUCCESS;
}

Response::Status
HyPerConn::respondConnectionUpdate(std::shared_ptr<ConnectionUpdateMessage const> message) {
   auto *weightUpdater = getComponentByType<BaseWeightUpdater>();
   if (weightUpdater) {
      mUpdateTimer->start();
      weightUpdater->updateState(message->mTime, message->mDeltaT);
      mUpdateTimer->stop();
   }
   return Response::SUCCESS;
}

Response::Status
HyPerConn::respondConnectionNormalize(std::shared_ptr<ConnectionNormalizeMessage const> message) {
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

Response::Status
HyPerConn::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseConnection::registerData(message);
   if (Response::completed(status)) {
      if (getComponentByType<BaseWeightUpdater>()) {
         mUpdateTimer = new Timer(getName(), "conn", "update");
         message->mDataRegistry->registerTimer(mUpdateTimer);
      }
   }
   return status;
}

} // namespace PV
