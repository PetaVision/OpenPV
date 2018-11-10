/*
 * HyPerDeliveryFacade.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDeliveryFacade.hpp"
#include "columns/Factory.hpp"

namespace PV {

HyPerDeliveryFacade::HyPerDeliveryFacade(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

HyPerDeliveryFacade::HyPerDeliveryFacade() {}

HyPerDeliveryFacade::~HyPerDeliveryFacade() {}

void HyPerDeliveryFacade::initialize(char const *name, PVParams *params, Communicator *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void HyPerDeliveryFacade::setObjectType() { mObjectType = "HyPerDeliveryFacade"; }

int HyPerDeliveryFacade::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseDelivery::ioParamsFillGroup(ioFlag);
   ioParam_accumulateType(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   if (ioFlag == PARAMS_IO_READ) {
      createDeliveryIntern();
   }
   if (mDeliveryIntern) {
      mDeliveryIntern->ioParams(ioFlag, false, false);
   }
   return status;
}

void HyPerDeliveryFacade::ioParam_accumulateType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "pvpatchAccumulateType", &mAccumulateTypeString, "convolve");
   if (ioFlag == PARAMS_IO_READ) {
      pvAssert(mAccumulateTypeString and mAccumulateTypeString[0]);
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char *c = mAccumulateTypeString; *c != '\0'; c++) {
         *c = (char)tolower((int)*c);
      }

      if (strcmp(mAccumulateTypeString, "convolve") == 0) {
         mAccumulateType = HyPerDelivery::CONVOLVE;
      }
      else if (strcmp(mAccumulateTypeString, "stochastic") == 0) {
         mAccumulateType = HyPerDelivery::STOCHASTIC;
      }
      else {
         if (mCommunicator->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s error: pvpatchAccumulateType \"%s\" is unrecognized.\n",
                  getDescription_c(),
                  mAccumulateTypeString);
            ErrorLog().printf("  Allowed values are \"convolve\" or \"stochastic\".\n");
         }
         MPI_Barrier(mCommunicator->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
   pvAssert(!parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   FatalIf(
         mReceiveGpu and mAccumulateType == HyPerDelivery::STOCHASTIC,
         "%s sets receiveGpu to true and pvpatchAccumulateType to stochastic, "
         "but stochastic release has not been implemented on the GPU.\n",
         getDescription_c());
}

void HyPerDeliveryFacade::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "updateGSynFromPostPerspective",
         &mUpdateGSynFromPostPerspective,
         mUpdateGSynFromPostPerspective);
}

void HyPerDeliveryFacade::createDeliveryIntern() {
   // Check channel number for noupdate
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      mDeliveryIntern = nullptr;
      return;
   }
   BaseObject *baseObject = nullptr;
   if (getReceiveGpu()) {
#ifdef PV_USE_CUDA
      if (getUpdateGSynFromPostPerspective()) {
         baseObject = Factory::instance()->createByKeyword(
               "PostsynapticPerspectiveGPUDelivery", name, parameters(), mCommunicator);
      }
      else {
         baseObject = Factory::instance()->createByKeyword(
               "PresynapticPerspectiveGPUDelivery", name, parameters(), mCommunicator);
      }
#else //
      pvAssert(0); // If PV_USE_CUDA is off, receiveGpu should always be false.
#endif // PV_USE_CUDA
   }
   else {
      switch (mAccumulateType) {
         case HyPerDelivery::CONVOLVE:
            if (getUpdateGSynFromPostPerspective()) {
               baseObject = Factory::instance()->createByKeyword(
                     "PostsynapticPerspectiveConvolveDelivery", name, parameters(), mCommunicator);
            }
            else {
               baseObject = Factory::instance()->createByKeyword(
                     "PresynapticPerspectiveConvolveDelivery", name, parameters(), mCommunicator);
            }
            break;
         case HyPerDelivery::STOCHASTIC:
            if (getUpdateGSynFromPostPerspective()) {
               baseObject = Factory::instance()->createByKeyword(
                     "PostsynapticPerspectiveStochasticDelivery",
                     name,
                     parameters(),
                     mCommunicator);
            }
            else {
               baseObject = Factory::instance()->createByKeyword(
                     "PresynapticPerspectiveStochasticDelivery", name, parameters(), mCommunicator);
            }
            break;
         default:
            pvAssert(0); // CONVOLVE and STOCHASTIC are the only allowed possibilities
            break;
      }
   }
   if (baseObject != nullptr) {
      mDeliveryIntern = dynamic_cast<HyPerDelivery *>(baseObject);
      pvAssert(mDeliveryIntern);
   }
}

Response::Status HyPerDeliveryFacade::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // DeliveryIntern needs to know the ConnectionData and the WeightsPair.
   if (mDeliveryIntern) {
      Response::Status internStatus = mDeliveryIntern->respond(message);
      if (internStatus == Response::POSTPONE) {
         return Response::POSTPONE;
      }
#ifdef PV_USE_CUDA
      mUsingGPUFlag = mDeliveryIntern->isUsingGPU();
#endif // PV_USE_CUDA
   }

   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
HyPerDeliveryFacade::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   auto status = BaseDelivery::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (mDeliveryIntern) {
      status = mDeliveryIntern->respond(message);
   }
   return status;
}
#endif // PV_USE_CUDA

Response::Status HyPerDeliveryFacade::allocateDataStructures() {
   auto status = BaseDelivery::allocateDataStructures();
   if (Response::completed(status) and mDeliveryIntern != nullptr) {
      auto internMessage = std::make_shared<AllocateDataStructuresMessage>();
      status             = mDeliveryIntern->respond(internMessage);
   }
   return status;
}

Response::Status
HyPerDeliveryFacade::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = BaseDelivery::initializeState(message);
   if (Response::completed(status) and mDeliveryIntern != nullptr) {
      status = mDeliveryIntern->respond(message);
   }
   return status;
}

Response::Status HyPerDeliveryFacade::copyInitialStateToGPU() {
   auto status = Response::SUCCESS;
   if (mDeliveryIntern) {
      auto copyMessage = std::make_shared<CopyInitialStateToGPUMessage>();
      status           = mDeliveryIntern->respond(copyMessage);
   }
   return status;
}

void HyPerDeliveryFacade::deliver(float *destBuffer) {
   // The internal delivery object mDeliveryIntern added itself to the post layer during
   // communicate; the post layer will call that delivery object as well.
   // Therefore nothing needs to be done here.
}

void HyPerDeliveryFacade::deliverUnitInput(float *recvBuffer) {
   // The internal delivery object mDeliveryIntern added itself to the post layer during
   // communicate; the post layer will call that delivery object as well.
   // Therefore nothing needs to be done here.
}

bool HyPerDeliveryFacade::isAllInputReady() {
   return getChannelCode() == CHANNEL_NOUPDATE ? true : mDeliveryIntern->isAllInputReady();
}

} // end namespace PV
