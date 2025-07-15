/*
 * HyPerDeliveryCreator.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDeliveryCreator.hpp"
#include "columns/Factory.hpp"

namespace PV {

HyPerDeliveryCreator::HyPerDeliveryCreator(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

HyPerDeliveryCreator::HyPerDeliveryCreator() {}

HyPerDeliveryCreator::~HyPerDeliveryCreator() {}

void HyPerDeliveryCreator::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void HyPerDeliveryCreator::setObjectType() { mObjectType = "HyPerDeliveryCreator"; }

int HyPerDeliveryCreator::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseObject::ioParamsFillGroup(ioSwitch);
   ioParam_receiveGpu(ioSwitch);
   ioParam_pvpatchAccumulateType(ioSwitch);
   ioParam_updateGSynFromPostPerspective(ioSwitch);
   return status;
}

void HyPerDeliveryCreator::ioParam_receiveGpu(ParamsIOSwitch ioSwitch) {
#ifdef PV_USE_CUDA
   bool warnIfAbsent = true;
#else
   bool warnIfAbsent = false;
#endif // PV_USE_CUDA
   mParamsIO->ioParam(ioSwitch, "receiveGpu", &mReceiveGpu, warnIfAbsent);
#ifndef PV_USE_CUDA
   if (mCommunicator->globalCommRank() == 0) {
      FatalIf(
            mReceiveGpu,
            "%s: receiveGpu is set to true in params, but PetaVision was compiled without GPU "
            "acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

void HyPerDeliveryCreator::ioParam_pvpatchAccumulateType(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "pvpatchAccumulateType", &mAccumulateTypeString);
   if (ioSwitch == ParamsIOSwitch::Read) {
      FatalIf(
            mAccumulateTypeString.empty(),
            "%s \"%s\" string parameter pvpatchAccumulateType cannot be empty or null.\n",
            mParamsIO->getKeyword().c_str(),
            getName());
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char &c : mAccumulateTypeString) {
         c = (char)tolower(static_cast<int>(c));
      }

      if (mAccumulateTypeString == "convolve") {
         mAccumulateType = CONVOLVE;
      }
      else if (mAccumulateTypeString == "stochastic") {
         mAccumulateType = STOCHASTIC;
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
         std::exit(EXIT_FAILURE);
      }
      pvAssert(!mParamsIO->presentAndNotBeenRead("receiveGpu"));
      FatalIf(
            mReceiveGpu and mAccumulateType == STOCHASTIC,
            "%s sets receiveGpu to true and pvpatchAccumulateType to stochastic, "
            "but stochastic release has not been implemented on the GPU.\n",
            getDescription_c());
   }
}

void HyPerDeliveryCreator::ioParam_updateGSynFromPostPerspective(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "updateGSynFromPostPerspective", &mUpdateGSynFromPostPerspective);
}

HyPerDelivery *HyPerDeliveryCreator::create() {
   char const *perspective = getUpdateGSynFromPostPerspective() ? "Post" : "Pre";

   char const *type;
   if (getReceiveGpu()) {
      type = "GPU";
   }
   else {
      switch (mAccumulateType) {
         case CONVOLVE: type   = "Convolve"; break;
         case STOCHASTIC: type = "Stochastic"; break;
         default: pvAssert(0); break;
      }
   }
   std::string keyword("");
   keyword.append(perspective).append("synapticPerspective").append(type).append("Delivery");
   BaseObject *baseObject = Factory::instance()->createByKeyword(keyword.c_str(), this);

   HyPerDelivery *deliveryObject = dynamic_cast<HyPerDelivery *>(baseObject);
   pvAssert(deliveryObject); // All possible keywords should generate HyPerDelivery-derived objects.
   return deliveryObject;
}

} // end namespace PV
