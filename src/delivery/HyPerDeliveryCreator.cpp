/*
 * HyPerDeliveryCreator.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDeliveryCreator.hpp"
#include "columns/Factory.hpp"

namespace PV {

HyPerDeliveryCreator::HyPerDeliveryCreator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerDeliveryCreator::HyPerDeliveryCreator() {}

HyPerDeliveryCreator::~HyPerDeliveryCreator() { free(mAccumulateTypeString); }

void HyPerDeliveryCreator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void HyPerDeliveryCreator::setObjectType() { mObjectType = "HyPerDeliveryCreator"; }

int HyPerDeliveryCreator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   ioParam_receiveGpu(ioFlag);
   ioParam_pvpatchAccumulateType(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   return status;
}

void HyPerDeliveryCreator::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   bool warnIfAbsent = true;
#else
   bool warnIfAbsent = false;
#endif // PV_USE_CUDA
   parameters()->ioParamValue(
         ioFlag, name, "receiveGpu", &mReceiveGpu, mReceiveGpu /*default*/, warnIfAbsent);
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

void HyPerDeliveryCreator::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "pvpatchAccumulateType", &mAccumulateTypeString, "convolve");
   if (ioFlag == PARAMS_IO_READ) {
      pvAssert(mAccumulateTypeString and mAccumulateTypeString[0]);
      // Convert string to lowercase so that capitalization doesn't matter.
      for (char *c = mAccumulateTypeString; *c != '\0'; c++) {
         *c = (char)tolower((int)*c);
      }

      if (strcmp(mAccumulateTypeString, "convolve") == 0) {
         mAccumulateType = CONVOLVE;
      }
      else if (strcmp(mAccumulateTypeString, "stochastic") == 0) {
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
         exit(EXIT_FAILURE);
      }
      pvAssert(!parameters()->presentAndNotBeenRead(name, "receiveGpu"));
      FatalIf(
            mReceiveGpu and mAccumulateType == STOCHASTIC,
            "%s sets receiveGpu to true and pvpatchAccumulateType to stochastic, "
            "but stochastic release has not been implemented on the GPU.\n",
            getDescription_c());
   }
}

void HyPerDeliveryCreator::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "updateGSynFromPostPerspective",
         &mUpdateGSynFromPostPerspective,
         mUpdateGSynFromPostPerspective);
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
