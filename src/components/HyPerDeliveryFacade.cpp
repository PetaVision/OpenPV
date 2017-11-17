/*
 * HyPerDeliveryFacade.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDeliveryFacade.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

HyPerDeliveryFacade::HyPerDeliveryFacade(char const *name, HyPerCol *hc) { initialize(name, hc); }

HyPerDeliveryFacade::HyPerDeliveryFacade() {}

HyPerDeliveryFacade::~HyPerDeliveryFacade() {}

int HyPerDeliveryFacade::initialize(char const *name, HyPerCol *hc) {
   return BaseDelivery::initialize(name, hc);
}

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
   parent->parameters()->ioParamString(
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
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s error: pvpatchAccumulateType \"%s\" is unrecognized.\n",
                  getDescription_c(),
                  mAccumulateTypeString);
            ErrorLog().printf("  Allowed values are \"convolve\" or \"stochastic\".\n");
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   FatalIf(
         mReceiveGpu and mAccumulateType == HyPerDelivery::STOCHASTIC,
         "%s sets receiveGpu to true and pvpatchAccumulateType to stochastic, "
         "but stochastic release has not been implemented on the GPU.\n",
         getDescription_c());
}

void HyPerDeliveryFacade::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
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
         // baseObject = Factory::instance()->createByKeyword(
         //       "PostsynapticPerspectiveGPUDelivery", name, parent);
      }
      else {
         // baseObject = Factory::instance()->createByKeyword(
         //       "PresynapticPerspectiveGPUDelivery", name, parent);
      }
#else //
      pvAssert(0); // If PV_USE_CUDA is off, receiveGpu should always be false.
#endif // PV_USE_CUDA
   }
   else {
      switch (mAccumulateType) {
         case HyPerDelivery::CONVOLVE:
            if (getUpdateGSynFromPostPerspective()) {
               // baseObject = Factory::instance()->createByKeyword(
               //       "PostsynapticPerspectiveColvolveDelivery", name, parent);
            }
            else {
               baseObject = Factory::instance()->createByKeyword(
                     "PresynapticPerspectiveConvolveDelivery", name, parent);
            }
            break;
         case HyPerDelivery::STOCHASTIC:
            if (getUpdateGSynFromPostPerspective()) {
               // baseObject = Factory::instance()->createByKeyword(
               //       "PostsynapticPerspectiveStochasticDelivery", name, parent);
            }
            else {
               baseObject = Factory::instance()->createByKeyword(
                     "PresynapticPerspectiveStochasticDelivery", name, parent);
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

int HyPerDeliveryFacade::allocateDataStructures() {
   int status = BaseDelivery::allocateDataStructures();
   if (mDeliveryIntern) {
      mDeliveryIntern->setNumArbors(mNumArbors);
      mDeliveryIntern->setPreAndPostLayers(mPreLayer, mPostLayer);
      mDeliveryIntern->respond(std::make_shared<AllocateDataMessage>());
   }
   return status;
}

double HyPerDeliveryFacade::convertToRateDeltaTimeFactor(double timeConstantTau) const {
   double dtFactor = 1.0;
   if (mConvertRateToSpikeCount) {
      double dt = parent->getDeltaTime();
      if (timeConstantTau > 0) {
         double exp_dt_tau = exp(-dt / timeConstantTau);
         dtFactor          = (1.0 - exp_dt_tau) / exp_dt_tau;
         // the above factor was chosen so that for a constant input of G_SYN to an excitatory
         // conductance G_EXC, then G_EXC -> G_SYN as t -> inf
      }
      else {
         dtFactor = dt;
      }
   }
   return dtFactor;
}

void HyPerDeliveryFacade::deliver(Weights *weights, Weights *postWeights) {
   if (mDeliveryIntern) {
      int numArbors = weights->getNumArbors();
      FatalIf(
            numArbors != (int)mDelay.size(),
            "%s has %d %s, but the number of delays is %d.\n",
            getDescription_c(),
            numArbors,
            numArbors == 1 ? "arbor" : "arbors",
            (int)mDelay.size());
      mDeliveryIntern->deliver(weights, postWeights);
   }
}

} // end namespace PV
