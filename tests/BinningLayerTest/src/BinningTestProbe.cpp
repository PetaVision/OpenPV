/*
 * BinningTestProbe.cpp
 *
 *  Created on: Jan 15, 2015
 *      Author: slundquist
 */

#include "BinningTestProbe.hpp"
#include "include/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
#include "layers/BinningLayer.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Response.hpp"
#include "probes/TargetLayerComponent.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/ComponentBasedObject.hpp>
#include <columns/Messages.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/BinningActivityBuffer.hpp>
#include <components/PhaseParam.hpp>

#include <cmath>
#include <functional>

namespace PV {

BinningTestProbe::BinningTestProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}
void BinningTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   mProbeTargetLayerLocator = std::make_shared<TargetLayerComponent>(name, params);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   BaseObject::initialize(name, params, comm);
}

Response::Status
BinningTestProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Set target layer and trigger layer
   status = status + mProbeTargetLayerLocator->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mBinningLayer = dynamic_cast<BinningLayer *>(mProbeTargetLayerLocator->getTargetLayer());
   FatalIf(
         getBinningLayer() == nullptr,
         "%s requires the target layer to be a BinningLayer.\n",
         getDescription_c());
   return Response::SUCCESS;
}

void BinningTestProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

int BinningTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   mProbeTargetLayerLocator->ioParamsFillGroup(ioFlag);
   return status;
}

Response::Status
BinningTestProbe::outputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   if (message->mTime == 0.0) {
      return Response::SUCCESS;
   }
   // Grab layer size
   const PVLayerLoc *loc = getBinningLayer()->getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;
   int nxGlobal          = loc->nxGlobal;
   int nyGlobal          = loc->nyGlobal;
   int nxExt             = nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = ny + loc->halo.lt + loc->halo.rt;
   int nxGlobalExt       = nxGlobal + loc->halo.lt + loc->halo.rt;
   int nyGlobalExt       = nyGlobal + loc->halo.lt + loc->halo.rt;
   // Grab the activity layer of current layer
   auto *publisherComponent = getBinningLayer()->getComponentByType<BasePublisherComponent>();
   FatalIf(
         publisherComponent == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         getBinningLayer()->getDescription_c());
   const float *A = publisherComponent->getLayerData();

   // Grab BinSigma from BinningLayer, which is contained in a component.
   auto *activityComponent = getBinningLayer()->getComponentByType<ComponentBasedObject>();
   pvAssert(activityComponent);
   auto *binningActivityBuffer = activityComponent->getComponentByType<BinningActivityBuffer>();
   pvAssert(binningActivityBuffer);
   const float binSigma = binningActivityBuffer->getBinSigma();

   // We only care about restricted space
   for (int iY = loc->halo.up; iY < ny + loc->halo.up; iY++) {
      for (int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++) {
         for (int iF = 0; iF < nf; iF++) {
            int origIndexGlobal   = kIndex(iX + kx0, iY + ky0, 0, nxGlobalExt, nyGlobalExt, 1);
            int binningIndexLocal = kIndex(iX, iY, iF, nxExt, nyExt, nf);
            float observedValue   = A[binningIndexLocal];
            if (binSigma == 0) {
               // Based on the input image, F index should be floor(origIndex/255*32), except
               // that if origIndex==255, F index should be 31.
               float binnedIndex =
                     std::floor((float)origIndexGlobal / 255.0f * 32.0f) - (origIndexGlobal == 255);
               float correctValue = iF == binnedIndex;
               FatalIf(
                     observedValue != correctValue,
                     "%s, extended global location x=%d, y=%d, f=%d, expected %f, observed %f.\n",
                     getBinningLayer()->getDescription_c(),
                     iX + kx0,
                     iY + ky0,
                     iF,
                     (double)correctValue,
                     (double)observedValue);
            }
            else {
               // Map feature index to the center of its bin
               float binCenter = ((float)iF + 0.5f) / nf; // Assumes maxBin is 1 and minBin is zero
               // Determine number of bins away the input value is from the bin center
               float inputValue   = (float)origIndexGlobal / 255.0f;
               float binOffset    = (binCenter - inputValue) * (float)loc->nf;
               float correctValue = exp(-binOffset * binOffset / (2 * binSigma * binSigma));
               FatalIf(
                     std::fabs(observedValue - correctValue) > 0.0001f,
                     "%s, extended global location x=%d, y=%d, f=%d, expected %f, observed %f.\n",
                     getBinningLayer()->getDescription_c(),
                     iX + kx0,
                     iY + ky0,
                     iF,
                     (double)correctValue,
                     (double)observedValue);
            }
         }
      }
   }
   return Response::SUCCESS;
}

Response::Status
BinningTestProbe::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = Response::SUCCESS;
   int targetLayerPhase = getBinningLayer()->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

Response::Status
BinningTestProbe::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

} // end namespace PV
