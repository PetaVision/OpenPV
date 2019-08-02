/*
 * AbstractNormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "AbstractNormProbe.hpp"
#include "layers/HyPerLayer.hpp"
#include <limits>

namespace PV {

AbstractNormProbe::AbstractNormProbe() : LayerProbe() {}

AbstractNormProbe::AbstractNormProbe(const char *name, PVParams *params, Communicator const *comm)
      : LayerProbe() {
   initialize(name, params, comm);
}

AbstractNormProbe::~AbstractNormProbe() {
   free(normDescription);
   normDescription = NULL;
   free(maskLayerName);
   maskLayerName = NULL;
   // Don't free mMaskLayerData, which belongs to the layer named by maskLayerName.
}

void AbstractNormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   LayerProbe::initialize(name, params, comm);
   setNormDescription();
}

int AbstractNormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_maskLayerName(ioFlag);
   return status;
}

void AbstractNormProbe::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "maskLayerName", &maskLayerName, NULL, false /*warnIfAbsent*/);
}

Response::Status
AbstractNormProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LayerProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   assert(targetLayer);
   if (maskLayerName && maskLayerName[0]) {
      mMaskLayerData = message->mObjectTable->findObject<BasePublisherComponent>(maskLayerName);
      FatalIf(
            mMaskLayerData == nullptr,
            "%s: maskLayerName \"%s\" does not have a BasePublisherComponent.\n",
            getDescription_c(),
            maskLayerName);

      PVLayerLoc const *maskLoc = mMaskLayerData->getLayerLoc();
      PVLayerLoc const *loc     = targetLayer->getLayerLoc();
      pvAssert(maskLoc != nullptr && loc != nullptr);
      FatalIf(
            maskLoc->nxGlobal != loc->nxGlobal or maskLoc->nyGlobal != loc->nyGlobal,
            "%s: maskLayerName \"%s\" does not have the same x and y dimensions.\n"
            "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
            getDescription_c(),
            maskLayerName,
            maskLoc->nxGlobal,
            maskLoc->nyGlobal,
            maskLoc->nf,
            loc->nxGlobal,
            loc->nyGlobal,
            loc->nf);

      FatalIf(
            maskLoc->nf != 1 and maskLoc->nf != loc->nf,
            "%s: maskLayerName \"%s\" must either have the same number of features as this layer, "
            "or one feature.\n",
            "    mask (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
            getDescription_c(),
            maskLayerName,
            maskLoc->nxGlobal,
            maskLoc->nyGlobal,
            maskLoc->nf,
            loc->nxGlobal,
            loc->nyGlobal,
            loc->nf);
      pvAssert(maskLoc->nx == loc->nx && maskLoc->ny == loc->ny);
      singleFeatureMask = maskLoc->nf == 1 && loc->nf != 1;
   }
   return Response::SUCCESS;
}

int AbstractNormProbe::setNormDescription() { return setNormDescriptionToString("norm"); }

int AbstractNormProbe::setNormDescriptionToString(char const *s) {
   normDescription = strdup(s);
   return normDescription ? PV_SUCCESS : PV_FAILURE;
}

void AbstractNormProbe::calcValues(double timeValue) {
   double *valuesBuffer = this->getValuesBuffer();
   for (int b = 0; b < this->getNumValues(); b++) {
      valuesBuffer[b] = getValueInternal(timeValue, b);
   }
   MPI_Allreduce(
         MPI_IN_PLACE,
         valuesBuffer,
         getNumValues(),
         MPI_DOUBLE,
         MPI_SUM,
         mCommunicator->communicator());
}

Response::Status AbstractNormProbe::outputState(double simTime, double deltaTime) {
   getValues(simTime);
   double *valuesBuffer = this->getValuesBuffer();
   if (!mOutputStreams.empty()) {
      int nBatch = getNumValues();
      int nk     = getTargetLayer()->getNumGlobalNeurons();
      for (int b = 0; b < nBatch; b++) {
         output(b).printf("%6.3f, %d, %8d, %f", simTime, b, nk, valuesBuffer[b]);
         output(b) << std::endl;
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
