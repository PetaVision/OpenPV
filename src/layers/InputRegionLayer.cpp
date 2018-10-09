/*
 * InputRegionLayer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionLayer.hpp"
#include "components/DependentBoundaryConditions.hpp"
#include "components/DependentPhaseParam.hpp"

namespace PV {

InputRegionLayer::InputRegionLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InputRegionLayer::InputRegionLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int InputRegionLayer::initialize_base() { return PV_SUCCESS; }

int InputRegionLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void InputRegionLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

PhaseParam *InputRegionLayer::createPhaseParam() { return new DependentPhaseParam(name, parent); }

BoundaryConditions *InputRegionLayer::createBoundaryConditions() {
   return new DependentBoundaryConditions(name, parent);
}

LayerInputBuffer *InputRegionLayer::createLayerInput() { return nullptr; }

InternalStateBuffer *InputRegionLayer::createInternalState() { return nullptr; }

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

void InputRegionLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parameters()->handleUnnecessaryParameter(name, "triggerLayerName");
   }
}

void InputRegionLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      sparseLayer = false;
      parameters()->handleUnnecessaryParameter(name, "sparseLayer");
   }
}

void InputRegionLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   if (ioFlag == PARAMS_IO_READ) {
      mUpdateGpu = false;
      parameters()->handleUnnecessaryParameter(name, "updateGpu");
   }
#endif // PV_USE_CUDA
}

Response::Status
InputRegionLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   setOriginalLayer();
   pvAssert(mOriginalLayer);
   if (!mOriginalLayer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE; // Make sure original layer has all the information we need to copy
   }
   mOriginalLayer->makeInputRegionsPointer();
   checkLayerDimensions();
   synchronizeMarginWidth(mOriginalLayer);
   mOriginalLayer->synchronizeMarginWidth(this);
   return Response::SUCCESS;
}

void InputRegionLayer::setOriginalLayer() {
   auto *originalLayerNameParam = getComponentByType<OriginalLayerNameParam>();
   pvAssert(originalLayerNameParam);

   ComponentBasedObject *originalObject = nullptr;
   try {
      originalObject = originalLayerNameParam->findLinkedObject(mTable);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalObject);

   mOriginalLayer = dynamic_cast<InputLayer *>(originalObject);
   if (mOriginalLayer == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
               getDescription_c(),
               originalObject->getName());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

void InputRegionLayer::checkLayerDimensions() {
   pvAssert(mOriginalLayer);
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   pvAssert(srcLoc != nullptr && loc != nullptr);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal
       || srcLoc->nf != loc->nf) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               mOriginalLayer->getName());
         errorMessage.printf(
               "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
               srcLoc->nxGlobal,
               srcLoc->nyGlobal,
               srcLoc->nf,
               loc->nxGlobal,
               loc->nyGlobal,
               loc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   pvAssert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
}

Response::Status InputRegionLayer::allocateDataStructures() {
   if (!mOriginalLayer->getDataStructuresAllocatedFlag()) {
      // original layer needs to create InputRegionsAllBatchElements first
      return Response::POSTPONE;
   }
   // mActivity must be null when parent allocate is called, since original layer will allocate it.
   Response::Status status = HyPerLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

int InputRegionLayer::setActivity() {
   for (int k = 0; k < getNumExtendedAllBatches(); k++) {
      getActivity()[k] = mOriginalLayer->getInputRegionsAllBatchElements()[k];
   }
   return PV_SUCCESS;
}

bool InputRegionLayer::needUpdate(double timed, double dt) const { return false; }

InputRegionLayer::~InputRegionLayer() {}

} /* namespace PV */
