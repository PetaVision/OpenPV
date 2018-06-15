/*
 * InputRegionLayer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionLayer.hpp"

namespace PV {

InputRegionLayer::InputRegionLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InputRegionLayer::InputRegionLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int InputRegionLayer::initialize_base() {
   numChannels = 0;
   return PV_SUCCESS;
}

int InputRegionLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void InputRegionLayer::setObserverTable() {
   HyPerLayer::setObserverTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

void InputRegionLayer::ioParam_phase(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // Phase will be copied from original layer in CommunicateInitInfo stage,
      // but probably is not needed.
      parent->parameters()->handleUnnecessaryParameter(name, "phase");
   }
}

void InputRegionLayer::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // mirrorBCflag will be copied from original layer in CommunicateInitInfo stage.
      parent->parameters()->handleUnnecessaryParameter(name, "mirrorBCflag");
   }
}

void InputRegionLayer::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // mirrorBCflag will be copied from original layer in CommunicateInitInfo stage.
      parent->parameters()->handleUnnecessaryParameter(name, "valueBC");
   }
}

void InputRegionLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initVTypeString = nullptr;
      parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

void InputRegionLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerLayerName");
   }
}

void InputRegionLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      sparseLayer = false;
      parent->parameters()->handleUnnecessaryParameter(name, "sparseLayer");
   }
}

void InputRegionLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   if (ioFlag == PARAMS_IO_READ) {
      mUpdateGpu = false;
      parent->parameters()->handleUnnecessaryParameter(name, "updateGpu");
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
   phase        = mOriginalLayer->getPhase();
   mirrorBCflag = mOriginalLayer->useMirrorBCs();
   valueBC      = mOriginalLayer->getValueBC();
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
      originalObject = originalLayerNameParam->findLinkedObject(mObserverTable.getObjectMap());
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
      if (parent->columnId() == 0) {
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
   return HyPerLayer::allocateDataStructures();
}

void InputRegionLayer::allocateV() { clayer->V = nullptr; }

void InputRegionLayer::allocateActivity() {
   int const numItems = getNumExtendedAllBatches();
   PVLayerCube *cube  = (PVLayerCube *)calloc(pvcube_size(numItems), sizeof(char));
   FatalIf(cube == nullptr, "Unable to allocate PVLayerCube for %s\n", getDescription_c());
   cube->size       = pvcube_size(numItems);
   cube->numItems   = numItems;
   cube->loc        = *getLayerLoc();
   cube->data       = mOriginalLayer->getInputRegionsAllBatchElements();
   clayer->activity = cube;
}

int InputRegionLayer::setActivity() { return PV_SUCCESS; }

int InputRegionLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (parent->columnId() == 0) {
      ErrorLog().printf(
            "%s: layers derived from InputRegionLayer do not have GSyn channels (requireChannel "
            "called "
            "with channel %d)\n",
            getDescription_c(),
            channelNeeded);
   }
   return PV_FAILURE;
}

void InputRegionLayer::allocateGSyn() { pvAssert(GSyn == nullptr); }

bool InputRegionLayer::needUpdate(double timed, double dt) { return false; }

InputRegionLayer::~InputRegionLayer() {}

} /* namespace PV */
