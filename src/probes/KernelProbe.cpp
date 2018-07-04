/*
 * KernelPactchProbe.cpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#include "KernelProbe.hpp"
#include "components/SharedWeights.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

KernelProbe::KernelProbe() { initialize_base(); }

KernelProbe::KernelProbe(const char *probename, HyPerCol *hc) {
   initialize_base();
   int status = initialize(probename, hc);
   assert(status == PV_SUCCESS);
}

KernelProbe::~KernelProbe() {}

int KernelProbe::initialize_base() { return PV_SUCCESS; }

int KernelProbe::initialize(const char *probename, HyPerCol *hc) {
   int status = BaseConnectionProbe::initialize(probename, hc);
   assert(name && parent);

   return status;
}

int KernelProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseConnectionProbe::ioParamsFillGroup(ioFlag);
   ioParam_kernelIndex(ioFlag);
   ioParam_arborId(ioFlag);
   ioParam_outputWeights(ioFlag);
   ioParam_outputPlasticIncr(ioFlag);
   ioParam_outputPatchIndices(ioFlag);
   return status;
}

void KernelProbe::ioParam_kernelIndex(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "kernelIndex", &kernelIndex, 0);
}

void KernelProbe::ioParam_arborId(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "arborId", &arborID, 0);
}

void KernelProbe::ioParam_outputWeights(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "outputWeights", &outputWeights, true /*default value*/);
}

void KernelProbe::ioParam_outputPlasticIncr(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "outputPlasticIncr", &outputPlasticIncr, false /*default value*/);
}

void KernelProbe::ioParam_outputPatchIndices(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "outputPatchIndices", &outputPatchIndices, false /*default value*/);
}

void KernelProbe::initNumValues() { setNumValues(-1); }

Response::Status
KernelProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseHyPerConnProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mTargetConn);

   auto *sharedWeights = mTargetConn->getComponentByType<SharedWeights>();
   FatalIf(
         sharedWeights == nullptr,
         "%s target connection \"%s\" does not have a SharedWeights component.\n",
         getDescription_c(),
         mTargetConn->getName());

   if (!sharedWeights->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until target connection \"%s\" has finished its CommunicateInitInfo "
               "stage.\n",
               getDescription_c(),
               mTargetConn->getName());
      }
      return Response::POSTPONE;
   }
   if (sharedWeights->getSharedWeights() == false) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: %s is not using shared weights.\n",
               getDescription_c(),
               mTargetConn->getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   mPatchSize = mTargetConn->getComponentByType<PatchSize>();
   FatalIf(
         mPatchSize == nullptr,
         "%s target connection \"%s\" does not have a PatchSize component.\n",
         getDescription_c(),
         mTargetConn->getName());

   return Response::SUCCESS;
}

Response::Status KernelProbe::allocateDataStructures() {
   auto status = BaseHyPerConnProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   auto *arborList = mTargetConn->getComponentByType<ArborList>();
   if (getArbor() < 0 || getArbor() >= arborList->getNumAxonalArbors()) {
      Fatal().printf(
            "KernelProbe \"%s\" arborId %d is out of bounds. (min 0, max %d)\n",
            name,
            getArbor(),
            arborList->getNumAxonalArbors() - 1);
   }

   if (getKernelIndex() < 0 || getKernelIndex() >= mWeights->getNumDataPatches()) {
      Fatal().printf(
            "KernelProbe \"%s\" kernelIndex %d is out of bounds.  "
            "(min 0, max %d)\n",
            name,
            getKernelIndex(),
            mWeights->getNumDataPatches() - 1);
   }

   pvAssert(mWeights); // Was set in CommunicateInitInfo stage
   if (!mWeights->getDataStructuresAllocatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until target connection \"%s\" has finished its "
               "AllocateDataStructures stage.\n",
               getDescription_c(),
               mTargetConn->getName());
      }
      return Response::POSTPONE;
   }
   mWeightData = mWeights->getDataReadOnly(getArbor());

   if (outputPlasticIncr) {
      auto *hebbianUpdater = mTargetConn->getComponentByType<HebbianUpdater>();
      FatalIf(
            hebbianUpdater == nullptr,
            "%s target connection \"%s\" does not have a HebbianUpdater component.\n",
            getDescription_c(),
            mTargetConn->getName());
      FatalIf(
            !hebbianUpdater->getPlasticityFlag(),
            "%s target connection \"%s\" is not plastic, but outputPlasticIncr is set.\n",
            getDescription_c(),
            mTargetConn->getName());
      if (!hebbianUpdater->getDataStructuresAllocatedFlag()) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            InfoLog().printf(
                  "%s must wait until target connection \"%s\" has finished its "
                  "AllocateDataStructures stage.\n",
                  getDescription_c(),
                  mTargetConn->getName());
         }
         return Response::POSTPONE;
      }
      mDeltaWeightData = hebbianUpdater->getDeltaWeightsDataStart(getArbor());
   }

   if (!mOutputStreams.empty()) {
      output(0) << "Probe \"" << name << "\", kernel index " << getKernelIndex() << ", arbor index "
                << getArbor() << ".\n";
   }
   if (getOutputPatchIndices()) {
      patchIndices();
   }

   return Response::SUCCESS;
}

Response::Status KernelProbe::outputState(double timed) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   Communicator *icComm = parent->getCommunicator();
   const int rank       = icComm->commRank();
   int nxp              = getPatchSize()->getPatchSizeX();
   int nyp              = getPatchSize()->getPatchSizeY();
   int nfp              = getPatchSize()->getPatchSizeF();
   int patchSize        = nxp * nyp * nfp;

   const float *wdata  = mWeightData + patchSize * kernelIndex;
   const float *dwdata = outputPlasticIncr ? mDeltaWeightData + patchSize * kernelIndex : nullptr;
   output(0) << "Time " << timed << ", Conn \"" << getTargetConn()->getName() << ", nxp=" << nxp
             << ", nyp=" << nyp << ", nfp=" << nfp << "\n";
   for (int f = 0; f < nfp; f++) {
      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            int k = kIndex(x, y, f, nxp, nyp, nfp);
            output(0) << "    x=" << x << ", y=" << y << ", f=" << f << " (index " << k << "):";
            if (getOutputWeights()) {
               output(0) << "  weight=" << wdata[k];
            }
            if (getOutputPlasticIncr()) {
               output(0) << "  dw=" << dwdata[k];
            }
            output(0) << "\n";
         }
      }
   }

   return Response::SUCCESS;
}

int KernelProbe::patchIndices() {
   pvAssert(!mOutputStreams.empty());
   int nxp = getPatchSize()->getPatchSizeX();
   int nyp = getPatchSize()->getPatchSizeY();
   int nfp = getPatchSize()->getPatchSizeF();

   auto geometry         = mWeights->getGeometry();
   int nPreExt           = geometry->getNumPatches();
   const PVLayerLoc *loc = &geometry->getPreLoc();
   int nxPre             = loc->nx;
   int nyPre             = loc->ny;
   int nfPre             = loc->nf;
   int nxPreExt          = nxPre + loc->halo.lt + loc->halo.rt;
   int nyPreExt          = nyPre + loc->halo.dn + loc->halo.up;
   for (int kPre = 0; kPre < nPreExt; kPre++) {
      Patch const &patch = mWeights->getPatch(kPre);
      int const offset   = patch.offset;
      int xOffset        = kxPos(offset, nxp, nyp, nfp);
      int yOffset        = kyPos(offset, nxp, nyp, nfp);
      int kxPre          = kxPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.lt;
      int kyPre          = kyPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.up;
      int kfPre          = featureIndex(kPre, nxPreExt, nyPreExt, nfPre);
      output(0) << "    presynaptic neuron " << kPre;
      output(0) << " (x=" << kxPre << ", y=" << kyPre << ", f=" << kfPre;
      output(0) << ") uses kernel index " << mWeights->calcDataIndexFromPatchIndex(kPre);
      output(0) << ", starting at x=" << xOffset << ", y=" << yOffset << "\n";
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
