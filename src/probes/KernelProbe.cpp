/*
 * KernelPactchProbe.cpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#include "KernelProbe.hpp"

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
   auto *targetHyPerConn = getTargetHyPerConn();
   assert(targetHyPerConn);
   if (targetHyPerConn->getSharedWeights() == false) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: %s is not using shared weights.\n",
               getDescription_c(),
               targetHyPerConn->getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

Response::Status KernelProbe::allocateDataStructures() {
   auto status = BaseHyPerConnProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   auto *targetHyPerConn = getTargetHyPerConn();
   assert(targetHyPerConn);
   if (getKernelIndex() < 0 || getKernelIndex() >= targetHyPerConn->getNumDataPatches()) {
      Fatal().printf(
            "KernelProbe \"%s\": kernelIndex %d is out of bounds.  "
            "(min 0, max %d)\n",
            name,
            getKernelIndex(),
            targetHyPerConn->getNumDataPatches() - 1);
   }
   if (getArbor() < 0 || getArbor() >= getTargetHyPerConn()->getNumAxonalArbors()) {
      Fatal().printf(
            "KernelProbe \"%s\": arborId %d is out of bounds. (min 0, max %d)\n",
            name,
            getArbor(),
            getTargetHyPerConn()->getNumAxonalArbors() - 1);
   }

   if (!mOutputStreams.empty()) {
      output(0) << "Probe \"" << name << "\", kernel index " << getKernelIndex() << ", arbor index "
                << getArbor() << ".\n";
   }
   if (getOutputPatchIndices()) {
      patchIndices(targetHyPerConn);
   }

   return Response::SUCCESS;
}

Response::Status KernelProbe::outputState(double timed) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   Communicator *icComm  = parent->getCommunicator();
   const int rank        = icComm->commRank();
   auto *targetHyPerConn = getTargetHyPerConn();
   assert(targetHyPerConn != nullptr);
   int nxp       = targetHyPerConn->getPatchSizeX();
   int nyp       = targetHyPerConn->getPatchSizeY();
   int nfp       = targetHyPerConn->getPatchSizeF();
   int patchSize = nxp * nyp * nfp;

   const float *wdata = targetHyPerConn->getWeightsDataStart(arborID) + patchSize * kernelIndex;
   const float *dwdata =
         outputPlasticIncr
               ? targetHyPerConn->getDeltaWeightsDataStart(arborID) + patchSize * kernelIndex
               : NULL;
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

int KernelProbe::patchIndices(HyPerConn *conn) {
   pvAssert(!mOutputStreams.empty());
   int nxp     = conn->getPatchSizeX();
   int nyp     = conn->getPatchSizeY();
   int nfp     = conn->getPatchSizeF();
   int nPreExt = conn->getNumGeometryPatches();
   assert(nPreExt == conn->getPre()->getNumExtended());
   const PVLayerLoc *loc = conn->getPre()->getLayerLoc();
   const PVHalo *halo    = &loc->halo;
   int nxPre             = loc->nx;
   int nyPre             = loc->ny;
   int nfPre             = loc->nf;
   int nxPreExt          = nxPre + loc->halo.lt + loc->halo.rt;
   int nyPreExt          = nyPre + loc->halo.dn + loc->halo.up;
   for (int kPre = 0; kPre < nPreExt; kPre++) {
      Patch const *patch = conn->getPatch(kPre);
      int xOffset        = kxPos(patch->offset, nxp, nyp, nfp);
      int yOffset        = kyPos(patch->offset, nxp, nyp, nfp);
      int kxPre          = kxPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.lt;
      int kyPre          = kyPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.up;
      int kfPre          = featureIndex(kPre, nxPreExt, nyPreExt, nfPre);
      output(0) << "    presynaptic neuron " << kPre;
      output(0) << " (x=" << kxPre << ", y=" << kyPre << ", f=" << kfPre;
      output(0) << ") uses kernel index " << conn->calcDataIndexFromPatchIndex(kPre);
      output(0) << ", starting at x=" << xOffset << ", y=" << yOffset << "\n";
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
