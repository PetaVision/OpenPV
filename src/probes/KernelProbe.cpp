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

int KernelProbe::initNumValues() { return setNumValues(-1); }

int KernelProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseHyPerConnProbe::communicateInitInfo(message);
   assert(targetHyPerConn);
   if (getTargetHyPerConn()->usingSharedWeights() == false) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: %s is not using shared weights.\n",
               getDescription_c(),
               targetConn->getDescription_c());
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

int KernelProbe::allocateDataStructures() {
   int status = BaseHyPerConnProbe::allocateDataStructures();
   assert(getTargetConn());
   if (getKernelIndex() < 0 || getKernelIndex() >= getTargetHyPerConn()->getNumDataPatches()) {
      Fatal().printf(
            "KernelProbe \"%s\": kernelIndex %d is out of bounds.  "
            "(min 0, max %d)\n",
            name,
            getKernelIndex(),
            getTargetHyPerConn()->getNumDataPatches() - 1);
   }
   if (getArbor() < 0 || getArbor() >= getTargetConn()->numberOfAxonalArborLists()) {
      Fatal().printf(
            "KernelProbe \"%s\": arborId %d is out of bounds. (min 0, max %d)\n",
            name,
            getArbor(),
            getTargetConn()->numberOfAxonalArborLists() - 1);
   }

   if (!mOutputStreams.empty()) {
      output(0) << "Probe \"" << name << "\", kernel index " << getKernelIndex() << ", arbor index "
                << getArbor() << ".\n";
   }
   if (getOutputPatchIndices()) {
      patchIndices(getTargetHyPerConn());
   }

   return status;
}

int KernelProbe::outputState(double timed) {
   Communicator *icComm = parent->getCommunicator();
   const int rank       = icComm->commRank();
   if (mOutputStreams.empty()) {
      return PV_SUCCESS;
   }
   assert(getTargetConn() != NULL);
   int nxp       = getTargetHyPerConn()->xPatchSize();
   int nyp       = getTargetHyPerConn()->yPatchSize();
   int nfp       = getTargetHyPerConn()->fPatchSize();
   int patchSize = nxp * nyp * nfp;

   const float *wdata = getTargetHyPerConn()->get_wDataStart(arborID) + patchSize * kernelIndex;
   const float *dwdata =
         outputPlasticIncr
               ? getTargetHyPerConn()->get_dwDataStart(arborID) + patchSize * kernelIndex
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

   return PV_SUCCESS;
}

int KernelProbe::patchIndices(HyPerConn *conn) {
   pvAssert(!mOutputStreams.empty());
   int nxp     = conn->xPatchSize();
   int nyp     = conn->yPatchSize();
   int nfp     = conn->fPatchSize();
   int nPreExt = conn->getNumWeightPatches();
   assert(nPreExt == conn->preSynapticLayer()->getNumExtended());
   const PVLayerLoc *loc = conn->preSynapticLayer()->getLayerLoc();
   const PVHalo *halo    = &loc->halo;
   int nxPre             = loc->nx;
   int nyPre             = loc->ny;
   int nfPre             = loc->nf;
   int nxPreExt          = nxPre + loc->halo.lt + loc->halo.rt;
   int nyPreExt          = nyPre + loc->halo.dn + loc->halo.up;
   for (int kPre = 0; kPre < nPreExt; kPre++) {
      PVPatch *w  = conn->getWeights(kPre, arborID);
      int xOffset = kxPos(w->offset, nxp, nyp, nfp);
      int yOffset = kyPos(w->offset, nxp, nyp, nfp);
      int kxPre   = kxPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.lt;
      int kyPre   = kyPos(kPre, nxPreExt, nyPreExt, nfPre) - loc->halo.up;
      int kfPre   = featureIndex(kPre, nxPreExt, nyPreExt, nfPre);
      output(0) << "    presynaptic neuron " << kPre;
      output(0) << " (x=" << kxPre << ", y=" << kyPre << ", f=" << kfPre;
      output(0) << ") uses kernel index " << conn->patchIndexToDataIndex(kPre);
      output(0) << ", starting at x=" << xOffset << ", y=" << yOffset << "\n";
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
