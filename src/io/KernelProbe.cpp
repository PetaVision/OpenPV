/*
 * KernelPactchProbe.cpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#include "KernelProbe.hpp"

namespace PV {

KernelProbe::KernelProbe() {
   initialize_base();
}

KernelProbe::KernelProbe(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborID) {
   initialize_base();
   initialize(probename, filename, hc, kernelIndex, arborID);
}

KernelProbe::~KernelProbe() {
}

int KernelProbe::initialize_base() {
   return PV_SUCCESS;
}

int KernelProbe::initialize(const char * probename, const char * filename, HyPerCol * hc, int kernel, int arbor) {
   PVParams * params = hc->parameters();
   kernelIndex = kernel;
   arborID = arbor;
   outputWeights = params->value(probename, "outputWeights", true) != 0.f;
   outputPlasticIncr = params->value(probename, "outputPlasticIncr", false) != 0.f;
   outputPatchIndices = params->value(probename, "outputPatchIndices", false) != 0.f;
   BaseConnectionProbe::initialize(probename, filename, hc);
   if(fp) fprintf(fp, "Probe \"%s\", kernel index %d, arbor index %d.\n", name, kernelIndex, arborID);
   return PV_SUCCESS;
}

int KernelProbe::outputState(float time, HyPerConn * c) {
#ifdef PV_USE_MPI
   InterColComm * icComm = c->getParent()->icCommunicator();
   const int rank = icComm->commRank();
   if( rank != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   KernelConn * kconn = dynamic_cast<KernelConn *>(c);
   if( kconn == NULL ) {
      fprintf(stderr, "KernelProbe \"%s\": connection \"%s\" is not a KernelConn.\n", name, c->getName() );
      return PV_FAILURE;
   }
   const PVPatch * w = kconn->getKernelPatch(arborID, kernelIndex);
   const pvdata_t * dw = outputPlasticIncr ? kconn->get_dKernelData(arborID, kernelIndex) : NULL;
   int nxp = kconn->xPatchSize();
   int nyp = kconn->yPatchSize();
   int nfp = kconn->fPatchSize();
   fprintf(fp, "Time %f, KernelConn \"%s\", nxp=%d, nyp=%d, nfp=%d\n",
           time, kconn->getName(),kconn->xPatchSize(), kconn->yPatchSize(), kconn->fPatchSize());
   for(int f=0; f<nfp; f++) {
      for(int y=0; y<nyp; y++) {
         for(int x=0; x<nxp; x++) {
            int k = kIndex(x,y,f,nxp,nyp,nfp);
            fprintf(fp, "    x=%d, y=%d, f=%d (index %d):", x, y, f, k);
            if(outputWeights) {
               fprintf(fp, "  weight=%f", w->data[k]);
            }
            if(outputPlasticIncr) {
               fprintf(fp, "  dw=%f", dw[k]);
            }
            fprintf(fp,"\n");
         }
      }
   }
   if(outputPatchIndices) {
      patchIndices(kconn);
   }

   return PV_SUCCESS;
}

int KernelProbe::patchIndices(KernelConn * kconn) {
   int nxp = kconn->xPatchSize();
   int nyp = kconn->yPatchSize();
   int nfp = kconn->fPatchSize();
   PVPatch * w = kconn->getKernelPatch(arborID, kernelIndex);
   int numSynapses = nxp*nyp*nfp;
   int nPreExt = kconn->numWeightPatches();
   assert(nPreExt == kconn->preSynapticLayer()->getNumExtended());
   const PVLayerLoc * loc = kconn->preSynapticLayer()->getLayerLoc();
   int marginWidth = loc->nb;
   int nxPre = loc->nx;
   int nyPre = loc->ny;
   int nfPre = loc->nf;
   int nxPreExt = nxPre+2*marginWidth;
   int nyPreExt = nyPre+2*marginWidth;
   for( int kPre = 0; kPre < nPreExt; kPre++ ) {
      pvdata_t * hData = kconn->getWeights(kPre, arborID)->data;
      pvdata_t * kData = w->data;
      if( hData >= kData && hData < kData+numSynapses) {
         int offset = hData-kData;
         int xOffset = kxPos(offset,nxp,nyp,nfp);
         int yOffset = kyPos(offset,nyp,nyp,nfp);
         int kxPre = kxPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
         int kyPre = kyPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
         int kfPre = featureIndex(kPre,nxPreExt,nyPreExt,nfPre);
         fprintf(fp,"    presynaptic neuron %d (x=%d, y=%d, f=%d) uses kernel index %d, starting at x=%d, y=%d\n",
                 kPre, kxPre, kyPre, kfPre, kernelIndex, xOffset, yOffset);
      }
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block
