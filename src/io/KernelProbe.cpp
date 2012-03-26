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

KernelProbe::KernelProbe(const char * probename, const char * filename, HyPerConn * conn, int kernelIndex, int arborID) {
   initialize_base();
   int status = initialize(probename, filename, conn, kernelIndex, arborID);
   assert(status == PV_SUCCESS);
}

KernelProbe::~KernelProbe() {
}

int KernelProbe::initialize_base() {
   return PV_SUCCESS;
}

int KernelProbe::initialize(const char * probename, const char * filename, HyPerConn * conn, int kernel, int arbor) {
   int status = BaseConnectionProbe::initialize(probename, filename, conn);
   if(status==PV_SUCCESS) {
      PVParams * params = conn->getParent()->parameters();
      kernelIndex = kernel;
      arborID = arbor;
      outputWeights = params->value(probename, "outputWeights", true) != 0.f;
      outputPlasticIncr = params->value(probename, "outputPlasticIncr", false) != 0.f;
      outputPatchIndices = params->value(probename, "outputPatchIndices", false) != 0.f;
      targetKConn = dynamic_cast<KernelConn *>(conn);
      if(targetKConn == NULL) {
         fprintf(stderr, "KernelProbe \"%s\": connection \"%s\" is not a KernelConn.\n", getName(), getTargetConn()->getName());
         status = PV_FAILURE;
      }
   }
   if(getFilePtr()) fprintf(getFilePtr(), "Probe \"%s\", kernel index %d, arbor index %d.\n", getName(), kernelIndex, arborID);
   return status;
}

int KernelProbe::outputState(float timef) {
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetKConn()->getParent()->icCommunicator();
   const int rank = icComm->commRank();
   if( rank != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   if( getTargetKConn() == NULL ) {
      fprintf(stderr, "KernelProbe \"%s\": connection \"%s\" is not a KernelConn.\n", getName(), getTargetConn()->getName() );
      return PV_FAILURE;
   }
   int nxp = getTargetKConn()->xPatchSize();
   int nyp = getTargetKConn()->yPatchSize();
   int nfp = getTargetKConn()->fPatchSize();
   int patchSize = nxp*nyp*nfp;

   const pvdata_t * wdata = getTargetKConn()->get_wDataStart(arborID)+patchSize*kernelIndex;
   const pvdata_t * dwdata = outputPlasticIncr ?
         getTargetKConn()->get_dwDataStart(arborID)+patchSize*kernelIndex : NULL;
   fprintf(getFilePtr(), "Time %f, KernelConn \"%s\", nxp=%d, nyp=%d, nfp=%d\n",
           timef, getTargetKConn()->getName(),nxp, nyp, nfp);
   for(int f=0; f<nfp; f++) {
      for(int y=0; y<nyp; y++) {
         for(int x=0; x<nxp; x++) {
            int k = kIndex(x,y,f,nxp,nyp,nfp);
            fprintf(getFilePtr(), "    x=%d, y=%d, f=%d (index %d):", x, y, f, k);
            if(outputWeights) {
               fprintf(getFilePtr(), "  weight=%f", wdata[k]); // fprintf(fp, "  weight=%f", w->data[k]);
            }
            if(outputPlasticIncr) {
               fprintf(getFilePtr(), "  dw=%f", dwdata[k]); // fprintf(fp, "  dw=%f", dw[k]);
            }
            fprintf(getFilePtr(),"\n");
         }
      }
   }
   if(outputPatchIndices) {
      patchIndices(getTargetKConn());
   }

   return PV_SUCCESS;
}

int KernelProbe::patchIndices(KernelConn * kconn) {
   int nxp = kconn->xPatchSize();
   int nyp = kconn->yPatchSize();
   int nfp = kconn->fPatchSize();
   // int numSynapses = nxp*nyp*nfp;
   // PVPatch * w = kconn->getKernelPatch(arborID, kernelIndex);
   int nPreExt = kconn->getNumWeightPatches();
   assert(nPreExt == kconn->preSynapticLayer()->getNumExtended());
   const PVLayerLoc * loc = kconn->preSynapticLayer()->getLayerLoc();
   int marginWidth = loc->nb;
   int nxPre = loc->nx;
   int nyPre = loc->ny;
   int nfPre = loc->nf;
   int nxPreExt = nxPre+2*marginWidth;
   int nyPreExt = nyPre+2*marginWidth;
   for( int kPre = 0; kPre < nPreExt; kPre++ ) {
      PVPatch * w = kconn->getWeights(kPre,arborID);
      int xOffset = kxPos(w->offset, nxp, nyp, nfp);
      int yOffset = kyPos(w->offset, nxp, nyp, nfp);
      int kxPre = kxPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
      int kyPre = kyPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
      int kfPre = featureIndex(kPre,nxPreExt,nyPreExt,nfPre);
      fprintf(getFilePtr(),"    presynaptic neuron %d (x=%d, y=%d, f=%d) uses kernel index %d, starting at x=%d, y=%d\n",
            kPre, kxPre, kyPre, kfPre, kconn->patchIndexToDataIndex(kPre), xOffset, yOffset);
   /*
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
    */
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block
