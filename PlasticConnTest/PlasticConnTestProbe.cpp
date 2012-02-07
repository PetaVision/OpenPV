/*
 * PlasticConnTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "PlasticConnTestProbe.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
PlasticConnTestProbe::PlasticConnTestProbe(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborId)
{
   initialize(probename, filename, hc, kernelIndex, arborId);
}


int PlasticConnTestProbe::initialize(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborId) {
   errorPresent = false;
   return KernelProbe::initialize(probename, filename, hc, kernelIndex, arborId);
}
/**
 * @timef
 * @l
 */
int PlasticConnTestProbe::outputState(float timef, HyPerConn * c) {
#ifdef PV_USE_MPI
   InterColComm * icComm = c->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return PV_SUCCESS;
   }
#endif // PV_USE_MPI
   KernelConn * kconn = dynamic_cast<KernelConn *>(c);
   if( kconn == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" is not a KernelConn.\n", name, c->getName() );
      return PV_FAILURE;
   }
   fprintf(fp, "    Time %f, connection \"%s\":\n", timef, kconn->getName());
   const PVPatch * wPatch = kconn->getKernelPatch(arborID, kernelIndex);
   const pvdata_t * w = wPatch->data;
   const pvdata_t * dw = kconn->get_dKernelData(arborID, kernelIndex);
   if( outputPlasticIncr && dw == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" has dKernelData(%d,%d) set to null.\n", name, kconn->getName(), kernelIndex, arborID);
      assert(false);
   }
   int nxp = kconn->xPatchSize();
   int nyp = kconn->yPatchSize();
   int nfp = kconn->fPatchSize();
   int status = PV_SUCCESS;
   for( int k=0; k<nxp*nyp*nfp; k++ ) {
      int x=kxPos(k,nxp,nyp,nfp);
      int wx = (nxp-1)/2 - x; // assumes connection is one-to-one
      if(outputWeights) {
         pvdata_t wCorrect = timef*wx;
         pvdata_t wObserved = w[k];
         if( fabs( (wObserved - wCorrect)/timef ) > 1e-4 ) {
            status = PV_FAILURE;
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: w = %f, should be %f\n", k, x, y, f, wObserved, wCorrect);
         }
      }
      if(timef > 0 && outputPlasticIncr && dw != NULL) {
         pvdata_t dwCorrect = wx;
         pvdata_t dwObserved = dw[k];
         if( dwObserved != dwCorrect ) {
            status = PV_FAILURE;
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: dw = %f, should be %f\n", k, x, y, f, dwObserved, dwCorrect);
         }
      }
   }
   assert(status==PV_SUCCESS);
   if( status == PV_SUCCESS ) {
      if( outputWeights ) fprintf(fp, "        All weights are correct.\n");
      if( outputPlasticIncr ) fprintf(fp, "        All plastic increments are correct.\n");
   }
   if(outputPatchIndices) {
      patchIndices(kconn);
   }

   return PV_SUCCESS;
}

PlasticConnTestProbe::~PlasticConnTestProbe() {
   if( !errorPresent ) {
      fprintf(fp, "No errors detected\n");
   }
}

}  // end of namespace PV block
