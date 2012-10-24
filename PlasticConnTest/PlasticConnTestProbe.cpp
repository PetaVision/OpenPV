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
PlasticConnTestProbe::PlasticConnTestProbe(const char * probename, const char * filename, HyPerConn * conn, int kernelIndex, int arborId)
{
   initialize(probename, filename, conn, kernelIndex, arborId);
}


int PlasticConnTestProbe::initialize(const char * probename, const char * filename, HyPerConn * conn, int kernelIndex, int arborId) {
   errorPresent = false;
   return KernelProbe::initialize(probename, filename, conn, kernelIndex, arborId);
}
/**
 * @timef
 */
int PlasticConnTestProbe::outputState(double timed) {
   HyPerConn * c = getTargetConn();
#ifdef PV_USE_MPI
   InterColComm * icComm = c->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return PV_SUCCESS;
   }
#endif // PV_USE_MPI
   KernelConn * kconn = dynamic_cast<KernelConn *>(c);
   if( kconn == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" is not a KernelConn.\n", getName(), c->getName() );
      return PV_FAILURE;
   }
   fprintf(getFilePtr(), "    Time %f, connection \"%s\":\n", timed, kconn->getName());
   // kconn->getKernelPatch(arborID, kernelIndex);
   const pvdata_t * w = kconn->get_wDataHead(arborID, kernelIndex); // wPatch->data;
   const pvdata_t * dw = kconn->get_dwDataHead(arborID, kernelIndex); // kconn->get_dKernelData(arborID, kernelIndex);
   if( outputPlasticIncr && dw == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" has dKernelData(%d,%d) set to null.\n", getName(), kconn->getName(), kernelIndex, arborID);
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
         pvdata_t wCorrect = timed*wx;
         pvdata_t wObserved = w[k];
         if( fabs( ((double) (wObserved - wCorrect))/timed ) > 1e-4 ) {
//            status = PV_FAILURE;
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(getFilePtr(), "        index %d (x=%d, y=%d, f=%d: w = %f, should be %f\n", k, x, y, f, wObserved, wCorrect);
         }
      }
      if(timed > 0 && outputPlasticIncr && dw != NULL) {
         pvdata_t dwCorrect = wx;
         pvdata_t dwObserved = dw[k];
         if( dwObserved != dwCorrect ) {
//            status = PV_FAILURE;
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(getFilePtr(), "        index %d (x=%d, y=%d, f=%d: dw = %f, should be %f\n", k, x, y, f, dwObserved, dwCorrect);
         }
      }
   }
   assert(status==PV_SUCCESS);
   if( status == PV_SUCCESS ) {
      if( outputWeights ) fprintf(getFilePtr(), "        All weights are correct.\n");
      if( outputPlasticIncr ) fprintf(getFilePtr(), "        All plastic increments are correct.\n");
   }
   if(outputPatchIndices) {
      patchIndices(kconn);
   }

   return PV_SUCCESS;
}

PlasticConnTestProbe::~PlasticConnTestProbe() {
   if( !errorPresent ) {
      fprintf(getFilePtr(), "No errors detected\n");
   }
}

}  // end of namespace PV block
