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
PlasticConnTestProbe::PlasticConnTestProbe(const char * probename, HyPerCol * hc)
{
   initialize(probename, hc);
}


int PlasticConnTestProbe::initialize(const char * probename, HyPerCol * hc) {
   errorPresent = false;
   return KernelProbe::initialize(probename, hc);
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
   assert(getTargetConn()!=NULL);
   FILE * fp = getStream()->fp;
   fprintf(fp, "    Time %f, connection \"%s\":\n", timed, getTargetConnName());
   const pvwdata_t * w = getTargetConn()->get_wDataHead(getArbor(), getKernelIndex());
   const pvdata_t * dw = getTargetConn()->get_dwDataHead(getArbor(), getKernelIndex());
   if( getOutputPlasticIncr() && dw == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" has dKernelData(%d,%d) set to null.\n", getName(), getTargetConnName(), getKernelIndex(), getArbor());
      assert(false);
   }
   int nxp = getTargetConn()->xPatchSize();
   int nyp = getTargetConn()->yPatchSize();
   int nfp = getTargetConn()->fPatchSize();
   int status = PV_SUCCESS;
   for( int k=0; k<nxp*nyp*nfp; k++ ) {
      int x=kxPos(k,nxp,nyp,nfp);
      int wx = (nxp-1)/2 - x; // assumes connection is one-to-one
      if(getOutputWeights()) {
         pvdata_t wCorrect = timed*wx;
         pvdata_t wObserved = w[k];
         if( fabs( ((double) (wObserved - wCorrect))/timed ) > 1e-4 ) {
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: w = %f, should be %f\n", k, x, y, f, wObserved, wCorrect);
         }
      }
      if(timed > 0 && getOutputPlasticIncr() && dw != NULL) {
         pvdata_t dwCorrect = wx;
         pvdata_t dwObserved = dw[k];
         if( dwObserved != dwCorrect ) {
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: dw = %f, should be %f\n", k, x, y, f, dwObserved, dwCorrect);
         }
      }
   }
   assert(status==PV_SUCCESS);
   if( status == PV_SUCCESS ) {
      if( getOutputWeights() )      fprintf(fp, "        All weights are correct.\n");
      if( getOutputPlasticIncr() ) fprintf(fp, "        All plastic increments are correct.\n");
   }
   if(getOutputPatchIndices()) {
      patchIndices(getTargetConn());
   }

   return PV_SUCCESS;
}

PlasticConnTestProbe::~PlasticConnTestProbe() {
   if( !errorPresent ) {
      fprintf(getStream()->fp, "No errors detected\n");
   }
}

}  // end of namespace PV block
