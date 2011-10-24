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
 * @time
 * @l
 */
int PlasticConnTestProbe::outputState(float time, HyPerConn * c) {
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
   // const PVPatch * w = kconn->getKernelPatch(arborID, kernelIndex);
   const PVPatch * wPatch = kconn->getKernelPatch(arborID, kernelIndex);
   const pvdata_t * w = wPatch->data;
   const pvdata_t * dw = kconn->get_dKernelData(arborID, kernelIndex);
   if( dw == NULL ) {
      fprintf(stderr, "PlasticConnTestProbe \"%s\": connection \"%s\" has dKernelData(%d,%d) set to null.\n", name, kconn->getName(), kernelIndex, arborID);
      return PV_FAILURE;
   }
   int nxp = kconn->xPatchSize();
   int nyp = kconn->yPatchSize();
   int nfp = kconn->fPatchSize();
   int status = PV_SUCCESS;
   for( int k=0; k<nxp*nyp*nfp; k++ ) {
      if(outputWeights) {
         int x=kxPos(k,nxp,nyp,nfp);
         pvdata_t wCorrect = ( time+kconn->getParent()->getDeltaTime() )*(nxp-1)/2;
         pvdata_t wObserved = w[k];
         if( fabs( (wObserved - wCorrect)/( time+kconn->getParent()->getDeltaTime() ) ) > 1e-4 ) {
            if( status == PV_SUCCESS ) {
               fprintf(fp, "    Time %f, connection \"%s\":\n", time, kconn->getName());
               status = PV_FAILURE;
            }
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: w = %f, should be %f\n", k, x, y, f, wObserved, wCorrect);
         }
      }
      if(outputPlasticIncr) {
         int x=kxPos(k,nxp,nyp,nfp);
         pvdata_t dwCorrect = (nxp-1)/2;
         pvdata_t dwObserved = dw[k];
         if( dwObserved != dwCorrect ) {
            if( status == PV_SUCCESS ) {
               fprintf(fp, "    Time %f, connection \"%s\":\n", time, kconn->getName());
               status = PV_FAILURE;
            }
            int y=kyPos(k,nxp,nyp,nfp);
            int f=featureIndex(k,nxp,nyp,nfp);
            fprintf(fp, "        index %d (x=%d, y=%d, f=%d: dw = %f, should be %f\n", k, x, y, f, dwObserved, dwCorrect);
         }
      }
   }
   assert(status==PV_SUCCESS);

   return PV_SUCCESS;
}

PlasticConnTestProbe::~PlasticConnTestProbe() {
   if( !errorPresent ) {
      fprintf(fp, "No errors detected\n");
   }
}

}  // end of namespace PV block
