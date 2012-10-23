/*
 * LCALIFLateralKernelConn.cpp
 *
 *  Created on: Oct 17, 2012
 *      Author: pschultz
 */

#include "LCALIFLateralKernelConn.hpp"

namespace PV {

LCALIFLateralKernelConn::LCALIFLateralKernelConn()
{
   initialize_base();
}

LCALIFLateralKernelConn::LCALIFLateralKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
}

LCALIFLateralKernelConn::~LCALIFLateralKernelConn()
{
   free(integratedSpikeCount); integratedSpikeCount = NULL;
}

int LCALIFLateralKernelConn::initialize_base() {
   integratedSpikeCount = NULL;
   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights * weightInit) {
   int status = KernelConn::initialize(name, hc, pre, post, filename, weightInit);
   const PVLayerLoc * preloc = pre->getLayerLoc();
   const PVLayerLoc * postloc = post->getLayerLoc();
   int nxpre = preloc->nx; int nxpost = postloc->nx;
   int nypre = preloc->ny; int nypost = postloc->ny;
   int nfpre = preloc->nf; int nfpost = postloc->nf;
   int nbpre = preloc->nb; int nbpost = postloc->nb;
   if (nxpre!=nxpost || nypre!=nypost || nfpre!=nfpost || nbpre!=nbpost) {
      if (parent->columnId()==0) {
         fprintf(stderr, "LCALIFLateralKernelConn: pre- and post-synaptic layers must have the same geometry (including margin width)\n");
         fprintf(stderr, "  Pre:  nx=%d, ny=%d, nf=%d, nb=%d\n", nxpre, nypre, nfpre, nbpre);
         fprintf(stderr, "  Post: nx=%d, ny=%d, nf=%d, nb=%d\n", nxpost, nypost, nfpost, nbpost);
      }
      abort();
   }
   integratedSpikeCount = (float *) malloc(pre->getNumExtended() * sizeof(float));
   for (int k=0; k<pre->getNumExtended(); k++) {
      integratedSpikeCount[k] = getTargetRateKHz(); // Spike counts initialized to target rate
   }
   return status;
}

int LCALIFLateralKernelConn::setParams(PVParams * params) {
   int status = KernelConn::setParams(params);
   integrationTimeConstant = readIntegrationTimeConstant();
   inhibitionTimeConstant = readInhibitionTimeConstant();
   targetRateKHz = 0.001 * readTargetRate();
   return status;
}

int LCALIFLateralKernelConn::update_dW(int axonId) {
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   updateIntegratedSpikeCount();
   float target_rate_sq = getTargetRateKHz()*getTargetRateKHz();
   float dt = parent->getDeltaTime();
   float tauINH = getInhibitionTimeConstant();
   const pvdata_t * preactbuf = integratedSpikeCount;
   const pvdata_t * postactbuf = integratedSpikeCount;

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonId);
      size_t offset = getAPostOffset(kExt, axonId);
      pvdata_t preact = preactbuf[kExt];
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(axonId, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            pvdata_t postact = postactRef[lineoffseta+k];
            dwdata[lineoffsetw + k] += dt/tauINH*(preact*postact-target_rate_sq)/target_rate_sq;
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
   }

   // Divide by (numNeurons/numKernels)
   int divisor = pre->getNumNeurons()/numKernelIndices;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp*nyp*nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] /= divisor;
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::updateWeights(int axonId) {
   if (plasticityFlag) {
      for (int kPre=0; kPre<getNumWeightPatches(); kPre++) {
         const PVPatch * p = getWeights(kPre, axonId);
         pvdata_t * dw_data = get_dwData(axonId,kPre);
         pvdata_t * w_data = get_wData(axonId,kPre);
         int nx = p->nx;
         int ny = p->ny;
         for (int y=0; y<ny; y++) {
            for (int x=0; x<nx; x++) {
               for (int f=0; f<nfp; f++) {
                  int idx = sxp*x + syp*y + sfp*f;
                  pvdata_t w = w_data[idx] + dw_data[idx];
                  if (w<0) w=0;
                  w_data[idx] = w;
               }
            }
         }
      }
   }
   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::updateIntegratedSpikeCount() {
   float exp_dt_tau = exp(-parent->getDeltaTime()/integrationTimeConstant);
   pvdata_t * activity = pre->getActivity();
   for (int kext=0; kext<getNumWeightPatches(); kext++) {
      integratedSpikeCount[kext] = exp_dt_tau * (integratedSpikeCount[kext]+activity[kext]);
   }
   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::checkpointWrite(const char * cpDir) {
   int status = KernelConn::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralKernelConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   write_pvdata(filename, parent->icCommunicator(), (double) parent->simulationTime(), integratedSpikeCount, pre->getLayerLoc(), PV_FLOAT_TYPE, /*extended*/ true, /*contiguous*/ false);
   return status;
}

int LCALIFLateralKernelConn::checkpointRead(const char * cpDir, double * timef) {
   int status = KernelConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralKernelConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   double timed;
   PVLayerLoc loc;
   memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   read_pvdata(filename, parent->icCommunicator(), &timed, integratedSpikeCount, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   return status;
}

} /* namespace PV */
