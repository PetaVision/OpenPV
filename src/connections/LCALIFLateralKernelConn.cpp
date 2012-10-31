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
   pvcube_delete(integratedSpikeCountCube); integratedSpikeCountCube = NULL; integratedSpikeCount = NULL;
   Communicator::freeDatatypes(mpi_datatype); mpi_datatype = NULL;
}

int LCALIFLateralKernelConn::initialize_base() {
   integratedSpikeCountCube = NULL;
   integratedSpikeCount = NULL;
   mpi_datatype = NULL;
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
   integratedSpikeCountCube = pvcube_new(pre->getLayerLoc(), pre->getNumExtended());
   integratedSpikeCount = integratedSpikeCountCube->data;
   for (int k=0; k<pre->getNumExtended(); k++) {
      integratedSpikeCount[k] = integrationTimeConstant*getTargetRateKHz(); // Spike counts initialized to equilibrium value
   }
   mpi_datatype = Communicator::newDatatypes(pre->getLayerLoc());
   if (mpi_datatype==NULL) {
      fprintf(stderr, "LCALIFLateralKernelConn \"%s\" error creating mpi_datatype\n", name);
      abort();
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
   float dt = (float) parent->getDeltaTime();
   float tauINH = getInhibitionTimeConstant();
   const pvdata_t * preactbuf = integratedSpikeCount;
   const pvdata_t * postactbuf = integratedSpikeCount;

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonId);
      size_t offset = getAPostOffset(kExt, axonId);
      pvdata_t preactrate = preactbuf[kExt]/integrationTimeConstant;
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvdata_t * dwdata = get_dwData(axonId, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            int postactindex = offset+lineoffseta+k;
            if (postactindex != kExt) { // Neurons don't inhibit themselves
               pvdata_t postactrate = postactbuf[postactindex]/integrationTimeConstant;
               pvdata_t dw = preactrate*postactrate-target_rate_sq;
               dwdata[lineoffsetw + k] += dw;
            }
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
   }

   // Normalize by dt/tauINH/(targetrate^2) and divide by (numNeurons/numKernels)
   int divisor = pre->getNumNeurons()/numKernelIndices;
   float normalizer = dt/tauINH/target_rate_sq/divisor;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp*nyp*nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] *= normalizer;
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::updateWeights(int axonId) {
   if (plasticityFlag) {
      for (int kernel=0; kernel<getNumDataPatches(); kernel++) {
         pvdata_t * dw_data = get_dwDataHead(axonId,kernel);
         pvdata_t * w_data = get_wDataHead(axonId,kernel);
         for (int y=0; y<nyp; y++) {
            for (int x=0; x<nxp; x++) {
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
   const pvdata_t * activity = pre->getLayerData(); // pre->getActivity();
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
   read_pvdata(filename, parent->icCommunicator(), &timed, integratedSpikeCount, pre->getLayerLoc(), PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   // Exchange borders
   if ( pre->useMirrorBCs() ) {
      for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
         pre->mirrorInteriorToBorder(borderId, integratedSpikeCountCube, integratedSpikeCountCube);
      }
   }
   parent->icCommunicator()->exchange(integratedSpikeCount, mpi_datatype, pre->getLayerLoc());

   return status;
}

} /* namespace PV */
