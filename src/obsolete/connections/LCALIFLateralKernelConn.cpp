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

LCALIFLateralKernelConn::LCALIFLateralKernelConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

LCALIFLateralKernelConn::~LCALIFLateralKernelConn()
{
   pvcube_delete(integratedSpikeCountCube); integratedSpikeCountCube = NULL; integratedSpikeCount = NULL;
   Communicator::freeDatatypes(mpi_datatype); mpi_datatype = NULL;
   free(interiorCounts[0]);
   free(interiorCounts); interiorCounts = NULL;
}

int LCALIFLateralKernelConn::initialize_base() {
   integratedSpikeCountCube = NULL;
   integratedSpikeCount = NULL;
   mpi_datatype = NULL;
   interiorCounts = NULL;
   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int LCALIFLateralKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_integrationTimeConstant(ioFlag);
   ioParam_inhibitionTimeConstant(ioFlag);
   ioParam_targetRate(ioFlag);
   ioParam_dWUpdatePeriod(ioFlag);
   return status;
}

// TODO: can we eliminate this class altogether and just use LCALIFLateralConn with sharedWeights = true?
void LCALIFLateralKernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void LCALIFLateralKernelConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   HyPerConn::ioParam_initialWeightUpdateTime(ioFlag);
   if (ioFlag==PARAMS_IO_READ) {
      dWUpdateTime = weightUpdateTime;
   }
}

void LCALIFLateralKernelConn::ioParam_integrationTimeConstant(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "integrationTimeConstant", &integrationTimeConstant, 1.0f);
}

void LCALIFLateralKernelConn::ioParam_inhibitionTimeConstant(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inhibitionTimeConstant", &inhibitionTimeConstant, 1.0f);
}

void LCALIFLateralKernelConn::ioParam_targetRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "targetRate", &targetRateHz, 1.0f);
   if (ioFlag==PARAMS_IO_READ) targetRateKHz = 0.001 * targetRateHz;
}

void LCALIFLateralKernelConn::ioParam_dWUpdatePeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dWUpdatePeriod", &dWUpdatePeriod, 1.0);
}

int LCALIFLateralKernelConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   const PVLayerLoc * preloc = pre->getLayerLoc();
   const PVLayerLoc * postloc = post->getLayerLoc();
   int nxpre = preloc->nx; int nxpost = postloc->nx;
   int nypre = preloc->ny; int nypost = postloc->ny;
   int nfpre = preloc->nf; int nfpost = postloc->nf;
   const PVHalo * haloPre = &preloc->halo; const PVHalo * haloPost = &postloc->halo;
   if (nxpre!=nxpost || nypre!=nypost || nfpre!=nfpost ||
       haloPre->lt!=haloPost->lt || haloPre->rt!=haloPost->rt || haloPre->dn!=haloPost->dn || haloPre->up!=haloPost->dn) {
      if (parent->columnId()==0) {
         fprintf(stderr, "LCALIFLateralKernelConn: pre- and post-synaptic layers must have the same geometry (including margin widths)\n");
         fprintf(stderr, "  Pre:  nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n", nxpre, nypre, nfpre, haloPre->lt, haloPre->rt, haloPre->dn, haloPre->up);
         fprintf(stderr, "  Post: nx=%d, ny=%d, nf=%d, nb=(%d,%d,%d,%d)\n", nxpost, nypost, nfpost, haloPost->lt, haloPost->rt, haloPost->dn, haloPost->up);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      abort();
   }
  return PV_SUCCESS;
}

int LCALIFLateralKernelConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();

   // Neurons don't inhibit themselves, only their neighbors; set self-interaction weights to mmzero.
   assert(nxp % 2 == 1 && nyp % 2 == 1 && getNumDataPatches()==nfp);
   for (int k=0; k<getNumDataPatches(); k++) {
      int n = kIndex((nxp-1)/2, (nyp-1)/2, k, nxp, nyp, nfp);
      get_wDataHead(0, k)[n] = 0.0f;
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

   // Compute the number of times each patch contributes to dw, for proper averaging.
   int num_arbors = numberOfAxonalArborLists();
   interiorCounts = (float **) calloc(num_arbors, sizeof(float *));
   if (interiorCounts==NULL) {
      fprintf(stderr, "LCALIFLateralKernelConn::initialize \"%s\" error: unable to allocate memory for interiorCounts pointer\n", name);
   }
   interiorCounts[0] = (float *) calloc(getNumDataPatches()*nxp*nyp*nfp, sizeof(float));
   if (interiorCounts[0]==NULL) {
      fprintf(stderr, "LCALIFLateralKernelConn::initialize \"%s\" error: unable to allocate memory for interiorCounts\n", name);
   }
   for (int arbor=1; arbor<num_arbors; arbor++) {
      interiorCounts[arbor] = interiorCounts[0]+arbor*getNumDataPatches()*nxp*nyp*nfp;
   }

   const PVLayerLoc * preloc = pre->getLayerLoc();
   int nxpre = preloc->nx;
   int nypre = preloc->ny;
   int nfpre = preloc->nf;

   int nExt = pre->getNumExtended();
   int sya = getPostExtStrides()->sy;
   int nxglob = preloc->nxGlobal;
   int nyglob = preloc->nyGlobal;
   int kx0 = preloc->kx0;
   int ky0 = preloc->ky0;
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      for(int kExt=0; kExt<nExt;kExt++) {
         int xglob = kxPos(kExt, nxpre + preloc->halo.lt + preloc->halo.rt, nypre + preloc->halo.dn + preloc->halo.up, nfpre) + kx0 - preloc->halo.lt;
         int yglob = kyPos(kExt, nypre + preloc->halo.lt + preloc->halo.rt, nypre + preloc->halo.dn + preloc->halo.up, nfpre) + ky0 - preloc->halo.up;
         if (xglob < 0 || xglob >= nxglob || yglob < 0 || yglob >= nyglob) {
            continue;
         }
         PVPatch * weights = getWeights(kExt,arbor);
         int offset = (int) getAPostOffset(kExt, arbor);
         int ny = weights->ny;
         int nk = weights->nx * nfp;
         int interiorCountOffset = get_wData(arbor, kExt)-get_wDataStart(arbor);
         int lineoffsetw = 0;
         int lineoffseta = 0;
         for( int y=0; y<ny; y++ ) {
            for( int k=0; k<nk; k++ ) {
               int postactindex = offset+lineoffseta+k;
               if (postactindex != kExt) { // Neurons don't inhibit themselves
                  interiorCounts[arbor][interiorCountOffset + lineoffsetw + k]++;
               }
            }
            lineoffsetw += syp;
            lineoffseta += sya;
         }
      }
   }
   int bufsize = numberOfAxonalArborLists() * getNumDataPatches() * nxp * nyp * nfp;
// TODO-CER-2014.3.26 - Ensure that reduction is done when not using MPI
#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, interiorCounts[0], bufsize, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif

   return status;
}

int LCALIFLateralKernelConn::update_dW(int axonId) {
   if (parent->simulationTime() < dWUpdateTime) {
      return PV_SUCCESS;
   }
   dWUpdateTime += dWUpdatePeriod;
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   updateIntegratedSpikeCount();
   float target_rate_sq = getTargetRateKHz()*getTargetRateKHz();
   const pvdata_t * preactbuf = integratedSpikeCount;
   const pvdata_t * postactbuf = integratedSpikeCount;

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt));

   const PVLayerLoc * preloc = pre->getLayerLoc();
   int nxpre = preloc->nx;
   int nypre = preloc->ny;
   int nfpre = preloc->nf;
   int nxglob = preloc->nxGlobal;
   int nyglob = preloc->nyGlobal;
   int kx0 = preloc->kx0;
   int ky0 = preloc->ky0;
   for(int kExt=0; kExt<nExt;kExt++) {
      int xglob = kxPos(kExt, nxpre + preloc->halo.lt + preloc->halo.rt, nypre + preloc->halo.dn + preloc->halo.up, nfpre) + kx0 - preloc->halo.lt;
      int yglob = kyPos(kExt, nxpre + preloc->halo.lt + preloc->halo.rt, nypre + preloc->halo.dn + preloc->halo.up, nfpre) + ky0 - preloc->halo.dn;
      if (xglob < 0 || xglob >= nxglob || yglob < 0 || yglob >= nyglob) {
         continue;
      }
      PVPatch * weights = getWeights(kExt,axonId);
      size_t offset = getAPostOffset(kExt, axonId);
      pvdata_t preactrate = preactbuf[kExt]/integrationTimeConstant;
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvwdata_t * dwdata = get_dwData(axonId, kExt);
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
   // Divide each dw by the number of correlations that contributed to that dw (divisorptr was summed over all MPI processes in initialization).
   // Also divide by target_rate_sq to normalize to a dimensionless quantity.
   // The nonlinear filter and the multiplication by dt/tauINH takes place in updateWeights, because the filter has to be applied after reduceKernels
   // and the multiplication by dt/tauINH needs to take place after the filter.
   int patch_size = nxp*nyp*nfp;
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      pvwdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      float * divisorptr = &interiorCounts[axonId][kernelindex*patch_size];
      for( int n=0; n<patch_size; n++ ) {
         assert(divisorptr[n]>0 || dwpatchdata[n]==0);
         if (divisorptr[n]>0) dwpatchdata[n] /= target_rate_sq * divisorptr[n];
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}

int LCALIFLateralKernelConn::updateWeights(int axonId) {
   if (plasticityFlag) {
      float normalizer = parent->getDeltaTime()/getInhibitionTimeConstant();
      for (int kernel=0; kernel<getNumDataPatches(); kernel++) {
         pvwdata_t * dw_data = get_dwDataHead(axonId,kernel);
         pvwdata_t * w_data = get_wDataHead(axonId,kernel);
         for (int y=0; y<nyp; y++) {
            for (int x=0; x<nxp; x++) {
               for (int f=0; f<nfp; f++) {
                  int idx = sxp*x + syp*y + sfp*f;
                  pvdata_t dw = dw_data[idx] * normalizer;
                  pvwdata_t w = w_data[idx] + (weightUpdatePeriod/parent->getDeltaTime())*dw;
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
   int status = HyPerConn::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralKernelConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   int status2 = HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), parent->simulationTime(), &integratedSpikeCount, 1/*numbands*/, false/*extended*/, pre->getLayerLoc());
   if (status2!=PV_SUCCESS) status = status2;

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_dW.pvp", cpDir, name);
   assert(chars_needed < PV_PATH_MAX);
   status2 = HyPerConn::writeWeights(NULL, get_dwDataStart(), getNumDataPatches(), filename, parent->simulationTime(), writeCompressedCheckpoints, true);
   if (status2!=PV_SUCCESS) status = status2;
   return status;
}

int LCALIFLateralKernelConn::checkpointRead(const char * cpDir, double * timef) {
   int status = HyPerConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralKernelConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   double timed;
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &integratedSpikeCount, 1/*numbands*/, /*extended*/ true, pre->getLayerLoc());
   // read_pvdata(filename, parent->icCommunicator(), &timed, integratedSpikeCount, pre->getLayerLoc(), PV_FLOAT_TYPE, /*extended*/ true, /*contiguous*/ false);
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
