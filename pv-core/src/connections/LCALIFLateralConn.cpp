/*
 * LCALIFLateralConn.cpp
 *
 *  Created on: Oct 3, 2012
 *      Author: pschultz
 */

#include "LCALIFLateralConn.hpp"

namespace PV {

LCALIFLateralConn::LCALIFLateralConn()
{
   initialize_base();
}

LCALIFLateralConn::LCALIFLateralConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   initialize_base();
   initialize(name, hc);
}

LCALIFLateralConn::~LCALIFLateralConn()
{
   pvcube_delete(integratedSpikeCountCube); integratedSpikeCountCube = NULL; integratedSpikeCount = NULL;
   Communicator::freeDatatypes(mpi_datatype); mpi_datatype = NULL;
}

int LCALIFLateralConn::initialize_base() {
   integratedSpikeCountCube = NULL;
   integratedSpikeCount = NULL;
   corrThresh = 1;
   return PV_SUCCESS;
}

int LCALIFLateralConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   return HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
}

int LCALIFLateralConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   PVParams * params = parent->parameters();
   ioParam_integrationTimeConstant(ioFlag);
   ioParam_inhibitionTimeConstant(ioFlag);
   ioParam_targetRate(ioFlag);
   ioParam_coorThresh(ioFlag);
   ioParam_dWMax(ioFlag);
   return status;
}

void LCALIFLateralConn::ioParam_integrationTimeConstant(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "integrationTimeConstant", &integrationTimeConstant, 1.0f);
}

void LCALIFLateralConn::ioParam_inhibitionTimeConstant(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inhibitionTimeConstant", &inhibitionTimeConstant, 1.0f);
}

void LCALIFLateralConn::ioParam_targetRate(enum ParamsIOFlag ioFlag) {
   float target_rate_hertz;
   if (ioFlag==PARAMS_IO_WRITE) target_rate_hertz = 1000.0f * targetRateKHz;
   parent->ioParamValue(ioFlag, name, "targetRate", &target_rate_hertz, 1.0f);
   if (ioFlag==PARAMS_IO_READ) targetRateKHz = 0.001*target_rate_hertz;
}

void LCALIFLateralConn::ioParam_coorThresh(enum ParamsIOFlag ioFlag) { // Should it be coorThresh or corrThresh?
   parent->ioParamValue(ioFlag, name, "coorThresh", &corrThresh, corrThresh);
}

int LCALIFLateralConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();
   assert(pre && post);
   const PVLayerLoc * preloc = pre->getLayerLoc();
   const PVLayerLoc * postloc = post->getLayerLoc();
   int nxpre = preloc->nx; int nxpost = postloc->nx;
   int nypre = preloc->ny; int nypost = postloc->ny;
   int nfpre = preloc->nf; int nfpost = postloc->nf;
   if (nxpre!=nxpost || nypre!=nypost || nfpre!=nfpost ||
       preloc->halo.lt!=postloc->halo.lt ||
       preloc->halo.rt!=postloc->halo.rt ||
       preloc->halo.dn!=postloc->halo.dn ||
       preloc->halo.up!=postloc->halo.up) {
      if (parent->columnId()==0) {
         fprintf(stderr, "LCALIFLateralConn: pre- and post-synaptic layers must have the same geometry (including margin width)\n");
         fprintf(stderr, "  Pre:  nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n", nxpre, nypre, nfpre, preloc->halo.lt, preloc->halo.rt, preloc->halo.dn, preloc->halo.up);
         fprintf(stderr, "  Post: nx=%d, ny=%d, nf=%d, nb=(%d,%d,%d,%d)\n", nxpost, nypost, nfpost, postloc->halo.lt, postloc->halo.rt, postloc->halo.dn, postloc->halo.up
);
      }
      status = PV_FAILURE;
   }
#ifdef PV_USE_MPI
   MPI_Barrier(parent->icCommunicator()->communicator());
#endif
   if (status!=PV_SUCCESS) {
      abort();
   }
   return status;
}

int LCALIFLateralConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   integratedSpikeCountCube = pvcube_new(pre->getLayerLoc(), pre->getNumExtended());
   integratedSpikeCount = integratedSpikeCountCube->data;
   memset(integratedSpikeCount, 0, pre->getNumExtended()*sizeof(*integratedSpikeCount)); // Spike counts initialized to 0

   //Loop through patches setting the self to self connection to 0
   //pvdata_t * gSyn_buffer_start = post->getChannel(channel);

   const PVPatchStrides * strides_restricted  = getPostNonextStrides();
   const int sx_restricted = strides_restricted->sx;
   const int sy_restricted = strides_restricted->sy;
   const int sf = strides_restricted->sf;

   for (int axonId = 0; axonId < numberOfAxonalArborLists(); axonId++){
      for (int kPre_extended=0; kPre_extended<getNumWeightPatches(); kPre_extended++) {

//         pvdata_t * gSyn_patch_start = getGSynPatchStart(kPre_extended, axonId);
//         int start_index_restricted = gSyn_patch_start - gSyn_buffer_start;
         int start_index_restricted = getGSynPatchStart(kPre_extended, axonId);

         const PVPatch * p = getWeights(kPre_extended, axonId);
         int nx_patch = p->nx;
         int ny_patch = p->ny;

         pvwdata_t * w_data = get_wData(axonId,kPre_extended);

         const PVLayerLoc * postloc = post->getLayerLoc();
         int nxpost = postloc->nx;
         int nypost = postloc->ny;
         int nfpost = postloc->nf;
         const PVHalo * halopost = &(postloc->halo);
         for (int ky_patch=0; ky_patch<ny_patch; ky_patch++) {
            for (int kx_patch=0; kx_patch<nx_patch; kx_patch++) {
               for (int kf_patch=0; kf_patch<nfp; kf_patch++) {
                  //Calculate indices of post weights
                  int kPost_restricted = start_index_restricted + sy_restricted*ky_patch + sx_restricted*kx_patch + sf*kf_patch;
                  int kPost_extended = kIndexExtended(kPost_restricted, nxpost, nypost, nfpost, halopost->lt, halopost->rt, halopost->dn, halopost->up);
                  int k_patch = sxp*kx_patch + syp*ky_patch + sfp*kf_patch;
                  if (kPost_extended == kPre_extended) {
                     w_data[k_patch] = 0;
                  }
               }
            }
         }
      }
   }

   // If reading from a checkpoint, loading the integratedSpikeCount requires exchanging border regions
   mpi_datatype = Communicator::newDatatypes(pre->getLayerLoc());
   if (mpi_datatype==NULL) {
      fprintf(stderr, "LCALIFLateralKernelConn \"%s\" error creating mpi_datatype\n", name);
      abort();
   }

   return status;
}

int LCALIFLateralConn::calc_dW(int axonId) {
   assert(axonId>=0 && axonId < numberOfAxonalArborLists());
   updateIntegratedSpikeCount();
   // pvdata_t * gSyn_buffer_start = post->getChannel(channel);
   pvdata_t target_rate_sq = getTargetRateKHz() * getTargetRateKHz();
   float dt_inh = parent->getDeltaTime()/inhibitionTimeConstant;
   const PVPatchStrides * strides_restricted  = getPostNonextStrides();
   const int sx_restricted = strides_restricted->sx;
   const int sy_restricted = strides_restricted->sy;
   const int sf = strides_restricted->sf;
   const PVLayerLoc * postloc = post->getLayerLoc();
   const int nxpost = postloc->nx;
   const int nypost = postloc->ny;
   const int nfpost = postloc->nf;
   const PVHalo * halopost = &(postloc->halo);

   for (int kPre_extended=0; kPre_extended<getNumWeightPatches(); kPre_extended++) {
      // The weight p(x,y,f) connects a presynaptic neuron in extended space to a postsynaptic neuron in restricted space.
      // We need to get the indices.  The presynaptic index is k.  To get the postsynaptic index, find
      // The memory location this weight is mapped to and subtract it from the start of the postsynaptic GSyn buffer.
//      pvdata_t * gSyn_patch_start = getGSynPatchStart(kPre_extended, axonId);
//      int start_index_restricted = gSyn_patch_start - gSyn_buffer_start;
      int start_index_restricted = getGSynPatchStart(kPre_extended, axonId);
      const PVPatch * p = getWeights(kPre_extended, axonId);
      pvwdata_t * dw_data = get_dwData(axonId,kPre_extended);
      int nx_patch = p->nx;
      int ny_patch = p->ny;
      for (int ky_patch=0; ky_patch<ny_patch; ky_patch++) {
         for (int kx_patch=0; kx_patch<nx_patch; kx_patch++) {
            for (int kf_patch=0; kf_patch<nfp; kf_patch++) {
               //Calculate indicies of post weights
               int kPost_restricted = start_index_restricted + sy_restricted*ky_patch + sx_restricted*kx_patch + sf*kf_patch;
               int kPost_extended = kIndexExtended(kPost_restricted, nxpost, nypost, nfpost, halopost->lt, halopost->rt, halopost->dn, halopost->up);
               if (kPost_extended != kPre_extended) { //Neuron shouldn't inhibit itself
                  pvdata_t delta_weight;
                  float pre_scale_dt_weight = (1/target_rate_sq) * ((integratedSpikeCount[kPre_extended]/integrationTimeConstant)
                        * (integratedSpikeCount[kPost_extended]/integrationTimeConstant) - target_rate_sq);
                  //Check to see if decorrelation is low enough to stop changing delta_weight
                  if (pre_scale_dt_weight > 0 && pre_scale_dt_weight < corrThresh){
                     delta_weight = 0;
                  }
                  else if(pre_scale_dt_weight >= corrThresh){
                     delta_weight = dt_inh * (pre_scale_dt_weight - corrThresh);
                  }
                  else{
                     delta_weight = dt_inh * pre_scale_dt_weight;
                  }
                  dw_data[sxp*kx_patch + syp*ky_patch + sfp*kf_patch] = delta_weight;
               }
            }
         }
      }
   }
   return PV_SUCCESS;
}

int LCALIFLateralConn::updateWeights(int axonId) {
   if (plasticityFlag) {
      for (int kPre_extended=0; kPre_extended<getNumWeightPatches(); kPre_extended++) {
         const PVPatch * p = getWeights(kPre_extended, axonId);
         pvwdata_t * dw_data = get_dwData(axonId,kPre_extended);
         pvwdata_t *  w_data = get_wData(axonId,kPre_extended);
         int nx_patch = p->nx;
         int ny_patch = p->ny;
         for (int ky_patch=0; ky_patch<ny_patch; ky_patch++) {
            for (int kx_patch=0; kx_patch<nx_patch; kx_patch++) {
               for (int kf_patch=0; kf_patch<nfp; kf_patch++) {
                  int k_patch = sxp*kx_patch + syp*ky_patch + sfp*kf_patch;
                  pvwdata_t w = w_data[k_patch] + dw_data[k_patch];
                  if (w<0) w=0;
                  w_data[k_patch] = w;
               }
            }
         }
      }
   }
   return PV_SUCCESS;
}

int LCALIFLateralConn::updateIntegratedSpikeCount() {
   float exp_dt_tau = exp(-parent->getDeltaTime()/integrationTimeConstant);
   const pvdata_t * activity = pre->getLayerData();// pre->getActivity(); // Gar owes Sheng another beer because of this bug. --Pete
   for (int kext=0; kext<getNumWeightPatches(); kext++) {
      integratedSpikeCount[kext] = exp_dt_tau * (integratedSpikeCount[kext]+activity[kext]);
   }
   return PV_SUCCESS;
}

int LCALIFLateralConn::checkpointWrite(const char * cpDir) {
   int status = HyPerConn::checkpointWrite(cpDir);
   // The following comment is obsolete as of Jan 10, 2013.  // This is kind of hacky, but we save the extended buffer integratedSpikeCount as if it were a nonextended buffer of size (nx+2*nb)-by-(ny+2*nb)
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   const PVLayerLoc * loc;
   loc = pre->getLayerLoc();
// Commented out Jan 10, 2013
   // memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   // loc.nx += 2*loc.nb;
   // loc.ny += 2*loc.nb;
   // loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   // loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   // loc.nb = 0;
   int status2 = HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), parent->simulationTime(), &integratedSpikeCount, 1/*numbands*/, true/*extended*/, loc);
   if (status2!=PV_SUCCESS) status = status2;
   return status;
}

int LCALIFLateralConn::checkpointRead(const char * cpDir, double* timef) {
   int status = HyPerConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_integratedSpikeCount.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_integratedSpikeCount.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   double timed;
   const PVLayerLoc * loc;
   loc = pre->getLayerLoc();
// Commented out Jan 10, 2013
   // memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   // loc.nx += 2*loc.nb;
   // loc.ny += 2*loc.nb;
   // loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   // loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   // loc.nb = 0;
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &integratedSpikeCount, 1/*numbands*/, /*extended*/ true, loc);
   // read_pvdata(filename, parent->icCommunicator(), &timed, integratedSpikeCount, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   // Exchange borders
   if ( pre->useMirrorBCs() ) {
      for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
         pre->mirrorInteriorToBorder(borderId, integratedSpikeCountCube, integratedSpikeCountCube);
      }
   }
   parent->icCommunicator()->exchange(integratedSpikeCount, mpi_datatype, loc);

   return status;
}

int LCALIFLateralConn::outputState(double timef, bool last)
{
   int status;
   io_timer->start();

   if (last) {
      printf("Writing last LCALIFLateral weights..%f\n",timef);
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, last);
      assert(status == 0);
   }else if ( (timef >= writeTime) && (writeStep >= 0) ) {
      //writeTime += writeStep; Done in HyperConn
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, false);
      assert(status == 0);

      // append to output file after original open
      //ioAppend = true;
   }

   // io timer already in HyPerConn::outputState, don't call twice
   io_timer->stop();

   status = HyPerConn::outputState(timef, last);

   return status;
}


} /* namespace PV */
