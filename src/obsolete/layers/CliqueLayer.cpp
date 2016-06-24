/*
 * CliqueLayer.cpp
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#include "CliqueLayer.hpp"
#include "../utils/conversions.h"
#include "../connections/HyPerConn.hpp"
#include "../connections/TransposeConn.hpp" // Needed by recvSynapticInputFromPostBase, even though that hasn't been rewritten for CliqueConn things
#include "ANNLayer.hpp"

#include <assert.h>

namespace PV {

CliqueLayer::CliqueLayer()
{
   initialize_base();
}

CliqueLayer::CliqueLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

CliqueLayer::~CliqueLayer()
{
}

int CliqueLayer::initialize_base()
{
   return PV_SUCCESS;
}

int CliqueLayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

int CliqueLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_Voffset(ioFlag);
   ioParam_Vgain(ioFlag);
   return status;
}

void CliqueLayer::ioParam_Voffset(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "Voffset", &Voffset, (pvdata_t) 0, true);
}

void CliqueLayer::ioParam_Vgain(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "Vgain", &Vgain, (pvdata_t) 2, true);
}

// CliqueLayer overrides recvAllSynapticInput with what used to be in HyPerLayer::recvAllSynapticInput,
// before recvAllSynapticInput changed to call connections' deliver method instead of layers' recv methods
// Since CliqueLayer never had recvSynapticInputFromPost or GPU receive methods, the initialization should
// prevent parameters that require them from being used?
int CliqueLayer::recvAllSynapticInput() {
   int status = PV_SUCCESS;
   //Only recvAllSynapticInput if we need an update
   if(needUpdate(parent->simulationTime(), parent->getDeltaTime())){
      bool switchGpu = false;
      //Start CPU timer here
      recvsyn_timer->start();

      for(std::vector<BaseConnection*>::iterator it = recvConns.begin(); it < recvConns.end(); it++){
         BaseConnection * baseConn = *it;
         HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);
         assert(conn != NULL);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         //Check if it's done with cpu connections
         if(!switchGpu && conn->getReceiveGpu()){
            //Copy GSyn over to GPU
            copyAllGSynToDevice();
#ifdef PV_USE_CUDA
            //Start gpu timer
            gpu_recvsyn_timer->start();
#endif
            switchGpu = true;
         }
#endif

         //Check if updating from post perspective
         HyPerLayer * pre = conn->preSynapticLayer();
         PVLayerCube cube;
         memcpy(&cube.loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
         cube.numItems = pre->getNumExtended();
         cube.size = sizeof(PVLayerCube);

         DataStore * store = parent->icCommunicator()->publisherStore(pre->getLayerId());
         int numArbors = conn->numberOfAxonalArborLists();

         for (int arbor=0; arbor<numArbors; arbor++) {
            int delay = conn->getDelay(arbor);
            cube.data = (pvdata_t *) store->buffer(LOCAL, delay);
            if(!conn->getUpdateGSynFromPostPerspective()){
               cube.isSparse = store->isSparse();
               if(cube.isSparse){
                  cube.numActive = *(store->numActiveBuffer(LOCAL, delay));
                  cube.activeIndices = store->activeIndicesBuffer(LOCAL, delay);
               }
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
               if(conn->getReceiveGpu()){
                  if (parent->columnId()==0) {
                     fprintf(stderr, "%s \"%s: CliqueLayer has not yet implemented receiving synaptic input on GPUs.\n",
                           parent->parameters()->groupKeywordFromName(name), name);
                  }
                  MPI_Barrier(parent->icCommunicator()->communicator());
                  exit(EXIT_FAILURE);
                  // status = recvSynapticInputGpu(conn, &cube, arbor);
                  // //No need to update GSyn since it's already living on gpu
                  // updatedDeviceGSyn = false;
               }
               else
#endif
               {
                  status = recvSynapticInput(conn, &cube, arbor);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
                  //CPU updated gsyn, need to update gsyn
                  updatedDeviceGSyn = true;
#endif
               }
            }
            else{
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
               if(conn->getReceiveGpu()){
                  if (parent->columnId()==0) {
                     fprintf(stderr, "%s \"%s: CliqueLayer has not yet implemented receiving synaptic input on GPUs.\n",
                           parent->parameters()->groupKeywordFromName(name), name);
                  }
                  MPI_Barrier(parent->icCommunicator()->communicator());
                  exit(EXIT_FAILURE);
                  // status = recvSynapticInputFromPostGpu(conn, &cube, arbor);
               }
               else
#endif
               {
                  if (parent->columnId()==0) {
                     fprintf(stderr, "%s \"%s: CliqueLayer has not yet implemented receiving from the postsynaptic perspective.\n",
                           parent->parameters()->groupKeywordFromName(name), name);
                  }
                  MPI_Barrier(parent->icCommunicator()->communicator());
                  exit(EXIT_FAILURE);
                  // status = recvSynapticInputFromPost(conn, &cube, arbor);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
                  updatedDeviceGSyn = true;
#endif
               }
            }
            assert(status == PV_SUCCESS || status == PV_BREAK);
            if (status == PV_BREAK){
               break; // breaks out of arbor loop
            }
         }
      }
#ifdef PV_USE_CUDA
      if(switchGpu){
         //Stop timer
         gpu_recvsyn_timer->stop();
      }
#endif
      recvsyn_timer->stop();
   }
   return status;
}

int CliqueLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity,
      int axonId)
{
   recvsyn_timer->start();
   // do not receive synaptic input from conn if plasticity flag is true
   if (conn->getPlasticityFlag()){
      return PV_BREAK;
   }
   enum ChannelType channel_type = conn->getChannel();
   if (channel_type == CHANNEL_EXC) {
      return recvSynapticInputBase(conn, activity, axonId);
   }
   // number of axons = patch size ^ (clique size - 1)
   int numCliques = conn->numberOfAxonalArborLists();
   int cliqueSize = 1
         + (int) rint(
               log2(numCliques)
                     / log2(
                           conn->xPatchSize() * conn->yPatchSize() * conn->fPatchSize()));
   if (cliqueSize == 1) {
      return recvSynapticInputBase(conn, activity, axonId);
   }

   assert(axonId == 0);
   // assume called only once
   //const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayer::recvSynapticInput: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, axonId, getNumExtended(), activity, this, conn);
   fflush(stdout);
#endif

   // get margin indices of pre layer
   // (whoops, not needed since we probably have to recompute all active indices in extended layer on the fly due to variable delays)
   //const int * marginIndices = conn->getPre()->getMarginIndices();
   //int numMargin = conn->getPre()->getNumMargin();

   const PVLayerLoc * preLoc = conn->getPre()->getLayerLoc();
   const int nfPre = preLoc->nf;
   //const int nxPre = preLoc->nx;
   //const int nyPre = preLoc->ny;
   const int nxPreExt = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   const int nyPreExt = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   int syPost = conn->getPostNonextStrides()->sy; // stride in layer

   // if pre and post are the same layers, make a clone of size PVPatch to hold temporary activity values
   // in order to eliminate generalize self-interactions
   // note that during learning, per and post may be separate instantiations
   bool self_flag = conn->getSelfFlag();
   pvdata_t * a_post_mask = NULL;
   const int a_post_size = conn->fPatchSize() * conn->xPatchSize() * conn->yPatchSize();
   a_post_mask = (pvdata_t *) calloc(a_post_size, sizeof(pvdata_t));
   assert(a_post_mask != NULL);
   // get linear index of cell at center of patch for self_flag == true
   const int k_post_self = (int) (a_post_size / 2); // if self_flag == true, a_post_size should be odd
   for (int k_post = 0; k_post < a_post_size; k_post++) {
      a_post_mask[k_post] = 1;
   }

   // gather active indices in extended layer
   // hard to pre-compute at HyPerLayer level because of variable delays
   int numActiveExt = 0;
   unsigned int * activeExt = (unsigned int *) calloc(conn->getPre()->getNumExtended(),
         sizeof(int));
   assert(activeExt != NULL);
   float * aPre = activity->data;
   for (int kPreExt = 0; kPreExt < conn->getPre()->getNumExtended(); kPreExt++) {
      if (aPre[kPreExt] == 0) continue;
      activeExt[numActiveExt++] = kPreExt;
   }

   // calc dimensions of wPostPatch
   // TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in HyPerConn::wPostPatches
   // TODO: following should be implemented as HyPerConn::calcPostPatchSize
   //const PVLayer * lPre = clayer;
   const float xScaleDiff = conn->getPost()->getXScale() - getXScale();
   const float yScaleDiff = conn->getPost()->getYScale() - getYScale();
   const float powXScale = pow(2.0f, (float) xScaleDiff);
   const float powYScale = pow(2.0f, (float) yScaleDiff);
   //const int prePad = lPre->loc.nb;
   const int nxPostPatch = (int) (conn->xPatchSize() * powXScale); // TODO: store in HyPerConn::wPostPatches
   const int nyPostPatch = (int) (conn->yPatchSize() * powYScale); // TODO: store in HyPerConn::wPostPatches
   //const int nfPostPatch = nf;

   // clique dimensions
   // a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
   // TODO: precompute clique dimensions during CliqueConn::initialize
   int nyCliqueRadius = (int) (nyPostPatch / 2);
   int nxCliqueRadius = (int) (nxPostPatch / 2);
   int cliquePatchSize = (2 * nxCliqueRadius + 1) * (2 * nyCliqueRadius + 1) * nfPre;
   //int numKernels = conn->numDataPatches();  // per arbor?
   //int numCliques = pow(cliquePatchSize, cliqueSize - 1);
   //assert(numCliques == conn->numberOfAxonalArborLists());

   // loop over all products of cliqueSize active presynaptic cells
   // outer loop is over presynaptic cells, each of which defines the center of a cliquePatch
   // inner loop is over all combinations of clique cells within cliquePatch boundaries, which may be shrunken
   // TODO: pre-allocate cliqueActiveIndices as CliqueConn::cliquePatchSize member variable
   int * cliqueActiveIndices = (int *) calloc(cliquePatchSize, sizeof(int));
   assert(cliqueActiveIndices != NULL);
   for (int kPreActive = 0; kPreActive < numActiveExt; kPreActive++) {
      int kPreExt = activeExt[kPreActive];

      // get indices of active elements in clique radius
      // watch out for shrunken patches!
      int numActiveElements = 0;
      int kxPreExt = kxPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      int kyPreExt = kyPos(kPreExt, nxPreExt, nyPreExt, nfPre);
      if (cliqueSize > 1) {
         for (int kyCliqueExt = (
               (kyPreExt - nyCliqueRadius) > 0 ? (kyPreExt - nyCliqueRadius) : 0);
               kyCliqueExt
                     < ((kyPreExt + nyCliqueRadius) <= nyPreExt ? (kyPreExt
                           + nyCliqueRadius) :
                           nyPreExt); kyCliqueExt++) {
            //if (kyCliqueExt < 0 || kyCliqueExt > nyPreExt) continue;
            for (int kxCliqueExt = (
                  (kxPreExt - nxCliqueRadius) > 0 ? (kxPreExt - nxCliqueRadius) : 0);
                  kxCliqueExt
                        < ((kxPreExt + nxCliqueRadius) <= nxPreExt ? (kxPreExt
                              + nxCliqueRadius) :
                              nxPreExt); kxCliqueExt++) {
               //if (kyCliqueExt < 0 || kyCliqueExt > nxPreExt) continue;
               for (int kfCliqueExt = 0; kfCliqueExt < nfPre; kfCliqueExt++) {
                  int kCliqueExt = kIndex(kxCliqueExt, kyCliqueExt, kfCliqueExt, nxPreExt,
                        nyPreExt, nfPre);
                  if ((aPre[kCliqueExt] == 0) || (kCliqueExt == kPreExt)) continue;
                  cliqueActiveIndices[numActiveElements++] = kCliqueExt;
               }
            }
         }
      } // cliqueSize > 1
      else {
         cliqueActiveIndices[numActiveElements++] = kPreExt; // each cell is its own clique if cliqueSize == 1
      }
      if (numActiveElements < (cliqueSize - 1)) continue;

      // loop over all active combinations of size=cliqueSize-1 in clique radius
      int numActiveCliques = (int) pow(numActiveElements, cliqueSize - 1);
      for (int kClique = 0; kClique < numActiveCliques; kClique++) {

         //initialize a_post_tmp
         if (self_flag) { // otherwise, a_post_mask is not modified and thus doesn't have to be updated
            for (int k_post = 0; k_post < a_post_size; k_post++) {
               a_post_mask[k_post] = 1;
            }
            a_post_mask[k_post_self] = 0;
         }

         // decompose kClique to compute product of active clique elements
         int arborNdx = 0;
         pvdata_t cliqueProd = aPre[kPreExt];
         int kResidue = kClique;
         int maxIndex = -1;
         for (int iProd = 0; iProd < cliqueSize - 1; iProd++) {
            int kPatchActive = (unsigned int) (kResidue
                  / pow(numActiveElements, cliqueSize - 1 - iProd - 1));

            // only apply each permutation of clique elements once, no element can contribute more than once
            if (kPatchActive <= maxIndex) {
               break;
            }
            else {
               maxIndex = kPatchActive;
            }
            int kCliqueExt = cliqueActiveIndices[kPatchActive];
            cliqueProd *= aPre[kCliqueExt];
            kResidue = kResidue
                  - kPatchActive
                        * (int) pow(numActiveElements, cliqueSize - 1 - iProd - 1);

            // compute arborIndex for this clique element
            int kxCliqueExt = kxPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kyCliqueExt = kyPos(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kfClique = featureIndex(kCliqueExt, nxPreExt, nyPreExt, nfPre);
            int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
            int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
            unsigned int kArbor = kIndex(kxPatch, kyPatch, kfClique,
                  (2 * nxCliqueRadius + 1), (2 * nyCliqueRadius + 1), nfPre);
            arborNdx += kArbor * (int) pow(cliquePatchSize, cliqueSize - 1 - iProd - 1);
            if ((arborNdx < 0) || (arborNdx >= numCliques)) {
               assert((arborNdx >= 0) && (arborNdx < numCliques));
            }
            // remove self-interactions if pre == post
            if (self_flag) {
               a_post_mask[kArbor] = 0;
            }
         } // iProd

         PVPatch * w_patch = conn->getWeights(kPreExt, arborNdx);

         const pvwdata_t * w_start = conn->get_wDataStart(arborNdx);
         int kernelIndex = conn->patchToDataLUT(kPreExt);
         const pvwdata_t * w_head = &(w_start[a_post_size * kernelIndex]);
         size_t w_offset = w_patch->offset; // w_patch->data - w_head;

         // WARNING - assumes weight and GSyn patches from task same size
         //         - assumes patch stride sf is 1

         int nkPost = conn->getPost()->getLayerLoc()->nf * w_patch->nx;
         int nyPost = w_patch->ny;
         int sywPatch = conn->yPatchStride(); // stride in patch

         pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());

         // TODO - unroll
         for (int y = 0; y < nyPost; y++) {
            pvpatch_accumulate2(nkPost,
                  (float *) (gSynPatchHead + conn->getGSynPatchStart(kPreExt, arborNdx) + y * syPost),
                  cliqueProd, (pvwdata_t *) (w_head + w_offset + y * sywPatch), // (w_patch->data + y * sywPatch),
                  (float *) (a_post_mask + w_offset + y * sywPatch));
         }

      } // kClique
   } // kPreActive
   free(activeExt);
   free(cliqueActiveIndices);
   free(a_post_mask);
   recvsyn_timer->stop();
   return PV_BREAK;
}

int CliqueLayer::recvSynapticInputBase(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   // copied from old HyPerLayer::recvSynapticInput when HyPerLayer receive methods were turned into HyPerConn deliver methods

   //Check if we need to update based on connection's channel
   if(conn->getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(GSyn && GSyn[conn->getChannel()]);

   //Simplified from HyPerConn::getConvertToRateDeltaTimeFactor() since
   //conn's post is this, and CliqueLayer doesn't override getChannelTimeConst()
   float dt_factor = conn->preSynapticActivityIsNotRate() ? parent->getDeltaTime() : 1.0f;
   //float dt_factor = getConvertToRateDeltaTimeFactor(conn);

   const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = this->getLayerLoc();


   assert(arborID >= 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif // DEBUG_OUTPUT


   //Clear all thread gsyn buffer
   if(thread_gSyn){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int i = 0; i < parent->getNumThreads() * getNumNeurons(); i++){
         int ti = i/getNumNeurons();
         int ni = i % getNumNeurons();
         thread_gSyn[ti][ni] = 0;
      }
   }

   int numLoop;
   if(activity->isSparse){
      numLoop = activity->numActive;
   }
   else{
      numLoop = numExtended;
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
   //TODO loop over active indicies here instead
   for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
      int kPre;
      if(activity->isSparse){
         kPre = activity->activeIndices[loopIndex];
      }
      else{
         kPre = loopIndex;
      }

      bool inWindow;
      //Post layer recieves synaptic input
      //Only with respect to post layer
      const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
      const PVLayerLoc * postLoc = this->getLayerLoc();
      int kPost = layerIndexExt(kPre, preLoc, postLoc);
#ifdef OBSOLETE // Marked obsolete Dec 2, 2014.  Use sharedWeights=false instead of windowing.
      inWindow = inWindowExt(arborID, kPost);
      if(!inWindow) continue;
#endif // OBSOLETE

      float a = activity->data[kPre] * dt_factor;
      // Activity < 0 is used by generative models --pete
      if (a == 0.0f) continue;

      //If we're using thread_gSyn, set this here
      pvdata_t * gSynPatchHead;
#ifdef PV_USE_OPENMP_THREADS
      if(thread_gSyn){
         int ti = omp_get_thread_num();
         gSynPatchHead = thread_gSyn[ti];
      }
      else{
         gSynPatchHead = this->getChannel(conn->getChannel());
      }
#else
      gSynPatchHead = this->getChannel(conn->getChannel());
#endif
      conn->deliverOnePreNeuronActivity(kPre, arborID, a, gSynPatchHead, conn->getRandState(kPre));
   }
#ifdef PV_USE_OPENMP_THREADS
   //Accumulate back into gSyn
   if(thread_gSyn){
      pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
      //Looping over neurons first to be thread safe
#pragma omp parallel for
      for(int ni = 0; ni < getNumNeurons(); ni++){
         //Different for maxpooling
         if(conn->getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
            for(int ti = 0; ti < parent->getNumThreads(); ti++){
               gSynPatchHead[ni] = gSynPatchHead[ni] < thread_gSyn[ti][ni] ? thread_gSyn[ti][ni] : gSynPatchHead[ni];
            }
         }
         else{
            for(int ti = 0; ti < parent->getNumThreads(); ti++){
               gSynPatchHead[ni] += thread_gSyn[ti][ni];
            }
         }
      }
   }
#endif

   return PV_SUCCESS;
}

int CliqueLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   return updateStateClique(timef, dt,loc, A, getV(),
         num_channels, gSynHead, this->Voffset, this->Vgain, this->AMax, this->AMin,
         this->VThresh, parent->columnId());
}

// TODO: direct clique input to separate GSyn: CHANNEL_CLIQUE
/*
int CliqueLayer::updateState(double timef, double dt)
{
   return updateStateClique(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
         getNumChannels(), GSyn[0], this->Voffset, this->Vgain, this->AMax, this->AMin,
         this->VThresh, clayer->columnId);
}
*/

int CliqueLayer::updateStateClique(double timef, double dt, const PVLayerLoc * loc,
      pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t Voffset,
      pvdata_t Vgain, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, int columnID)
{
   pv_debug_info("[%d]: CliqueLayer::updateState:", columnID);

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx * ny * nf;

   // Assumes that channels are contiguous in memory, i.e. GSyn[ch] = GSyn[0]+num_neurons*ch.  See allocateBuffers().
   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   //pvdata_t * gSynInhB = getChannelStart(gSynHead, CHANNEL_INHB, num_neurons);
// assume bottomUp input to gSynExc, target lateral input to gSynInh, distractor lateral input to gSynInhB
   for (int k = 0; k < num_neurons; k++) {
      V[k] = 0.0f;
      pvdata_t bottomUp_input = gSynExc[k];
      if (bottomUp_input <= 0.0f) {
         continue;
      }
      pvdata_t lateral_exc = gSynInh[k];
      //pvdata_t lateral_inh = gSynInhB[k];
      //pvdata_t lateral_denom = ((lateral_exc + fabs(lateral_inh)) > 0.0f) ? (lateral_exc + fabs(lateral_inh)) : 1.0f;

      //V[k] = bottomUp_input * (this->Voffset + this->Vgain * (lateral_exc - lateral_inh));
      V[k] = bottomUp_input * (Voffset + Vgain * (lateral_exc)); // - fabs(lateral_inh))); // / lateral_denom);
   } // k

   // resetGSynBuffers called by HyPerCol
   // resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead);
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // setActivity();
   applyVThresh_ANNLayer(num_neurons, V, AMin, VThresh, 0.0f/*AShift*/, 0.0f/*VWidth*/, A, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // applyVThresh();
   applyVMax_ANNLayer(num_neurons, V, AMax, A, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // applyVMax();

   //Moved to publish
   //updateActiveIndices();

   return 0;
}

} /* namespace PV */

