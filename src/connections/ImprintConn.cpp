/*
 * ImprintConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "ImprintConn.hpp"
namespace PV {

ImprintConn::ImprintConn(){
   initialize_base();
}

ImprintConn::ImprintConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) : KernelConn()
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

ImprintConn::~ImprintConn() {
   free(lastActiveTime);
   free(imprinted);
}

int ImprintConn::initialize_base() {
   imprintTimeThresh = 100;
   return PV_SUCCESS;
}

int ImprintConn::allocateDataStructures() {
   int status = KernelConn::allocateDataStructures();
   const PVLayerLoc * loc = pre->getLayerLoc();
   int nf = loc->nf;
   imprinted = (bool*) calloc(nf, sizeof(bool));
   lastActiveTime = (double*) malloc(nf * sizeof(double));
   for(int fi = 0; fi < nf; fi++){
      lastActiveTime[fi] = fi * weightUpdatePeriod;
   }
   return status;
}

int ImprintConn::setParams(PVParams * params) {
   int status = KernelConn::setParams(params);
   imprintTimeThresh = (double) params->value(name, "imprintTimeThresh", imprintTimeThresh);
   if(imprintTimeThresh <= weightUpdateTime){
      fprintf(stderr, "Warning: ImprintConn's imprintTimeThresh is smaller than weightUpdateTime. The algorithm will imprint on every weight update\n");
   }
   return status;
}

bool ImprintConn::imprintFeature(int arborId, int kExt){
   PVPatch * weights = getWeights(kExt,arborId);
   //If imprinting, patch must not be a shrunken patch
   if(weights->nx != nxp || weights->ny != nyp){
      return false;
   }
   //TODO random processor with broadcast
   //Only one processor doing imprinting
   //Making a synced random number based off of sim time and kExt
   long syncedRandNum = (parent->simulationTime()/weightUpdatePeriod)+kExt;
   if(syncedRandNum % parent->icCommunicator()->commSize() != parent->icCommunicator()->commRank()){  
      return false;
   }

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arborId));
   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(); 
   size_t offset = getAPostOffset(kExt, arborId);
   int ny = weights->ny;
   int nk = weights->nx * nfp;
   const pvdata_t * postactRef = &postactbuf[offset];
   pvdata_t * dwdata = get_dwData(arborId, kExt);
   int lineoffsetw = 0;
   int lineoffseta = 0;

   bool hasTexture = false;
   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
          pvdata_t aPost = postactRef[lineoffseta+k];
          if(aPost != 0){
             hasTexture = true;
          }
          //Multiply aPost by a big number so it overpowers whatever is on the weights already
          //For some reason, this needs a huge number to make a difference
          //normalizing should take care of the scale after
          dwdata[lineoffsetw + k] = 9999999999 * aPost;
      }
      lineoffsetw += syp;
      lineoffseta += sya;
   }
   if(hasTexture){
      return true;
   }
   //If there's no texture in whatever was grabbed
   //Reset dwdata to 0
   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
          dwdata[lineoffsetw + k] = 0;
      }
      lineoffsetw += syp;
      lineoffseta += sya;
   }
   return false;
}


int ImprintConn::update_dW(int arbor_ID){
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arbor_ID));
   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(); 
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   //getDelay(arbor_ID));  //delay is from pre to post, so use current post activity

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   for(int fi = 0; fi < preLoc->nf; fi++){
      imprinted[fi] = false;
      //Reset numActiveFeature
      numActiveFeature[fi] = 0;
   }
   for(int kExt=0; kExt<nExt;kExt++) {
      pvdata_t preact = preactbuf[kExt];
      //Check imprinting
      int preFi = featureIndex(kExt, preLoc->nx + 2*preLoc->nb, preLoc->ny + 2*preLoc->nb, preLoc->nf); 
      //TODO this code is grabbing the first patch it can find. Make imprinting random so that it can grab other patches
      if (lastActiveTime[preFi] <= parent->simulationTime() - imprintTimeThresh){
         if(!imprinted[preFi]){
            //Random chance (one in 5) to imprint
            if(rand() % 5 == 0){
               imprinted[preFi] = imprintFeature(arbor_ID, kExt);
            }
            if(imprinted[preFi]){
               std::cout << "Imprinted feature " << preFi << "\n";
            }
         }
      }

      if (skipPre(preact)) continue;

      numActiveFeature[preFi]++;
      lastActiveTime[preFi] = parent->simulationTime();

      //if (preact == 0.0f) continue;
      bool inWindow = true;
      // only check inWindow if number of arbors > 1
      if (this->numberOfAxonalArborLists()>1){
         if(useWindowPost){
            int kPost = layerIndexExt(kExt, preLoc, postLoc);
            inWindow = post->inWindowExt(arbor_ID, kPost);
         }
         else{
            inWindow = pre->inWindowExt(arbor_ID, kExt);
         }
         if(!inWindow) continue;
      }
      PVPatch * weights = getWeights(kExt,arbor_ID);
      size_t offset = getAPostOffset(kExt, arbor_ID);
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(arbor_ID, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
             pvdata_t aPost = postactRef[lineoffseta+k];
             dwdata[lineoffsetw + k] += updateRule_dW(preact, aPost);
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
   }

   // Divide by numActiveFeature in this timestep
   int preNf = preLoc->nf;
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      //Calculate pre feature index from patch index
      //kernelindex always spins over nf first, so just mod preNf to find out iF
      int preiF = kernelindex % preNf;
      int divisor = numActiveFeature[preiF];
      if(divisor != 0){
         int numpatchitems = nxp*nyp*nfp;
         pvdata_t * dwpatchdata = get_dwDataHead(arbor_ID,kernelindex);
         for( int n=0; n<numpatchitems; n++ ) {
            dwpatchdata[n] /= divisor;
         }
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;

}




} // end namespace PV
