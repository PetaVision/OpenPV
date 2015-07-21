/*
 * LocalKernelConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "LocalKernelConn.hpp"

namespace PV {

LocalKernelConn::LocalKernelConn() {
    initialize_base();
}

LocalKernelConn::LocalKernelConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, NULL, NULL);
}

int LocalKernelConn::initialize_base() {
   decay = .01;
   return PV_SUCCESS;
}  // end of LocalKernelConn::initialize_base()

int LocalKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = KernelConn::ioParamsFillGroup(ioFlag);
   ioParam_decay(ioFlag);
   return status;
}

void LocalKernelConn::ioParam_decay(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "decay", &decay, decay, true/*warnIfAbsent*/);
}

//int LocalKernelConn::defaultUpdateInd_dW(int arbor_ID, int kExt){
//   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arbor_ID));
//   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(); 
//   const PVLayerLoc * preLoc = pre->getLayerLoc();
//   const PVLayerLoc * postLoc = post->getLayerLoc();
//   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));
//
//   pvdata_t preact = preactbuf[kExt];
//   if (skipPre(preact)) return PV_CONTINUE;
//   //update numKernelActivations
//
//   int kernelIndex = patchIndexToDataIndex(kExt);
//   //Only increment if kernelIndex is restricted
//   int nxExt = preLoc->nx + 2*preLoc->nb;
//   int nyExt = preLoc->ny + 2*preLoc->nb;
//   int nf = preLoc->nf;
//   int extX = kxPos(kExt, nxExt, nyExt, nf);
//   int extY = kyPos(kExt, nxExt, nyExt, nf);
//   if(extX >= preLoc->nb && extX < preLoc->nx + preLoc->nb &&
//      extY >= preLoc->nb && extY < preLoc->ny + preLoc->nb){
//      numKernelActivations[kernelIndex]++;
//   }
//
//   //if (preact == 0.0f) continue;
//   bool inWindow = true;
//   // only check inWindow if number of arbors > 1
//   if (this->numberOfAxonalArborLists()>1){
//      if(useWindowPost){
//         int kPost = layerIndexExt(kExt, preLoc, postLoc);
//         inWindow = post->inWindowExt(arbor_ID, kPost);
//      }
//      else{
//         inWindow = pre->inWindowExt(arbor_ID, kExt);
//      }
//      if(!inWindow) return PV_CONTINUE;
//   }
//   PVPatch * weights = getWeights(kExt,arbor_ID);
//   size_t offset = getAPostOffset(kExt, arbor_ID);
//   int ny = weights->ny;
//   int nk = weights->nx * nfp;
//   const pvdata_t * postactRef = &postactbuf[offset];
//   pvdata_t * dwdata = get_dwData(arbor_ID, kExt);
//   pvdata_t * wdata = get_wData(arbor_ID, kExt);
//   int lineoffsetw = 0;
//   int lineoffseta = 0;
//   for( int y=0; y<ny; y++ ) {
//      for( int k=0; k<nk; k++ ) {
//         pvdata_t aPost = postactRef[lineoffseta+k];
//         dwdata[lineoffsetw + k] += updateRule_dW(preact, aPost) + decay*(wdata[lineoffsetw + k]);
//      }
//      lineoffsetw += syp;
//      lineoffseta += sya;
//   }
//   return PV_SUCCESS;
//}

int LocalKernelConn::updateWeights(int arbor_ID){
   lastUpdateTime = parent->simulationTime();
   // add dw to w
   for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
      pvwdata_t * w_data_start = get_wDataStart(kArbor);
      for( int dataPatchIdx = 0; dataPatchIdx < getNumDataPatches(); dataPatchIdx++){
         float sumsq = 0;
         for( int patchIdx=0; patchIdx<nxp*nyp*nfp; patchIdx++ ) {
            int k = dataPatchIdx*(nxp*nyp*nfp) + patchIdx;
            w_data_start[k] += get_dwDataStart(kArbor)[k];

            //if(parent->columnId() == 0 && isnan(w_data_start[k])){
            //   std::cout << "nan found, dataPatchIdx: " << dataPatchIdx << " patchIdx: " << patchIdx << " dw: " << get_dwDataStart(kArbor)[k] << " w: " << w_data_start[k] << "\n";
            //}
            
            ////Apply max
            //if(w_data_start[k] > 1){
            //   std::cout << "Applying max to kernel " << k << " at time " << parent->simulationTime() << " for the weight " << w_data_start[k] << "\n";
            //   w_data_start[k] = 1;
            //}
            ////Apply min
            //if(w_data_start[k] < -1){
            //   std::cout << "Applying min to kernel " << k << " at time " << parent->simulationTime() << " for the weight " << w_data_start[k] << "\n";
            //   w_data_start[k] = -1;
            //}
            sumsq += w_data_start[k] * w_data_start[k];
         }
         if(parent->columnId() == 0){
            std::cout << "Patch number " << dataPatchIdx << " l2 norm: " << sqrt(sumsq) << "\n";
         }
      }
   }
   return PV_BREAK;
}


}  // end of namespace PV block
