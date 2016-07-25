#include "MoviePvpTestLayer.hpp"

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(const char * name, HyPerCol * hc) {
   MoviePvp::initialize(name, hc);
}

int MoviePvpTestLayer::updateState(double time, double dt)
{
   MoviePvp::updateState(time, dt);
   std::cout << "Here\n";
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nbatch = loc->nbatch;
   int nbatchGlobal = loc->nbatchGlobal;
   int commBatch = parent->commBatch();
   int numBatchPerProc = parent->numCommBatches();

   for(int b = 0; b < nbatch; b++){
      pvdata_t * dataBatch = data + b * getNumExtended();
      int frameIdx;
      if(strcmp(getBatchMethod(), "byImage") == 0 || strcmp(getBatchMethod(), "bySpecified") == 0){
         frameIdx = (time-1) * nbatchGlobal + commBatch*numBatchPerProc + b;
         std::cout << "frameIdx:" << frameIdx << ", nbatchGlobal:" << nbatchGlobal << ", commBatch:"<< commBatch << ", numBatchPerProc:" << numBatchPerProc << ", b:" << b << "\n"; 
      }
      else if(strcmp(getBatchMethod(), "byMovie") == 0){
         frameIdx = b * 2 + (time-1);
      }
      for(int nkRes = 0; nkRes < getNumNeurons(); nkRes++){
         //Calculate extended index
         int nkExt = kIndexExtended(nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);  
         //checkVal is the value from batch index 0
         pvdata_t checkVal = dataBatch[nkExt];

         int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
         int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0; 
         int kf = featureIndex(nkRes, nx, ny, nf);

         pvdata_t expectedVal = kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf) + frameIdx*192;
         if(fabs(checkVal - expectedVal) >= 1e-5){
            pvError() << "ImageFileIO " << name << " test Expected: " << expectedVal << " Actual: " << checkVal << "\n";
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createMoviePvpTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new MoviePvpTestLayer(name, hc) : NULL;
}

}  // end namespace PV

