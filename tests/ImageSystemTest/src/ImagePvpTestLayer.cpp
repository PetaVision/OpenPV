#include "ImagePvpTestLayer.hpp"

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(const char * name, HyPerCol * hc) {
   ImagePvp::initialize(name, hc);
}

int ImagePvpTestLayer::updateStateWrapper(double time, double dt)
{
   ImagePvp::updateStateWrapper(time, dt);
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nbatch = loc->nbatch;
   int frameIdx = getPvpFrameIdx();
   for(int b = 0; b < nbatch; b++){
      pvdata_t * dataBatch = data + b * getNumExtended();
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
            std::cout << "ImageFileIO " << name << " test Expected: " << expectedVal << " Actual: " << checkVal << "\n";
            exit(-1);
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createImagePvpTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new ImagePvpTestLayer(name, hc) : NULL;
}

}

