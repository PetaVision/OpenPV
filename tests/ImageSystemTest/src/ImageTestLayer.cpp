#include "ImageTestLayer.hpp"

namespace PV {

   ImageTestLayer::ImageTestLayer(const char * name, HyPerCol * hc) {
      initialize(name, hc);
   }

   int ImageTestLayer::updateState(double time, double dt)
   {
      ImageLayer::updateState(time, dt);
      const PVLayerLoc * loc = getLayerLoc();
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int nbatch = loc->nbatch;
      for(int b = 0; b < nbatch; b++){
         pvdata_t * dataBatch = getActivity() + b * getNumExtended();
         for(int nkRes = 0; nkRes < getNumNeurons(); nkRes++){
            //Calculate extended index
            int nkExt = kIndexExtended(nkRes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);  
            //checkVal is the value from batch index 0
            pvdata_t checkVal = dataBatch[nkExt] * 255;

            int kxGlobal = kxPos(nkRes, nx, ny, nf) + loc->kx0;
            int kyGlobal = kyPos(nkRes, nx, ny, nf) + loc->ky0; 
            int kf = featureIndex(nkRes, nx, ny, nf);

            pvdata_t expectedVal = kIndex(kxGlobal, kyGlobal, kf, loc->nxGlobal, loc->nyGlobal, nf);
            if(fabs(checkVal - expectedVal) >= 1e-5){
               pvError() << "ImageFileIO test Expected: " << expectedVal << " Actual: " << checkVal << "\n";
            }
         }
      }
      return PV_SUCCESS;
   }

}
